// load_hf.zig — turn a HuggingFace model directory (safetensors + config.json)
// into the engine's Runtime / LayerSource / HeadSource.
//
// Layout: PyTorch stores a linear layer as [out_features, in_features] in C
// order, which is exactly the ANE 1x1-conv weight layout, so no transpose is
// needed. RoPE is the HF half-split convention (`rope_adjacent = false`), unlike
// llama-family GGUF files whose Q/K rows are permuted at conversion time.

const std = @import("std");
const safetensors = @import("safetensors.zig");
const hf = @import("hf.zig");
const model = @import("model.zig");
const sys = @import("sys.zig");

pub const Error = error{
    MissingTensor,
    DimensionMismatch,
    UnsupportedArchitecture,
};

/// All safetensors shards of one model directory.
pub const Shards = struct {
    allocator: std.mem.Allocator,
    list: []safetensors.Safetensors,

    pub fn open(allocator: std.mem.Allocator, dir: []const u8) !Shards {
        const paths = try hf.findShards(allocator, dir);
        defer hf.freeShards(allocator, paths);
        const list = try allocator.alloc(safetensors.Safetensors, paths.len);
        var opened: usize = 0;
        errdefer {
            for (list[0..opened]) |*s| s.deinit();
            allocator.free(list);
        }
        for (paths, 0..) |p, i| {
            list[i] = try safetensors.Safetensors.load(allocator, p);
            opened += 1;
        }
        return .{ .allocator = allocator, .list = list };
    }

    pub fn deinit(self: *Shards) void {
        for (self.list) |*s| s.deinit();
        self.allocator.free(self.list);
        self.* = undefined;
    }

    pub fn find(self: *const Shards, name: []const u8) ?safetensors.Tensor {
        for (self.list) |*s| {
            if (s.tensor(name)) |t| return t;
        }
        return null;
    }

    pub fn readF16(self: *const Shards, allocator: std.mem.Allocator, name: []const u8) ![]f16 {
        for (self.list) |*s| {
            if (s.has(name)) return s.readF16(allocator, name);
        }
        return Error.MissingTensor;
    }

    pub fn readF32(self: *const Shards, allocator: std.mem.Allocator, name: []const u8) ![]f32 {
        for (self.list) |*s| {
            if (s.has(name)) return s.readF32(allocator, name);
        }
        return Error.MissingTensor;
    }
};

/// Translate an HF config into the engine's format-independent config.
pub fn toModelConfig(c: hf.Config) !model.Config {
    if (c.intermediate_size == 0) return Error.UnsupportedArchitecture;
    if (c.hidden_size == 0 or c.num_hidden_layers == 0 or c.num_attention_heads == 0) {
        return Error.UnsupportedArchitecture;
    }
    return .{
        .arch = if (std.mem.startsWith(u8, c.arch, "Qwen"))
            "qwen2"
        else if (std.mem.startsWith(u8, c.arch, "Gemma"))
            "gemma2"
        else
            "llama",
        .hidden = c.hidden_size,
        .layers = c.num_hidden_layers,
        .heads = c.num_attention_heads,
        .kv_heads = c.num_key_value_heads,
        .head_dim = c.head_dim,
        .inter = c.intermediate_size,
        .vocab = c.vocab_size,
        .eps = c.rms_norm_eps,
        .rope_theta = c.rope_theta,
        .tie_embeddings = c.tie_word_embeddings,
        // HF keeps the rotate_half (half-split) RoPE layout.
        .rope_adjacent = false,
        .num_experts = c.num_experts,
        .experts_per_tok = c.num_experts_per_tok,
        .moe_inter = c.moe_intermediate_size,
        .shared_inter = c.shared_expert_intermediate_size,
        .norm_topk_prob = c.norm_topk_prob,
        .sparse_step = c.decoder_sparse_step,
        .mlp_only_mask = c.mlp_only_layers,
        .norm_unit_offset = std.mem.startsWith(u8, c.arch, "Gemma"),
        .embed_scale = if (std.mem.startsWith(u8, c.arch, "Gemma"))
            @sqrt(@as(f32, @floatFromInt(c.hidden_size)))
        else
            1.0,
        .use_gelu = std.mem.startsWith(u8, c.arch, "Gemma"),
        // Gemma 2's caps and window come from the config, like the GGUF metadata.
        .attn_logit_softcap = c.attn_logit_softcapping,
        .final_logit_softcap = c.final_logit_softcapping,
        .sliding_window = if (c.sliding_window > 0) c.sliding_window else c.sliding_window_size,
    };
}

/// Load a Linear weight as fp16 in [out][in] order, validating the shape.
fn linear(allocator: std.mem.Allocator, sh: *const Shards, name: []const u8, in_dim: u32, out_dim: u32) ![]f16 {
    const t = sh.find(name) orelse return Error.MissingTensor;
    if (t.numel() != @as(usize, in_dim) * out_dim) return Error.DimensionMismatch;
    if (t.rank() != 2 or t.shape[0] != out_dim or t.shape[1] != in_dim) return Error.DimensionMismatch;
    return sh.readF16(allocator, name);
}

fn norm(allocator: std.mem.Allocator, sh: *const Shards, name: []const u8, n: u32) ![]f32 {
    const v = try sh.readF32(allocator, name);
    if (v.len != n) {
        allocator.free(v);
        return Error.DimensionMismatch;
    }
    return v;
}

pub fn loadRuntime(allocator: std.mem.Allocator, sh: *const Shards, cfg: model.Config, progress: bool) !model.Runtime {
    var rt = model.Runtime{ .allocator = allocator, .config = cfg };
    errdefer rt.deinit();
    if (progress) sys.print("  runtime weights: embedding + norms\n", .{});
    rt.embed = try linear(allocator, sh, "model.embed_tokens.weight", cfg.hidden, cfg.vocab);
    rt.embed_owned = true;
    rt.final_norm = try norm(allocator, sh, "model.norm.weight", cfg.hidden);
    rt.norms = try allocator.alloc(model.Norm, cfg.layers);
    @memset(rt.norms, .{});
    var buf: [192]u8 = undefined;
    for (rt.norms, 0..) |*n, i| {
        const li: u32 = @intCast(i);
        n.attn = try norm(allocator, sh, std.fmt.bufPrint(&buf, "model.layers.{d}.input_layernorm.weight", .{li}) catch unreachable, cfg.hidden);
        n.ffn = try norm(allocator, sh, std.fmt.bufPrint(&buf, "model.layers.{d}.post_attention_layernorm.weight", .{li}) catch unreachable, cfg.hidden);
        if (sh.find(std.fmt.bufPrint(&buf, "model.layers.{d}.self_attn.q_norm.weight", .{li}) catch unreachable) != null) {
            n.q_norm = try norm(allocator, sh, std.fmt.bufPrint(&buf, "model.layers.{d}.self_attn.q_norm.weight", .{li}) catch unreachable, cfg.head_dim);
            n.k_norm = try norm(allocator, sh, std.fmt.bufPrint(&buf, "model.layers.{d}.self_attn.k_norm.weight", .{li}) catch unreachable, cfg.head_dim);
        }
        if (sh.find(std.fmt.bufPrint(&buf, "model.layers.{d}.self_attn.q_proj.bias", .{li}) catch unreachable) != null) {
            const bq = try sh.readF32(allocator, std.fmt.bufPrint(&buf, "model.layers.{d}.self_attn.q_proj.bias", .{li}) catch unreachable);
            defer allocator.free(bq);
            const bk = try sh.readF32(allocator, std.fmt.bufPrint(&buf, "model.layers.{d}.self_attn.k_proj.bias", .{li}) catch unreachable);
            defer allocator.free(bk);
            const bv = try sh.readF32(allocator, std.fmt.bufPrint(&buf, "model.layers.{d}.self_attn.v_proj.bias", .{li}) catch unreachable);
            defer allocator.free(bv);
            if (bq.len + bk.len + bv.len != cfg.qkvDim()) return Error.DimensionMismatch;
            const all = try allocator.alloc(f32, cfg.qkvDim());
            @memcpy(all[0..bq.len], bq);
            @memcpy(all[bq.len..][0..bk.len], bk);
            @memcpy(all[bq.len + bk.len ..][0..bv.len], bv);
            n.qkv_bias = all;
        }
    }
    return rt;
}

pub fn loadLayer(allocator: std.mem.Allocator, sh: *const Shards, cfg: model.Config, index: u32) !model.Matrices {
    var m = model.Matrices{};
    errdefer m.deinit(allocator);
    var buf: [192]u8 = undefined;
    const q = try linear(allocator, sh, std.fmt.bufPrint(&buf, "model.layers.{d}.self_attn.q_proj.weight", .{index}) catch unreachable, cfg.hidden, cfg.qDim());
    defer allocator.free(q);
    const k = try linear(allocator, sh, std.fmt.bufPrint(&buf, "model.layers.{d}.self_attn.k_proj.weight", .{index}) catch unreachable, cfg.hidden, cfg.kvDim());
    defer allocator.free(k);
    const v = try linear(allocator, sh, std.fmt.bufPrint(&buf, "model.layers.{d}.self_attn.v_proj.weight", .{index}) catch unreachable, cfg.hidden, cfg.kvDim());
    defer allocator.free(v);
    m.qkv = try allocator.alloc(f16, q.len + k.len + v.len);
    @memcpy(m.qkv[0..q.len], q);
    @memcpy(m.qkv[q.len..][0..k.len], k);
    @memcpy(m.qkv[q.len + k.len ..][0..v.len], v);
    m.o = try linear(allocator, sh, std.fmt.bufPrint(&buf, "model.layers.{d}.self_attn.o_proj.weight", .{index}) catch unreachable, cfg.qDim(), cfg.hidden);
    if (cfg.layerIsSparse(index)) {
        // The routed experts go in `m.moe` for the CPU to run; the shared expert takes
        // the dense gate/up/down slots so the ANE's existing FFN kernel serves it.
        m.moe = try loadMoeLayer(allocator, sh, cfg, index);
        if (cfg.shared_inter > 0) {
            m.gate = try linear(allocator, sh, std.fmt.bufPrint(&buf, "model.layers.{d}.mlp.shared_expert.gate_proj.weight", .{index}) catch unreachable, cfg.hidden, cfg.shared_inter);
            m.up = try linear(allocator, sh, std.fmt.bufPrint(&buf, "model.layers.{d}.mlp.shared_expert.up_proj.weight", .{index}) catch unreachable, cfg.hidden, cfg.shared_inter);
            m.down = try linear(allocator, sh, std.fmt.bufPrint(&buf, "model.layers.{d}.mlp.shared_expert.down_proj.weight", .{index}) catch unreachable, cfg.shared_inter, cfg.hidden);
        }
    } else {
        m.gate = try linear(allocator, sh, std.fmt.bufPrint(&buf, "model.layers.{d}.mlp.gate_proj.weight", .{index}) catch unreachable, cfg.hidden, cfg.inter);
        m.up = try linear(allocator, sh, std.fmt.bufPrint(&buf, "model.layers.{d}.mlp.up_proj.weight", .{index}) catch unreachable, cfg.hidden, cfg.inter);
        m.down = try linear(allocator, sh, std.fmt.bufPrint(&buf, "model.layers.{d}.mlp.down_proj.weight", .{index}) catch unreachable, cfg.inter, cfg.hidden);
    }
    return m;
}

pub fn loadHead(allocator: std.mem.Allocator, sh: *const Shards, cfg: model.Config, embed: []const f16) !model.HeadWeights {
    if (sh.find("lm_head.weight") == null) return .{ .data = embed, .owned = false };
    return .{ .data = try linear(allocator, sh, "lm_head.weight", cfg.hidden, cfg.vocab), .owned = true };
}

pub const HfLayers = struct {
    shards: *const Shards,
    cfg: model.Config,

    pub fn source(self: *HfLayers) model.LayerSource {
        return .{ .ctx = self, .loadFn = loadFn };
    }
    fn loadFn(ctx: *anyopaque, allocator: std.mem.Allocator, index: u32) anyerror!model.Matrices {
        const self: *HfLayers = @ptrCast(@alignCast(ctx));
        return loadLayer(allocator, self.shards, self.cfg, index);
    }
};

pub const HfHead = struct {
    shards: *const Shards,
    cfg: model.Config,
    embed: []const f16,

    pub fn source(self: *HfHead) model.HeadSource {
        return .{ .ctx = self, .loadFn = loadFn };
    }
    fn loadFn(ctx: *anyopaque, allocator: std.mem.Allocator) anyerror!model.HeadWeights {
        const self: *HfHead = @ptrCast(@alignCast(ctx));
        return loadHead(allocator, self.shards, self.cfg, self.embed);
    }
};

test "hf config maps to the engine config with half-split RoPE" {
    const c = hf.Config{
        .arch = "Qwen2ForCausalLM",
        .hidden_size = 896,
        .num_hidden_layers = 24,
        .num_attention_heads = 14,
        .num_key_value_heads = 2,
        .head_dim = 64,
        .intermediate_size = 4864,
        .vocab_size = 151936,
        .num_experts = 0,
        .num_experts_per_tok = 0,
        .moe_intermediate_size = 0,
        .shared_expert_intermediate_size = 0,
        .norm_topk_prob = false,
        .decoder_sparse_step = 1,
        .mlp_only_layers = 0,
        .attn_logit_softcapping = 0,
        .final_logit_softcapping = 0,
        .sliding_window = 0,
        .sliding_window_size = 0,
        .rms_norm_eps = 1e-6,
        .rope_theta = 1e6,
        .tie_word_embeddings = false,
        .max_position_embeddings = 32768,
        .bos_token_id = null,
        .eos_token_id = null,
    };
    const m = try toModelConfig(c);
    try std.testing.expectEqual(@as(u32, 896), m.hidden);
    try std.testing.expectEqual(@as(u32, 1152), m.qkvDim());
    try std.testing.expectEqualStrings("qwen2", m.arch);
    try std.testing.expect(!m.rope_adjacent);
}

/// Load every layer plus the runtime weights into a `ModelWeights`, for the
/// CPU reference in `anedvd verify` and `anedvd cpu`.
///
/// Mirrors load_gguf.loadWeights: the reference needs the whole model resident,
/// which is the opposite of what the engine does (it streams one layer at a
/// time), so this is deliberately a separate, simple path.
pub fn loadWeights(allocator: std.mem.Allocator, sh: *const Shards, cfg: model.Config, progress: bool) !model.ModelWeights {
    var mw = model.ModelWeights{ .allocator = allocator, .config = cfg };
    errdefer mw.deinit();
    if (progress) sys.print("loading HF weights: {d} layers\\n", .{cfg.layers});

    mw.embed = try linear(allocator, sh, "model.embed_tokens.weight", cfg.hidden, cfg.vocab);
    mw.final_norm = try norm(allocator, sh, "model.norm.weight", cfg.hidden);
    if (sh.find("lm_head.weight") != null) {
        mw.lm_head = try linear(allocator, sh, "lm_head.weight", cfg.hidden, cfg.vocab);
    }

    mw.layers = try allocator.alloc(model.LayerWeights, cfg.layers);
    @memset(mw.layers, .{});
    var buf: [192]u8 = undefined;
    for (mw.layers, 0..) |*lw, i| {
        const li: u32 = @intCast(i);
        lw.attn_norm = try norm(allocator, sh, std.fmt.bufPrint(&buf, "model.layers.{d}.input_layernorm.weight", .{li}) catch unreachable, cfg.hidden);
        lw.ffn_norm = try norm(allocator, sh, std.fmt.bufPrint(&buf, "model.layers.{d}.post_attention_layernorm.weight", .{li}) catch unreachable, cfg.hidden);

        const q = try linear(allocator, sh, std.fmt.bufPrint(&buf, "model.layers.{d}.self_attn.q_proj.weight", .{li}) catch unreachable, cfg.hidden, cfg.qDim());
        defer allocator.free(q);
        const k = try linear(allocator, sh, std.fmt.bufPrint(&buf, "model.layers.{d}.self_attn.k_proj.weight", .{li}) catch unreachable, cfg.hidden, cfg.kvDim());
        defer allocator.free(k);
        const v = try linear(allocator, sh, std.fmt.bufPrint(&buf, "model.layers.{d}.self_attn.v_proj.weight", .{li}) catch unreachable, cfg.hidden, cfg.kvDim());
        defer allocator.free(v);
        lw.qkv = try allocator.alloc(f16, q.len + k.len + v.len);
        @memcpy(lw.qkv[0..q.len], q);
        @memcpy(lw.qkv[q.len..][0..k.len], k);
        @memcpy(lw.qkv[q.len + k.len ..][0..v.len], v);

        lw.o = try linear(allocator, sh, std.fmt.bufPrint(&buf, "model.layers.{d}.self_attn.o_proj.weight", .{li}) catch unreachable, cfg.qDim(), cfg.hidden);
        if (cfg.layerIsSparse(li)) {
            // Sparse layer. The routed experts go in `moe` (they run on the CPU), and
            // the SHARED expert doubles as this layer's dense FFN: every token passes
            // through it, so putting it in gate/up/down lets the existing ANE FFN
            // kernel serve it unchanged, and the engine only adds the routed
            // experts' contribution on top.
            lw.moe = try loadMoeLayer(allocator, sh, cfg, li);
            if (cfg.shared_inter > 0) {
                const sg = try linear(allocator, sh, std.fmt.bufPrint(&buf, "model.layers.{d}.mlp.shared_expert.gate_proj.weight", .{li}) catch unreachable, cfg.hidden, cfg.shared_inter);
                defer allocator.free(sg);
                const su = try linear(allocator, sh, std.fmt.bufPrint(&buf, "model.layers.{d}.mlp.shared_expert.up_proj.weight", .{li}) catch unreachable, cfg.hidden, cfg.shared_inter);
                defer allocator.free(su);
                const sd = try linear(allocator, sh, std.fmt.bufPrint(&buf, "model.layers.{d}.mlp.shared_expert.down_proj.weight", .{li}) catch unreachable, cfg.shared_inter, cfg.hidden);
                defer allocator.free(sd);
                lw.gate = sg;
                lw.up = su;
                lw.down = sd;
            }
        } else {
            lw.gate = try linear(allocator, sh, std.fmt.bufPrint(&buf, "model.layers.{d}.mlp.gate_proj.weight", .{li}) catch unreachable, cfg.hidden, cfg.inter);
            lw.up = try linear(allocator, sh, std.fmt.bufPrint(&buf, "model.layers.{d}.mlp.up_proj.weight", .{li}) catch unreachable, cfg.hidden, cfg.inter);
            lw.down = try linear(allocator, sh, std.fmt.bufPrint(&buf, "model.layers.{d}.mlp.down_proj.weight", .{li}) catch unreachable, cfg.inter, cfg.hidden);
        }

        if (sh.find(std.fmt.bufPrint(&buf, "model.layers.{d}.self_attn.q_norm.weight", .{li}) catch unreachable) != null) {
            lw.q_norm = try norm(allocator, sh, std.fmt.bufPrint(&buf, "model.layers.{d}.self_attn.q_norm.weight", .{li}) catch unreachable, cfg.head_dim);
            lw.k_norm = try norm(allocator, sh, std.fmt.bufPrint(&buf, "model.layers.{d}.self_attn.k_norm.weight", .{li}) catch unreachable, cfg.head_dim);
        }
        if (sh.find(std.fmt.bufPrint(&buf, "model.layers.{d}.self_attn.q_proj.bias", .{li}) catch unreachable) != null) {
            const bq = try sh.readF32(allocator, std.fmt.bufPrint(&buf, "model.layers.{d}.self_attn.q_proj.bias", .{li}) catch unreachable);
            defer allocator.free(bq);
            const bk = try sh.readF32(allocator, std.fmt.bufPrint(&buf, "model.layers.{d}.self_attn.k_proj.bias", .{li}) catch unreachable);
            defer allocator.free(bk);
            const bv = try sh.readF32(allocator, std.fmt.bufPrint(&buf, "model.layers.{d}.self_attn.v_proj.bias", .{li}) catch unreachable);
            defer allocator.free(bv);
            if (bq.len + bk.len + bv.len != cfg.qkvDim()) return Error.DimensionMismatch;
            const all = try allocator.alloc(f32, cfg.qkvDim());
            @memcpy(all[0..bq.len], bq);
            @memcpy(all[bq.len..][0..bk.len], bk);
            @memcpy(all[bq.len + bk.len ..][0..bv.len], bv);
            lw.qkv_bias = all;
        }
    }
    return mw;
}

/// Load the sparse block of a MoE layer from safetensors.
///
/// The expert tensors are `model.layers.N.mlp.experts.{e}.{gate,up,down}_proj.weight`,
/// one per expert. They are stacked into a single [num_experts][inter][hidden]
/// allocation so the routing path can index an expert with a slice instead of
/// chasing per-expert allocations through a hash map on every token.
pub fn loadMoeLayer(allocator: std.mem.Allocator, sh: *const Shards, cfg: model.Config, index: u32) !model.MoeWeights {
    var moe = model.MoeWeights{
        .num_experts = cfg.num_experts,
        .inter = cfg.moe_inter,
        .shared_inter = cfg.shared_inter,
        .hidden_dim = cfg.hidden,
    };
    errdefer moe.deinit(allocator);

    const per_gate = @as(usize, cfg.moe_inter) * cfg.hidden;
    var buf: [192]u8 = undefined;
    moe.router = try linear(allocator, sh, std.fmt.bufPrint(&buf, "model.layers.{d}.mlp.gate.weight", .{index}) catch unreachable, cfg.hidden, cfg.num_experts);
    moe.gate = try allocator.alloc(f16, per_gate * cfg.num_experts);
    moe.up = try allocator.alloc(f16, per_gate * cfg.num_experts);
    moe.down = try allocator.alloc(f16, per_gate * cfg.num_experts);

    for (0..cfg.num_experts) |e| {
        const ge = try linear(allocator, sh, std.fmt.bufPrint(&buf, "model.layers.{d}.mlp.experts.{d}.gate_proj.weight", .{ index, e }) catch unreachable, cfg.hidden, cfg.moe_inter);
        defer allocator.free(ge);
        const ue = try linear(allocator, sh, std.fmt.bufPrint(&buf, "model.layers.{d}.mlp.experts.{d}.up_proj.weight", .{ index, e }) catch unreachable, cfg.hidden, cfg.moe_inter);
        defer allocator.free(ue);
        const de = try linear(allocator, sh, std.fmt.bufPrint(&buf, "model.layers.{d}.mlp.experts.{d}.down_proj.weight", .{ index, e }) catch unreachable, cfg.moe_inter, cfg.hidden);
        defer allocator.free(de);
        @memcpy(moe.gate[e * per_gate ..][0..per_gate], ge);
        @memcpy(moe.up[e * per_gate ..][0..per_gate], ue);
        @memcpy(moe.down[e * per_gate ..][0..per_gate], de);
    }

    if (cfg.shared_inter > 0) {
        moe.shared_gate = try linear(allocator, sh, std.fmt.bufPrint(&buf, "model.layers.{d}.mlp.shared_expert.gate_proj.weight", .{index}) catch unreachable, cfg.hidden, cfg.shared_inter);
        moe.shared_up = try linear(allocator, sh, std.fmt.bufPrint(&buf, "model.layers.{d}.mlp.shared_expert.up_proj.weight", .{index}) catch unreachable, cfg.hidden, cfg.shared_inter);
        moe.shared_down = try linear(allocator, sh, std.fmt.bufPrint(&buf, "model.layers.{d}.mlp.shared_expert.down_proj.weight", .{index}) catch unreachable, cfg.shared_inter, cfg.hidden);
        moe.shared_gate_lin = try linear(allocator, sh, std.fmt.bufPrint(&buf, "model.layers.{d}.mlp.shared_expert_gate.weight", .{index}) catch unreachable, cfg.hidden, 1);
    }
    return moe;
}
