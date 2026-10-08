// load_gguf.zig — turn a GGUF file into the engine's ModelWeights.
//
// Layout note: ggml stores a linear layer as ne = [in_features, out_features]
// while the payload is the original PyTorch [out_features, in_features] array in
// C order, i.e. `w[o * in + i]`. That is exactly the ANE 1x1-conv weight layout,
// so no transpose is needed. The dimension order is validated and a defensive
// transpose is applied if a file ever disagrees.

const std = @import("std");
const gguf = @import("gguf.zig");
const model = @import("model.zig");
const sys = @import("sys.zig");

pub const Error = error{
    MissingTensor,
    DimensionMismatch,
    UnsupportedArchitecture,
    UnsupportedQuantization,
};

/// Architectures this loader has been exercised on, or that share their GGUF
/// layout exactly (the same `*.block_count` config keys and `blk.N.*` tensor
/// names). Anything else is accepted but warned about, because a wrong
/// architecture guess produces fluent nonsense rather than an error.
pub const known_architectures = [_][]const u8{
    "llama",    "qwen2",   "qwen3",    "mistral",   "smollm",
    "smollm2",  "smollm3", "qwen2moe", "qwen3moe",  "granite",
    "stablelm", "olmo",    "olmo2",    "phi3",      "gemma",
    "gemma2",   "gemma3",  "gptneox",  "internlm2",
};

pub fn isKnownArchitecture(arch: []const u8) bool {
    for (known_architectures) |a| if (std.mem.eql(u8, arch, a)) return true;
    return false;
}

/// Architectures whose HF implementation uses `rotate_half` (half-split RoPE).
/// llama.cpp selects `LLAMA_ROPE_TYPE_NEOX` for these; everything else uses the
/// adjacent-pair layout, which is also the layout `convert_hf_to_gguf.py`
/// permutes Llama-family Q/K rows into.
///
/// Verified locally: llama -> adjacent (SmolLM2-135M), qwen2 -> half-split
/// (Qwen2.5-0.5B; the adjacent convention yields repetitive text). The rest of
/// this list mirrors llama.cpp's rope_type table but has not been reproduced
/// here — override with --rope-hf / --rope-adjacent if a model misbehaves.
const neox_archs = [_][]const u8{
    "qwen2",  "qwen2moe", "qwen2vl",  "qwen3", "qwen3moe",
    "phi3",   "gptneox",  "stablelm", "gemma", "gemma2",
    "gemma3", "olmo",     "olmo2",
};

pub fn ropeIsAdjacent(arch: []const u8) bool {
    for (neox_archs) |a| if (std.mem.eql(u8, arch, a)) return false;
    return true;
}

fn key(buf: []u8, arch: []const u8, suffix: []const u8) []const u8 {
    return std.fmt.bufPrint(buf, "{s}.{s}", .{ arch, suffix }) catch unreachable;
}

pub fn loadConfig(g: *const gguf.Gguf) !model.Config {
    var buf: [128]u8 = undefined;
    const arch = g.arch() orelse "llama";

    var cfg = model.Config{ .arch = arch };
    cfg.hidden = g.getU32(key(&buf, arch, "embedding_length")) orelse return Error.MissingTensor;
    cfg.layers = g.getU32(key(&buf, arch, "block_count")) orelse return Error.MissingTensor;
    cfg.heads = g.getU32(key(&buf, arch, "attention.head_count")) orelse return Error.MissingTensor;
    cfg.kv_heads = g.getU32(key(&buf, arch, "attention.head_count_kv")) orelse cfg.heads;
    cfg.inter = g.getU32(key(&buf, arch, "feed_forward_length")) orelse return Error.MissingTensor;
    cfg.head_dim = g.getU32(key(&buf, arch, "attention.key_length")) orelse (cfg.hidden / cfg.heads);
    cfg.eps = g.getF32(key(&buf, arch, "attention.layer_norm_rms_epsilon")) orelse 1e-5;
    cfg.rope_theta = g.getF32(key(&buf, arch, "rope.freq_base")) orelse 10000.0;
    cfg.vocab = g.getU32(key(&buf, arch, "vocab_size")) orelse blk: {
        const toks = g.getStringArray("tokenizer.ggml.tokens") orelse return Error.MissingTensor;
        break :blk @intCast(toks.len);
    };
    cfg.tie_embeddings = g.tensor("output.weight") == null;
    cfg.rope_adjacent = ropeIsAdjacent(arch);

    // MoE metadata (llama.cpp: <arch>.expert_count / .expert_used_count /
    // .expert_shared_count / .expert_feed_forward_length).
    cfg.num_experts = g.getU32(key(&buf, arch, "expert_count")) orelse 0;
    cfg.experts_per_tok = g.getU32(key(&buf, arch, "expert_used_count")) orelse 0;
    if (cfg.num_experts > 0) {
        if (cfg.experts_per_tok == 0) {
            sys.eprint("warning: {s} declares {d} experts but no expert_used_count; assuming 2.\n", .{ arch, cfg.num_experts });
            cfg.experts_per_tok = 2;
        }
        // The tensor shapes are authoritative, not the metadata. Qwen1.5-MoE ships
        // no expert_feed_forward_length at all, and its feed_forward_length (5632)
        // is the SHARED expert's width while the routed experts are 1408 — so
        // defaulting one from the other silently sizes every expert wrong, or
        // skips the shared expert entirely. Read the widths off the tensors the
        // same way llama.cpp does.
        const gate_exps = g.tensor(std.fmt.bufPrint(&buf, "blk.0.ffn_gate_exps.weight", .{}) catch unreachable);
        if (gate_exps) |t| {
            if (t.dims.len == 3) {
                // {hidden, moe_inter, num_experts} with the expert axis last.
                cfg.moe_inter = @intCast(t.dims[1]);
                if (t.dims[2] != cfg.num_experts) {
                    sys.eprint("warning: expert_count is {d} but ffn_gate_exps has {d} experts; trusting the tensor.\n", .{ cfg.num_experts, t.dims[2] });
                    cfg.num_experts = @intCast(t.dims[2]);
                }
            }
        } else {
            cfg.moe_inter = g.getU32(key(&buf, arch, "expert_feed_forward_length")) orelse cfg.inter;
        }
        // The shared expert's width comes from its tensor when the metadata omits it
        // (Qwen1.5-MoE does). A 0 here would silently drop the shared expert from
        // the forward pass, which is a wrong answer rather than a slow one.
        cfg.shared_inter = g.getU32(key(&buf, arch, "expert_shared_feed_forward_length")) orelse blk: {
            const shexp = g.tensor(std.fmt.bufPrint(&buf, "blk.0.ffn_gate_shexp.weight", .{}) catch unreachable) orelse break :blk 0;
            // {hidden, inter}: ggml's ne[0] is hidden and varies fastest, so the
            // shared width is dims[1]. Reading dims[0] yields the hidden size, which
            // is plausible enough to pass unnoticed.
            break :blk if (shexp.dims.len >= 2) @as(u32, @intCast(shexp.dims[1])) else cfg.inter;
        };
    }
    try cfg.validate();

    // A wrong guess here is silent: the model still runs and produces fluent
    // text. Say so once, loudly enough to be seen in a log.
    if (!isKnownArchitecture(arch)) {
        sys.eprint("warning: architecture \"{s}\" is not one this loader has been tested with.\n", .{arch});
        sys.eprint("  Treating it like llama: blk.N.* tensor names, RoPE {s}.\n", .{
            if (cfg.rope_adjacent) "adjacent-pair" else "half-split",
        });
        sys.eprint("  If the output is fluent nonsense, force the other convention with --rope-hf / --rope-adjacent.\n", .{});
    }
    return cfg;
}

/// Load a Linear weight as fp16 in ANE conv layout [out][in].
fn loadLinear(allocator: std.mem.Allocator, g: *const gguf.Gguf, name: []const u8, in_dim: u32, out_dim: u32) ![]f16 {
    const t = g.tensor(name) orelse return Error.MissingTensor;
    const want: u64 = @as(u64, in_dim) * out_dim;
    if (t.elemCount() != want) return Error.DimensionMismatch;

    const src = try g.readF16(allocator, name);
    errdefer allocator.free(src);

    if (t.dims.len >= 2 and t.dims[0] == in_dim and t.dims[1] == out_dim) {
        return src; // already [out][in]
    }
    if (t.dims.len >= 2 and t.dims[0] == out_dim and t.dims[1] == in_dim) {
        const dst = try allocator.alloc(f16, src.len);
        for (0..out_dim) |o| {
            for (0..in_dim) |i| dst[o * in_dim + i] = src[i * out_dim + o];
        }
        allocator.free(src);
        return dst;
    }
    return Error.DimensionMismatch;
}

/// Zero-copy view of an fp16 tensor's payload inside the mapped file.
///
/// The view aliases the mapping, so the caller's `Gguf` must outlive the
/// `Runtime` that holds it. That is why `model_open.Loaded` keeps the `Gguf`
/// alive next to the weights, and why `loadWeights` (used for the CPU
/// reference, which outlives the file) copies through `readF16` instead.
fn viewF16(g: *const gguf.Gguf, name: []const u8, in_dim: u32, out_dim: u32) ?[]f16 {
    const t = g.tensor(name) orelse return null;
    if (t.ttype != .f16) return null;
    if (t.elemCount() != @as(u64, in_dim) * out_dim) return null;
    if (t.dims.len < 2 or t.dims[0] != in_dim or t.dims[1] != out_dim) return null;
    const bytes = g.tensorBytes(t) catch return null;
    if (bytes.len % 2 != 0) return null;
    const aligned: []align(2) const u8 = @alignCast(bytes);
    return @constCast(std.mem.bytesAsSlice(f16, aligned));
}

fn loadNorm(allocator: std.mem.Allocator, g: *const gguf.Gguf, name: []const u8, n: u32) ![]f32 {
    const v = try g.readF32(allocator, name);
    if (v.len != n) {
        allocator.free(v);
        return Error.DimensionMismatch;
    }
    return v;
}

pub fn loadWeights(allocator: std.mem.Allocator, g: *const gguf.Gguf, progress: bool) !model.ModelWeights {
    const cfg = try loadConfig(g);
    var mw = model.ModelWeights{ .allocator = allocator, .config = cfg };
    errdefer mw.deinit();

    const hidden: usize = cfg.hidden;
    const q_dim: usize = cfg.qDim();
    const kv_dim: usize = cfg.kvDim();
    const inter: usize = cfg.inter;

    if (progress) sys.print("loading weights: {d} layers, hidden {d}, heads {d}/{d}, inter {d}, vocab {d}\n", .{ cfg.layers, cfg.hidden, cfg.heads, cfg.kv_heads, cfg.inter, cfg.vocab });

    mw.embed = try loadLinear(allocator, g, "token_embd.weight", cfg.hidden, cfg.vocab);
    mw.final_norm = try loadNorm(allocator, g, "output_norm.weight", cfg.hidden);
    if (g.tensor("output.weight") != null) {
        mw.lm_head = try loadLinear(allocator, g, "output.weight", cfg.hidden, cfg.vocab);
    }

    mw.layers = try allocator.alloc(model.LayerWeights, cfg.layers);
    @memset(mw.layers, .{});
    var buf: [128]u8 = undefined;
    for (mw.layers, 0..) |*lw, i| {
        const li: u32 = @intCast(i);
        if (progress and (i % 4 == 0 or i + 1 == cfg.layers)) {
            sys.print("  layer {d}/{d}\n", .{ i + 1, cfg.layers });
        }
        lw.attn_norm = try loadNorm(allocator, g, std.fmt.bufPrint(&buf, "blk.{d}.attn_norm.weight", .{li}) catch unreachable, cfg.hidden);
        lw.ffn_norm = try loadNorm(allocator, g, std.fmt.bufPrint(&buf, "blk.{d}.ffn_norm.weight", .{li}) catch unreachable, cfg.hidden);

        const q = try loadLinear(allocator, g, std.fmt.bufPrint(&buf, "blk.{d}.attn_q.weight", .{li}) catch unreachable, cfg.hidden, cfg.qDim());
        defer allocator.free(q);
        const k = try loadLinear(allocator, g, std.fmt.bufPrint(&buf, "blk.{d}.attn_k.weight", .{li}) catch unreachable, cfg.hidden, cfg.kvDim());
        defer allocator.free(k);
        const v = try loadLinear(allocator, g, std.fmt.bufPrint(&buf, "blk.{d}.attn_v.weight", .{li}) catch unreachable, cfg.hidden, cfg.kvDim());
        defer allocator.free(v);

        lw.qkv = try allocator.alloc(f16, q.len + k.len + v.len);
        @memcpy(lw.qkv[0..q.len], q);
        @memcpy(lw.qkv[q.len..][0..k.len], k);
        @memcpy(lw.qkv[q.len + k.len ..][0..v.len], v);

        lw.o = try loadLinear(allocator, g, std.fmt.bufPrint(&buf, "blk.{d}.attn_output.weight", .{li}) catch unreachable, cfg.qDim(), cfg.hidden);
        // Qwen3: per-head q/k normalisation, needed by the CPU reference too.
        if (g.tensor(std.fmt.bufPrint(&buf, "blk.{d}.attn_q_norm.weight", .{li}) catch unreachable) != null) {
            lw.q_norm = try loadNorm(allocator, g, std.fmt.bufPrint(&buf, "blk.{d}.attn_q_norm.weight", .{li}) catch unreachable, cfg.head_dim);
            lw.k_norm = try loadNorm(allocator, g, std.fmt.bufPrint(&buf, "blk.{d}.attn_k_norm.weight", .{li}) catch unreachable, cfg.head_dim);
        }

        // Qwen2 (and a few others) add biases to the attention projections.
        if (g.tensor(std.fmt.bufPrint(&buf, "blk.{d}.attn_q.bias", .{li}) catch unreachable) != null) {
            const bq = try g.readF32(allocator, std.fmt.bufPrint(&buf, "blk.{d}.attn_q.bias", .{li}) catch unreachable);
            defer allocator.free(bq);
            const bk = try g.readF32(allocator, std.fmt.bufPrint(&buf, "blk.{d}.attn_k.bias", .{li}) catch unreachable);
            defer allocator.free(bk);
            const bv = try g.readF32(allocator, std.fmt.bufPrint(&buf, "blk.{d}.attn_v.bias", .{li}) catch unreachable);
            defer allocator.free(bv);
            if (bq.len + bk.len + bv.len != cfg.qkvDim()) return Error.DimensionMismatch;
            const all = try allocator.alloc(f32, cfg.qkvDim());
            @memcpy(all[0..bq.len], bq);
            @memcpy(all[bq.len..][0..bk.len], bk);
            @memcpy(all[bq.len + bk.len ..][0..bv.len], bv);
            lw.qkv_bias = all;
        }
        if (g.tensor(std.fmt.bufPrint(&buf, "blk.{d}.attn_output.bias", .{li}) catch unreachable) != null) {
            lw.o_bias = try g.readF32(allocator, std.fmt.bufPrint(&buf, "blk.{d}.attn_output.bias", .{li}) catch unreachable);
        }
        if (cfg.layerIsSparse(li)) {
            // Sparse layer: the dense ffn_gate/up/down are the shared expert's.
            lw.moe = try loadMoeLayer(allocator, g, cfg, li);
        } else {
            lw.gate = try loadLinear(allocator, g, std.fmt.bufPrint(&buf, "blk.{d}.ffn_gate.weight", .{li}) catch unreachable, cfg.hidden, cfg.inter);
            lw.up = try loadLinear(allocator, g, std.fmt.bufPrint(&buf, "blk.{d}.ffn_up.weight", .{li}) catch unreachable, cfg.hidden, cfg.inter);
            lw.down = try loadLinear(allocator, g, std.fmt.bufPrint(&buf, "blk.{d}.ffn_down.weight", .{li}) catch unreachable, cfg.inter, cfg.hidden);
        }
    }

    _ = hidden;
    _ = q_dim;
    _ = kv_dim;
    _ = inter;
    return mw;
}

test "config keys are built per architecture" {
    var buf: [128]u8 = undefined;
    try std.testing.expectEqualStrings("qwen2.embedding_length", key(&buf, "qwen2", "embedding_length"));
    try std.testing.expectEqualStrings("llama.block_count", key(&buf, "llama", "block_count"));
}

// ---------------------------------------------------------------- streaming

/// Everything needed at run time, without the per-layer matrices.
pub fn loadRuntime(allocator: std.mem.Allocator, g: *const gguf.Gguf, cfg: model.Config, progress: bool) !model.Runtime {
    var rt = model.Runtime{ .allocator = allocator, .config = cfg };
    errdefer rt.deinit();
    if (progress) sys.print("  runtime weights: embedding + norms\n", .{});
    if (viewF16(g, "token_embd.weight", cfg.hidden, cfg.vocab)) |view| {
        rt.embed = view;
        rt.embed_owned = false; // aliases the mmap; nothing to free
        if (progress) sys.print("    embedding: zero-copy from the mapped file\n", .{});
    } else {
        rt.embed = try loadLinear(allocator, g, "token_embd.weight", cfg.hidden, cfg.vocab);
        rt.embed_owned = true;
    }
    rt.final_norm = try loadNorm(allocator, g, "output_norm.weight", cfg.hidden);
    rt.norms = try allocator.alloc(model.Norm, cfg.layers);
    @memset(rt.norms, .{});
    var buf: [128]u8 = undefined;
    for (rt.norms, 0..) |*n, i| {
        const li: u32 = @intCast(i);
        n.attn = try loadNorm(allocator, g, std.fmt.bufPrint(&buf, "blk.{d}.attn_norm.weight", .{li}) catch unreachable, cfg.hidden);
        n.ffn = try loadNorm(allocator, g, std.fmt.bufPrint(&buf, "blk.{d}.ffn_norm.weight", .{li}) catch unreachable, cfg.hidden);
        if (g.tensor(std.fmt.bufPrint(&buf, "blk.{d}.attn_q.bias", .{li}) catch unreachable) != null) {
            const bq = try g.readF32(allocator, std.fmt.bufPrint(&buf, "blk.{d}.attn_q.bias", .{li}) catch unreachable);
            defer allocator.free(bq);
            const bk = try g.readF32(allocator, std.fmt.bufPrint(&buf, "blk.{d}.attn_k.bias", .{li}) catch unreachable);
            defer allocator.free(bk);
            const bv = try g.readF32(allocator, std.fmt.bufPrint(&buf, "blk.{d}.attn_v.bias", .{li}) catch unreachable);
            defer allocator.free(bv);
            if (bq.len + bk.len + bv.len != cfg.qkvDim()) return Error.DimensionMismatch;
            const all = try allocator.alloc(f32, cfg.qkvDim());
            @memcpy(all[0..bq.len], bq);
            @memcpy(all[bq.len..][0..bk.len], bk);
            @memcpy(all[bq.len + bk.len ..][0..bv.len], bv);
            n.qkv_bias = all;
        }
        if (g.tensor(std.fmt.bufPrint(&buf, "blk.{d}.attn_output.bias", .{li}) catch unreachable) != null) {
            n.o_bias = try g.readF32(allocator, std.fmt.bufPrint(&buf, "blk.{d}.attn_output.bias", .{li}) catch unreachable);
        }
        // Qwen3 normalises each head's q/k before RoPE.
        if (g.tensor(std.fmt.bufPrint(&buf, "blk.{d}.attn_q_norm.weight", .{li}) catch unreachable) != null) {
            n.q_norm = try loadNorm(allocator, g, std.fmt.bufPrint(&buf, "blk.{d}.attn_q_norm.weight", .{li}) catch unreachable, cfg.head_dim);
            n.k_norm = try loadNorm(allocator, g, std.fmt.bufPrint(&buf, "blk.{d}.attn_k_norm.weight", .{li}) catch unreachable, cfg.head_dim);
        }
    }
    return rt;
}

/// One layer's compile-time-only matrices.
pub fn loadLayer(allocator: std.mem.Allocator, g: *const gguf.Gguf, cfg: model.Config, index: u32) !model.Matrices {
    var m = model.Matrices{};
    errdefer m.deinit(allocator);
    var buf: [128]u8 = undefined;
    const q = try loadLinear(allocator, g, std.fmt.bufPrint(&buf, "blk.{d}.attn_q.weight", .{index}) catch unreachable, cfg.hidden, cfg.qDim());
    defer allocator.free(q);
    const k = try loadLinear(allocator, g, std.fmt.bufPrint(&buf, "blk.{d}.attn_k.weight", .{index}) catch unreachable, cfg.hidden, cfg.kvDim());
    defer allocator.free(k);
    const v = try loadLinear(allocator, g, std.fmt.bufPrint(&buf, "blk.{d}.attn_v.weight", .{index}) catch unreachable, cfg.hidden, cfg.kvDim());
    defer allocator.free(v);
    m.qkv = try allocator.alloc(f16, q.len + k.len + v.len);
    @memcpy(m.qkv[0..q.len], q);
    @memcpy(m.qkv[q.len..][0..k.len], k);
    @memcpy(m.qkv[q.len + k.len ..][0..v.len], v);
    m.o = try loadLinear(allocator, g, std.fmt.bufPrint(&buf, "blk.{d}.attn_output.weight", .{index}) catch unreachable, cfg.qDim(), cfg.hidden);
    m.gate = try loadLinear(allocator, g, std.fmt.bufPrint(&buf, "blk.{d}.ffn_gate.weight", .{index}) catch unreachable, cfg.hidden, cfg.inter);
    m.up = try loadLinear(allocator, g, std.fmt.bufPrint(&buf, "blk.{d}.ffn_up.weight", .{index}) catch unreachable, cfg.hidden, cfg.inter);
    m.down = try loadLinear(allocator, g, std.fmt.bufPrint(&buf, "blk.{d}.ffn_down.weight", .{index}) catch unreachable, cfg.inter, cfg.hidden);
    return m;
}

/// lm-head weights; ties to the embedding when the file has no `output.weight`.
pub fn loadHead(allocator: std.mem.Allocator, g: *const gguf.Gguf, cfg: model.Config, embed: []const f16) !model.HeadWeights {
    if (g.tensor("output.weight") == null) return .{ .data = embed, .owned = false };
    return .{ .data = try loadLinear(allocator, g, "output.weight", cfg.hidden, cfg.vocab), .owned = true };
}

/// LayerSource backed by a GGUF file.
pub const GgufLayers = struct {
    g: *const gguf.Gguf,
    cfg: model.Config,

    pub fn source(self: *GgufLayers) model.LayerSource {
        return .{ .ctx = self, .loadFn = loadFn };
    }
    fn loadFn(ctx: *anyopaque, allocator: std.mem.Allocator, index: u32) anyerror!model.Matrices {
        const self: *GgufLayers = @ptrCast(@alignCast(ctx));
        return loadLayer(allocator, self.g, self.cfg, index);
    }
};

/// HeadSource backed by a GGUF file (falls back to the tied embedding).
pub const GgufHead = struct {
    g: *const gguf.Gguf,
    cfg: model.Config,
    embed: []const f16,

    pub fn source(self: *GgufHead) model.HeadSource {
        return .{ .ctx = self, .loadFn = loadFn };
    }
    fn loadFn(ctx: *anyopaque, allocator: std.mem.Allocator) anyerror!model.HeadWeights {
        const self: *GgufHead = @ptrCast(@alignCast(ctx));
        return loadHead(allocator, self.g, self.cfg, self.embed);
    }
};

/// Load the sparse block of a MoE layer from GGUF.
///
/// The expert tensors are 3-D (`blk.N.ffn_gate_exps` etc.) with the expert index
/// as the LAST declared dimension, which in ggml's layout means experts are the
/// slowest-varying axis and each expert's slice is contiguous. That is convenient
/// (one read, no gather) but the two orders are not the same for every tensor, so
/// each is checked against the expert's expected [out][in] shape rather than
/// assumed.
pub fn loadMoeLayer(allocator: std.mem.Allocator, g: *const gguf.Gguf, cfg: model.Config, index: u32) !model.MoeWeights {
    var buf: [128]u8 = undefined;
    var moe = model.MoeWeights{
        .num_experts = cfg.num_experts,
        .inter = cfg.moe_inter,
        .shared_inter = cfg.shared_inter,
        .hidden_dim = cfg.hidden,
    };
    errdefer moe.deinit(allocator);

    const per_gate = @as(usize, cfg.moe_inter) * cfg.hidden;
    moe.router = try loadLinear(allocator, g, std.fmt.bufPrint(&buf, "blk.{d}.ffn_gate_inp.weight", .{index}) catch unreachable, cfg.hidden, cfg.num_experts);
    moe.gate = try allocator.alloc(f16, per_gate * cfg.num_experts);
    moe.up = try allocator.alloc(f16, per_gate * cfg.num_experts);
    moe.down = try allocator.alloc(f16, per_gate * cfg.num_experts);

    try loadExperts(allocator, g, std.fmt.bufPrint(&buf, "blk.{d}.ffn_gate_exps.weight", .{index}) catch unreachable, moe.gate, cfg.num_experts, cfg.moe_inter, cfg.hidden);
    try loadExperts(allocator, g, std.fmt.bufPrint(&buf, "blk.{d}.ffn_up_exps.weight", .{index}) catch unreachable, moe.up, cfg.num_experts, cfg.moe_inter, cfg.hidden);
    try loadExperts(allocator, g, std.fmt.bufPrint(&buf, "blk.{d}.ffn_down_exps.weight", .{index}) catch unreachable, moe.down, cfg.num_experts, cfg.hidden, cfg.moe_inter);

    if (cfg.shared_inter > 0) {
        moe.shared_gate = try loadLinear(allocator, g, std.fmt.bufPrint(&buf, "blk.{d}.ffn_gate_shexp.weight", .{index}) catch unreachable, cfg.hidden, cfg.shared_inter);
        moe.shared_up = try loadLinear(allocator, g, std.fmt.bufPrint(&buf, "blk.{d}.ffn_up_shexp.weight", .{index}) catch unreachable, cfg.hidden, cfg.shared_inter);
        moe.shared_down = try loadLinear(allocator, g, std.fmt.bufPrint(&buf, "blk.{d}.ffn_down_shexp.weight", .{index}) catch unreachable, cfg.shared_inter, cfg.hidden);
        moe.shared_gate_lin = try loadLinear(allocator, g, std.fmt.bufPrint(&buf, "blk.{d}.ffn_gate_inp_shexp.weight", .{index}) catch unreachable, cfg.hidden, 1);
    }
    return moe;
}

/// Split a stacked expert tensor into `dst`, which is [num_experts][out][in].
///
/// `out`/`in` are what the CPU path needs, one expert at a time. The file may
/// store an expert as [out][in] or [in][out] depending on which dimension ggml
/// put first, so the declared dims decide; guessing produces plausible garbage.
fn loadExperts(
    allocator: std.mem.Allocator,
    g: *const gguf.Gguf,
    name: []const u8,
    dst: []f16,
    num_experts: u32,
    out_dim: u32,
    in_dim: u32,
) !void {
    const t = g.tensor(name) orelse return Error.MissingTensor;
    const per_expert: u64 = @as(u64, out_dim) * in_dim;
    if (t.elemCount() != per_expert * num_experts) return Error.DimensionMismatch;
    if (t.dims.len != 3) return Error.DimensionMismatch;

    // The expert axis is whichever dimension equals num_experts.
    var expert_axis: ?usize = null;
    for (t.dims, 0..) |d, i| {
        if (d == num_experts) {
            expert_axis = i;
            break;
        }
    }
    const axis = expert_axis orelse return Error.DimensionMismatch;

    // ggml's ne[0] varies fastest, so the element at (i0, i1, i2) is at
    // i0 + ne0*(i1 + ne1*i2). Requiring the expert axis to be last keeps each
    // expert's slice contiguous, which is the case llama.cpp writes.
    if (axis != 2) return Error.DimensionMismatch;

    const src = try g.readF16(allocator, name);
    defer allocator.free(src);

    // Within an expert the remaining dims are [d0, d1] with d0 fastest.
    const d0 = t.dims[0];
    const d1 = t.dims[1];
    if (d0 == in_dim and d1 == out_dim) {
        // [in-fastest, out] -> an expert's slice is out-major with in contiguous,
        // i.e. exactly [out][in]. Straight copy.
        for (0..num_experts) |e| {
            @memcpy(dst[e * per_expert ..][0..per_expert], src[e * per_expert ..][0..per_expert]);
        }
    } else if (d0 == out_dim and d1 == in_dim) {
        // [out-fastest, in] -> the slice is in-major; transpose it.
        for (0..num_experts) |e| {
            const slice = src[e * per_expert ..][0..per_expert];
            const out = dst[e * per_expert ..][0..per_expert];
            for (0..out_dim) |o| {
                for (0..in_dim) |i| {
                    out[o * in_dim + i] = slice[i * out_dim + o];
                }
            }
        }
    } else {
        return Error.DimensionMismatch;
    }
}
