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
    try cfg.validate();
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
        lw.gate = try loadLinear(allocator, g, std.fmt.bufPrint(&buf, "blk.{d}.ffn_gate.weight", .{li}) catch unreachable, cfg.hidden, cfg.inter);
        lw.up = try loadLinear(allocator, g, std.fmt.bufPrint(&buf, "blk.{d}.ffn_up.weight", .{li}) catch unreachable, cfg.hidden, cfg.inter);
        lw.down = try loadLinear(allocator, g, std.fmt.bufPrint(&buf, "blk.{d}.ffn_down.weight", .{li}) catch unreachable, cfg.inter, cfg.hidden);
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
