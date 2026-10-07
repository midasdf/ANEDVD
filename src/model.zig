// model.zig — format-independent model description and weight container.
//
// Loaders (GGUF, safetensors) fill this in; the engine consumes it. All linear
// weights are stored in the ANE's conv layout: row-major [out_features][in_features],
// fp16. GGUF stores the transpose of that, so the GGUF loader transposes while
// dequantising; HF safetensors already matches.

const std = @import("std");

pub const Config = struct {
    /// ggml architecture name, e.g. "llama", "qwen2", "qwen3".
    arch: []const u8 = "llama",
    hidden: u32 = 0,
    layers: u32 = 0,
    heads: u32 = 0,
    kv_heads: u32 = 0,
    head_dim: u32 = 0,
    inter: u32 = 0,
    vocab: u32 = 0,
    eps: f32 = 1e-5,
    rope_theta: f32 = 10000.0,
    tie_embeddings: bool = false,
    /// GGUF (llama.cpp) permutes Q/K rows into the adjacent-pair RoPE layout;
    /// HF safetensors keeps the half-split layout. Set by the loader.
    rope_adjacent: bool = false,

    pub fn qDim(self: Config) u32 {
        return self.heads * self.head_dim;
    }
    pub fn kvDim(self: Config) u32 {
        return self.kv_heads * self.head_dim;
    }
    pub fn qkvDim(self: Config) u32 {
        return self.qDim() + 2 * self.kvDim();
    }
    pub fn validate(self: Config) !void {
        if (self.hidden == 0 or self.layers == 0 or self.heads == 0 or
            self.head_dim == 0 or self.inter == 0 or self.vocab == 0)
            return error.IncompleteConfig;
        if (self.kv_heads == 0) return error.IncompleteConfig;
        if (self.heads % self.kv_heads != 0) return error.InvalidGqaGroup;
        if (self.heads * self.head_dim != self.hidden) {
            // Qwen-style models can have head_dim * heads != hidden; that is fine
            // as long as the projections are consistent, so only warn via error
            // when head_dim was never derived.
        }
    }
};

pub const LayerWeights = struct {
    attn_norm: []f32 = &.{}, // [hidden]
    ffn_norm: []f32 = &.{}, // [hidden]
    /// [(q + k + v) dims][hidden]
    qkv: []f16 = &.{},
    /// [hidden][q_dim]
    o: []f16 = &.{},
    /// [inter][hidden]
    gate: []f16 = &.{},
    /// [inter][hidden]
    up: []f16 = &.{},
    /// [hidden][inter]
    down: []f16 = &.{},
    /// Optional attention biases (Qwen2 adds them, Llama does not): [qkv dims].
    qkv_bias: ?[]f32 = null,
    /// Optional attention-output bias: [hidden].
    o_bias: ?[]f32 = null,

    pub fn deinit(self: *LayerWeights, allocator: std.mem.Allocator) void {
        if (self.attn_norm.len > 0) allocator.free(self.attn_norm);
        if (self.ffn_norm.len > 0) allocator.free(self.ffn_norm);
        if (self.qkv.len > 0) allocator.free(self.qkv);
        if (self.o.len > 0) allocator.free(self.o);
        if (self.gate.len > 0) allocator.free(self.gate);
        if (self.up.len > 0) allocator.free(self.up);
        if (self.down.len > 0) allocator.free(self.down);
        if (self.qkv_bias) |b| allocator.free(b);
        if (self.o_bias) |b| allocator.free(b);
        self.* = .{};
    }
};

pub const ModelWeights = struct {
    allocator: std.mem.Allocator,
    config: Config,
    /// [vocab][hidden]
    embed: []f16 = &.{},
    /// [hidden]
    final_norm: []f32 = &.{},
    /// [vocab][hidden]; null means "tied to embed".
    lm_head: ?[]f16 = null,
    layers: []LayerWeights = &.{},

    pub fn deinit(self: *ModelWeights) void {
        if (self.embed.len > 0) self.allocator.free(self.embed);
        if (self.final_norm.len > 0) self.allocator.free(self.final_norm);
        if (self.lm_head) |h| self.allocator.free(h);
        for (self.layers) |*l| l.deinit(self.allocator);
        if (self.layers.len > 0) self.allocator.free(self.layers);
        self.* = undefined;
    }

    pub fn headWeights(self: *const ModelWeights) []const f16 {
        return self.lm_head orelse self.embed;
    }
};

test "config dims" {
    const c = Config{ .hidden = 896, .layers = 24, .heads = 14, .kv_heads = 2, .head_dim = 64, .inter = 4864, .vocab = 151936 };
    try std.testing.expectEqual(@as(u32, 896), c.qDim());
    try std.testing.expectEqual(@as(u32, 128), c.kvDim());
    try std.testing.expectEqual(@as(u32, 1152), c.qkvDim());
    try c.validate();
}
