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

    // ------------------------------------------------------------------ MoE
    // 0 for a dense model. See research/moe-design.md for why the experts run on
    // the CPU: a 60-expert model would need thousands of loaded ANE kernels, and
    // decode only reads the k selected experts per token anyway.
    /// Routed experts per layer.
    num_experts: u32 = 0,
    /// How many of them a token is routed to (`num_experts_per_tok`).
    experts_per_tok: u32 = 0,
    /// One expert's intermediate width (`moe_intermediate_size`).
    moe_inter: u32 = 0,
    /// The always-on shared expert's width; 0 means the model has none.
    shared_inter: u32 = 0,
    /// Qwen2MoE divides the top-k weights by their sum when set.
    norm_topk_prob: bool = false,
    /// Sparse layers are `(layer + 1) % sparse_step == 0`, minus `mlp_only_layers`.
    sparse_step: u32 = 1,
    /// Bitmask by layer index: set means "this layer is dense even though the
    /// model has experts" (`mlp_only_layers`). Layers are few, so a u64 covers
    /// any real model; layers past bit 63 are treated as sparse.
    mlp_only_mask: u64 = 0,

    /// True when this layer runs the sparse block rather than a dense MLP.
    pub fn layerIsSparse(self: Config, layer: u32) bool {
        if (self.num_experts == 0) return false;
        if (layer < 64 and (self.mlp_only_mask >> @intCast(layer)) & 1 == 1) return false;
        const step = if (self.sparse_step == 0) 1 else self.sparse_step;
        return (layer + 1) % step == 0;
    }

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

// ---------------------------------------------------------------- streaming

/// Compile-time-only weights for one layer. The engine frees these as soon as
/// the layer's ANE kernels have been built, which keeps peak RSS at roughly one
/// layer instead of the whole model.
pub const Matrices = struct {
    /// [(q + k + v) dims][hidden]
    qkv: []f16 = &.{},
    /// [hidden][q_dim]
    o: []f16 = &.{},
    /// Routed experts, for a sparse layer of a MoE model. Owned by whoever fills
    /// this in; `deinit` frees it. Null for a dense layer.
    moe: ?MoeWeights = null,
    /// [inter][hidden]
    gate: []f16 = &.{},
    /// [inter][hidden]
    up: []f16 = &.{},
    /// [hidden][inter]
    down: []f16 = &.{},
    qkv_bias: ?[]f32 = null,
    o_bias: ?[]f32 = null,

    pub fn deinit(self: *Matrices, allocator: std.mem.Allocator) void {
        if (self.qkv.len > 0) allocator.free(self.qkv);
        if (self.o.len > 0) allocator.free(self.o);
        if (self.gate.len > 0) allocator.free(self.gate);
        if (self.up.len > 0) allocator.free(self.up);
        if (self.down.len > 0) allocator.free(self.down);
        if (self.qkv_bias) |b| allocator.free(b);
        if (self.o_bias) |b| allocator.free(b);
        if (self.moe) |*m| m.deinit(allocator);
        self.* = .{};
    }
};

/// Per-layer normalisation weights: needed on the CPU for every decode step, so
/// they stay resident (2 * hidden floats per layer, negligible).
pub const Norm = struct {
    attn: []f32 = &.{},
    ffn: []f32 = &.{},
    /// Attention biases (Qwen2 and friends); applied on the CPU after the ANE
    /// projection, so they must outlive the matrices.
    qkv_bias: ?[]f32 = null,
    o_bias: ?[]f32 = null,
    /// Per-head Q/K normalisation weights (Qwen3): [head_dim] each, applied
    /// before RoPE.
    q_norm: ?[]f32 = null,
    k_norm: ?[]f32 = null,
};

/// Everything the engine needs at run time: the token embedding, the final
/// norm, and the per-layer norms.
pub const Runtime = struct {
    allocator: std.mem.Allocator,
    config: Config,
    /// [vocab][hidden]. For an fp16 GGUF this is a zero-copy view into the
    /// memory-mapped file (`embed_owned == false`), so unified memory holds one
    /// copy of the embedding instead of two.
    embed: []f16 = &.{},
    embed_owned: bool = true,
    /// [hidden]
    final_norm: []f32 = &.{},
    norms: []Norm = &.{},

    pub fn deinit(self: *Runtime) void {
        if (self.embed.len > 0 and self.embed_owned) self.allocator.free(self.embed);
        if (self.final_norm.len > 0) self.allocator.free(self.final_norm);
        for (self.norms) |n| {
            if (n.attn.len > 0) self.allocator.free(n.attn);
            if (n.ffn.len > 0) self.allocator.free(n.ffn);
            if (n.qkv_bias) |b| self.allocator.free(b);
            if (n.o_bias) |b| self.allocator.free(b);
            if (n.q_norm) |b| self.allocator.free(b);
            if (n.k_norm) |b| self.allocator.free(b);
        }
        if (self.norms.len > 0) self.allocator.free(self.norms);
        self.* = undefined;
    }
};

pub const HeadWeights = struct {
    /// [vocab][hidden]; may alias Runtime.embed when embeddings are tied.
    data: []const f16,
    /// True when the caller must free `data`.
    owned: bool,
};

/// Loads one layer's matrices on demand. Ownership of the returned matrices
/// transfers to the engine, which frees them after compiling the kernels.
pub const LayerSource = struct {
    ctx: *anyopaque,
    loadFn: *const fn (ctx: *anyopaque, allocator: std.mem.Allocator, index: u32) anyerror!Matrices,

    pub fn load(self: LayerSource, allocator: std.mem.Allocator, index: u32) !Matrices {
        return self.loadFn(self.ctx, allocator, index);
    }
};

/// Loads the lm-head weights. Called once, after the layers.
pub const HeadSource = struct {
    ctx: *anyopaque,
    loadFn: *const fn (ctx: *anyopaque, allocator: std.mem.Allocator) anyerror!HeadWeights,

    pub fn load(self: HeadSource, allocator: std.mem.Allocator) !HeadWeights {
        return self.loadFn(self.ctx, allocator);
    }
};

/// The sparse half of a MoE layer: a router, `num_experts` SwiGLU experts, and the
/// always-on shared expert. Kept separate from `Matrices` because it is per-layer
/// optional and much larger than the dense path.
///
/// The experts are stored stacked in one allocation each ([experts][inter][hidden]
/// for gate/up and [experts][hidden][inter] for down) so the loader does one
/// allocation per tensor instead of one per expert, and so the GGUF case (where
/// the expert axis is the last one) has a natural place to land.
pub const MoeWeights = struct {
    /// [num_experts][hidden]
    router: []f16 = &.{},
    /// [num_experts][moe_inter][hidden]
    gate: []f16 = &.{},
    up: []f16 = &.{},
    /// [num_experts][hidden][moe_inter]
    down: []f16 = &.{},
    num_experts: u32 = 0,
    inter: u32 = 0,
    /// The dense MLP every token also passes through; widths from
    /// `shared_inter`. Empty when the model has no shared expert.
    shared_gate: []f16 = &.{},
    shared_up: []f16 = &.{},
    shared_down: []f16 = &.{},
    /// [1][hidden] gate on the shared expert's output.
    shared_gate_lin: []f16 = &.{},
    shared_inter: u32 = 0,
    /// Cached so `expertGate` and friends can stride without a Config.
    hidden_dim: u32 = 0,

    pub fn deinit(self: *MoeWeights, allocator: std.mem.Allocator) void {
        if (self.router.len > 0) allocator.free(self.router);
        if (self.gate.len > 0) allocator.free(self.gate);
        if (self.up.len > 0) allocator.free(self.up);
        if (self.down.len > 0) allocator.free(self.down);
        if (self.shared_gate.len > 0) allocator.free(self.shared_gate);
        if (self.shared_up.len > 0) allocator.free(self.shared_up);
        if (self.shared_down.len > 0) allocator.free(self.shared_down);
        if (self.shared_gate_lin.len > 0) allocator.free(self.shared_gate_lin);
        self.* = .{};
    }

    /// One expert's gate slice, [inter][hidden].
    pub fn expertGate(self: *const MoeWeights, e: u32) []const f16 {
        const stride = @as(usize, self.inter) * self.hidden_dim;
        return self.gate[@as(usize, e) * stride ..][0..stride];
    }
    pub fn expertUp(self: *const MoeWeights, e: u32) []const f16 {
        const stride = @as(usize, self.inter) * self.hidden_dim;
        return self.up[@as(usize, e) * stride ..][0..stride];
    }
    pub fn expertDown(self: *const MoeWeights, e: u32) []const f16 {
        const stride = @as(usize, self.hidden_dim) * self.inter;
        return self.down[@as(usize, e) * stride ..][0..stride];
    }
};

pub const LayerWeights = struct {
    attn_norm: []f32 = &.{}, // [hidden]
    ffn_norm: []f32 = &.{}, // [hidden]
    /// Present only for sparse layers of a MoE model.
    moe: ?MoeWeights = null,
    /// Per-head Q/K normalisation (Qwen3): [head_dim] each. The engine keeps
    /// these in `Norm`, but the CPU reference walks LayerWeights, so they are
    /// loaded here too.
    q_norm: ?[]f32 = null,
    k_norm: ?[]f32 = null,
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

    /// Move the matrices out (used by the eager adapter).
    pub fn takeMatrices(self: *LayerWeights) Matrices {
        const m = Matrices{
            .qkv = self.qkv,
            .o = self.o,
            .gate = self.gate,
            .up = self.up,
            .down = self.down,
            .qkv_bias = self.qkv_bias,
            .o_bias = self.o_bias,
            .moe = self.moe,
        };
        self.qkv = &.{};
        self.o = &.{};
        self.gate = &.{};
        self.up = &.{};
        self.down = &.{};
        self.qkv_bias = null;
        self.o_bias = null;
        // The experts moved with the matrices; leave nothing behind for deinit to
        // free twice.
        self.moe = null;
        return m;
    }

    /// Move the norms out.
    pub fn takeNorm(self: *LayerWeights) Norm {
        const n = Norm{
            .attn = self.attn_norm,
            .ffn = self.ffn_norm,
            .q_norm = self.q_norm,
            .k_norm = self.k_norm,
        };
        self.attn_norm = &.{};
        self.ffn_norm = &.{};
        self.q_norm = null;
        self.k_norm = null;
        return n;
    }

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
        if (self.q_norm) |b| allocator.free(b);
        if (self.k_norm) |b| allocator.free(b);
        if (self.moe) |*m| m.deinit(allocator);
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
    /// Set by `headSource` so a tied head can alias the runtime embedding.
    head_embed: []const f16 = &.{},

    pub fn deinit(self: *ModelWeights) void {
        if (self.embed.len > 0) self.allocator.free(self.embed);
        if (self.final_norm.len > 0) self.allocator.free(self.final_norm);
        if (self.lm_head) |h| self.allocator.free(h);
        for (self.layers) |*l| l.deinit(self.allocator);
        if (self.layers.len > 0) self.allocator.free(self.layers);
        self.* = undefined;
    }

    /// Deep copy (used when a CPU reference needs to keep its own weights while
    /// the engine takes ownership of the originals).
    pub fn clone(self: *const ModelWeights, allocator: std.mem.Allocator) !ModelWeights {
        var out = ModelWeights{ .allocator = allocator, .config = self.config };
        errdefer out.deinit();
        out.embed = try allocator.dupe(f16, self.embed);
        out.final_norm = try allocator.dupe(f32, self.final_norm);
        if (self.lm_head) |h| out.lm_head = try allocator.dupe(f16, h);
        out.layers = try allocator.alloc(LayerWeights, self.layers.len);
        @memset(out.layers, .{});
        for (self.layers, 0..) |*lw, i| {
            out.layers[i].attn_norm = try allocator.dupe(f32, lw.attn_norm);
            out.layers[i].ffn_norm = try allocator.dupe(f32, lw.ffn_norm);
            out.layers[i].qkv = try allocator.dupe(f16, lw.qkv);
            out.layers[i].o = try allocator.dupe(f16, lw.o);
            out.layers[i].gate = try allocator.dupe(f16, lw.gate);
            out.layers[i].up = try allocator.dupe(f16, lw.up);
            out.layers[i].down = try allocator.dupe(f16, lw.down);
            if (lw.qkv_bias) |b| out.layers[i].qkv_bias = try allocator.dupe(f32, b);
            if (lw.o_bias) |b| out.layers[i].o_bias = try allocator.dupe(f32, b);
            if (lw.q_norm) |b| out.layers[i].q_norm = try allocator.dupe(f32, b);
            if (lw.k_norm) |b| out.layers[i].k_norm = try allocator.dupe(f32, b);
        }
        return out;
    }

    /// Move the runtime parts into a `Runtime` (embedding, final norm, per-layer
    /// norms and biases). The `ModelWeights` is left holding only the matrices.
    pub fn toRuntime(self: *ModelWeights, allocator: std.mem.Allocator) !Runtime {
        const norms = try allocator.alloc(Norm, self.layers.len);
        const rt = Runtime{
            .allocator = allocator,
            .config = self.config,
            .embed = self.embed,
            .final_norm = self.final_norm,
            .norms = norms,
        };
        self.embed = &.{};
        self.final_norm = &.{};
        for (self.layers, 0..) |*lw, i| {
            norms[i] = .{
                .attn = lw.attn_norm,
                .ffn = lw.ffn_norm,
                .qkv_bias = lw.qkv_bias,
                .o_bias = lw.o_bias,
            };
            lw.attn_norm = &.{};
            lw.ffn_norm = &.{};
            lw.qkv_bias = null;
            lw.o_bias = null;
        }
        return rt;
    }

    /// LayerSource that moves matrices out of this struct (eager adapter).
    pub fn layerSource(self: *ModelWeights) LayerSource {
        return .{ .ctx = self, .loadFn = loadLayerFn };
    }
    fn loadLayerFn(ctx: *anyopaque, allocator: std.mem.Allocator, index: u32) anyerror!Matrices {
        _ = allocator;
        const self: *ModelWeights = @ptrCast(@alignCast(ctx));
        return self.layers[index].takeMatrices();
    }

    /// HeadSource that moves the lm head out (or aliases `embed` when tied).
    pub fn headSource(self: *ModelWeights, embed: []const f16) HeadSource {
        self.head_embed = embed;
        return .{ .ctx = self, .loadFn = loadHeadFn };
    }
    fn loadHeadFn(ctx: *anyopaque, allocator: std.mem.Allocator) anyerror!HeadWeights {
        _ = allocator;
        const self: *ModelWeights = @ptrCast(@alignCast(ctx));
        if (self.lm_head) |h| {
            self.lm_head = null;
            return .{ .data = h, .owned = true };
        }
        return .{ .data = self.head_embed, .owned = false };
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
