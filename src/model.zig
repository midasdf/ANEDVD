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

    // ------------------------------------------------------------------ Gemma
    /// Gemma's RMSNorm is `(1 + w)`, not `w`. Baking the offset into the weights at
    /// load time keeps every compute path unchanged: `rmsnorm` already multiplies by
    /// the weight it is given. Confirmed against `GemmaRMSNorm.forward`.
    ///
    /// Whether to set this depends on the SOURCE, and getting it wrong is the difference
    /// between "Paris" and a run of dots:
    ///   - an HF checkpoint stores the raw parameter (mean ~0.19 on gemma-2-2b), so the
    ///     offset must be added;
    ///   - the GGUF converter has already applied it (measured mean 1.1927 on
    ///     `blk.0.attn_norm.weight`, which is the effective `1 + w`), so adding it again
    ///     doubles every norm.
    /// `load_hf` sets it for Gemma; `load_gguf` clears it.
    norm_unit_offset: bool = false,
    /// Gemma scales the embedding by `sqrt(hidden_size)` on the way in
    /// (`GemmaTextScaledWordEmbedding`). 1.0 for every other architecture.
    embed_scale: f32 = 1.0,
    /// Gemma 2 caps attention logits as `softcap * tanh(x / softcap)` before softmax;
    /// 0 disables. Confirmed against Gemma2Attention.
    attn_logit_softcap: f32 = 0,
    /// The same cap applied to the final logits; 0 disables.
    final_logit_softcap: f32 = 0,
    /// Per-Layer Embeddings: the width of the per-layer input vector. Gemma 4 is 256. 0 means
    /// the architecture has none, which is every other model here.
    ple_dim: u32 = 0,
    /// The vocabulary of the per-layer embedding table, which may differ from `vocab`.
    ple_vocab: u32 = 0,
    /// The RoPE base for sliding layers when it differs from `rope_theta`. Gemma 4 uses 1e6 on
    /// its global layers and 1e4 on its sliding ones; 0 means one base for every layer.
    rope_theta_swa: f32 = 0,
    /// A per-layer output scale (`blk.N.layer_output_scale`) exists and must be applied.
    layer_output_scale: bool = false,
    /// `blk.N.post_norm.weight` exists: the norm applied after PLE's per-layer projection.
    ///
    /// I first described this as "a fifth residual norm per layer", inferring it from the name.
    /// It is not — the reference calls it `post_per_layer_input_norm`, it takes the projected
    /// per-layer vector (width `hidden`) and sits inside the PLE block, not on the residual.
    /// Renamed in the reference's terms would be clearer, but the GGUF name is what the loader
    /// looks for, so the field keeps it and the comment carries the meaning.
    post_norm: bool = false,
    /// How many of the LAST layers share K/V with earlier ones instead of computing their own.
    /// Gemma 4 E2B shares 20 of its 35. A shared layer attends with its own queries but reads
    /// the keys and values of the last non-shared layer of the same attention type.
    kv_shared_layers: u32 = 0,
    /// The head dimension used by sliding layers when it differs from the global ones. Gemma 4
    /// is 256 on sliding layers and 512 on global ones; 0 means every layer uses `head_dim`.
    head_dim_swa: u32 = 0,
    /// Sliding-window size for the layers that use it; 0 disables windowing.
    sliding_window: u32 = 0,
    /// Gemma 2 alternates full and sliding attention. The rule, from the reference:
    /// `sliding_attention if (layer + 1) % 2 else full_attention` — so layer 0 SLIDES
    /// and layer 1 is global. Getting this backwards produces real words in the wrong
    /// order rather than obvious garbage.
    swa_pattern: u32 = 2,
    /// An explicit per-layer sliding list, used when the model ships one instead of a
    /// repeating pattern (Gemma 4's `layer_types`). Bit `l` of `swa_layers[l / 64]` set means
    /// layer `l` uses the window. Checked before `swa_all` and `swa_pattern`; 128 layers is
    /// the ceiling, which is twice the largest model this has been pointed at.
    swa_explicit: bool = false,
    swa_layers: [2]u64 = .{ 0, 0 },

    /// Every layer uses the sliding window rather than alternating. Mistral is the case:
    /// llama.cpp calls `set_swa_pattern(0, ..)`, whose rule is `n_pattern == 0 || ...`, so
    /// all layers slide. Applying Gemma 2's alternating pattern to Mistral would window only
    /// half its layers and attend globally on the rest.
    swa_all: bool = false,
    /// True when the FFN uses `gelu(gate) * up` rather than `silu(gate) * up`.
    ///
    /// Gemma 2 sets `hidden_activation = "gelu_pytorch_tanh"`; every other architecture
    /// here uses SiLU. The two are far apart (at x = -3 they differ 40x), and picking
    /// wrongly keeps the text fluent while making every FFN output wrong.
    use_gelu: bool = false,
    /// Explicit attention scale (1/sqrt(x)), overriding the usual 1/sqrt(head_dim).
    ///
    /// Gemma 2 sets `f_attention_scale = 1/sqrt(n_embd / n_head)` for every size except
    /// 27B, and n_embd/n_head is NOT head_dim for gemma-2-2b (2304/8 = 288 against a
    /// key length of 256), so the two formulas differ by 6%.
    attn_scale: f32 = 0,
    /// True when the model "sandwiches" each sublayer between two more norms
    /// (Gemma 2): the attention output and the MLP output are both normalised before
    /// they join the residual.
    sandwich_norms: bool = false,

    /// True when this layer attends within the sliding window rather than globally.
    ///
    /// The reference builds `layer_types` as
    /// `"sliding_attention" if (i + 1) % 2 else "full_attention"`, i.e. layer 0 slides.
    /// Writing this as `layer % 2 != 0` inverts every layer.
    pub fn layerIsSliding(self: Config, layer: u32) bool {
        if (self.sliding_window == 0) return false;
        // An explicit list wins over both other rules: Gemma 4 ships one and it is the only
        // thing that describes its pattern, which is not a repeating length.
        if (self.swa_explicit) {
            if (layer >= 128) return false;
            const word: u6 = @intCast(layer / 64);
            const bit: u6 = @intCast(layer % 64);
            return (self.swa_layers[word] >> bit) & 1 != 0;
        }
        if (self.swa_all) return true; // Mistral: llama.cpp's `n_pattern == 0` case
        if (self.swa_pattern == 0) return false;
        return (layer + 1) % self.swa_pattern != 0;
    }

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

    /// The RoPE base a particular layer uses.
    pub fn layerRopeTheta(self: Config, layer: u32) f32 {
        if (self.rope_theta_swa > 0 and self.layerIsSliding(layer)) return self.rope_theta_swa;
        return self.rope_theta;
    }

    /// The first layer that shares K/V rather than computing it. 0 when none do.
    pub fn firstKvSharedLayer(self: Config) u32 {
        if (self.kv_shared_layers == 0 or self.kv_shared_layers >= self.layers) return self.layers;
        return self.layers - self.kv_shared_layers;
    }
    pub fn isKvShared(self: Config, layer: u32) bool {
        return layer >= self.firstKvSharedLayer();
    }
    /// Where this layer's keys and values come from. A layer that computes its own is its own
    /// donor. A shared layer borrows from the **last non-shared layer of the same attention
    /// type** — sliding layers from a sliding donor, global from a global — which is what the
    /// reference does (`prev_layers[::-1].index(current_layer_type)`), because the two types
    /// have different head dimensions and must not be mixed.
    pub fn kvDonor(self: Config, layer: u32) u32 {
        if (!self.isKvShared(layer)) return layer;
        const first = self.firstKvSharedLayer();
        const want_sliding = self.layerIsSliding(layer);
        var i: u32 = first;
        while (i > 0) {
            i -= 1;
            if (self.layerIsSliding(i) == want_sliding) return i;
        }
        return layer; // no donor of the right type: keep its own, which the loader will have
    }

    /// The head dimension a particular layer uses. Sliding and global layers may differ: Gemma 4
    /// is 256 on sliding layers against 512 on global ones, so a single `head_dim` cannot
    /// describe it. `head_dim_swa == 0` means every layer uses `head_dim`.
    pub fn layerHeadDim(self: Config, layer: u32) u32 {
        if (self.head_dim_swa > 0 and self.layerIsSliding(layer)) return self.head_dim_swa;
        return self.head_dim;
    }
    pub fn layerQDim(self: Config, layer: u32) usize {
        return @as(usize, self.heads) * self.layerHeadDim(layer);
    }
    pub fn layerKvDim(self: Config, layer: u32) usize {
        return @as(usize, self.kv_heads) * self.layerHeadDim(layer);
    }
    pub fn layerQkvDim(self: Config, layer: u32) usize {
        return self.layerQDim(layer) + 2 * self.layerKvDim(layer);
    }
    /// The widest per-layer q/kv widths, for buffers that hold one layer at a time.
    pub fn maxQDim(self: Config) u32 {
        return self.heads * @max(self.head_dim, self.head_dim_swa);
    }
    pub fn maxKvDim(self: Config) u32 {
        return self.kv_heads * @max(self.head_dim, self.head_dim_swa);
    }
    /// The widest per-layer qkv row, for buffers that hold one layer at a time.
    pub fn maxQkvDim(self: Config) usize {
        const hd = @max(self.head_dim, self.head_dim_swa);
        return @as(usize, self.qkv_heads_count(hd)) * hd;
    }
    fn qkv_heads_count(self: Config, hd: u32) u32 {
        _ = hd;
        return self.heads + 2 * self.kv_heads;
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
    /// PLE, Gemma 4 only. `ple_gate` is `[ple_dim][hidden]`, `ple_proj` is `[hidden][ple_dim]`
    /// and `ple_post_norm` is `[hidden]` — the norm applied after the projection. All null when
    /// the model has no per-layer embeddings, which is every other architecture here.
    ple_gate: ?[]f16 = null,
    ple_proj: ?[]f16 = null,
    ple_post_norm: ?[]f32 = null,
    /// `blk.N.layer_output_scale`: a scalar every layer ends with, `h *= scale`. Null when the
    /// file has none.
    layer_output_scale: ?f32 = null,

    pub fn deinit(self: *Matrices, allocator: std.mem.Allocator) void {
        if (self.qkv.len > 0) allocator.free(self.qkv);
        if (self.o.len > 0) allocator.free(self.o);
        if (self.gate.len > 0) allocator.free(self.gate);
        if (self.up.len > 0) allocator.free(self.up);
        if (self.down.len > 0) allocator.free(self.down);
        if (self.qkv_bias) |b| allocator.free(b);
        if (self.o_bias) |b| allocator.free(b);
        if (self.ple_gate) |m| allocator.free(m);
        if (self.ple_proj) |m| allocator.free(m);
        if (self.ple_post_norm) |m| allocator.free(m);
        if (self.moe) |*m| m.deinit(allocator);
        self.* = .{};
    }
};

/// Per-layer normalisation weights: needed on the CPU for every decode step, so
/// they stay resident (2 * hidden floats per layer, negligible).
pub const Norm = struct {
    attn: []f32 = &.{},
    ffn: []f32 = &.{},
    /// Gemma 2's sandwich norms; empty for every other architecture.
    post_attn: []f32 = &.{},
    post_ffw: []f32 = &.{},
    /// Attention biases (Qwen2 and friends); applied on the CPU after the ANE
    /// projection, so they must outlive the matrices.
    qkv_bias: ?[]f32 = null,
    o_bias: ?[]f32 = null,
    /// Per-head Q/K normalisation weights (Qwen3): [head_dim] each, applied
    /// before RoPE.
    q_norm: ?[]f32 = null,
    k_norm: ?[]f32 = null,
    /// Gemma 2's sandwich norms: [hidden] each, null when the model has none.
    post_attn_norm: ?[]f32 = null,
    post_ffw_norm: ?[]f32 = null,
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
/// Where an expert's weights live. A MoE model cannot be materialised: Qwen1.5-MoE
/// is 25 GB of fp16 experts on disk-as-f16 against an 8 GB machine, and copying it
/// into anonymous memory swaps the machine to a crawl (997 MB of swap and 71M
/// pageins were measured). The file is instead left mapped, where its untouched
/// pages are clean and evictable, and only the k experts a token routes to are
/// dequantised — into a small per-layer cache, not into 25 GB.
pub const MoeSource = struct {
    /// The mapping the expert tensors live in.
    gguf: *const @import("gguf.zig").Gguf,
    /// Tensor names, one per projection.
    gate_name: []const u8 = "",
    up_name: []const u8 = "",
    down_name: []const u8 = "",
    /// Bytes per expert (identical for gate and up; down differs in element order
    /// only, so the element count is what matters and each read derives its own).
    gate_elems: usize = 0,
    down_elems: usize = 0,
};

pub const MoeWeights = struct {
    /// Streaming source. When set, `gate`/`up`/`down` are empty and
    /// `moeExpertAccum` reads the experts through `expertScratch` instead.
    source: ?MoeSource = null,
    /// Scratch an expert is dequantised into: three slots, because
    /// `moeExpertAccum` consumes gate, up and down together and they must not
    /// alias. Layout: [3][max(gate_elems, down_elems)]. One expert's worth of
    /// memory in total (about 35 MB for Qwen1.5-MoE), instead of 25 GB.
    expert_scratch: []f16 = &.{},
    /// f32 expert scratch for the batched prefill path: one read serves every column
    /// that chose the expert, so it is held rather than streamed per token.
    expert_scratch_f32_alt: []f32 = &.{},
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
    /// Elements per projection slot in `expert_scratch`.
    expert_slot: usize = 0,

    pub fn deinit(self: *MoeWeights, allocator: std.mem.Allocator) void {
        if (self.router.len > 0) allocator.free(self.router);
        if (self.gate.len > 0) allocator.free(self.gate);
        if (self.up.len > 0) allocator.free(self.up);
        if (self.down.len > 0) allocator.free(self.down);
        if (self.shared_gate.len > 0) allocator.free(self.shared_gate);
        if (self.shared_up.len > 0) allocator.free(self.shared_up);
        if (self.shared_down.len > 0) allocator.free(self.shared_down);
        if (self.shared_gate_lin.len > 0) allocator.free(self.shared_gate_lin);
        if (self.expert_scratch.len > 0) allocator.free(self.expert_scratch);
        if (self.expert_scratch_f32_alt.len > 0) allocator.free(self.expert_scratch_f32_alt);
        // The streaming source owns its tensor names (they must outlive the loader).
        if (self.source) |src| {
            if (src.gate_name.len > 0) allocator.free(src.gate_name);
            if (src.up_name.len > 0) allocator.free(src.up_name);
            if (src.down_name.len > 0) allocator.free(src.down_name);
        }
        self.* = .{};
    }

    /// Dequantise one expert's `which` projection into `scratch`, returning the view.
    ///
    /// `which` is 0 = gate, 1 = up, 2 = down. Only valid for a streaming source; the
    /// eager path returns the in-memory slice instead, which is why both go through
    /// these accessors rather than the engine reaching into `gate`/`up`/`down`.
    /// Stride between projection slots, derived on demand.
    ///
    /// `expert_slot` was only set by the streaming loader, so on an eager layer it was
    /// 0 and gate, up and down all wrote slot 0 — each overwriting the last. The
    /// matmul then multiplied the same tensor three times and the logits were wrong in
    /// a way that looked like a broken model rather than a broken stride.
    fn slotStride(self: *const MoeWeights, scratch_len: usize) usize {
        if (self.expert_slot > 0) return self.expert_slot;
        const per = @as(usize, self.inter) * self.hidden_dim;
        if (per == 0 or scratch_len < per * 3) return per;
        return per;
    }

    pub fn loadExpert(self: *const MoeWeights, e: u32, which: u2, scratch: []f16) []const f16 {
        const src = self.source orelse return switch (which) {
            0 => self.expertGate(e),
            1 => self.expertUp(e),
            else => self.expertDown(e),
        };
        const name = switch (which) {
            0 => src.gate_name,
            1 => src.up_name,
            else => src.down_name,
        };
        const elems = switch (which) {
            2 => src.down_elems,
            else => src.gate_elems,
        };
        // Slot per projection, so the three live at once.
        const dst = scratch[@as(usize, which) * self.slotStride(scratch.len) ..][0..elems];
        src.gguf.readExpertF16(name, e, self.num_experts, dst) catch |err| {
            // A streaming read that fails leaves the scratch zeroed rather than
            // reusing the previous expert's weights, which would be silently wrong.
            @memset(dst, 0);
            std.debug.print("moe: expert {d} of {s} failed: {s}\n", .{ e, name, @errorName(err) });
            return dst;
        };
        return dst;
    }

    /// Dequantise one expert's `which` projection into `scratch` as f32.
    ///
    /// The batched prefill path uses this so an expert's read is not immediately thrown
    /// away: it is read once and multiplied against every column that chose it. When
    /// the layer is eager there is nothing to read, so this reports empty and the
    /// caller uses the fp16 accessors.
    pub fn loadExpertF32(self: *const MoeWeights, e: u32, which: u2, scratch: []f32) []const f32 {
        const src = self.source orelse return &.{};
        const name = switch (which) {
            0 => src.gate_name,
            1 => src.up_name,
            else => src.down_name,
        };
        const elems = switch (which) {
            2 => src.down_elems,
            else => src.gate_elems,
        };
        // One slot per projection: a single shared slot made gate, up and down
        // overwrite each other, so the matmul saw the same tensor three times and
        // decode produced immediate EOS while prefill (which reads them in a
        // different order) happened to survive.
        const dst = scratch[@as(usize, which) * self.slotStride(scratch.len) ..][0..elems];
        src.gguf.readExpertF32(name, e, self.num_experts, dst) catch |err| {
            @memset(dst, 0);
            std.debug.print("moe: expert {d} of {s} failed: {s}\n", .{ e, name, @errorName(err) });
            return dst;
        };
        return dst;
    }

    /// True when the experts are read from the mapping rather than held in memory.
    pub fn streaming(self: *const MoeWeights) bool {
        return self.source != null;
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
    /// Gemma 2's sandwich norms: [hidden] each, null when the model has none.
    post_attn_norm: ?[]f32 = null,
    post_ffw_norm: ?[]f32 = null,
    /// PLE (Gemma 4): `[ple_dim][hidden]`, `[hidden][ple_dim]` and `[hidden]`. Null when the
    /// model has no per-layer embeddings.
    ple_gate: ?[]f16 = null,
    ple_proj: ?[]f16 = null,
    ple_post_norm: ?[]f32 = null,
    /// `blk.N.layer_output_scale`: every layer ends with `h *= scale`.
    layer_output_scale: ?f32 = null,
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
            .ple_gate = self.ple_gate,
            .ple_proj = self.ple_proj,
            .ple_post_norm = self.ple_post_norm,
            .layer_output_scale = self.layer_output_scale,
        };
        self.qkv = &.{};
        self.o = &.{};
        self.gate = &.{};
        self.up = &.{};
        self.down = &.{};
        self.qkv_bias = null;
        self.o_bias = null;
        self.ple_gate = null;
        self.ple_proj = null;
        self.ple_post_norm = null;
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
            .post_attn = self.post_attn_norm orelse &.{},
            .post_ffw = self.post_ffw_norm orelse &.{},
        };
        self.attn_norm = &.{};
        self.ffn_norm = &.{};
        self.q_norm = null;
        self.k_norm = null;
        self.post_attn_norm = null;
        self.post_ffw_norm = null;
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
            if (lw.post_attn_norm) |b| out.layers[i].post_attn_norm = try allocator.dupe(f32, b);
            if (lw.post_ffw_norm) |b| out.layers[i].post_ffw_norm = try allocator.dupe(f32, b);
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

test "Gemma 2's sliding layers start at layer 0" {
    // The reference builds `layer_types` as
    //   "sliding_attention" if (i + 1) % 2 else "full_attention"
    // so layer 0 slides and layer 1 is global. Writing it as `layer % 2 != 0` inverts
    // every layer, which produces real words in the wrong order rather than garbage.
    const gemma2 = Config{ .sliding_window = 4096, .swa_pattern = 2 };
    try std.testing.expect(gemma2.layerIsSliding(0));
    try std.testing.expect(!gemma2.layerIsSliding(1));
    try std.testing.expect(gemma2.layerIsSliding(2));
    try std.testing.expect(!gemma2.layerIsSliding(3));

    // No window (or no pattern) means every layer is global, which is the case for
    // every non-Gemma model and for Gemma 1.
    const plain = Config{};
    try std.testing.expect(!plain.layerIsSliding(0));
    try std.testing.expect(!plain.layerIsSliding(1));
    const gemma1 = Config{ .sliding_window = 0, .swa_pattern = 2 };
    try std.testing.expect(!gemma1.layerIsSliding(0));
}

test "Gemma's unit-offset norm is a per-source decision" {
    // The HF checkpoint and the GGUF differ in whether `(1 + w)` is already applied,
    // and this one flag is the difference between a correct answer and a run of dots.
    // The defaults must keep every non-Gemma model unaffected.
    const plain = Config{};
    try std.testing.expect(!plain.norm_unit_offset);
    try std.testing.expect(!plain.use_gelu);
    try std.testing.expectEqual(@as(f32, 1.0), plain.embed_scale);
    try std.testing.expectEqual(@as(f32, 0), plain.attn_scale);
    try std.testing.expectEqual(@as(f32, 0), plain.attn_logit_softcap);
}

test "Mistral windows every layer; Gemma 2 alternates" {
    // Two architectures, two different rules, and getting either wrong makes attention see
    // the wrong keys. llama.cpp's `set_swa_pattern(n_pattern, ..)` is
    // `is_swa[il] = n_pattern == 0 || ...`, so `n_pattern == 0` means EVERY layer slides —
    // that is Mistral. Gemma 2 instead alternates, `sliding if (layer + 1) % 2`.
    const mistral = Config{ .sliding_window = 4096, .swa_all = true };
    for (0..6) |li| try std.testing.expect(mistral.layerIsSliding(@intCast(li)));

    const gemma2 = Config{ .sliding_window = 4096, .swa_pattern = 2 };
    try std.testing.expect(gemma2.layerIsSliding(0));
    try std.testing.expect(!gemma2.layerIsSliding(1));
    try std.testing.expect(gemma2.layerIsSliding(2));
    try std.testing.expect(!gemma2.layerIsSliding(3));

    // No window at all: nothing slides, whatever the pattern says. This is every model whose
    // config has no `sliding_window` key, which is most of them.
    const plain = Config{ .swa_all = true };
    try std.testing.expect(!plain.layerIsSliding(0));
    try std.testing.expect(!plain.layerIsSliding(1));
}

test "a sliding pattern of six is five sliding layers then one global" {
    // Gemma 2 alternates 1:1 (`swa_pattern = 2`). Gemma 3 repeats over six, and the config key
    // that says so used to be read nowhere, so a Gemma 3 model silently got Gemma 2's pattern.
    // The reference rule is the same shape for both: `sliding if (layer + 1) % pattern`.
    const gemma3 = Config{ .sliding_window = 1024, .swa_pattern = 6 };
    for (0..5) |li| try std.testing.expect(gemma3.layerIsSliding(@intCast(li)));
    try std.testing.expect(!gemma3.layerIsSliding(5)); // the sixth layer is global
    for (6..11) |li| try std.testing.expect(gemma3.layerIsSliding(@intCast(li)));
    try std.testing.expect(!gemma3.layerIsSliding(11));
}

test "an explicit per-layer list overrides both the pattern and swa_all" {
    // Gemma 4 ships `layer_types` as a list rather than a repeating length, and its first
    // layers read [sliding x4, full, sliding x4, full, ...] — which no modulo describes.
    var cfg = Config{ .sliding_window = 512, .swa_explicit = true };
    // bit l set => layer l slides
    cfg.swa_layers[0] = 0b01111; // layers 0-3 sliding, layer 4 full
    for (0..4) |li| try std.testing.expect(cfg.layerIsSliding(@intCast(li)));
    try std.testing.expect(!cfg.layerIsSliding(4));
    try std.testing.expect(!cfg.layerIsSliding(5));

    // It wins over the other two rules, both of which would answer differently here.
    cfg.swa_all = true;
    try std.testing.expect(!cfg.layerIsSliding(4));
    cfg.swa_pattern = 2;
    try std.testing.expect(!cfg.layerIsSliding(4));

    // No window at all still means nothing slides, whatever the list says.
    var none = Config{ .swa_explicit = true };
    none.swa_layers[0] = 0xFFFF_FFFF_FFFF_FFFF;
    try std.testing.expect(!none.layerIsSliding(0));

    // Beyond the 128-layer ceiling it must answer rather than read past the array.
    try std.testing.expect(!cfg.layerIsSliding(200));
}

test "sliding layers can use their own head dimension" {
    // Gemma 4: sliding layers 256, global layers 512, with one KV head. A single `head_dim`
    // could not describe it, and the KV cache is allocated per layer precisely because of this.
    var cfg = Config{
        .heads = 8,
        .kv_heads = 1,
        .head_dim = 512,
        .head_dim_swa = 256,
        .sliding_window = 512,
        .swa_explicit = true,
    };
    cfg.swa_layers[0] = 0b001; // only layer 0 slides; layer 1 is global

    try std.testing.expectEqual(@as(u32, 256), cfg.layerHeadDim(0));
    try std.testing.expectEqual(@as(u32, 512), cfg.layerHeadDim(1));
    try std.testing.expectEqual(@as(usize, 8 * 256), cfg.layerQDim(0));
    try std.testing.expectEqual(@as(usize, 8 * 512), cfg.layerQDim(1));
    try std.testing.expectEqual(@as(usize, 256), cfg.layerKvDim(0)); // one KV head
    try std.testing.expectEqual(@as(usize, 512), cfg.layerKvDim(1));
    try std.testing.expectEqual(@as(usize, 8 * 256 + 2 * 256), cfg.layerQkvDim(0));
    // Buffers holding one layer at a time must fit the widest layer.
    try std.testing.expectEqual(@as(usize, 8 * 512 + 2 * 512), cfg.maxQkvDim());

    // With no sliding head dim set, every layer is unchanged — which is every other model here.
    const plain = Config{ .heads = 14, .kv_heads = 2, .head_dim = 64 };
    try std.testing.expectEqual(@as(u32, 64), plain.layerHeadDim(0));
    try std.testing.expectEqual(@as(u32, 64), plain.layerHeadDim(7));
    try std.testing.expectEqual(@as(usize, plain.qkvDim()), plain.maxQkvDim());
    try std.testing.expectEqual(@as(usize, plain.kvDim()), plain.layerKvDim(3));
}

test "a shared layer borrows K/V from the last non-shared layer of the same type" {
    // Gemma 4 E2B: 35 layers, 20 shared, and its pattern is [sliding x4, full] repeated. The
    // reference takes the last non-shared layer of the same type, not simply the layer 20 back,
    // because sliding and global layers have different head dimensions (256 against 512).
    const layers: u32 = 35;
    const shared: u32 = 20;
    var cfg = Config{ .layers = layers, .kv_shared_layers = shared, .sliding_window = 512, .swa_explicit = true };
    // [sliding x4, full] repeated: layer 4, 9, 14, 19, 24, 29, 34 are full.
    for (0..layers) |li| {
        if (li % 5 != 4) cfg.swa_layers[li / 64] |= @as(u64, 1) << @intCast(li % 64);
    }
    try std.testing.expectEqual(@as(u32, 15), cfg.firstKvSharedLayer());
    try std.testing.expect(!cfg.isKvShared(14));
    try std.testing.expect(cfg.isKvShared(15));

    // The pattern is [sliding, sliding, sliding, sliding, full], so the non-shared SLIDING
    // layers are 0-3, 5-8, 10-13 and the non-shared FULL ones are 4, 9, 14.
    // Layer 15 slides, so it borrows from 13, the last sliding layer before 15 — not from 14,
    // which is full and has a different head dimension.
    try std.testing.expectEqual(@as(u32, 13), cfg.kvDonor(15));
    try std.testing.expectEqual(@as(u32, 13), cfg.kvDonor(16)); // 15 is shared, so not a donor
    // 17 slides as well (only li % 5 == 4 is full), so it also borrows from 13.
    try std.testing.expectEqual(@as(u32, 13), cfg.kvDonor(17));
    try std.testing.expectEqual(@as(u32, 13), cfg.kvDonor(18));
    // 19 IS full (19 % 5 == 4) and is the first shared full layer: donor 14.
    try std.testing.expectEqual(@as(u32, 14), cfg.kvDonor(19));
    try std.testing.expectEqual(@as(u32, 14), cfg.kvDonor(24));
    // A layer that computes its own K/V is its own donor.
    try std.testing.expectEqual(@as(u32, 3), cfg.kvDonor(3));
    try std.testing.expectEqual(@as(u32, 14), cfg.kvDonor(14));

    // No sharing configured: every layer is its own donor, which is every other model.
    const plain = Config{ .layers = 26 };
    try std.testing.expectEqual(@as(u32, 26), plain.firstKvSharedLayer());
    for ([_]u32{ 0, 1, 25 }) |li| try std.testing.expectEqual(li, plain.kvDonor(li));
}

test "the RoPE base can differ per layer type" {
    // Gemma 4: 1e6 on its global layers, 1e4 on its sliding ones. Applying one base everywhere
    // would rotate the sliding layers at the wrong frequency.
    var cfg = Config{ .rope_theta = 1_000_000.0, .rope_theta_swa = 10_000.0, .sliding_window = 512, .swa_explicit = true };
    cfg.swa_layers[0] = 0b01; // layer 0 slides
    try std.testing.expectEqual(@as(f32, 10_000.0), cfg.layerRopeTheta(0));
    try std.testing.expectEqual(@as(f32, 1_000_000.0), cfg.layerRopeTheta(1));

    // One base for all layers is every other model here.
    const plain = Config{ .rope_theta = 10000.0 };
    try std.testing.expectEqual(@as(f32, 10000.0), plain.layerRopeTheta(0));
    try std.testing.expectEqual(@as(f32, 10000.0), plain.layerRopeTheta(7));
}
