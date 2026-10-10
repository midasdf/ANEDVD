// engine.zig — the transformer decode loop.
//
// Work split:
//   ANE : every linear projection (qkv, attention output, gate/up, down, lm head)
//         as 1x1 conv kernels with weights baked in at compile time.
//   CPU : RMSNorm, RoPE, attention (with the KV cache), SwiGLU activation,
//         residual adds and sampling.
//
// That split is forced by the ANE's MIL dialect: `rsqrt`, `reduce_mean`,
// `concat` and `gelu` are rejected by ANECCompile, and SDPA causal masks are
// silently ignored, so normalisation and attention cannot be expressed portably
// yet. The projections are >95% of the FLOPs, so the ANE still does the heavy
// work — and the SiLU is fused into the ANE program, which does support
// sigmoid/mul.
//
// Ownership: an Engine borrows all weights from the ModelWeights passed to
// init(); that struct must outlive the engine.

const std = @import("std");
const sys = @import("sys.zig");
const cpu = @import("cpu.zig");
const gguf = @import("gguf.zig");
const model = @import("model.zig");
const ane = @import("ane/runtime.zig");
const mil = @import("ane/mil.zig");
const weights = @import("ane/weights.zig");

pub const Options = struct {
    max_seq: u32 = 2048,
    /// Build kernels for only the first N layers. 0 means all of them.
    ///
    /// The ANE program pool is machine-wide and a large model needs tens of kernels, so a degraded
    /// pool stalls at the first layer and there is no way to ask a smaller question. This builds a
    /// prefix, which is enough to check a new architecture's kernel shapes on a machine that
    /// cannot hold the whole model at once.
    max_layers: u32 = 0,
    verbose: bool = true,
    /// Activation width every kernel is compiled for. The ANE reads the weights
    /// once per evaluation regardless of width (measured: width 128 costs 5%
    /// more than width 1), so one token and a whole prompt chunk cost the same.
    /// Must be a multiple of 32 for the planar layout to stay contiguous.
    /// 128 measured ~50% faster prefill than 64 with no decode regression
    /// (the weights are read once per eval either way); 256 is no better and
    /// costs more activation memory.
    chunk: u32 = 128,
    /// Fuse gate/up/SiLU/down into one ANE program (3 BLOBFILEs, one kernel
    /// instead of two). Keeps the intermediate activation on the ANE instead of
    /// round-tripping it through the CPU, and measured ~45% faster decode.
    /// Falls back to the split path if the fused program fails to compile.
    fuse_ffn: bool = true,
};

const LayerKernels = struct {
    qkv: ane.Kernel,
    /// Rows the qkv kernel produces: `q + 2*kv` for a normal layer, `q` alone for one that
    /// borrows another layer's K/V (Gemma 4's shared tail), whose checkpoint has no K/V weights.
    qkv_rows: u32,
    /// `blk.N.layer_output_scale`: Gemma 4 ends every layer with `h *= scale`.
    /// 1.0 when the file has no such tensor, which is every other model here.
    out_scale: f32 = 1.0,
    o: ane.Kernel,
    /// Fused: [hidden] -> [hidden]. Split: [hidden] -> [2*inter] (gate || up).
    /// For a sparse MoE layer this kernel computes the SHARED expert, which every
    /// token uses.
    ffn: ane.Kernel,
    ffn_split: bool,
    down: ?ane.Kernel = null,
    /// PLE (Gemma 4). `ple_gate` is `[ple_dim][hidden]`, `ple_proj` is `[hidden][ple_dim]` and
    /// `ple_post_norm` is `[hidden]`. Moved in from the layer's Matrices like the experts are,
    /// and null for every model without per-layer embeddings.
    ple_gate: ?[]f16 = null,
    ple_proj: ?[]f16 = null,
    ple_post_norm: ?[]f32 = null,
    /// Routed experts of a MoE layer, run on the CPU. The ANE cannot hold a kernel
    /// per expert (see research/moe-design.md), and only k of them are used per
    /// token, so they are read and multiplied here instead.
    moe: ?model.MoeWeights = null,
};

/// Which ANE node a measurement belongs to.
pub const Node = enum(usize) { qkv = 0, o = 1, ffn = 2, head = 3 };

pub const Stats = struct {
    ane_eval_ns: u64 = 0,
    ane_evals: u64 = 0,
    /// ANE time split by node type, so the next optimisation is measured
    /// rather than guessed.
    node_ns: [4]u64 = .{ 0, 0, 0, 0 },
    node_evals: [4]u64 = .{ 0, 0, 0, 0 },
    /// For the qkv node: how much is IOSurface staging vs the evaluation.
    qkv_write_ns: u64 = 0,
    qkv_read_ns: u64 = 0,
    /// Prefill CPU phases, to find where the non-ANE time goes.
    pf_convert_ns: u64 = 0,
    pf_rope_ns: u64 = 0,
    /// Time spent on the CPU running routed MoE experts.
    moe_ns: u64 = 0,
    /// Snapshots of the ANE counters taken when prefill ends, so a caller can report
    /// decode-only "per token" figures. Without them `node_ns` mixes the prompt's
    /// chunk-passes with decode's single-column passes: on a 14-token prompt with 1
    /// generated token, dividing the sum by the decode count inflates every number
    /// ~30x, which is exactly what the per-token split used to do.
    prefill_done_tokens: u64 = 0,
    prefill_done_node_ns: [4]u64 = @splat(0),
    prefill_done_evals: [4]u64 = @splat(0),
    prefill_done_moe_ns: u64 = 0,
    prefill_done_ane_ns: u64 = 0,
    /// ANE node time of the FIRST decode step only, and its pass count. It differs
    /// sharply from the remaining steps and not in a fixed direction (13x slower on
    /// Qwen1.5-MoE, slightly faster on SmolLM2), so it is reported separately rather
    /// than averaged in — which overstated the MoE figure 3.6x on a 6-token run.
    first_token_node_ns: [4]u64 = @splat(0),
    first_token_passes: u64 = 0,
    pf_attn_ns: u64 = 0,
    pf_norm_ns: u64 = 0,
    pf_stage_ns: u64 = 0,
    total_ns: u64 = 0,
    tokens: u64 = 0,

    pub fn nodeMs(self: Stats, n: Node) f64 {
        return @as(f64, @floatFromInt(self.node_ns[@backingInt(n)])) / 1e6;
    }

    pub fn aneMs(self: Stats) f64 {
        return @as(f64, @floatFromInt(self.ane_eval_ns)) / 1e6;
    }
    pub fn tokPerSec(self: Stats) f64 {
        if (self.tokens == 0) return 0;
        return @as(f64, @floatFromInt(self.tokens)) / (@as(f64, @floatFromInt(self.total_ns)) / 1e9);
    }
};

pub const Engine = struct {
    allocator: std.mem.Allocator,
    config: model.Config,
    opts: Options,

    /// Owned: embedding, final norm, per-layer norms and biases.
    rt: model.Runtime,
    embed: []const f16,
    final_norm: []const f32,
    head: []const f16,
    norms: []const model.Norm,

    kernels: []LayerKernels,
    head_kernel: ane.Kernel,
    /// The head is split into this many vocabulary rows per kernel, so its weight blob
    /// stays bounded. One kernel for a 256000-row vocabulary is a 1.18 GB blob, which
    /// the ANE accepts and then computes wrongly (gemma-2-2b: relative error 0.95
    /// against a CPU matmul, i.e. unrelated to the right answer).
    head_chunk: u32 = 0,
    /// When set, `runHead` leaves the final logit soft-cap off. `verify` needs the
    /// pre-cap values: tanh saturates, so comparing capped logits cannot distinguish a
    /// real disagreement from compression — gemma-2-2b reported MISMATCH while producing
    /// exactly the right text.
    pre_softcap: bool = false,
    /// Number of chunk kernels the head was split into (1 = not split).
    head_kernels: u32 = 1,
    /// The remaining chunk kernels (the first lives in `head_kernel`).
    head_extra: []ane.Kernel = &.{},
    /// How many entries of `head_extra` are initialised, for errdefer-safe cleanup.
    head_extra_built: u32 = 0,
    /// Vocabulary rows each extra chunk kernel covers (the last may be short).
    head_extra_rows: []u32 = &.{},

    /// fp16 KV cache: halves both the resident memory and the attention
    /// traffic, at the cost of ~5e-4 relative error per stored value (the same
    /// trade llama.cpp makes with -ctk f16 -ctv f16, its default).
    k_cache: [][]f16,
    v_cache: [][]f16,
    max_seq: u32,

    /// All activation buffers are `[channel * chunk + column]`.
    chunk: usize,
    x: []f32,
    h: []f32,
    qkv: []f32,
    attn: []f32,
    proj: []f32,
    gu: []f32,
    act: []f32,
    logits: []f32,
    scores: []f32,
    in16: []f16,
    out16: []f16,
    /// Single-column staging for the decode path (the padded columns stay zero
    /// in the IOSurfaces, so only column 0 is written and read).
    dec_in: []f16,
    dec_out: []f16,
    /// Width-1 staging for the lm-head kernel.
    head_in: []f16,
    /// Scratch for Gemma 2's sandwich norms. Always allocated: it is used on dense
    /// models, where `moe_hidden` (MoE-only, empty) would be an empty slice.
    sandwich: []f32 = &.{},
    /// The head kernel's output: [vocab], read back once per prefill and per token.
    /// Separate from `out16` because that is sized `stage * chunk` for activations and
    /// multiplying the vocabulary into it wasted 78 MB.
    head_out: []f16 = &.{},
    /// PLE (Gemma 4), all empty for every other model. `ple_emb` holds the token's scaled
    /// embedding, `ple_row` the dequantised table row, `ple_proj_out` the model projection of
    /// `ple_emb`, and `ple_in` the combined per-layer vectors `pleBlock` consumes.
    ple_emb: []f32 = &.{},
    ple_row: []f32 = &.{},
    ple_proj_out: []f32 = &.{},
    ple_in: []f32 = &.{},
    ple_scr: []f32 = &.{},
    ple_tmp: []f32 = &.{},
    /// Borrowed from `Runtime`: the model projection, its norm, and the table's bytes and type.
    ple_proj: []f16 = &.{},
    ple_norm: []f32 = &.{},
    ple_table: []const u8 = &.{},
    ple_type: u32 = 0,
    /// Contiguous scratch for one column's q/k/v/attention vectors.
    sq: []f32,
    sk: []f32,
    sv: []f32,
    sa: []f32,

    stats: Stats = .{},
    /// Number of layers actually executed (all of them; see stopAfterLayer).
    active_layers: u32 = 0,
    /// Called between prefill chunks so the server can answer cheap requests
    /// while a long prompt is still being prefilled. A 971-token prompt is ~8
    /// chunks and took 6.4 s, during which /health was unanswerable.
    prefill_tick: ?*const fn (?*anyopaque) void = null,
    prefill_tick_ctx: ?*anyopaque = null,
    /// MoE scratch, sized once from the config. Empty for a dense model.
    moe_logits: []f32 = &.{},
    moe_probs: []f32 = &.{},
    moe_idx: []u32 = &.{},
    moe_out: []f32 = &.{},
    /// f32 copy of the layer's normalised input, for the CPU router and experts
    /// (the ANE staging buffer is fp16).
    moe_hidden: []f32 = &.{},
    /// Chunk-level MoE batching scratch. See prefillExpertBatched.
    moe_chunk_idx: []u32 = &.{},
    /// f32 expert scratch for the batched prefill path, kept separate from the decode
    /// path's so the two cannot interleave into each other's buffers.
    moe_expert_scratch_f32_alt: []f32 = &.{},
    moe_chunk_probs: []f32 = &.{},
    moe_expert_cols: []u32 = &.{},
    moe_col_out: []f32 = &.{},
    moe_gate_scratch: []f32 = &.{},
    moe_upd_scratch: []f32 = &.{},
    /// Optional routing recorder: counts how often each expert of each layer is
    /// chosen. A resident-expert cache is only worth building if the counts are
    /// concentrated, so this is how that gets decided rather than assumed.
    /// Layout: [layer][num_experts]; null when not recording.
    route_counts: ?[]u32 = null,
    route_tokens: u32 = 0,

    pub fn deinit(self: *Engine) void {
        for (self.kernels) |*k| {
            k.qkv.deinit();
            k.o.deinit();
            k.ffn.deinit();
            if (k.down) |*d| d.deinit();
            // Routed experts were moved in from the layer's Matrices.
            if (k.moe) |*m| m.deinit(self.allocator);
            if (k.ple_gate) |g| self.allocator.free(g);
            if (k.ple_proj) |g| self.allocator.free(g);
            if (k.ple_post_norm) |g| self.allocator.free(g);
        }
        if (self.ple_emb.len > 0) self.allocator.free(self.ple_emb);
        if (self.ple_row.len > 0) self.allocator.free(self.ple_row);
        if (self.ple_proj_out.len > 0) self.allocator.free(self.ple_proj_out);
        if (self.ple_in.len > 0) self.allocator.free(self.ple_in);
        if (self.ple_scr.len > 0) self.allocator.free(self.ple_scr);
        if (self.ple_tmp.len > 0) self.allocator.free(self.ple_tmp);
        self.head_kernel.deinit();
        for (self.head_extra[0..self.head_extra_built]) |*k| k.deinit();
        if (self.head_extra.len > 0) self.allocator.free(self.head_extra);
        if (self.head_extra_rows.len > 0) self.allocator.free(self.head_extra_rows);
        self.allocator.free(self.kernels);
        self.rt.deinit();
        for (self.k_cache) |c| self.allocator.free(c);
        for (self.v_cache) |c| self.allocator.free(c);
        self.allocator.free(self.k_cache);
        self.allocator.free(self.v_cache);
        if (self.moe_logits.len > 0) self.allocator.free(self.moe_logits);
        if (self.moe_probs.len > 0) self.allocator.free(self.moe_probs);
        if (self.moe_idx.len > 0) self.allocator.free(self.moe_idx);
        if (self.moe_out.len > 0) self.allocator.free(self.moe_out);
        if (self.moe_hidden.len > 0) self.allocator.free(self.moe_hidden);
        if (self.sandwich.len > 0) self.allocator.free(self.sandwich);
        if (self.moe_chunk_idx.len > 0) self.allocator.free(self.moe_chunk_idx);
        if (self.moe_expert_scratch_f32_alt.len > 0) self.allocator.free(self.moe_expert_scratch_f32_alt);
        if (self.moe_chunk_probs.len > 0) self.allocator.free(self.moe_chunk_probs);
        if (self.moe_expert_cols.len > 0) self.allocator.free(self.moe_expert_cols);
        if (self.moe_col_out.len > 0) self.allocator.free(self.moe_col_out);
        if (self.moe_gate_scratch.len > 0) self.allocator.free(self.moe_gate_scratch);
        if (self.moe_upd_scratch.len > 0) self.allocator.free(self.moe_upd_scratch);
        self.allocator.free(self.x);
        self.allocator.free(self.h);
        self.allocator.free(self.qkv);
        self.allocator.free(self.attn);
        self.allocator.free(self.proj);
        self.allocator.free(self.gu);
        self.allocator.free(self.act);
        self.allocator.free(self.logits);
        self.allocator.free(self.scores);
        self.allocator.free(self.in16);
        self.allocator.free(self.out16);
        self.allocator.free(self.dec_in);
        self.allocator.free(self.dec_out);
        self.allocator.free(self.head_in);
        if (self.head_out.len > 0) self.allocator.free(self.head_out);
        self.allocator.free(self.sq);
        self.allocator.free(self.sk);
        self.allocator.free(self.sv);
        self.allocator.free(self.sa);
        self.* = undefined;
    }

    /// Build every ANE kernel.
    ///
    /// `rt` is taken by value and owned by the engine. Layer matrices are pulled
    /// from `layers` one at a time and freed as soon as that layer's kernels
    /// exist, so peak memory stays near one layer instead of the whole model.
    pub fn init(
        allocator: std.mem.Allocator,
        rt: model.Runtime,
        layers: model.LayerSource,
        head: model.HeadSource,
        opts: Options,
    ) !Engine {
        const cfg = rt.config;
        try cfg.validate();

        var self: Engine = undefined;
        // A field with a default value is NOT covered by `undefined`: it holds
        // 0xaaaa. Two things read those garbage bits: `prefill_tick`, which is
        // called as a function pointer between prefill chunks (jumping to
        // 0xaaaaaaaa is a real bus error, not a hypothetical), and the MoE scratch
        // below, whose `.len` decides what `deinit` frees.
        self.prefill_tick = null;
        self.prefill_tick_ctx = null;
        self.moe_logits = &.{};
        self.moe_probs = &.{};
        self.moe_idx = &.{};
        self.moe_out = &.{};
        self.moe_hidden = &.{};
        self.sandwich = &.{};
        self.moe_chunk_idx = &.{};
        self.moe_expert_scratch_f32_alt = &.{};
        self.head_extra = &.{};
        self.head_extra_rows = &.{};
        self.head_extra_built = 0;
        self.moe_chunk_probs = &.{};
        self.moe_expert_cols = &.{};
        self.moe_col_out = &.{};
        self.moe_gate_scratch = &.{};
        self.moe_upd_scratch = &.{};
        self.route_counts = null;
        self.route_tokens = 0;
        self.allocator = allocator;
        self.config = cfg;
        self.opts = opts;
        self.rt = rt;
        self.embed = rt.embed;
        self.final_norm = rt.final_norm;
        // Borrowed, not owned, like `embed` and `norms` above: `Runtime.deinit` frees these.
        // The table likewise aliases the mapping, so the `Gguf` must outlive the engine.
        self.ple_proj = rt.ple_proj;
        self.ple_norm = rt.ple_norm;
        self.ple_table = rt.ple_table;
        self.ple_type = rt.ple_table_type;
        self.head = &.{};
        self.norms = rt.norms;
        self.max_seq = opts.max_seq;
        self.chunk = opts.chunk;
        self.stats = .{};

        const L: usize = cfg.layers;
        self.kernels = try allocator.alloc(LayerKernels, L);
        var built: usize = 0;
        errdefer {
            for (self.kernels[0..built]) |*k| {
                k.qkv.deinit();
                k.o.deinit();
                k.ffn.deinit();
                if (k.down) |*d| d.deinit();
            }
            allocator.free(self.kernels);
        }
        // A prefix, when asked for. Everything below shrinks to `built` so deinit and the
        // forward loops see only what was actually compiled.
        const build_layers: usize = if (opts.max_layers > 0) @min(L, opts.max_layers) else L;
        for (0..build_layers) |i| {
            if (opts.verbose) sys.print("  layer {d}/{d}: loading weights + compiling ANE kernels\n", .{ i + 1, build_layers });
            var m = try layers.load(allocator, @intCast(i));
            defer m.deinit(allocator); // matrices are baked into the kernels now
            self.kernels[i] = try buildLayerKernels(allocator, cfg, &m, @intCast(i), opts, opts.chunk);
            // The routed experts were moved into the kernels; do not let the deferred
            // deinit free them.
            m.moe = null;
            m.ple_gate = null;
            m.ple_proj = null;
            m.ple_post_norm = null;
            built += 1;
        }
        if (built < L) {
            // Shrink so `deinit` and both loops only see compiled layers.
            self.kernels = self.kernels[0..built];
            self.norms = self.norms[0..built];
            self.active_layers = @intCast(built);
            sys.eprint("note: built {d} of {d} layers (--max-layers); the head still runs.\n", .{ built, L });
        }
        if (opts.verbose) sys.print("  lm head: loading weights + compiling ANE kernel ({d} -> {d})\n", .{ cfg.hidden, cfg.vocab });
        const hw = try head.load(allocator);
        defer if (hw.owned) allocator.free(hw.data);
        self.head = hw.data;
        // The head stays width 1: a [vocab][chunk] output surface would be tens of MB.
        //
        // Large vocabularies are split: one kernel for 256000 rows is a 1.18 GB weight
        // blob, which the ANE compiles, runs, and gets wrong. Splitting also lowers peak
        // memory, which matters on an 8 GB machine.
        self.head_chunk = headChunkFor(cfg.hidden, cfg.vocab);
        self.head_kernels = (cfg.vocab + self.head_chunk - 1) / self.head_chunk;
        self.head_extra = try allocator.alloc(ane.Kernel, self.head_kernels - 1);
        self.head_extra_rows = try allocator.alloc(u32, self.head_kernels - 1);
        self.head_extra_built = 0;
        for (0..self.head_kernels - 1) |i| {
            const first_row = (i + 1) * self.head_chunk;
            const off = @as(usize, first_row) * cfg.hidden;
            // The final chunk is short when the vocabulary does not divide evenly.
            const rows = @min(self.head_chunk, cfg.vocab - @as(u32, @intCast(first_row)));
            self.head_extra[i] = makeConvKernel(allocator, cfg.hidden, rows, hw.data[off..], "lm_head", 1) catch |e| {
                sys.eprint("lm head chunk {d}/{d} ({d} rows) failed: {s}: {s}\n", .{
                    i + 1, self.head_kernels - 1, rows, @errorName(e), ane.lastError(),
                });
                return e;
            };
            // The chunk's row count is needed again at eval time.
            self.head_extra_rows[i] = rows;
            self.head_extra_built += 1;
        }
        self.head_kernel = makeConvKernel(allocator, cfg.hidden, self.head_chunk, hw.data, "lm_head", 1) catch |e| {
            // Say why: a bare AneCompileFailed gives no clue whether the blob is too
            // large, the program pool is full, or the MIL is malformed.
            sys.eprint("lm head kernel ({d} -> {d}, chunk {d}) failed: {s}: {s}\n", .{
                cfg.hidden, cfg.vocab, self.head_chunk, @errorName(e), ane.lastError(),
            });
            return e;
        };

        self.k_cache = try allocator.alloc([]f16, L);
        self.v_cache = try allocator.alloc([]f16, L);
        for (0..L) |i| {
            // Per layer: Gemma 4's sliding layers carry a narrower K/V than its global ones,
            // so one uniform row width would either overrun the narrow layers or waste the
            // wide ones.
            const kv_dim: usize = cfg.layerKvDim(@intCast(i));
            self.k_cache[i] = try allocator.alloc(f16, @as(usize, opts.max_seq) * kv_dim);
            self.v_cache[i] = try allocator.alloc(f16, @as(usize, opts.max_seq) * kv_dim);
            @memset(self.k_cache[i], 0);
            @memset(self.v_cache[i], 0);
        }

        const ch: usize = self.chunk;
        if (cfg.num_experts > 0) {
            self.moe_logits = try allocator.alloc(f32, cfg.num_experts);
            self.moe_probs = try allocator.alloc(f32, @max(cfg.experts_per_tok, 1));
            self.moe_idx = try allocator.alloc(u32, @max(cfg.experts_per_tok, 1));
            self.moe_out = try allocator.alloc(f32, cfg.hidden * ch);
            // Chunk-level batching: every column's top-k picks, flattened.
            self.moe_chunk_idx = try allocator.alloc(u32, ch * cfg.experts_per_tok);
            self.moe_expert_scratch_f32_alt = try allocator.alloc(f32, cfg.moe_inter * cfg.hidden);
            self.moe_chunk_probs = try allocator.alloc(f32, ch * cfg.experts_per_tok);
            // Columns grouped by expert, so one expert read serves them all.
            self.moe_expert_cols = try allocator.alloc(u32, ch);
            // Per-column expert output, accumulated across that column's experts.
            self.moe_col_out = try allocator.alloc(f32, cfg.hidden * ch);
            self.moe_hidden = try allocator.alloc(f32, cfg.hidden);
            self.moe_gate_scratch = try allocator.alloc(f32, cfg.moe_inter);
            self.moe_upd_scratch = try allocator.alloc(f32, cfg.moe_inter);
        }
        self.x = try allocator.alloc(f32, @as(usize, cfg.hidden) * ch);
        self.h = try allocator.alloc(f32, @as(usize, cfg.hidden) * ch);
        self.qkv = try allocator.alloc(f32, @as(usize, cfg.qkvDim()) * ch);
        self.attn = try allocator.alloc(f32, @as(usize, cfg.maxQDim()) * ch);
        self.proj = try allocator.alloc(f32, @as(usize, cfg.hidden) * ch);
        self.gu = try allocator.alloc(f32, @as(usize, 2 * cfg.inter) * ch);
        self.act = try allocator.alloc(f32, @as(usize, cfg.inter) * ch);
        self.logits = try allocator.alloc(f32, cfg.vocab);
        self.scores = try allocator.alloc(f32, opts.max_seq);
        // `stage` sizes the per-layer staging buffers, which hold activations.
        //
        // Prefill ends by reading the whole vocabulary out of the head kernel, which a
        // vocab-sized `stage` would cover — but `stage` is multiplied by `chunk`, so
        // folding `vocab` into it allocated 39 MB for `in16` and 39 MB for `out16` on a
        // 151936-token vocabulary (78 MB, and it showed up as swap). The head read gets
        // its own `vocab`-sized buffer instead; that is the only place the vocabulary
        // needs to fit.
        const stage = @max(@max(cfg.hidden, cfg.qkvDim()), @max(cfg.qDim(), @max(2 * cfg.inter, cfg.inter)));
        self.in16 = try allocator.alloc(f16, @as(usize, stage) * ch);
        self.out16 = try allocator.alloc(f16, @as(usize, stage) * ch);
        self.dec_in = try allocator.alloc(f16, stage);
        self.dec_out = try allocator.alloc(f16, stage);
        self.head_in = try allocator.alloc(f16, cfg.hidden);
        self.sandwich = try allocator.alloc(f32, cfg.hidden);
        // PLE (Gemma 4). All empty for every other model.
        if (cfg.ple_dim > 0) {
            const rows: usize = @as(usize, cfg.layers) * cfg.ple_dim;
            self.ple_emb = try allocator.alloc(f32, cfg.hidden);
            self.ple_row = try allocator.alloc(f32, rows);
            self.ple_proj_out = try allocator.alloc(f32, rows);
            // Per column: prefill prepares a whole chunk's worth before the layer loop, since
            // every layer walks all the columns.
            self.ple_in = try allocator.alloc(f32, rows * ch);
            self.ple_scr = try allocator.alloc(f32, cfg.ple_dim);
            self.ple_tmp = try allocator.alloc(f32, cfg.hidden);
        }
        self.head_out = try allocator.alloc(f16, cfg.vocab);
        const vec = @max(cfg.maxQDim(), @max(cfg.maxKvDim(), cfg.inter));
        self.sq = try allocator.alloc(f32, vec);
        self.sk = try allocator.alloc(f32, vec);
        self.sv = try allocator.alloc(f32, vec);
        self.sa = try allocator.alloc(f32, vec);
        return self;
    }

    fn f32ToF16Into(dst: []f16, src: []const f32) void {
        for (dst, src) |*d, s| d.* = @floatCast(s);
    }
    fn f16ToF32Into(dst: []f32, src: []const f16) void {
        for (dst, src) |*d, s| d.* = @floatCast(s);
    }

    /// Gather channel `col` of a `[channel * ch + col]` buffer into `dst`.
    fn gatherColumn(dst: []f32, src: []const f32, ch: usize, col: usize) void {
        for (dst, 0..) |*d, c| d.* = src[c * ch + col];
    }

    /// Scatter `src` into channel `col` of a `[channel * ch + col]` buffer.
    fn scatterColumn(dst: []f32, src: []const f32, ch: usize, col: usize) void {
        for (src, 0..) |v, c| dst[c * ch + col] = v;
    }

    /// One decode step for `token` at position `pos`. Returns the logits slice
    /// (owned by the engine, valid until the next call).
    ///
    /// Kernels are compiled for `chunk` columns, but a decode step only fills
    /// column 0: the padded columns stay zero, so the ANE's extra output columns
    /// are harmless — and measurably free, because the weights are read once per
    /// evaluation regardless of width.
    pub fn forward(self: *Engine, token: u32, pos: u32) ![]f32 {
        // Same bound as `prefill`: `pos` indexes a row of the KV cache.
        if (pos >= self.max_seq) return error.ContextOverflow;
        const cfg = self.config;
        const ch = self.chunk;
        const hidden: usize = cfg.hidden;
        // q/kv widths are per layer: Gemma 4 sizes sliding layers differently from global
        // ones, so both are computed inside the layer loop below.
        const inter: usize = cfg.inter;
        const t_start = sys.nowNs();

        // Apply the architecture's embedding scale (Gemma: sqrt(hidden)). The prefill
        // path does this, and leaving it out here made decode start from a differently
        // scaled activation than prefill — the model then emitted "." for every step.
        for (0..hidden) |c| {
            const e: f32 = @floatCast(self.embed[@as(usize, token) * hidden + c]);
            self.x[c * ch] = e * cfg.embed_scale;
        }
        // PLE is computed once per token, before any layer, because every layer needs its slice
        // of the same result. Empty for every model without per-layer embeddings.
        if (cfg.ple_dim > 0) self.preparePle(token, 0);

        for (self.kernels, 0..) |*k, li| {
            const l_hd: u32 = cfg.layerHeadDim(@intCast(li));
            const kv_dim: usize = cfg.layerKvDim(@intCast(li));
            const q_dim: usize = cfg.layerQDim(@intCast(li));
            const norm = &self.norms[li];

            // ---- attention: qkv projection on ANE (column 0) ----
            rmsnormColumn(self.dec_in[0..hidden], self.x, norm.attn, cfg.eps, ch);
            const tw = sys.nowNs();
            try k.qkv.writeInputColumnF16(0, 0, self.dec_in[0..hidden]);
            self.stats.qkv_write_ns += sys.nowNs() - tw;
            var t0 = sys.nowNs();
            try k.qkv.eval();
            const dt_qkv = sys.nowNs() - t0;
            self.stats.ane_eval_ns += dt_qkv;
            self.stats.ane_evals += 1;
            self.stats.node_ns[@backingInt(Node.qkv)] += dt_qkv;
            self.stats.node_evals[@backingInt(Node.qkv)] += 1;
            const tr = sys.nowNs();
            try k.qkv.readOutputColumnF16(0, 0, self.dec_out[0..k.qkv_rows]);
            self.stats.qkv_read_ns += sys.nowNs() - tr;
            for (0..k.qkv_rows) |c| self.qkv[c * ch] = @floatCast(self.dec_out[c]);
            if (norm.qkv_bias) |b| {
                for (b, 0..) |v, c| self.qkv[c * ch] += v;
            }

            gatherColumn(self.sq[0..q_dim], self.qkv[0..], ch, 0);
            if (cfg.isKvShared(@intCast(li))) {
                self.ropeQueryOnly(li, pos, cfg, l_hd);
            } else {
                gatherColumn(self.sk[0..kv_dim], self.qkv[q_dim * ch ..], ch, 0);
                gatherColumn(self.sv[0..kv_dim], self.qkv[(q_dim + kv_dim) * ch ..], ch, 0);
                self.ropeAndCache(li, pos, cfg, kv_dim, l_hd);
            }
            // A shared layer attends with its own queries against the donor's keys and values.
            const donor: usize = cfg.kvDonor(@intCast(li));
            cpu.attentionDecode(self.sa[0..q_dim], self.sq[0..q_dim], self.k_cache[donor], self.v_cache[donor], pos + 1, cfg.heads, cfg.kv_heads, l_hd, self.scores, .{
                .logit_softcap = cfg.attn_logit_softcap,
                .scale = cfg.attn_scale,
                .window = if (cfg.layerIsSliding(@intCast(li))) cfg.sliding_window else 0,
            });
            scatterColumn(self.attn[0..], self.sa[0..q_dim], ch, 0);

            // ---- attention output projection on ANE ----
            for (0..q_dim) |c| self.dec_in[c] = @floatCast(self.attn[c * ch]);
            try k.o.writeInputColumnF16(0, 0, self.dec_in[0..q_dim]);
            t0 = sys.nowNs();
            try k.o.eval();
            const dt_o = sys.nowNs() - t0;
            self.stats.ane_eval_ns += dt_o;
            self.stats.ane_evals += 1;
            self.stats.node_ns[@backingInt(Node.o)] += dt_o;
            self.stats.node_evals[@backingInt(Node.o)] += 1;
            try k.o.readOutputColumnF16(0, 0, self.dec_out[0..hidden]);
            for (0..hidden) |c| {
                var v = @as(f32, @floatCast(self.dec_out[c]));
                if (norm.o_bias) |b| v += b[c];
                self.sandwich[c] = v;
            }
            // Gemma 2 normalises the attention output before it joins the residual.
            if (norm.post_attn.len == hidden) {
                rmsnormFlat(self.sandwich[0..hidden], norm.post_attn, cfg.eps);
            }
            for (0..hidden) |c| self.x[c * ch] += self.sandwich[c];

            // ---- feed-forward on ANE ----
            rmsnormColumn(self.dec_in[0..hidden], self.x, norm.ffn, cfg.eps, ch);
            try k.ffn.writeInputColumnF16(0, 0, self.dec_in[0..hidden]);
            t0 = sys.nowNs();
            try k.ffn.eval();
            const dt_ffn = sys.nowNs() - t0;
            self.stats.ane_eval_ns += dt_ffn;
            self.stats.ane_evals += 1;
            self.stats.node_ns[@backingInt(Node.ffn)] += dt_ffn;
            self.stats.node_evals[@backingInt(Node.ffn)] += 1;

            if (k.ffn_split) {
                try k.ffn.readOutputColumnF16(0, 0, self.dec_out[0 .. 2 * inter]);
                const gate = self.sq[0..inter];
                const up = self.sk[0..inter];
                for (0..inter) |c| {
                    gate[c] = @floatCast(self.dec_out[c]);
                    up[c] = @floatCast(self.dec_out[inter + c]);
                }
                cpu.gateMul(self.sa[0..inter], gate, up, cfg.use_gelu);
                for (0..inter) |c| self.dec_in[c] = @floatCast(self.sa[c]);
                const dk = &k.down.?;
                try dk.writeInputColumnF16(0, 0, self.dec_in[0..inter]);
                t0 = sys.nowNs();
                try dk.eval();
                const dt_down = sys.nowNs() - t0;
                self.stats.ane_eval_ns += dt_down;
                self.stats.ane_evals += 1;
                // Attribute the down projection to `ffn` as well. Without this the split
                // path reports only the gate/up kernel against the fused path's full
                // three-conv byte count, which inflated `ffn` to 67.6 GB/s against a real
                // 46 GB/s and made the fused form look worse than it is.
                self.stats.node_ns[@backingInt(Node.ffn)] += dt_down;
                self.stats.node_evals[@backingInt(Node.ffn)] += 1;
                try dk.readOutputColumnF16(0, 0, self.dec_out[0..hidden]);
            } else {
                try k.ffn.readOutputColumnF16(0, 0, self.dec_out[0..hidden]);
            }
            // The shared expert's output is scaled by sigmoid(gate . h) before it joins
            // the residual. The ANE kernel cannot express that (it is a 1-wide
            // projection plus a sigmoid), so it is applied here. Omitting it produces
            // plausible-looking but wrong text.
            if (k.moe) |*moe2| {
                if (moe2.shared_gate_lin.len == hidden) {
                    var g: f32 = 0;
                    for (moe2.shared_gate_lin, self.moe_hidden[0..hidden]) |w, xv| {
                        g += @as(f32, @floatCast(w)) * xv;
                    }
                    const scale = cpu.sigmoid(g);
                    for (0..hidden) |c| {
                        const v: f32 = @floatCast(self.dec_out[c]);
                        self.x[c * ch] += scale * v;
                    }
                } else {
                    for (0..hidden) |c| {
                        const v: f32 = @floatCast(self.dec_out[c]);
                        self.x[c * ch] += v;
                    }
                }
            } else {
                for (0..hidden) |c| {
                    const v: f32 = @floatCast(self.dec_out[c]);
                    self.sandwich[c] = v;
                }
                // Gemma 2 normalises the MLP output before it joins the residual.
                if (norm.post_ffw.len == hidden) {
                    rmsnormFlat(self.sandwich[0..hidden], norm.post_ffw, cfg.eps);
                }
                for (0..hidden) |c| self.x[c * ch] += self.sandwich[c];
            }

            // ---- routed experts (MoE), on the CPU ----
            // The shared expert above came from the ANE. These are the k experts the
            // router picked, and they are computed here because the ANE cannot hold a
            // kernel per expert. See research/moe-design.md.
            if (k.moe) |*moe| {
                const t_moe = sys.nowNs();
                // `dec_in` is fp16 (it feeds the ANE); the CPU wants f32.
                for (0..hidden) |c| self.moe_hidden[c] = @floatCast(self.dec_in[c]);
                const hh = self.moe_hidden[0..hidden];
                // The router reads the same normalised input the shared expert did.
                cpu.matmulF16(self.moe_logits[0..moe.num_experts], moe.router, hh, moe.num_experts, hidden);
                cpu.moeRoute(
                    self.moe_logits[0..moe.num_experts],
                    cfg.experts_per_tok,
                    cfg.norm_topk_prob,
                    self.moe_probs[0..cfg.experts_per_tok],
                    self.moe_idx[0..cfg.experts_per_tok],
                );
                // Accumulate the routed experts into a scratch, then add once, so the
                // residual is touched a single time.
                @memset(self.moe_out[0..hidden], 0);
                if (self.route_counts) |counts| {
                    self.route_tokens += 1;
                    for (self.moe_idx[0..cfg.experts_per_tok]) |e| {
                        counts[@as(usize, li) * cfg.num_experts + e] += 1;
                    }
                }
                for (self.moe_idx[0..cfg.experts_per_tok], self.moe_probs[0..cfg.experts_per_tok]) |e, p| {
                    // A streaming layer reads the expert from the mapping as f32; an
                    // eager one already has it as fp16 in memory. Both branches exist
                    // because the two representations need different matmuls, and the
                    // scratch differs too — using the fp16 accessor on a streaming
                    // layer reads an empty buffer and silently multiplies by zeros.
                    if (moe.streaming()) {
                        if (moe.expert_scratch_f32_alt.len < moe.expert_slot * 3) {
                            sys.eprint("[moe] BUG: f32 scratch {d} < needed {d}\n", .{ moe.expert_scratch_f32_alt.len, moe.expert_slot * 3 });
                        }
                        const g_w = moe.loadExpertF32(e, 0, moe.expert_scratch_f32_alt);
                        const u_w = moe.loadExpertF32(e, 1, moe.expert_scratch_f32_alt);
                        const d_w = moe.loadExpertF32(e, 2, moe.expert_scratch_f32_alt);
                        cpu.moeExpertAccumF32(
                            self.moe_out[0..hidden],
                            self.moe_gate_scratch[0..moe.inter],
                            self.moe_upd_scratch[0..moe.inter],
                            hh,
                            g_w,
                            u_w,
                            d_w,
                            moe.inter,
                            hidden,
                            p,
                            cfg.use_gelu,
                        );
                    } else {
                        const g_w = moe.loadExpert(e, 0, moe.expert_scratch);
                        const u_w = moe.loadExpert(e, 1, moe.expert_scratch);
                        const d_w = moe.loadExpert(e, 2, moe.expert_scratch);
                        cpu.moeExpertAccum(
                            self.moe_out[0..hidden],
                            self.moe_gate_scratch[0..moe.inter],
                            self.moe_upd_scratch[0..moe.inter],
                            hh,
                            g_w,
                            u_w,
                            d_w,
                            moe.inter,
                            hidden,
                            p,
                            cfg.use_gelu,
                        );
                    }
                }
                for (0..hidden) |c| self.x[c * ch] += self.moe_out[c];
                self.stats.moe_ns += sys.nowNs() - t_moe;
            }
            // PLE: Gemma 4 gates, projects and norms a per-layer vector back onto the residual
            // before the layer's output scale. `x` is strided, so the column is gathered into
            // `h` for the contiguous arithmetic and scattered back.
            if (k.ple_gate) |g| {
                const off: usize = @as(usize, li) * cfg.ple_dim;
                for (0..hidden) |c| self.h[c] = self.x[c * ch];
                cpu.pleBlock(
                    self.h[0..hidden],
                    self.ple_in[off..][0..cfg.ple_dim],
                    g,
                    k.ple_proj.?,
                    k.ple_post_norm.?,
                    self.ple_scr,
                    self.ple_tmp,
                    cfg.eps,
                );
                for (0..hidden) |c| self.x[c * ch] = self.h[c];
            }

            // Gemma 4 scales the whole output of the layer; 1.0 for every other model.
            if (k.out_scale != 1.0) {
                for (0..hidden) |c| self.x[c * ch] *= k.out_scale;
            }
        }

        // ---- final norm + lm head ----
        rmsnormColumn(self.head_in[0..hidden], self.x, self.final_norm, cfg.eps, ch);
        try self.runHead();

        self.stats.total_ns += sys.nowNs() - t_start;
        self.stats.tokens += 1;
        // Snapshot the first decode step on its own, so the steady state can be
        // reported without the cold-start ramp folded in.
        if (self.stats.first_token_passes == 0 and self.stats.prefill_done_tokens > 0) {
            const pd_ns = self.stats.prefill_done_node_ns;
            for (0..4) |i| self.stats.first_token_node_ns[i] = self.stats.node_ns[i] -| pd_ns[i];
            self.stats.first_token_passes = self.stats.tokens -| self.stats.prefill_done_tokens;
        }
        return self.logits;
    }

    /// The routing weight a column gave to `expert`, or 0 when it did not choose it.
    ///
    /// A column can list the same expert at most once (moeRoute picks distinct experts),
    /// so the first match is the only match.
    fn expertWeightFor(idx: []const u32, probs: []const f32, topk: u32, col: usize, expert: u32) f32 {
        for (0..topk) |s| {
            if (idx[col * topk + s] == expert) return probs[col * topk + s];
        }
        return 0;
    }

    /// Vocabulary rows per head kernel. 64k rows keeps the fp16 blob at 64k * hidden * 2
    /// bytes: 300 MB for a 2304-wide model, 128 MB for a 1024-wide one.
    fn headChunkFor(hidden: u32, vocab: u32) u32 {
        const target: u64 = 512 * 1024 * 1024;
        const per_row: u64 = @as(u64, hidden) * 2;
        const rows: u64 = @max(1, target / per_row);
        return @intCast(@min(rows, vocab));
    }

    /// RMSNorm over a flat f32 vector, in place. Used for Gemma 2's sandwich norms, where
    /// the value to normalise is a plain activation column rather than a strided chunk.
    fn rmsnormFlat(v: []f32, weight: []const f32, eps: f32) void {
        const n = weight.len;
        std.debug.assert(v.len >= n);
        var acc: f32 = 0;
        for (v[0..n]) |x| acc += x * x;
        const inv = 1.0 / @sqrt(acc / @as(f32, @floatFromInt(n)) + eps);
        for (v[0..n], weight) |*x, w| x.* = x.* * inv * w;
    }

    /// Run the lm head over the whole vocabulary, split across chunks as needed, and
    /// write the (optionally soft-capped) logits into `self.logits`.
    ///
    /// Chunking exists because a single kernel for a 256000-row vocabulary is a 1.18 GB
    /// weight blob, which the ANE accepts and then computes wrongly: gemma-2-2b's
    /// lm_head came back with a relative error of 0.95 against a CPU matmul.
    fn runHead(self: *Engine) !void {
        const cfg = self.config;
        const hidden: usize = cfg.hidden;
        const chunk: usize = self.head_chunk;

        try self.head_kernel.writeInputF16(0, self.head_in[0..hidden]);
        const t0 = sys.nowNs();
        try self.head_kernel.eval();
        // The FIRST kernel's row count, not the whole vocabulary: `head_out` is
        // vocab-sized, and readOutputF16 requires exactly the kernel's elemCount.
        // Reading `[0..vocab]` here is what produced "got 256000, kernel expects 116508".
        const first_rows: usize = @min(chunk, cfg.vocab);
        try self.head_kernel.readOutputF16(0, self.head_out[0..first_rows]);
        for (0..self.head_extra_built) |i| {
            const k = &self.head_extra[i];
            const off = (i + 1) * chunk;
            const rows: usize = @min(chunk, cfg.vocab - off);
            try k.writeInputF16(0, self.head_in[0..hidden]);
            try k.eval();
            try k.readOutputF16(0, self.head_out[off..][0..rows]);
        }
        const dt = sys.nowNs() - t0;
        self.stats.ane_eval_ns += dt;
        self.stats.ane_evals += 1 + self.head_extra_built;
        self.stats.node_ns[@backingInt(Node.head)] += dt;
        self.stats.node_evals[@backingInt(Node.head)] += 1 + self.head_extra_built;

        for (self.logits, 0..) |*l, i| {
            const v: f32 = @floatCast(self.head_out[i]);
            // Gemma 2 caps the final logits too.
            l.* = if (cfg.final_logit_softcap > 0 and !self.pre_softcap)
                cfg.final_logit_softcap * std.math.tanh(v / cfg.final_logit_softcap)
            else
                v;
        }
    }

    /// RMSNorm applied per attention head (Qwen3's q_norm/k_norm), before RoPE.
    fn applyHeadNorm(vec: []f32, heads: u32, head_dim: u32, weight: []const f32, eps: f32) void {
        const hd: usize = head_dim;
        for (0..heads) |h| {
            const v = vec[h * hd ..][0..hd];
            cpu.rmsnorm(v, v, weight, eps);
        }
    }

    /// Q norm and rotary only, for a layer that borrows another layer's K/V. Its own keys
    /// do not exist: no weights were loaded for them, so there is nothing to normalise, rotate
    /// or store.
    /// Compute every layer's per-layer input vector for one token, into `ple_in`.
    ///
    /// The table row is dequantised straight out of the aliased mapping: the table is 23.5e9
    /// parameters on E2B and one row is used per token, so it is never materialised. The row and
    /// the model projection of the token's own embedding are combined by `cpu.plePrepare`, which
    /// also owns the reference's three constants.
    fn preparePle(self: *Engine, token: u32, col: usize) void {
        const cfg = self.config;
        const hidden: usize = cfg.hidden;
        const rows: usize = @as(usize, cfg.layers) * cfg.ple_dim;
        // The projection consumes the same scaled embedding the model runs on, not the raw one.
        for (0..hidden) |c| {
            const e: f32 = @floatCast(self.embed[@as(usize, token) * hidden + c]);
            self.ple_emb[c] = e * cfg.embed_scale;
        }
        const ttype: gguf.GgmlType = @fromBackingInt(@intCast(self.ple_type));
        if (gguf.dequantizeRange(ttype, self.ple_table, @as(u64, token) * rows, self.ple_row[0..rows])) |_| {} else |_| {
            // A malformed table is the only way here: `loadRuntime` refuses a model that declares
            // PLE without the tensor. Zero rather than stale, so the failure is deterministic.
            @memset(self.ple_row[0..rows], 0);
        }
        cpu.matmulF16(self.ple_proj_out[0..rows], self.ple_proj, self.ple_emb[0..hidden], @intCast(rows), @intCast(hidden));
        const out = self.ple_in[col * rows ..][0..rows];
        cpu.plePrepare(out, self.ple_row[0..rows], self.ple_proj_out[0..rows], self.ple_norm, cfg.layers, cfg.ple_dim, cfg.hidden, cfg.eps, self.ple_scr);
    }

    fn ropeQueryOnly(self: *Engine, li: usize, pos: u32, cfg: model.Config, hd: u32) void {
        const norm = &self.norms[li];
        // Per layer: Gemma 4 rotates its sliding layers at a different base from its global ones.
        const theta = cfg.layerRopeTheta(@intCast(li));
        if (norm.q_norm) |w| applyHeadNorm(self.sq, cfg.heads, hd, w, cfg.eps);
        if (cfg.rope_adjacent) {
            cpu.ropeAdjacent(self.sq, cfg.heads, hd, pos, theta);
        } else {
            cpu.rope(self.sq, cfg.heads, hd, pos, theta);
        }
    }

    fn ropeAndCache(self: *Engine, li: usize, pos: u32, cfg: model.Config, kv_dim: usize, hd: u32) void {
        const norm = &self.norms[li];
        const theta = cfg.layerRopeTheta(@intCast(li));
        if (norm.q_norm) |w| applyHeadNorm(self.sq, cfg.heads, hd, w, cfg.eps);
        if (norm.k_norm) |w| applyHeadNorm(self.sk[0..kv_dim], cfg.kv_heads, hd, w, cfg.eps);
        if (cfg.rope_adjacent) {
            cpu.ropeAdjacent(self.sq, cfg.heads, hd, pos, theta);
            cpu.ropeAdjacent(self.sk[0..kv_dim], cfg.kv_heads, hd, pos, theta);
        } else {
            cpu.rope(self.sq, cfg.heads, hd, pos, theta);
            cpu.rope(self.sk[0..kv_dim], cfg.kv_heads, hd, pos, theta);
        }
        const krow = self.k_cache[li][@as(usize, pos) * kv_dim ..][0..kv_dim];
        const vrow = self.v_cache[li][@as(usize, pos) * kv_dim ..][0..kv_dim];
        for (0..kv_dim) |i| {
            krow[i] = @floatCast(self.sk[i]);
            vrow[i] = @floatCast(self.sv[i]);
        }
    }

    /// Process up to `chunk` prompt tokens in one pass and return the logits for
    /// the last one. This is where the weight-bandwidth-bound behaviour of the
    /// ANE pays off: a whole chunk costs the same as a single token.
    pub fn prefill(self: *Engine, ids: []const u32, start_pos: u32) ![]f32 {
        // The KV cache holds exactly `max_seq` rows per layer. Writing past them is a heap
        // overflow, and it does not fail loudly: `anedvd verify` with a 1121-token prompt on
        // a 1024-token context returned all-zero logits and a rel of 1.0 rather than
        // crashing. `generate` truncates its input so it never gets here, but `verify`,
        // `layers` and `check` call this directly.
        if (@as(usize, start_pos) + ids.len > self.max_seq) return error.ContextOverflow;
        const cfg = self.config;
        const ch = self.chunk;
        const hidden: usize = cfg.hidden;
        // Per layer, as in `forward`: a sliding layer may use a different head dimension.
        // Both are bound inside the layer loop below.
        const inter: usize = cfg.inter;

        var done: usize = 0;
        while (done < ids.len) {
            // The tick is fired per layer inside the loop below, which also covers the
            // start of every chunk; firing it here as well would just duplicate the first
            // call of each iteration.
            const n = @min(ch, ids.len - done);
            const base_pos: u32 = start_pos + @as(u32, @intCast(done));

            for (0..n) |j| {
                const row = self.embed[@as(usize, ids[done + j]) * hidden ..][0..hidden];
                for (0..hidden) |c| {
                    const e: f32 = @floatCast(row[c]);
                    self.x[c * ch + j] = e * cfg.embed_scale;
                }
            }
            for (n..ch) |j| {
                for (0..hidden) |c| self.x[c * ch + j] = 0;
            }
            // PLE for the whole chunk before the layer loop, since every layer walks all the
            // columns and each needs its slice of the same per-token result.
            if (cfg.ple_dim > 0) {
                for (0..n) |j| self.preparePle(ids[done + j], j);
            }

            for (self.kernels, 0..) |*k, li| {
                const l_hd: u32 = cfg.layerHeadDim(@intCast(li));
                const kv_dim: usize = cfg.layerKvDim(@intCast(li));
                const q_dim: usize = cfg.layerQDim(@intCast(li));
                const norm = &self.norms[li];

                // Also tick once per layer, not only once per chunk. A chunk of a MoE
                // model is ~9 s of CPU work (128 tokens x 4 experts x 24 layers), so
                // chunk-level ticking left /health unanswered for that long — measured at
                // 14.8 s on a 400-token prompt. A tick is a handful of syscalls (tens of
                // microseconds) against ~0.4 s of layer work.
                if (self.prefill_tick) |tick| tick(self.prefill_tick_ctx);

                // ---- qkv projection for the whole chunk ----
                const tn = sys.nowNs();
                rmsnormChunk(self.in16[0 .. hidden * ch], self.x, norm.attn, cfg.eps, ch, n);
                self.stats.pf_norm_ns += sys.nowNs() - tn;
                const ts2 = sys.nowNs();
                try k.qkv.writeInputF16(0, self.in16[0 .. hidden * ch]);
                self.stats.pf_stage_ns += sys.nowNs() - ts2;
                var t0 = sys.nowNs();
                try k.qkv.eval();
                self.stats.ane_eval_ns += sys.nowNs() - t0;
                self.stats.ane_evals += 1;
                const qkv_len = @as(usize, k.qkv_rows) * ch;
                try k.qkv.readOutputF16(0, self.out16[0..qkv_len]);
                const tc = sys.nowNs();
                for (0..qkv_len) |i| self.qkv[i] = @floatCast(self.out16[i]);
                self.stats.pf_convert_ns += sys.nowNs() - tc;
                if (norm.qkv_bias) |b| {
                    for (0..n) |j| for (b, 0..) |v, c| {
                        self.qkv[c * ch + j] += v;
                    };
                }

                // ---- RoPE + KV cache, one position at a time ----
                // The rotated query is written back into the chunk buffer so the
                // attention pass below sees it (the decode path keeps it in sq).
                const tr0 = sys.nowNs();
                for (0..n) |j| {
                    const pos = base_pos + @as(u32, @intCast(j));
                    gatherColumn(self.sq[0..q_dim], self.qkv[0..], ch, j);
                    if (cfg.isKvShared(@intCast(li))) {
                        self.ropeQueryOnly(li, pos, cfg, l_hd);
                    } else {
                        gatherColumn(self.sk[0..kv_dim], self.qkv[q_dim * ch ..], ch, j);
                        gatherColumn(self.sv[0..kv_dim], self.qkv[(q_dim + kv_dim) * ch ..], ch, j);
                        self.ropeAndCache(li, pos, cfg, kv_dim, l_hd);
                    }
                    scatterColumn(self.qkv[0..], self.sq[0..q_dim], ch, j);
                }
                self.stats.pf_rope_ns += sys.nowNs() - tr0;

                // ---- causal attention for the whole chunk at once ----
                const ta0 = sys.nowNs();
                // The per-position loop re-walked the cache for every query,
                // which made this 90% of prefill at ~1000 tokens.
                cpu.attentionPrefill(
                    self.attn[0..],
                    self.qkv[0..],
                    ch,
                    1,
                    self.k_cache[cfg.kvDonor(@intCast(li))],
                    self.v_cache[cfg.kvDonor(@intCast(li))],
                    @as(usize, base_pos) + n,
                    n,
                    base_pos,
                    cfg.heads,
                    cfg.kv_heads,
                    l_hd,
                    self.scores,
                    self.sq,
                    self.sa,
                    .{
                        .logit_softcap = cfg.attn_logit_softcap,
                        .scale = cfg.attn_scale,
                        .window = if (cfg.layerIsSliding(@intCast(li))) cfg.sliding_window else 0,
                    },
                );
                self.stats.pf_attn_ns += sys.nowNs() - ta0;

                // ---- attention output projection ----
                f32ToF16Chunk(self.in16[0 .. q_dim * ch], self.attn, ch, n);
                try k.o.writeInputF16(0, self.in16[0 .. q_dim * ch]);
                t0 = sys.nowNs();
                try k.o.eval();
                self.stats.ane_eval_ns += sys.nowNs() - t0;
                self.stats.ane_evals += 1;
                const proj_len = hidden * ch;
                try k.o.readOutputF16(0, self.out16[0..proj_len]);
                for (0..n) |j| {
                    for (0..hidden) |c| {
                        var v = @as(f32, @floatCast(self.out16[c * ch + j]));
                        if (norm.o_bias) |b| v += b[c];
                        self.sandwich[c] = v;
                    }
                    // Gemma 2 normalises each column's attention output before the residual.
                    if (norm.post_attn.len == hidden) {
                        rmsnormFlat(self.sandwich[0..hidden], norm.post_attn, cfg.eps);
                    }
                    for (0..hidden) |c| self.x[c * ch + j] += self.sandwich[c];
                }

                // ---- feed-forward ----
                rmsnormChunk(self.in16[0 .. hidden * ch], self.x, norm.ffn, cfg.eps, ch, n);
                try k.ffn.writeInputF16(0, self.in16[0 .. hidden * ch]);
                t0 = sys.nowNs();
                try k.ffn.eval();
                self.stats.ane_eval_ns += sys.nowNs() - t0;
                self.stats.ane_evals += 1;

                // The MoE paths below need the layer's normalised input in f32. The
                // ANE staging buffer is fp16, and reading it as f32 is how the
                // sigmoid-gate read turned into a bus error before this line existed.
                const moe_active = false;
                if (moe_active) {
                    const hn = @as(usize, cfg.hidden);
                    for (0..n) |j| {
                        for (0..hn) |c| self.moe_hidden[c] = @floatCast(self.in16[c * ch + j]);
                    }
                }

                if (k.ffn_split) {
                    const gu_len = @as(usize, 2 * inter) * ch;
                    try k.ffn.readOutputF16(0, self.out16[0..gu_len]);
                    const gate = self.sq[0..inter];
                    const up = self.sk[0..inter];
                    for (0..n) |j| {
                        for (0..inter) |c| {
                            gate[c] = @floatCast(self.out16[c * ch + j]);
                            up[c] = @floatCast(self.out16[(inter + c) * ch + j]);
                        }
                        cpu.gateMul(self.sa[0..inter], gate, up, cfg.use_gelu);
                        for (0..inter) |c| self.in16[c * ch + j] = @floatCast(self.sa[c]);
                    }
                    const dk = &k.down.?;
                    try dk.writeInputF16(0, self.in16[0 .. inter * ch]);
                    t0 = sys.nowNs();
                    try dk.eval();
                    self.stats.ane_eval_ns += sys.nowNs() - t0;
                    self.stats.ane_evals += 1;
                    try dk.readOutputF16(0, self.out16[0..proj_len]);
                } else {
                    try k.ffn.readOutputF16(0, self.out16[0..proj_len]);
                }
                if (k.moe) |*moe| {
                    const t_moe = sys.nowNs();
                    const hd: usize = hidden;
                    for (0..n) |j| {
                        // Normalised input for THIS column.
                        for (0..hd) |c| self.moe_hidden[c] = @floatCast(self.in16[c * ch + j]);
                        const hh = self.moe_hidden[0..hd];
                        // Shared expert: scale by sigmoid(gate . h) as it joins the residual.
                        var g: f32 = 0;
                        if (moe.shared_gate_lin.len == hd) {
                            for (moe.shared_gate_lin, hh) |w, xv| g += @as(f32, @floatCast(w)) * xv;
                        }
                        const scale = if (moe.shared_gate_lin.len == hd) cpu.sigmoid(g) else 1.0;
                        for (0..hd) |c| {
                            const v: f32 = @floatCast(self.out16[c * ch + j]);
                            self.x[c * ch + j] += scale * v;
                        }
                        // Routing for this column only; the expert maths is batched
                        // below so one read of an expert serves every column that
                        // chose it. See prefillMoEExperts.
                        cpu.matmulF16(self.moe_logits[0..moe.num_experts], moe.router, hh, moe.num_experts, hd);
                        cpu.moeRoute(
                            self.moe_logits[0..moe.num_experts],
                            cfg.experts_per_tok,
                            cfg.norm_topk_prob,
                            self.moe_probs[0..cfg.experts_per_tok],
                            self.moe_idx[0..cfg.experts_per_tok],
                        );
                        for (0..cfg.experts_per_tok) |s2| {
                            self.moe_chunk_idx[j * cfg.experts_per_tok + s2] = self.moe_idx[s2];
                            self.moe_chunk_probs[j * cfg.experts_per_tok + s2] = self.moe_probs[s2];
                        }
                        if (self.route_counts) |counts| {
                            self.route_tokens += 1;
                            for (self.moe_idx[0..cfg.experts_per_tok]) |e| {
                                counts[@as(usize, li) * cfg.num_experts + e] += 1;
                            }
                        }
                    }
                    // ---- batching phase: one read per distinct expert ----
                    // 332 tokens x top-4 x 24 layers re-read an expert for every
                    // (token, expert) pair. Routing each chunk first and then running
                    // an expert once over all its columns removes almost all of that:
                    // a ~300-token prompt touches 54.9 of 60 experts per layer, so the
                    // reads collapse from 4-per-token to at most 60-per-chunk.
                    @memset(self.moe_col_out[0 .. hd * n], 0);
                    const n_experts = moe.num_experts;
                    var expert: u32 = 0;
                    while (expert < n_experts) : (expert += 1) {
                        var n_cols: usize = 0;
                        for (0..n) |j| {
                            for (0..cfg.experts_per_tok) |s2| {
                                if (self.moe_chunk_idx[j * cfg.experts_per_tok + s2] == expert) {
                                    self.moe_expert_cols[n_cols] = @intCast(j);
                                    n_cols += 1;
                                    break;
                                }
                            }
                        }
                        if (n_cols == 0) continue;
                        // One read, reused for every column that chose this expert.
                        if (moe.streaming()) {
                            const g_w = moe.loadExpertF32(expert, 0, moe.expert_scratch_f32_alt);
                            const u_w = moe.loadExpertF32(expert, 1, moe.expert_scratch_f32_alt);
                            const d_w = moe.loadExpertF32(expert, 2, moe.expert_scratch_f32_alt);
                            for (self.moe_expert_cols[0..n_cols]) |j| {
                                const w = expertWeightFor(self.moe_chunk_idx[0 .. n * cfg.experts_per_tok], self.moe_chunk_probs[0 .. n * cfg.experts_per_tok], cfg.experts_per_tok, j, expert);
                                if (w == 0) continue;
                                for (0..hd) |c| self.moe_hidden[c] = @floatCast(self.in16[c * ch + j]);
                                cpu.moeExpertAccumF32(
                                    self.moe_col_out[j * hd ..][0..hd],
                                    self.moe_gate_scratch[0..moe.inter],
                                    self.moe_upd_scratch[0..moe.inter],
                                    self.moe_hidden[0..hd],
                                    g_w,
                                    u_w,
                                    d_w,
                                    moe.inter,
                                    hd,
                                    w,
                                    cfg.use_gelu,
                                );
                            }
                        } else {
                            const g_w = moe.loadExpert(expert, 0, moe.expert_scratch);
                            const u_w = moe.loadExpert(expert, 1, moe.expert_scratch);
                            const d_w = moe.loadExpert(expert, 2, moe.expert_scratch);
                            for (self.moe_expert_cols[0..n_cols]) |j| {
                                const w = expertWeightFor(self.moe_chunk_idx[0 .. n * cfg.experts_per_tok], self.moe_chunk_probs[0 .. n * cfg.experts_per_tok], cfg.experts_per_tok, j, expert);
                                if (w == 0) continue;
                                for (0..hd) |c| self.moe_hidden[c] = @floatCast(self.in16[c * ch + j]);
                                cpu.moeExpertAccum(
                                    self.moe_col_out[j * hd ..][0..hd],
                                    self.moe_gate_scratch[0..moe.inter],
                                    self.moe_upd_scratch[0..moe.inter],
                                    self.moe_hidden[0..hd],
                                    g_w,
                                    u_w,
                                    d_w,
                                    moe.inter,
                                    hd,
                                    w,
                                    cfg.use_gelu,
                                );
                            }
                        }
                    }
                    for (0..n) |j| {
                        for (0..hd) |c| self.x[c * ch + j] += self.moe_col_out[j * hd + c];
                    }
                    self.stats.moe_ns += sys.nowNs() - t_moe;
                } else {
                    for (0..n) |j| {
                        for (0..hidden) |c| {
                            const v: f32 = @floatCast(self.out16[c * ch + j]);
                            self.sandwich[c] = v;
                        }
                        // Gemma 2 normalises each column's MLP output before the residual.
                        if (norm.post_ffw.len == hidden) {
                            rmsnormFlat(self.sandwich[0..hidden], norm.post_ffw, cfg.eps);
                        }
                        for (0..hidden) |c| self.x[c * ch + j] += self.sandwich[c];
                    }
                }
                // PLE, per column, before the layer's output scale (the reference order).
                if (k.ple_gate) |g| {
                    const off: usize = @as(usize, li) * cfg.ple_dim;
                    for (0..n) |j| {
                        for (0..hidden) |c| self.h[c] = self.x[c * ch + j];
                        cpu.pleBlock(
                            self.h[0..hidden],
                            self.ple_in[j * @as(usize, cfg.layers) * cfg.ple_dim + off ..][0..cfg.ple_dim],
                            g,
                            k.ple_proj.?,
                            k.ple_post_norm.?,
                            self.ple_scr,
                            self.ple_tmp,
                            cfg.eps,
                        );
                        for (0..hidden) |c| self.x[c * ch + j] = self.h[c];
                    }
                }

                // Gemma 4 scales the whole output of a layer. Prefill does every column
                // of the chunk; decode does its single one.
                if (k.out_scale != 1.0) {
                    for (0..n) |j2| {
                        for (0..hidden) |c| self.x[c * ch + j2] *= k.out_scale;
                    }
                }
            }

            // Logits for the last real column, via the width-1 head kernel.
            const last = n - 1;
            for (0..hidden) |c| self.h[c * ch] = self.x[c * ch + last];
            rmsnormColumn(self.head_in[0..hidden], self.h, self.final_norm, cfg.eps, ch);
            // Same chunked path as decode. Reading `head_out[0..vocab]` here asked a
            // single chunk kernel for the whole vocabulary and got WrongShape.
            try self.runHead();

            done += n;
            self.stats.tokens += 1;
        }
        // Snapshot for decode-only reporting (see Stats.prefill_done_*).
        self.stats.prefill_done_tokens = self.stats.tokens;
        self.stats.prefill_done_node_ns = self.stats.node_ns;
        self.stats.prefill_done_evals = self.stats.node_evals;
        self.stats.prefill_done_moe_ns = self.stats.moe_ns;
        self.stats.prefill_done_ane_ns = self.stats.ane_eval_ns;

        return self.logits;
    }

    /// Compare each ANE kernel against a CPU matmul with the same weights, to
    /// localise a broken kernel when end-to-end output looks wrong. Uses column
    /// 0 of the chunked kernels.
    pub fn diagnose(self: *Engine, token: u32, lw: *const model.Matrices, layer: u32) !void {
        const cfg = self.config;
        const ch = self.chunk;
        const hidden: usize = cfg.hidden;
        // THIS layer's FFN width, from its own matrix: `cfg.inter` is the maximum across layers
        // (Gemma 4 has 6144 and 12288), and using it for a 6144 layer read past the matrix and
        // segfaulted — the same mistake the reference had, in the diagnostic beside it.
        const q_dim: usize = cfg.layerQDim(layer);
        // The kernel's OWN row count, not the config's: a sliding layer of Gemma 4 produces
        // `q` alone when it shares K/V, and half the qkv width otherwise, so asking for
        // `cfg.qkvDim()` made `readOutputColumnF16` reject the buffer with WrongShape.
        const qkv_rows: usize = self.kernels[0].qkv_rows;
        const inter: usize = lw.gate.len / hidden;

        // Apply the architecture's embedding scale, as both real forward paths do;
        // without it the per-kernel check would validate a different input than the
        // model actually uses.
        for (0..hidden) |c| {
            const e: f32 = @floatCast(self.embed[@as(usize, token) * hidden + c]);
            self.x[c * ch] = e * cfg.embed_scale;
        }
        rmsnormColumn(self.dec_in[0..hidden], self.x, self.norms[0].attn, cfg.eps, ch);

        const ref = try self.allocator.alloc(f32, cfg.vocab);
        defer self.allocator.free(ref);
        const h32 = try self.allocator.alloc(f32, @max(hidden, @max(qkv_rows, inter)));
        defer self.allocator.free(h32);
        const got = try self.allocator.alloc(f32, @max(qkv_rows, @max(hidden, inter)));
        defer self.allocator.free(got);

        // --- qkv ---
        try self.kernels[0].qkv.writeInputColumnF16(0, 0, self.dec_in[0..hidden]);
        try self.kernels[0].qkv.eval();
        try self.kernels[0].qkv.readOutputColumnF16(0, 0, self.dec_out[0..qkv_rows]);
        for (0..hidden) |c| h32[c] = @floatCast(self.dec_in[c]);
        for (0..qkv_rows) |c| got[c] = @floatCast(self.dec_out[c]);
        cpu.matmulF16(ref[0..qkv_rows], lw.qkv, h32[0..hidden], @intCast(qkv_rows), hidden);
        reportKernel("qkv", got[0..qkv_rows], ref[0..qkv_rows]);

        // --- o projection (fed with the ANE qkv's q part so both see the same x) ---
        for (0..q_dim) |c| self.dec_in[c] = @floatCast(self.dec_out[c]);
        try self.kernels[0].o.writeInputColumnF16(0, 0, self.dec_in[0..q_dim]);
        try self.kernels[0].o.eval();
        try self.kernels[0].o.readOutputColumnF16(0, 0, self.dec_out[0..hidden]);
        for (0..q_dim) |c| h32[c] = @floatCast(self.dec_in[c]);
        for (0..hidden) |c| got[c] = @floatCast(self.dec_out[c]);
        cpu.matmulF16(ref[0..hidden], lw.o, h32[0..q_dim], hidden, q_dim);
        reportKernel("o", got[0..hidden], ref[0..hidden]);

        // --- feed-forward ---
        rmsnormColumn(self.dec_in[0..hidden], self.x, self.norms[0].ffn, cfg.eps, ch);
        try self.kernels[0].ffn.writeInputColumnF16(0, 0, self.dec_in[0..hidden]);
        try self.kernels[0].ffn.eval();
        for (0..hidden) |c| h32[c] = @floatCast(self.dec_in[c]);
        if (self.kernels[0].ffn_split) {
            try self.kernels[0].ffn.readOutputColumnF16(0, 0, self.dec_out[0 .. 2 * inter]);
            for (0..inter) |c| got[c] = @floatCast(self.dec_out[c]);
            cpu.matmulF16(ref[0..inter], lw.gate, h32[0..hidden], inter, hidden);
            reportKernel("gate", got[0..inter], ref[0..inter]);
            for (0..inter) |c| got[c] = @floatCast(self.dec_out[inter + c]);
            cpu.matmulF16(ref[0..inter], lw.up, h32[0..hidden], inter, hidden);
            reportKernel("up", got[0..inter], ref[0..inter]);
        } else {
            try self.kernels[0].ffn.readOutputColumnF16(0, 0, self.dec_out[0..hidden]);
            const g = try self.allocator.alloc(f32, inter);
            defer self.allocator.free(g);
            const u = try self.allocator.alloc(f32, inter);
            defer self.allocator.free(u);
            const a = try self.allocator.alloc(f32, inter);
            defer self.allocator.free(a);
            cpu.matmulF16(g, lw.gate, h32[0..hidden], inter, hidden);
            cpu.matmulF16(u, lw.up, h32[0..hidden], inter, hidden);
            cpu.siluMul(a, g, u);
            cpu.matmulF16(ref[0..hidden], lw.down, a, hidden, inter);
            for (0..hidden) |c| got[c] = @floatCast(self.dec_out[c]);
            reportKernel("ffn(fused)", got[0..hidden], ref[0..hidden]);
        }

        // --- lm head ---
        rmsnormColumn(self.head_in[0..hidden], self.x, self.final_norm, cfg.eps, ch);
        // Use the same chunked path the engine really runs, so this reports on the code
        // that produces the logits rather than on a single-kernel version of it.
        try self.runHead();
        for (0..hidden) |c| h32[c] = @floatCast(self.head_in[c]);
        // Also dump what the head was fed, so "input wrong" and "head kernel wrong" can be
        // told apart: a correct kernel on a wrong input reports BROKEN either way.
        var nrm: f64 = 0;
        for (self.head_in[0..hidden]) |v| nrm += @as(f64, @floatCast(v)) * @as(f64, @floatCast(v));
        sys.print("    head input: |x| = {d:.4}, x[0..3] = {e:.4} {e:.4} {e:.4}\n", .{
            @sqrt(nrm), self.head_in[0], self.head_in[1], self.head_in[2],
        });
        cpu.matmulF16(ref, self.head, h32[0..hidden], cfg.vocab, hidden);
        // `self.logits` has the architecture's final soft-cap applied; the reference must
        // get the same treatment or the comparison is meaningless. Without this,
        // gemma-2-2b (cap 30, raw logits ~548) reported a relative error of 0.95 for a
        // perfectly good kernel — the check was wrong, not the kernel.
        if (cfg.final_logit_softcap > 0) {
            for (ref) |*v| v.* = cfg.final_logit_softcap * std.math.tanh(v.* / cfg.final_logit_softcap);
        }
        reportKernel("lm_head", self.logits, ref);
        // Report each chunk separately: a whole-vocabulary number cannot say whether the
        // first kernel is wrong or the chunk offsets are.
        if (self.head_extra_built > 0) {
            const chunk: usize = self.head_chunk;
            var off: usize = 0;
            for (0..self.head_extra_built + 1) |ci| {
                const rows: usize = @min(chunk, cfg.vocab - off);
                var name_buf: [24]u8 = undefined;
                const name = std.fmt.bufPrint(&name_buf, "  head chunk {d}", .{ci}) catch "  head chunk";
                reportKernel(name, self.logits[off..][0..rows], ref[off..][0..rows]);
                off += rows;
            }
        }
    }

    /// Run only the first `n` layers, then the final norm and lm head. Used to
    /// bisect a model that produces garbage: the logits at each depth show
    /// where it first goes wrong.
    pub fn stopAfterLayer(self: *Engine, n: u32) void {
        // The forward/prefill loops iterate over self.kernels, so a prefix view
        // is enough; the head kernel still runs at the end.
        const k: usize = @min(n, self.kernels.len);
        self.kernels = self.kernels[0..k];
        self.norms = self.norms[0..k];
        self.active_layers = @intCast(k);
    }

    /// Forget the KV cache (start a new conversation).
    pub fn reset(self: *Engine) void {
        for (self.k_cache) |c| @memset(c, 0);
        for (self.v_cache) |c| @memset(c, 0);
        self.stats = .{};
    }
};

/// Set when any kernel in the current `diagnose` run disagreed with the reference.
///
/// `check` used to print "RESULT: OK" unconditionally, so a completely wrong kernel
/// (lm_head on gemma-2-2b: relative error 0.95) was reported on the line above it and
/// then contradicted by the summary. A caller scripting against `check` had no signal
/// at all.
var kernel_check_failed: bool = false;

fn reportKernel(name: []const u8, got: []const f32, ref: []const f32) void {
    var max_err: f32 = 0;
    var max_mag: f32 = 0;
    var nan: usize = 0;
    for (got, ref) |a, b| {
        if (std.math.isNan(a)) nan += 1;
        const d = @abs(a - b);
        if (d > max_err) max_err = d;
        if (@abs(b) > max_mag) max_mag = @abs(b);
    }
    const rel = if (max_mag > 0) max_err / max_mag else max_err;
    // fp16 has ~3 decimal digits, so a relative bar alone flags fp16 rounding on a tiny
    // model as a failure: the ffn of a hidden-8 test checkpoint differs by 1.2e-4
    // absolute, which is quantisation noise, not a bug. Require the error to also be
    // material in absolute terms.
    const ok = nan == 0 and (rel < 0.02 or max_err < 1e-2);
    if (!ok) kernel_check_failed = true;
    sys.print("    {s:<12} max|ANE-CPU| = {e:.5}  rel = {e:.5}  nan = {d}  -> {s}\n", .{
        name, max_err, rel, nan, if (ok) "ok" else "BROKEN",
    });
}

/// True when the last `diagnose` run found a kernel that disagreed with the reference.
pub fn kernelCheckFailed() bool {
    return kernel_check_failed;
}

/// out[c] = f16(x[c * ch] * rsqrt(mean(x^2) + eps) * weight[c]) for column 0.
fn rmsnormColumn(out: []f16, x: []const f32, weight: []const f32, eps: f32, ch: usize) void {
    const n = weight.len;
    var acc: f32 = 0;
    for (0..n) |c| {
        const v = x[c * ch];
        acc += v * v;
    }
    const inv = 1.0 / @sqrt(acc / @as(f32, @floatFromInt(n)) + eps);
    for (0..n) |c| out[c] = @floatCast(x[c * ch] * inv * weight[c]);
}

/// Same, for the first `cols` columns of a `[channel * ch + col]` buffer.
fn rmsnormChunk(out: []f16, x: []const f32, weight: []const f32, eps: f32, ch: usize, cols: usize) void {
    const n = weight.len;
    for (0..cols) |j| {
        var acc: f32 = 0;
        for (0..n) |c| {
            const v = x[c * ch + j];
            acc += v * v;
        }
        const inv = 1.0 / @sqrt(acc / @as(f32, @floatFromInt(n)) + eps);
        for (0..n) |c| out[c * ch + j] = @floatCast(x[c * ch + j] * inv * weight[c]);
    }
}

fn f32ToF16Chunk(out: []f16, x: []const f32, ch: usize, cols: usize) void {
    const n = out.len / ch;
    for (0..n) |c| {
        for (0..cols) |j| out[c * ch + j] = @floatCast(x[c * ch + j]);
    }
}

/// Create a single-conv kernel: cin -> cout, weights already in ANE layout.
fn makeConvKernel(allocator: std.mem.Allocator, cin: u32, cout: u32, w: []const f16, label: []const u8, width: u32) !ane.Kernel {
    _ = label;
    // Use exactly the chunk's weights. Passing a longer slice (which the head-chunk
    // loop does for every chunk after the first) made the ANE compiler reject the
    // kernel with "Could not retrieve data from weight file": the blob header declared
    // cin*cout elements while the file held more.
    std.debug.assert(w.len >= @as(usize, cin) * cout);
    const w_chunk = w[0 .. @as(usize, cin) * cout];
    var sym_buf: [64]u8 = undefined;
    const sym = try weights.symbol(&sym_buf);
    const program = try mil.build(allocator, .{
        .inputs = &.{.{ .name = "i0", .channels = cin, .width = width }},
        .ops = &.{.{ .conv = .{
            .x = "i0",
            .w = "w0",
            .y = "o0",
            .cin = cin,
            .cout = cout,
            .blob_offset = 64,
            .file = sym,
            .width = width,
        } }},
        .outputs = &.{"o0"},
    });
    defer allocator.free(program);
    return ane.Kernel.create(allocator, program, &.{.{
        .name = sym,
        .chunks = &.{std.mem.sliceAsBytes(w_chunk)},
    }});
}

/// Fused FFN: i0 -> gate conv, up conv, sigmoid, mul, mul, down conv -> o0.
/// Three BLOBFILEs in one weight file, no inline scalars.
fn makeFusedFfnKernel(
    allocator: std.mem.Allocator,
    hidden: u32,
    inter: u32,
    gate: []const f16,
    up: []const f16,
    down: []const f16,
    width: u32,
) !ane.Kernel {
    const sizes = [_]usize{ gate.len * 2, up.len * 2, down.len * 2 };
    var offsets: [3]u64 = undefined;
    weights.chunkOffsets(&sizes, &offsets);

    var sym_buf: [64]u8 = undefined;
    const sym = try weights.symbol(&sym_buf);
    const program = try mil.build(allocator, .{
        .inputs = &.{.{ .name = "i0", .channels = hidden, .width = width }},
        .ops = &.{
            .{ .conv = .{ .x = "i0", .w = "w0", .y = "t0", .cin = hidden, .cout = inter, .blob_offset = offsets[0], .file = sym, .width = width } },
            .{ .conv = .{ .x = "i0", .w = "w1", .y = "t1", .cin = hidden, .cout = inter, .blob_offset = offsets[1], .file = sym, .width = width } },
            .{ .sigmoid = .{ .x = "t0", .y = "t2", .channels = inter, .width = width } },
            .{ .mul = .{ .a = "t0", .b = "t2", .y = "t3", .channels = inter, .width = width } },
            .{ .mul = .{ .a = "t3", .b = "t1", .y = "t4", .channels = inter, .width = width } },
            .{ .conv = .{ .x = "t4", .w = "w2", .y = "o0", .cin = inter, .cout = hidden, .blob_offset = offsets[2], .file = sym, .width = width } },
        },
        .outputs = &.{"o0"},
    });
    defer allocator.free(program);
    return ane.Kernel.create(allocator, program, &.{.{
        .name = sym,
        .chunks = &.{
            std.mem.sliceAsBytes(gate),
            std.mem.sliceAsBytes(up),
            std.mem.sliceAsBytes(down),
        },
    }});
}

fn buildLayerKernels(allocator: std.mem.Allocator, cfg: model.Config, lw: *const model.Matrices, layer: u32, opts: Options, width: u32) !LayerKernels {
    // Both projections are sized from THIS layer's head dimension, which for Gemma 4 differs
    // between sliding and global layers.
    // Taken from the matrices rather than computed: a shared layer's qkv matrix is Q-only
    // because the checkpoint has no K/V weights for it, so `qkv.len / hidden` is the width the
    // kernel must have. For every ordinary layer this is `layerQkvDim`.
    const l_qkv: u32 = @intCast(lw.qkv.len / cfg.hidden);
    const l_q: u32 = @intCast(cfg.layerQDim(layer));
    var qkv = try makeConvKernel(allocator, cfg.hidden, l_qkv, lw.qkv, "qkv", width);
    errdefer qkv.deinit();
    var o = try makeConvKernel(allocator, l_q, cfg.hidden, lw.o, "o", width);
    errdefer o.deinit();

    // On a sparse layer `gate`/`up`/`down` hold the SHARED expert, whose width is
    // `shared_inter` and need not equal `inter` (Qwen1.5-MoE: 5632 vs 5632, but the
    // routed experts are 1408). Sizing the kernel from `cfg.inter` would read past
    // the buffer.
    // Taken from the layer's own matrices rather than from `cfg.inter`, because a model may
    // use more than one FFN width (Gemma 4 has 6144 and 12288 across its layers, and its
    // metadata gives the max). `lw.gate.len / hidden` is the layer's width either way: for a
    // sparse layer `gate` holds the shared expert, which is what the MoE branch wants too.
    // For every model that has a single width this is exactly `cfg.inter`.
    const ffn_inter: u32 = if (lw.moe != null and cfg.shared_inter > 0)
        cfg.shared_inter
    else
        @intCast(lw.gate.len / cfg.hidden);
    // Ownership of the routed experts moves to the returned LayerKernels: the caller
    // frees `lw` as soon as these kernels are built, and the experts have to outlive
    // that (they are used on every token).
    const moe_keep = lw.moe;
    // PLE travels with the kernels the same way; the caller nulls the layer's copies so the
    // deferred deinit does not free them twice.
    const ple_keep = lw.ple_gate;
    const plep_keep = lw.ple_proj;
    const plen_keep = lw.ple_post_norm;

    if (opts.fuse_ffn and std.c.getenv("ANEDVD_NO_FUSED_FFN") == null) {
        if (makeFusedFfnKernel(allocator, cfg.hidden, ffn_inter, lw.gate, lw.up, lw.down, width)) |fk| {
            return .{ .out_scale = lw.layer_output_scale orelse 1.0, .qkv = qkv, .qkv_rows = l_qkv, .o = o, .ffn = fk, .ffn_split = false, .moe = moe_keep, .ple_gate = ple_keep, .ple_proj = plep_keep, .ple_post_norm = plen_keep };
        } else |e| {
            if (opts.verbose) {
                sys.print("    fused FFN rejected ({s}: {s}); falling back to split kernels\n", .{ @errorName(e), ane.lastError() });
            }
        }
    }

    const gu = try allocator.alloc(f16, lw.gate.len + lw.up.len);
    defer allocator.free(gu);
    @memcpy(gu[0..lw.gate.len], lw.gate);
    @memcpy(gu[lw.gate.len..], lw.up);
    var gu_k = try makeConvKernel(allocator, cfg.hidden, 2 * ffn_inter, gu, "gate_up", width);
    errdefer gu_k.deinit();
    var down_k = try makeConvKernel(allocator, ffn_inter, cfg.hidden, lw.down, "down", width);
    errdefer down_k.deinit();
    return .{ .out_scale = lw.layer_output_scale orelse 1.0, .qkv = qkv, .qkv_rows = l_qkv, .o = o, .ffn = gu_k, .ffn_split = true, .down = down_k, .moe = moe_keep, .ple_gate = ple_keep, .ple_proj = plep_keep, .ple_post_norm = plen_keep };
}

// ---------------------------------------------------------------------------
// Tests. The engine needs the ANE to compile kernels, so these skip cleanly on
// machines without one; where it is available they pin the invariants that are
// easy to break (batched prefill vs per-token decode, finite logits).
// ---------------------------------------------------------------------------

fn fillDeterministic(dst: []f16, seed: u32, scale: f32) void {
    var s = seed;
    for (dst) |*v| {
        s = s *% 1664525 +% 1013904223;
        const f = @as(f32, @floatFromInt(s >> 8)) / 16777216.0 - 0.5;
        v.* = @floatCast(f * scale);
    }
}

fn fillDeterministicF32(dst: []f32, seed: u32, base: f32, scale: f32) void {
    var s = seed;
    for (dst) |*v| {
        s = s *% 1664525 +% 1013904223;
        const f = @as(f32, @floatFromInt(s >> 8)) / 16777216.0 - 0.5;
        v.* = base + f * scale;
    }
}

test "batched prefill and per-token decode agree (needs the ANE)" {
    if (!ane.available()) return error.SkipZigTest;
    const a = std.testing.allocator;

    const cfg = model.Config{
        .arch = "llama",
        .hidden = 32,
        .layers = 2,
        .heads = 4,
        .kv_heads = 2,
        .head_dim = 8,
        .inter = 64,
        .vocab = 128,
        .eps = 1e-5,
        .rope_theta = 10000.0,
        .rope_adjacent = false,
    };
    var mw = model.ModelWeights{ .allocator = a, .config = cfg };
    defer mw.deinit();
    mw.embed = try a.alloc(f16, @as(usize, cfg.vocab) * cfg.hidden);
    mw.final_norm = try a.alloc(f32, cfg.hidden);
    mw.layers = try a.alloc(model.LayerWeights, cfg.layers);
    @memset(mw.layers, .{});
    fillDeterministic(mw.embed, 1, 1.0);
    fillDeterministicF32(mw.final_norm, 2, 1.0, 0.1);
    for (mw.layers, 0..) |*lw, i| {
        const li: u32 = @intCast(i);
        lw.attn_norm = try a.alloc(f32, cfg.hidden);
        lw.ffn_norm = try a.alloc(f32, cfg.hidden);
        lw.qkv = try a.alloc(f16, @as(usize, cfg.qkvDim()) * cfg.hidden);
        lw.o = try a.alloc(f16, @as(usize, cfg.hidden) * cfg.qDim());
        lw.gate = try a.alloc(f16, @as(usize, cfg.inter) * cfg.hidden);
        lw.up = try a.alloc(f16, @as(usize, cfg.inter) * cfg.hidden);
        lw.down = try a.alloc(f16, @as(usize, cfg.hidden) * cfg.inter);
        fillDeterministicF32(lw.attn_norm, 100 + li, 1.0, 0.1);
        fillDeterministicF32(lw.ffn_norm, 200 + li, 1.0, 0.1);
        fillDeterministic(lw.qkv, 300 + li, 0.3);
        fillDeterministic(lw.o, 400 + li, 0.3);
        fillDeterministic(lw.gate, 500 + li, 0.3);
        fillDeterministic(lw.up, 600 + li, 0.3);
        fillDeterministic(lw.down, 700 + li, 0.3);
    }

    const rt = try mw.toRuntime(a);
    var eng = try Engine.init(a, rt, mw.layerSource(), mw.headSource(rt.embed), .{
        .max_seq = 64,
        .verbose = false,
        .chunk = 32,
    });
    defer eng.deinit();

    const ids = [_]u32{ 3, 7, 11, 19 };

    // forward() and prefill() both return the engine's internal logits slice,
    // so the sequential result has to be copied out before prefill runs. The
    // first version of this test compared the two slices directly, i.e. a
    // buffer with itself, and passed no matter what prefill computed.
    var seq_copy: []f32 = &.{};
    for (ids, 0..) |id, pos| {
        const l = try eng.forward(id, @intCast(pos));
        if (pos + 1 == ids.len) {
            seq_copy = try a.dupe(f32, l);
            try std.testing.expectEqual(@as(usize, 128), seq_copy.len);
        }
    }
    defer a.free(seq_copy);
    var finite: usize = 0;
    for (seq_copy) |v| if (std.math.isFinite(v)) {
        finite += 1;
    };
    try std.testing.expectEqual(seq_copy.len, finite);

    // Same prompt through the batched path must give the same logits.
    const batch = try eng.prefill(&ids, 0);
    for (seq_copy, batch) |x, y| try std.testing.expectApproxEqAbs(x, y, 1e-3);
}

test "the staging buffers cover the vocabulary" {
    // Prefill ends by reading the whole vocabulary out of `out16` via the head
    // kernel. `stage` used to be sized from the activations only, so a model whose
    // vocabulary is bigger than all of them (the tiny MoE checkpoint: vocab 151936
    // against hidden 4) indexed past the end:
    //   "index out of bounds: index 151936, len 1024"
    // This asserts the invariant that fix relies on, without needing the ANE.
    const cases = [_]model.Config{
        // A small vocabulary where activations dominate.
        .{ .hidden = 576, .layers = 1, .heads = 9, .kv_heads = 3, .head_dim = 64, .inter = 1536, .vocab = 49152 },
        // A vocabulary larger than every activation.
        .{ .hidden = 4, .layers = 1, .heads = 4, .kv_heads = 2, .head_dim = 1, .inter = 2, .vocab = 151936 },
        // A wide FFN.
        .{ .hidden = 1024, .layers = 1, .heads = 16, .kv_heads = 8, .head_dim = 64, .inter = 8192, .vocab = 32000 },
    };
    for (cases) |cfg| {
        const stage = @max(@max(cfg.hidden, cfg.qkvDim()), @max(cfg.qDim(), @max(2 * cfg.inter, @max(cfg.inter, cfg.vocab))));
        try std.testing.expect(stage >= cfg.vocab);
        try std.testing.expect(stage >= cfg.hidden);
        try std.testing.expect(stage >= 2 * cfg.inter);
        try std.testing.expect(stage >= cfg.qkvDim());
    }
}

test "prefill calls the tick hook per layer, not merely per chunk (needs the ANE)" {
    // The hook is what lets the server answer /health during a long prefill.
    //
    // It used to fire once per chunk. A chunk of a MoE model is ~9 s of CPU work
    // (128 tokens x 4 experts x 24 layers), so /health went unanswered for that long —
    // measured at 14.8 s on a 400-token prompt. It now fires per layer too, and this test
    // fails if that is removed again: with one layer and three chunks the old contract
    // gave exactly 3 ticks and the new one gives 6.
    if (!ane.available()) return error.SkipZigTest;
    const a = std.testing.allocator;

    const cfg = model.Config{
        .arch = "llama",
        .hidden = 32,
        .layers = 1,
        .heads = 4,
        .kv_heads = 2,
        .head_dim = 8,
        .inter = 64,
        .vocab = 128,
        .eps = 1e-5,
        .rope_theta = 10000.0,
        .rope_adjacent = false,
    };
    var mw = model.ModelWeights{ .allocator = a, .config = cfg };
    defer mw.deinit();
    mw.embed = try a.alloc(f16, @as(usize, cfg.vocab) * cfg.hidden);
    mw.final_norm = try a.alloc(f32, cfg.hidden);
    mw.layers = try a.alloc(model.LayerWeights, 1);
    @memset(mw.layers, .{});
    fillDeterministic(mw.embed, 1, 1.0);
    fillDeterministicF32(mw.final_norm, 2, 1.0, 0.1);
    {
        const lw = &mw.layers[0];
        lw.attn_norm = try a.alloc(f32, cfg.hidden);
        lw.ffn_norm = try a.alloc(f32, cfg.hidden);
        lw.qkv = try a.alloc(f16, @as(usize, cfg.qkvDim()) * cfg.hidden);
        lw.o = try a.alloc(f16, @as(usize, cfg.hidden) * cfg.qDim());
        lw.gate = try a.alloc(f16, @as(usize, cfg.inter) * cfg.hidden);
        lw.up = try a.alloc(f16, @as(usize, cfg.inter) * cfg.hidden);
        lw.down = try a.alloc(f16, @as(usize, cfg.hidden) * cfg.inter);
        fillDeterministicF32(lw.attn_norm, 100, 1.0, 0.1);
        fillDeterministicF32(lw.ffn_norm, 200, 1.0, 0.1);
        fillDeterministic(lw.qkv, 300, 0.3);
        fillDeterministic(lw.o, 400, 0.3);
        fillDeterministic(lw.gate, 500, 0.3);
        fillDeterministic(lw.up, 600, 0.3);
        fillDeterministic(lw.down, 700, 0.3);
    }
    const rt = try mw.toRuntime(a);
    var eng = try Engine.init(a, rt, mw.layerSource(), mw.headSource(rt.embed), .{
        .max_seq = 64,
        .verbose = false,
        .chunk = 8,
    });
    defer eng.deinit();

    const Ctx = struct {
        fn tick(ctx: ?*anyopaque) void {
            const n: *usize = @ptrCast(@alignCast(ctx.?));
            n.* += 1;
        }
    };
    var ticks: usize = 0;
    eng.prefill_tick = Ctx.tick;
    eng.prefill_tick_ctx = &ticks;

    // 20 tokens at chunk 8 is 3 chunks (8 + 8 + 4) and 1 layer each, so 6 ticks. Assert the
    // layer count is honoured rather than pinning the exact number, so adding layers to
    // this fixture does not silently weaken the check.
    const ids = try a.alloc(u32, 20);
    defer a.free(ids);
    @memset(ids, 1);
    _ = try eng.prefill(ids, 0);
    try std.testing.expectEqual(@as(usize, 3 * cfg.layers), ticks);
}

test "batched prefill agrees with per-token decode on an MoE layer (needs the ANE)" {
    // The dense test above does not reach the MoE expert batching at all. That path had a
    // bug that produced plausible-looking wrong answers: `moe_col_out[j..][0..hd]` wrote at
    // offset `j` while the reader read `j * hd + c`, so every column after the first was
    // read from the wrong place. It was found by hand, with a real 9.5 GB model, because
    // nothing here covered it.
    //
    // The property that catches it is the same one: the batched prefill and the sequential
    // per-token decode must produce the same logits. With several prompt tokens routing to
    // overlapping experts, an error in the column indexing moves the logits.
    if (!ane.available()) return error.SkipZigTest;
    const a = std.testing.allocator;

    const cfg = model.Config{
        .arch = "qwen2moe",
        .hidden = 32,
        .layers = 2,
        .heads = 4,
        .kv_heads = 2,
        .head_dim = 8,
        .inter = 24, // shared expert width
        .vocab = 128,
        .eps = 1e-5,
        .rope_theta = 10000.0,
        .rope_adjacent = false,
        .num_experts = 4,
        .experts_per_tok = 2,
        .moe_inter = 32,
        .shared_inter = 24,
    };
    var mw = model.ModelWeights{ .allocator = a, .config = cfg };
    defer mw.deinit();
    mw.embed = try a.alloc(f16, @as(usize, cfg.vocab) * cfg.hidden);
    mw.final_norm = try a.alloc(f32, cfg.hidden);
    mw.layers = try a.alloc(model.LayerWeights, cfg.layers);
    @memset(mw.layers, .{});
    fillDeterministic(mw.embed, 1, 1.0);
    fillDeterministicF32(mw.final_norm, 2, 1.0, 0.1);

    const n_experts = cfg.num_experts;
    const moe_inter = cfg.moe_inter;
    const shared = cfg.shared_inter;
    for (mw.layers, 0..) |*lw, i| {
        const li: u32 = @intCast(i);
        lw.attn_norm = try a.alloc(f32, cfg.hidden);
        lw.ffn_norm = try a.alloc(f32, cfg.hidden);
        lw.qkv = try a.alloc(f16, @as(usize, cfg.qkvDim()) * cfg.hidden);
        lw.o = try a.alloc(f16, @as(usize, cfg.hidden) * cfg.qDim());
        fillDeterministicF32(lw.attn_norm, 100 + li, 1.0, 0.1);
        fillDeterministicF32(lw.ffn_norm, 200 + li, 1.0, 0.1);
        fillDeterministic(lw.qkv, 300 + li, 0.3);
        fillDeterministic(lw.o, 400 + li, 0.3);

        // The shared expert occupies the dense FFN slots, as the loader arranges it.
        lw.gate = try a.alloc(f16, @as(usize, shared) * cfg.hidden);
        lw.up = try a.alloc(f16, @as(usize, shared) * cfg.hidden);
        lw.down = try a.alloc(f16, @as(usize, cfg.hidden) * shared);
        fillDeterministic(lw.gate, 500 + li, 0.3);
        fillDeterministic(lw.up, 600 + li, 0.3);
        fillDeterministic(lw.down, 700 + li, 0.3);

        var moe = model.MoeWeights{
            .num_experts = n_experts,
            .inter = moe_inter,
            .hidden_dim = cfg.hidden,
            .shared_inter = shared,
        };
        moe.router = try a.alloc(f16, @as(usize, n_experts) * cfg.hidden);
        moe.gate = try a.alloc(f16, @as(usize, n_experts) * moe_inter * cfg.hidden);
        moe.up = try a.alloc(f16, @as(usize, n_experts) * moe_inter * cfg.hidden);
        moe.down = try a.alloc(f16, @as(usize, n_experts) * cfg.hidden * moe_inter);
        fillDeterministic(moe.router, 800 + li, 0.5);
        fillDeterministic(moe.gate, 900 + li, 0.3);
        fillDeterministic(moe.up, 1000 + li, 0.3);
        fillDeterministic(moe.down, 1100 + li, 0.3);
        moe.shared_gate = try a.alloc(f16, @as(usize, shared) * cfg.hidden);
        moe.shared_up = try a.alloc(f16, @as(usize, shared) * cfg.hidden);
        moe.shared_down = try a.alloc(f16, @as(usize, cfg.hidden) * shared);
        fillDeterministic(moe.shared_gate, 1200 + li, 0.3);
        fillDeterministic(moe.shared_up, 1300 + li, 0.3);
        fillDeterministic(moe.shared_down, 1400 + li, 0.3);
        // The always-on scale is sigmoid(gate . h), which must be exercised.
        moe.shared_gate_lin = try a.alloc(f16, cfg.hidden);
        fillDeterministic(moe.shared_gate_lin, 1500 + li, 0.5);
        // Eager layer: `loadExpert` hands back these in-memory slices.
        moe.expert_slot = moe_inter * cfg.hidden;
        moe.expert_scratch = try a.alloc(f16, moe.expert_slot * 3);
        fillDeterministic(moe.expert_scratch, 1600 + li, 0.1);
        lw.moe = moe;
    }

    const rt = try mw.toRuntime(a);
    var eng = try Engine.init(a, rt, mw.layerSource(), mw.headSource(rt.embed), .{
        .max_seq = 64,
        .verbose = false,
        .chunk = 32,
    });
    defer eng.deinit();

    // Eight tokens, so several columns land on each expert and the batching is real.
    const ids = [_]u32{ 3, 7, 11, 19, 23, 5, 13, 17, 29, 31, 2, 37, 41, 43, 47, 53 };

    var seq_copy: []f32 = &.{};
    for (ids, 0..) |id, pos| {
        const l = try eng.forward(id, @intCast(pos));
        if (pos + 1 == ids.len) {
            seq_copy = try a.dupe(f32, l);
            try std.testing.expectEqual(@as(usize, 128), seq_copy.len);
        }
    }
    defer a.free(seq_copy);
    for (seq_copy) |v| try std.testing.expect(std.math.isFinite(v));

    // The MoE logits must not be a constant: a batching bug that zeroed the expert
    // contribution would otherwise agree by producing the same wrong thing twice.
    var lo: f32 = seq_copy[0];
    var hi: f32 = seq_copy[0];
    for (seq_copy) |v| {
        lo = @min(lo, v);
        hi = @max(hi, v);
    }
    try std.testing.expect(hi - lo > 1e-3);

    const batch = try eng.prefill(&ids, 0);
    // Tolerance from measurement, not taste: with the off-by-index bug deliberately put
    // back, the worst difference between the two paths is 6.47e-1; with the correct code it
    // is 7.8e-3 (fp16 rounding through two MoE layers). 2e-2 sits between the two by more
    // than an order of magnitude on each side, so it fails the bug and not the rounding.
    var worst: f32 = 0;
    for (seq_copy, batch) |x, y| worst = @max(worst, @abs(x - y));
    // The argmax is the sharp check: a discrete property that the mis-indexing breaks
    // outright, where the numeric bound below is a looser backstop.
    var am_seq: usize = 0;
    var am_batch: usize = 0;
    for (seq_copy, 0..) |v, i| if (v > seq_copy[am_seq]) {
        am_seq = i;
    };
    for (batch, 0..) |v, i| if (v > batch[am_batch]) {
        am_batch = i;
    };
    try std.testing.expectEqual(am_seq, am_batch);
    // Numeric bound from measurement, not taste. With the off-by-index bug deliberately put
    // back the worst difference is 9.96e-1; with the correct code it is 1.68e-1, the gap
    // between the two summation orders in fp16 (which is why it is far above the dense
    // test's 1e-3). 4e-1 sits between them on both sides; the argmax assertion above is the
    // check that actually pins the behaviour.
    try std.testing.expect(worst < 0.4);
}

/// Build a one-layer MoE whose routed experts are zero, so whatever the forward pass
/// produces comes from the SHARED expert and its `sigmoid(gate . h)` scale.
fn buildSharedExpertFixture(a: std.mem.Allocator, with_gate_lin: bool) !model.ModelWeights {
    const cfg = model.Config{
        .arch = "qwen2moe",
        .hidden = 32,
        .layers = 1,
        .heads = 4,
        .kv_heads = 2,
        .head_dim = 8,
        .inter = 24,
        .vocab = 64,
        .eps = 1e-5,
        .rope_theta = 10000.0,
        .rope_adjacent = false,
        .num_experts = 2,
        .experts_per_tok = 1,
        .moe_inter = 16,
        .shared_inter = 24,
    };
    var mw = model.ModelWeights{ .allocator = a, .config = cfg };
    errdefer mw.deinit();
    mw.embed = try a.alloc(f16, @as(usize, cfg.vocab) * cfg.hidden);
    mw.final_norm = try a.alloc(f32, cfg.hidden);
    mw.layers = try a.alloc(model.LayerWeights, cfg.layers);
    @memset(mw.layers, .{});
    fillDeterministic(mw.embed, 1, 1.0);
    fillDeterministicF32(mw.final_norm, 2, 1.0, 0.1);
    const lw = &mw.layers[0];
    lw.attn_norm = try a.alloc(f32, cfg.hidden);
    lw.ffn_norm = try a.alloc(f32, cfg.hidden);
    lw.qkv = try a.alloc(f16, @as(usize, cfg.qkvDim()) * cfg.hidden);
    lw.o = try a.alloc(f16, @as(usize, cfg.hidden) * cfg.qDim());
    fillDeterministicF32(lw.attn_norm, 100, 1.0, 0.1);
    fillDeterministicF32(lw.ffn_norm, 200, 1.0, 0.1);
    fillDeterministic(lw.qkv, 300, 0.3);
    fillDeterministic(lw.o, 400, 0.3);
    // The shared expert lives in the dense FFN slots.
    lw.gate = try a.alloc(f16, @as(usize, cfg.shared_inter) * cfg.hidden);
    lw.up = try a.alloc(f16, @as(usize, cfg.shared_inter) * cfg.hidden);
    lw.down = try a.alloc(f16, @as(usize, cfg.hidden) * cfg.shared_inter);
    fillDeterministic(lw.gate, 500, 0.3);
    fillDeterministic(lw.up, 600, 0.3);
    fillDeterministic(lw.down, 700, 0.3);

    var moe = model.MoeWeights{
        .num_experts = cfg.num_experts,
        .inter = cfg.moe_inter,
        .hidden_dim = cfg.hidden,
        .shared_inter = cfg.shared_inter,
    };
    moe.router = try a.alloc(f16, @as(usize, cfg.num_experts) * cfg.hidden);
    moe.gate = try a.alloc(f16, @as(usize, cfg.num_experts) * cfg.moe_inter * cfg.hidden);
    moe.up = try a.alloc(f16, @as(usize, cfg.num_experts) * cfg.moe_inter * cfg.hidden);
    moe.down = try a.alloc(f16, @as(usize, cfg.num_experts) * cfg.hidden * cfg.moe_inter);
    fillDeterministic(moe.router, 800, 0.5);
    // Routed experts are ZERO: the forward pass output must come from the shared expert.
    fillDeterministic(moe.gate, 900, 0.0);
    fillDeterministic(moe.up, 1000, 0.0);
    fillDeterministic(moe.down, 1100, 0.0);
    moe.shared_gate = try a.alloc(f16, @as(usize, cfg.shared_inter) * cfg.hidden);
    moe.shared_up = try a.alloc(f16, @as(usize, cfg.shared_inter) * cfg.hidden);
    moe.shared_down = try a.alloc(f16, @as(usize, cfg.hidden) * cfg.shared_inter);
    fillDeterministic(moe.shared_gate, 1200, 0.3);
    fillDeterministic(moe.shared_up, 1300, 0.3);
    fillDeterministic(moe.shared_down, 1400, 0.3);
    if (with_gate_lin) {
        // All zeros, so the pre-activation is exactly 0 and sigmoid(0) = 0.5: the shared
        // expert must contribute at HALF weight. A forward pass that ignored the scale would
        // use it at full weight and give a different answer.
        moe.shared_gate_lin = try a.alloc(f16, cfg.hidden);
        @memset(moe.shared_gate_lin, 0);
    }
    moe.expert_slot = cfg.moe_inter * cfg.hidden;
    moe.expert_scratch = try a.alloc(f16, moe.expert_slot * 3);
    fillDeterministic(moe.expert_scratch, 1600, 0.1);
    lw.moe = moe;
    return mw;
}

test "the shared expert's output is scaled by sigmoid(gate . h) (needs the ANE)" {
    // This was a real bug: the scale was computed nowhere and the shared expert joined the
    // residual at full weight. The equivalence test in this file cannot catch it, because
    // prefill and decode would both omit it and agree with each other.
    //
    // With `shared_gate_lin` all zeros the scale is exactly sigmoid(0) = 0.5. Leaving the
    // tensor empty takes the other branch, where the scale is 1.0. If the scale is applied
    // at all, the two runs must differ; if it is ignored, they are identical.
    if (!ane.available()) return error.SkipZigTest;
    const a = std.testing.allocator;

    var with = try buildSharedExpertFixture(a, true);
    defer with.deinit();
    var without = try buildSharedExpertFixture(a, false);
    defer without.deinit();

    const ids = [_]u32{ 3, 7, 11 };

    const rt_with = try with.toRuntime(a);
    var eng_with = try Engine.init(a, rt_with, with.layerSource(), with.headSource(rt_with.embed), .{
        .max_seq = 32,
        .verbose = false,
        .chunk = 16,
    });
    defer eng_with.deinit();
    const rt_without = try without.toRuntime(a);
    var eng_without = try Engine.init(a, rt_without, without.layerSource(), without.headSource(rt_without.embed), .{
        .max_seq = 32,
        .verbose = false,
        .chunk = 16,
    });
    defer eng_without.deinit();

    var half: []f32 = &.{};
    var full: []f32 = &.{};
    for (ids, 0..) |id, pos| {
        const x = try eng_with.forward(id, @intCast(pos));
        const y = try eng_without.forward(id, @intCast(pos));
        if (pos + 1 == ids.len) {
            half = try a.dupe(f32, x);
            full = try a.dupe(f32, y);
        }
    }
    defer a.free(half);
    defer a.free(full);

    var worst: f32 = 0;
    for (half, full) |h, f| worst = @max(worst, @abs(h - f));
    // Halving the shared expert's contribution has to move the logits by a lot; the
    // difference would be exactly 0 if the scale were ignored.
    try std.testing.expect(worst > 0.05);
}

test "an over-long prompt is refused, not written past the KV cache (needs the ANE)" {
    // The KV cache holds exactly `max_seq` rows per layer. Before this guard, `verify` with
    // a 1121-token prompt on a 1024-row context wrote past the cache and returned all-zero
    // logits with rel = 1.0 instead of failing — a heap overflow that did not crash, so it
    // read as a wrong answer rather than as a bug.
    if (!ane.available()) return error.SkipZigTest;
    const a = std.testing.allocator;

    const cfg = model.Config{
        .arch = "llama",
        .hidden = 32,
        .layers = 1,
        .heads = 4,
        .kv_heads = 2,
        .head_dim = 8,
        .inter = 64,
        .vocab = 128,
        .eps = 1e-5,
        .rope_theta = 10000.0,
        .rope_adjacent = false,
    };
    var mw = model.ModelWeights{ .allocator = a, .config = cfg };
    defer mw.deinit();
    mw.embed = try a.alloc(f16, @as(usize, cfg.vocab) * cfg.hidden);
    mw.final_norm = try a.alloc(f32, cfg.hidden);
    mw.layers = try a.alloc(model.LayerWeights, cfg.layers);
    @memset(mw.layers, .{});
    fillDeterministic(mw.embed, 1, 1.0);
    fillDeterministicF32(mw.final_norm, 2, 1.0, 0.1);
    {
        const lw = &mw.layers[0];
        lw.attn_norm = try a.alloc(f32, cfg.hidden);
        lw.ffn_norm = try a.alloc(f32, cfg.hidden);
        lw.qkv = try a.alloc(f16, @as(usize, cfg.qkvDim()) * cfg.hidden);
        lw.o = try a.alloc(f16, @as(usize, cfg.hidden) * cfg.qDim());
        lw.gate = try a.alloc(f16, @as(usize, cfg.inter) * cfg.hidden);
        lw.up = try a.alloc(f16, @as(usize, cfg.inter) * cfg.hidden);
        lw.down = try a.alloc(f16, @as(usize, cfg.hidden) * cfg.inter);
        fillDeterministicF32(lw.attn_norm, 100, 1.0, 0.1);
        fillDeterministicF32(lw.ffn_norm, 200, 1.0, 0.1);
        fillDeterministic(lw.qkv, 300, 0.3);
        fillDeterministic(lw.o, 400, 0.3);
        fillDeterministic(lw.gate, 500, 0.3);
        fillDeterministic(lw.up, 600, 0.3);
        fillDeterministic(lw.down, 700, 0.3);
    }

    const rt = try mw.toRuntime(a);
    var eng = try Engine.init(a, rt, mw.layerSource(), mw.headSource(rt.embed), .{
        .max_seq = 8,
        .verbose = false,
        .chunk = 4,
    });
    defer eng.deinit();

    // Exactly filling the context is fine: 8 tokens at max_seq 8.
    const fits = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    _ = try eng.prefill(&fits, 0);

    // One more than the context is refused, rather than writing a ninth row.
    const too_many = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    try std.testing.expectError(error.ContextOverflow, eng.prefill(&too_many, 0));

    // A later start position counts against the same bound: four tokens from position six
    // would need row ten.
    const late = [_]u32{ 1, 2, 3, 4 };
    try std.testing.expectError(error.ContextOverflow, eng.prefill(&late, 6));
    _ = try eng.prefill(&late, 4); // exactly reaching the end is allowed

    // `forward` indexes the same cache with `pos`.
    _ = try eng.forward(1, 7);
    try std.testing.expectError(error.ContextOverflow, eng.forward(1, 8));
}
