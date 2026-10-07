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
const model = @import("model.zig");
const ane = @import("ane/runtime.zig");
const mil = @import("ane/mil.zig");
const weights = @import("ane/weights.zig");

pub const Options = struct {
    max_seq: u32 = 2048,
    verbose: bool = true,
    /// Activation width every kernel is compiled for. The ANE reads the weights
    /// once per evaluation regardless of width (measured: width 128 costs 5%
    /// more than width 1), so one token and a whole prompt chunk cost the same.
    /// Must be a multiple of 32 for the planar layout to stay contiguous.
    chunk: u32 = 64,
    /// Fuse gate/up/SiLU/down into one ANE program (3 BLOBFILEs, one kernel
    /// instead of two). Keeps the intermediate activation on the ANE instead of
    /// round-tripping it through the CPU, and measured ~45% faster decode.
    /// Falls back to the split path if the fused program fails to compile.
    fuse_ffn: bool = true,
};

const LayerKernels = struct {
    qkv: ane.Kernel,
    o: ane.Kernel,
    /// Fused: [hidden] -> [hidden]. Split: [hidden] -> [2*inter] (gate || up).
    ffn: ane.Kernel,
    ffn_split: bool,
    down: ?ane.Kernel = null,
};

pub const Stats = struct {
    ane_eval_ns: u64 = 0,
    ane_evals: u64 = 0,
    total_ns: u64 = 0,
    tokens: u64 = 0,

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

    k_cache: [][]f32,
    v_cache: [][]f32,
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
    /// Contiguous scratch for one column's q/k/v/attention vectors.
    sq: []f32,
    sk: []f32,
    sv: []f32,
    sa: []f32,

    stats: Stats = .{},

    pub fn deinit(self: *Engine) void {
        for (self.kernels) |*k| {
            k.qkv.deinit();
            k.o.deinit();
            k.ffn.deinit();
            if (k.down) |*d| d.deinit();
        }
        self.head_kernel.deinit();
        self.allocator.free(self.kernels);
        self.rt.deinit();
        for (self.k_cache) |c| self.allocator.free(c);
        for (self.v_cache) |c| self.allocator.free(c);
        self.allocator.free(self.k_cache);
        self.allocator.free(self.v_cache);
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
        self.allocator = allocator;
        self.config = cfg;
        self.opts = opts;
        self.rt = rt;
        self.embed = rt.embed;
        self.final_norm = rt.final_norm;
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
        for (0..L) |i| {
            if (opts.verbose) sys.print("  layer {d}/{d}: loading weights + compiling ANE kernels\n", .{ i + 1, L });
            var m = try layers.load(allocator, @intCast(i));
            defer m.deinit(allocator); // matrices are baked into the kernels now
            self.kernels[i] = try buildLayerKernels(allocator, cfg, &m, opts, opts.chunk);
            built += 1;
        }
        if (opts.verbose) sys.print("  lm head: loading weights + compiling ANE kernel ({d} -> {d})\n", .{ cfg.hidden, cfg.vocab });
        const hw = try head.load(allocator);
        defer if (hw.owned) allocator.free(hw.data);
        self.head = hw.data;
        // The head stays width 1: a [vocab][chunk] output surface would be tens of MB.
        self.head_kernel = try makeConvKernel(allocator, cfg.hidden, cfg.vocab, hw.data, "lm_head", 1);

        const kv_dim: usize = cfg.kvDim();
        self.k_cache = try allocator.alloc([]f32, L);
        self.v_cache = try allocator.alloc([]f32, L);
        for (0..L) |i| {
            self.k_cache[i] = try allocator.alloc(f32, @as(usize, opts.max_seq) * kv_dim);
            self.v_cache[i] = try allocator.alloc(f32, @as(usize, opts.max_seq) * kv_dim);
            @memset(self.k_cache[i], 0);
            @memset(self.v_cache[i], 0);
        }

        const ch: usize = self.chunk;
        self.x = try allocator.alloc(f32, @as(usize, cfg.hidden) * ch);
        self.h = try allocator.alloc(f32, @as(usize, cfg.hidden) * ch);
        self.qkv = try allocator.alloc(f32, @as(usize, cfg.qkvDim()) * ch);
        self.attn = try allocator.alloc(f32, @as(usize, cfg.qDim()) * ch);
        self.proj = try allocator.alloc(f32, @as(usize, cfg.hidden) * ch);
        self.gu = try allocator.alloc(f32, @as(usize, 2 * cfg.inter) * ch);
        self.act = try allocator.alloc(f32, @as(usize, cfg.inter) * ch);
        self.logits = try allocator.alloc(f32, cfg.vocab);
        self.scores = try allocator.alloc(f32, opts.max_seq);
        const stage = @max(@max(cfg.hidden, cfg.qkvDim()), @max(cfg.qDim(), @max(2 * cfg.inter, cfg.inter)));
        self.in16 = try allocator.alloc(f16, @as(usize, stage) * ch);
        self.out16 = try allocator.alloc(f16, @as(usize, stage) * ch);
        self.dec_in = try allocator.alloc(f16, stage);
        self.dec_out = try allocator.alloc(f16, stage);
        self.head_in = try allocator.alloc(f16, cfg.hidden);
        const vec = @max(cfg.qDim(), @max(cfg.kvDim(), cfg.inter));
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
        const cfg = self.config;
        const ch = self.chunk;
        const hidden: usize = cfg.hidden;
        const kv_dim: usize = cfg.kvDim();
        const q_dim: usize = cfg.qDim();
        const inter: usize = cfg.inter;
        const t_start = sys.nowNs();

        for (0..hidden) |c| {
            self.x[c * ch] = @floatCast(self.embed[@as(usize, token) * hidden + c]);
        }

        for (self.kernels, 0..) |*k, li| {
            const norm = &self.norms[li];

            // ---- attention: qkv projection on ANE (column 0) ----
            rmsnormColumn(self.dec_in[0..hidden], self.x, norm.attn, cfg.eps, ch);
            try k.qkv.writeInputColumnF16(0, 0, self.dec_in[0..hidden]);
            var t0 = sys.nowNs();
            try k.qkv.eval();
            self.stats.ane_eval_ns += sys.nowNs() - t0;
            self.stats.ane_evals += 1;
            try k.qkv.readOutputColumnF16(0, 0, self.dec_out[0..cfg.qkvDim()]);
            for (0..cfg.qkvDim()) |c| self.qkv[c * ch] = @floatCast(self.dec_out[c]);
            if (norm.qkv_bias) |b| {
                for (b, 0..) |v, c| self.qkv[c * ch] += v;
            }

            gatherColumn(self.sq[0..q_dim], self.qkv[0..], ch, 0);
            gatherColumn(self.sk[0..kv_dim], self.qkv[q_dim * ch ..], ch, 0);
            gatherColumn(self.sv[0..kv_dim], self.qkv[(q_dim + kv_dim) * ch ..], ch, 0);
            self.ropeAndCache(li, pos, cfg, kv_dim);
            cpu.attentionDecode(self.sa[0..q_dim], self.sq[0..q_dim], self.k_cache[li], self.v_cache[li], pos + 1, cfg.heads, cfg.kv_heads, cfg.head_dim, self.scores);
            scatterColumn(self.attn[0..], self.sa[0..q_dim], ch, 0);

            // ---- attention output projection on ANE ----
            for (0..q_dim) |c| self.dec_in[c] = @floatCast(self.attn[c * ch]);
            try k.o.writeInputColumnF16(0, 0, self.dec_in[0..q_dim]);
            t0 = sys.nowNs();
            try k.o.eval();
            self.stats.ane_eval_ns += sys.nowNs() - t0;
            self.stats.ane_evals += 1;
            try k.o.readOutputColumnF16(0, 0, self.dec_out[0..hidden]);
            for (0..hidden) |c| {
                var v = @as(f32, @floatCast(self.dec_out[c]));
                if (norm.o_bias) |b| v += b[c];
                self.x[c * ch] += v;
            }

            // ---- feed-forward on ANE ----
            rmsnormColumn(self.dec_in[0..hidden], self.x, norm.ffn, cfg.eps, ch);
            try k.ffn.writeInputColumnF16(0, 0, self.dec_in[0..hidden]);
            t0 = sys.nowNs();
            try k.ffn.eval();
            self.stats.ane_eval_ns += sys.nowNs() - t0;
            self.stats.ane_evals += 1;

            if (k.ffn_split) {
                try k.ffn.readOutputColumnF16(0, 0, self.dec_out[0 .. 2 * inter]);
                const gate = self.sq[0..inter];
                const up = self.sk[0..inter];
                for (0..inter) |c| {
                    gate[c] = @floatCast(self.dec_out[c]);
                    up[c] = @floatCast(self.dec_out[inter + c]);
                }
                cpu.siluMul(self.sa[0..inter], gate, up);
                for (0..inter) |c| self.dec_in[c] = @floatCast(self.sa[c]);
                const dk = &k.down.?;
                try dk.writeInputColumnF16(0, 0, self.dec_in[0..inter]);
                t0 = sys.nowNs();
                try dk.eval();
                self.stats.ane_eval_ns += sys.nowNs() - t0;
                self.stats.ane_evals += 1;
                try dk.readOutputColumnF16(0, 0, self.dec_out[0..hidden]);
            } else {
                try k.ffn.readOutputColumnF16(0, 0, self.dec_out[0..hidden]);
            }
            for (0..hidden) |c| self.x[c * ch] += @floatCast(self.dec_out[c]);
        }

        // ---- final norm + lm head (width-1 kernel) ----
        rmsnormColumn(self.head_in[0..hidden], self.x, self.final_norm, cfg.eps, ch);
        try self.head_kernel.writeInputF16(0, self.head_in[0..hidden]);
        const t4 = sys.nowNs();
        try self.head_kernel.eval();
        self.stats.ane_eval_ns += sys.nowNs() - t4;
        self.stats.ane_evals += 1;
        try self.head_kernel.readOutputF16(0, self.out16[0..cfg.vocab]);
        for (self.logits, 0..) |*l, i| l.* = @floatCast(self.out16[i]);

        self.stats.total_ns += sys.nowNs() - t_start;
        self.stats.tokens += 1;
        return self.logits;
    }

    fn ropeAndCache(self: *Engine, li: usize, pos: u32, cfg: model.Config, kv_dim: usize) void {
        if (cfg.rope_adjacent) {
            cpu.ropeAdjacent(self.sq, cfg.heads, cfg.head_dim, pos, cfg.rope_theta);
            cpu.ropeAdjacent(self.sk[0..kv_dim], cfg.kv_heads, cfg.head_dim, pos, cfg.rope_theta);
        } else {
            cpu.rope(self.sq, cfg.heads, cfg.head_dim, pos, cfg.rope_theta);
            cpu.rope(self.sk[0..kv_dim], cfg.kv_heads, cfg.head_dim, pos, cfg.rope_theta);
        }
        @memcpy(self.k_cache[li][@as(usize, pos) * kv_dim ..][0..kv_dim], self.sk[0..kv_dim]);
        @memcpy(self.v_cache[li][@as(usize, pos) * kv_dim ..][0..kv_dim], self.sv[0..kv_dim]);
    }

    /// Process up to `chunk` prompt tokens in one pass and return the logits for
    /// the last one. This is where the weight-bandwidth-bound behaviour of the
    /// ANE pays off: a whole chunk costs the same as a single token.
    pub fn prefill(self: *Engine, ids: []const u32, start_pos: u32) ![]f32 {
        const cfg = self.config;
        const ch = self.chunk;
        const hidden: usize = cfg.hidden;
        const kv_dim: usize = cfg.kvDim();
        const q_dim: usize = cfg.qDim();
        const inter: usize = cfg.inter;

        var done: usize = 0;
        while (done < ids.len) {
            const n = @min(ch, ids.len - done);
            const base_pos: u32 = start_pos + @as(u32, @intCast(done));

            for (0..n) |j| {
                const row = self.embed[@as(usize, ids[done + j]) * hidden ..][0..hidden];
                for (0..hidden) |c| self.x[c * ch + j] = @floatCast(row[c]);
            }
            for (n..ch) |j| {
                for (0..hidden) |c| self.x[c * ch + j] = 0;
            }

            for (self.kernels, 0..) |*k, li| {
                const norm = &self.norms[li];

                // ---- qkv projection for the whole chunk ----
                rmsnormChunk(self.in16[0 .. hidden * ch], self.x, norm.attn, cfg.eps, ch, n);
                try k.qkv.writeInputF16(0, self.in16[0 .. hidden * ch]);
                var t0 = sys.nowNs();
                try k.qkv.eval();
                self.stats.ane_eval_ns += sys.nowNs() - t0;
                self.stats.ane_evals += 1;
                const qkv_len = @as(usize, cfg.qkvDim()) * ch;
                try k.qkv.readOutputF16(0, self.out16[0..qkv_len]);
                for (0..qkv_len) |i| self.qkv[i] = @floatCast(self.out16[i]);
                if (norm.qkv_bias) |b| {
                    for (0..n) |j| for (b, 0..) |v, c| {
                        self.qkv[c * ch + j] += v;
                    };
                }

                // ---- RoPE + KV cache, one position at a time ----
                // The rotated query is written back into the chunk buffer so the
                // attention pass below sees it (the decode path keeps it in sq).
                for (0..n) |j| {
                    const pos = base_pos + @as(u32, @intCast(j));
                    gatherColumn(self.sq[0..q_dim], self.qkv[0..], ch, j);
                    gatherColumn(self.sk[0..kv_dim], self.qkv[q_dim * ch ..], ch, j);
                    gatherColumn(self.sv[0..kv_dim], self.qkv[(q_dim + kv_dim) * ch ..], ch, j);
                    self.ropeAndCache(li, pos, cfg, kv_dim);
                    scatterColumn(self.qkv[0..], self.sq[0..q_dim], ch, j);
                }

                // ---- causal attention, one position at a time ----
                for (0..n) |j| {
                    const pos = base_pos + @as(u32, @intCast(j));
                    gatherColumn(self.sq[0..q_dim], self.qkv[0..], ch, j);
                    cpu.attentionDecode(self.sa[0..q_dim], self.sq[0..q_dim], self.k_cache[li], self.v_cache[li], pos + 1, cfg.heads, cfg.kv_heads, cfg.head_dim, self.scores);
                    scatterColumn(self.attn[0..], self.sa[0..q_dim], ch, j);
                }

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
                        self.x[c * ch + j] += v;
                    }
                }

                // ---- feed-forward ----
                rmsnormChunk(self.in16[0 .. hidden * ch], self.x, norm.ffn, cfg.eps, ch, n);
                try k.ffn.writeInputF16(0, self.in16[0 .. hidden * ch]);
                t0 = sys.nowNs();
                try k.ffn.eval();
                self.stats.ane_eval_ns += sys.nowNs() - t0;
                self.stats.ane_evals += 1;

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
                        cpu.siluMul(self.sa[0..inter], gate, up);
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
                for (0..n) |j| {
                    for (0..hidden) |c| self.x[c * ch + j] += @floatCast(self.out16[c * ch + j]);
                }
            }

            // Logits for the last real column, via the width-1 head kernel.
            const last = n - 1;
            for (0..hidden) |c| self.h[c * ch] = self.x[c * ch + last];
            rmsnormColumn(self.head_in[0..hidden], self.h, self.final_norm, cfg.eps, ch);
            try self.head_kernel.writeInputF16(0, self.head_in[0..hidden]);
            const th = sys.nowNs();
            try self.head_kernel.eval();
            self.stats.ane_eval_ns += sys.nowNs() - th;
            self.stats.ane_evals += 1;
            try self.head_kernel.readOutputF16(0, self.out16[0..cfg.vocab]);
            for (self.logits, 0..) |*l, i| l.* = @floatCast(self.out16[i]);

            done += n;
            self.stats.tokens += 1;
        }
        return self.logits;
    }

    /// Compare each ANE kernel against a CPU matmul with the same weights, to
    /// localise a broken kernel when end-to-end output looks wrong. Uses column
    /// 0 of the chunked kernels.
    pub fn diagnose(self: *Engine, token: u32, lw: *const model.Matrices) !void {
        const cfg = self.config;
        const ch = self.chunk;
        const hidden: usize = cfg.hidden;
        const q_dim: usize = cfg.qDim();
        const inter: usize = cfg.inter;

        for (0..hidden) |c| self.x[c * ch] = @floatCast(self.embed[@as(usize, token) * hidden + c]);
        rmsnormColumn(self.dec_in[0..hidden], self.x, self.norms[0].attn, cfg.eps, ch);

        const ref = try self.allocator.alloc(f32, cfg.vocab);
        defer self.allocator.free(ref);
        const h32 = try self.allocator.alloc(f32, @max(hidden, @max(cfg.qkvDim(), inter)));
        defer self.allocator.free(h32);
        const got = try self.allocator.alloc(f32, @max(cfg.qkvDim(), @max(hidden, inter)));
        defer self.allocator.free(got);

        // --- qkv ---
        try self.kernels[0].qkv.writeInputColumnF16(0, 0, self.dec_in[0..hidden]);
        try self.kernels[0].qkv.eval();
        try self.kernels[0].qkv.readOutputColumnF16(0, 0, self.dec_out[0..cfg.qkvDim()]);
        for (0..hidden) |c| h32[c] = @floatCast(self.dec_in[c]);
        for (0..cfg.qkvDim()) |c| got[c] = @floatCast(self.dec_out[c]);
        cpu.matmulF16(ref[0..cfg.qkvDim()], lw.qkv, h32[0..hidden], cfg.qkvDim(), hidden);
        reportKernel("qkv", got[0..cfg.qkvDim()], ref[0..cfg.qkvDim()]);

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
        try self.head_kernel.writeInputF16(0, self.head_in[0..hidden]);
        try self.head_kernel.eval();
        try self.head_kernel.readOutputF16(0, self.out16[0..cfg.vocab]);
        for (0..hidden) |c| h32[c] = @floatCast(self.head_in[c]);
        for (self.logits, 0..) |*l, i| l.* = @floatCast(self.out16[i]);
        cpu.matmulF16(ref, self.head, h32[0..hidden], cfg.vocab, hidden);
        reportKernel("lm_head", self.logits, ref);
    }

    /// Forget the KV cache (start a new conversation).
    pub fn reset(self: *Engine) void {
        for (self.k_cache) |c| @memset(c, 0);
        for (self.v_cache) |c| @memset(c, 0);
        self.stats = .{};
    }
};

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
    sys.print("    {s:<12} max|ANE-CPU| = {e:.5}  rel = {e:.5}  nan = {d}  -> {s}\n", .{
        name, max_err, rel, nan, if (rel < 0.02 and nan == 0) "ok" else "BROKEN",
    });
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
    std.debug.assert(w.len == @as(usize, cin) * cout);
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
        .chunks = &.{std.mem.sliceAsBytes(w)},
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

fn buildLayerKernels(allocator: std.mem.Allocator, cfg: model.Config, lw: *const model.Matrices, opts: Options, width: u32) !LayerKernels {
    var qkv = try makeConvKernel(allocator, cfg.hidden, cfg.qkvDim(), lw.qkv, "qkv", width);
    errdefer qkv.deinit();
    var o = try makeConvKernel(allocator, cfg.qDim(), cfg.hidden, lw.o, "o", width);
    errdefer o.deinit();

    if (opts.fuse_ffn) {
        if (makeFusedFfnKernel(allocator, cfg.hidden, cfg.inter, lw.gate, lw.up, lw.down, width)) |fk| {
            return .{ .qkv = qkv, .o = o, .ffn = fk, .ffn_split = false };
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
    var gu_k = try makeConvKernel(allocator, cfg.hidden, 2 * cfg.inter, gu, "gate_up", width);
    errdefer gu_k.deinit();
    var down_k = try makeConvKernel(allocator, cfg.inter, cfg.hidden, lw.down, "down", width);
    errdefer down_k.deinit();
    return .{ .qkv = qkv, .o = o, .ffn = gu_k, .ffn_split = true, .down = down_k };
}
