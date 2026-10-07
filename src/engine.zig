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
    /// EXPERIMENTAL: fuse gate/up/SiLU/down into one ANE program (3 BLOBFILEs).
    /// Off by default: the fused program is numerically wrong at real model
    /// sizes (see README "Known issues"), while the split path is exact.
    fuse_ffn: bool = false,
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

    embed: []const f16,
    final_norm: []const f32,
    head: []const f16,
    layers_weights: []const model.LayerWeights,

    kernels: []LayerKernels,
    head_kernel: ane.Kernel,

    k_cache: [][]f32,
    v_cache: [][]f32,
    max_seq: u32,

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
        self.* = undefined;
    }

    /// Build every ANE kernel. Weights stay owned by `mw`.
    pub fn init(allocator: std.mem.Allocator, mw: *const model.ModelWeights, opts: Options) !Engine {
        const cfg = mw.config;
        try cfg.validate();

        var self: Engine = undefined;
        self.allocator = allocator;
        self.config = cfg;
        self.opts = opts;
        self.embed = mw.embed;
        self.final_norm = mw.final_norm;
        self.head = mw.headWeights();
        self.layers_weights = mw.layers;
        self.max_seq = opts.max_seq;
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
        for (mw.layers, 0..) |*lw, i| {
            if (opts.verbose) sys.print("  layer {d}/{d}: compiling ANE kernels\n", .{ i + 1, L });
            self.kernels[i] = try buildLayerKernels(allocator, cfg, lw, opts);
            built += 1;
        }
        if (opts.verbose) sys.print("  lm head: compiling ANE kernel ({d} -> {d})\n", .{ cfg.hidden, cfg.vocab });
        self.head_kernel = try makeConvKernel(allocator, cfg.hidden, cfg.vocab, self.head, "lm_head");

        const kv_dim: usize = cfg.kvDim();
        self.k_cache = try allocator.alloc([]f32, L);
        self.v_cache = try allocator.alloc([]f32, L);
        for (0..L) |i| {
            self.k_cache[i] = try allocator.alloc(f32, @as(usize, opts.max_seq) * kv_dim);
            self.v_cache[i] = try allocator.alloc(f32, @as(usize, opts.max_seq) * kv_dim);
            @memset(self.k_cache[i], 0);
            @memset(self.v_cache[i], 0);
        }

        self.x = try allocator.alloc(f32, cfg.hidden);
        self.h = try allocator.alloc(f32, cfg.hidden);
        self.qkv = try allocator.alloc(f32, cfg.qkvDim());
        self.attn = try allocator.alloc(f32, cfg.qDim());
        self.proj = try allocator.alloc(f32, cfg.hidden);
        self.gu = try allocator.alloc(f32, 2 * cfg.inter);
        self.act = try allocator.alloc(f32, cfg.inter);
        self.logits = try allocator.alloc(f32, cfg.vocab);
        self.scores = try allocator.alloc(f32, opts.max_seq);
        const stage = @max(@max(cfg.hidden, cfg.qkvDim()), @max(cfg.qDim(), @max(2 * cfg.inter, cfg.inter)));
        self.in16 = try allocator.alloc(f16, stage);
        self.out16 = try allocator.alloc(f16, @max(stage, cfg.vocab));
        return self;
    }

    fn f32ToF16Into(dst: []f16, src: []const f32) void {
        for (dst, src) |*d, s| d.* = @floatCast(s);
    }
    fn f16ToF32Into(dst: []f32, src: []const f16) void {
        for (dst, src) |*d, s| d.* = @floatCast(s);
    }

    /// One decode step for `token` at position `pos`. Returns the logits slice
    /// (owned by the engine, valid until the next call).
    pub fn forward(self: *Engine, token: u32, pos: u32) ![]f32 {
        const cfg = self.config;
        const hidden: usize = cfg.hidden;
        const kv_dim: usize = cfg.kvDim();
        const q_dim: usize = cfg.qDim();
        const inter: usize = cfg.inter;
        const t_start = sys.nowNs();

        f16ToF32Into(self.x, self.embed[@as(usize, token) * hidden ..][0..hidden]);

        for (self.kernels, 0..) |*k, li| {
            const lw = &self.layers_weights[li];

            // ---- attention: qkv projection on ANE ----
            cpu.rmsnorm(self.h, self.x, lw.attn_norm, cfg.eps);
            f32ToF16Into(self.in16[0..hidden], self.h);
            try k.qkv.writeInputF16(0, self.in16[0..hidden]);
            var t0 = sys.nowNs();
            try k.qkv.eval();
            self.stats.ane_eval_ns += sys.nowNs() - t0;
            self.stats.ane_evals += 1;
            try k.qkv.readOutputF16(0, self.out16[0..cfg.qkvDim()]);
            f16ToF32Into(self.qkv, self.out16[0..cfg.qkvDim()]);
            if (lw.qkv_bias) |b| cpu.addInPlace(self.qkv, b);

            const q = self.qkv[0..q_dim];
            const kk = self.qkv[q_dim..][0..kv_dim];
            const vv = self.qkv[q_dim + kv_dim ..][0..kv_dim];
            if (cfg.rope_adjacent) {
                cpu.ropeAdjacent(q, cfg.heads, cfg.head_dim, pos, cfg.rope_theta);
                cpu.ropeAdjacent(kk, cfg.kv_heads, cfg.head_dim, pos, cfg.rope_theta);
            } else {
                cpu.rope(q, cfg.heads, cfg.head_dim, pos, cfg.rope_theta);
                cpu.rope(kk, cfg.kv_heads, cfg.head_dim, pos, cfg.rope_theta);
            }

            @memcpy(self.k_cache[li][@as(usize, pos) * kv_dim ..][0..kv_dim], kk);
            @memcpy(self.v_cache[li][@as(usize, pos) * kv_dim ..][0..kv_dim], vv);

            cpu.attentionDecode(self.attn, q, self.k_cache[li], self.v_cache[li], pos + 1, cfg.heads, cfg.kv_heads, cfg.head_dim, self.scores);

            // ---- attention output projection on ANE ----
            f32ToF16Into(self.in16[0..q_dim], self.attn);
            try k.o.writeInputF16(0, self.in16[0..q_dim]);
            t0 = sys.nowNs();
            try k.o.eval();
            self.stats.ane_eval_ns += sys.nowNs() - t0;
            self.stats.ane_evals += 1;
            try k.o.readOutputF16(0, self.out16[0..hidden]);
            f16ToF32Into(self.proj, self.out16[0..hidden]);
            if (lw.o_bias) |b| cpu.addInPlace(self.proj, b);
            cpu.addInPlace(self.x, self.proj);

            // ---- feed-forward on ANE ----
            cpu.rmsnorm(self.h, self.x, lw.ffn_norm, cfg.eps);
            f32ToF16Into(self.in16[0..hidden], self.h);
            try k.ffn.writeInputF16(0, self.in16[0..hidden]);
            t0 = sys.nowNs();
            try k.ffn.eval();
            self.stats.ane_eval_ns += sys.nowNs() - t0;
            self.stats.ane_evals += 1;

            if (k.ffn_split) {
                try k.ffn.readOutputF16(0, self.out16[0 .. 2 * inter]);
                f16ToF32Into(self.gu, self.out16[0 .. 2 * inter]);
                cpu.siluMul(self.act, self.gu[0..inter], self.gu[inter..][0..inter]);
                f32ToF16Into(self.in16[0..inter], self.act);
                const dk = &k.down.?;
                try dk.writeInputF16(0, self.in16[0..inter]);
                t0 = sys.nowNs();
                try dk.eval();
                self.stats.ane_eval_ns += sys.nowNs() - t0;
                self.stats.ane_evals += 1;
                try dk.readOutputF16(0, self.out16[0..hidden]);
            } else {
                try k.ffn.readOutputF16(0, self.out16[0..hidden]);
            }
            f16ToF32Into(self.proj, self.out16[0..hidden]);
            cpu.addInPlace(self.x, self.proj);
        }

        // ---- final norm + lm head on ANE ----
        cpu.rmsnorm(self.h, self.x, self.final_norm, cfg.eps);
        f32ToF16Into(self.in16[0..hidden], self.h);
        try self.head_kernel.writeInputF16(0, self.in16[0..hidden]);
        const t4 = sys.nowNs();
        try self.head_kernel.eval();
        self.stats.ane_eval_ns += sys.nowNs() - t4;
        self.stats.ane_evals += 1;
        try self.head_kernel.readOutputF16(0, self.out16[0..cfg.vocab]);
        f16ToF32Into(self.logits, self.out16[0..cfg.vocab]);

        self.stats.total_ns += sys.nowNs() - t_start;
        self.stats.tokens += 1;
        return self.logits;
    }

    /// Compare each ANE kernel against a CPU matmul with the same weights, to
    /// localise a broken kernel when end-to-end output looks wrong.
    pub fn diagnose(self: *Engine, token: u32) !void {
        const cfg = self.config;
        const hidden: usize = cfg.hidden;
        const q_dim: usize = cfg.qDim();
        const inter: usize = cfg.inter;
        const lw = &self.layers_weights[0];

        f16ToF32Into(self.x, self.embed[@as(usize, token) * hidden ..][0..hidden]);
        cpu.rmsnorm(self.h, self.x, lw.attn_norm, cfg.eps);

        const ref = try self.allocator.alloc(f32, cfg.vocab);
        defer self.allocator.free(ref);

        // --- qkv ---
        f32ToF16Into(self.in16[0..hidden], self.h);
        try self.kernels[0].qkv.writeInputF16(0, self.in16[0..hidden]);
        try self.kernels[0].qkv.eval();
        try self.kernels[0].qkv.readOutputF16(0, self.out16[0..cfg.qkvDim()]);
        f16ToF32Into(self.qkv, self.out16[0..cfg.qkvDim()]);
        cpu.matmulF16(ref[0..cfg.qkvDim()], lw.qkv, self.h, cfg.qkvDim(), hidden);
        reportKernel("qkv", self.qkv, ref[0..cfg.qkvDim()]);

        // --- o projection (input: the ANE qkv's q part, so both use the same x) ---
        f32ToF16Into(self.in16[0..q_dim], self.qkv[0..q_dim]);
        try self.kernels[0].o.writeInputF16(0, self.in16[0..q_dim]);
        try self.kernels[0].o.eval();
        try self.kernels[0].o.readOutputF16(0, self.out16[0..hidden]);
        f16ToF32Into(self.proj, self.out16[0..hidden]);
        cpu.matmulF16(ref[0..hidden], lw.o, self.qkv[0..q_dim], hidden, q_dim);
        reportKernel("o", self.proj, ref[0..hidden]);

        // --- fused FFN ---
        f32ToF16Into(self.in16[0..hidden], self.h);
        try self.kernels[0].ffn.writeInputF16(0, self.in16[0..hidden]);
        try self.kernels[0].ffn.eval();
        if (self.kernels[0].ffn_split) {
            try self.kernels[0].ffn.readOutputF16(0, self.out16[0 .. 2 * inter]);
            f16ToF32Into(self.gu, self.out16[0 .. 2 * inter]);
            cpu.matmulF16(ref[0..inter], lw.gate, self.h, inter, hidden);
            reportKernel("gate", self.gu[0..inter], ref[0..inter]);
            cpu.matmulF16(ref[0..inter], lw.up, self.h, inter, hidden);
            reportKernel("up", self.gu[inter..][0..inter], ref[0..inter]);
        } else {
            try self.kernels[0].ffn.readOutputF16(0, self.out16[0..hidden]);
            f16ToF32Into(self.proj, self.out16[0..hidden]);
            const g = try self.allocator.alloc(f32, inter);
            defer self.allocator.free(g);
            const u = try self.allocator.alloc(f32, inter);
            defer self.allocator.free(u);
            const a = try self.allocator.alloc(f32, inter);
            defer self.allocator.free(a);
            cpu.matmulF16(g, lw.gate, self.h, inter, hidden);
            cpu.matmulF16(u, lw.up, self.h, inter, hidden);
            cpu.siluMul(a, g, u);
            cpu.matmulF16(ref[0..hidden], lw.down, a, hidden, inter);
            reportKernel("ffn(fused)", self.proj, ref[0..hidden]);
        }

        // --- lm head ---
        cpu.rmsnorm(self.h, self.x, self.final_norm, cfg.eps);
        f32ToF16Into(self.in16[0..hidden], self.h);
        try self.head_kernel.writeInputF16(0, self.in16[0..hidden]);
        try self.head_kernel.eval();
        try self.head_kernel.readOutputF16(0, self.out16[0..cfg.vocab]);
        f16ToF32Into(self.logits, self.out16[0..cfg.vocab]);
        cpu.matmulF16(ref, self.head, self.h, cfg.vocab, hidden);
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

/// Create a single-conv kernel: cin -> cout, weights already in ANE layout.
fn makeConvKernel(allocator: std.mem.Allocator, cin: u32, cout: u32, w: []const f16, label: []const u8) !ane.Kernel {
    std.debug.assert(w.len == @as(usize, cin) * cout);
    var blob = try weights.pack(allocator, &.{.{ .name = label, .data = w }});
    defer blob.deinit();
    var sym_buf: [64]u8 = undefined;
    const sym = try weights.symbolFor(&sym_buf, 0);
    const program = try mil.build(allocator, .{
        .inputs = &.{.{ .name = "i0", .channels = cin }},
        .ops = &.{.{ .conv = .{
            .x = "i0",
            .w = "w0",
            .y = "o0",
            .cin = cin,
            .cout = cout,
            .blob_offset = blob.offsets[0],
            .file = sym,
        } }},
        .outputs = &.{"o0"},
    });
    defer allocator.free(program);
    return ane.Kernel.create(allocator, program, &.{.{ .name = sym, .data = blob.bytes }});
}

/// Fused FFN: i0 -> gate conv, up conv, sigmoid, mul, mul, down conv -> o0.
/// Three BLOBFILEs and no inline scalars, so it stays inside the 16-slot budget.
fn makeFusedFfnKernel(
    allocator: std.mem.Allocator,
    hidden: u32,
    inter: u32,
    gate: []const f16,
    up: []const f16,
    down: []const f16,
) !ane.Kernel {
    var blob = try weights.pack(allocator, &.{
        .{ .name = "gate", .data = gate },
        .{ .name = "up", .data = up },
        .{ .name = "down", .data = down },
    });
    defer blob.deinit();
    var s0: [64]u8 = undefined;
    var s1: [64]u8 = undefined;
    var s2: [64]u8 = undefined;
    const n0 = try weights.symbolFor(&s0, 0);
    const n1 = try weights.symbolFor(&s1, 1);
    const n2 = try weights.symbolFor(&s2, 2);
    const program = try mil.build(allocator, .{
        .inputs = &.{.{ .name = "i0", .channels = hidden }},
        .ops = &.{
            .{ .conv = .{ .x = "i0", .w = "w0", .y = "t0", .cin = hidden, .cout = inter, .blob_offset = blob.offsets[0], .file = n0 } },
            .{ .conv = .{ .x = "i0", .w = "w1", .y = "t1", .cin = hidden, .cout = inter, .blob_offset = blob.offsets[1], .file = n1 } },
            .{ .sigmoid = .{ .x = "t0", .y = "t2", .channels = inter } },
            .{ .mul = .{ .a = "t0", .b = "t2", .y = "t3", .channels = inter } },
            .{ .mul = .{ .a = "t3", .b = "t1", .y = "t4", .channels = inter } },
            .{ .conv = .{ .x = "t4", .w = "w2", .y = "o0", .cin = inter, .cout = hidden, .blob_offset = blob.offsets[2], .file = n2 } },
        },
        .outputs = &.{"o0"},
    });
    defer allocator.free(program);
    // All three symbols point at the same file; the shim writes it once per name.
    return ane.Kernel.create(allocator, program, &.{
        .{ .name = n0, .data = blob.bytes },
        .{ .name = n1, .data = blob.bytes },
        .{ .name = n2, .data = blob.bytes },
    });
}

fn buildLayerKernels(allocator: std.mem.Allocator, cfg: model.Config, lw: *const model.LayerWeights, opts: Options) !LayerKernels {
    var qkv = try makeConvKernel(allocator, cfg.hidden, cfg.qkvDim(), lw.qkv, "qkv");
    errdefer qkv.deinit();
    var o = try makeConvKernel(allocator, cfg.qDim(), cfg.hidden, lw.o, "o");
    errdefer o.deinit();

    if (opts.fuse_ffn) {
        if (makeFusedFfnKernel(allocator, cfg.hidden, cfg.inter, lw.gate, lw.up, lw.down)) |fk| {
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
    var gu_k = try makeConvKernel(allocator, cfg.hidden, 2 * cfg.inter, gu, "gate_up");
    errdefer gu_k.deinit();
    var down_k = try makeConvKernel(allocator, cfg.inter, cfg.hidden, lw.down, "down");
    errdefer down_k.deinit();
    return .{ .qkv = qkv, .o = o, .ffn = gu_k, .ffn_split = true, .down = down_k };
}
