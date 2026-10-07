// cpu.zig — the non-matmul half of the transformer, kept on the CPU.
//
// The ANE is very good at the big matrix products but its MIL dialect rejects
// `rsqrt`, `reduce_mean`, `concat` and `gelu`, and it silently ignores causal
// masks in SDPA. Normalisation, RoPE, attention, activations and sampling are
// therefore done here; only the FLOP-heavy projections run on the ANE.

const std = @import("std");

/// out = x * rsqrt(mean(x^2) + eps) * weight
pub fn rmsnorm(out: []f32, x: []const f32, weight: []const f32, eps: f32) void {
    std.debug.assert(out.len == x.len and weight.len == x.len);
    var acc: f32 = 0;
    for (x) |v| acc += v * v;
    const mean = acc / @as(f32, @floatFromInt(x.len));
    const inv = 1.0 / @sqrt(mean + eps);
    for (out, x, weight) |*o, xi, wi| o.* = xi * inv * wi;
}

/// In-place rotary position embedding (Llama-style, adjacent pairs).
/// `vec` holds `heads * head_dim` values; rotation applies per head.
pub fn rope(vec: []f32, heads: u32, head_dim: u32, pos: u32, theta: f32) void {
    const hd: usize = head_dim;
    const half = hd / 2;
    for (0..heads) |h| {
        const base = h * hd;
        for (0..half) |i| {
            const freq = 1.0 / std.math.pow(f32, theta, @as(f32, @floatFromInt(2 * i)) / @as(f32, @floatFromInt(hd)));
            const angle = @as(f32, @floatFromInt(pos)) * freq;
            const c = @cos(angle);
            const s = @sin(angle);
            const a = vec[base + i];
            const b = vec[base + i + half];
            vec[base + i] = a * c - b * s;
            vec[base + i + half] = a * s + b * c;
        }
    }
}

/// In-place RoPE in the llama.cpp / GGUF convention: adjacent pairs (2j, 2j+1).
///
/// `convert_hf_to_gguf.py` permutes the Q and K weight rows of Llama-family
/// models (llama, qwen2/3, smollm, ...) into this layout, so a GGUF model must
/// be rotated this way — using the HF half-split convention here silently
/// produces fluent-looking garbage.
pub fn ropeAdjacent(vec: []f32, heads: u32, head_dim: u32, pos: u32, theta: f32) void {
    const hd: usize = head_dim;
    const half = hd / 2;
    for (0..heads) |h| {
        const base = h * hd;
        for (0..half) |j| {
            const freq = 1.0 / std.math.pow(f32, theta, @as(f32, @floatFromInt(2 * j)) / @as(f32, @floatFromInt(hd)));
            const angle = @as(f32, @floatFromInt(pos)) * freq;
            const c = @cos(angle);
            const s = @sin(angle);
            const a = vec[base + 2 * j];
            const b = vec[base + 2 * j + 1];
            vec[base + 2 * j] = a * c - b * s;
            vec[base + 2 * j + 1] = a * s + b * c;
        }
    }
}

/// SiLU(gate) * up, elementwise.
pub fn siluMul(out: []f32, gate: []const f32, up: []const f32) void {
    for (out, gate, up) |*o, g, u| {
        const s = 1.0 / (1.0 + @exp(-g));
        o.* = g * s * u;
    }
}

pub fn addInPlace(dst: []f32, src: []const f32) void {
    for (dst, src) |*d, s| d.* += s;
}

/// Softmax over `scores[0..n]` in place.
pub fn softmax(scores: []f32) void {
    if (scores.len == 0) return;
    var max: f32 = scores[0];
    for (scores[1..]) |v| max = @max(max, v);
    var sum: f32 = 0;
    for (scores) |*v| {
        v.* = @exp(v.* - max);
        sum += v.*;
    }
    if (sum == 0) return;
    const inv = 1.0 / sum;
    for (scores) |*v| v.* *= inv;
}

/// Multi-head attention for one decode step.
///
///   q     : [n_heads * head_dim]           (query for the current token)
///   k_cache/v_cache : [n_kv_heads * head_dim] per past position
///   out   : [n_heads * head_dim]
///
/// GQA: query head h reads KV head h / (n_heads / n_kv_heads).
pub fn attentionDecode(
    out: []f32,
    q: []const f32,
    k_cache: []const f32,
    v_cache: []const f32,
    n_past: usize,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    scores_scratch: []f32,
) void {
    const hd: usize = head_dim;
    const kv_dim: usize = @as(usize, n_kv_heads) * hd;
    const group = n_heads / n_kv_heads;
    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));

    std.debug.assert(scores_scratch.len >= n_past);

    for (0..n_heads) |h| {
        const kvh = h / group;
        const q_h = q[h * hd ..][0..hd];
        const scores = scores_scratch[0..n_past];
        for (0..n_past) |t| {
            const k_t = k_cache[t * kv_dim + kvh * hd ..][0..hd];
            var dot: f32 = 0;
            for (q_h, k_t) |a, b| dot += a * b;
            scores[t] = dot * scale;
        }
        softmax(scores);
        const o_h = out[h * hd ..][0..hd];
        @memset(o_h, 0);
        for (0..n_past) |t| {
            const v_t = v_cache[t * kv_dim + kvh * hd ..][0..hd];
            const w = scores[t];
            for (o_h, v_t) |*o, vv| o.* += w * vv;
        }
    }
}

/// y = W x with W stored as fp16 [out][in] (the ANE conv layout).
pub fn matmulF16(out: []f32, w: []const f16, x: []const f32, out_dim: usize, in_dim: usize) void {
    for (0..out_dim) |o| {
        var acc: f32 = 0;
        for (0..in_dim) |i| acc += @as(f32, @floatCast(w[o * in_dim + i])) * x[i];
        out[o] = acc;
    }
}

pub fn argmax(values: []const f32) u32 {
    var best: usize = 0;
    for (values, 0..) |v, i| {
        if (v > values[best]) best = i;
    }
    return @intCast(best);
}

/// One candidate token during sampling.
pub const Candidate = struct {
    logit: f32,
    index: u32,
};

pub const SamplerParams = struct {
    temperature: f32 = 1.0,
    /// 0 disables top-k.
    top_k: usize = 0,
    /// 1.0 disables nucleus sampling.
    top_p: f32 = 1.0,
    /// >1.0 discourages tokens already in `recent` (llama.cpp semantics).
    repetition_penalty: f32 = 1.0,
    /// Flat subtraction for tokens already present (OpenAI semantics).
    presence_penalty: f32 = 0.0,
    /// Subtracted once per occurrence (OpenAI semantics).
    frequency_penalty: f32 = 0.0,
    /// Logits below `max_logit - min_keep_delta * temperature` are ignored
    /// entirely: their softmax weight is < e^-20 and cannot be sampled.
    min_keep_delta: f32 = 20.0,
};

fn compareCandidates(_: void, a: Candidate, b: Candidate) bool {
    return a.logit > b.logit;
}

/// Apply repetition / presence / frequency penalties to `logits` in place.
pub fn applyPenalties(logits: []f32, params: SamplerParams, recent: []const u32) void {
    if (params.repetition_penalty == 1.0 and params.presence_penalty == 0.0 and params.frequency_penalty == 0.0) return;
    for (recent) |id| {
        if (id >= logits.len) continue;
        var v = logits[id];
        if (params.repetition_penalty != 1.0) {
            v = if (v > 0) v / params.repetition_penalty else v * params.repetition_penalty;
        }
        v -= params.presence_penalty;
        v -= params.frequency_penalty;
        logits[id] = v;
    }
}

/// Temperature / top-k / top-p sampler with penalties.
///
/// `scratch` must hold at least `logits.len` candidates and is reused across
/// calls. Returns the chosen token id.
pub fn sample(
    logits: []f32,
    params: SamplerParams,
    recent: []const u32,
    rng: *u32,
    scratch: []Candidate,
) u32 {
    std.debug.assert(scratch.len >= logits.len);
    applyPenalties(logits, params, recent);

    if (params.temperature <= 0) return argmax(logits);

    // Pass 1: max (used as the softmax reference and the cut-off).
    var max_v: f32 = -std.math.inf(f32);
    for (logits) |v| max_v = @max(max_v, v);

    const cut = max_v - params.min_keep_delta * params.temperature;

    // Pass 2: collect the candidates worth considering.
    var n: usize = 0;
    for (logits, 0..) |v, i| {
        if (v >= cut) {
            scratch[n] = .{ .logit = v, .index = @intCast(i) };
            n += 1;
        }
    }
    if (n == 0) return argmax(logits);

    // Top-k: keep only the k best (sorting n candidates, n is small).
    var keep = n;
    if (params.top_k > 0 and params.top_k < n) {
        std.mem.sort(Candidate, scratch[0..n], {}, compareCandidates);
        keep = params.top_k;
    }

    // Probabilities over the kept set.
    var sum: f32 = 0;
    for (scratch[0..keep]) |*c| {
        c.logit = @exp((c.logit - max_v) / params.temperature);
        sum += c.logit;
    }
    if (sum <= 0) return argmax(logits);

    // Top-p: sort and truncate at the nucleus (only needed if enabled).
    if (params.top_p < 1.0) {
        std.mem.sort(Candidate, scratch[0..keep], {}, compareCandidates);
        var acc: f32 = 0;
        var cut_n: usize = keep;
        for (scratch[0..keep], 0..) |c, i| {
            acc += c.logit / sum;
            if (acc >= params.top_p) {
                cut_n = i + 1;
                break;
            }
        }
        keep = cut_n;
        sum = 0;
        for (scratch[0..keep]) |c| sum += c.logit;
        if (sum <= 0) return argmax(logits);
    }

    rng.* = rng.* *% 1664525 +% 1013904223;
    const r = @as(f32, @floatFromInt(rng.* >> 8)) / 16777216.0;
    var target = r * sum;
    for (scratch[0..keep]) |c| {
        target -= c.logit;
        if (target <= 0) return c.index;
    }
    return scratch[0].index;
}

test "rmsnorm normalises to unit rms" {
    var out: [4]f32 = undefined;
    const x = [_]f32{ 1, 2, 3, 4 };
    const w = [_]f32{ 1, 1, 1, 1 };
    rmsnorm(&out, &x, &w, 0);
    var acc: f32 = 0;
    for (out) |v| acc += v * v;
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), acc, 1e-4);
}

test "ropeAdjacent preserves vector norm" {
    var v = [_]f32{ 1, 0, 0, 1, 0, 1, 1, 0 };
    var before: f32 = 0;
    for (v) |x| before += x * x;
    ropeAdjacent(&v, 2, 4, 7, 100000.0);
    var after: f32 = 0;
    for (v) |x| after += x * x;
    try std.testing.expectApproxEqAbs(before, after, 1e-4);
}

test "rope preserves vector norm" {
    var v = [_]f32{ 1, 0, 0, 0, 0, 1, 0, 0 };
    const before: f32 = blk: {
        var s: f32 = 0;
        for (v) |x| s += x * x;
        break :blk s;
    };
    rope(&v, 2, 4, 3, 10000.0);
    var after: f32 = 0;
    for (v) |x| after += x * x;
    try std.testing.expectApproxEqAbs(before, after, 1e-4);
}

test "softmax sums to one" {
    var s = [_]f32{ 1, 2, 3, 4 };
    softmax(&s);
    var sum: f32 = 0;
    for (s) |v| sum += v;
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), sum, 1e-5);
}

test "siluMul matches reference values" {
    var out: [2]f32 = undefined;
    siluMul(&out, &[_]f32{ 0.0, 1.0 }, &[_]f32{ 1.0, 1.0 });
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), out[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.7310586), out[1], 1e-5);
}

test "attentionDecode single past position returns v" {
    var out: [4]f32 = undefined;
    const q = [_]f32{ 1, 0, 1, 0 };
    const k = [_]f32{ 1, 0, 1, 0 };
    const v = [_]f32{ 0.5, 0.25, 0.125, 0.0625 };
    var scratch: [1]f32 = undefined;
    attentionDecode(&out, &q, &k, &v, 1, 2, 2, 2, &scratch);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), out[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.125), out[2], 1e-5);
}

test "sampling: temperature 0 is greedy" {
    var seed: u32 = 1;
    var logits = [_]f32{ 0.1, 5.0, 0.2 };
    var cand: [3]Candidate = undefined;
    try std.testing.expectEqual(@as(u32, 1), sample(&logits, .{ .temperature = 0 }, &.{}, &seed, &cand));
}

test "sampling: top_k 1 picks the maximum" {
    var seed: u32 = 1;
    var logits = [_]f32{ 0.1, 5.0, 4.9 };
    var cand: [3]Candidate = undefined;
    for (0..16) |_| {
        try std.testing.expectEqual(@as(u32, 1), sample(&logits, .{ .temperature = 1.0, .top_k = 1 }, &.{}, &seed, &cand));
    }
}

test "sampling: top_p 0.1 keeps only the dominant token" {
    var seed: u32 = 7;
    var logits = [_]f32{ 20.0, 0.0, 0.0 };
    var cand: [3]Candidate = undefined;
    for (0..16) |_| {
        try std.testing.expectEqual(@as(u32, 0), sample(&logits, .{ .temperature = 1.0, .top_p = 0.1 }, &.{}, &seed, &cand));
    }
}

test "sampling: every drawn token has non-zero probability" {
    var seed: u32 = 12345;
    var logits = [_]f32{ 1.0, 1.0, 1.0, 1.0 };
    var cand: [4]Candidate = undefined;
    var counts: [4]u32 = @splat(0);
    for (0..4000) |_| {
        const t = sample(&logits, .{ .temperature = 1.0, .top_p = 0.9 }, &.{}, &seed, &cand);
        counts[t] += 1;
    }
    for (counts) |c| try std.testing.expect(c > 500); // roughly uniform
}

test "penalties push repeated tokens down" {
    var logits = [_]f32{ 5.0, 4.0 };
    applyPenalties(&logits, .{ .repetition_penalty = 2.0 }, &.{0});
    try std.testing.expectApproxEqAbs(@as(f32, 2.5), logits[0], 1e-6);
    var logits2 = [_]f32{ 5.0, 4.0 };
    applyPenalties(&logits2, .{ .presence_penalty = 1.0, .frequency_penalty = 0.5 }, &.{ 0, 0 });
    try std.testing.expectApproxEqAbs(@as(f32, 2.5), logits2[0], 1e-6);
}
