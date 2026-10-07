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

/// Simple temperature + top-k sampler. `rng_state` is a caller-owned LCG seed.
pub fn sampleTopK(values: []const f32, temperature: f32, top_k: usize, rng_state: *u32) u32 {
    if (temperature <= 0 or top_k == 1) return argmax(values);

    const n = values.len;
    var best_val: f32 = -std.math.inf(f32);
    for (values) |v| best_val = @max(best_val, v);

    // exp((v - max)/T) into a reusable accumulator via a two-pass top-k scan
    // (small vocab-sized scratch is avoided by scanning k times).
    const k = @min(top_k, n);
    var sum: f32 = 0;
    // Compute the normaliser over the top-k only: find the k-th largest value.
    var threshold: f32 = -std.math.inf(f32);
    if (k < n) {
        var cut: f32 = -std.math.inf(f32);
        for (0..k) |_| {
            var local: f32 = -std.math.inf(f32);
            for (values) |v| {
                if (v > cut and v < local) local = v;
                if (v > local) local = v;
            }
            // simple selection: track the largest value strictly below `cut`
            var next: f32 = -std.math.inf(f32);
            for (values) |v| {
                if (v < cut and v > next) next = v;
            }
            if (next == -std.math.inf(f32)) break;
            cut = next;
        }
        threshold = cut;
    }
    for (values) |v| {
        if (v >= threshold) sum += @exp((v - best_val) / temperature);
    }
    if (sum <= 0) return argmax(values);

    rng_state.* = rng_state.* *% 1664525 +% 1013904223;
    const r = @as(f32, @floatFromInt(rng_state.* >> 8)) / 16777216.0;
    var target = r * sum;
    for (values, 0..) |v, i| {
        if (v < threshold) continue;
        target -= @exp((v - best_val) / temperature);
        if (target <= 0) return @intCast(i);
    }
    return argmax(values);
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

test "sampleTopK with top_k 1 is argmax" {
    var seed: u32 = 1;
    const logits = [_]f32{ 0.1, 5.0, 0.2 };
    try std.testing.expectEqual(@as(u32, 1), sampleTopK(&logits, 1.0, 1, &seed));
}
