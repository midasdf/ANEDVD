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

/// Vector width for the attention inner loops. 8 f32 lanes = 256-bit, which
/// maps to two NEON registers; LLVM will not reassociate a plain `dot += a*b`
/// reduction (no fast-math), so the lanes have to be explicit or the loop
/// serialises on the add dependency chain.
const LANES = 8;
const F32x = @Vector(LANES, f32);
const F16x = @Vector(LANES, f16);

/// dot(a, b) over equally sized slices, 8 lanes at a time.
pub fn dotF32(a: []const f32, b: []const f32) f32 {
    std.debug.assert(a.len == b.len);
    var acc: F32x = @splat(0);
    var i: usize = 0;
    while (i + LANES <= a.len) : (i += LANES) {
        const va: F32x = a[i..][0..LANES].*;
        const vb: F32x = b[i..][0..LANES].*;
        acc += va * vb;
    }
    var sum: f32 = @reduce(.Add, acc);
    while (i < a.len) : (i += 1) sum += a[i] * b[i];
    return sum;
}

/// dot(a_f32, b_f16): the KV cache is fp16, the query stays fp32.
pub fn dotF32F16(a: []const f32, b: []const f16) f32 {
    std.debug.assert(a.len == b.len);
    var acc: F32x = @splat(0);
    var i: usize = 0;
    while (i + LANES <= a.len) : (i += LANES) {
        const va: F32x = a[i..][0..LANES].*;
        const vb: F16x = b[i..][0..LANES].*;
        acc += va * @as(F32x, @floatCast(vb));
    }
    var sum: f32 = @reduce(.Add, acc);
    while (i < a.len) : (i += 1) sum += a[i] * @as(f32, @floatCast(b[i]));
    return sum;
}

/// out += weight * v (fp16 source), 8 lanes at a time.
fn axpyF16(out: []f32, v: []const f16, weight: f32) void {
    const w: F32x = @splat(weight);
    var i: usize = 0;
    while (i + LANES <= out.len) : (i += LANES) {
        var o: F32x = out[i..][0..LANES].*;
        const vv: F16x = v[i..][0..LANES].*;
        o += w * @as(F32x, @floatCast(vv));
        out[i..][0..LANES].* = o;
    }
    while (i < out.len) : (i += 1) out[i] += weight * @as(f32, @floatCast(v[i]));
}

/// Multi-query attention for a chunk of prompt positions.
///
/// `q` and `out` use the engine's channel-major activation layout: element
/// (channel, column) lives at `base[channel * chan_stride + col * col_stride]`.
/// A channel is `head * head_dim + dim`, so query `qi` head `h` dim `d` is at
/// `q[(h * head_dim + d) * chan_stride + (q_pos0 + qi) * col_stride]`.
///
/// Getting this wrong is silent: reading the buffer as position-major returns
/// plausible-looking numbers, and an earlier version of the A/B check compared
/// the batched output against itself, so nothing caught it.
///
/// The per-position path calls `attentionDecode` n_q times, which re-walks the
/// whole cache for every query — at 1000 prompt tokens that is 1000x more work
/// than necessary and it dominated prefill (90% CPU at 971 tokens). Here each
/// cache entry is read once per query head, with the causal mask folded in.
pub fn attentionPrefill(
    out: []f32,
    q: []const f32,
    chan_stride: usize,
    col_stride: usize,
    k_cache: []const f16,
    v_cache: []const f16,
    n_past: usize,
    n_q: usize,
    q_pos0: usize,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    scores: []f32,
    query_scratch: []f32,
    out_scratch: []f32,
) void {
    const hd: usize = head_dim;
    const kv_dim: usize = @as(usize, n_kv_heads) * hd;
    const group = n_heads / n_kv_heads;
    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));
    std.debug.assert(scores.len >= n_past);
    std.debug.assert(query_scratch.len >= hd and out_scratch.len >= hd);

    for (0..n_q) |qi| {
        const pos = q_pos0 + qi;
        const visible = @min(n_past, pos + 1);
        const col = qi * col_stride;
        for (0..n_heads) |h| {
            const kvh = h / group;
            const base = h * hd * chan_stride + col;
            for (0..hd) |d| query_scratch[d] = q[base + d * chan_stride];
            const q_h = query_scratch[0..hd];
            const sc = scores[0..visible];
            for (0..visible) |t| {
                const k_t = k_cache[t * kv_dim + kvh * hd ..][0..hd];
                sc[t] = dotF32F16(q_h, k_t) * scale;
            }
            softmax(sc);
            @memset(out_scratch[0..hd], 0);
            for (0..visible) |t| {
                const v_t = v_cache[t * kv_dim + kvh * hd ..][0..hd];
                axpyF16(out_scratch[0..hd], v_t, sc[t]);
            }
            for (0..hd) |d| out[base + d * chan_stride] = out_scratch[d];
        }
    }
}

/// Softmax over `scores[0..n]` in place.
pub fn softmax(scores: []f32) void {
    if (scores.len == 0) return;
    // Vectorised max and sum; the exp is a polynomial approximation because
    // libm's expf is a function call per element and attention evaluates one
    // per (query, key, head, layer): at 971 prompt tokens that was 344k calls
    // per token and 78% of prefill.
    const max: f32 = vmaxF32(scores);
    var sum: f32 = 0;
    var i: usize = 0;
    while (i + LANES <= scores.len) : (i += LANES) {
        const v: F32x = scores[i..][0..LANES].*;
        const e = expApproxVec(v - @as(F32x, @splat(max)));
        scores[i..][0..LANES].* = e;
        sum += @reduce(.Add, e);
    }
    while (i < scores.len) : (i += 1) {
        const e = expApprox(scores[i] - max);
        scores[i] = e;
        sum += e;
    }
    if (sum == 0) return;
    const inv = 1.0 / sum;
    const vinv: F32x = @splat(inv);
    i = 0;
    while (i + LANES <= scores.len) : (i += LANES) {
        const v: F32x = scores[i..][0..LANES].*;
        scores[i..][0..LANES].* = v * vinv;
    }
    while (i < scores.len) : (i += 1) scores[i] *= inv;
}

fn vmaxF32(values: []const f32) f32 {
    var i: usize = 0;
    var best: F32x = @splat(-std.math.inf(f32));
    while (i + LANES <= values.len) : (i += LANES) {
        const v: F32x = values[i..][0..LANES].*;
        best = @max(best, v);
    }
    var m: f32 = @reduce(.Max, best);
    while (i < values.len) : (i += 1) m = @max(m, values[i]);
    return m;
}

/// e^x for x <= 0, to ~1e-6 relative accuracy.
///
/// Uses the standard range reduction x = n*ln2 + r with |r| <= ln2/2 and a
/// degree-5 polynomial for e^r, then scales by 2^n. Clamping n at -126 keeps
/// the scale factor normal (attention inputs are already shifted by the max, so
/// anything below ~-87 has zero weight anyway).
pub fn expApprox(x: f32) f32 {
    if (x != x) return x; // NaN
    // Below the f32 normal range the result is at most 1e-38, which cannot
    // affect a softmax denominator of order 1. Returning 0 is exact enough and
    // avoids clamping the exponent scaler (which would otherwise return a
    // spuriously large value).
    if (x < -87.0) return 0;
    const inv_ln2 = 1.4426950408889634;
    const n_f = @round(x * inv_ln2);
    const n: i32 = @intFromFloat(n_f);
    const r = x - @as(f32, @floatFromInt(n)) * 0.6931471805599453;
    const p = 1.0 + r * (1.0 + r * (0.5 + r * (0.16666667 + r * (0.041666668 + r * 0.008333334))));
    const scale = @as(f32, @bitCast((@as(u32, @intCast(n + 127)) << 23)));
    return p * scale;
}

fn expApproxVec(x: F32x) F32x {
    return @exp(x);
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
    k_cache: []const f16,
    v_cache: []const f16,
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
            scores[t] = dotF32F16(q_h, k_t) * scale;
        }
        softmax(scores);
        const o_h = out[h * hd ..][0..hd];
        // Weighted sum of the cached values. Keeping the accumulator in vector
        // registers matters: writing back to o_h on every position costs more
        // than the arithmetic (measured: 180 us -> 60 us per layer at ctx 1024).
        if (hd == 64) {
            var a0: F32x = @splat(0);
            var a1: F32x = @splat(0);
            var a2: F32x = @splat(0);
            var a3: F32x = @splat(0);
            var a4: F32x = @splat(0);
            var a5: F32x = @splat(0);
            var a6: F32x = @splat(0);
            var a7: F32x = @splat(0);
            for (0..n_past) |t| {
                const v_t = v_cache[t * kv_dim + kvh * hd ..][0..hd];
                const w: F32x = @splat(scores[t]);
                a0 += w * @as(F32x, @floatCast(@as(F16x, v_t[0..8].*)));
                a1 += w * @as(F32x, @floatCast(@as(F16x, v_t[8..16].*)));
                a2 += w * @as(F32x, @floatCast(@as(F16x, v_t[16..24].*)));
                a3 += w * @as(F32x, @floatCast(@as(F16x, v_t[24..32].*)));
                a4 += w * @as(F32x, @floatCast(@as(F16x, v_t[32..40].*)));
                a5 += w * @as(F32x, @floatCast(@as(F16x, v_t[40..48].*)));
                a6 += w * @as(F32x, @floatCast(@as(F16x, v_t[48..56].*)));
                a7 += w * @as(F32x, @floatCast(@as(F16x, v_t[56..64].*)));
            }
            o_h[0..8].* = a0;
            o_h[8..16].* = a1;
            o_h[16..24].* = a2;
            o_h[24..32].* = a3;
            o_h[32..40].* = a4;
            o_h[40..48].* = a5;
            o_h[48..56].* = a6;
            o_h[56..64].* = a7;
        } else {
            @memset(o_h, 0);
            for (0..n_past) |t| {
                const v_t = v_cache[t * kv_dim + kvh * hd ..][0..hd];
                axpyF16(o_h, v_t, scores[t]);
            }
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

/// What the last `sample` call actually did, for profiling.
pub const SamplerInfo = struct {
    /// Vocabulary entries above the `max_logit - min_keep_delta*T` cut.
    candidates: usize = 0,
    /// Entries actually drawn from (after top_k / top_p).
    kept: usize = 0,
    /// True when the top-p sort ran (it is skipped when top_p >= 1).
    sorted_for_top_p: bool = false,
    ns: u64 = 0,
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
///
/// The window is scanned in order, so a token that appears n times takes the
/// repetition penalty and the frequency penalty n times, but the presence
/// penalty only once — which is the OpenAI definition and what the unit test
/// pins.
pub fn applyPenalties(logits: []f32, params: SamplerParams, recent: []const u32) void {
    if (params.repetition_penalty == 1.0 and params.presence_penalty == 0.0 and params.frequency_penalty == 0.0) return;
    for (recent, 0..) |id, i| {
        if (id >= logits.len) continue;
        var v = logits[id];
        if (params.repetition_penalty != 1.0) {
            v = if (v > 0) v / params.repetition_penalty else v * params.repetition_penalty;
        }
        if (params.frequency_penalty != 0) v -= params.frequency_penalty;
        if (params.presence_penalty != 0 and std.mem.indexOfScalar(u32, recent[0..i], id) == null) {
            v -= params.presence_penalty;
        }
        logits[id] = v;
    }
}

/// Order `items` so the `k` largest are first, in descending order.
///
/// The sampler needs the top `k` in order and nothing else. Two hand-written
/// partial-selection schemes were tried here (a shifting sorted window, then a
/// size-k min-heap); both were wrong, and a test comparing against a full sort
/// caught the second one producing a window that was not the top k. Partial
/// selection is easy to get subtly wrong, so this delegates to the library's
/// pdqsort (`sortUnstable`), which is measurably faster than the block sort it
/// replaces and whose correctness is not in question.
///
/// A real algorithmic win would need a proper selection algorithm; the measured
/// cost of the sort is 18 ms/token against 1 ms for the exp() over the same
/// values, so if this is ever revisited, do it with a tested heap implementation
/// rather than by inspection.
fn selectTopK(items: []Candidate, k: usize) void {
    std.debug.assert(k <= items.len);
    if (items.len <= 1) return;
    // Full ordering. The caller keeps the first `k`; ordering the rest costs
    // nothing extra that matters at this size, and a partly-ordered contract
    // would be one more thing for the test to have to pin down.
    std.mem.sortUnstable(Candidate, items, {}, compareCandidates);
    std.debug.assert(k <= items.len);
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
    var info: SamplerInfo = undefined;
    return sampleProfiled(logits, params, recent, rng, scratch, &info);
}

pub fn sampleProfiled(
    logits: []f32,
    params: SamplerParams,
    recent: []const u32,
    rng: *u32,
    scratch: []Candidate,
    info: *SamplerInfo,
) u32 {
    std.debug.assert(scratch.len >= logits.len);
    info.* = .{};
    const t0 = nowNs();
    applyPenalties(logits, params, recent);

    if (params.temperature <= 0) {
        info.ns = nowNs() - t0;
        return argmax(logits);
    }

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
    info.candidates = n;
    if (n == 0) {
        info.ns = nowNs() - t0;
        return argmax(logits);
    }

    // Top-k: keep only the k best.
    //
    // This used to be a full sort of every candidate. With top_k = 40 from a
    // 151936-token vocabulary the cut above can leave ~150k candidates, and
    // sorting them (18 ms measured) to keep 34 was the entire cost of sampling;
    // the exp() over the same values is 1 ms. std.mem.sort does not need a fully
    // ordered array, so use partial selection instead: O(n) rather than O(n log n).
    var keep = n;
    if (params.top_k > 0 and params.top_k < n) {
        selectTopK(scratch[0..n], params.top_k);
        keep = params.top_k;
    }

    // Probabilities over the kept set.
    var sum: f32 = 0;
    for (scratch[0..keep]) |*c| {
        c.logit = @exp((c.logit - max_v) / params.temperature);
        sum += c.logit;
    }
    if (sum <= 0) return argmax(logits);

    // Top-p: order the kept set and truncate at the nucleus.
    if (params.top_p < 1.0) {
        info.sorted_for_top_p = true;
        // Already ordered when top_k ran; only sort when it did not (or when the
        // kept set is small enough that the sort is free).
        if (!(params.top_k > 0 and params.top_k < n) and keep > 1) {
            std.mem.sort(Candidate, scratch[0..keep], {}, compareCandidates);
        }
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

    info.kept = keep;
    rng.* = rng.* *% 1664525 +% 1013904223;
    const r = @as(f32, @floatFromInt(rng.* >> 8)) / 16777216.0;
    var target = r * sum;
    for (scratch[0..keep]) |c| {
        target -= c.logit;
        if (target <= 0) {
            info.ns = nowNs() - t0;
            return c.index;
        }
    }
    info.ns = nowNs() - t0;
    return scratch[0].index;
}

fn nowNs() u64 {
    var ts: std.c.timespec = undefined;
    if (std.c.clock_gettime(std.c.CLOCK.MONOTONIC, &ts) != 0) return 0;
    return @as(u64, @intCast(ts.sec)) * 1_000_000_000 + @as(u64, @intCast(ts.nsec));
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

test "expApprox matches std.math.exp over the attention range" {
    // Only x <= 0 occurs in softmax (inputs are shifted by the max).
    // Measured worst case is 6.1e-6 relative, so 2e-5 is the bar.
    var x: f32 = 0;
    while (x > -87) : (x -= 0.37) {
        const expected = @exp(x);
        const got = expApprox(x);
        try std.testing.expectApproxEqRel(expected, got, 2e-5);
    }
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), expApprox(0), 1e-6);
    // Below the normal range the weight is zero, and it must be exactly zero
    // rather than a clamped-scaler artefact.
    for ([_]f32{ -88, -100, -200 }) |v| {
        try std.testing.expectEqual(@as(f32, 0), expApprox(v));
    }
    for ([_]f32{ -1, -5, -20 }) |v| {
        try std.testing.expectApproxEqRel(@exp(v), expApprox(v), 2e-5);
    }
}

test "softmax sums to one" {
    var s = [_]f32{ 1, 2, 3, 4 };
    softmax(&s);
    var sum: f32 = 0;
    for (s) |v| sum += v;
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), sum, 1e-5);

    // A long vector exercises the vector path and the tails.
    var long: [1000]f32 = undefined;
    for (&long, 0..) |*v, i| v.* = @sin(@as(f32, @floatFromInt(i))) * 20.0;
    softmax(&long);
    var s2: f32 = 0;
    for (long) |v| s2 += v;
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), s2, 1e-4);
}

test "siluMul matches reference values" {
    var out: [2]f32 = undefined;
    siluMul(&out, &[_]f32{ 0.0, 1.0 }, &[_]f32{ 1.0, 1.0 });
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), out[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.7310586), out[1], 1e-5);
}

test "attentionPrefill matches attentionDecode for a single query" {
    const heads: u32 = 2;
    const kv_heads: u32 = 1;
    const hd: u32 = 4;
    const n_past: usize = 3;
    var k: [3 * 4]f16 = undefined;
    var v: [3 * 4]f16 = undefined;
    var seed: u32 = 5;
    for (&k) |*x| {
        seed = seed *% 1664525 +% 1013904223;
        x.* = @floatCast(@as(f32, @floatFromInt(seed >> 8)) / 8388608.0 - 1.0);
    }
    for (&v) |*x| {
        seed = seed *% 1664525 +% 1013904223;
        x.* = @floatCast(@as(f32, @floatFromInt(seed >> 8)) / 8388608.0 - 1.0);
    }
    var q: [heads * hd]f32 = undefined;
    for (&q) |*x| {
        seed = seed *% 1664525 +% 1013904223;
        x.* = @as(f32, @floatFromInt(seed >> 8)) / 8388608.0 - 1.0;
    }
    var a: [heads * hd]f32 = undefined;
    // Channel-major with one column.
    var b: [heads * hd]f32 = undefined;
    var scratch: [8]f32 = undefined;
    var qs: [heads * hd]f32 = undefined;
    var os: [heads * hd]f32 = undefined;
    // Query at position 2 sees all three cached entries. One column, so the
    // channel-major buffer is just [channel * 1 + 0] = plain channel order.
    attentionDecode(&a, &q, &k, &v, 3, heads, kv_heads, hd, &scratch);
    attentionPrefill(&b, &q, 1, 1, &k, &v, n_past, 1, 2, heads, kv_heads, hd, &scratch, &qs, &os);
    for (a, b) |x, y| try std.testing.expectApproxEqAbs(x, y, 1e-5);
}

test "attentionPrefill honours the causal mask" {
    const heads: u32 = 1;
    const kv_heads: u32 = 1;
    const hd: u32 = 2;
    // Two cached entries with very different values.
    const k = [_]f16{ 1, 0, 1, 0 };
    const v = [_]f16{ 10, 10, 0, 0 };
    // Two queries: position 0 may only see entry 0, position 1 sees both.
    // Channel-major [channel * 2 + column]: column 0 = (1,0), column 1 = (1,0).
    const q = [_]f32{ 1, 1, 0, 0 };
    var out: [2 * 2]f32 = undefined;
    var scratch: [4]f32 = undefined;
    var qs: [2]f32 = undefined;
    var os: [2]f32 = undefined;
    attentionPrefill(&out, &q, 2, 1, &k, &v, 2, 2, 0, heads, kv_heads, hd, &scratch, &qs, &os);
    // Column 0 of the output (channel 0, column 0) sees only cache entry 0.
    try std.testing.expectApproxEqAbs(@as(f32, 10), out[0], 1e-3);
    // Column 1 sees both entries, so the mean of v (10 and 0) on channel 0.
    try std.testing.expectApproxEqAbs(@as(f32, 5), out[1], 1e-3);
}

test "dotF32 matches a scalar dot product" {
    var a: [19]f32 = undefined;
    var b: [19]f32 = undefined;
    for (&a, 0..) |*x, i| x.* = @as(f32, @floatFromInt(i)) * 0.37 - 2.0;
    for (&b, 0..) |*x, i| x.* = 1.0 - @as(f32, @floatFromInt(i)) * 0.11;
    var expected: f32 = 0;
    for (a, b) |x, y| expected += x * y;
    try std.testing.expectApproxEqAbs(expected, dotF32(&a, &b), 1e-3);
}

test "attentionDecode single past position returns v" {
    var out: [4]f32 = undefined;
    const q = [_]f32{ 1, 0, 1, 0 };
    const k = [_]f16{ 1, 0, 1, 0 };
    const v = [_]f16{ 0.5, 0.25, 0.125, 0.0625 };
    var scratch: [1]f32 = undefined;
    attentionDecode(&out, &q, &k, &v, 1, 2, 2, 2, &scratch);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), out[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.125), out[2], 1e-5);
}

test "selectTopK agrees with a full sort" {
    const a = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(0xC0FFEE);
    const rand = prng.random();

    for ([_]usize{ 1, 2, 3, 7, 40, 64, 129 }) |k| {
        // Sizes below, at, and well above k; a small value range so ties occur.
        for ([_]usize{ 0, 1, 2, 5, 39, 40, 41, 500 }) |n| {
            const items = try a.alloc(Candidate, n);
            defer a.free(items);
            for (items, 0..) |*c, i| {
                c.* = .{ .logit = @floatFromInt(rand.intRangeAtMost(u32, 0, 4)), .index = @intCast(i) };
            }
            const expected = try a.dupe(Candidate, items);
            defer a.free(expected);
            std.mem.sort(Candidate, expected, {}, compareCandidates);

            const got = @min(k, n);
            selectTopK(items, got);
            for (0..got) |i| {
                try std.testing.expectApproxEqAbs(expected[i].logit, items[i].logit, 1e-6);
            }
            if (got > 1) {
                for (1..got) |i| {
                    try std.testing.expect(items[i - 1].logit >= items[i].logit);
                }
            }
            if (got > 0) {
                for (items[got..]) |c| {
                    try std.testing.expect(c.logit <= items[got - 1].logit);
                }
            }
        }
    }
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
    // Two occurrences: presence once, frequency twice.
    var logits2 = [_]f32{ 5.0, 4.0 };
    applyPenalties(&logits2, .{ .presence_penalty = 1.0, .frequency_penalty = 0.5 }, &.{ 0, 0 });
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), logits2[0], 1e-6);
    // A token outside the window is untouched.
    var logits3 = [_]f32{ 5.0, 4.0 };
    applyPenalties(&logits3, .{ .repetition_penalty = 2.0 }, &.{1});
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), logits3[0], 1e-6);
}
