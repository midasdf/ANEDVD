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

/// Architectures this loader will not warn about: the ones exercised on a real
/// model here (llama, qwen2, qwen3, qwen2moe, gemma2, and the smollm family),
/// plus the ones that share those GGUF layouts exactly — the same
/// `*.block_count` config keys and `blk.N.*` tensor names.
///
/// Being on this list is not a statement that the architecture was run. Mistral
/// and SmolLM3 are here on layout alone; Mistral's sliding window turned out to be
/// ignored entirely until it was fixed from the config, which is exactly the kind
/// of thing only running a model catches. Anything not listed is accepted with a
/// warning, because a wrong architecture guess produces fluent nonsense rather
/// than an error.
pub const known_architectures = [_][]const u8{
    "llama",    "qwen2",   "qwen3",    "mistral",   "smollm",
    "smollm2",  "smollm3", "qwen2moe", "qwen3moe",  "granite",
    "stablelm", "olmo",    "olmo2",    "phi3",      "gemma",
    "gemma2",   "gemma3",  "gptneox",  "internlm2",
};

pub fn isKnownArchitecture(arch: []const u8) bool {
    for (known_architectures) |a| if (std.mem.eql(u8, arch, a)) return true;
    return false;
}

/// Architectures whose GGUF this loader recognises but cannot load correctly yet. These are
/// refused rather than warned about, because "treat it like llama" is guaranteed wrong for
/// them: the output is fluent nonsense, which costs a user hours before they conclude the
/// model is broken. Each entry names what is missing so the work is scoped, not guessed.
///
/// The Gemma 4 metadata below is not inferred — it is read from the official
/// `google/gemma-4-E2B-it-qat-q4_0-gguf` header:
///
///   gemma4.block_count = 35
///   gemma4.attention.key_length = 512        gemma4.attention.key_length_swa = 256
///   gemma4.attention.value_length = 512      gemma4.attention.value_length_swa = 256
///   gemma4.attention.sliding_window = 512
///   gemma4.attention.sliding_window_pattern = [true x4, false, ...]   (per layer, not a modulo)
///   gemma4.attention.shared_kv_layers = 20
///   gemma4.embedding_length_per_layer_input = 256
///   gemma4.feed_forward_length = [6144 x15, 12288 x20, ...]           (per layer)
///   gemma4.rope.freq_base = 1e6              gemma4.rope.freq_base_swa = 1e4
///   tokenizer.ggml.tokens = 262144 entries
pub const unsupported_architectures = [_]struct { name: []const u8, why: []const u8 }{
    .{
        .name = "gemma4",
        // Steps 1-3 are done (per-layer sliding list, per-layer FFN widths, per-layer head
        // dims); these two remain.
        .why = "KV sharing across 20 layers and per-layer input embeddings (PLE)",
    },
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

    // Refuse before any key is looked up: these architectures are missing dimensions the
    // loader itself assumes, so the lookup would fail on a missing key with a message about
    // the key rather than about the architecture.
    for (unsupported_architectures) |u| {
        if (std.mem.eql(u8, arch, u.name)) {
            sys.eprint("error: \"{s}\" is recognised but not implemented yet.\n", .{arch});
            sys.eprint("  It needs {s}.\n", .{u.why});
            sys.eprint("  Loading it as llama would produce fluent nonsense, so this is refused.\n", .{});
            return Error.UnsupportedArchitecture;
        }
    }

    var cfg = model.Config{ .arch = arch };
    // Name the missing key. A bare `MissingTensor` from a config read sends the reader
    // looking for a broken tensor, when the actual problem is that this file does not
    // carry the metadata an architecture of this name needs — which happens when the
    // architecture string is wrong or the file is for a different runtime.
    cfg.hidden = g.getU32(key(&buf, arch, "embedding_length")) orelse {
        sys.eprint("gguf: no {s} in this file (arch \"{s}\").\n", .{ key(&buf, arch, "embedding_length"), arch });
        return Error.MissingTensor;
    };
    cfg.layers = g.getU32(key(&buf, arch, "block_count")) orelse {
        sys.eprint("gguf: no {s} in this file.\n", .{key(&buf, arch, "block_count")});
        return Error.MissingTensor;
    };
    cfg.heads = g.getU32(key(&buf, arch, "attention.head_count")) orelse {
        sys.eprint("gguf: no {s} in this file.\n", .{key(&buf, arch, "attention.head_count")});
        return Error.MissingTensor;
    };
    cfg.kv_heads = g.getU32(key(&buf, arch, "attention.head_count_kv")) orelse cfg.heads;
    // `feed_forward_length` is a scalar for most architectures and an ARRAY for Gemma 4, whose
    // layers are not all the same width (E2B is [6144 x15, 12288 x20, ...]). The scalar read
    // returned null on an array, so the load failed on a missing key; the max is what scratch
    // buffers must be sized for, and each layer's own width comes from its tensor below.
    cfg.inter = interFromMetadata(g, &buf, arch) orelse {
        sys.eprint("gguf: no {s} in this file.\n", .{key(&buf, arch, "feed_forward_length")});
        return Error.MissingTensor;
    };
    cfg.head_dim = g.getU32(key(&buf, arch, "attention.key_length")) orelse (cfg.hidden / cfg.heads);
    cfg.eps = g.getF32(key(&buf, arch, "attention.layer_norm_rms_epsilon")) orelse 1e-5;
    cfg.rope_theta = g.getF32(key(&buf, arch, "rope.freq_base")) orelse 10000.0;
    cfg.vocab = g.getU32(key(&buf, arch, "vocab_size")) orelse blk: {
        // No vocab_size and no token list: report both, since either could be the cause.
        const toks = g.getStringArray("tokenizer.ggml.tokens") orelse {
            sys.eprint("gguf: no {s} and no tokenizer.ggml.tokens to fall back on.\n", .{key(&buf, arch, "vocab_size")});
            return Error.MissingTensor;
        };
        break :blk @intCast(toks.len);
    };
    cfg.tie_embeddings = g.tensor("output.weight") == null;
    cfg.rope_adjacent = ropeIsAdjacent(arch);
    // Gemma: `(1 + w)` RMSNorm and a sqrt(hidden) embedding scale, both confirmed
    // against the reference implementation rather than guessed.
    if (std.mem.startsWith(u8, arch, "gemma")) {
        // The GGUF converter has ALREADY applied Gemma's `(1 + w)` to the norm tensors.
        // Measured: blk.0.attn_norm.weight has mean 1.1927, which is the *effective*
        // factor (1 + w) for a converged model, where the raw HF parameter would have
        // mean ~0.19. Applying the offset again doubles it and the model emits "."
        // instead of an answer, so it is NOT applied for GGUF.
        cfg.norm_unit_offset = false;
        // The GGUF stores UNSCALED embedding weights (measured: mean|w| ~ 1e-5 on
        // gemma-2-2b, where the HF standard deviation is ~0.01 and a baked-in
        // sqrt(2304) would give ~0.5), so the scale must be applied here.
        cfg.embed_scale = @sqrt(@as(f32, @floatFromInt(cfg.hidden)));
        // Gemma 2 adds logit soft-capping and alternating sliding-window attention.
        // Both are read from the file, not assumed.
        cfg.use_gelu = true; // gemma 2: hidden_activation = "gelu_pytorch_tanh"
        // gemma2 uses 1/sqrt(n_embd/n_head) (llama.cpp: f_attention_scale), which is not
        // 1/sqrt(head_dim) when key_length differs from n_embd/n_head.
        if (std.mem.eql(u8, arch, "gemma2") and cfg.heads > 0) {
            cfg.attn_scale = 1.0 / @sqrt(@as(f32, @floatFromInt(cfg.hidden)) / @as(f32, @floatFromInt(cfg.heads)));
        }
        cfg.attn_logit_softcap = g.getF32(key(&buf, arch, "attn_logit_softcapping")) orelse 0;
        cfg.final_logit_softcap = g.getF32(key(&buf, arch, "final_logit_softcapping")) orelse 0;
        cfg.sliding_window = g.getU32(key(&buf, arch, "attention.sliding_window")) orelse 0;
        // gemma2 and later sandwich each sublayer between two more norms.
        if (g.tensor(std.fmt.bufPrint(&buf, "blk.0.post_attention_norm.weight", .{}) catch unreachable) != null) {
            cfg.sandwich_norms = true;
        }
    }

    // Mistral (and any other llama-family model that declares one) uses the sliding window
    // on EVERY layer, unlike Gemma 2's alternating pattern. Ignoring it made those models
    // attend globally over the whole context.
    if (!std.mem.startsWith(u8, arch, "gemma")) {
        if (g.getU32(key(&buf, arch, "attention.sliding_window"))) |w| {
            cfg.sliding_window = w;
            cfg.swa_all = true;
        }
    }
    // Gemma 4 carries two head dimensions: `key_length` for the global layers and
    // `key_length_swa` for the sliding ones (512 and 256). Config's `head_dim` is the
    // global/default one, so the plain `key_length` read above already filled it.
    if (g.getU32(key(&buf, arch, "attention.key_length_swa"))) |swa_hd| {
        if (swa_hd > 0 and swa_hd != cfg.head_dim) cfg.head_dim_swa = swa_hd;
    }

    // The pattern key overrides the architecture default in both directions: Gemma 3 needs 6
    // where Gemma 2 uses 2, and a non-Gemma model may declare one too. Two shapes exist: a
    // repeating length (Gemma 2/3) and an explicit per-layer list (Gemma 4, whose
    // `sliding_window_pattern` is an array of bools matching HF's `layer_types` one for one —
    // `true` means that layer slides).
    if (g.getValue(key(&buf, arch, "attention.sliding_window_pattern"))) |v| {
        if (v.asBoolArray()) |flags| {
            cfg.swa_explicit = true;
            cfg.swa_all = false;
            if (flags.len > 128) {
                sys.eprint("warning: {d} sliding-window flags, only the first 128 are used.\n", .{flags.len});
            }
            for (flags, 0..) |slides, li| {
                if (!slides or li >= 128) continue;
                cfg.swa_layers[li / 64] |= @as(u64, 1) << @intCast(li % 64);
            }
        } else if (v.asU32()) |pat| {
            if (pat > 0) {
                cfg.swa_pattern = pat;
                cfg.swa_all = false;
            }
        }
    }

    // MoE metadata (llama.cpp: <arch>.expert_count / .expert_used_count /
    // .expert_shared_count / .expert_feed_forward_length).
    cfg.num_experts = g.getU32(key(&buf, arch, "expert_count")) orelse 0;
    cfg.experts_per_tok = g.getU32(key(&buf, arch, "expert_used_count")) orelse 0;
    if (cfg.num_experts > 0) {
        if (cfg.experts_per_tok == 0) {
            sys.eprint("warning: {s} declares {d} experts but no expert_used_count; assuming 2.\n", .{ arch, cfg.num_experts });
            cfg.experts_per_tok = 2;
        }
        // The tensor shapes are authoritative, not the metadata. Qwen1.5-MoE ships
        // no expert_feed_forward_length at all, and its feed_forward_length (5632)
        // is the SHARED expert's width while the routed experts are 1408 — so
        // defaulting one from the other silently sizes every expert wrong, or
        // skips the shared expert entirely. Read the widths off the tensors the
        // same way llama.cpp does.
        const gate_exps = g.tensor(std.fmt.bufPrint(&buf, "blk.0.ffn_gate_exps.weight", .{}) catch unreachable);
        if (gate_exps) |t| {
            if (t.dims.len == 3) {
                // {hidden, moe_inter, num_experts} with the expert axis last.
                cfg.moe_inter = @intCast(t.dims[1]);
                if (t.dims[2] != cfg.num_experts) {
                    sys.eprint("warning: expert_count is {d} but ffn_gate_exps has {d} experts; trusting the tensor.\n", .{ cfg.num_experts, t.dims[2] });
                    cfg.num_experts = @intCast(t.dims[2]);
                }
            }
        } else {
            cfg.moe_inter = g.getU32(key(&buf, arch, "expert_feed_forward_length")) orelse cfg.inter;
        }
        // The shared expert's width comes from its tensor when the metadata omits it
        // (Qwen1.5-MoE does). A 0 here would silently drop the shared expert from
        // the forward pass, which is a wrong answer rather than a slow one.
        cfg.shared_inter = g.getU32(key(&buf, arch, "expert_shared_feed_forward_length")) orelse blk: {
            const shexp = g.tensor(std.fmt.bufPrint(&buf, "blk.0.ffn_gate_shexp.weight", .{}) catch unreachable) orelse break :blk 0;
            // {hidden, inter}: ggml's ne[0] is hidden and varies fastest, so the
            // shared width is dims[1]. Reading dims[0] yields the hidden size, which
            // is plausible enough to pass unnoticed.
            break :blk if (shexp.dims.len >= 2) @as(u32, @intCast(shexp.dims[1])) else cfg.inter;
        };
    }
    try cfg.validate();

    // A wrong guess here is silent: the model still runs and produces fluent
    // text. Say so once, loudly enough to be seen in a log.
    if (!isKnownArchitecture(arch)) {
        sys.eprint("warning: architecture \"{s}\" is not one this loader has been tested with.\n", .{arch});
        sys.eprint("  Treating it like llama: blk.N.* tensor names, RoPE {s}.\n", .{
            if (cfg.rope_adjacent) "adjacent-pair" else "half-split",
        });
        sys.eprint("  If the output is fluent nonsense, force the other convention with --rope-hf / --rope-adjacent.\n", .{});
    }
    return cfg;
}

/// Load a Linear weight as fp16 in ANE conv layout [out][in].
/// `feed_forward_length`, scalar or array, reduced to the widest layer.
fn interFromMetadata(g: *const gguf.Gguf, buf: []u8, arch: []const u8) ?u32 {
    const v = g.getValue(key(buf, arch, "feed_forward_length")) orelse return null;
    if (v.asU32()) |n| return n;
    const arr = switch (v) {
        .array => |a| a,
        else => return null,
    };
    var max: u32 = 0;
    for (0..arr.data.len()) |i| {
        const n = switch (arr.data) {
            .u32 => |d| d[i],
            .i32 => |d| if (d[i] >= 0) @as(u32, @intCast(d[i])) else continue,
            .u64 => |d| std.math.cast(u32, d[i]) orelse continue,
            else => continue,
        };
        if (n > max) max = n;
    }
    return if (max == 0) null else max;
}

/// The `out_dim` a 2-D tensor actually has, given its `in_dim`. Shapes are authoritative: a
/// model may declare one `feed_forward_length` and use several.
fn tensorOutDim(g: *const gguf.Gguf, name: []const u8, in_dim: u32) ?u32 {
    const t = g.tensor(name) orelse return null;
    if (t.dims.len < 2 or t.dims[0] != in_dim) return null;
    return std.math.cast(u32, t.dims[1]);
}

fn loadLinear(allocator: std.mem.Allocator, g: *const gguf.Gguf, name: []const u8, in_dim: u32, out_dim: u32) ![]f16 {
    const t = g.tensor(name) orelse {
        sys.eprint("gguf: no tensor {s}\n", .{name});
        return Error.MissingTensor;
    };
    const want: u64 = @as(u64, in_dim) * out_dim;
    if (t.elemCount() != want) {
        sys.eprint("gguf: {s} has {d} elements, expected {d} ({d}x{d}), dims {any}\n", .{
            name, t.elemCount(), want, in_dim, out_dim, t.dims,
        });
        return Error.DimensionMismatch;
    }

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
///
/// The view aliases the mapping, so the caller's `Gguf` must outlive the
/// `Runtime` that holds it. That is why `model_open.Loaded` keeps the `Gguf`
/// alive next to the weights, and why `loadWeights` (used for the CPU
/// reference, which outlives the file) copies through `readF16` instead.
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

/// Load an RMSNorm, applying Gemma's `(1 + w)` convention when the model uses it.
fn loadNormFor(allocator: std.mem.Allocator, g: *const gguf.Gguf, name: []const u8, n: u32, cfg: model.Config) ![]f32 {
    const v = try loadNorm(allocator, g, name, n);
    if (cfg.norm_unit_offset) {
        for (v) |*x| x.* += 1.0;
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
    mw.final_norm = try loadNormFor(allocator, g, "output_norm.weight", cfg.hidden, cfg);
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
        lw.attn_norm = try loadNormFor(allocator, g, std.fmt.bufPrint(&buf, "blk.{d}.attn_norm.weight", .{li}) catch unreachable, cfg.hidden, cfg);
        lw.ffn_norm = try loadNormFor(allocator, g, std.fmt.bufPrint(&buf, "blk.{d}.ffn_norm.weight", .{li}) catch unreachable, cfg.hidden, cfg);
        if (cfg.sandwich_norms) {
            lw.post_attn_norm = try loadNormFor(allocator, g, std.fmt.bufPrint(&buf, "blk.{d}.post_attention_norm.weight", .{li}) catch unreachable, cfg.hidden, cfg);
            lw.post_ffw_norm = try loadNormFor(allocator, g, std.fmt.bufPrint(&buf, "blk.{d}.post_ffw_norm.weight", .{li}) catch unreachable, cfg.hidden, cfg);
        }

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
            lw.q_norm = try loadNormFor(allocator, g, std.fmt.bufPrint(&buf, "blk.{d}.attn_q_norm.weight", .{li}) catch unreachable, cfg.head_dim, cfg);
            lw.k_norm = try loadNormFor(allocator, g, std.fmt.bufPrint(&buf, "blk.{d}.attn_k_norm.weight", .{li}) catch unreachable, cfg.head_dim, cfg);
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
        if (cfg.layerIsSparse(li)) {
            // Sparse layer: the dense ffn_gate/up/down are the shared expert's.
            lw.moe = try loadMoeLayer(allocator, g, cfg, li);
        } else {
            const gname = std.fmt.bufPrint(&buf, "blk.{d}.ffn_gate.weight", .{li}) catch unreachable;
            const inter_l = tensorOutDim(g, gname, cfg.hidden) orelse cfg.inter;
            lw.gate = try loadLinear(allocator, g, gname, cfg.hidden, inter_l);
            lw.up = try loadLinear(allocator, g, std.fmt.bufPrint(&buf, "blk.{d}.ffn_up.weight", .{li}) catch unreachable, cfg.hidden, inter_l);
            lw.down = try loadLinear(allocator, g, std.fmt.bufPrint(&buf, "blk.{d}.ffn_down.weight", .{li}) catch unreachable, inter_l, cfg.hidden);
        }
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
    rt.final_norm = try loadNormFor(allocator, g, "output_norm.weight", cfg.hidden, cfg);
    rt.norms = try allocator.alloc(model.Norm, cfg.layers);
    @memset(rt.norms, .{});
    var buf: [128]u8 = undefined;
    for (rt.norms, 0..) |*n, i| {
        const li: u32 = @intCast(i);
        n.attn = try loadNormFor(allocator, g, std.fmt.bufPrint(&buf, "blk.{d}.attn_norm.weight", .{li}) catch unreachable, cfg.hidden, cfg);
        if (cfg.sandwich_norms) {
            n.post_attn = try loadNormFor(allocator, g, std.fmt.bufPrint(&buf, "blk.{d}.post_attention_norm.weight", .{li}) catch unreachable, cfg.hidden, cfg);
            n.post_ffw = try loadNormFor(allocator, g, std.fmt.bufPrint(&buf, "blk.{d}.post_ffw_norm.weight", .{li}) catch unreachable, cfg.hidden, cfg);
        }
        n.ffn = try loadNormFor(allocator, g, std.fmt.bufPrint(&buf, "blk.{d}.ffn_norm.weight", .{li}) catch unreachable, cfg.hidden, cfg);
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
            n.q_norm = try loadNormFor(allocator, g, std.fmt.bufPrint(&buf, "blk.{d}.attn_q_norm.weight", .{li}) catch unreachable, cfg.head_dim, cfg);
            n.k_norm = try loadNormFor(allocator, g, std.fmt.bufPrint(&buf, "blk.{d}.attn_k_norm.weight", .{li}) catch unreachable, cfg.head_dim, cfg);
        }
    }
    return rt;
}

/// One layer's compile-time-only matrices.
pub fn loadLayer(allocator: std.mem.Allocator, g: *const gguf.Gguf, cfg: model.Config, index: u32) !model.Matrices {
    var m = model.Matrices{};
    errdefer m.deinit(allocator);
    var buf: [192]u8 = undefined;
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
    if (cfg.layerIsSparse(index)) {
        // Routed experts run on the CPU; the shared expert takes the dense slots so
        // the ANE's existing FFN kernel serves it unchanged.
        m.moe = try loadMoeLayerStreaming(allocator, g, cfg, index);
        if (cfg.shared_inter > 0) {
            m.gate = try loadLinear(allocator, g, std.fmt.bufPrint(&buf, "blk.{d}.ffn_gate_shexp.weight", .{index}) catch unreachable, cfg.hidden, cfg.shared_inter);
            m.up = try loadLinear(allocator, g, std.fmt.bufPrint(&buf, "blk.{d}.ffn_up_shexp.weight", .{index}) catch unreachable, cfg.hidden, cfg.shared_inter);
            m.down = try loadLinear(allocator, g, std.fmt.bufPrint(&buf, "blk.{d}.ffn_down_shexp.weight", .{index}) catch unreachable, cfg.shared_inter, cfg.hidden);
        }
    } else {
        const gname = std.fmt.bufPrint(&buf, "blk.{d}.ffn_gate.weight", .{index}) catch unreachable;
        const inter_l = tensorOutDim(g, gname, cfg.hidden) orelse cfg.inter;
        m.gate = try loadLinear(allocator, g, gname, cfg.hidden, inter_l);
        m.up = try loadLinear(allocator, g, std.fmt.bufPrint(&buf, "blk.{d}.ffn_up.weight", .{index}) catch unreachable, cfg.hidden, inter_l);
        m.down = try loadLinear(allocator, g, std.fmt.bufPrint(&buf, "blk.{d}.ffn_down.weight", .{index}) catch unreachable, inter_l, cfg.hidden);
    }
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

/// Load the sparse block of a MoE layer from GGUF.
///
/// The expert tensors are 3-D (`blk.N.ffn_gate_exps` etc.) with the expert index
/// as the LAST declared dimension, which in ggml's layout means experts are the
/// slowest-varying axis and each expert's slice is contiguous. That is convenient
/// (one read, no gather) but the two orders are not the same for every tensor, so
/// each is checked against the expert's expected [out][in] shape rather than
/// assumed.
/// Load a MoE layer WITHOUT materialising its experts.
///
/// Only the router and the shared expert are copied; the routed experts stay in the
/// mapping and are dequantised one at a time as the router asks for them. This is
/// what makes a MoE model runnable at all here: Qwen1.5-MoE's experts are 25 GB as
/// fp16 against this machine's 8 GB, and materialising them swapped the machine
/// (997 MB used, 71M pageins) until decode was ~30x slower than the arithmetic
/// justifies. The mapped file's untouched pages are clean, so the OS can evict them
/// under pressure instead of paging out anonymous memory.
pub fn loadMoeLayerStreaming(
    allocator: std.mem.Allocator,
    g: *const gguf.Gguf,
    cfg: model.Config,
    index: u32,
) !model.MoeWeights {
    var buf: [128]u8 = undefined;
    var moe = model.MoeWeights{
        .num_experts = cfg.num_experts,
        .inter = cfg.moe_inter,
        .shared_inter = cfg.shared_inter,
        .hidden_dim = cfg.hidden,
    };
    errdefer moe.deinit(allocator);

    const gate_elems = @as(usize, cfg.moe_inter) * cfg.hidden;
    const down_elems = @as(usize, cfg.hidden) * cfg.moe_inter;
    // Three slots so gate, up and down can be live at once (moeExpertAccum takes
    // all three), each big enough for the larger projection.
    const slot = @max(gate_elems, down_elems);
    moe.expert_slot = slot;
    // No fp16 scratch on a streaming layer: `loadExpertF32` is the only accessor the
    // streaming paths use, so the fp16 buffer was 17.3 MB per layer allocated and
    // never read (415 MB across 24 layers). It is still needed for the eager loader,
    // which holds its experts as fp16 and uses `loadExpert`.
    // Three slots, one per projection, because the three are live simultaneously.
    moe.expert_scratch_f32_alt = try allocator.alloc(f32, slot * 3);

    moe.router = try loadLinear(allocator, g, std.fmt.bufPrint(&buf, "blk.{d}.ffn_gate_inp.weight", .{index}) catch unreachable, cfg.hidden, cfg.num_experts);

    // The tensor names must outlive this call, so they are duplicated and owned by
    // MoeWeights. Assign them to `moe.source` FIRST and let the single
    // `errdefer moe.deinit` above handle cleanup: giving each name its own errdefer as
    // well, and then assigning them, frees them twice when a later `try` fails. That
    // was a real double free at layer 1.
    moe.source = .{
        .gguf = g,
        .gate_name = try allocator.dupe(u8, std.fmt.bufPrint(&buf, "blk.{d}.ffn_gate_exps.weight", .{index}) catch unreachable),
        .up_name = try allocator.dupe(u8, std.fmt.bufPrint(&buf, "blk.{d}.ffn_up_exps.weight", .{index}) catch unreachable),
        .down_name = try allocator.dupe(u8, std.fmt.bufPrint(&buf, "blk.{d}.ffn_down_exps.weight", .{index}) catch unreachable),
        .gate_elems = gate_elems,
        .down_elems = down_elems,
    };

    if (cfg.shared_inter > 0) {
        moe.shared_gate = try loadLinear(allocator, g, std.fmt.bufPrint(&buf, "blk.{d}.ffn_gate_shexp.weight", .{index}) catch unreachable, cfg.hidden, cfg.shared_inter);
        moe.shared_up = try loadLinear(allocator, g, std.fmt.bufPrint(&buf, "blk.{d}.ffn_up_shexp.weight", .{index}) catch unreachable, cfg.hidden, cfg.shared_inter);
        moe.shared_down = try loadLinear(allocator, g, std.fmt.bufPrint(&buf, "blk.{d}.ffn_down_shexp.weight", .{index}) catch unreachable, cfg.shared_inter, cfg.hidden);
        // A 1-D vector of length hidden, not a [1][hidden] matrix: ggml stores the
        // shared-expert gate as {n_embd}, which loadLinear's rank-2 checks reject.
        {
            const gate_f32 = try g.readF32(allocator, std.fmt.bufPrint(&buf, "blk.{d}.ffn_gate_inp_shexp.weight", .{index}) catch unreachable);
            defer allocator.free(gate_f32);
            if (gate_f32.len != cfg.hidden) return Error.DimensionMismatch;
            moe.shared_gate_lin = try allocator.alloc(f16, cfg.hidden);
            for (gate_f32, 0..) |v, i| moe.shared_gate_lin[i] = @floatCast(v);
        }
    }
    return moe;
}

pub fn loadMoeLayer(allocator: std.mem.Allocator, g: *const gguf.Gguf, cfg: model.Config, index: u32) !model.MoeWeights {
    var buf: [128]u8 = undefined;
    var moe = model.MoeWeights{
        .num_experts = cfg.num_experts,
        .inter = cfg.moe_inter,
        .shared_inter = cfg.shared_inter,
        .hidden_dim = cfg.hidden,
    };
    errdefer moe.deinit(allocator);

    const per_gate = @as(usize, cfg.moe_inter) * cfg.hidden;
    // Three slots, one per projection: a zero stride made gate, up and down share
    // slot 0, which silently multiplied the same tensor three times.
    moe.expert_slot = per_gate;
    moe.expert_scratch = try allocator.alloc(f16, per_gate * 3);
    moe.router = try loadLinear(allocator, g, std.fmt.bufPrint(&buf, "blk.{d}.ffn_gate_inp.weight", .{index}) catch unreachable, cfg.hidden, cfg.num_experts);
    moe.gate = try allocator.alloc(f16, per_gate * cfg.num_experts);
    moe.up = try allocator.alloc(f16, per_gate * cfg.num_experts);
    moe.down = try allocator.alloc(f16, per_gate * cfg.num_experts);

    try loadExperts(allocator, g, std.fmt.bufPrint(&buf, "blk.{d}.ffn_gate_exps.weight", .{index}) catch unreachable, moe.gate, cfg.num_experts, cfg.moe_inter, cfg.hidden);
    try loadExperts(allocator, g, std.fmt.bufPrint(&buf, "blk.{d}.ffn_up_exps.weight", .{index}) catch unreachable, moe.up, cfg.num_experts, cfg.moe_inter, cfg.hidden);
    try loadExperts(allocator, g, std.fmt.bufPrint(&buf, "blk.{d}.ffn_down_exps.weight", .{index}) catch unreachable, moe.down, cfg.num_experts, cfg.hidden, cfg.moe_inter);

    if (cfg.shared_inter > 0) {
        moe.shared_gate = try loadLinear(allocator, g, std.fmt.bufPrint(&buf, "blk.{d}.ffn_gate_shexp.weight", .{index}) catch unreachable, cfg.hidden, cfg.shared_inter);
        moe.shared_up = try loadLinear(allocator, g, std.fmt.bufPrint(&buf, "blk.{d}.ffn_up_shexp.weight", .{index}) catch unreachable, cfg.hidden, cfg.shared_inter);
        moe.shared_down = try loadLinear(allocator, g, std.fmt.bufPrint(&buf, "blk.{d}.ffn_down_shexp.weight", .{index}) catch unreachable, cfg.shared_inter, cfg.hidden);
        // A 1-D vector of length hidden, not a [1][hidden] matrix: ggml stores the
        // shared-expert gate as {n_embd} and loadLinear's rank-2 checks reject it.
        {
            const gate_f32 = try g.readF32(allocator, std.fmt.bufPrint(&buf, "blk.{d}.ffn_gate_inp_shexp.weight", .{index}) catch unreachable);
            defer allocator.free(gate_f32);
            if (gate_f32.len != cfg.hidden) return Error.DimensionMismatch;
            moe.shared_gate_lin = try allocator.alloc(f16, cfg.hidden);
            for (gate_f32, 0..) |v, i| moe.shared_gate_lin[i] = @floatCast(v);
        }
    }
    return moe;
}

/// Split a stacked expert tensor into `dst`, which is [num_experts][out][in].
///
/// `out`/`in` are what the CPU path needs, one expert at a time. The file may
/// store an expert as [out][in] or [in][out] depending on which dimension ggml
/// put first, so the declared dims decide; guessing produces plausible garbage.
fn loadExperts(
    allocator: std.mem.Allocator,
    g: *const gguf.Gguf,
    name: []const u8,
    dst: []f16,
    num_experts: u32,
    out_dim: u32,
    in_dim: u32,
) !void {
    const t = g.tensor(name) orelse {
        sys.eprint("gguf: no expert tensor {s}\n", .{name});
        return Error.MissingTensor;
    };
    const per_expert: u64 = @as(u64, out_dim) * in_dim;
    if (t.elemCount() != per_expert * num_experts) {
        sys.eprint("gguf: {s} has {d} elements, expected {d} ({d} experts x {d}x{d}), dims {any}\n", .{
            name, t.elemCount(), per_expert * num_experts, num_experts, out_dim, in_dim, t.dims,
        });
        return Error.DimensionMismatch;
    }
    if (t.dims.len != 3) {
        sys.eprint("gguf: {s} is not 3-D (dims {any})\n", .{ name, t.dims });
        return Error.DimensionMismatch;
    }

    // The expert axis is whichever dimension equals num_experts.
    var expert_axis: ?usize = null;
    for (t.dims, 0..) |d, i| {
        if (d == num_experts) {
            expert_axis = i;
            break;
        }
    }
    const axis = expert_axis orelse {
        sys.eprint("gguf: {s} dims {any} has no axis equal to {d} experts\n", .{ name, t.dims, num_experts });
        return Error.DimensionMismatch;
    };

    // ggml's ne[0] varies fastest, so the element at (i0, i1, i2) is at
    // i0 + ne0*(i1 + ne1*i2). Requiring the expert axis to be last keeps each
    // expert's slice contiguous, which is the case llama.cpp writes.
    if (axis != 2) return Error.DimensionMismatch;

    const src = try g.readF16(allocator, name);
    defer allocator.free(src);

    // Within an expert the remaining dims are [d0, d1] with d0 fastest.
    const d0 = t.dims[0];
    const d1 = t.dims[1];
    if (d0 == in_dim and d1 == out_dim) {
        // [in-fastest, out] -> an expert's slice is out-major with in contiguous,
        // i.e. exactly [out][in]. Straight copy.
        for (0..num_experts) |e| {
            @memcpy(dst[e * per_expert ..][0..per_expert], src[e * per_expert ..][0..per_expert]);
        }
    } else if (d0 == out_dim and d1 == in_dim) {
        // [out-fastest, in] -> the slice is in-major; transpose it.
        for (0..num_experts) |e| {
            const slice = src[e * per_expert ..][0..per_expert];
            const out = dst[e * per_expert ..][0..per_expert];
            for (0..out_dim) |o| {
                for (0..in_dim) |i| {
                    out[o * in_dim + i] = slice[i * out_dim + o];
                }
            }
        }
    } else {
        return Error.DimensionMismatch;
    }
}

test "a config missing a required key is reported, not silently zeroed" {
    // A bare `MissingTensor` from a config read sent the reader looking for a broken
    // tensor. The loader now names the key, which is the difference between "this file
    // is for another runtime" and "your download is corrupt".
    const a = std.testing.allocator;
    var bytes: std.ArrayList(u8) = .empty;
    defer bytes.deinit(a);
    try putInt(&bytes, a, u32, 0x4655_4747);
    try putInt(&bytes, a, u32, 3);
    try putInt(&bytes, a, u64, 0); // no tensors
    try putInt(&bytes, a, u64, 2); // two kv pairs
    try putStr(&bytes, a, "general.architecture", "llama");
    try putInt(&bytes, a, u64, "llama.embedding_length".len);
    try bytes.appendSlice(a, "llama.embedding_length");
    try putInt(&bytes, a, u32, 4);
    try putInt(&bytes, a, u32, 576);
    while (bytes.items.len % 32 != 0) try bytes.append(a, 0);

    var g = try gguf.Gguf.fromBytes(a, bytes.items);
    defer g.deinit();
    // block_count is deliberately absent.
    try std.testing.expectError(Error.MissingTensor, loadConfig(&g));
}

test "a complete config still loads" {
    // Guards the diagnostics above against swallowing real errors: a file with every
    // required key must load, not report one missing.
    const a = std.testing.allocator;
    var bytes: std.ArrayList(u8) = .empty;
    defer bytes.deinit(a);
    try putInt(&bytes, a, u32, 0x4655_4747);
    try putInt(&bytes, a, u32, 3);
    try putInt(&bytes, a, u64, 0);
    try putInt(&bytes, a, u64, 6);
    try putStr(&bytes, a, "general.architecture", "llama");
    try putU32Kv(&bytes, a, "llama.embedding_length", 576);
    try putU32Kv(&bytes, a, "llama.block_count", 4);
    try putU32Kv(&bytes, a, "llama.attention.head_count", 9);
    try putU32Kv(&bytes, a, "llama.feed_forward_length", 1536);
    try putU32Kv(&bytes, a, "llama.vocab_size", 1024);
    while (bytes.items.len % 32 != 0) try bytes.append(a, 0);

    var g = try gguf.Gguf.fromBytes(a, bytes.items);
    defer g.deinit();
    const cfg = try loadConfig(&g);
    try std.testing.expectEqual(@as(u32, 576), cfg.hidden);
    try std.testing.expectEqual(@as(u32, 4), cfg.layers);
    try std.testing.expectEqual(@as(u32, 9), cfg.heads);
    try std.testing.expectEqual(@as(u32, 1024), cfg.vocab);
}

fn putInt(buf: *std.ArrayList(u8), a: std.mem.Allocator, comptime T: type, v: T) !void {
    var tmp: [@sizeOf(T)]u8 = undefined;
    std.mem.writeInt(T, &tmp, v, .little);
    try buf.appendSlice(a, &tmp);
}

fn putStr(buf: *std.ArrayList(u8), a: std.mem.Allocator, k: []const u8, value: []const u8) !void {
    try putInt(buf, a, u64, k.len);
    try buf.appendSlice(a, k);
    try putInt(buf, a, u32, 8); // string
    try putInt(buf, a, u64, value.len);
    try buf.appendSlice(a, value);
}

fn putU32Kv(buf: *std.ArrayList(u8), a: std.mem.Allocator, k: []const u8, value: u32) !void {
    try putInt(buf, a, u64, k.len);
    try buf.appendSlice(a, k);
    try putInt(buf, a, u32, 4); // u32
    try putInt(buf, a, u32, value);
}

/// Write a tensor-info entry: name, rank-`n` dims, type, offset. Needed by the MoE tests
/// below, which must present a `ffn_gate_exps.weight` for the config to read widths from.
fn putTensor(buf: *std.ArrayList(u8), a: std.mem.Allocator, name: []const u8, dims: []const u64, ty: u32, offset: u64) !void {
    try putInt(buf, a, u64, name.len);
    try buf.appendSlice(a, name);
    try putInt(buf, a, u32, @intCast(dims.len));
    for (dims) |d| try putInt(buf, a, u64, d);
    try putInt(buf, a, u32, ty);
    try putInt(buf, a, u64, offset);
}

test "MoE widths come from the tensor shapes, not the metadata" {
    // The subtlety that matters: Qwen1.5-MoE ships no `expert_feed_forward_length`, and its
    // `feed_forward_length` (5632) is the SHARED expert's width while the routed experts are
    // 1408. Defaulting one from the other sizes every expert wrong or drops the shared
    // expert entirely, which is a wrong answer rather than a slow one. The tensor shapes win.
    const a = std.testing.allocator;
    var bytes: std.ArrayList(u8) = .empty;
    defer bytes.deinit(a);

    try putInt(&bytes, a, u32, 0x4655_4747);
    try putInt(&bytes, a, u32, 3);
    try putInt(&bytes, a, u64, 1); // one tensor
    try putInt(&bytes, a, u64, 8); // eight kv pairs
    try putStr(&bytes, a, "general.architecture", "qwen2moe");
    try putU32Kv(&bytes, a, "qwen2moe.embedding_length", 2048);
    try putU32Kv(&bytes, a, "qwen2moe.block_count", 1);
    try putU32Kv(&bytes, a, "qwen2moe.attention.head_count", 16);
    try putU32Kv(&bytes, a, "qwen2moe.feed_forward_length", 5632);
    try putU32Kv(&bytes, a, "qwen2moe.vocab_size", 1024);
    try putU32Kv(&bytes, a, "qwen2moe.expert_count", 60);
    try putU32Kv(&bytes, a, "qwen2moe.expert_used_count", 4);
    // The tensor disagrees with the declared expert count on purpose: 64 against 60, and
    // its middle dimension is the routed width (1408), not the shared width (5632).
    try putTensor(&bytes, a, "blk.0.ffn_gate_exps.weight", &.{ 2048, 1408, 64 }, 2, 0);
    while (bytes.items.len % 32 != 0) try bytes.append(a, 0);

    var g = try gguf.Gguf.fromBytes(a, bytes.items);
    defer g.deinit();
    const cfg = try loadConfig(&g);
    try std.testing.expectEqual(@as(u32, 1408), cfg.moe_inter); // from dims[1]
    try std.testing.expectEqual(@as(u32, 64), cfg.num_experts); // from dims[2], not the kv
    try std.testing.expectEqual(@as(u32, 4), cfg.experts_per_tok);
    // feed_forward_length stays the SHARED width; it is a different number.
    try std.testing.expectEqual(@as(u32, 5632), cfg.inter);
}

test "a Gemma GGUF does NOT need the (1 + w) offset, because it is already applied" {
    // The other half of the asymmetry pinned in `load_hf`. Measured on gemma-2-2b:
    // `blk.0.attn_norm.weight` has mean 1.1927, which IS the effective `(1 + w)` for a
    // converged model; the raw HF parameter would have mean ~0.19. Adding the offset again
    // doubles every norm and the model emits "." instead of an answer.
    const a = std.testing.allocator;
    var bytes: std.ArrayList(u8) = .empty;
    defer bytes.deinit(a);

    try putInt(&bytes, a, u32, 0x4655_4747);
    try putInt(&bytes, a, u32, 3);
    try putInt(&bytes, a, u64, 1); // blk.0.post_attention_norm.weight
    try putInt(&bytes, a, u64, 8); // eight kv pairs, exactly as written below
    try putStr(&bytes, a, "general.architecture", "gemma2");
    try putU32Kv(&bytes, a, "gemma2.embedding_length", 2304);
    try putU32Kv(&bytes, a, "gemma2.block_count", 26);
    try putU32Kv(&bytes, a, "gemma2.attention.head_count", 8);
    try putU32Kv(&bytes, a, "gemma2.attention.head_count_kv", 4);
    try putU32Kv(&bytes, a, "gemma2.attention.key_length", 256);
    try putU32Kv(&bytes, a, "gemma2.feed_forward_length", 9216);
    try putU32Kv(&bytes, a, "gemma2.vocab_size", 256000);
    // The soft-caps are optional (`orelse 0`), so they are left out here deliberately.
    try putTensor(&bytes, a, "blk.0.post_attention_norm.weight", &.{2304}, 0, 0);
    while (bytes.items.len % 32 != 0) try bytes.append(a, 0);

    var g = try gguf.Gguf.fromBytes(a, bytes.items);
    defer g.deinit();
    const cfg = try loadConfig(&g);
    try std.testing.expect(!cfg.norm_unit_offset); // NOT applied: the file already has it
    try std.testing.expect(cfg.use_gelu);
    try std.testing.expectApproxEqAbs(@sqrt(@as(f32, 2304)), cfg.embed_scale, 1e-4);
    // The sandwich norms are detected from the tensor being present.
    try std.testing.expect(cfg.sandwich_norms);
    // attn_scale is 1/sqrt(n_embd/n_head) = 1/sqrt(288), not 1/sqrt(256).
    try std.testing.expect(cfg.attn_scale > 0);
    try std.testing.expectApproxEqAbs(1.0 / @sqrt(288.0), cfg.attn_scale, 1e-5);
}

test "a Mistral GGUF's sliding window is read, not ignored" {
    // The loader read `attention.sliding_window` only inside the gemma branch, so a Mistral
    // file with the key attended globally over the whole context. This asserts the key is
    // honoured and that all layers slide, which is Mistral's rule rather than Gemma 2's.
    const a = std.testing.allocator;
    var bytes: std.ArrayList(u8) = .empty;
    defer bytes.deinit(a);

    try putInt(&bytes, a, u32, 0x4655_4747);
    try putInt(&bytes, a, u32, 3);
    try putInt(&bytes, a, u64, 0);
    try putInt(&bytes, a, u64, 7); // exactly the kv pairs written below
    try putStr(&bytes, a, "general.architecture", "mistral");
    try putU32Kv(&bytes, a, "mistral.embedding_length", 256);
    try putU32Kv(&bytes, a, "mistral.block_count", 2);
    try putU32Kv(&bytes, a, "mistral.attention.head_count", 8);
    try putU32Kv(&bytes, a, "mistral.feed_forward_length", 512);
    try putU32Kv(&bytes, a, "mistral.attention.sliding_window", 4096);
    try putU32Kv(&bytes, a, "mistral.vocab_size", 1024);
    while (bytes.items.len % 32 != 0) try bytes.append(a, 0);

    var g = try gguf.Gguf.fromBytes(a, bytes.items);
    defer g.deinit();
    const cfg = try loadConfig(&g);
    try std.testing.expectEqual(@as(u32, 4096), cfg.sliding_window);
    try std.testing.expect(cfg.swa_all);
    try std.testing.expect(cfg.layerIsSliding(0));
    try std.testing.expect(cfg.layerIsSliding(1)); // every layer, not alternating
    // Mistral is a llama-family model: adjacent RoPE and no Gemma-specific flags.
    try std.testing.expect(cfg.rope_adjacent);
    try std.testing.expect(!cfg.use_gelu);
    try std.testing.expect(!cfg.norm_unit_offset);
}

test "a gemma4 GGUF is refused rather than loaded as llama" {
    // Gemma 4 differs from llama in ways that cannot be guessed: per-layer head dims and FFN
    // widths, an explicit per-layer sliding/full list, KV sharing and per-layer input
    // embeddings. "Treat it like llama" would run and produce fluent nonsense, so the loader
    // must refuse it. Every key below is one the official Gemma 4 E2B GGUF actually carries.
    const a = std.testing.allocator;
    var bytes: std.ArrayList(u8) = .empty;
    defer bytes.deinit(a);

    try putInt(&bytes, a, u32, 0x4655_4747);
    try putInt(&bytes, a, u32, 3);
    try putInt(&bytes, a, u64, 0);
    try putInt(&bytes, a, u64, 8); // exactly the pairs written below
    try putStr(&bytes, a, "general.architecture", "gemma4");
    try putU32Kv(&bytes, a, "gemma4.embedding_length", 1536);
    try putU32Kv(&bytes, a, "gemma4.block_count", 35);
    try putU32Kv(&bytes, a, "gemma4.attention.head_count", 8);
    try putU32Kv(&bytes, a, "gemma4.attention.head_count_kv", 1);
    try putU32Kv(&bytes, a, "gemma4.feed_forward_length", 6144);
    try putU32Kv(&bytes, a, "gemma4.attention.key_length", 512);
    try putU32Kv(&bytes, a, "gemma4.attention.sliding_window", 512);
    while (bytes.items.len % 32 != 0) try bytes.append(a, 0);

    var g = try gguf.Gguf.fromBytes(a, bytes.items);
    defer g.deinit();
    try std.testing.expectError(Error.UnsupportedArchitecture, loadConfig(&g));

    // A nearby name that shares no such difference must still take the ordinary path, so this
    // test cannot pass merely by refusing everything beginning with "gemma".
    const a2 = std.testing.allocator;
    var bytes2: std.ArrayList(u8) = .empty;
    defer bytes2.deinit(a2);
    try putInt(&bytes2, a2, u32, 0x4655_4747);
    try putInt(&bytes2, a2, u32, 3);
    try putInt(&bytes2, a2, u64, 0);
    try putInt(&bytes2, a2, u64, 7);
    try putStr(&bytes2, a2, "general.architecture", "gemma2");
    try putU32Kv(&bytes2, a2, "gemma2.embedding_length", 2304);
    try putU32Kv(&bytes2, a2, "gemma2.block_count", 26);
    try putU32Kv(&bytes2, a2, "gemma2.attention.head_count", 8);
    try putU32Kv(&bytes2, a2, "gemma2.feed_forward_length", 9216);
    try putU32Kv(&bytes2, a2, "gemma2.attention.sliding_window", 4096);
    try putU32Kv(&bytes2, a2, "gemma2.vocab_size", 256000);
    while (bytes2.items.len % 32 != 0) try bytes2.append(a2, 0);
    var g2 = try gguf.Gguf.fromBytes(a2, bytes2.items);
    defer g2.deinit();
    const cfg2 = try loadConfig(&g2);
    try std.testing.expectEqual(@as(u32, 2304), cfg2.hidden);
    try std.testing.expect(!cfg2.swa_all); // Gemma 2 alternates, it does not slide everywhere
}

test "a per-layer sliding list is read from a GGUF bool array" {
    // Gemma 4 writes `attention.sliding_window_pattern` as an array of bools, one per layer,
    // matching HF's `layer_types`. The repeating-length form (Gemma 2/3) is a scalar, and both
    // shapes have to reach `Config` — this drives the array one, since `gemma4` itself is
    // refused until the rest of the architecture is implemented.
    const a = std.testing.allocator;
    var bytes: std.ArrayList(u8) = .empty;
    defer bytes.deinit(a);

    const flags = [_]bool{ true, true, false, true, true };
    try putInt(&bytes, a, u32, 0x4655_4747);
    try putInt(&bytes, a, u32, 3);
    try putInt(&bytes, a, u64, 0);
    try putInt(&bytes, a, u64, 8);
    try putStr(&bytes, a, "general.architecture", "llama");
    try putU32Kv(&bytes, a, "llama.embedding_length", 256);
    try putU32Kv(&bytes, a, "llama.block_count", 5);
    try putU32Kv(&bytes, a, "llama.attention.head_count", 4);
    try putU32Kv(&bytes, a, "llama.feed_forward_length", 512);
    try putU32Kv(&bytes, a, "llama.vocab_size", 1024);
    try putU32Kv(&bytes, a, "llama.attention.sliding_window", 512);
    {
        // name + type 9 (array) + element type 7 (bool) + count + payload
        const name = "llama.attention.sliding_window_pattern";
        try putInt(&bytes, a, u64, name.len);
        try bytes.appendSlice(a, name);
        try putInt(&bytes, a, u32, 9);
        try putInt(&bytes, a, u32, 7);
        try putInt(&bytes, a, u64, flags.len);
        for (flags) |f| try bytes.append(a, @intFromBool(f));
    }
    while (bytes.items.len % 32 != 0) try bytes.append(a, 0);

    var g = try gguf.Gguf.fromBytes(a, bytes.items);
    defer g.deinit();
    const cfg = try loadConfig(&g);
    try std.testing.expect(cfg.swa_explicit);
    try std.testing.expect(cfg.layerIsSliding(0));
    try std.testing.expect(cfg.layerIsSliding(1));
    try std.testing.expect(!cfg.layerIsSliding(2)); // the false in the list
    try std.testing.expect(cfg.layerIsSliding(3));
    // The scalar window and the array coexist: swa_all must not also be set.
    try std.testing.expect(!cfg.swa_all);
}

test "per-layer FFN widths: the metadata array gives the max, the tensors give each layer" {
    // Gemma 4 declares `feed_forward_length` as an array ([6144 x15, 12288 x20, ...]) and its
    // layers differ. The scalar read returned null on an array so the load died on a missing
    // key; the max is what scratch buffers need, and each layer's width must come from its own
    // tensor, which is why the loader no longer passes `cfg.inter` to `loadLinear`.
    const a = std.testing.allocator;
    var bytes: std.ArrayList(u8) = .empty;
    defer bytes.deinit(a);

    // Two layers: layer 0 is 64 wide, layer 1 is 128, so the metadata max is 128.
    const hidden: u32 = 32;
    try putInt(&bytes, a, u32, 0x4655_4747);
    try putInt(&bytes, a, u32, 3);
    try putInt(&bytes, a, u64, 2); // two tensors
    try putInt(&bytes, a, u64, 6);
    try putStr(&bytes, a, "general.architecture", "llama");
    try putU32Kv(&bytes, a, "llama.embedding_length", hidden);
    try putU32Kv(&bytes, a, "llama.block_count", 2);
    try putU32Kv(&bytes, a, "llama.attention.head_count", 4);
    try putU32Kv(&bytes, a, "llama.vocab_size", 64);
    {
        const name = "llama.feed_forward_length";
        try putInt(&bytes, a, u64, name.len);
        try bytes.appendSlice(a, name);
        try putInt(&bytes, a, u32, 9); // array
        try putInt(&bytes, a, u32, 4); // of u32
        try putInt(&bytes, a, u64, 2);
        try putInt(&bytes, a, u32, 64);
        try putInt(&bytes, a, u32, 128);
    }
    // tensor infos: ffn_gate for both layers, F32 so the geometry is what is being tested
    for ([_]struct { n: []const u8, out: u32 }{ .{ .n = "blk.0.ffn_gate.weight", .out = 64 }, .{ .n = "blk.1.ffn_gate.weight", .out = 128 } }) |t| {
        try putTensor(&bytes, a, t.n, &[_]u64{ hidden, t.out }, 0, 0);
    }
    while (bytes.items.len % 32 != 0) try bytes.append(a, 0);

    var g = try gguf.Gguf.fromBytes(a, bytes.items);
    defer g.deinit();
    const cfg = try loadConfig(&g);
    try std.testing.expectEqual(@as(u32, 128), cfg.inter); // the max, for scratch sizing
    // ...and each layer's own width, which is what the loader now feeds to loadLinear.
    try std.testing.expectEqual(@as(u32, 64), tensorOutDim(&g, "blk.0.ffn_gate.weight", hidden).?);
    try std.testing.expectEqual(@as(u32, 128), tensorOutDim(&g, "blk.1.ffn_gate.weight", hidden).?);
    // A missing tensor or a wrong in_dim must not be mistaken for a width.
    try std.testing.expect(tensorOutDim(&g, "blk.9.ffn_gate.weight", hidden) == null);
    try std.testing.expect(tensorOutDim(&g, "blk.0.ffn_gate.weight", 7) == null);
}

test "gemma4's second head dimension is read from key_length_swa" {
    // `key_length` is the global layers' (512) and `key_length_swa` the sliding layers' (256).
    // Without this the sliding layers would be run at 512 and the KV cache sized wrong.
    const a = std.testing.allocator;
    var bytes: std.ArrayList(u8) = .empty;
    defer bytes.deinit(a);

    try putInt(&bytes, a, u32, 0x4655_4747);
    try putInt(&bytes, a, u32, 3);
    try putInt(&bytes, a, u64, 0);
    try putInt(&bytes, a, u64, 9);
    try putStr(&bytes, a, "general.architecture", "llama");
    try putU32Kv(&bytes, a, "llama.embedding_length", 1536);
    try putU32Kv(&bytes, a, "llama.block_count", 2);
    try putU32Kv(&bytes, a, "llama.attention.head_count", 8);
    try putU32Kv(&bytes, a, "llama.attention.head_count_kv", 1);
    try putU32Kv(&bytes, a, "llama.feed_forward_length", 6144);
    try putU32Kv(&bytes, a, "llama.vocab_size", 1024);
    try putU32Kv(&bytes, a, "llama.attention.key_length", 512);
    try putU32Kv(&bytes, a, "llama.attention.key_length_swa", 256);
    try putU32Kv(&bytes, a, "llama.attention.sliding_window", 512);
    while (bytes.items.len % 32 != 0) try bytes.append(a, 0);

    var g = try gguf.Gguf.fromBytes(a, bytes.items);
    defer g.deinit();
    const cfg = try loadConfig(&g);
    try std.testing.expectEqual(@as(u32, 512), cfg.head_dim);
    try std.testing.expectEqual(@as(u32, 256), cfg.head_dim_swa);
    // head_count_kv is 1, so a KV row is the head dimension itself. This fixture carries no
    // sliding list, so every layer is non-sliding and uses the global 512 — the 256 is what a
    // sliding layer will use once `layerIsSliding` says so.
    try std.testing.expectEqual(@as(usize, 512), cfg.layerKvDim(0));
    try std.testing.expectEqual(@as(usize, 512), cfg.layerKvDim(1));
    var sliding = cfg;
    sliding.sliding_window = 512;
    sliding.swa_explicit = true;
    sliding.swa_layers[0] = 0b01; // layer 0 slides
    try std.testing.expectEqual(@as(usize, 256), sliding.layerKvDim(0));
    try std.testing.expectEqual(@as(usize, 512), sliding.layerKvDim(1));
    try std.testing.expectEqual(@as(u32, 8 * 512), sliding.maxQDim());
}
