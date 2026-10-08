// generate.zig — shared generation session used by the CLI and the HTTP server.
//
// Owns the sampling loop and prompt formatting so `anedvd run` and
// `anedvd serve` behave identically.

const std = @import("std");
const cpu = @import("cpu.zig");
const engine_mod = @import("engine.zig");
const tokenizer_mod = @import("tokenizer.zig");

pub const Role = enum {
    system,
    user,
    assistant,

    pub fn fromString(s: []const u8) ?Role {
        if (std.mem.eql(u8, s, "system")) return .system;
        if (std.mem.eql(u8, s, "user")) return .user;
        if (std.mem.eql(u8, s, "assistant")) return .assistant;
        return null;
    }
    pub fn toString(self: Role) []const u8 {
        return switch (self) {
            .system => "system",
            .user => "user",
            .assistant => "assistant",
        };
    }
};

pub const Message = struct {
    role: Role,
    content: []const u8,
};

pub const Params = struct {
    max_tokens: u32 = 256,
    /// Sampling knobs; `temperature <= 0` means greedy.
    sampler: cpu.SamplerParams = .{ .temperature = 0.7, .top_k = 40 },
    seed: u32 = 0x12345678,
    /// How many recent tokens the penalties consider.
    penalty_window: usize = 256,
};

pub const StopReason = enum {
    stop, // EOS / stop token
    length, // max_tokens reached
    abort, // emitter asked to stop (client disconnected)

    pub fn toString(self: StopReason) []const u8 {
        return switch (self) {
            .stop => "stop",
            .length => "length",
            .abort => "abort",
        };
    }
};

pub const Stats = struct {
    prompt_tokens: u32 = 0,
    completion_tokens: u32 = 0,
    prefill_ns: u64 = 0,
    decode_ns: u64 = 0,
    /// Time spent inside ANE evaluations during the decode phase.
    decode_ane_ns: u64 = 0,
    /// ANE time inside the prefill phase.
    prefill_ane_ns: u64 = 0,
    /// Prompt tokens that were already in the KV cache and did not need to be
    /// recomputed (multi-turn).
    prefill_reused: u32 = 0,
    /// Sampling cost over the whole run, and the worst-case candidate count.
    sample_ns: u64 = 0,
    sample_candidates_max: usize = 0,
    sample_candidates_total: u64 = 0,
    stop_reason: StopReason = .stop,

    pub fn decodeToksPerSec(self: Stats) f64 {
        if (self.decode_ns == 0) return 0;
        return @as(f64, @floatFromInt(self.completion_tokens)) / (@as(f64, @floatFromInt(self.decode_ns)) / 1e9);
    }
};

/// Called once per generated token with the decoded UTF-8 piece.
/// Return false to abort generation (e.g. the HTTP client went away).
pub const Emitter = struct {
    ctx: ?*anyopaque = null,
    func: *const fn (ctx: ?*anyopaque, piece: []const u8, token_id: u32) bool,

    pub fn emit(self: Emitter, piece: []const u8, token_id: u32) bool {
        return self.func(self.ctx, piece, token_id);
    }
};

/// Used when the sampler scratch cannot be allocated: sampling then falls back
/// to greedy decoding instead of failing the whole session.
var empty_candidates: [0]cpu.Candidate = .{};

pub const Session = struct {
    allocator: std.mem.Allocator,
    engine: *engine_mod.Engine,
    tokenizer: *const tokenizer_mod.Tokenizer,
    rng: u32 = 0x12345678,
    /// Ids that terminate generation (any explicit stop tokens).
    stop_ids: []const u32 = &.{},
    /// The tokenizer's EOS token, if it has one.
    eos_stop: ?u32 = null,
    /// Scratch for the sampler (one candidate per vocabulary entry). Set by
    /// `init`; without it the sampler falls back to greedy decoding.
    candidates: []cpu.Candidate,
    /// Token ids currently held in the engine's KV cache. Multi-turn chat only
    /// has to prefill the part that changed, since the transcript grows by
    /// appending.
    cached: std.ArrayList(u32) = .empty,

    pub fn init(allocator: std.mem.Allocator, engine: *engine_mod.Engine, tok: *const tokenizer_mod.Tokenizer) Session {
        const candidates = allocator.alloc(cpu.Candidate, tok.vocabSize()) catch &empty_candidates;
        // Probe buffer sized for the whole vocabulary: 151936 candidates is
        // 1.2 MB, trivial next to the weights, and it makes the concentration
        // measurement exact instead of capped.
        return .{
            .allocator = allocator,
            .engine = engine,
            .tokenizer = tok,
            .rng = 0x12345678,
            .candidates = candidates,
        };
    }

    pub fn deinit(self: *Session) void {
        if (self.candidates.len > 0) self.allocator.free(self.candidates);
        self.cached.deinit(self.allocator);
    }

    /// Drop the cached prefix (start a fresh conversation).
    pub fn reset(self: *Session) void {
        self.cached.clearRetainingCapacity();
        self.engine.reset();
    }

    fn isStop(self: *const Session, id: u32) bool {
        if (self.eos_stop) |eos| {
            if (id == eos) return true;
        }
        for (self.stop_ids) |s| if (s == id) return true;
        return false;
    }

    /// Stop when the model emits EOS. Owned by the session, so callers never
    /// have to allocate a stop list.
    pub fn stopOnEos(self: *Session) void {
        self.eos_stop = self.tokenizer.eosId();
    }

    /// Build a prompt string from chat messages; same as `formatChatFor`.
    pub fn formatChat(self: *const Session, allocator: std.mem.Allocator, messages: []const Message, default_system: ?[]const u8) ![]u8 {
        return formatChatFor(allocator, self.tokenizer, messages, default_system);
    }

    fn commonPrefix(a: []const u32, b: []const u32) usize {
        const n = @min(a.len, b.len);
        var i: usize = 0;
        while (i < n and a[i] == b[i]) : (i += 1) {}
        return i;
    }

    /// Prefill `prompt_ids`, then sample up to `params.max_tokens` tokens.
    /// `emitter` receives each decoded piece as it is produced.
    pub fn generate(self: *Session, prompt_ids: []const u32, params: Params, emitter: Emitter) !Stats {
        var stats = Stats{};
        const eng = self.engine;
        self.rng = params.seed;

        // Keep the prompt inside the KV cache, leaving room to generate.
        const max_seq: u32 = eng.max_seq;
        var ids = prompt_ids;
        if (ids.len + params.max_tokens + 1 > max_seq) {
            const keep = max_seq -| params.max_tokens -| 1;
            if (keep == 0) return error.PromptTooLong;
            ids = ids[ids.len - keep ..];
        }
        stats.prompt_tokens = @intCast(ids.len);

        // Reuse the KV prefix: a growing transcript only needs the new suffix,
        // which is what makes multi-turn chat cheap.
        const reuse = commonPrefix(self.cached.items, ids);
        stats.prefill_reused = @intCast(@min(reuse, ids.len));
        if (reuse == 0) eng.reset();
        self.cached.shrinkRetainingCapacity(reuse);

        // One batched pass over the new tokens: the ANE is weight-bandwidth
        // bound, so a chunk of tokens costs about the same as a single one.
        const p0 = nowNs();
        const ane_p0 = eng.stats.ane_eval_ns;
        var logits: []f32 = try eng.prefill(ids[reuse..], @intCast(reuse));
        const p1 = nowNs();
        stats.prefill_ane_ns = eng.stats.ane_eval_ns - ane_p0;
        stats.prefill_ns = p1 - p0;
        try self.cached.appendSlice(self.allocator, ids[reuse..]);

        // Recent tokens for the repetition / presence / frequency penalties.
        var recent: std.ArrayList(u32) = .empty;
        defer recent.deinit(self.allocator);
        const window = params.penalty_window;
        for (ids) |id| {
            if (recent.items.len >= window) _ = recent.orderedRemove(0);
            try recent.append(self.allocator, id);
        }

        var pos: u32 = @intCast(ids.len);
        const ane_before = eng.stats.ane_eval_ns;
        const d0 = nowNs();
        var produced: u32 = 0;
        while (produced < params.max_tokens and pos < max_seq) : (produced += 1) {
            var sinfo: cpu.SamplerInfo = undefined;
            const next = if (self.candidates.len >= logits.len)
                cpu.sampleProfiled(logits, params.sampler, recent.items, &self.rng, self.candidates, &sinfo)
            else
                cpu.argmax(logits);
            stats.sample_ns += sinfo.ns;
            stats.sample_candidates_total += sinfo.candidates;
            stats.sample_candidates_max = @max(stats.sample_candidates_max, sinfo.candidates);

            if (self.isStop(next)) {
                stats.stop_reason = .stop;
                break;
            }
            const piece = try self.tokenizer.tokenBytes(self.allocator, next);
            defer self.allocator.free(piece);
            if (!emitter.emit(piece, next)) {
                stats.stop_reason = .abort;
                produced += 1;
                break;
            }
            if (recent.items.len >= window) _ = recent.orderedRemove(0);
            try recent.append(self.allocator, next);
            try self.cached.append(self.allocator, next);
            logits = try eng.forward(next, pos);
            pos += 1;
            if (produced + 1 == params.max_tokens) stats.stop_reason = .length;
        }
        if (stats.stop_reason == .stop and produced >= params.max_tokens) stats.stop_reason = .length;
        stats.decode_ns = nowNs() - d0;
        stats.decode_ane_ns = eng.stats.ane_eval_ns - ane_before;
        stats.completion_tokens = produced;
        return stats;
    }
};

/// Build a prompt from chat messages using `tok`'s special tokens.
///
/// ChatML when the vocabulary has the markers (Qwen, SmolLM2, …), otherwise a
/// plain "User:/Assistant:" transcript. The result always ends with the
/// assistant header, so generation continues from there.
pub fn formatChatFor(
    allocator: std.mem.Allocator,
    tok: *const tokenizer_mod.Tokenizer,
    messages: []const Message,
    default_system: ?[]const u8,
) ![]u8 {
    var out = std.ArrayList(u8).empty;
    errdefer out.deinit(allocator);
    const chatml = tok.tokenId("<|im_start|>") != null and tok.tokenId("<|im_end|>") != null;
    // Zephyr/TinyLlama style: <|user|>\n...<|assistant|>\n, with EOS between turns.
    // The markers are NOT vocabulary entries in those files (only ChatML models
    // put their markers in the vocabulary), so this is selected from the GGUF
    // `tokenizer.chat_template` text, not by tokenId lookup.
    const zephyr = !chatml and std.mem.indexOf(u8, tok.chat_template, "<|user|>") != null;

    if (zephyr) {
        // The template uses the EOS *piece text* as the separator; emitting the
        // literal string keeps this independent of how EOS is spelled.
        const eos_text = if (tok.eosId()) |e| tok.tokenText(e) else "";
        if (default_system) |sys_text| {
            try out.appendSlice(allocator, "<|system|>\n");
            try out.appendSlice(allocator, sys_text);
            try out.appendSlice(allocator, eos_text);
        }
        for (messages) |m| {
            try out.appendSlice(allocator, "<|");
            try out.appendSlice(allocator, m.role.toString());
            try out.appendSlice(allocator, "|>\n");
            try out.appendSlice(allocator, m.content);
            try out.appendSlice(allocator, eos_text);
        }
        try out.appendSlice(allocator, "<|assistant|>\n");
        return out.toOwnedSlice(allocator);
    }

    if (chatml) {
        var wrote_system = false;
        for (messages) |m| {
            if (m.role == .system) {
                try out.appendSlice(allocator, "<|im_start|>system\n");
                try out.appendSlice(allocator, m.content);
                try out.appendSlice(allocator, "<|im_end|>\n");
                wrote_system = true;
            }
        }
        if (!wrote_system) {
            if (default_system) |sys_text| {
                try out.appendSlice(allocator, "<|im_start|>system\n");
                try out.appendSlice(allocator, sys_text);
                try out.appendSlice(allocator, "<|im_end|>\n");
            }
        }
        for (messages) |m| {
            if (m.role == .system) continue;
            try out.appendSlice(allocator, "<|im_start|>");
            try out.appendSlice(allocator, m.role.toString());
            try out.appendSlice(allocator, "\n");
            try out.appendSlice(allocator, m.content);
            try out.appendSlice(allocator, "<|im_end|>\n");
        }
        try out.appendSlice(allocator, "<|im_start|>assistant\n");
    } else {
        if (default_system) |sys_text| {
            try out.appendSlice(allocator, sys_text);
            try out.appendSlice(allocator, "\n\n");
        }
        for (messages) |m| {
            switch (m.role) {
                .system => {
                    try out.appendSlice(allocator, m.content);
                    try out.appendSlice(allocator, "\n\n");
                },
                .user => {
                    try out.appendSlice(allocator, "User: ");
                    try out.appendSlice(allocator, m.content);
                    try out.appendSlice(allocator, "\n");
                },
                .assistant => {
                    try out.appendSlice(allocator, "Assistant: ");
                    try out.appendSlice(allocator, m.content);
                    try out.appendSlice(allocator, "\n");
                },
            }
        }
        try out.appendSlice(allocator, "Assistant:");
    }
    return out.toOwnedSlice(allocator);
}

fn nowNs() u64 {
    var ts: std.c.timespec = undefined;
    if (std.c.clock_gettime(std.c.CLOCK.MONOTONIC, &ts) != 0) return 0;
    return @as(u64, @intCast(ts.sec)) * 1_000_000_000 + @as(u64, @intCast(ts.nsec));
}

test "chatml prompt layout" {
    const a = std.testing.allocator;
    var tok = try tokenizer_mod.Tokenizer.fromTokenizerJsonSlice(a, chatmlFixture());
    defer tok.deinit();
    const msgs = [_]Message{
        .{ .role = .user, .content = "hi" },
        .{ .role = .assistant, .content = "hello" },
        .{ .role = .user, .content = "bye" },
    };
    const prompt = try formatChatFor(a, &tok, &msgs, null);
    defer a.free(prompt);
    try std.testing.expectEqualStrings(
        "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\nhello<|im_end|>\n<|im_start|>user\nbye<|im_end|>\n<|im_start|>assistant\n",
        prompt,
    );
}

test "chatml prompt inserts the default system message once" {
    const a = std.testing.allocator;
    var tok = try tokenizer_mod.Tokenizer.fromTokenizerJsonSlice(a, chatmlFixture());
    defer tok.deinit();
    const msgs = [_]Message{.{ .role = .user, .content = "hi" }};
    const prompt = try formatChatFor(a, &tok, &msgs, "be nice");
    defer a.free(prompt);
    try std.testing.expect(std.mem.startsWith(u8, prompt, "<|im_start|>system\nbe nice<|im_end|>\n"));
    // An explicit system message wins over the default.
    const with_system = [_]Message{
        .{ .role = .system, .content = "explicit" },
        .{ .role = .user, .content = "hi" },
    };
    const p2 = try formatChatFor(a, &tok, &with_system, "be nice");
    defer a.free(p2);
    try std.testing.expect(std.mem.startsWith(u8, p2, "<|im_start|>system\nexplicit<|im_end|>\n"));
    try std.testing.expect(std.mem.indexOf(u8, p2, "be nice") == null);
}

test "plain transcript when the vocabulary has no ChatML markers" {
    const a = std.testing.allocator;
    var tok = try tokenizer_mod.Tokenizer.fromTokenizerJsonSlice(a, plainFixture());
    defer tok.deinit();
    const msgs = [_]Message{
        .{ .role = .user, .content = "hi" },
        .{ .role = .assistant, .content = "hello" },
    };
    const prompt = try formatChatFor(a, &tok, &msgs, "sys");
    defer a.free(prompt);
    try std.testing.expectEqualStrings("sys\n\nUser: hi\nAssistant: hello\nAssistant:", prompt);
}

fn chatmlFixture() []const u8 {
    return
    \\{"model":{"type":"BPE","vocab":{"<|im_start|>":0,"<|im_end|>":1,"a":2},"merges":[]}}
    ;
}

fn plainFixture() []const u8 {
    return
    \\{"model":{"type":"BPE","vocab":{"a":0,"b":1},"merges":[]}}
    ;
}

test "role parsing" {
    try std.testing.expectEqual(Role.user, Role.fromString("user").?);
    try std.testing.expectEqual(Role.assistant, Role.fromString("assistant").?);
    try std.testing.expectEqual(Role.system, Role.fromString("system").?);
    try std.testing.expect(Role.fromString("tool") == null);
}

test "stop reason strings match the OpenAI finish_reason vocabulary" {
    try std.testing.expectEqualStrings("stop", StopReason.stop.toString());
    try std.testing.expectEqualStrings("length", StopReason.length.toString());
}
