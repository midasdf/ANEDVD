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
    temperature: f32 = 0.7,
    top_k: usize = 40,
    seed: u32 = 0x12345678,
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

pub const Session = struct {
    allocator: std.mem.Allocator,
    engine: *engine_mod.Engine,
    tokenizer: *const tokenizer_mod.Tokenizer,
    rng: u32 = 0x12345678,
    /// Ids that terminate generation (EOS plus any explicit stop tokens).
    stop_ids: []const u32 = &.{},

    pub fn init(allocator: std.mem.Allocator, engine: *engine_mod.Engine, tok: *const tokenizer_mod.Tokenizer) Session {
        return .{ .allocator = allocator, .engine = engine, .tokenizer = tok, .rng = 0x12345678 };
    }

    fn isStop(self: *const Session, id: u32) bool {
        for (self.stop_ids) |s| if (s == id) return true;
        return false;
    }

    /// Build a prompt string from chat messages.
    ///
    /// Uses ChatML when the vocabulary has the markers (Qwen, SmolLM2, …) and
    /// falls back to a plain "User:/Assistant:" transcript otherwise.
    pub fn formatChat(self: *const Session, allocator: std.mem.Allocator, messages: []const Message, default_system: ?[]const u8) ![]u8 {
        var out = std.ArrayList(u8).empty;
        errdefer out.deinit(allocator);
        const tok = self.tokenizer;
        const chatml = tok.tokenId("<|im_start|>") != null and tok.tokenId("<|im_end|>") != null;

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

    /// Prefill `prompt_ids`, then sample up to `params.max_tokens` tokens.
    /// `emitter` receives each decoded piece as it is produced.
    pub fn generate(self: *Session, prompt_ids: []const u32, params: Params, emitter: Emitter) !Stats {
        var stats = Stats{};
        const eng = self.engine;
        eng.reset();
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

        // One batched pass over the whole prompt: the ANE is weight-bandwidth
        // bound, so a chunk of tokens costs about the same as a single one.
        const p0 = nowNs();
        var logits: []f32 = try eng.prefill(ids, 0);
        stats.prefill_ns = nowNs() - p0;

        var pos: u32 = @intCast(ids.len);
        const d0 = nowNs();
        var produced: u32 = 0;
        while (produced < params.max_tokens and pos < max_seq) : (produced += 1) {
            const next = if (params.temperature <= 0)
                cpu.argmax(logits)
            else
                cpu.sampleTopK(logits, params.temperature, params.top_k, &self.rng);

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
            logits = try eng.forward(next, pos);
            pos += 1;
            if (produced + 1 == params.max_tokens) stats.stop_reason = .length;
        }
        if (stats.stop_reason == .stop and produced >= params.max_tokens) stats.stop_reason = .length;
        stats.decode_ns = nowNs() - d0;
        stats.completion_tokens = produced;
        return stats;
    }
};

fn nowNs() u64 {
    var ts: std.c.timespec = undefined;
    if (std.c.clock_gettime(std.c.CLOCK.MONOTONIC, &ts) != 0) return 0;
    return @as(u64, @intCast(ts.sec)) * 1_000_000_000 + @as(u64, @intCast(ts.nsec));
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
