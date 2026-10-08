// server.zig — OpenAI- and Anthropic-compatible HTTP API + the built-in WebUI.
//
// Routes:
//   GET  /                      WebUI (src/webui.html)
//   GET  /health                liveness probe
//   GET  /v1/models             OpenAI model list
//   POST /v1/chat/completions   OpenAI chat completions (streaming and not)
//   POST /v1/completions        OpenAI legacy completions
//   POST /v1/messages           Anthropic messages (streaming and not)
//
// Requests are handled one at a time: there is a single ANE engine and a single
// KV cache, so concurrent generations would corrupt each other. A second client
// simply waits.

const std = @import("std");
const sys = @import("sys.zig");
const Buf = @import("buf.zig").Buf;
const http = @import("http.zig");
const generate = @import("generate.zig");

const WEBUI_HTML = @embedFile("webui.html");

pub const Options = struct {
    host: []const u8 = "127.0.0.1",
    port: u16 = 8080,
    model_name: []const u8 = "anedvd",
    default_system: ?[]const u8 = null,
    default_max_tokens: u32 = 512,
    default_top_p: f32 = 1.0,
    default_repetition_penalty: f32 = 1.0,
};

pub const Server = struct {
    allocator: std.mem.Allocator,
    session: *generate.Session,
    opts: Options,
    /// Listener, so the mid-generation hook can poll for new connections.
    listener: ?*http.Server = null,
    /// Requests served while a generation was in flight.
    served_during_generation: u64 = 0,
    /// Generation requests turned away with 503 because the engine was busy.
    busy_rejections: u64 = 0,
    /// A connection accepted mid-generation that has not sent its request yet,
    /// kept alive between generated tokens instead of being dropped.
    pending: ?http.Conn = null,
    /// The engine's context window, reported when a prompt has to be truncated.
    context_limit: u32 = 0,

    pub fn run(self: *Server) !void {
        var srv = try http.Server.open(self.opts.host, self.opts.port);
        defer srv.close();
        self.listener = &srv;
        sys.print("ANEDVD server listening on http://{s}:{d}\n", .{ self.opts.host, self.opts.port });
        sys.print("  model: {s}\n", .{self.opts.model_name});
        sys.print("  endpoints: /v1/chat/completions (OpenAI), /v1/messages (Anthropic), /v1/completions, /v1/models, / (WebUI)\n", .{});
        sys.print("  requests are served one at a time (single ANE engine)\n", .{});

        while (true) {
            // Non-blocking so a connection that arrives mid-generation is
            // noticed by servicePending() rather than queueing behind it.
            srv.setNonBlocking(true);
            const maybe = srv.acceptIfPending();
            srv.setNonBlocking(false);
            var conn = maybe orelse {
                sys.sleepMs(2);
                continue;
            };
            self.handle(&conn) catch |e| {
                if (e != error.WriteFailed and e != error.ConnectionClosed) {
                    sys.eprint("request failed: {s}\n", .{@errorName(e)});
                }
            };
            conn.close();
        }
    }

    fn handle(self: *Server, conn: *http.Conn) !void {
        var arena = std.heap.ArenaAllocator.init(self.allocator);
        defer arena.deinit();
        const a = arena.allocator();

        var buf: std.ArrayList(u8) = .empty;
        defer buf.deinit(a);
        const req = conn.readRequest(a, &buf) catch |e| {
            if (e == error.ConnectionClosed) return e;
            return sendError(conn, 400, "malformed HTTP request");
        };
        const path = req.path();

        if (std.mem.eql(u8, req.method, "OPTIONS")) {
            try conn.beginWithLength(204, "No Content", "text/plain", 0);
            return;
        }
        if (std.mem.eql(u8, req.method, "GET") and (std.mem.eql(u8, path, "/") or std.mem.eql(u8, path, "/index.html"))) {
            try conn.sendBody(200, "OK", "text/html; charset=utf-8", WEBUI_HTML);
            return;
        }
        if (std.mem.eql(u8, req.method, "GET") and std.mem.eql(u8, path, "/health")) {
            try conn.sendBody(200, "OK", "application/json", "{\"status\":\"ok\"}");
            return;
        }
        if (std.mem.eql(u8, req.method, "GET") and std.mem.eql(u8, path, "/v1/models")) {
            return self.sendModels(conn);
        }
        if (std.mem.eql(u8, req.method, "POST") and std.mem.eql(u8, path, "/v1/chat/completions")) {
            return self.chatCompletions(conn, a, req.body);
        }
        if (std.mem.eql(u8, req.method, "POST") and std.mem.eql(u8, path, "/v1/completions")) {
            return self.completions(conn, a, req.body);
        }
        if (std.mem.eql(u8, req.method, "POST") and std.mem.eql(u8, path, "/v1/messages")) {
            return self.anthropicMessages(conn, a, req.body);
        }
        return sendError(conn, 404, "unknown endpoint");
    }

    fn sendModels(self: *Server, conn: *http.Conn) !void {
        var b = Buf.init(self.allocator);
        defer b.deinit();
        try b.print("{{\"object\":\"list\",\"data\":[{{\"id\":\"{s}\",\"object\":\"model\",\"created\":{d},\"owned_by\":\"anedvd\"}}]}}", .{
            self.opts.model_name, sys.unixTime(),
        });
        try conn.sendBody(200, "OK", "application/json", b.slice());
    }

    // ------------------------------------------------------------ OpenAI chat

    fn chatCompletions(self: *Server, conn: *http.Conn, a: std.mem.Allocator, body: []const u8) !void {
        const parsed = std.json.parseFromSliceLeaky(std.json.Value, a, body, .{}) catch
            return sendError(conn, 400, "invalid JSON body");
        if (parsed != .object) return sendError(conn, 400, "body must be a JSON object");
        const root = parsed.object;

        const model_name = getModelName(root, self.opts.model_name);
        const max_tokens: u32 = intToU32(getInt(root, "max_tokens", self.opts.default_max_tokens), 1);
        const temperature: f32 = getFloat(root, "temperature", 0.7);
        const top_k: usize = intToU32(getInt(root, "top_k", 40), 1);
        const stream = getBool(root, "stream", false);
        const stops = try parseStops(a, root);

        const messages = (try parseMessages(a, root)) orelse
            return sendError(conn, 400, "messages must be a non-empty array of {role, content}");
        if (messages.len == 0) return sendError(conn, 400, "messages must be non-empty");

        const prompt = try self.session.formatChat(a, messages, self.opts.default_system);
        const ids = try self.session.tokenizer.encode(a, prompt, true);
        if (ids.len == 0) return sendError(conn, 400, "prompt encoded to zero tokens");

        const params = generate.Params{
            .max_tokens = max_tokens,
            .sampler = .{
                .temperature = temperature,
                .top_k = top_k,
                .top_p = getFloat(root, "top_p", self.opts.default_top_p),
                .repetition_penalty = getFloat(root, "repetition_penalty", self.opts.default_repetition_penalty),
                .presence_penalty = getFloat(root, "presence_penalty", 0.0),
                .frequency_penalty = getFloat(root, "frequency_penalty", 0.0),
            },
        };
        const created = sys.unixTime();

        if (!stream) {
            var collector = Collector.init(self.allocator);
            collector.server = self;
            defer collector.deinit();
            const stats = try self.session.generate(ids, params, .{ .ctx = &collector, .func = collectEmit });

            self.logStats(stats);
            var b = Buf.init(self.allocator);
            defer b.deinit();
            try b.print("{{\"id\":\"chatcmpl-{d}\",\"object\":\"chat.completion\",\"created\":{d},\"model\":\"", .{ created, created });
            try jsonString(&b, model_name);
            try b.print("\",\"choices\":[{{\"index\":0,\"message\":{{\"role\":\"assistant\",\"content\":\"", .{});
            try jsonString(&b, collector.text.items);
            try b.print("\"}},\"finish_reason\":\"{s}\"}}],\"usage\":{{\"prompt_tokens\":{d},\"completion_tokens\":{d},\"total_tokens\":{d},\"prompt_tokens_dropped\":{d}}}}}", .{
                finishReason(stats.stop_reason), stats.prompt_tokens, stats.completion_tokens, stats.prompt_tokens + stats.completion_tokens, stats.prompt_tokens_dropped,
            });
            return conn.sendBody(200, "OK", "application/json", b.slice());
        }

        // Prefill happens before any token is emitted, and for a long prompt it
        // is seconds long, so hook the server into the chunk loop.
        self.session.engine.prefill_tick = prefillTick;
        self.session.engine.prefill_tick_ctx = self;
        defer {
            self.session.engine.prefill_tick = null;
            self.session.engine.prefill_tick_ctx = null;
        }

        try conn.beginStream(200, "OK", "text/event-stream");
        var sink = SseSink.init(self.allocator, conn, .openai, model_name, created);
        sink.server = self;
        defer sink.deinit();
        sink.setStops(stops);
        const stats = try self.session.generate(ids, params, .{ .ctx = &sink, .func = sseEmit });
        self.logStats(stats);
        if (!conn.alive) return;
        return sink.finishOpenAi(stats);
    }

    /// Called between generated tokens. Answers any connection that has already
    /// sent a complete request and needs no generation (health, model list) so a
    /// long generation cannot make the server look dead. Anything heavier is
    /// answered with 503, and a connection that has not sent its request yet is
    /// held until it does.
    ///
    /// The held connection matters: a client connects at the TCP handshake and
    /// only then writes its request. Dropping it because no bytes were ready yet
    /// closes the socket under the client, which sees an empty reply — measured
    /// at a 40 ms connect-to-request delay.
    pub fn servicePending(self: *Server) void {
        var srv = self.listener orelse return;
        srv.setNonBlocking(true);
        const maybe = srv.acceptIfPending();
        srv.setNonBlocking(false);

        // Take a newly accepted connection only if we have no other waiting one.
        if (maybe) |fresh| {
            if (self.pending == null) {
                self.pending = fresh;
            } else {
                var extra = fresh;
                extra.close();
            }
        }
        var conn = self.pending orelse return;

        // Do not wait: a poll that blocks would cost its timeout on every token
        // (5 ms x 400 tokens was 1.2 s of added generation time). Keeping the
        // connection in `pending` instead of dropping it is what makes the 0 ms
        // check safe -- a real client's request arrives within a token, and the
        // connection is still here when it does.
        if (!conn.hasPendingInput(0)) return; // keep it for the next token

        self.pending = null;
        defer conn.close();

        var arena = std.heap.ArenaAllocator.init(self.allocator);
        defer arena.deinit();
        const a = arena.allocator();
        var buf: std.ArrayList(u8) = .empty;
        defer buf.deinit(a);
        const req = conn.readRequest(a, &buf) catch |e| {
            if (e != error.ConnectionClosed) {
                conn.sendBody(400, "Bad Request", "application/json", "{\"error\":{\"message\":\"malformed request\"}}") catch {};
            }
            return;
        };
        const path = req.path();

        // The cheap read-only routes are answered on the spot.
        if (std.mem.eql(u8, req.method, "GET") and std.mem.eql(u8, path, "/health")) {
            conn.sendBody(200, "OK", "application/json", "{\"status\":\"ok\"}") catch return;
            self.served_during_generation += 1;
            return;
        }
        if (std.mem.eql(u8, req.method, "GET") and std.mem.eql(u8, path, "/v1/models")) {
            self.sendModels(&conn) catch return;
            self.served_during_generation += 1;
            return;
        }

        // Anything that needs the engine cannot be served while it is busy.
        // Dropping the connection here left the client with an empty reply and no
        // status at all, which looks like a crash; answer with a real HTTP error
        // and Retry-After so a client can tell the difference and back off.
        self.busy_rejections += 1;
        tryBusyResponse(&conn, self.busy_rejections);
    }

    /// Close a connection parked between tokens, at shutdown.
    pub fn closePending(self: *Server) void {
        if (self.pending) |*c| c.close();
        self.pending = null;
    }

    /// One line per request so multi-turn prefix reuse is observable in the
    /// server log.
    fn logStats(self: *Server, stats: generate.Stats) void {
        sys.eprint("[req] prompt {d} ({d} reused, {d} new), +{d} tokens, prefill {d:.2} s, decode {d:.1} tok/s, {s}\n", .{
            stats.prompt_tokens,
            stats.prefill_reused,
            stats.prompt_tokens -| stats.prefill_reused,
            stats.completion_tokens,
            @as(f64, @floatFromInt(stats.prefill_ns)) / 1e9,
            stats.decodeToksPerSec(),
            stats.stop_reason.toString(),
        });
        if (stats.prompt_tokens_dropped > 0) {
            sys.eprint("[req] WARNING: dropped {d} leading prompt tokens ({d} sent, {d} used) to fit the {d}-token context\n", .{
                stats.prompt_tokens_dropped,
                stats.prompt_tokens_sent,
                stats.prompt_tokens,
                self.context_limit,
            });
        }
    }

    // ------------------------------------------------------------ OpenAI legacy

    fn completions(self: *Server, conn: *http.Conn, a: std.mem.Allocator, body: []const u8) !void {
        const parsed = std.json.parseFromSliceLeaky(std.json.Value, a, body, .{}) catch
            return sendError(conn, 400, "invalid JSON body");
        if (parsed != .object) return sendError(conn, 400, "body must be a JSON object");
        const root = parsed.object;

        const model_name = getModelName(root, self.opts.model_name);
        const prompt = getString(root, "prompt") orelse return sendError(conn, 400, "prompt is required");
        const max_tokens: u32 = intToU32(getInt(root, "max_tokens", self.opts.default_max_tokens), 1);
        const temperature: f32 = getFloat(root, "temperature", 0.7);
        const stream = getBool(root, "stream", false);
        const stops = try parseStops(a, root);

        const ids = try self.session.tokenizer.encode(a, prompt, true);
        if (ids.len == 0) return sendError(conn, 400, "prompt encoded to zero tokens");
        const params = generate.Params{
            .max_tokens = max_tokens,
            .sampler = .{
                .temperature = temperature,
                .top_p = getFloat(root, "top_p", self.opts.default_top_p),
                .repetition_penalty = getFloat(root, "repetition_penalty", self.opts.default_repetition_penalty),
            },
        };
        const created = sys.unixTime();

        if (!stream) {
            var collector = Collector.init(self.allocator);
            collector.server = self;
            defer collector.deinit();
            const stats = try self.session.generate(ids, params, .{ .ctx = &collector, .func = collectEmit });

            var b = Buf.init(self.allocator);
            defer b.deinit();
            try b.print("{{\"id\":\"cmpl-{d}\",\"object\":\"text_completion\",\"created\":{d},\"model\":\"", .{ created, created });
            try jsonString(&b, model_name);
            try b.print("\",\"choices\":[{{\"text\":\"", .{});
            try jsonString(&b, collector.text.items);
            try b.print("\",\"index\":0,\"finish_reason\":\"{s}\"}}],\"usage\":{{\"prompt_tokens\":{d},\"completion_tokens\":{d},\"total_tokens\":{d},\"prompt_tokens_dropped\":{d}}}}}", .{
                finishReason(stats.stop_reason), stats.prompt_tokens, stats.completion_tokens, stats.prompt_tokens + stats.completion_tokens, stats.prompt_tokens_dropped,
            });
            return conn.sendBody(200, "OK", "application/json", b.slice());
        }

        // Prefill happens before any token is emitted, and for a long prompt it
        // is seconds long, so hook the server into the chunk loop.
        self.session.engine.prefill_tick = prefillTick;
        self.session.engine.prefill_tick_ctx = self;
        defer {
            self.session.engine.prefill_tick = null;
            self.session.engine.prefill_tick_ctx = null;
        }

        try conn.beginStream(200, "OK", "text/event-stream");
        var sink = SseSink.init(self.allocator, conn, .openai_legacy, model_name, created);
        sink.server = self;
        defer sink.deinit();
        sink.setStops(stops);
        _ = try self.session.generate(ids, params, .{ .ctx = &sink, .func = sseEmit });
        if (!conn.alive) return;
        return sink.finishOpenAiLegacy();
    }

    // ------------------------------------------------------------ Anthropic

    fn anthropicMessages(self: *Server, conn: *http.Conn, a: std.mem.Allocator, body: []const u8) !void {
        const parsed = std.json.parseFromSliceLeaky(std.json.Value, a, body, .{}) catch
            return sendError(conn, 400, "invalid JSON body");
        if (parsed != .object) return sendError(conn, 400, "body must be a JSON object");
        const root = parsed.object;

        const model_name = getModelName(root, self.opts.model_name);
        const max_tokens: u32 = intToU32(getInt(root, "max_tokens", self.opts.default_max_tokens), 1);
        const temperature: f32 = getFloat(root, "temperature", 0.7);
        const stream = getBool(root, "stream", false);
        const system = getString(root, "system");
        const stops = try parseStops(a, root);

        const messages = (try parseMessages(a, root)) orelse
            return sendError(conn, 400, "messages must be a non-empty array of {role, content}");
        if (messages.len == 0) return sendError(conn, 400, "messages must be non-empty");

        const prompt = try self.session.formatChat(a, messages, system orelse self.opts.default_system);
        const ids = try self.session.tokenizer.encode(a, prompt, true);
        if (ids.len == 0) return sendError(conn, 400, "prompt encoded to zero tokens");

        const params = generate.Params{
            .max_tokens = max_tokens,
            .sampler = .{
                .temperature = temperature,
                .top_p = getFloat(root, "top_p", self.opts.default_top_p),
                .repetition_penalty = getFloat(root, "repetition_penalty", self.opts.default_repetition_penalty),
            },
        };
        const created = sys.unixTime();
        const msg_id = try std.fmt.allocPrint(a, "msg_{d}", .{created});

        if (!stream) {
            var collector = Collector.init(self.allocator);
            collector.server = self;
            defer collector.deinit();
            const stats = try self.session.generate(ids, params, .{ .ctx = &collector, .func = collectEmit });

            var b = Buf.init(self.allocator);
            defer b.deinit();
            try b.print("{{\"id\":\"{s}\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"", .{msg_id});
            try jsonString(&b, model_name);
            try b.print("\",\"content\":[{{\"type\":\"text\",\"text\":\"", .{});
            try jsonString(&b, collector.text.items);
            try b.print("\"}}],\"stop_reason\":\"{s}\",\"stop_sequence\":null,\"usage\":{{\"input_tokens\":{d},\"output_tokens\":{d}}}}}", .{
                anthropicStop(stats.stop_reason), stats.prompt_tokens, stats.completion_tokens,
            });
            return conn.sendBody(200, "OK", "application/json", b.slice());
        }

        // Prefill happens before any token is emitted, and for a long prompt it
        // is seconds long, so hook the server into the chunk loop.
        self.session.engine.prefill_tick = prefillTick;
        self.session.engine.prefill_tick_ctx = self;
        defer {
            self.session.engine.prefill_tick = null;
            self.session.engine.prefill_tick_ctx = null;
        }

        try conn.beginStream(200, "OK", "text/event-stream");
        var sink = SseSink.init(self.allocator, conn, .anthropic, model_name, created);
        sink.server = self;
        defer sink.deinit();
        sink.message_id = msg_id;
        sink.setStops(stops);
        try sink.anthropicPrelude();
        const stats = try self.session.generate(ids, params, .{ .ctx = &sink, .func = sseEmit });
        if (!conn.alive) return;
        return sink.finishAnthropic(stats);
    }
};

// ---------------------------------------------------------------- helpers

fn sendError(conn: *http.Conn, status: u16, message: []const u8) !void {
    var b: [512]u8 = undefined;
    const body = std.fmt.bufPrint(&b, "{{\"error\":{{\"message\":\"{s}\",\"type\":\"invalid_request_error\"}}}}", .{message}) catch return;
    const reason = if (status == 404) "Not Found" else "Bad Request";
    try conn.sendBody(status, reason, "application/json", body);
}

fn finishReason(r: generate.StopReason) []const u8 {
    return switch (r) {
        .stop => "stop",
        .length => "length",
        .abort => "stop",
    };
}

fn anthropicStop(r: generate.StopReason) []const u8 {
    return switch (r) {
        .stop => "end_turn",
        .length => "max_tokens",
        .abort => "end_turn",
    };
}

fn intToU32(v: i64, min: u32) u32 {
    if (v < @as(i64, min)) return min;
    if (v > std.math.maxInt(u32)) return std.math.maxInt(u32);
    return @intCast(v);
}

/// Like getString, but treats "" as absent so a client that sends an empty
/// model id still gets the server's default name echoed back.
fn getModelName(obj: std.json.ObjectMap, fallback: []const u8) []const u8 {
    const s = getString(obj, "model") orelse return fallback;
    return if (s.len == 0) fallback else s;
}

fn getString(obj: std.json.ObjectMap, key: []const u8) ?[]const u8 {
    const v = obj.get(key) orelse return null;
    return switch (v) {
        .string => |s| s,
        else => null,
    };
}

fn getInt(obj: std.json.ObjectMap, key: []const u8, default: i64) i64 {
    const v = obj.get(key) orelse return default;
    return switch (v) {
        .integer => |i| i,
        .float => |f| @intFromFloat(f),
        else => default,
    };
}

fn getFloat(obj: std.json.ObjectMap, key: []const u8, default: f32) f32 {
    const v = obj.get(key) orelse return default;
    return switch (v) {
        .integer => |i| @floatFromInt(i),
        .float => |f| @floatCast(f),
        else => default,
    };
}

fn getBool(obj: std.json.ObjectMap, key: []const u8, default: bool) bool {
    const v = obj.get(key) orelse return default;
    return switch (v) {
        .bool => |b| b,
        else => default,
    };
}

/// `stop` may be a string or an array of strings.
fn parseStops(a: std.mem.Allocator, obj: std.json.ObjectMap) ![]const []const u8 {
    const v = obj.get("stop") orelse return &.{};
    switch (v) {
        .string => |s| {
            const out = try a.alloc([]const u8, 1);
            out[0] = s;
            return out;
        },
        .array => |arr| {
            var list: std.ArrayList([]const u8) = .empty;
            for (arr.items) |item| {
                if (item == .string) try list.append(a, item.string);
            }
            return try list.toOwnedSlice(a);
        },
        else => return &.{},
    }
}

/// Message content is either a string or an array of content blocks.
fn contentToString(a: std.mem.Allocator, v: std.json.Value) !?[]const u8 {
    switch (v) {
        .string => |s| return s,
        .array => |arr| {
            var out: std.ArrayList(u8) = .empty;
            for (arr.items) |block| {
                switch (block) {
                    .object => |o| {
                        if (getString(o, "text")) |t| {
                            if (out.items.len > 0) try out.append(a, '\n');
                            try out.appendSlice(a, t);
                        }
                    },
                    .string => |s| try out.appendSlice(a, s),
                    else => {},
                }
            }
            return try out.toOwnedSlice(a);
        },
        else => return null,
    }
}

fn parseMessages(a: std.mem.Allocator, root: std.json.ObjectMap) !?[]generate.Message {
    const v = root.get("messages") orelse return null;
    if (v != .array) return null;
    var list: std.ArrayList(generate.Message) = .empty;
    for (v.array.items) |item| {
        if (item != .object) continue;
        const o = item.object;
        const role_s = getString(o, "role") orelse continue;
        const role = generate.Role.fromString(role_s) orelse continue;
        const content_v = o.get("content") orelse continue;
        const content = (try contentToString(a, content_v)) orelse continue;
        try list.append(a, .{ .role = role, .content = content });
    }
    return try list.toOwnedSlice(a);
}

/// Escape `s` as a JSON string body into `b`.
fn jsonString(b: *Buf, s: []const u8) !void {
    for (s) |ch| {
        switch (ch) {
            '"' => try b.appendSlice("\\\""),
            '\\' => try b.appendSlice("\\\\"),
            '\n' => try b.appendSlice("\\n"),
            '\r' => try b.appendSlice("\\r"),
            '\t' => try b.appendSlice("\\t"),
            8 => try b.appendSlice("\\b"),
            12 => try b.appendSlice("\\f"),
            else => {
                if (ch < 0x20) {
                    try b.print("\\u{x:0>4}", .{ch});
                } else {
                    try b.append(ch);
                }
            },
        }
    }
}

/// Batched writer that escapes JSON strings straight onto the socket.
const JsonSink = struct {
    conn: *http.Conn,
    buf: [4096]u8 = undefined,
    len: usize = 0,

    fn flush(self: *JsonSink) !void {
        if (self.len == 0) return;
        try self.conn.write(self.buf[0..self.len]);
        self.len = 0;
    }
    fn write(self: *JsonSink, bytes: []const u8) !void {
        if (bytes.len > self.buf.len - self.len) try self.flush();
        if (bytes.len > self.buf.len) return self.conn.write(bytes);
        @memcpy(self.buf[self.len..][0..bytes.len], bytes);
        self.len += bytes.len;
    }
    fn print(self: *JsonSink, comptime fmt: []const u8, args: anytype) !void {
        var tmp: [2048]u8 = undefined;
        const s = std.fmt.bufPrint(&tmp, fmt, args) catch return error.WriteFailed;
        try self.write(s);
    }
    fn string(self: *JsonSink, s: []const u8) !void {
        for (s) |ch| {
            switch (ch) {
                '"' => try self.write("\\\""),
                '\\' => try self.write("\\\\"),
                '\n' => try self.write("\\n"),
                '\r' => try self.write("\\r"),
                '\t' => try self.write("\\t"),
                8 => try self.write("\\b"),
                12 => try self.write("\\f"),
                else => {
                    if (ch < 0x20) {
                        var tmp: [8]u8 = undefined;
                        const esc = std.fmt.bufPrint(&tmp, "\\u{x:0>4}", .{ch}) catch continue;
                        try self.write(esc);
                    } else {
                        try self.write(&[_]u8{ch});
                    }
                },
            }
        }
    }
};

/// Accumulates a whole completion for non-streaming responses.
const Collector = struct {
    allocator: std.mem.Allocator,
    text: std.ArrayList(u8) = .empty,
    /// Set for routes that should service liveness probes mid-generation.
    server: ?*Server = null,

    fn init(allocator: std.mem.Allocator) Collector {
        return .{ .allocator = allocator };
    }
    fn deinit(self: *Collector) void {
        self.text.deinit(self.allocator);
    }
};

fn collectEmit(ctx: ?*anyopaque, piece: []const u8, token_id: u32) bool {
    _ = token_id;
    const c: *Collector = @ptrCast(@alignCast(ctx.?));
    if (c.server) |srv| srv.servicePending();
    c.text.appendSlice(c.allocator, piece) catch return false;
    return true;
}

// ---------------------------------------------------------------- SSE

const Style = enum { openai, openai_legacy, anthropic };

/// Decides how much of `pending` is safe to emit.
///
/// With stop sequences configured we must never emit a prefix of one, so the
/// last `holdback` bytes are retained until more text arrives (or the stream
/// ends). Returns the emit slice plus whether a stop sequence was found.
const HoldbackResult = struct { emit: []const u8, hit_stop: bool };

fn holdbackStep(pending: []const u8, stops: []const []const u8, holdback: usize) HoldbackResult {
    for (stops) |s| {
        if (std.mem.indexOf(u8, pending, s)) |idx| {
            return .{ .emit = pending[0..idx], .hit_stop = true };
        }
    }
    if (pending.len > holdback) {
        return .{ .emit = pending[0 .. pending.len - holdback], .hit_stop = false };
    }
    return .{ .emit = pending[0..0], .hit_stop = false };
}

/// Frames generated text as Server-Sent Events, with stop-sequence holdback so
/// a stop string is never emitted partially.
const SseSink = struct {
    allocator: std.mem.Allocator,
    conn: *http.Conn,
    style: Style,
    model: []const u8,
    created: i64,
    message_id: []const u8 = "",
    /// Set for streaming routes so the emitter can service liveness probes.
    server: ?*Server = null,
    pending: std.ArrayList(u8) = .empty,
    stops: []const []const u8 = &.{},
    holdback: usize = 0,
    hit_stop: bool = false,

    fn init(allocator: std.mem.Allocator, conn: *http.Conn, style: Style, model: []const u8, created: i64) SseSink {
        return .{ .allocator = allocator, .conn = conn, .style = style, .model = model, .created = created };
    }

    fn deinit(self: *SseSink) void {
        self.pending.deinit(self.allocator);
    }

    fn setStops(self: *SseSink, stops: []const []const u8) void {
        self.stops = stops;
        var max: usize = 0;
        for (stops) |s| max = @max(max, s.len);
        self.holdback = if (max > 0) max - 1 else 0;
    }

    fn frameText(self: *SseSink, text: []const u8) !void {
        var js = JsonSink{ .conn = self.conn };
        switch (self.style) {
            .openai => {
                try js.print("data: {{\"id\":\"chatcmpl-{d}\",\"object\":\"chat.completion.chunk\",\"created\":{d},\"model\":\"", .{ self.created, self.created });
                try js.string(self.model);
                try js.write("\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"");
                try js.string(text);
                try js.write("\"},\"finish_reason\":null}]}\n\n");
            },
            .openai_legacy => {
                try js.print("data: {{\"id\":\"cmpl-{d}\",\"object\":\"text_completion\",\"created\":{d},\"model\":\"", .{ self.created, self.created });
                try js.string(self.model);
                try js.write("\",\"choices\":[{\"text\":\"");
                try js.string(text);
                try js.write("\",\"index\":0,\"finish_reason\":null}]}\n\n");
            },
            .anthropic => {
                try js.write("event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"");
                try js.string(text);
                try js.write("\"}}\n\n");
            },
        }
        try js.flush();
    }

    /// Returns false when a stop sequence was hit (generation should end).
    fn push(self: *SseSink, piece: []const u8) !bool {
        try self.pending.appendSlice(self.allocator, piece);
        const step = holdbackStep(self.pending.items, self.stops, self.holdback);
        if (step.emit.len > 0) try self.frameText(step.emit);
        if (step.hit_stop) {
            self.hit_stop = true;
            self.pending.clearRetainingCapacity();
            return false;
        }
        const keep = self.pending.items.len - step.emit.len;
        std.mem.copyForwards(u8, self.pending.items[0..keep], self.pending.items[step.emit.len..]);
        self.pending.shrinkRetainingCapacity(keep);
        return true;
    }

    fn flushPending(self: *SseSink) !void {
        if (self.pending.items.len > 0 and !self.hit_stop) {
            try self.frameText(self.pending.items);
            self.pending.clearRetainingCapacity();
        }
    }

    fn finishOpenAi(self: *SseSink, stats: generate.Stats) !void {
        try self.flushPending();
        var js = JsonSink{ .conn = self.conn };
        try js.print("data: {{\"id\":\"chatcmpl-{d}\",\"object\":\"chat.completion.chunk\",\"created\":{d},\"model\":\"", .{ self.created, self.created });
        try js.string(self.model);
        // The final chunk carries usage, which is where a client learns that its
        // prompt was truncated (prompt_tokens_dropped > 0) instead of having to
        // infer it from a token count smaller than what it sent.
        try js.print("\",\"choices\":[{{\"index\":0,\"delta\":{{}},\"finish_reason\":\"{s}\"}}],\"usage\":{{\"prompt_tokens\":{d},\"completion_tokens\":{d},\"total_tokens\":{d},\"prompt_tokens_dropped\":{d}}}}}\n\n", .{
            finishReason(stats.stop_reason),
            stats.prompt_tokens,
            stats.completion_tokens,
            stats.prompt_tokens + stats.completion_tokens,
            stats.prompt_tokens_dropped,
        });
        try js.write("data: [DONE]\n\n");
        try js.flush();
    }

    fn finishOpenAiLegacy(self: *SseSink) !void {
        try self.flushPending();
        var js = JsonSink{ .conn = self.conn };
        try js.write("data: [DONE]\n\n");
        try js.flush();
    }

    fn anthropicPrelude(self: *SseSink) !void {
        var js = JsonSink{ .conn = self.conn };
        try js.write("event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"");
        try js.string(self.message_id);
        try js.write("\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"");
        try js.string(self.model);
        try js.write("\",\"content\":[],\"stop_reason\":null,\"usage\":{\"input_tokens\":0,\"output_tokens\":0}}}\n\n");
        try js.write("event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n");
        try js.flush();
    }

    fn finishAnthropic(self: *SseSink, stats: generate.Stats) !void {
        try self.flushPending();
        var js = JsonSink{ .conn = self.conn };
        try js.write("event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n");
        try js.print("event: message_delta\ndata: {{\"type\":\"message_delta\",\"delta\":{{\"stop_reason\":\"{s}\",\"stop_sequence\":null}},\"usage\":{{\"input_tokens\":{d},\"output_tokens\":{d}}}}}\n\n", .{
            anthropicStop(stats.stop_reason), stats.prompt_tokens, stats.completion_tokens,
        });
        try js.write("event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n");
        try js.flush();
    }
};

fn prefillTick(ctx: ?*anyopaque) void {
    const self: *Server = @ptrCast(@alignCast(ctx orelse return));
    self.servicePending();
}

/// 503 with Retry-After for a request that needs the busy engine.
///
/// Retry-After is what makes this a usable answer rather than a mystery: a
/// client can back off instead of treating the empty reply as a broken server.
fn tryBusyResponse(conn: *http.Conn, so_far: u64) void {
    const body = "{\"error\":{\"message\":\"another generation is in progress; retry shortly\",\"type\":\"server_busy\"}}";
    conn.beginWithLength(503, "Service Unavailable", "application/json", body.len) catch return;
    conn.write("Retry-After: 1\r\n") catch return;
    conn.write("\r\n") catch return;
    conn.write(body) catch return;
    sys.eprint("[busy] rejected a request while generating ({d} so far)\n", .{so_far});
}

fn sseEmit(ctx: ?*anyopaque, piece: []const u8, token_id: u32) bool {
    _ = token_id;
    const sink: *SseSink = @ptrCast(@alignCast(ctx.?));
    if (sink.server) |srv| srv.servicePending();
    return sink.push(piece) catch {
        sink.conn.alive = false;
        return false;
    };
}

test "json string escaping" {
    const a = std.testing.allocator;
    var b = Buf.init(a);
    defer b.deinit();
    try jsonString(&b, "a\"b\\c\nd\te");
    try std.testing.expectEqualStrings("a\\\"b\\\\c\\nd\\te", b.slice());
}

test "stop-sequence holdback never leaks a partial stop string" {
    const stops = [_][]const u8{"three"};
    const holdback = 4; // max stop length - 1

    // Feed the text in token-sized pieces; collect everything that gets framed.
    var framed: std.ArrayList(u8) = .empty;
    defer framed.deinit(std.testing.allocator);
    var pending: std.ArrayList(u8) = .empty;
    defer pending.deinit(std.testing.allocator);

    const pieces = [_][]const u8{ "one ", "two ", "th", "ree", " four" };
    var stopped = false;
    for (pieces) |p| {
        try pending.appendSlice(std.testing.allocator, p);
        const step = holdbackStep(pending.items, &stops, holdback);
        try framed.appendSlice(std.testing.allocator, step.emit);
        if (step.hit_stop) {
            stopped = true;
            pending.clearRetainingCapacity();
            break;
        }
        const keep = pending.items.len - step.emit.len;
        std.mem.copyForwards(u8, pending.items[0..keep], pending.items[step.emit.len..]);
        pending.shrinkRetainingCapacity(keep);
    }
    try std.testing.expect(stopped);
    try std.testing.expectEqualStrings("one two ", framed.items);
    // The stop string itself must never have been emitted.
    try std.testing.expect(std.mem.indexOf(u8, framed.items, "three") == null);
}

test "holdback with no stop sequences emits everything" {
    const step = holdbackStep("hello world", &.{}, 0);
    try std.testing.expectEqualStrings("hello world", step.emit);
    try std.testing.expect(!step.hit_stop);
}

test "holdback at the very start emits nothing" {
    const stops = [_][]const u8{"STOP"};
    const step = holdbackStep("STOP rest", &stops, 3);
    try std.testing.expectEqualStrings("", step.emit);
    try std.testing.expect(step.hit_stop);
}

test "finish reasons map to each API's vocabulary" {
    try std.testing.expectEqualStrings("length", finishReason(.length));
    try std.testing.expectEqualStrings("stop", finishReason(.abort));
    try std.testing.expectEqualStrings("max_tokens", anthropicStop(.length));
    try std.testing.expectEqualStrings("end_turn", anthropicStop(.stop));
}
