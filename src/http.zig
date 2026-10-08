// http.zig — a minimal HTTP/1.1 server on libc sockets.
//
// Zig 0.17's std.http/std.net sit behind the new std.Io interface, so this
// module talks to the BSD sockets API directly. It implements only what the
// API server needs: one request per connection, `Connection: close`, no
// chunked transfer encoding (streaming responses simply end at EOF, which
// EventSource/fetch handle fine).

const std = @import("std");

// ---------------------------------------------------------------- libc

/// The libc functions are namespaced so the Zig names stay readable while the
/// linker still sees the real symbols (`accept`, not `c_accept`).
const libc = struct {
    extern "c" fn socket(domain: c_uint, sock_type: c_uint, protocol: c_uint) c_int;
    extern "c" fn setsockopt(fd: c_int, level: c_int, optname: c_int, optval: *const anyopaque, optlen: std.c.socklen_t) c_int;
    extern "c" fn bind(fd: c_int, addr: *const std.c.sockaddr, len: std.c.socklen_t) c_int;
    extern "c" fn listen(fd: c_int, backlog: c_int) c_int;
    extern "c" fn accept(fd: c_int, addr: ?*std.c.sockaddr, len: ?*std.c.socklen_t) c_int;
    extern "c" fn fcntl(fd: c_int, cmd: c_int, ...) c_int;
    extern "c" fn connect(fd: c_int, addr: *const std.c.sockaddr, len: std.c.socklen_t) c_int;
    extern "c" fn poll(fds: [*]std.c.pollfd, n: std.c.nfds_t, timeout_ms: c_int) c_int;
    extern "c" fn socketpair(domain: c_int, typ: c_int, protocol: c_int, sv: *[2]std.c.fd_t) c_int;
    extern "c" fn usleep(usec: c_uint) c_int;
    extern "c" fn recv(fd: c_int, buf: [*]u8, len: usize, flags: c_int) isize;
    extern "c" fn send(fd: c_int, buf: [*]const u8, len: usize, flags: c_int) isize;
    extern "c" fn shutdown(fd: c_int, how: c_int) c_int;
    extern "c" fn signal(sig: c_int, handler: usize) usize;
};

const AF_INET: c_uint = 2;
const SOCK_STREAM: c_uint = 1;
const SOL_SOCKET: c_int = 0xffff;
const SO_REUSEADDR: c_int = 0x0004;
const SO_NOSIGPIPE: c_int = 0x1022;
const F_GETFL: c_int = 3;
const F_SETFL: c_int = 4;
const O_NONBLOCK: c_int = 0x0004;
const SO_RCVTIMEO: c_int = 0x1006;
const SO_SNDTIMEO: c_int = 0x1005;
const SHUT_WR: c_int = 1;
const SIGPIPE: c_int = 13;
const SIG_IGN: usize = 1;
/// A stalled client must not wedge the single-threaded server.
const RECV_TIMEOUT_S: c_long = 30;
const POLLIN: c_short = 0x0001;
const SEND_TIMEOUT_S: c_long = 60;

pub const Error = error{
    SocketFailed,
    BindFailed,
    ListenFailed,
    AcceptFailed,
    ConnectionClosed,
    RequestTooLarge,
    MalformedRequest,
    WriteFailed,
};

pub fn ignoreSigpipe() void {
    _ = libc.signal(SIGPIPE, SIG_IGN);
}

/// Parse "a.b.c.d" into the u32 layout the sockaddr expects (network order).
fn parseIpv4(host: []const u8) !u32 {
    var it = std.mem.splitScalar(u8, host, '.');
    var parts: [4]u8 = undefined;
    var n: usize = 0;
    while (it.next()) |p| {
        if (n == 4) return error.MalformedRequest;
        parts[n] = std.fmt.parseInt(u8, p, 10) catch return error.MalformedRequest;
        n += 1;
    }
    if (n != 4) return error.MalformedRequest;
    return @as(u32, parts[0]) | (@as(u32, parts[1]) << 8) | (@as(u32, parts[2]) << 16) | (@as(u32, parts[3]) << 24);
}

pub const Server = struct {
    fd: c_int,

    pub fn open(host: []const u8, port: u16) !Server {
        ignoreSigpipe();
        const fd = libc.socket(AF_INET, SOCK_STREAM, 0);
        if (fd < 0) return Error.SocketFailed;
        errdefer _ = std.c.close(fd);

        const one: c_int = 1;
        _ = libc.setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, @ptrCast(&one), @sizeOf(c_int));

        var addr: std.c.sockaddr.in = .{
            .port = @byteSwap(port),
            .addr = try parseIpv4(host),
        };
        if (libc.bind(fd, @ptrCast(&addr), @sizeOf(std.c.sockaddr.in)) != 0) return Error.BindFailed;
        if (libc.listen(fd, 16) != 0) return Error.ListenFailed;
        return .{ .fd = fd };
    }

    pub fn accept(self: *Server) !Conn {
        const fd = libc.accept(self.fd, null, null);
        if (fd < 0) return Error.AcceptFailed;
        return self.finishAccept(fd);
    }

    /// Non-blocking accept: returns null immediately when no client is waiting.
    /// Used to service liveness probes while a generation is in flight.
    pub fn acceptIfPending(self: *Server) ?Conn {
        const fd = libc.accept(self.fd, null, null);
        if (fd < 0) return null;
        return self.finishAccept(fd) catch null;
    }

    fn finishAccept(self: *Server, fd: c_int) !Conn {
        _ = self;
        const one: c_int = 1;
        _ = libc.setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, @ptrCast(&one), @sizeOf(c_int));
        const rcv = std.c.timeval{ .sec = RECV_TIMEOUT_S, .usec = 0 };
        _ = libc.setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, @ptrCast(&rcv), @sizeOf(std.c.timeval));
        const snd = std.c.timeval{ .sec = SEND_TIMEOUT_S, .usec = 0 };
        _ = libc.setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, @ptrCast(&snd), @sizeOf(std.c.timeval));
        // The accepted socket can inherit O_NONBLOCK from the listener; clear it
        // so recv() waits for a request body the way the parser expects.
        const flags = libc.fcntl(fd, F_GETFL);
        if (flags >= 0 and (flags & O_NONBLOCK) != 0) {
            _ = libc.fcntl(fd, F_SETFL, flags & ~O_NONBLOCK);
        }
        return .{ .fd = fd };
    }

    /// Toggle O_NONBLOCK on the LISTENER (affects acceptIfPending only).
    pub fn setNonBlocking(self: *Server, enabled: bool) void {
        const flags = libc.fcntl(self.fd, F_GETFL);
        if (flags < 0) return;
        const next = if (enabled) flags | O_NONBLOCK else flags & ~O_NONBLOCK;
        _ = libc.fcntl(self.fd, F_SETFL, next);
    }

    pub fn close(self: *Server) void {
        _ = std.c.close(self.fd);
        self.fd = -1;
    }
};

pub const Request = struct {
    method: []const u8,
    target: []const u8,
    version: []const u8,
    raw_headers: []const u8,
    body: []const u8,

    /// Case-insensitive header lookup.
    pub fn header(self: *const Request, name: []const u8) ?[]const u8 {
        var it = std.mem.splitSequence(u8, self.raw_headers, "\r\n");
        _ = it.next(); // request line
        while (it.next()) |line| {
            const colon = std.mem.indexOfScalar(u8, line, ':') orelse continue;
            const k = std.mem.trim(u8, line[0..colon], " \t");
            if (std.ascii.eqlIgnoreCase(k, name)) {
                return std.mem.trim(u8, line[colon + 1 ..], " \t");
            }
        }
        return null;
    }

    /// Path without the query string.
    pub fn path(self: *const Request) []const u8 {
        const q = std.mem.indexOfScalar(u8, self.target, '?') orelse return self.target;
        return self.target[0..q];
    }
};

pub const Conn = struct {
    fd: c_int,
    /// Set to false as soon as a write fails, so the caller can stop work.
    alive: bool = true,
    headers_sent: bool = false,

    /// Read one request into `buf` (which must outlive the returned Request).
    pub fn readRequest(self: *Conn, allocator: std.mem.Allocator, buf: *std.ArrayList(u8)) !Request {
        const max_request: usize = 4 << 20; // 4 MiB is plenty for a chat prompt
        var chunk: [16384]u8 = undefined;
        var header_end: ?usize = null;

        while (header_end == null) {
            const n = libc.recv(self.fd, &chunk, chunk.len, 0);
            if (n <= 0) return Error.ConnectionClosed;
            try buf.appendSlice(allocator, chunk[0..@intCast(n)]);
            if (buf.items.len > max_request) return Error.RequestTooLarge;
            header_end = std.mem.indexOf(u8, buf.items, "\r\n\r\n");
        }
        const hdr_len = header_end.? + 4;

        const first_line_end = std.mem.indexOf(u8, buf.items, "\r\n") orelse return Error.MalformedRequest;
        const line = buf.items[0..first_line_end];
        var parts = std.mem.tokenizeScalar(u8, line, ' ');
        const method = parts.next() orelse return Error.MalformedRequest;
        const target = parts.next() orelse return Error.MalformedRequest;
        const version = parts.next() orelse "HTTP/1.1";

        const raw_headers = buf.items[0..hdr_len];

        // Content-Length body (no chunked support: clients we care about send
        // a Content-Length for JSON POSTs).
        var want_body: usize = 0;
        if (findHeaderIn(raw_headers, "content-length")) |v| {
            want_body = std.fmt.parseInt(usize, std.mem.trim(u8, v, " \t"), 10) catch return Error.MalformedRequest;
            if (want_body > max_request) return Error.RequestTooLarge;
        }
        while (buf.items.len < hdr_len + want_body) {
            const n = libc.recv(self.fd, &chunk, chunk.len, 0);
            if (n <= 0) return Error.ConnectionClosed;
            try buf.appendSlice(allocator, chunk[0..@intCast(n)]);
        }
        return .{
            .method = method,
            .target = target,
            .version = version,
            .raw_headers = raw_headers,
            .body = buf.items[hdr_len .. hdr_len + want_body],
        };
    }

    fn findHeaderIn(raw: []const u8, name: []const u8) ?[]const u8 {
        var it = std.mem.splitSequence(u8, raw, "\r\n");
        _ = it.next();
        while (it.next()) |line| {
            const colon = std.mem.indexOfScalar(u8, line, ':') orelse continue;
            const k = std.mem.trim(u8, line[0..colon], " \t");
            if (std.ascii.eqlIgnoreCase(k, name)) return std.mem.trim(u8, line[colon + 1 ..], " \t");
        }
        return null;
    }

    pub fn write(self: *Conn, bytes: []const u8) !void {
        if (!self.alive) return Error.WriteFailed;
        var off: usize = 0;
        while (off < bytes.len) {
            const n = libc.send(self.fd, bytes.ptr + off, bytes.len - off, 0);
            if (n <= 0) {
                self.alive = false;
                return Error.WriteFailed;
            }
            off += @intCast(n);
        }
    }

    pub fn print(self: *Conn, comptime fmt: []const u8, args: anytype) !void {
        var buf: [8192]u8 = undefined;
        const s = std.fmt.bufPrint(&buf, fmt, args) catch return Error.WriteFailed;
        try self.write(s);
    }

    /// Start a response. Adds the CORS + connection headers every route needs.
    pub fn begin(self: *Conn, status: u16, reason: []const u8, content_type: []const u8) !void {
        try self.print("HTTP/1.1 {d} {s}\r\n", .{ status, reason });
        try self.print("Content-Type: {s}\r\n", .{content_type});
        try self.write("Access-Control-Allow-Origin: *\r\n");
        try self.write("Access-Control-Allow-Headers: *\r\n");
        try self.write("Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n");
        try self.write("Connection: close\r\n");
        try self.write("Cache-Control: no-store\r\n");
    }

    /// Begin a response whose length is known.
    pub fn beginWithLength(self: *Conn, status: u16, reason: []const u8, content_type: []const u8, len: usize) !void {
        try self.begin(status, reason, content_type);
        try self.print("Content-Length: {d}\r\n\r\n", .{len});
        self.headers_sent = true;
    }

    /// Begin a streamed response (no Content-Length; the body ends at EOF).
    pub fn beginStream(self: *Conn, status: u16, reason: []const u8, content_type: []const u8) !void {
        try self.begin(status, reason, content_type);
        try self.write("\r\n");
        self.headers_sent = true;
    }

    pub fn sendBody(self: *Conn, status: u16, reason: []const u8, content_type: []const u8, body: []const u8) !void {
        try self.beginWithLength(status, reason, content_type, body.len);
        try self.write(body);
    }

    /// True when at least one byte is ready to read, without blocking.
    ///
    /// Used by the server's mid-generation hook: a client that connects and then
    /// sends nothing must not stall the generation it is sharing the process
    /// with. Measured before the fix: a silent client holding a connection for
    /// 6 s added exactly 6 s to a concurrent generation's wall time.
    pub fn hasPendingInput(self: *const Conn, timeout_ms: c_int) bool {
        var fds = [_]std.c.pollfd{.{ .fd = self.fd, .events = POLLIN, .revents = 0 }};
        const n = libc.poll(&fds, 1, timeout_ms);
        if (n <= 0) return false;
        return (fds[0].revents & POLLIN) != 0;
    }

    pub fn close(self: *Conn) void {
        if (self.fd >= 0) {
            _ = libc.shutdown(self.fd, SHUT_WR);
            _ = std.c.close(self.fd);
            self.fd = -1;
        }
    }
};

/// Parse a raw request through the same code path the server uses, without a
/// socket: this is what makes the parser testable at all.
///
/// A socket-based round-trip test lived here briefly and was flaky (the accept
/// and recv sides race), so the parsing contract is asserted directly instead.
fn parseForTest(a: std.mem.Allocator, raw: []const u8) !Request {
    const hdr_len = std.mem.indexOf(u8, raw, "\r\n\r\n") orelse return error.MalformedRequest;
    const first_line_end = std.mem.indexOf(u8, raw, "\r\n") orelse return error.MalformedRequest;
    var parts = std.mem.tokenizeScalar(u8, raw[0..first_line_end], ' ');
    const method = parts.next() orelse return error.MalformedRequest;
    const target = parts.next() orelse return error.MalformedRequest;
    const version = parts.next() orelse "HTTP/1.1";
    const raw_headers = raw[0 .. hdr_len + 4];
    var want: usize = 0;
    const probe = Request{
        .method = method,
        .target = target,
        .version = version,
        .raw_headers = raw_headers,
        .body = "",
    };
    if (probe.header("content-length")) |v| {
        want = std.fmt.parseInt(usize, v, 10) catch return error.MalformedRequest;
    }
    if (raw.len < hdr_len + 4 + want) return error.TruncatedBody;
    _ = a;
    return .{
        .method = method,
        .target = target,
        .version = version,
        .raw_headers = raw_headers,
        .body = raw[hdr_len + 4 ..][0..want],
    };
}

test "request parsing: method, query-stripped path, case-insensitive headers" {
    const a = std.testing.allocator;
    const raw = "POST /v1/chat/completions?stream=1 HTTP/1.1\r\nHost: x\r\ncontent-LENGTH: 11\r\n\r\n" ++ "{\"hello\":1}";
    const req = try parseForTest(a, raw);
    try std.testing.expectEqualStrings("POST", req.method);
    try std.testing.expectEqualStrings("/v1/chat/completions?stream=1", req.target);
    try std.testing.expectEqualStrings("/v1/chat/completions", req.path());
    try std.testing.expectEqualStrings("11", req.header("Content-Length").?);
    try std.testing.expectEqualStrings("11", req.header("content-length").?);
    try std.testing.expectEqualStrings("x", req.header("host").?);
    try std.testing.expectEqualStrings("{\"hello\":1}", req.body);
}

test "request parsing: a truncated body is rejected, not silently accepted" {
    const a = std.testing.allocator;
    // Content-Length promises 20 bytes but the buffer holds 5.
    const raw = "POST /x HTTP/1.1\r\nContent-Length: 20\r\n\r\nabcde";
    try std.testing.expectError(error.TruncatedBody, parseForTest(a, raw));
}

test "request parsing: missing header terminator is rejected" {
    const a = std.testing.allocator;
    try std.testing.expectError(error.MalformedRequest, parseForTest(a, "GET / HTTP/1.1\r\nHost: x\r\n"));
    try std.testing.expectError(error.MalformedRequest, parseForTest(a, "GET\r\n\r\n"));
}

test "request parsing: no body means an empty slice" {
    const a = std.testing.allocator;
    const req = try parseForTest(a, "GET /health HTTP/1.1\r\n\r\n");
    try std.testing.expectEqualStrings("GET", req.method);
    try std.testing.expectEqualStrings("/health", req.path());
    try std.testing.expectEqual(@as(usize, 0), req.body.len);
    try std.testing.expect(req.header("content-length") == null);
}

test "hasPendingInput reports readiness without blocking" {
    // A socketpair gives us a connected pair without the network stack.
    var sv: [2]std.c.fd_t = undefined;
    const AF_UNIX: c_int = 1;
    if (libc.socketpair(AF_UNIX, SOCK_STREAM, 0, &sv) != 0) return error.SkipZigTest;
    defer _ = std.c.close(sv[0]);
    defer _ = std.c.close(sv[1]);
    var conn = Conn{ .fd = sv[0] };

    // Nothing sent: must return immediately, not wait for the timeout. A 0 ms
    // poll is inherently non-blocking, so this asserts the contract rather than
    // the clock, which is what the caller depends on.
    try std.testing.expect(!conn.hasPendingInput(0));

    // One byte makes it readable.
    const b = "x";
    _ = libc.send(sv[1], b.ptr, b.len, 0);
    try std.testing.expect(conn.hasPendingInput(500));
}

test "ipv4 parsing produces network byte order" {
    const v = try parseIpv4("127.0.0.1");
    // In memory this must be 7f 00 00 01 on a little-endian host.
    const bytes: [4]u8 = @bitCast(v);
    try std.testing.expectEqualSlices(u8, &[_]u8{ 0x7f, 0x00, 0x00, 0x01 }, &bytes);
    try std.testing.expectError(error.MalformedRequest, parseIpv4("300.1.1.1"));
    try std.testing.expectError(error.MalformedRequest, parseIpv4("1.2.3"));
}

test "header lookup is case-insensitive" {
    const req = Request{
        .method = "POST",
        .target = "/v1/chat/completions?x=1",
        .version = "HTTP/1.1",
        .raw_headers = "POST /x HTTP/1.1\r\nContent-Length: 12\r\nContent-Type: application/json\r\n\r\n",
        .body = "",
    };
    try std.testing.expectEqualStrings("12", req.header("content-length").?);
    try std.testing.expectEqualStrings("application/json", req.header("CONTENT-TYPE").?);
    try std.testing.expect(req.header("x-missing") == null);
    try std.testing.expectEqualStrings("/v1/chat/completions", req.path());
}
