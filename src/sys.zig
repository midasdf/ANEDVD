// sys.zig — thin, stable platform layer.
//
// Zig 0.17 moved file IO behind the new std.Io interface; this project uses
// libc directly (we already link libc for the ANE shim) so the code stays
// readable and independent of that churn.

const std = @import("std");

// ---------------------------------------------------------------- printing

/// Write to stdout without going through std.Io.
pub fn print(comptime fmt: []const u8, args: anytype) void {
    var tmp: [8192]u8 = undefined;
    const s = std.fmt.bufPrint(&tmp, fmt, args) catch return;
    writeAll(1, s);
}

/// Write to stderr.
pub fn eprint(comptime fmt: []const u8, args: anytype) void {
    var tmp: [8192]u8 = undefined;
    const s = std.fmt.bufPrint(&tmp, fmt, args) catch return;
    writeAll(2, s);
}

pub fn writeAll(fd: std.c.fd_t, bytes: []const u8) void {
    var off: usize = 0;
    while (off < bytes.len) {
        const n = std.c.write(fd, bytes.ptr + off, bytes.len - off);
        if (n <= 0) return;
        off += @intCast(n);
    }
}

// ---------------------------------------------------------------- little-endian

pub inline fn readU16LE(b: []const u8) u16 {
    return @as(u16, b[0]) | (@as(u16, b[1]) << 8);
}

pub inline fn readU32LE(b: []const u8) u32 {
    return @as(u32, b[0]) | (@as(u32, b[1]) << 8) | (@as(u32, b[2]) << 16) | (@as(u32, b[3]) << 24);
}

pub inline fn readU64LE(b: []const u8) u64 {
    return @as(u64, readU32LE(b[0..4])) | (@as(u64, readU32LE(b[4..8])) << 32);
}

pub inline fn readI32LE(b: []const u8) i32 {
    return @bitCast(readU32LE(b));
}

pub inline fn readF32LE(b: []const u8) f32 {
    return @bitCast(readU32LE(b));
}

pub inline fn readF16LE(b: []const u8) f16 {
    return @bitCast(readU16LE(b));
}

pub inline fn writeU16LE(dst: []u8, v: u16) void {
    dst[0] = @truncate(v);
    dst[1] = @truncate(v >> 8);
}

pub inline fn writeU32LE(dst: []u8, v: u32) void {
    dst[0] = @truncate(v);
    dst[1] = @truncate(v >> 8);
    dst[2] = @truncate(v >> 16);
    dst[3] = @truncate(v >> 24);
}

pub inline fn writeU64LE(dst: []u8, v: u64) void {
    writeU32LE(dst[0..4], @truncate(v));
    writeU32LE(dst[4..8], @truncate(v >> 32));
}

// ---------------------------------------------------------------- paths

/// Copy `path` into `buf` and NUL-terminate it, for libc calls.
pub fn pathZ(path: []const u8, buf: []u8) ![:0]const u8 {
    if (path.len + 1 > buf.len) return error.PathTooLong;
    @memcpy(buf[0..path.len], path);
    buf[path.len] = 0;
    return buf[0..path.len :0];
}

// ---------------------------------------------------------------- files

pub const MappedFile = struct {
    fd: std.c.fd_t = -1,
    data: []align(std.heap.page_size_min) const u8 = &.{},

    /// Memory-map a file read-only.
    pub fn open(path: []const u8) !MappedFile {
        var pbuf: [4096]u8 = undefined;
        const p = try pathZ(path, &pbuf);
        const fd = std.c.open(p.ptr, .{ .ACCMODE = .RDONLY, .CLOEXEC = true });
        if (fd < 0) return error.OpenFailed;
        errdefer _ = std.c.close(fd);
        var st: std.c.Stat = undefined;
        if (std.c.fstat(fd, &st) != 0) return error.StatFailed;
        const size: usize = @intCast(st.size);
        if (size == 0) return .{ .fd = fd, .data = &.{} };
        const raw = std.c.mmap(null, size, .{ .READ = true }, .{ .TYPE = .PRIVATE }, fd, 0);
        if (raw == std.c.MAP_FAILED) return error.MmapFailed;
        const ptr: [*]align(std.heap.page_size_min) const u8 = @ptrCast(@alignCast(raw));
        // NOT MADV_RANDOM, despite MoE expert access being scattered across a 9.5 GB
        // file. Measured over four interleaved pairs of single-token runs, MADV_RANDOM
        // LOST every pair on minor page faults (median 391208 against 383327, 0 wins of
        // 4), so the kernel's default readahead is the better choice here — presumably
        // because the experts a generation actually revisits are spatially clustered by
        // layer. Recorded so it is not "fixed" again without measuring.
        return .{ .fd = fd, .data = ptr[0..size] };
    }

    pub fn close(self: *MappedFile) void {
        if (self.data.len > 0) _ = std.c.munmap(@ptrCast(@alignCast(self.data.ptr)), self.data.len);
        if (self.fd >= 0) _ = std.c.close(self.fd);
        self.* = .{};
    }
};

pub fn readFileAlloc(allocator: std.mem.Allocator, path: []const u8, max_bytes: usize) ![]u8 {
    var f = try MappedFile.open(path);
    defer f.close();
    if (f.data.len > max_bytes) return error.FileTooLarge;
    return allocator.dupe(u8, f.data);
}

pub fn writeFile(path: []const u8, bytes: []const u8) !void {
    var pbuf: [4096]u8 = undefined;
    const p = try pathZ(path, &pbuf);
    const fd = std.c.open(p.ptr, .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true, .CLOEXEC = true }, @as(std.c.mode_t, 0o644));
    if (fd < 0) return error.CreateFailed;
    defer _ = std.c.close(fd);
    var off: usize = 0;
    while (off < bytes.len) {
        const n = std.c.write(fd, bytes.ptr + off, bytes.len - off);
        if (n <= 0) return error.WriteFailed;
        off += @intCast(n);
    }
}

pub fn fileSize(path: []const u8) !u64 {
    var pbuf: [4096]u8 = undefined;
    const p = try pathZ(path, &pbuf);
    const fd = std.c.open(p.ptr, .{ .ACCMODE = .RDONLY, .CLOEXEC = true });
    if (fd < 0) return error.OpenFailed;
    defer _ = std.c.close(fd);
    var st: std.c.Stat = undefined;
    if (std.c.fstat(fd, &st) != 0) return error.StatFailed;
    return @intCast(st.size);
}

pub fn mkdirp(path: []const u8) !void {
    var pbuf: [4096]u8 = undefined;
    const p = try pathZ(path, &pbuf);
    var i: usize = 1;
    while (i <= path.len) : (i += 1) {
        if (i == path.len or path[i] == '/') {
            var tmp: [4096]u8 = undefined;
            if (i + 1 > tmp.len) return error.PathTooLong;
            @memcpy(tmp[0..i], path[0..i]);
            tmp[i] = 0;
            _ = std.c.mkdir(&tmp, @as(std.c.mode_t, 0o755));
        }
    }
    _ = p;
}

// ---------------------------------------------------------------- time

/// Unix epoch seconds (for API `created` fields).
pub fn unixTime() i64 {
    var ts: std.c.timespec = undefined;
    if (std.c.clock_gettime(std.c.CLOCK.REALTIME, &ts) != 0) return 0;
    return @intCast(ts.sec);
}

/// Sleep for `ms` milliseconds. std.Thread.sleep no longer exists in 0.17 and
/// std.Io.sleep needs an Io, so this uses the POSIX primitive directly.
pub fn sleepMs(ms: u64) void {
    var ts = std.c.timespec{
        .sec = @intCast(ms / 1000),
        .nsec = @intCast((ms % 1000) * std.time.ns_per_ms),
    };
    while (nanosleep(&ts, &ts) != 0) {}
}

extern "c" fn nanosleep(req: *const std.c.timespec, rem: *std.c.timespec) c_int;

pub fn nowNs() u64 {
    var ts: std.c.timespec = undefined;
    if (std.c.clock_gettime(std.c.CLOCK.MONOTONIC, &ts) != 0) return 0;
    return @as(u64, @intCast(ts.sec)) * 1_000_000_000 + @as(u64, @intCast(ts.nsec));
}

test "little-endian readers" {
    const b = [_]u8{ 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08 };
    try std.testing.expectEqual(@as(u32, 0x04030201), readU32LE(b[0..4]));
    try std.testing.expectEqual(@as(u64, 0x0807060504030201), readU64LE(b[0..8]));
    try std.testing.expectEqual(@as(u16, 0x0201), readU16LE(b[0..2]));
}
