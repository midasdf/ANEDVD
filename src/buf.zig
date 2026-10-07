// buf.zig — minimal growable byte buffer.
//
// Zig 0.17's std.ArrayList/std.Io APIs are in flux, so this module keeps the
// engine's string building independent of them.

const std = @import("std");

pub const Buf = struct {
    allocator: std.mem.Allocator,
    items: []u8 = &.{},
    len: usize = 0,

    pub fn init(allocator: std.mem.Allocator) Buf {
        return .{ .allocator = allocator };
    }

    pub fn deinit(self: *Buf) void {
        if (self.items.len > 0) self.allocator.free(self.items);
        self.* = .{ .allocator = self.allocator };
    }

    pub fn ensure(self: *Buf, extra: usize) !void {
        const need = self.len + extra;
        if (need <= self.items.len) return;
        var cap = if (self.items.len == 0) @as(usize, 4096) else self.items.len;
        while (cap < need) cap *= 2;
        const new = try self.allocator.alloc(u8, cap);
        if (self.len > 0) @memcpy(new[0..self.len], self.items[0..self.len]);
        if (self.items.len > 0) self.allocator.free(self.items);
        self.items = new;
    }

    pub fn appendSlice(self: *Buf, bytes: []const u8) !void {
        try self.ensure(bytes.len);
        @memcpy(self.items[self.len..][0..bytes.len], bytes);
        self.len += bytes.len;
    }

    pub fn append(self: *Buf, byte: u8) !void {
        try self.ensure(1);
        self.items[self.len] = byte;
        self.len += 1;
    }

    /// printf-style append; lines longer than the scratch buffer are rejected.
    pub fn print(self: *Buf, comptime fmt: []const u8, args: anytype) !void {
        var tmp: [16384]u8 = undefined;
        const s = std.fmt.bufPrint(&tmp, fmt, args) catch |e| switch (e) {
            error.NoSpaceLeft => return error.FormatTooLong,
        };
        try self.appendSlice(s);
    }

    pub fn slice(self: *const Buf) []const u8 {
        return self.items[0..self.len];
    }

    /// Hand ownership of the buffer contents to the caller.
    pub fn toOwnedSlice(self: *Buf) ![]u8 {
        const out = try self.allocator.alloc(u8, self.len);
        if (self.len > 0) @memcpy(out, self.items[0..self.len]);
        if (self.items.len > 0) self.allocator.free(self.items);
        self.items = &.{};
        self.len = 0;
        return out;
    }
};

test "buf append and print" {
    const allocator = std.testing.allocator;
    var b = Buf.init(allocator);
    defer b.deinit();
    try b.appendSlice("hello");
    try b.append(' ');
    try b.print("world {d}", .{42});
    try std.testing.expectEqualStrings("hello world 42", b.slice());
}
