// weights.zig — pack tensors into the ANE weight-file format.
//
// Verified format (matches maderix/ANE and Espresso, and reproduced locally):
//
//   [0,64)      file header          byte0 = 0x01, byte4 = 0x02
//   per tensor:
//     [H,H+64)  chunk header         0xEF 0xBE 0xAD 0xDE at H, version 1 at H+4,
//                                    u32 payload size at H+8, u32 data offset
//                                    (always 128) at H+16
//     [H+64,..) fp16 payload         row-major [Cout][Cin][kh][kw]
//
// The MIL `BLOBFILE(offset = ...)` value is the payload's absolute file offset
// minus 64, i.e. for chunk k: 64*(k+1) + sum(size of previous payloads).
// (Checked against the published fused_ffn.mil fixture: offsets 64 and 176 for
// a 48-byte first payload.)

const std = @import("std");
const sys = @import("../sys.zig");

pub const CHUNK_HEADER = 64;
pub const FILE_HEADER = 64;

pub const Tensor = struct {
    /// Logical name (used for diagnostics only; the file symbol is positional).
    name: []const u8,
    /// fp16 payload in the exact order the MIL weight tensor expects.
    data: []const f16,
};

pub const Packed = struct {
    allocator: std.mem.Allocator,
    bytes: []u8,
    /// BLOBFILE offset for each tensor, in input order.
    offsets: []u64,

    pub fn deinit(self: *Packed) void {
        self.allocator.free(self.bytes);
        self.allocator.free(self.offsets);
        self.* = undefined;
    }
};

/// MIL weight symbol for a kernel's single weight file.
pub fn symbol(buf: []u8) ![:0]const u8 {
    const s = "@model_path/weights/w.bin";
    if (buf.len < s.len + 1) return error.NoSpaceLeft;
    @memcpy(buf[0..s.len], s);
    buf[s.len] = 0;
    return buf[0..s.len :0];
}

/// BLOBFILE offset for chunk `k` of a file laid out as
/// [64-byte file header][64-byte chunk header + payload]*, where the MIL offset
/// is the payload's absolute offset minus 64.
pub fn chunkOffsets(sizes: []const usize, out: []u64) void {
    var cursor: u64 = 64; // file header
    for (sizes, 0..) |size, k| {
        out[k] = cursor; // payload starts after this chunk's 64-byte header
        cursor += 64 + size;
    }
}

pub fn pack(allocator: std.mem.Allocator, tensors: []const Tensor) !Packed {
    var total: usize = FILE_HEADER;
    for (tensors) |t| total += CHUNK_HEADER + t.data.len * 2;

    const bytes = try allocator.alloc(u8, total);
    errdefer allocator.free(bytes);
    @memset(bytes, 0);

    const offsets = try allocator.alloc(u64, tensors.len);
    errdefer allocator.free(offsets);

    // File header.
    bytes[0] = 0x01;
    bytes[4] = 0x02;

    var cursor: usize = FILE_HEADER;
    for (tensors, 0..) |t, k| {
        const payload_size = t.data.len * 2;
        const h = cursor;
        bytes[h + 0] = 0xEF;
        bytes[h + 1] = 0xBE;
        bytes[h + 2] = 0xAD;
        bytes[h + 3] = 0xDE;
        bytes[h + 4] = 0x01;
        sys.writeU32LE(bytes[h + 8 ..][0..4], @intCast(payload_size));
        // Absolute payload offset for THIS chunk (128, 240, 352, ...).
        sys.writeU32LE(bytes[h + 16 ..][0..4], @intCast(h + CHUNK_HEADER));

        const payload_off = h + CHUNK_HEADER;
        const dst = bytes[payload_off..][0..payload_size];
        @memcpy(dst, std.mem.sliceAsBytes(t.data));

        // MIL BLOBFILE offset = absolute payload offset - 64.
        offsets[k] = payload_off - 64;
        cursor = payload_off + payload_size;
    }
    std.debug.assert(cursor == total);
    return .{ .allocator = allocator, .bytes = bytes, .offsets = offsets };
}

/// Convert f32 to f16 (round to nearest even, as @floatCast does).
pub fn f32ToF16(allocator: std.mem.Allocator, src: []const f32) ![]f16 {
    const out = try allocator.alloc(f16, src.len);
    for (src, 0..) |v, i| out[i] = @floatCast(v);
    return out;
}

pub fn f16ToF32(allocator: std.mem.Allocator, src: []const f16) ![]f32 {
    const out = try allocator.alloc(f32, src.len);
    for (src, 0..) |v, i| out[i] = @floatCast(v);
    return out;
}

test "weight blob layout matches the published fixture" {
    const allocator = std.testing.allocator;
    // Two tensors of 24 fp16 values each (48 bytes), mirroring fused_ffn.mil.
    var a: [24]f16 = undefined;
    var b: [24]f16 = undefined;
    for (&a, 0..) |*v, i| v.* = @floatCast(@as(f32, @floatFromInt(i)) * 0.5);
    for (&b, 0..) |*v, i| v.* = @floatCast(@as(f32, @floatFromInt(i)) * -0.25);
    const blob = try pack(allocator, &.{ .{ .name = "W1", .data = &a }, .{ .name = "W3", .data = &b } });
    defer blob.deinit();

    // Chunk 0 payload at 128 -> BLOBFILE offset 64; chunk 1 payload at 240 -> 176.
    try std.testing.expectEqual(@as(u64, 64), blob.offsets[0]);
    try std.testing.expectEqual(@as(u64, 176), blob.offsets[1]);
    try std.testing.expectEqual(@as(usize, 288), blob.bytes.len);

    // Chunk headers.
    try std.testing.expectEqual(@as(u8, 0xEF), blob.bytes[64]);
    try std.testing.expectEqual(@as(u8, 0xDE), blob.bytes[67]);
    try std.testing.expectEqual(@as(u32, 48), sys.readU32LE(blob.bytes[72..76]));
    try std.testing.expectEqual(@as(u32, 48), sys.readU32LE(blob.bytes[176..180]));

    // `data_off` is the ABSOLUTE offset of each chunk's own payload (128 then
    // 240), not a constant. Assuming a constant compiles fine and silently
    // feeds the wrong weights to every chunk after the first; this is checked
    // against the published ffn_blob_ref.bin byte-for-byte.
    try std.testing.expectEqual(@as(u32, 128), sys.readU32LE(blob.bytes[80..84]));
    try std.testing.expectEqual(@as(u32, 240), sys.readU32LE(blob.bytes[192..196]));

    // Payload round-trip.
    const got: []const f16 = @alignCast(std.mem.bytesAsSlice(f16, blob.bytes[128..176]));
    for (a, 0..) |v, i| try std.testing.expectEqual(v, got[i]);
}

test "chunkOffsets agrees with pack()" {
    const allocator = std.testing.allocator;
    const sizes = [_]usize{ 48, 96, 24 };
    var a: [48]f16 = undefined;
    var b: [96]f16 = undefined;
    var c: [24]f16 = undefined;
    for (&a) |*v| v.* = 1;
    for (&b) |*v| v.* = 2;
    for (&c) |*v| v.* = 3;
    var blob = try pack(allocator, &.{
        .{ .name = "a", .data = &a },
        .{ .name = "b", .data = &b },
        .{ .name = "c", .data = &c },
    });
    defer blob.deinit();
    var computed: [3]u64 = undefined;
    chunkOffsets(&sizes, &computed);
    for (computed, blob.offsets) |x, y| try std.testing.expectEqual(y, x);
}

test "f32ToF16 round-trips exactly-representable values" {
    const allocator = std.testing.allocator;
    const src = [_]f32{ 0.0, 1.0, -2.5, 0.125 };
    const half = try f32ToF16(allocator, &src);
    defer allocator.free(half);
    const back = try f16ToF32(allocator, half);
    defer allocator.free(back);
    for (src, 0..) |v, i| try std.testing.expectEqual(v, back[i]);
}
