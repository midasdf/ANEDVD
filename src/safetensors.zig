//! Minimal, dependency-free reader for HuggingFace `safetensors` weight files.
//!
//! Format: an 8-byte little-endian `u64` giving the length `N` of a JSON header,
//! then `N` header bytes, then the raw tensor payload. Tensor `data_offsets` are
//! relative to the first byte after the header.
//!
//! Only std is used. Zig 0.17's `std.Io` requires an `Io` instance for every
//! filesystem call; because the public API here is `(allocator, path)`, loading
//! uses `std.Io.Threaded.global_single_threaded`, the std-sanctioned singleton
//! for library code that does not take an `Io` parameter. All operations used
//! (open/read/stat) are synchronous and need no concurrency support.

const std = @import("std");
const Allocator = std.mem.Allocator;
const Io = std.Io;

/// Element type of a tensor as declared by the safetensors header.
///
/// Non-exhaustive: unrecognised on-disk dtype strings are rejected at parse time
/// rather than being silently coerced into a wrong representation.
pub const Dtype = enum(u8) {
    f64,
    f32,
    f16,
    bf16,
    i64,
    i32,
    i16,
    i8,
    u8,
    bool_,
    _,

    /// Maps a safetensors dtype string (e.g. `"BF16"`) to a `Dtype`.
    /// Returns null for dtypes this module does not support.
    pub fn fromString(s: []const u8) ?Dtype {
        const table = .{
            .{ "F64", Dtype.f64 },
            .{ "F32", Dtype.f32 },
            .{ "F16", Dtype.f16 },
            .{ "BF16", Dtype.bf16 },
            .{ "I64", Dtype.i64 },
            .{ "I32", Dtype.i32 },
            .{ "I16", Dtype.i16 },
            .{ "I8", Dtype.i8 },
            .{ "U8", Dtype.u8 },
            .{ "BOOL", Dtype.bool_ },
        };
        inline for (table) |entry| {
            if (std.mem.eql(u8, s, entry[0])) return entry[1];
        }
        return null;
    }

    /// Canonical safetensors spelling of this dtype.
    pub fn toString(self: Dtype) []const u8 {
        return switch (self) {
            .f64 => "F64",
            .f32 => "F32",
            .f16 => "F16",
            .bf16 => "BF16",
            .i64 => "I64",
            .i32 => "I32",
            .i16 => "I16",
            .i8 => "I8",
            .u8 => "U8",
            .bool_ => "BOOL",
            else => "UNKNOWN",
        };
    }

    /// Bytes per element, or null for dtypes this module cannot size.
    pub fn byteSize(self: Dtype) ?usize {
        return switch (self) {
            .f64, .i64 => 8,
            .f32, .i32 => 4,
            .f16, .bf16, .i16 => 2,
            .i8, .u8, .bool_ => 1,
            else => null,
        };
    }
};

/// A tensor view. `data` borrows the owning `Safetensors` file bytes and is
/// valid only until `deinit` is called (or, for shards returned by
/// `findTensorInShards`, until that shard is closed).
pub const Tensor = struct {
    name: []const u8,
    dtype: Dtype,
    shape: []const u64,
    data: []const u8,

    /// Number of elements (product of `shape`; 1 for a scalar).
    pub fn numel(self: Tensor) usize {
        var n: usize = 1;
        for (self.shape) |d| n = std.math.mul(usize, n, @intCast(d)) catch return 0;
        return n;
    }

    pub fn rank(self: Tensor) usize {
        return self.shape.len;
    }
};

pub const Error = error{
    /// File is shorter than the 8-byte header length prefix.
    HeaderTooSmall,
    /// Declared header length overruns the file.
    InvalidHeaderLength,
    /// Header JSON is malformed, or an entry is not an object.
    InvalidHeader,
    /// A tensor entry is missing `dtype` / `shape` / `data_offsets`, or they
    /// have the wrong JSON type.
    InvalidTensorEntry,
    /// `dtype` names a type this module does not implement.
    UnsupportedDtype,
    /// `data_offsets` are reversed, out of the payload, or do not match
    /// `shape` × element size.
    DataSizeMismatch,
    /// Requested tensor is not present in this file.
    TensorNotFound,
};

pub const Safetensors = struct {
    gpa: Allocator,
    /// Backing storage for names, shapes and header temporaries. Held until
    /// `deinit` because every name/shape slice points into it.
    arena: std.heap.ArenaAllocator,
    /// Whole file, including the 8-byte prefix and the JSON header.
    bytes: []const u8,
    /// Insertion-ordered index of tensors (header order).
    map: std.array_hash_map.String(Tensor),

    /// Reads and parses every tensor header in `path` (absolute, or relative to
    /// the process working directory). The whole file is read into memory once;
    /// tensor `data` slices alias that buffer.
    pub fn load(allocator: Allocator, path: []const u8) !Safetensors {
        const bytes = try Io.Dir.cwd().readFileAlloc(ioInstance(), path, allocator, .unlimited);
        errdefer allocator.free(bytes);
        return initOwned(allocator, bytes);
    }

    /// Like `load`, but takes the already-read file contents. The buffer is
    /// copied, so the caller keeps ownership of `bytes`.
    pub fn fromBytes(allocator: Allocator, bytes: []const u8) !Safetensors {
        const owned = try allocator.dupe(u8, bytes);
        errdefer allocator.free(owned);
        return initOwned(allocator, owned);
    }

    fn initOwned(allocator: Allocator, bytes: []const u8) !Safetensors {
        var self: Safetensors = .{
            .gpa = allocator,
            .arena = std.heap.ArenaAllocator.init(allocator),
            .bytes = bytes,
            .map = .empty,
        };
        errdefer {
            self.map.deinit(allocator);
            self.arena.deinit();
        }

        if (bytes.len < 8) return Error.HeaderTooSmall;
        const header_len_u64 = std.mem.readInt(u64, bytes[0..8], .little);
        const header_len = std.math.cast(usize, header_len_u64) orelse return Error.InvalidHeaderLength;
        if (header_len > bytes.len - 8) return Error.InvalidHeaderLength;
        const header = bytes[8 .. 8 + header_len];
        const payload = bytes[8 + header_len ..];

        const a = self.arena.allocator();
        const root = std.json.parseFromSliceLeaky(std.json.Value, a, header, .{}) catch
            return Error.InvalidHeader;
        if (root != .object) return Error.InvalidHeader;
        try self.parseHeader(a, root.object, payload);

        return self;
    }

    fn parseHeader(
        self: *Safetensors,
        a: Allocator,
        obj: std.json.ObjectMap,
        payload: []const u8,
    ) !void {
        var it = obj.iterator();
        while (it.next()) |kv| {
            const name = kv.key_ptr.*;
            if (std.mem.eql(u8, name, "__metadata__")) continue;
            const value = kv.value_ptr.*;
            if (value != .object) return Error.InvalidTensorEntry;
            const entry = value.object;

            const dtype_value = entry.get("dtype") orelse return Error.InvalidTensorEntry;
            if (dtype_value != .string) return Error.InvalidTensorEntry;
            const dtype = Dtype.fromString(dtype_value.string) orelse return Error.UnsupportedDtype;
            const elem_size = dtype.byteSize() orelse return Error.UnsupportedDtype;

            const shape_value = entry.get("shape") orelse return Error.InvalidTensorEntry;
            if (shape_value != .array) return Error.InvalidTensorEntry;
            const dims = shape_value.array.items;
            const shape = try a.alloc(u64, dims.len);
            var numel: usize = 1;
            for (dims, 0..) |dim, i| {
                if (dim != .integer or dim.integer < 0) return Error.InvalidTensorEntry;
                const d: u64 = @intCast(dim.integer);
                shape[i] = d;
                numel = std.math.mul(usize, numel, std.math.cast(usize, d) orelse
                    return Error.DataSizeMismatch) catch return Error.DataSizeMismatch;
            }

            const offsets_value = entry.get("data_offsets") orelse return Error.InvalidTensorEntry;
            if (offsets_value != .array) return Error.InvalidTensorEntry;
            const offsets = offsets_value.array.items;
            if (offsets.len != 2) return Error.InvalidTensorEntry;
            if (offsets[0] != .integer or offsets[1] != .integer) return Error.InvalidTensorEntry;
            if (offsets[0].integer < 0 or offsets[1].integer < 0) return Error.DataSizeMismatch;
            const begin: usize = std.math.cast(usize, @as(u64, @intCast(offsets[0].integer))) orelse
                return Error.DataSizeMismatch;
            const end: usize = std.math.cast(usize, @as(u64, @intCast(offsets[1].integer))) orelse
                return Error.DataSizeMismatch;
            if (begin > end or end > payload.len) return Error.DataSizeMismatch;

            const data = payload[begin..end];
            const expected = std.math.mul(usize, numel, elem_size) catch return Error.DataSizeMismatch;
            if (expected != data.len) return Error.DataSizeMismatch;

            try self.map.put(self.gpa, name, .{
                .name = name,
                .dtype = dtype,
                .shape = shape,
                .data = data,
            });
        }
    }

    /// Releases the file buffer, the header arena and the tensor index.
    pub fn deinit(self: *Safetensors) void {
        self.map.deinit(self.gpa);
        self.arena.deinit();
        self.gpa.free(self.bytes);
        self.* = undefined;
    }

    /// Tensor names in header order.
    pub fn names(self: *const Safetensors) []const []const u8 {
        return self.map.keys();
    }

    /// Number of tensors (excludes `__metadata__`).
    pub fn len(self: *const Safetensors) usize {
        return self.map.count();
    }

    pub fn has(self: *const Safetensors, name: []const u8) bool {
        return self.map.contains(name);
    }

    /// Borrowed view of `name`, or null when absent.
    pub fn tensor(self: *const Safetensors, name: []const u8) ?Tensor {
        return self.map.get(name);
    }

    /// Decodes `name` into freshly-allocated `f32` values, converting from any
    /// supported dtype (including bf16, which has no Zig primitive type).
    /// Caller owns the result.
    pub fn readF32(self: *const Safetensors, allocator: Allocator, name: []const u8) ![]f32 {
        const t = self.tensor(name) orelse return Error.TensorNotFound;
        const out = try allocator.alloc(f32, t.numel());
        errdefer allocator.free(out);
        try convertTo(f32, t, out);
        return out;
    }

    /// Decodes `name` into freshly-allocated `f16` values. Caller owns the result.
    pub fn readF16(self: *const Safetensors, allocator: Allocator, name: []const u8) ![]f16 {
        const t = self.tensor(name) orelse return Error.TensorNotFound;
        const out = try allocator.alloc(f16, t.numel());
        errdefer allocator.free(out);
        try convertTo(f16, t, out);
        return out;
    }
};

/// The `Io` implementation used when no `Io` is supplied by the caller.
/// Documented std escape hatch for libraries whose public API has no `Io`
/// parameter; it is synchronous and supports no concurrency.
pub fn ioInstance() Io {
    return Io.Threaded.global_single_threaded.io();
}

/// Converts a tensor of any supported dtype into `out` (length must equal
/// `t.numel()`).
pub fn convertTo(comptime Out: type, t: Tensor, out: []Out) !void {
    if (out.len != t.numel()) return Error.DataSizeMismatch;
    const bytes = t.data;
    switch (t.dtype) {
        .f64 => for (out, 0..) |*o, i| {
            o.* = @floatCast(readFloat(f64, bytes, i));
        },
        .f32 => for (out, 0..) |*o, i| {
            o.* = @floatCast(readFloat(f32, bytes, i));
        },
        .f16 => for (out, 0..) |*o, i| {
            o.* = @floatCast(readFloat(f16, bytes, i));
        },
        .bf16 => for (out, 0..) |*o, i| {
            o.* = @floatCast(bf16ToF32(readUint(u16, bytes, i)));
        },
        .i64 => for (out, 0..) |*o, i| {
            o.* = @floatFromInt(readSigned(i64, bytes, i));
        },
        .i32 => for (out, 0..) |*o, i| {
            o.* = @floatFromInt(readSigned(i32, bytes, i));
        },
        .i16 => for (out, 0..) |*o, i| {
            o.* = @floatFromInt(readSigned(i16, bytes, i));
        },
        .i8 => for (out, 0..) |*o, i| {
            o.* = @floatFromInt(readSigned(i8, bytes, i));
        },
        .u8 => for (out, 0..) |*o, i| {
            o.* = @floatFromInt(bytes[i]);
        },
        .bool_ => for (out, 0..) |*o, i| {
            o.* = if (bytes[i] != 0) 1 else 0;
        },
        else => return Error.UnsupportedDtype,
    }
}

fn readUint(comptime T: type, bytes: []const u8, index: usize) T {
    const n = @sizeOf(T);
    var buf: [n]u8 = undefined;
    @memcpy(&buf, bytes[index * n ..][0..n]);
    return std.mem.readInt(T, &buf, .little);
}

/// Reads a two's-complement little-endian signed integer of `T`'s width.
fn readSigned(comptime T: type, bytes: []const u8, index: usize) T {
    return @bitCast(readUint(@Int(.unsigned, @bitSizeOf(T)), bytes, index));
}

fn readFloat(comptime F: type, bytes: []const u8, index: usize) F {
    return @bitCast(readUint(@Int(.unsigned, @bitSizeOf(F)), bytes, index));
}

/// bf16 is the top 16 bits of an IEEE-754 binary32.
pub fn bf16ToF32(bits: u16) f32 {
    const wide: u32 = @as(u32, bits) << 16;
    return @bitCast(wide);
}

/// Rounds a binary32 to its nearest bf16 (round-half-to-even), returned in the
/// low 16 bits.
pub fn f32ToBf16(value: f32) u16 {
    const bits: u32 = @bitCast(value);
    if ((bits & 0x7FFF_FFFF) > 0x7F80_0000) {
        // NaN: keep it a NaN, preserve the payload's top bits.
        return @intCast((bits >> 16) | 0x0040);
    }
    const lsb: u32 = (bits >> 16) & 1;
    return @intCast((bits +% 0x7FFF +% lsb) >> 16);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

const testing = std.testing;

/// Builds a complete safetensors image (prefix + header + payload).
fn buildImage(allocator: Allocator, header_json: []const u8, payload: []const u8) ![]u8 {
    var buf: std.ArrayList(u8) = .empty;
    errdefer buf.deinit(allocator);
    try buf.ensureUnusedCapacity(allocator, 8 + header_json.len + payload.len);
    var len_bytes: [8]u8 = undefined;
    std.mem.writeInt(u64, &len_bytes, header_json.len, .little);
    buf.appendSliceAssumeCapacity(&len_bytes);
    buf.appendSliceAssumeCapacity(header_json);
    buf.appendSliceAssumeCapacity(payload);
    return buf.toOwnedSlice(allocator);
}

/// Appends `v` to `list` as little-endian bytes (safetensors is little-endian
/// on every host, so tests must not rely on native byte order).
fn appendLE(allocator: Allocator, list: *std.ArrayList(u8), comptime T: type, v: T) !void {
    var out: [@sizeOf(T)]u8 = undefined;
    std.mem.writeInt(@Int(.unsigned, @bitSizeOf(T)), &out, @bitCast(v), .little);
    try list.appendSlice(allocator, &out);
}

const test_header =
    \\{"__metadata__":{"format":"pt"},
    \\ "f32_a":{"dtype":"F32","shape":[2,2],"data_offsets":[0,16]},
    \\ "f16_b":{"dtype":"F16","shape":[3],"data_offsets":[16,22]},
    \\ "bf16_c":{"dtype":"BF16","shape":[2],"data_offsets":[22,26]},
    \\ "i32_d":{"dtype":"I32","shape":[3],"data_offsets":[26,38]},
    \\ "bool_e":{"dtype":"BOOL","shape":[2],"data_offsets":[38,40]}}
;

fn buildPayload(allocator: Allocator) ![]u8 {
    var payload: std.ArrayList(u8) = .empty;
    errdefer payload.deinit(allocator);
    const f32s = [_]f32{ 1.5, -2.25, 0.0, 42.0 };
    for (f32s) |v| try appendLE(allocator, &payload, f32, v);
    const f16s = [_]f16{ 1.0, 0.5, -3.0 };
    for (f16s) |v| try appendLE(allocator, &payload, f16, v);
    try appendLE(allocator, &payload, u16, f32ToBf16(1.0));
    try appendLE(allocator, &payload, u16, f32ToBf16(-0.5));
    const i32s = [_]i32{ 1, -2, 3 };
    for (i32s) |v| try appendLE(allocator, &payload, i32, v);
    try payload.appendSlice(allocator, &.{ 0, 1 });
    try testing.expectEqual(@as(usize, 40), payload.items.len);
    return payload.toOwnedSlice(allocator);
}

test "bf16 helpers round-trip exact values" {
    try testing.expectEqual(@as(u16, 0x3F80), f32ToBf16(1.0));
    try testing.expectEqual(@as(u16, 0xBF00), f32ToBf16(-0.5));
    try testing.expectEqual(@as(u16, 0x0000), f32ToBf16(0.0));
    try testing.expectEqual(@as(u16, 0x8000), f32ToBf16(-0.0));
    try testing.expectEqual(@as(f32, 1.0), bf16ToF32(0x3F80));
    try testing.expectEqual(@as(f32, -0.5), bf16ToF32(0xBF00));
    // 1 + 2^-8 sits exactly halfway between bf16 neighbours -> round to even.
    try testing.expectEqual(@as(u16, 0x3F80), f32ToBf16(1.0 + 0.00390625));
    try testing.expect(std.math.isNan(bf16ToF32(f32ToBf16(std.math.nan(f32)))));
}

test "safetensors: header parsing, metadata, dtypes and conversions" {
    const allocator = testing.allocator;
    const payload = try buildPayload(allocator);
    defer allocator.free(payload);
    const image = try buildImage(allocator, test_header, payload);
    defer allocator.free(image);

    var st = try Safetensors.fromBytes(allocator, image);
    defer st.deinit();

    // __metadata__ is not a tensor; header order is preserved.
    try testing.expectEqual(@as(usize, 5), st.len());
    const names = st.names();
    try testing.expectEqualStrings("f32_a", names[0]);
    try testing.expectEqualStrings("bool_e", names[4]);
    try testing.expect(st.has("f32_a"));
    try testing.expect(!st.has("__metadata__"));
    try testing.expect(!st.has("missing"));
    try testing.expect(st.tensor("missing") == null);

    const t = st.tensor("f32_a").?;
    try testing.expectEqual(Dtype.f32, t.dtype);
    try testing.expectEqual(@as(usize, 2), t.rank());
    try testing.expectEqual(@as(u64, 2), t.shape[0]);
    try testing.expectEqual(@as(u64, 2), t.shape[1]);
    try testing.expectEqual(@as(usize, 4), t.numel());
    try testing.expectEqual(@as(usize, 16), t.data.len);
    try testing.expectEqualStrings("f32_a", t.name);
    try testing.expectEqualStrings("F32", t.dtype.toString());

    const f32s = try st.readF32(allocator, "f32_a");
    defer allocator.free(f32s);
    try testing.expectEqualSlices(f32, &.{ 1.5, -2.25, 0.0, 42.0 }, f32s);

    const f16s = try st.readF16(allocator, "f16_b");
    defer allocator.free(f16s);
    try testing.expectEqualSlices(f16, &.{ 1.0, 0.5, -3.0 }, f16s);

    // f16 tensor widened to f32
    const f16_as_f32 = try st.readF32(allocator, "f16_b");
    defer allocator.free(f16_as_f32);
    try testing.expectEqualSlices(f32, &.{ 1.0, 0.5, -3.0 }, f16_as_f32);

    // bf16 has no Zig primitive type; it must decode to the exact f32 values.
    const bf16s = try st.readF32(allocator, "bf16_c");
    defer allocator.free(bf16s);
    try testing.expectEqualSlices(f32, &.{ 1.0, -0.5 }, bf16s);
    const bf16_as_f16 = try st.readF16(allocator, "bf16_c");
    defer allocator.free(bf16_as_f16);
    try testing.expectEqualSlices(f16, &.{ 1.0, -0.5 }, bf16_as_f16);

    // integer tensor widened to f32 and narrowed to f16
    const ints = try st.readF32(allocator, "i32_d");
    defer allocator.free(ints);
    try testing.expectEqualSlices(f32, &.{ 1.0, -2.0, 3.0 }, ints);

    const bools = try st.readF32(allocator, "bool_e");
    defer allocator.free(bools);
    try testing.expectEqualSlices(f32, &.{ 0.0, 1.0 }, bools);

    // f32 -> f16 narrowing is exact for these values.
    const narrowed = try st.readF16(allocator, "f32_a");
    defer allocator.free(narrowed);
    try testing.expectEqualSlices(f16, &.{ 1.5, -2.25, 0.0, 42.0 }, narrowed);

    try testing.expectError(Error.TensorNotFound, st.readF32(allocator, "nope"));
}

test "safetensors: every supported dtype is readable" {
    const allocator = testing.allocator;
    const header =
        \\{"f64":{"dtype":"F64","shape":[1],"data_offsets":[0,8]},
        \\ "f32":{"dtype":"F32","shape":[1],"data_offsets":[8,12]},
        \\ "f16":{"dtype":"F16","shape":[1],"data_offsets":[12,14]},
        \\ "bf16":{"dtype":"BF16","shape":[1],"data_offsets":[14,16]},
        \\ "i64":{"dtype":"I64","shape":[1],"data_offsets":[16,24]},
        \\ "i32":{"dtype":"I32","shape":[1],"data_offsets":[24,28]},
        \\ "i16":{"dtype":"I16","shape":[1],"data_offsets":[28,30]},
        \\ "i8":{"dtype":"I8","shape":[1],"data_offsets":[30,31]},
        \\ "u8":{"dtype":"U8","shape":[1],"data_offsets":[31,32]},
        \\ "bool":{"dtype":"BOOL","shape":[1],"data_offsets":[32,33]}}
    ;
    var payload: std.ArrayList(u8) = .empty;
    defer payload.deinit(allocator);
    try appendLE(allocator, &payload, f64, -1.25);
    try appendLE(allocator, &payload, f32, 2.5);
    try appendLE(allocator, &payload, f16, -0.75);
    try appendLE(allocator, &payload, u16, f32ToBf16(4.0));
    try appendLE(allocator, &payload, i64, -9);
    try appendLE(allocator, &payload, i32, 11);
    try appendLE(allocator, &payload, i16, -13);
    try appendLE(allocator, &payload, i8, -15);
    try appendLE(allocator, &payload, u8, 200);
    try payload.appendSlice(allocator, &.{1});
    try testing.expectEqual(@as(usize, 33), payload.items.len);

    const image = try buildImage(allocator, header, payload.items);
    defer allocator.free(image);
    var st = try Safetensors.fromBytes(allocator, image);
    defer st.deinit();

    const expected = [_]struct { []const u8, f32 }{
        .{ "f64", -1.25 },
        .{ "f32", 2.5 },
        .{ "f16", -0.75 },
        .{ "bf16", 4.0 },
        .{ "i64", -9.0 },
        .{ "i32", 11.0 },
        .{ "i16", -13.0 },
        .{ "i8", -15.0 },
        .{ "u8", 200.0 },
        .{ "bool", 1.0 },
    };
    for (expected) |case| {
        const values = try st.readF32(allocator, case[0]);
        defer allocator.free(values);
        try testing.expectEqual(@as(usize, 1), values.len);
        try testing.expectEqual(case[1], values[0]);
    }
}

test "safetensors: load from disk and deinit frees everything" {
    const allocator = testing.allocator;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();

    const payload = try buildPayload(allocator);
    defer allocator.free(payload);
    const image = try buildImage(allocator, test_header, payload);
    defer allocator.free(image);

    try tmp.dir.writeFile(testing.io, .{ .sub_path = "model.safetensors", .data = image });

    const path = try std.fs.path.join(allocator, &.{ ".zig-cache", "tmp", &tmp.sub_path, "model.safetensors" });
    defer allocator.free(path);

    var st = try Safetensors.load(allocator, path);
    defer st.deinit();
    try testing.expectEqual(@as(usize, 5), st.len());
    const values = try st.readF32(allocator, "i32_d");
    defer allocator.free(values);
    try testing.expectEqualSlices(f32, &.{ 1.0, -2.0, 3.0 }, values);

    try testing.expectError(error.FileNotFound, Safetensors.load(allocator, "definitely/not/here.safetensors"));
}

test "safetensors: rejects malformed files with specific errors" {
    const allocator = testing.allocator;

    try testing.expectError(Error.HeaderTooSmall, Safetensors.fromBytes(allocator, "abc"));
    try testing.expectError(Error.HeaderTooSmall, Safetensors.fromBytes(allocator, ""));
    // header length that cannot exist
    try testing.expectError(Error.InvalidHeaderLength, Safetensors.fromBytes(allocator, &@as([8]u8, @splat(0xFF))));

    // malformed JSON header
    const bad_json = try buildImage(allocator, "{not json", &.{});
    defer allocator.free(bad_json);
    try testing.expectError(Error.InvalidHeader, Safetensors.fromBytes(allocator, bad_json));

    // valid JSON, but not an object
    const not_object = try buildImage(allocator, "[1,2,3]", &.{});
    defer allocator.free(not_object);
    try testing.expectError(Error.InvalidHeader, Safetensors.fromBytes(allocator, not_object));

    // Declared header longer than the file.
    var truncated: [12]u8 = undefined;
    std.mem.writeInt(u64, truncated[0..8], 100, .little);
    truncated[8..].* = "abcd".*;
    try testing.expectError(Error.InvalidHeaderLength, Safetensors.fromBytes(allocator, &truncated));

    // dtype string that we do not implement
    const bad_dtype =
        \\{"t":{"dtype":"F8_E4M3","shape":[1],"data_offsets":[0,1]}}
    ;
    const img1 = try buildImage(allocator, bad_dtype, &.{0});
    defer allocator.free(img1);
    try testing.expectError(Error.UnsupportedDtype, Safetensors.fromBytes(allocator, img1));

    // offsets beyond the payload
    const bad_offsets =
        \\{"t":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}
    ;
    const img2 = try buildImage(allocator, bad_offsets, &.{ 0, 0 });
    defer allocator.free(img2);
    try testing.expectError(Error.DataSizeMismatch, Safetensors.fromBytes(allocator, img2));

    // shape/size disagreement (4 elements declared, 8 bytes of payload)
    const bad_shape =
        \\{"t":{"dtype":"F32","shape":[4],"data_offsets":[0,8]}}
    ;
    const img3 = try buildImage(allocator, bad_shape, &.{ 0, 0, 0, 0, 0, 0, 0, 0 });
    defer allocator.free(img3);
    try testing.expectError(Error.DataSizeMismatch, Safetensors.fromBytes(allocator, img3));

    // reversed offsets
    const reversed =
        \\{"t":{"dtype":"F32","shape":[1],"data_offsets":[4,0]}}
    ;
    const img4 = try buildImage(allocator, reversed, &.{ 0, 0, 0, 0 });
    defer allocator.free(img4);
    try testing.expectError(Error.DataSizeMismatch, Safetensors.fromBytes(allocator, img4));

    // missing keys / non-object entry
    const no_shape =
        \\{"t":{"dtype":"F32","data_offsets":[0,4]}}
    ;
    const img5 = try buildImage(allocator, no_shape, &.{ 0, 0, 0, 0 });
    defer allocator.free(img5);
    try testing.expectError(Error.InvalidTensorEntry, Safetensors.fromBytes(allocator, img5));

    const scalar =
        \\{"t": 5}
    ;
    const img6 = try buildImage(allocator, scalar, &.{});
    defer allocator.free(img6);
    try testing.expectError(Error.InvalidTensorEntry, Safetensors.fromBytes(allocator, img6));
}

test "safetensors: zero-sized tensor and scalar shape" {
    const allocator = testing.allocator;
    const header =
        \\{"empty":{"dtype":"F32","shape":[0],"data_offsets":[0,0]},
        \\ "scalar":{"dtype":"F32","shape":[],"data_offsets":[0,4]}}
    ;
    var payload: [4]u8 = undefined;
    std.mem.writeInt(u32, &payload, @bitCast(@as(f32, 7.5)), .little);
    const image = try buildImage(allocator, header, &payload);
    defer allocator.free(image);
    var st = try Safetensors.fromBytes(allocator, image);
    defer st.deinit();

    try testing.expectEqual(@as(usize, 0), st.tensor("empty").?.numel());
    const empty = try st.readF32(allocator, "empty");
    defer allocator.free(empty);
    try testing.expectEqual(@as(usize, 0), empty.len);

    try testing.expectEqual(@as(usize, 1), st.tensor("scalar").?.numel());
    const scalar = try st.readF32(allocator, "scalar");
    defer allocator.free(scalar);
    try testing.expectEqualSlices(f32, &.{7.5}, scalar);
}

test "safetensors: padded header and escaped/unicode tensor names" {
    const allocator = testing.allocator;
    // Some writers pad the JSON header with spaces to an 8-byte boundary, and
    // tensor names may be JSON-escaped (here a non-ASCII codepoint).
    const header = "{\"t\\u00e9nsor\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]}}   \n ";
    var payload: [4]u8 = undefined;
    std.mem.writeInt(u32, &payload, @bitCast(@as(f32, 3.5)), .little);
    const image = try buildImage(allocator, header, &payload);
    defer allocator.free(image);

    var st = try Safetensors.fromBytes(allocator, image);
    defer st.deinit();
    try testing.expectEqual(@as(usize, 1), st.len());
    try testing.expectEqualStrings("t\xc3\xa9nsor", st.names()[0]);
    const values = try st.readF32(allocator, "t\xc3\xa9nsor");
    defer allocator.free(values);
    try testing.expectEqualSlices(f32, &.{3.5}, values);
}

test "safetensors: dtype table" {
    try testing.expectEqual(Dtype.f32, Dtype.fromString("F32").?);
    try testing.expectEqual(Dtype.bf16, Dtype.fromString("BF16").?);
    try testing.expectEqual(Dtype.bool_, Dtype.fromString("BOOL").?);
    try testing.expectEqual(@as(?Dtype, null), Dtype.fromString("F8_E5M2"));
    try testing.expectEqual(@as(?Dtype, null), Dtype.fromString(""));
    try testing.expectEqual(@as(?usize, 1), @as(?usize, Dtype.bool_.byteSize()));
    try testing.expectEqual(@as(?Dtype, null), Dtype.fromString("nope"));
    try testing.expectEqual(Dtype.u8, Dtype.fromString("U8").?);
    try testing.expectEqual(Dtype.f64, Dtype.fromString("F64").?);
    try testing.expectEqual(Dtype.i64, Dtype.fromString("I64").?);
    try testing.expectEqual(Dtype.i16, Dtype.fromString("I16").?);
    try testing.expectEqual(Dtype.i8, Dtype.fromString("I8").?);
    try testing.expectEqualStrings("BOOL", Dtype.bool_.toString());
    try testing.expectEqual(@as(?usize, 8), Dtype.f64.byteSize());
    try testing.expectEqual(@as(?usize, 2), Dtype.bf16.byteSize());
    try testing.expectEqual(@as(?usize, 1), Dtype.i8.byteSize());
    // non-exhaustive tag fallback
    const bogus: Dtype = @fromBackingInt(@intCast(200));
    try testing.expectEqual(@as(?usize, null), bogus.byteSize());
    try testing.expectEqualStrings("UNKNOWN", bogus.toString());
    var buf: [1]f32 = undefined;
    try testing.expectError(Error.UnsupportedDtype, convertTo(f32, .{
        .name = "x",
        .dtype = bogus,
        .shape = &.{1},
        .data = &.{0},
    }, &buf));
}
