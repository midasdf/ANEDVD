//! Dependency-free (std-only) reader for GGUF v2/v3 model files plus
//! bit-exact ggml dequantization.
//!
//! Design notes
//! ------------
//! * `Gguf.load` memory-maps the file (falling back to a single heap read when
//!   the platform/allocator cannot map it).  Tensor payloads are decoded
//!   straight out of that mapping, so multi-GB files are never copied twice:
//!   `readF32`/`readF16` only materialize the requested tensor.
//! * Metadata strings, tensor names and string arrays are zero-copy slices of
//!   the mapped file; everything else (tensor table, dims, non-string arrays)
//!   lives in an arena owned by the `Gguf` and is released by `deinit`.
//! * Integer/float metadata and tensor payloads are decoded explicitly
//!   little-endian, so the reader is endian-independent.
//! * Quantization follows the published ggml reference (`ggml-quants.c`);
//!   `tests/fixtures/dequant/*.f32` was generated from that reference and
//!   re-verified bit-for-bit against the compiled C code by
//!   `tools/crosscheck_dequant.sh`.

const std = @import("std");
const Allocator = std.mem.Allocator;
const Io = std.Io;

/// GGUF magic: "GGUF" as a little-endian u32.
pub const magic_number: u32 = 0x4655_4747;

/// ggml tensor types, values as stored in the GGUF type field.
/// Non-exhaustive: unknown ids decode to `_` and are rejected by
/// `dequantizeBytes` with `error.UnsupportedType`.
pub const GgmlType = enum(u32) {
    f32 = 0,
    f16 = 1,
    q4_0 = 2,
    q4_1 = 3,
    q5_0 = 6,
    q5_1 = 7,
    q8_0 = 8,
    q8_1 = 9,
    q2_k = 10,
    q3_k = 11,
    q4_k = 12,
    q5_k = 13,
    q6_k = 14,
    q8_k = 15,
    bf16 = 30,
    _,

    /// Number of stored elements per block (1 for unquantized types).
    pub fn blockElems(self: GgmlType) ?usize {
        return switch (self) {
            .f32, .f16, .bf16 => 1,
            .q4_0, .q4_1, .q5_0, .q5_1, .q8_0, .q8_1 => 32,
            .q2_k, .q3_k, .q4_k, .q5_k, .q6_k, .q8_k => 256,
            else => null,
        };
    }

    /// Number of stored bytes per block.
    pub fn blockBytes(self: GgmlType) ?usize {
        return switch (self) {
            .f32 => 4,
            .f16, .bf16 => 2,
            .q4_0 => 18,
            .q4_1 => 20,
            .q5_0 => 22,
            .q5_1 => 24,
            .q8_0 => 34,
            .q8_1 => 36,
            .q2_k => 84,
            .q3_k => 110,
            .q4_k => 144,
            .q5_k => 176,
            .q6_k => 210,
            .q8_k => 292,
            else => null,
        };
    }

    /// True when the type is implemented by `dequantizeBytes`.
    pub fn isSupported(self: GgmlType) bool {
        return self.blockElems() != null;
    }

    pub fn name(self: GgmlType) []const u8 {
        return switch (self) {
            .f32 => "f32",
            .f16 => "f16",
            .q4_0 => "q4_0",
            .q4_1 => "q4_1",
            .q5_0 => "q5_0",
            .q5_1 => "q5_1",
            .q8_0 => "q8_0",
            .q8_1 => "q8_1",
            .q2_k => "q2_k",
            .q3_k => "q3_k",
            .q4_k => "q4_k",
            .q5_k => "q5_k",
            .q6_k => "q6_k",
            .q8_k => "q8_k",
            .bf16 => "bf16",
            else => "unknown",
        };
    }
};

/// One tensor description from the GGUF tensor-info table.
pub const Tensor = struct {
    /// Zero-copy slice of the mapped file.
    name: []const u8,
    /// ggml order: `dims[0]` is the fastest-varying dimension (row length).
    dims: []const u64,
    ttype: GgmlType,
    /// Byte offset from the start of the tensor-data section.
    offset: u64,

    /// Product of all dimensions (saturating on malformed input).
    pub fn elemCount(self: Tensor) u64 {
        var n: u64 = 1;
        for (self.dims) |d| n *|= d;
        return n;
    }

    /// Number of dimensions.
    pub fn nDims(self: Tensor) usize {
        return self.dims.len;
    }

    /// Stored byte size, or null for unsupported/unknown types.
    pub fn byteSize(self: Tensor) ?u64 {
        const epb = self.ttype.blockElems() orelse return null;
        const bpb = self.ttype.blockBytes() orelse return null;
        const n = self.elemCount();
        if (n % epb != 0) return null;
        return std.math.mul(u64, n / epb, bpb) catch null;
    }
};

/// GGUF metadata value types (ids 0..12 as stored in the file).
pub const ValueType = enum(u32) {
    u8 = 0,
    i8 = 1,
    u16 = 2,
    i16 = 3,
    u32 = 4,
    i32 = 5,
    f32 = 6,
    boolean = 7,
    string = 8,
    array = 9,
    u64 = 10,
    i64 = 11,
    f64 = 12,
};

/// Typed backing storage for metadata arrays.
pub const ArrayData = union(enum) {
    u8: []const u8,
    i8: []const i8,
    u16: []const u16,
    i16: []const i16,
    u32: []const u32,
    i32: []const i32,
    f32: []const f32,
    boolean: []const bool,
    string: []const []const u8,
    u64: []const u64,
    i64: []const i64,
    f64: []const f64,

    pub fn len(self: ArrayData) usize {
        return switch (self) {
            inline else => |s| s.len,
        };
    }
};

pub const Array = struct {
    elem_type: ValueType,
    data: ArrayData,
};

pub const Value = union(ValueType) {
    u8: u8,
    i8: i8,
    u16: u16,
    i16: i16,
    u32: u32,
    i32: i32,
    f32: f32,
    boolean: bool,
    string: []const u8,
    array: Array,
    u64: u64,
    i64: i64,
    f64: f64,

    /// Reads any integer kind that fits into a u32.
    pub fn asU32(self: Value) ?u32 {
        return switch (self) {
            .u8 => |v| v,
            .i8 => |v| if (v >= 0) @intCast(v) else null,
            .u16 => |v| v,
            .i16 => |v| if (v >= 0) @intCast(v) else null,
            .u32 => |v| v,
            .i32 => |v| if (v >= 0) @intCast(v) else null,
            .u64 => |v| std.math.cast(u32, v),
            .i64 => |v| std.math.cast(u32, v),
            .boolean => |v| @intFromBool(v),
            else => null,
        };
    }

    pub fn asU64(self: Value) ?u64 {
        return switch (self) {
            .u8 => |v| v,
            .i8 => |v| if (v >= 0) @intCast(v) else null,
            .u16 => |v| v,
            .i16 => |v| if (v >= 0) @intCast(v) else null,
            .u32 => |v| v,
            .i32 => |v| if (v >= 0) @intCast(v) else null,
            .u64 => |v| v,
            .i64 => |v| if (v >= 0) @intCast(v) else null,
            .boolean => |v| @intFromBool(v),
            else => null,
        };
    }

    pub fn asI64(self: Value) ?i64 {
        return switch (self) {
            .u8 => |v| v,
            .i8 => |v| v,
            .u16 => |v| v,
            .i16 => |v| v,
            .u32 => |v| v,
            .i32 => |v| v,
            .u64 => |v| std.math.cast(i64, v),
            .i64 => |v| v,
            .boolean => |v| @intFromBool(v),
            else => null,
        };
    }

    pub fn asF32(self: Value) ?f32 {
        return switch (self) {
            .f32 => |v| v,
            .f64 => |v| @floatCast(v),
            .u8 => |v| @floatFromInt(v),
            .i8 => |v| @floatFromInt(v),
            .u16 => |v| @floatFromInt(v),
            .i16 => |v| @floatFromInt(v),
            .u32 => |v| @floatFromInt(v),
            .i32 => |v| @floatFromInt(v),
            .u64 => |v| @floatFromInt(v),
            .i64 => |v| @floatFromInt(v),
            .boolean => |v| @floatFromInt(@intFromBool(v)),
            else => null,
        };
    }

    pub fn asBool(self: Value) ?bool {
        return switch (self) {
            .boolean => |v| v,
            .u8 => |v| v != 0,
            .i8 => |v| v != 0,
            .u16 => |v| v != 0,
            .i16 => |v| v != 0,
            .u32 => |v| v != 0,
            .i32 => |v| v != 0,
            .u64 => |v| v != 0,
            .i64 => |v| v != 0,
            else => null,
        };
    }

    pub fn asString(self: Value) ?[]const u8 {
        return switch (self) {
            .string => |v| v,
            else => null,
        };
    }

    /// Only succeeds for an array whose element type is `string`.
    pub fn asStringArray(self: Value) ?[]const []const u8 {
        return switch (self) {
            .array => |a| switch (a.data) {
                .string => |s| s,
                else => null,
            },
            else => null,
        };
    }
};

pub const Kv = struct {
    key: []const u8,
    value: Value,
};

/// A parsed GGUF file.  `deinit` releases the mapping/arena.
pub const Gguf = struct {
    allocator: Allocator,
    arena: std.heap.ArenaAllocator,
    /// Whole-file bytes (mapping memory, or an owned heap copy).
    data: []const u8,
    mapping: ?Io.File.MemoryMap = null,
    owned: ?[]u8 = null,
    tensors: []const Tensor = &.{},
    kv: []const Kv = &.{},
    index: std.StringHashMapUnmanaged(u32) = .empty,
    version: u32 = 3,
    alignment: u64 = 32,
    /// Absolute file offset where the tensor-data section starts.
    tensor_data_offset: u64 = 0,

    /// Memory-maps `path` (or reads it once if mapping is unavailable) and
    /// parses the header, metadata and tensor table.
    pub fn load(allocator: Allocator, path: []const u8) !Gguf {
        return loadWithIo(allocator, Io.Threaded.global_single_threaded.io(), path);
    }

    /// Like `load`, but uses a caller-provided `std.Io` implementation.
    pub fn loadWithIo(allocator: Allocator, io: Io, path: []const u8) !Gguf {
        var file = try Io.Dir.cwd().openFile(io, path, .{});
        defer file.close(io);
        const stat = try file.stat(io);
        const size = stat.size;
        if (size == 0) return error.TruncatedFile;
        if (size > std.math.maxInt(usize)) return error.FileTooBig;

        var g = Gguf{ .allocator = allocator, .arena = std.heap.ArenaAllocator.init(allocator), .data = &.{} };
        errdefer g.deinit();

        if (Io.File.MemoryMap.create(io, file, .{
            .len = @intCast(size),
            .protection = .{ .read = true, .write = false },
        })) |mm| {
            g.mapping = mm;
            g.data = mm.memory;
        } else |_| {
            // Fallback for platforms/Io implementations without memory mapping.
            const buf = try allocator.alloc(u8, @intCast(size));
            errdefer allocator.free(buf);
            var reader = file.reader(io, &.{});
            // readSliceAll fills the whole buffer or returns an error.
            try reader.interface.readSliceAll(buf);
            g.owned = buf;
            g.data = buf;
        }

        try g.parse();
        return g;
    }

    /// Parses an in-memory image.  The bytes are copied, so the caller may
    /// free `bytes` immediately.
    pub fn fromBytes(allocator: Allocator, bytes: []const u8) !Gguf {
        var g = Gguf{ .allocator = allocator, .arena = std.heap.ArenaAllocator.init(allocator), .data = &.{} };
        errdefer g.deinit();
        const copy = try allocator.alloc(u8, bytes.len);
        @memcpy(copy, bytes);
        g.owned = copy;
        g.data = copy;
        try g.parse();
        return g;
    }

    pub fn deinit(self: *Gguf) void {
        if (self.mapping) |*mm| mm.destroy(Io.Threaded.global_single_threaded.io());
        if (self.owned) |buf| self.allocator.free(buf);
        self.arena.deinit();
        self.* = undefined;
    }

    /// Number of tensors in the file.
    pub fn tensorCount(self: *const Gguf) usize {
        return self.tensors.len;
    }

    /// Tensor `i` (asserts `i < tensorCount()`).
    pub fn tensorAt(self: *const Gguf, i: usize) Tensor {
        std.debug.assert(i < self.tensors.len);
        return self.tensors[i];
    }

    /// Looks a tensor up by name.
    pub fn tensor(self: *const Gguf, name: []const u8) ?Tensor {
        const i = self.index.get(name) orelse return null;
        return self.tensors[i];
    }

    /// All metadata key/value pairs in file order.
    pub fn metadata(self: *const Gguf) []const Kv {
        return self.kv;
    }

    /// Raw metadata value for `key`.
    pub fn getValue(self: *const Gguf, key: []const u8) ?Value {
        const i = self.index.get(key) orelse return null;
        return self.kv[i].value;
    }

    pub fn getString(self: *const Gguf, key: []const u8) ?[]const u8 {
        return (self.getValue(key) orelse return null).asString();
    }

    pub fn getU32(self: *const Gguf, key: []const u8) ?u32 {
        return (self.getValue(key) orelse return null).asU32();
    }

    pub fn getU64(self: *const Gguf, key: []const u8) ?u64 {
        return (self.getValue(key) orelse return null).asU64();
    }

    pub fn getI32(self: *const Gguf, key: []const u8) ?i32 {
        return std.math.cast(i32, (self.getValue(key) orelse return null).asI64() orelse return null);
    }

    pub fn getF32(self: *const Gguf, key: []const u8) ?f32 {
        return (self.getValue(key) orelse return null).asF32();
    }

    pub fn getBool(self: *const Gguf, key: []const u8) ?bool {
        return (self.getValue(key) orelse return null).asBool();
    }

    /// Only succeeds for an array of strings (e.g. `tokenizer.ggml.tokens`).
    pub fn getStringArray(self: *const Gguf, key: []const u8) ?[]const []const u8 {
        return (self.getValue(key) orelse return null).asStringArray();
    }

    /// `general.architecture`.
    pub fn arch(self: *const Gguf) ?[]const u8 {
        return self.getString("general.architecture");
    }

    /// Borrows the mapped bytes backing tensor `t`.
    pub fn tensorBytes(self: *const Gguf, t: Tensor) ![]const u8 {
        const bpb = t.ttype.blockBytes() orelse return error.UnsupportedType;
        const epb = t.ttype.blockElems() orelse return error.UnsupportedType;
        const n = t.elemCount();
        if (n % epb != 0) return error.InvalidTensorShape;
        const nbytes = std.math.mul(u64, n / epb, bpb) catch return error.TensorDataOutOfRange;
        const start = std.math.add(u64, self.tensor_data_offset, t.offset) catch return error.TensorDataOutOfRange;
        const end = std.math.add(u64, start, nbytes) catch return error.TensorDataOutOfRange;
        if (end > self.data.len) return error.TensorDataOutOfRange;
        return self.data[@intCast(start)..@intCast(end)];
    }

    /// Dequantizes tensor `name` into a freshly allocated f32 slice.
    /// The caller owns the result.
    pub fn readF32(self: *const Gguf, allocator: Allocator, name: []const u8) ![]f32 {
        const t = self.tensor(name) orelse return error.TensorNotFound;
        const n = t.elemCount();
        if (n > std.math.maxInt(usize)) return error.TensorTooLarge;
        const out = try allocator.alloc(f32, @intCast(n));
        errdefer allocator.free(out);
        try self.dequantizeTensor(t, out);
        return out;
    }

    /// Dequantizes tensor `name` into f16 (lossy for types wider than f16).
    /// The caller owns the result.
    /// Bytes of ONE expert inside a stacked expert tensor.
    ///
    /// `blk.N.ffn_{gate,up,down}_exps` keeps its experts on the slowest axis, so
    /// expert `e` of `n_experts` is the contiguous range `[e/n, (e+1)/n)` of the
    /// tensor's payload. This is what makes expert streaming possible: a 9.5 GB
    /// MoE cannot be resident on an 8 GB machine, but the four experts a token
    /// actually routes to are a few MB, and they can be read without touching the
    /// other 56.
    pub fn expertBytes(self: *const Gguf, name: []const u8, expert: u32, n_experts: u32) ![]const u8 {
        if (n_experts == 0 or expert >= n_experts) return error.InvalidTensorShape;
        const all = try self.tensorBytes(self.tensor(name) orelse return error.TensorNotFound);
        if (all.len % n_experts != 0) return error.InvalidTensorShape;
        const per = all.len / n_experts;
        return all[@as(usize, expert) * per ..][0..per];
    }

    /// Dequantise ONE expert of a stacked expert tensor into `out` as fp16.
    ///
    /// This is the streaming path: it touches only that expert's bytes, so the
    /// other `n_experts - 1` never become resident. On Qwen1.5-MoE (9.5 GB file,
    /// 8 GB machine) the eager alternative cannot run at all.
    ///
    /// The expert is a contiguous 1/n slice of the payload because the expert axis
    /// is ggml's slowest, and the slice starts on a block boundary since every
    /// block size divides the per-expert element count.
    pub fn readExpertF16(
        self: *const Gguf,
        name: []const u8,
        expert: u32,
        n_experts: u32,
        out: []f16,
    ) !void {
        const t = self.tensor(name) orelse return error.TensorNotFound;
        const per_expert: u64 = @as(u64, @intCast(out.len));
        if (t.elemCount() != per_expert * n_experts) return error.DimensionMismatch;
        const epb = t.ttype.blockElems() orelse return error.UnsupportedType;
        if (per_expert % epb != 0) return error.InvalidTensorShape;
        const bpb = t.ttype.blockBytes() orelse return error.UnsupportedType;
        const per_bytes = per_expert / epb * bpb;

        const start = self.tensor_data_offset + t.offset + @as(u64, expert) * per_bytes;
        const end = start + per_bytes;
        if (end > self.data.len) return error.TensorDataOutOfRange;
        const src = self.data[@intCast(start)..@intCast(end)];

        var buf: [2048]f32 = undefined;
        var done: usize = 0;
        while (done < out.len) {
            const m = @min(buf.len, out.len - done);
            try dequantizeRange(t.ttype, src, done, buf[0..m]);
            f32ToF16Slice(buf[0..m], out[done..][0..m]);
            done += m;
        }
    }

    pub fn readF16(self: *const Gguf, allocator: Allocator, name: []const u8) ![]f16 {
        const t = self.tensor(name) orelse return error.TensorNotFound;
        const n = t.elemCount();
        if (n > std.math.maxInt(usize)) return error.TensorTooLarge;
        const out = try allocator.alloc(f16, @intCast(n));
        errdefer allocator.free(out);
        const src = try self.tensorBytes(t);

        if (t.ttype == .f16) {
            for (out, 0..) |*o, i| o.* = @bitCast(readU16(src, 2 * i));
            return out;
        }

        // Chunked so that wide types never need a full f32 copy of the tensor.
        // 2048 is a multiple of every block size we support (1, 32, 256).
        var buf: [2048]f32 = undefined;
        var done: usize = 0;
        while (done < out.len) {
            const m = @min(buf.len, out.len - done);
            try dequantizeRange(t.ttype, src, done, buf[0..m]);
            f32ToF16Slice(buf[0..m], out[done..][0..m]);
            done += m;
        }
        return out;
    }

    /// Dequantizes one tensor into `out` (`out.len` must equal `elemCount`).
    pub fn dequantizeTensor(self: *const Gguf, t: Tensor, out: []f32) !void {
        if (out.len != t.elemCount()) return error.InvalidTensorShape;
        const src = try self.tensorBytes(t);
        try dequantizeBytes(t.ttype, src, out);
    }

    // ------------------------------------------------------------- parsing

    fn parse(self: *Gguf) !void {
        const a = self.arena.allocator();
        var cur = Cursor{ .data = self.data };

        if (try cur.takeU32() != magic_number) return error.NotGgufFile;
        const version = try cur.takeU32();
        if (version < 2 or version > 3) return error.UnsupportedVersion;
        self.version = version;

        const tensor_count = try cur.takeU64();
        const kv_count = try cur.takeU64();
        // Every entry needs at least a few bytes; reject absurd counts early
        // so that corrupt headers cannot trigger huge allocations.
        if (tensor_count > cur.remaining() + 1) return error.TooManyTensors;
        if (kv_count > cur.remaining() + 1) return error.TooManyMetadataEntries;

        const kv = try a.alloc(Kv, @intCast(kv_count));
        for (kv) |*entry| {
            const key = try cur.takeString();
            const value = try readValue(&cur, a);
            entry.* = .{ .key = key, .value = value };
        }
        self.kv = kv;

        try self.index.ensureTotalCapacity(a, @intCast(kv_count));
        for (kv, 0..) |entry, i| {
            // Duplicate keys: first one wins, like llama.cpp.
            const gop = self.index.getOrPutAssumeCapacity(entry.key);
            if (!gop.found_existing) gop.value_ptr.* = @intCast(i);
        }

        if (self.getValue("general.alignment")) |v| {
            const align_val = v.asU64() orelse return error.InvalidMetadata;
            if (align_val == 0 or align_val > (1 << 24) or !std.math.isPowerOfTwo(align_val)) {
                return error.InvalidMetadata;
            }
            self.alignment = align_val;
        }

        const tensors = try a.alloc(Tensor, @intCast(tensor_count));
        for (tensors) |*t| {
            const name = try cur.takeString();
            const n_dims = try cur.takeU32();
            if (n_dims == 0 or n_dims > 8) return error.InvalidTensorShape;
            const dims = try a.alloc(u64, n_dims);
            for (dims) |*d| d.* = try cur.takeU64();
            const raw_type = try cur.takeU32();
            const offset = try cur.takeU64();
            t.* = .{
                .name = name,
                .dims = dims,
                .ttype = @fromBackingInt(@intCast(raw_type)),
                .offset = offset,
            };
        }
        self.tensors = tensors;

        // Tensor names live in the same lookup map as metadata keys, so the
        // whole table has to be reserved before inserting them.
        try self.index.ensureTotalCapacity(a, @intCast(kv_count + tensor_count));
        for (tensors, 0..) |t, i| {
            const gop = self.index.getOrPutAssumeCapacity(t.name);
            if (!gop.found_existing) gop.value_ptr.* = @intCast(i);
        }

        self.tensor_data_offset = alignForward(cur.pos, self.alignment);
        if (self.tensor_data_offset > self.data.len) return error.TruncatedFile;
    }
};

// ------------------------------------------------------------- dequantizing

const raw_f16 = f16;

/// Dequantizes `src` (whole number of blocks) into `out`.
/// `error.UnsupportedType` for ggml types this reader does not implement,
/// `error.InvalidTensorShape`/`error.TruncatedFile` for malformed input.
pub fn dequantizeBytes(ttype: GgmlType, src: []const u8, out: []f32) !void {
    return dequantizeRange(ttype, src, 0, out);
}

/// Like `dequantizeBytes`, but starts at element `elem_offset` (which must be
/// a multiple of the type's block size).  Used to convert wide types into f16
/// in bounded chunks.
/// Convert f32 to fp16 a vector at a time.
///
/// The scalar `@floatCast` loop this replaces was the gap between the raw
/// dequantiser (733 M elements/s) and the streaming reader (286 M elements/s): the
/// dequantise itself was never the wall, the trailing conversion pass was.
///
/// `@floatCast` on a vector does the same round-to-nearest conversion the scalar
/// form does, so results are unchanged — unlike a bit-truncation, which is faster
/// still and silently wrong in the last bit.
pub fn f32ToF16Slice(src: []const f32, dst: []f16) void {
    std.debug.assert(src.len == dst.len);
    const L = 8;
    var i: usize = 0;
    while (i + L <= src.len) : (i += L) {
        const v: @Vector(L, f32) = src[i..][0..L].*;
        const h: @Vector(L, f16) = @floatCast(v);
        dst[i..][0..L].* = @bitCast(h);
    }
    while (i < src.len) : (i += 1) dst[i] = @floatCast(src[i]);
}

pub fn dequantizeRange(ttype: GgmlType, src: []const u8, elem_offset: u64, out: []f32) !void {
    const epb = ttype.blockElems() orelse return error.UnsupportedType;
    const bpb = ttype.blockBytes() orelse return error.UnsupportedType;
    if (out.len == 0) return;
    if (out.len % epb != 0) return error.InvalidTensorShape;
    if (elem_offset % epb != 0) return error.InvalidTensorShape;

    const start = std.math.mul(u64, elem_offset / epb, bpb) catch return error.TruncatedFile;
    if (start > src.len) return error.TruncatedFile;
    const body = src[@intCast(start)..];

    const blocks = out.len / epb;
    const need = std.math.mul(usize, blocks, bpb) catch return error.TruncatedFile;
    if (body.len < need) return error.TruncatedFile;
    const b = body[0..need];

    switch (ttype) {
        .f32 => dequantF32(b, out),
        .f16 => dequantF16(b, out),
        .bf16 => dequantBf16(b, out),
        .q4_0 => dequantQ4_0(b, out),
        .q4_1 => dequantQ4_1(b, out),
        .q5_0 => dequantQ5_0(b, out),
        .q5_1 => dequantQ5_1(b, out),
        .q8_0 => dequantQ8_0(b, out),
        .q8_1 => dequantQ8_1(b, out),
        .q2_k => dequantQ2K(b, out),
        .q3_k => dequantQ3K(b, out),
        .q4_k => dequantQ4K(b, out),
        .q5_k => dequantQ5K(b, out),
        .q6_k => dequantQ6K(b, out),
        .q8_k => dequantQ8K(b, out),
        else => return error.UnsupportedType,
    }
}

inline fn readU16(b: []const u8, off: usize) u16 {
    return std.mem.readInt(u16, b[off..][0..2], .little);
}

inline fn readU32(b: []const u8, off: usize) u32 {
    return std.mem.readInt(u32, b[off..][0..4], .little);
}

inline fn readI8(b: []const u8, off: usize) i8 {
    return @bitCast(b[off]);
}

inline fn f16ToF32(h: u16) f32 {
    const v: raw_f16 = @bitCast(h);
    return @floatCast(v);
}

inline fn f32ToBf16Bits(bits: u16) f32 {
    return @bitCast(@as(u32, bits) << 16);
}

fn dequantF32(b: []const u8, out: []f32) void {
    for (out, 0..) |*o, i| o.* = @bitCast(readU32(b, 4 * i));
}

fn dequantF16(b: []const u8, out: []f32) void {
    for (out, 0..) |*o, i| o.* = f16ToF32(readU16(b, 2 * i));
}

fn dequantBf16(b: []const u8, out: []f32) void {
    for (out, 0..) |*o, i| o.* = f32ToBf16Bits(readU16(b, 2 * i));
}

fn dequantQ4_0(b: []const u8, out: []f32) void {
    const qk = 32;
    for (0..out.len / qk) |i| {
        const blk = b[i * 18 ..][0..18];
        const d = f16ToF32(readU16(blk, 0));
        for (0..qk / 2) |j| {
            const byte = blk[2 + j];
            const x0: f32 = @floatFromInt(@as(i32, byte & 0x0F) - 8);
            const x1: f32 = @floatFromInt(@as(i32, byte >> 4) - 8);
            out[i * qk + j] = x0 * d;
            out[i * qk + j + qk / 2] = x1 * d;
        }
    }
}

fn dequantQ4_1(b: []const u8, out: []f32) void {
    const qk = 32;
    for (0..out.len / qk) |i| {
        const blk = b[i * 20 ..][0..20];
        const d = f16ToF32(readU16(blk, 0));
        const m = f16ToF32(readU16(blk, 2));
        for (0..qk / 2) |j| {
            const byte = blk[4 + j];
            const x0: f32 = @floatFromInt(byte & 0x0F);
            const x1: f32 = @floatFromInt(byte >> 4);
            out[i * qk + j] = x0 * d + m;
            out[i * qk + j + qk / 2] = x1 * d + m;
        }
    }
}

/// Shared shape of q5_0/q5_1: fp16 pair header, u32 high bits, low nibbles.
fn dequantQ5(b: []const u8, out: []f32, comptime block_bytes: usize, comptime has_min: bool) void {
    const qk = 32;
    const qh_off = if (has_min) @as(usize, 4) else 2;
    const qs_off = qh_off + 4;
    const minus: i32 = if (has_min) 0 else 16;
    for (0..out.len / qk) |i| {
        const blk = b[i * block_bytes ..][0..block_bytes];
        const d = f16ToF32(readU16(blk, 0));
        const m = if (has_min) f16ToF32(readU16(blk, 2)) else 0.0;
        const qh = readU32(blk, qh_off);
        for (0..qk / 2) |j| {
            const byte = blk[qs_off + j];
            const sh_lo: u5 = @intCast(j);
            const sh_hi: u5 = @intCast(j + 12);
            const xh_0: u32 = ((qh >> sh_lo) << 4) & 0x10;
            const xh_1: u32 = (qh >> sh_hi) & 0x10;
            const x0: f32 = @floatFromInt(@as(i32, @intCast((byte & 0x0F) | xh_0)) - minus);
            const x1: f32 = @floatFromInt(@as(i32, @intCast((byte >> 4) | xh_1)) - minus);
            out[i * qk + j] = x0 * d + m;
            out[i * qk + j + qk / 2] = x1 * d + m;
        }
    }
}

fn dequantQ5_0(b: []const u8, out: []f32) void {
    dequantQ5(b, out, 22, false);
}

fn dequantQ5_1(b: []const u8, out: []f32) void {
    dequantQ5(b, out, 24, true);
}

fn dequantQ8_0(b: []const u8, out: []f32) void {
    const qk = 32;
    for (0..out.len / qk) |i| {
        const blk = b[i * 34 ..][0..34];
        const d = f16ToF32(readU16(blk, 0));
        // Vectorised: 32 contiguous i8 -> f32 -> scale. Q8_0 is a third of the
        // real Qwen1.5-MoE's expert traffic (every down_exps tensor), so it is
        // worth the same treatment as Q4_K. Identical arithmetic, pinned by the
        // dequant fixtures.
        const raw: @Vector(32, i8) = @bitCast(blk[2..][0..32].*);
        const q: @Vector(32, f32) = @floatFromInt(raw);
        out[i * qk ..][0..32].* = @as(@Vector(32, f32), @splat(d)) * q;
    }
}

fn dequantQ8_1(b: []const u8, out: []f32) void {
    // `s` (offset 2) is d*sum(qs) and is not needed for dequantization.
    const qk = 32;
    for (0..out.len / qk) |i| {
        const blk = b[i * 36 ..][0..36];
        const d = f16ToF32(readU16(blk, 0));
        for (0..qk) |j| {
            const q: f32 = @floatFromInt(readI8(blk, 4 + j));
            out[i * qk + j] = q * d;
        }
    }
}

fn dequantQ2K(b: []const u8, out: []f32) void {
    const qk = 256;
    for (0..out.len / qk) |ib| {
        const blk = b[ib * 84 ..][0..84];
        const d = f16ToF32(readU16(blk, 80));
        const min = f16ToF32(readU16(blk, 82));
        const scales = blk[0..16];
        var q: usize = 16;
        var y: usize = ib * qk;
        var is: usize = 0;
        for (0..2) |_| {
            var shift: u8 = 0;
            for (0..4) |_| {
                var sc = scales[is];
                is += 1;
                var dl = d * @as(f32, @floatFromInt(sc & 0x0F));
                var ml = min * @as(f32, @floatFromInt(sc >> 4));
                for (0..16) |l| {
                    const v: f32 = @floatFromInt((blk[q + l] >> @intCast(shift)) & 3);
                    out[y + l] = dl * v - ml;
                }
                y += 16;
                sc = scales[is];
                is += 1;
                dl = d * @as(f32, @floatFromInt(sc & 0x0F));
                ml = min * @as(f32, @floatFromInt(sc >> 4));
                for (0..16) |l| {
                    const v: f32 = @floatFromInt((blk[q + l + 16] >> @intCast(shift)) & 3);
                    out[y + l] = dl * v - ml;
                }
                y += 16;
                shift += 2;
            }
            q += 32;
        }
    }
}

fn dequantQ3K(b: []const u8, out: []f32) void {
    const qk = 256;
    const kmask1: u32 = 0x0303_0303;
    const kmask2: u32 = 0x0f0f_0f0f;
    for (0..out.len / qk) |ib| {
        const blk = b[ib * 110 ..][0..110];
        const d_all = f16ToF32(readU16(blk, 108));
        const hm = blk[0..32];
        const qs = blk[32..96];

        // Unpack the 16 6-bit scales stored in the 12 bytes at offset 96.
        var aux = [4]u32{ readU32(blk, 96), readU32(blk, 100), readU32(blk, 104), 0 };
        const tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
        var scales: [16]i8 = undefined;
        for (0..4) |k| {
            const w = aux[k];
            for (0..4) |byte_i| {
                const byte: u8 = @truncate(w >> @intCast(8 * byte_i));
                scales[k * 4 + byte_i] = @bitCast(byte);
            }
        }

        var m: u8 = 1;
        var is: usize = 0;
        var q: usize = 0;
        var y: usize = ib * qk;
        for (0..2) |_| {
            var shift: u8 = 0;
            for (0..4) |_| {
                var dl = d_all * @as(f32, @floatFromInt(@as(i32, scales[is]) - 32));
                is += 1;
                for (0..16) |l| {
                    const low: i32 = @intCast((qs[q + l] >> @intCast(shift)) & 3);
                    const v: i32 = low - (if (hm[l] & m != 0) @as(i32, 0) else 4);
                    out[y + l] = dl * @as(f32, @floatFromInt(v));
                }
                y += 16;
                dl = d_all * @as(f32, @floatFromInt(@as(i32, scales[is]) - 32));
                is += 1;
                for (0..16) |l| {
                    const low: i32 = @intCast((qs[q + l + 16] >> @intCast(shift)) & 3);
                    const v: i32 = low - (if (hm[l + 16] & m != 0) @as(i32, 0) else 4);
                    out[y + l] = dl * @as(f32, @floatFromInt(v));
                }
                y += 16;
                shift += 2;
                m <<= 1;
            }
            q += 32;
        }
    }
}

/// `get_scale_min_k4` from ggml-quants.c: 6-bit scale/min pairs packed in
/// `scales[12]`.
inline fn getScaleMinK4(j: usize, q: []const u8) struct { d: u8, m: u8 } {
    if (j < 4) {
        return .{ .d = q[j] & 63, .m = q[j + 4] & 63 };
    }
    return .{
        .d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4),
        .m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4),
    };
}

fn dequantQ4K(b: []const u8, out: []f32) void {
    const qk = 256;
    for (0..out.len / qk) |ib| {
        const blk = b[ib * 144 ..][0..144];
        const d = f16ToF32(readU16(blk, 0));
        const min = f16ToF32(readU16(blk, 2));
        const scales = blk[4..16];
        var q: usize = 16;
        var is: usize = 0;
        for (0..4) |_| {
            const s0 = getScaleMinK4(is, scales);
            const d1 = d * @as(f32, @floatFromInt(s0.d));
            const m1 = min * @as(f32, @floatFromInt(s0.m));
            const s1 = getScaleMinK4(is + 1, scales);
            const d2 = d * @as(f32, @floatFromInt(s1.d));
            const m2 = min * @as(f32, @floatFromInt(s1.m));
            const y = ib * qk + (is / 2) * 64;
            // Vectorised: 32 contiguous bytes become 32 f32 in two steps. The
            // arithmetic is identical to the scalar form, so the dequant fixtures
            // still pin it bit-for-bit; what changes is that the compiler emits one
            // convert and one multiply-add per 32 elements instead of scalar ops.
            //
            // This inner loop is the measured wall on the MoE streaming path
            // (219 MB/s, cold and warm alike), and it also runs once per layer for
            // every dense GGUF load.
            const raw: @Vector(32, u8) = blk[q..][0..32].*;
            const lo: @Vector(32, f32) = @floatFromInt(raw & @as(@Vector(32, u8), @splat(0x0F)));
            const hi: @Vector(32, f32) = @floatFromInt(raw >> @as(@Vector(32, u3), @splat(4)));
            out[y..][0..32].* = @as(@Vector(32, f32), @splat(d1)) * lo - @as(@Vector(32, f32), @splat(m1));
            out[y + 32 ..][0..32].* = @as(@Vector(32, f32), @splat(d2)) * hi - @as(@Vector(32, f32), @splat(m2));
            q += 32;
            is += 2;
        }
    }
}

fn dequantQ5K(b: []const u8, out: []f32) void {
    const qk = 256;
    for (0..out.len / qk) |ib| {
        const blk = b[ib * 176 ..][0..176];
        const d = f16ToF32(readU16(blk, 0));
        const min = f16ToF32(readU16(blk, 2));
        const scales = blk[4..16];
        const qh = blk[16..48];
        var ql: usize = 48;
        var is: usize = 0;
        var mask1: u8 = 1;
        var mask2: u8 = 2;
        for (0..4) |_| {
            const s0 = getScaleMinK4(is, scales);
            const d1 = d * @as(f32, @floatFromInt(s0.d));
            const m1 = min * @as(f32, @floatFromInt(s0.m));
            const s1 = getScaleMinK4(is + 1, scales);
            const d2 = d * @as(f32, @floatFromInt(s1.d));
            const m2 = min * @as(f32, @floatFromInt(s1.m));
            const y = ib * qk + (is / 2) * 64;
            for (0..32) |l| {
                const hi: u8 = if (qh[l] & mask1 != 0) 16 else 0;
                const v: f32 = @floatFromInt((blk[ql + l] & 0x0F) + hi);
                out[y + l] = d1 * v - m1;
            }
            for (0..32) |l| {
                const hi: u8 = if (qh[l] & mask2 != 0) 16 else 0;
                const v: f32 = @floatFromInt((blk[ql + l] >> 4) + hi);
                out[y + 32 + l] = d2 * v - m2;
            }
            ql += 32;
            is += 2;
            mask1 <<= 2;
            mask2 <<= 2;
        }
    }
}

fn dequantQ6K(b: []const u8, out: []f32) void {
    const qk = 256;
    for (0..out.len / qk) |ib| {
        const blk = b[ib * 210 ..][0..210];
        const d = f16ToF32(readU16(blk, 208));
        for (0..2) |half| {
            const ql = blk[half * 64 ..][0..64];
            const qh = blk[128 + half * 32 ..][0..32];
            const sc = blk[192 + half * 8 ..][0..8];
            const y = ib * qk + half * 128;
            for (0..32) |l| {
                const is = l / 16;
                const q1: i32 = @as(i32, (ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                const q2: i32 = @as(i32, (ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                const q3: i32 = @as(i32, (ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                const q4: i32 = @as(i32, (ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                out[y + l + 0] = d * @as(f32, @floatFromInt(@as(i8, @bitCast(sc[is + 0])))) * @as(f32, @floatFromInt(q1));
                out[y + l + 32] = d * @as(f32, @floatFromInt(@as(i8, @bitCast(sc[is + 2])))) * @as(f32, @floatFromInt(q2));
                out[y + l + 64] = d * @as(f32, @floatFromInt(@as(i8, @bitCast(sc[is + 4])))) * @as(f32, @floatFromInt(q3));
                out[y + l + 96] = d * @as(f32, @floatFromInt(@as(i8, @bitCast(sc[is + 6])))) * @as(f32, @floatFromInt(q4));
            }
        }
    }
}

fn dequantQ8K(b: []const u8, out: []f32) void {
    const qk = 256;
    for (0..out.len / qk) |ib| {
        const blk = b[ib * 292 ..][0..292];
        const d: f32 = @bitCast(readU32(blk, 0));
        for (0..qk) |j| {
            const q: f32 = @floatFromInt(readI8(blk, 4 + j));
            out[ib * qk + j] = q * d;
        }
    }
}

// ------------------------------------------------------------------ helpers

/// 0.17 removed the `**` array-repeat operator.
fn zeros(comptime n: usize) [n]u8 {
    return @splat(0);
}

const Cursor = struct {
    data: []const u8,
    pos: usize = 0,

    fn remaining(self: Cursor) usize {
        return self.data.len - self.pos;
    }

    fn need(self: Cursor, n: usize) !void {
        if (self.remaining() < n) return error.TruncatedFile;
    }

    fn takeU8(self: *Cursor) !u8 {
        try self.need(1);
        const v = self.data[self.pos];
        self.pos += 1;
        return v;
    }

    fn takeU16(self: *Cursor) !u16 {
        try self.need(2);
        const v = readU16(self.data, self.pos);
        self.pos += 2;
        return v;
    }

    fn takeU32(self: *Cursor) !u32 {
        try self.need(4);
        const v = readU32(self.data, self.pos);
        self.pos += 4;
        return v;
    }

    fn takeU64(self: *Cursor) !u64 {
        try self.need(8);
        const v = std.mem.readInt(u64, self.data[self.pos..][0..8], .little);
        self.pos += 8;
        return v;
    }

    /// GGUF strings are `u64 len` + raw bytes (no terminator).
    fn takeString(self: *Cursor) ![]const u8 {
        const n = try self.takeU64();
        if (n > self.remaining()) return error.TruncatedFile;
        const len: usize = @intCast(n);
        const s = self.data[self.pos .. self.pos + len];
        self.pos += len;
        return s;
    }
};

fn readValue(cur: *Cursor, a: Allocator) !Value {
    const raw = try cur.takeU32();
    const vt = std.enums.fromInt(ValueType, raw) orelse return error.UnsupportedMetadataType;
    return switch (vt) {
        .u8 => .{ .u8 = try cur.takeU8() },
        .i8 => .{ .i8 = @bitCast(try cur.takeU8()) },
        .u16 => .{ .u16 = try cur.takeU16() },
        .i16 => .{ .i16 = @bitCast(try cur.takeU16()) },
        .u32 => .{ .u32 = try cur.takeU32() },
        .i32 => .{ .i32 = @bitCast(try cur.takeU32()) },
        .f32 => .{ .f32 = @bitCast(try cur.takeU32()) },
        .boolean => .{ .boolean = (try cur.takeU8()) != 0 },
        .string => .{ .string = try cur.takeString() },
        .u64 => .{ .u64 = try cur.takeU64() },
        .i64 => .{ .i64 = @bitCast(try cur.takeU64()) },
        .f64 => .{ .f64 = @bitCast(try cur.takeU64()) },
        .array => .{ .array = try readArray(cur, a) },
    };
}

fn readArray(cur: *Cursor, a: Allocator) !Array {
    const raw = try cur.takeU32();
    const et = std.enums.fromInt(ValueType, raw) orelse return error.UnsupportedMetadataType;
    if (et == .array) return error.UnsupportedMetadataType; // nested arrays do not exist in GGUF
    const count64 = try cur.takeU64();
    if (count64 > cur.remaining() + 1) return error.TruncatedFile;
    const count: usize = @intCast(count64);

    const data: ArrayData = switch (et) {
        .u8 => .{ .u8 = blk: {
            const s = try a.alloc(u8, count);
            for (s) |*v| v.* = try cur.takeU8();
            break :blk s;
        } },
        .i8 => .{ .i8 = blk: {
            const s = try a.alloc(i8, count);
            for (s) |*v| v.* = @bitCast(try cur.takeU8());
            break :blk s;
        } },
        .u16 => .{ .u16 = blk: {
            const s = try a.alloc(u16, count);
            for (s) |*v| v.* = try cur.takeU16();
            break :blk s;
        } },
        .i16 => .{ .i16 = blk: {
            const s = try a.alloc(i16, count);
            for (s) |*v| v.* = @bitCast(try cur.takeU16());
            break :blk s;
        } },
        .u32 => .{ .u32 = blk: {
            const s = try a.alloc(u32, count);
            for (s) |*v| v.* = try cur.takeU32();
            break :blk s;
        } },
        .i32 => .{ .i32 = blk: {
            const s = try a.alloc(i32, count);
            for (s) |*v| v.* = @bitCast(try cur.takeU32());
            break :blk s;
        } },
        .f32 => .{ .f32 = blk: {
            const s = try a.alloc(f32, count);
            for (s) |*v| v.* = @bitCast(try cur.takeU32());
            break :blk s;
        } },
        .u64 => .{ .u64 = blk: {
            const s = try a.alloc(u64, count);
            for (s) |*v| v.* = try cur.takeU64();
            break :blk s;
        } },
        .i64 => .{ .i64 = blk: {
            const s = try a.alloc(i64, count);
            for (s) |*v| v.* = @bitCast(try cur.takeU64());
            break :blk s;
        } },
        .f64 => .{ .f64 = blk: {
            const s = try a.alloc(f64, count);
            for (s) |*v| v.* = @bitCast(try cur.takeU64());
            break :blk s;
        } },
        .boolean => .{ .boolean = blk: {
            const s = try a.alloc(bool, count);
            for (s) |*v| v.* = (try cur.takeU8()) != 0;
            break :blk s;
        } },
        .string => .{ .string = blk: {
            const s = try a.alloc([]const u8, count);
            for (s) |*v| v.* = try cur.takeString();
            break :blk s;
        } },
        .array => unreachable,
    };
    return .{ .elem_type = et, .data = data };
}

fn alignForward(offset: usize, alignment: u64) u64 {
    const a: usize = @intCast(alignment);
    return ((offset + a - 1) / a) * a;
}

// ===========================================================================
// Tests
// ===========================================================================

const testing = std.testing;

/// `<repo>/tests/fixtures`, derived from this file's own path so the tests do
/// not depend on the process working directory.
/// Resolve a path under `<repo>/tests/fixtures`.
///
/// Zig 0.17's `@src().file` is only the basename, so the repository root cannot
/// be derived from it; try the plausible roots in order instead.
pub fn fixturePath(a: Allocator, rel: []const u8) ![]u8 {
    const candidates = [_][]const u8{ "tests/fixtures", "src/../tests/fixtures", "../tests/fixtures" };
    for (candidates) |c| {
        const p = try std.fs.path.join(a, &.{ c, rel });
        if (std.fs.path.isAbsolute(p)) return p;
        // Keep the first candidate: callers report FileNotFound themselves.
        if (std.mem.eql(u8, c, "tests/fixtures")) return p;
        a.free(p);
    }
    return error.FileNotFound;
}

fn readFixture(a: Allocator, rel: []const u8) ![]u8 {
    const path = try fixturePath(a, rel);
    defer a.free(path);
    return Io.Dir.cwd().readFileAlloc(testing.io, path, a, .limited(1 << 28));
}

fn expectBitEqual(expected: []const f32, actual: []const f32) !void {
    try testing.expectEqual(expected.len, actual.len);
    for (expected, actual, 0..) |e, got, i| {
        const eb: u32 = @bitCast(e);
        const gb: u32 = @bitCast(got);
        if (eb != gb) {
            std.debug.print("element {d}: expected {d} (0x{x:0>8}), got {d} (0x{x:0>8})\n", .{ i, e, eb, got, gb });
            return error.TestExpectedEqual;
        }
    }
}

/// Minimal GGUF v3 writer used to build test files in memory.
const Builder = struct {
    a: Allocator,
    buf: std.ArrayList(u8) = .empty,
    tensors: u32 = 0,
    kvs: u32 = 0,

    fn init(a: Allocator) Builder {
        return .{ .a = a };
    }

    fn deinit(self: *Builder) void {
        self.buf.deinit(self.a);
    }

    fn int(self: *Builder, comptime T: type, v: T) !void {
        var tmp: [@sizeOf(T)]u8 = undefined;
        std.mem.writeInt(T, &tmp, v, .little);
        try self.buf.appendSlice(self.a, &tmp);
    }

    fn str(self: *Builder, s: []const u8) !void {
        try self.int(u64, s.len);
        try self.buf.appendSlice(self.a, s);
    }

    fn header(self: *Builder, version: u32, tensor_count: u32, kv_count: u32) !void {
        try self.int(u32, magic_number);
        try self.int(u32, version);
        try self.int(u64, tensor_count);
        try self.int(u64, kv_count);
        self.tensors = tensor_count;
        self.kvs = kv_count;
    }

    fn kvString(self: *Builder, key: []const u8, v: []const u8) !void {
        try self.str(key);
        try self.int(u32, 8);
        try self.str(v);
    }

    fn kvU32(self: *Builder, key: []const u8, v: u32) !void {
        try self.str(key);
        try self.int(u32, 4);
        try self.int(u32, v);
    }

    fn kvI32(self: *Builder, key: []const u8, v: i32) !void {
        try self.str(key);
        try self.int(u32, 5);
        try self.int(i32, v);
    }

    fn kvF32(self: *Builder, key: []const u8, v: f32) !void {
        try self.str(key);
        try self.int(u32, 6);
        try self.int(u32, @bitCast(v));
    }

    fn kvBool(self: *Builder, key: []const u8, v: bool) !void {
        try self.str(key);
        try self.int(u32, 7);
        try self.int(u8, @intFromBool(v));
    }

    fn kvStringArray(self: *Builder, key: []const u8, items: []const []const u8) !void {
        try self.str(key);
        try self.int(u32, 9);
        try self.int(u32, 8);
        try self.int(u64, items.len);
        for (items) |s| try self.str(s);
    }

    fn tensorInfo(self: *Builder, name: []const u8, dims: []const u64, ttype: u32, offset: u64) !void {
        try self.str(name);
        try self.int(u32, @intCast(dims.len));
        for (dims) |d| try self.int(u64, d);
        try self.int(u32, ttype);
        try self.int(u64, offset);
    }

    fn padTo(self: *Builder, alignment: u64) !void {
        while (self.buf.items.len % alignment != 0) try self.buf.append(self.a, 0);
    }
};

fn f16Bits(v: f16) u16 {
    return @bitCast(v);
}

/// Builds a complete tiny GGUF v3 image:
///   w.f32 [4] f32, w.f16 [2,3] f16, w.q4_0 [32] q4_0
/// plus metadata of every shape the reader cares about.
fn buildTinyGguf(a: Allocator, alignment: u32) ![]u8 {
    var b = Builder.init(a);
    errdefer b.deinit();

    try b.header(3, 3, 8);
    try b.kvString("general.architecture", "llama");
    try b.kvString("general.name", "in-memory-fixture");
    try b.kvU32("general.alignment", alignment);
    try b.kvU32("llama.block_count", 1);
    try b.kvI32("fixture.negative", -7);
    try b.kvF32("llama.rope.freq_base", 10000.0);
    try b.kvBool("fixture.bool", true);
    try b.kvStringArray("fixture.strings", &.{ "a", "bb" });

    const f32_data = [_]u8{ 0x00, 0x00, 0xC0, 0x3F, 0x00, 0x00, 0x00, 0xC0, 0x00, 0x00, 0x50, 0x40, 0x00, 0x00, 0x00, 0x3F };
    const f16_data = [_]u8{ 0x00, 0x3C, 0x00, 0x41, 0x00, 0xBA, 0x00, 0x44, 0x00, 0xBE, 0x00, 0x34 };
    const q4_0_data = [_]u8{ 0x00, 0x38 } ++ [_]u8{0x21} ++ zeros(15);

    const off0 = 0;
    const off1 = alignForward(f32_data.len, alignment);
    const off2 = alignForward(off1 + f16_data.len, alignment);

    try b.tensorInfo("w.f32", &.{4}, 0, off0);
    try b.tensorInfo("w.f16", &.{ 2, 3 }, 1, off1);
    try b.tensorInfo("w.q4_0", &.{32}, 2, off2);

    try b.padTo(alignment);
    try b.buf.appendSlice(a, &f32_data);
    while (b.buf.items.len % alignment != 0) try b.buf.append(a, 0);
    try b.buf.appendSlice(a, &f16_data);
    while (b.buf.items.len % alignment != 0) try b.buf.append(a, 0);
    try b.buf.appendSlice(a, &q4_0_data);

    return b.buf.toOwnedSlice(a);
}

test "ggml type table matches the published ggml block layouts" {
    try testing.expectEqual(@as(usize, 32), GgmlType.q4_0.blockElems().?);
    try testing.expectEqual(@as(usize, 18), GgmlType.q4_0.blockBytes().?);
    try testing.expectEqual(@as(usize, 20), GgmlType.q4_1.blockBytes().?);
    try testing.expectEqual(@as(usize, 22), GgmlType.q5_0.blockBytes().?);
    try testing.expectEqual(@as(usize, 24), GgmlType.q5_1.blockBytes().?);
    try testing.expectEqual(@as(usize, 34), GgmlType.q8_0.blockBytes().?);
    try testing.expectEqual(@as(usize, 36), GgmlType.q8_1.blockBytes().?);
    try testing.expectEqual(@as(usize, 256), GgmlType.q4_k.blockElems().?);
    try testing.expectEqual(@as(usize, 84), GgmlType.q2_k.blockBytes().?);
    try testing.expectEqual(@as(usize, 110), GgmlType.q3_k.blockBytes().?);
    try testing.expectEqual(@as(usize, 144), GgmlType.q4_k.blockBytes().?);
    try testing.expectEqual(@as(usize, 176), GgmlType.q5_k.blockBytes().?);
    try testing.expectEqual(@as(usize, 210), GgmlType.q6_k.blockBytes().?);
    try testing.expectEqual(@as(usize, 292), GgmlType.q8_k.blockBytes().?);
    try testing.expectEqual(@as(u32, 30), @backingInt(GgmlType.bf16));
    // Unknown ids land on the non-exhaustive `_` tag and are unsupported.
    const unknown: GgmlType = @fromBackingInt(@intCast(16)); // iq2_xxs
    try testing.expect(unknown.blockBytes() == null);
    try testing.expect(!unknown.isSupported());
    try testing.expectError(error.UnsupportedType, dequantizeBytes(unknown, &.{}, &.{}));
}

test "dequantize: hand-computed legacy blocks (published ggml formulas)" {
    var out: [32]f32 = undefined;

    // q4_0: d = 0.5, qs[0] = 0x21  =>  x[0] = (1-8)*d = -3.5, x[16] = (2-8)*d = -3.0
    //                                 x[1] = x[17] = (0-8)*d = -4.0
    const q4_0 = [_]u8{ 0x00, 0x38 } ++ [_]u8{0x21} ++ zeros(15);
    try dequantizeBytes(.q4_0, &q4_0, &out);
    try testing.expectEqual(@as(f32, -3.5), out[0]);
    try testing.expectEqual(@as(f32, -3.0), out[16]);
    try testing.expectEqual(@as(f32, -4.0), out[1]);
    try testing.expectEqual(@as(f32, -4.0), out[17]);
    try testing.expectEqual(@as(f32, -4.0), out[31]);

    // q4_1: d = 0.5, m = 0.25, qs[0] = 0x21  =>  x[0] = 1*d+m = 0.75, x[16] = 2*d+m = 1.25
    const q4_1 = [_]u8{ 0x00, 0x38, 0x00, 0x34 } ++ [_]u8{0x21} ++ zeros(15);
    try dequantizeBytes(.q4_1, &q4_1, &out);
    try testing.expectEqual(@as(f32, 0.75), out[0]);
    try testing.expectEqual(@as(f32, 1.25), out[16]);
    try testing.expectEqual(@as(f32, 0.25), out[1]);

    // q5_0: d = 0.25, qh = 0x00010001 (byte order 01 00 01 00), qs[0] = 0x21
    //   qh bit 0 -> x[0]   = ((1 | 0x10) - 16)*d =  1*d =  0.25
    //   qh bit 16 -> x[16] = ((2 | 0x10) - 16)*d =  2*d =  0.50
    //   qh bits 1/17 clear -> x[1] = x[17] = (0 - 16)*d = -4.0
    const q5_0 = [_]u8{ 0x00, 0x34, 0x01, 0x00, 0x01, 0x00 } ++ [_]u8{0x21} ++ zeros(15);
    try dequantizeBytes(.q5_0, &q5_0, &out);
    try testing.expectEqual(@as(f32, 0.25), out[0]);
    try testing.expectEqual(@as(f32, 0.5), out[16]);
    try testing.expectEqual(@as(f32, -4.0), out[1]);
    try testing.expectEqual(@as(f32, -4.0), out[17]);

    // q5_1: d = 0.5, m = 1.0, same high bits => x[0] = (1|16)*d+m = 9.5,
    //       x[16] = (2|16)*d+m = 10.0, x[1] = (0)*d+m = 1.0
    const q5_1 = [_]u8{ 0x00, 0x38, 0x00, 0x3C, 0x01, 0x00, 0x01, 0x00 } ++ [_]u8{0x21} ++ zeros(15);
    try dequantizeBytes(.q5_1, &q5_1, &out);
    try testing.expectEqual(@as(f32, 9.5), out[0]);
    try testing.expectEqual(@as(f32, 10.0), out[16]);
    try testing.expectEqual(@as(f32, 1.0), out[1]);

    // q8_0: d = 0.25, qs = {-128, 127, -1, 0...} => -32.0, 31.75, -0.25, 0.0
    const q8_0 = [_]u8{ 0x00, 0x34, 0x80, 0x7F, 0xFF } ++ zeros(29);
    try dequantizeBytes(.q8_0, &q8_0, &out);
    try testing.expectEqual(@as(f32, -32.0), out[0]);
    try testing.expectEqual(@as(f32, 31.75), out[1]);
    try testing.expectEqual(@as(f32, -0.25), out[2]);
    try testing.expectEqual(@as(f32, 0.0), out[3]);

    // q8_1: same but d at offset 0, s at 2, quants at 4.
    const q8_1 = [_]u8{ 0x00, 0x34, 0x00, 0x00, 0x80, 0x7F, 0xFF } ++ zeros(29);
    try dequantizeBytes(.q8_1, &q8_1, &out);
    try testing.expectEqual(@as(f32, -32.0), out[0]);
    try testing.expectEqual(@as(f32, 31.75), out[1]);
    try testing.expectEqual(@as(f32, -0.25), out[2]);

    // Raw formats.
    var raw_out: [3]f32 = undefined;
    try dequantizeBytes(.f16, &[_]u8{ 0x00, 0x3C, 0x00, 0xC0, 0x00, 0x38 }, &raw_out);
    try testing.expectEqualSlices(f32, &.{ 1.0, -2.0, 0.5 }, &raw_out);

    try dequantizeBytes(.bf16, &[_]u8{ 0x80, 0x3F, 0x00, 0xC0, 0x00, 0x80 }, &raw_out);
    try testing.expectEqualSlices(f32, &.{ 1.0, -2.0, -0.0 }, &raw_out);

    try dequantizeBytes(.f32, &[_]u8{ 0x00, 0x00, 0x80, 0x3F, 0x00, 0x00, 0x00, 0xC0, 0x00, 0x00, 0x20, 0x40 }, &raw_out);
    try testing.expectEqualSlices(f32, &.{ 1.0, -2.0, 2.5 }, &raw_out);
}

test "dequantize: hand-computed q4_K block" {
    // Block: d = 1.0, dmin = 0.5; scales[0..] decode via get_scale_min_k4 to
    //   j=0: d=2, m=1 (q[0]=2, q[4]=1)      j=1: d=3, m=0 (q[1]=3)
    //   j=2: d=4, m=0 (q[2]=4)              j=3: d=5, m=0 (q[3]=5)
    //   j=4..7: 6-bit path, low nibbles of q[8..11] = d {6,7,8,9}, mins 0
    // qs[0] = 0x21 => low nibble 1 (group 0 low, d1=2, m1=0.5) and
    //                 high nibble 2 (group 0 high, d2=3, m2=0)
    // qs[32] = 0x0F => group 1 low nibble 15 with d1=4, m1=0
    var blk = zeros(144);
    blk[0] = 0x00;
    blk[1] = 0x3C; // d = 1.0
    blk[2] = 0x00;
    blk[3] = 0x38; // dmin = 0.5
    blk[4] = 2;
    blk[5] = 3;
    blk[6] = 4;
    blk[7] = 5;
    blk[8] = 1; // m for j=0
    blk[12] = 6; // scales[8] -> d for j=4
    blk[13] = 7;
    blk[14] = 8;
    blk[15] = 9;
    blk[16] = 0x21; // qs[0]
    blk[16 + 32] = 0x0F; // qs[32]

    var out: [256]f32 = undefined;
    try dequantizeBytes(.q4_k, &blk, &out);
    try testing.expectEqual(@as(f32, 1.5), out[0]); // 2*1 - 0.5
    try testing.expectEqual(@as(f32, -0.5), out[1]); // 2*0 - 0.5
    try testing.expectEqual(@as(f32, -0.5), out[31]);
    try testing.expectEqual(@as(f32, 6.0), out[32]); // 3*2 - 0
    try testing.expectEqual(@as(f32, 0.0), out[33]);
    try testing.expectEqual(@as(f32, 60.0), out[64]); // 4*15 - 0
    try testing.expectEqual(@as(f32, 0.0), out[96]);
    try testing.expectEqual(@as(f32, 0.0), out[128]);
    try testing.expectEqual(@as(f32, 0.0), out[255]);
}

test "dequantize: bit-exact against the ggml reference fixtures" {
    const a = testing.allocator;
    const cases = [_]struct { t: GgmlType, name: []const u8 }{
        .{ .t = .f32, .name = "f32" },
        .{ .t = .f16, .name = "f16" },
        .{ .t = .bf16, .name = "bf16" },
        .{ .t = .q4_0, .name = "q4_0" },
        .{ .t = .q4_1, .name = "q4_1" },
        .{ .t = .q5_0, .name = "q5_0" },
        .{ .t = .q5_1, .name = "q5_1" },
        .{ .t = .q8_0, .name = "q8_0" },
        .{ .t = .q8_1, .name = "q8_1" },
        .{ .t = .q2_k, .name = "q2_k" },
        .{ .t = .q3_k, .name = "q3_k" },
        .{ .t = .q4_k, .name = "q4_k" },
        .{ .t = .q5_k, .name = "q5_k" },
        .{ .t = .q6_k, .name = "q6_k" },
        .{ .t = .q8_k, .name = "q8_k" },
    };
    for (cases) |c| {
        const bin_name = try std.fmt.allocPrint(a, "dequant/{s}.bin", .{c.name});
        defer a.free(bin_name);
        const f32_name = try std.fmt.allocPrint(a, "dequant/{s}.f32", .{c.name});
        defer a.free(f32_name);

        const bin = try readFixture(a, bin_name);
        defer a.free(bin);
        const expected_bytes = try readFixture(a, f32_name);
        defer a.free(expected_bytes);
        try testing.expectEqual(@as(usize, 0), expected_bytes.len % 4);

        const expected = try a.alloc(f32, expected_bytes.len / 4);
        defer a.free(expected);
        @memcpy(std.mem.sliceAsBytes(expected), expected_bytes);
        const out = try a.alloc(f32, expected.len);
        defer a.free(out);
        try dequantizeBytes(c.t, bin, out);
        expectBitEqual(expected, out) catch |err| {
            std.debug.print("type {s} ({s}) differs from the ggml reference\n", .{ c.name, @tagName(c.t) });
            return err;
        };
    }
}

test "gguf: builds, parses and loads a tiny v3 file" {
    const a = testing.allocator;
    const bytes = try buildTinyGguf(a, 32);
    defer a.free(bytes);

    // In-memory parse.
    var g = try Gguf.fromBytes(a, bytes);
    defer g.deinit();
    try testing.expectEqual(@as(usize, 3), g.tensorCount());
    try testing.expectEqualStrings("llama", g.arch().?);
    try testing.expectEqualStrings("in-memory-fixture", g.getString("general.name").?);
    try testing.expectEqual(@as(u32, 1), g.getU32("llama.block_count").?);
    try testing.expectEqual(@as(i32, -7), g.getI32("fixture.negative").?);
    try testing.expectEqual(@as(f32, 10000.0), g.getF32("llama.rope.freq_base").?);
    try testing.expectEqual(true, g.getBool("fixture.bool").?);
    try testing.expectEqualStrings("bb", g.getStringArray("fixture.strings").?[1]);
    try testing.expect(g.tensor("does.not.exist") == null);
    try testing.expectEqual(@as(u64, 6), g.tensor("w.f16").?.elemCount());
    try testing.expectEqualSlices(u64, &.{ 2, 3 }, g.tensor("w.f16").?.dims);

    const wf32 = try g.readF32(a, "w.f32");
    defer a.free(wf32);
    try testing.expectEqualSlices(f32, &.{ 1.5, -2.0, 3.25, 0.5 }, wf32);

    const wf16 = try g.readF32(a, "w.f16");
    defer a.free(wf16);
    try testing.expectEqualSlices(f32, &.{ 1.0, 2.5, -0.75, 4.0, -1.5, 0.25 }, wf16);

    const wf16_half = try g.readF16(a, "w.f16");
    defer a.free(wf16_half);
    try testing.expectEqual(@as(f16, 1.0), wf16_half[0]);
    try testing.expectEqual(@as(f16, -1.5), wf16_half[4]);

    const wq4 = try g.readF32(a, "w.q4_0");
    defer a.free(wq4);
    try testing.expectEqual(@as(f32, -3.5), wq4[0]);
    try testing.expectEqual(@as(f32, -3.0), wq4[16]);

    // Same bytes through the file path (mmap or read fallback).
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(testing.io, .{ .sub_path = "tiny.gguf", .data = bytes });
    const path = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}/tiny.gguf", .{tmp.sub_path[0..]});
    defer a.free(path);

    var g2 = try Gguf.loadWithIo(a, testing.io, path);
    defer g2.deinit();
    try testing.expect(g2.mapping != null);
    try testing.expectEqual(@as(usize, 3), g2.tensorCount());
    const wf32_2 = try g2.readF32(a, "w.f32");
    defer a.free(wf32_2);
    try testing.expectEqualSlices(f32, wf32, wf32_2);

    // The default entry point uses the global std.Io implementation.
    var g3 = try Gguf.load(a, path);
    defer g3.deinit();
    try testing.expectEqualStrings("llama", g3.arch().?);
}

test "gguf: honors general.alignment" {
    const a = testing.allocator;
    const bytes = try buildTinyGguf(a, 64);
    defer a.free(bytes);
    var g = try Gguf.fromBytes(a, bytes);
    defer g.deinit();

    try testing.expectEqual(@as(u64, 64), g.alignment);
    try testing.expectEqual(@as(u64, 0), g.tensor_data_offset % 64);
    const wf32 = try g.readF32(a, "w.f32");
    defer a.free(wf32);
    try testing.expectEqualSlices(f32, &.{ 1.5, -2.0, 3.25, 0.5 }, wf32);
    const wq4 = try g.readF32(a, "w.q4_0");
    defer a.free(wq4);
    try testing.expectEqual(@as(f32, -3.5), wq4[0]);
}

test "gguf: reads the independently generated tiny_v3.gguf fixture" {
    const a = testing.allocator;
    const path = try fixturePath(a, "gguf/tiny_v3.gguf");
    defer a.free(path);
    var g = try Gguf.loadWithIo(a, testing.io, path);
    defer g.deinit();

    try testing.expectEqual(@as(u32, 3), g.version);
    try testing.expectEqual(@as(usize, 4), g.tensorCount());
    try testing.expectEqualStrings("llama", g.arch().?);
    try testing.expectEqualStrings("tiny-fixture", g.getString("general.name").?);
    try testing.expectEqual(@as(u64, 1234567890123), g.getU64("fixture.u64").?);
    try testing.expectEqual(@as(u64, 0xFFFF_FFFF_FFFF_FFFF - 1234567890122), @as(u64, @bitCast(@as(i64, g.getValue("fixture.i64").?.i64))));
    try testing.expectEqual(@as(f32, 10000.0), g.getF32("llama.rope.freq_base").?);
    try testing.expectEqual(true, g.getBool("fixture.bool").?);
    try testing.expectEqual(@as(u32, 200), g.getU32("fixture.u8").?);
    try testing.expectEqual(@as(u32, 60000), g.getU32("fixture.u16").?);

    const arr = g.getValue("fixture.u32_array").?;
    try testing.expectEqual(@as(usize, 3), arr.array.data.len());
    try testing.expectEqualSlices(u32, &.{ 1, 2, 3 }, arr.array.data.u32);
    try testing.expectEqual(@as(usize, 0), g.getValue("fixture.empty_array").?.array.data.len());

    const strings = g.getStringArray("fixture.string_array").?;
    try testing.expectEqual(@as(usize, 3), strings.len);
    try testing.expectEqualStrings("ccc", strings[2]);

    // Tensor table.
    const t_f32 = g.tensor("w.f32").?;
    try testing.expectEqual(GgmlType.f32, t_f32.ttype);
    try testing.expectEqualSlices(u64, &.{4}, t_f32.dims);
    try testing.expectEqual(@as(u64, 4), t_f32.elemCount());
    try testing.expectEqual(@as(u64, 16), t_f32.byteSize().?);

    const t_f16 = g.tensor("w.f16").?;
    try testing.expectEqualSlices(u64, &.{ 2, 3 }, t_f16.dims);
    try testing.expectEqual(@as(u64, 12), t_f16.byteSize().?);

    const wf32 = try g.readF32(a, "w.f32");
    defer a.free(wf32);
    try testing.expectEqualSlices(f32, &.{ 1.5, -2.0, 3.25, 0.5 }, wf32);

    const wf16 = try g.readF32(a, "w.f16");
    defer a.free(wf16);
    try testing.expectEqualSlices(f32, &.{ 1.0, 2.5, -0.75, 4.0, -1.5, 0.25 }, wf16);

    const wf16_half = try g.readF16(a, "w.f16");
    defer a.free(wf16_half);
    try testing.expectEqualSlices(f16, &.{ 1.0, 2.5, -0.75, 4.0, -1.5, 0.25 }, wf16_half);

    const wq4 = try g.readF32(a, "w.q4_0");
    defer a.free(wq4);
    try testing.expectEqual(@as(f32, -3.5), wq4[0]);
    try testing.expectEqual(@as(f32, -3.0), wq4[16]);
    try testing.expectEqual(@as(f32, -4.0), wq4[31]);

    const wq8 = try g.readF32(a, "w.q8_0");
    defer a.free(wq8);
    try testing.expectEqual(@as(f32, -32.0), wq8[0]);
    try testing.expectEqual(@as(f32, -31.75), wq8[1]);
    try testing.expectEqual(@as(f32, -24.25), wq8[31]);

    // f16 conversion path for a non-f16 tensor (lossy but exact here).
    const wq4_half = try g.readF16(a, "w.q4_0");
    defer a.free(wq4_half);
    try testing.expectEqual(@as(f16, -3.5), wq4_half[0]);
    try testing.expectEqual(@as(f16, -3.0), wq4_half[16]);
}

test "gguf: rejects malformed files" {
    const a = testing.allocator;

    // Bad magic.
    var bad = try buildTinyGguf(a, 32);
    defer a.free(bad);
    bad[0] = 'X';
    try testing.expectError(error.NotGgufFile, Gguf.fromBytes(a, bad));

    // Unsupported version.
    var bad_version = try buildTinyGguf(a, 32);
    defer a.free(bad_version);
    bad_version[4] = 4;
    try testing.expectError(error.UnsupportedVersion, Gguf.fromBytes(a, bad_version));

    // Truncated tensor table.
    var short = try buildTinyGguf(a, 32);
    defer a.free(short);
    try testing.expectError(error.TruncatedFile, Gguf.fromBytes(a, short[0 .. short.len - 400]));

    // Empty input.
    try testing.expectError(error.TruncatedFile, Gguf.fromBytes(a, &.{}));

    // Unsupported tensor type: same file, but w.f32 declared as iq2_xxs (16).
    // Rebuild with a patched type id: find the tensor-info table is fiddly, so
    // build a dedicated one-tensor file instead.
    var b = Builder.init(a);
    defer b.deinit();
    try b.header(3, 1, 0);
    try b.tensorInfo("bad", &.{32}, 16, 0);
    try b.padTo(32);
    try b.buf.appendSlice(a, &(zeros(200)));
    var g = try Gguf.fromBytes(a, b.buf.items);
    defer g.deinit();
    try testing.expectError(error.UnsupportedType, g.tensorBytes(g.tensor("bad").?));
    const out = try a.alloc(f32, 32);
    defer a.free(out);
    try testing.expectError(error.UnsupportedType, g.dequantizeTensor(g.tensor("bad").?, out));

    // Shape/type mismatch and out-of-range offsets.
    var b2 = Builder.init(a);
    defer b2.deinit();
    try b2.header(3, 2, 0);
    try b2.tensorInfo("odd", &.{48}, 2, 0); // 48 is not a multiple of 32
    try b2.tensorInfo("past_end", &.{32}, 2, 1 << 20);
    try b2.padTo(32);
    try b2.buf.appendSlice(a, &(zeros(64)));
    var g2 = try Gguf.fromBytes(a, b2.buf.items);
    defer g2.deinit();
    try testing.expectError(error.InvalidTensorShape, g2.tensorBytes(g2.tensor("odd").?));
    try testing.expectError(error.TensorDataOutOfRange, g2.tensorBytes(g2.tensor("past_end").?));
    try testing.expectError(error.TensorNotFound, g2.readF32(a, "nope"));
}

test "gguf: dequantizeRange converts in block-aligned chunks" {
    const a = testing.allocator;
    const bin = try readFixture(a, "dequant/q6_k.bin");
    defer a.free(bin);
    const expected_bytes = try readFixture(a, "dequant/q6_k.f32");
    defer a.free(expected_bytes);
    const expected = try a.alloc(f32, expected_bytes.len / 4);
    defer a.free(expected);
    @memcpy(std.mem.sliceAsBytes(expected), expected_bytes);

    // Element-wise over 128-element chunks, i.e. half a K-block each.
    var out = try a.alloc(f32, expected.len);
    defer a.free(out);
    var done: usize = 0;
    while (done < out.len) : (done += 256) {
        try dequantizeRange(.q6_k, bin, done, out[done .. done + 256]);
    }
    try expectBitEqual(expected, out);

    // Element offsets must be block-aligned.
    try testing.expectError(error.InvalidTensorShape, dequantizeRange(.q6_k, bin, 64, out[0..256]));
}
