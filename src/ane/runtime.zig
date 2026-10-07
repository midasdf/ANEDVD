// runtime.zig — Zig binding for the ANE shim (src/ane/shim.m).
//
// Owns the lifecycle of compiled ANE kernels and knows how to move fp16 data
// in and out of the ANE's planar tensor layout.
//
// Layout facts (verified on A18 Pro / macOS 27.0.1):
//   * A tensor [1, C, H, W] is stored as C planes of `plane_stride` bytes.
//   * plane_stride = max(64, row_stride * H), row_stride = max(64, W * elem_size).
//   * For fp16 with W >= 32 the layout is effectively contiguous; for small W
//     (decode: W == 1) every channel still occupies its own 64-byte plane.
//   * The authoritative numbers come from the loaded model
//     (modelAttributes -> NetworkStatusList -> LiveInputList/LiveOutputList),
//     never from guesses.

const std = @import("std");

pub const Dtype = enum(c_int) { fp16 = 0, fp32 = 1, unknown = 2 };

pub const TensorInfo = extern struct {
    name: [64]u8,
    dtype: c_int,
    batches: c_int,
    channels: c_int,
    height: c_int,
    width: c_int,
    plane_stride: usize,
    row_stride: usize,
    depth_stride: usize,
    batch_stride: usize,
    nbytes: usize,

    pub fn nameSlice(self: *const TensorInfo) []const u8 {
        return std.mem.sliceTo(&self.name, 0);
    }
    pub fn channelsUsize(self: *const TensorInfo) usize {
        return @intCast(self.channels);
    }
    pub fn widthUsize(self: *const TensorInfo) usize {
        return @intCast(self.width);
    }
    pub fn heightUsize(self: *const TensorInfo) usize {
        return @intCast(self.height);
    }
    /// Number of logical elements (batches * channels * height * width).
    pub fn elemCount(self: *const TensorInfo) usize {
        return @as(usize, @intCast(self.batches)) * self.channelsUsize() *
            self.heightUsize() * self.widthUsize();
    }
    pub fn elemSize(self: *const TensorInfo) usize {
        return if (self.dtype == @backingInt(Dtype.fp32)) 4 else 2;
    }
    /// True when the ANE layout happens to be plain row-major (no padding).
    pub fn isContiguous(self: *const TensorInfo) bool {
        return self.row_stride == self.widthUsize() * self.elemSize() and
            self.plane_stride == self.row_stride * self.heightUsize() and
            self.batch_stride == self.plane_stride * self.channelsUsize();
    }
};

extern fn ane_shim_init() c_int;
extern fn ane_shim_ready() c_int;
extern fn ane_shim_last_error() [*:0]const u8;
extern fn ane_shim_compile_count() c_int;
extern fn ane_shim_kernel_create(
    mil: [*]const u8,
    mil_len: usize,
    w_names: [*]const [*:0]const u8,
    w_data: [*]const [*]const u8,
    w_lens: [*]const usize,
    n_weights: c_int,
) ?*anyopaque;
extern fn ane_shim_kernel_free(k: *anyopaque) void;
extern fn ane_shim_kernel_eval(k: *anyopaque) c_int;
extern fn ane_shim_input_count(k: *const anyopaque) c_int;
extern fn ane_shim_output_count(k: *const anyopaque) c_int;
extern fn ane_shim_input_info(k: *const anyopaque, idx: c_int, out: *TensorInfo) c_int;
extern fn ane_shim_output_info(k: *const anyopaque, idx: c_int, out: *TensorInfo) c_int;
extern fn ane_shim_input_base(k: *const anyopaque, idx: c_int) ?[*]u8;
extern fn ane_shim_output_base(k: *const anyopaque, idx: c_int) ?[*]u8;
extern fn ane_shim_input_lock(k: *const anyopaque, idx: c_int) c_int;
extern fn ane_shim_input_unlock(k: *const anyopaque, idx: c_int) c_int;
extern fn ane_shim_output_lock(k: *const anyopaque, idx: c_int) c_int;
extern fn ane_shim_output_unlock(k: *const anyopaque, idx: c_int) c_int;
extern fn ane_shim_input_capacity(k: *const anyopaque, idx: c_int) usize;
extern fn ane_shim_output_capacity(k: *const anyopaque, idx: c_int) usize;
extern fn ane_shim_last_eval_ns(k: *const anyopaque) u64;

/// One weight file handed to a kernel. `name` must be a MIL weight symbol such
/// as "@model_path/weights/w0.bin".
pub const WeightFile = struct {
    name: [:0]const u8,
    data: []const u8,
};

pub const Error = error{
    AneUnavailable,
    AneCompileFailed,
    AneEvalFailed,
    SurfaceUnavailable,
    SurfaceLockFailed,
    WrongShape,
    TooManyTensors,
};

pub fn lastError() []const u8 {
    return std.mem.span(ane_shim_last_error());
}

/// Whether the ANE private framework could be loaded and its classes resolved.
pub fn available() bool {
    return ane_shim_init() == 0;
}

pub fn compileCount() i32 {
    return ane_shim_compile_count();
}

fn dupeZ(allocator: std.mem.Allocator, s: []const u8) ![:0]u8 {
    const buf = try allocator.alloc(u8, s.len + 1);
    @memcpy(buf[0..s.len], s);
    buf[s.len] = 0;
    return buf[0..s.len :0];
}

pub const Kernel = struct {
    allocator: std.mem.Allocator,
    handle: *anyopaque,
    in_info: []TensorInfo,
    out_info: []TensorInfo,
    /// Names of the MIL weight symbols, kept alive for the kernel's lifetime.
    weight_names: [][:0]u8,
    /// Bytes written to the input surfaces at creation (they start zeroed and
    /// padding is never touched afterwards).
    input_scratch: []u8,

    pub fn create(
        allocator: std.mem.Allocator,
        mil: []const u8,
        weights: []const WeightFile,
    ) !Kernel {
        if (!available()) return Error.AneUnavailable;

        const names = try allocator.alloc([*:0]const u8, weights.len);
        defer allocator.free(names);
        const datas = try allocator.alloc([*]const u8, weights.len);
        defer allocator.free(datas);
        const lens = try allocator.alloc(usize, weights.len);
        defer allocator.free(lens);
        var owned_names = try allocator.alloc([:0]u8, weights.len);
        errdefer {
            for (owned_names) |n| allocator.free(n);
            allocator.free(owned_names);
        }
        for (weights, 0..) |w, i| {
            owned_names[i] = try dupeZ(allocator, w.name);
            names[i] = owned_names[i].ptr;
            datas[i] = w.data.ptr;
            lens[i] = w.data.len;
        }

        const h = ane_shim_kernel_create(mil.ptr, mil.len, names.ptr, datas.ptr, lens.ptr, @intCast(weights.len)) orelse {
            for (owned_names) |n| allocator.free(n);
            allocator.free(owned_names);
            return Error.AneCompileFailed;
        };

        const ni: usize = @intCast(ane_shim_input_count(h));
        const no: usize = @intCast(ane_shim_output_count(h));
        if (ni == 0 or no == 0 or ni > 16 or no > 16) {
            ane_shim_kernel_free(h);
            for (owned_names) |n| allocator.free(n);
            allocator.free(owned_names);
            return Error.TooManyTensors;
        }

        const in_info = try allocator.alloc(TensorInfo, ni);
        errdefer allocator.free(in_info);
        const out_info = try allocator.alloc(TensorInfo, no);
        errdefer allocator.free(out_info);
        for (0..ni) |i| {
            if (ane_shim_input_info(h, @intCast(i), &in_info[i]) == 0) return Error.SurfaceUnavailable;
        }
        for (0..no) |i| {
            if (ane_shim_output_info(h, @intCast(i), &out_info[i]) == 0) return Error.SurfaceUnavailable;
        }

        var max_in: usize = 0;
        for (in_info) |*t| max_in = @max(max_in, t.nbytes);

        return .{
            .allocator = allocator,
            .handle = h,
            .in_info = in_info,
            .out_info = out_info,
            .weight_names = owned_names,
            .input_scratch = try allocator.alloc(u8, max_in),
        };
    }

    pub fn deinit(self: *Kernel) void {
        ane_shim_kernel_free(self.handle);
        for (self.weight_names) |n| self.allocator.free(n);
        self.allocator.free(self.weight_names);
        self.allocator.free(self.in_info);
        self.allocator.free(self.out_info);
        self.allocator.free(self.input_scratch);
        self.* = undefined;
    }

    pub fn inputCount(self: *const Kernel) usize {
        return self.in_info.len;
    }
    pub fn outputCount(self: *const Kernel) usize {
        return self.out_info.len;
    }
    pub fn inputInfo(self: *const Kernel, idx: usize) TensorInfo {
        return self.in_info[idx];
    }
    pub fn outputInfo(self: *const Kernel, idx: usize) TensorInfo {
        return self.out_info[idx];
    }

    pub fn eval(self: *const Kernel) !void {
        if (ane_shim_kernel_eval(self.handle) == 0) return Error.AneEvalFailed;
    }

    pub fn lastEvalNs(self: *const Kernel) u64 {
        return ane_shim_last_eval_ns(self.handle);
    }

    /// Write fp16 values (row-major [channels][height][width]) into input `idx`,
    /// honouring the ANE's planar layout. Unused padding is left as-is.
    pub fn writeInputF16(self: *const Kernel, idx: usize, values: []const f16) !void {
        const info = self.in_info[idx];
        if (values.len != info.elemCount()) return Error.WrongShape;
        if (ane_shim_input_lock(self.handle, @intCast(idx)) == 0) return Error.SurfaceLockFailed;
        defer _ = ane_shim_input_unlock(self.handle, @intCast(idx));
        const base = ane_shim_input_base(self.handle, @intCast(idx)) orelse return Error.SurfaceUnavailable;
        scatterF16(base, info, values);
    }

    /// Read fp16 values from output `idx` back into row-major order.
    pub fn readOutputF16(self: *const Kernel, idx: usize, out: []f16) !void {
        const info = self.out_info[idx];
        if (out.len != info.elemCount()) return Error.WrongShape;
        if (ane_shim_output_lock(self.handle, @intCast(idx)) == 0) return Error.SurfaceLockFailed;
        defer _ = ane_shim_output_unlock(self.handle, @intCast(idx));
        const base = ane_shim_output_base(self.handle, @intCast(idx)) orelse return Error.SurfaceUnavailable;
        gatherF16(base, info, out);
    }

    /// Zero every input surface (padding included). Called once at startup so
    /// later partial writes cannot leak stale bytes into a kernel.
    pub fn zeroInputs(self: *const Kernel) void {
        for (self.in_info, 0..) |info, i| {
            const cap = ane_shim_input_capacity(self.handle, @intCast(i));
            if (cap == 0) continue;
            if (ane_shim_input_lock(self.handle, @intCast(i)) == 0) continue;
            const base = ane_shim_input_base(self.handle, @intCast(i)) orelse continue;
            @memset(base[0..cap], 0);
            _ = ane_shim_input_unlock(self.handle, @intCast(i));
            _ = info;
        }
    }
};

fn scatterF16(base: [*]u8, info: TensorInfo, values: []const f16) void {
    const w = info.widthUsize();
    const h = info.heightUsize();
    const c = info.channelsUsize();
    if (info.isContiguous()) {
        const bytes = std.mem.sliceAsBytes(values);
        @memcpy(base[0..bytes.len], bytes);
        return;
    }
    var i: usize = 0;
    for (0..c) |ch| {
        for (0..h) |y| {
            const src = values[i..][0..w];
            i += w;
            const dst = base + ch * info.plane_stride + y * info.row_stride;
            const bytes = std.mem.sliceAsBytes(src);
            @memcpy(dst[0..bytes.len], bytes);
        }
    }
}

fn gatherF16(base: [*]const u8, info: TensorInfo, out: []f16) void {
    const w = info.widthUsize();
    const h = info.heightUsize();
    const c = info.channelsUsize();
    if (info.isContiguous()) {
        const bytes = std.mem.sliceAsBytes(out);
        @memcpy(bytes, base[0..bytes.len]);
        return;
    }
    var i: usize = 0;
    for (0..c) |ch| {
        for (0..h) |y| {
            const dst = out[i..][0..w];
            i += w;
            const src = base + ch * info.plane_stride + y * info.row_stride;
            const bytes = std.mem.sliceAsBytes(dst);
            @memcpy(bytes, src[0..bytes.len]);
        }
    }
}

test "tensor info helpers" {
    var t: TensorInfo = std.mem.zeroes(TensorInfo);
    t.dtype = @backingInt(Dtype.fp16);
    t.batches = 1;
    t.channels = 4;
    t.height = 1;
    t.width = 1;
    t.row_stride = 64;
    t.plane_stride = 64;
    t.batch_stride = 256;
    try std.testing.expectEqual(@as(usize, 4), t.elemCount());
    try std.testing.expect(!t.isContiguous());
    t.width = 32;
    t.row_stride = 64;
    t.plane_stride = 64;
    t.batch_stride = 256;
    try std.testing.expect(t.isContiguous());
}
