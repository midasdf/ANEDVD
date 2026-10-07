// mil.zig — generate ANE-native MIL programs.
//
// Only ops verified to compile on ANEC are emitted here:
//   * conv (1x1) is the workhorse: on this ANE `matmul` is rejected with
//     kIOReturnUnsupported (0x1d) during evaluation, so every matrix product is
//     expressed as a 1x1 convolution, exactly like CoreML's own lowering.
//   * add (elementwise, alpha defaults to 1.0 — inline scalars cost one of the
//     16 BLOBFILE slots, so they are avoided).
// `rsqrt`, `concat`, `reduce_mean` and `gelu` are NOT available; activations,
// normalisation and attention therefore run on the CPU in this engine.

const std = @import("std");
const Buf = @import("../buf.zig").Buf;

pub const Input = struct {
    name: []const u8,
    channels: u32,
};

pub const Op = union(enum) {
    /// y = conv1x1(x, W) with W taken from a BLOBFILE chunk.
    conv: struct {
        x: []const u8,
        w: []const u8,
        y: []const u8,
        cin: u32,
        cout: u32,
        blob_offset: u64,
        file: []const u8,
    },
    /// y = a + b (same shape).
    add: struct {
        a: []const u8,
        b: []const u8,
        y: []const u8,
        channels: u32,
    },
    /// y = sigmoid(x)
    sigmoid: struct {
        x: []const u8,
        y: []const u8,
        channels: u32,
    },
    /// y = a * b (same shape)
    mul: struct {
        a: []const u8,
        b: []const u8,
        y: []const u8,
        channels: u32,
    },
};

pub const Spec = struct {
    inputs: []const Input,
    ops: []const Op,
    /// Tensor names returned by the program, in the order they should be read.
    outputs: []const []const u8,
};

const BUILD_INFO =
    "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, " ++
    "{\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, " ++
    "{\"coremltools-version\", \"9.0\"}})]";

const CONST_HEADER =
    \\        string c_pad_type = const()[name = string("c_pad_type"), val = string("valid")];
    \\        tensor<int32, [2]> c_strides = const()[name = string("c_strides"), val = tensor<int32, [2]>([1, 1])];
    \\        tensor<int32, [4]> c_pad = const()[name = string("c_pad"), val = tensor<int32, [4]>([0, 0, 0, 0])];
    \\        tensor<int32, [2]> c_dilations = const()[name = string("c_dilations"), val = tensor<int32, [2]>([1, 1])];
    \\        int32 c_groups = const()[name = string("c_groups"), val = int32(1)];
    \\
;

/// Build MIL text for `spec`. Caller owns the returned slice.
pub fn build(allocator: std.mem.Allocator, spec: Spec) ![]u8 {
    var b = Buf.init(allocator);
    defer b.deinit();

    try b.appendSlice("program(1.3)\n");
    try b.appendSlice(BUILD_INFO);
    try b.appendSlice("\n{\n    func main<ios18>(");
    for (spec.inputs, 0..) |in, i| {
        if (i > 0) try b.appendSlice(", ");
        try b.print("tensor<fp16, [1, {d}, 1, 1]> {s}", .{ in.channels, in.name });
    }
    try b.appendSlice(") {\n");
    try b.appendSlice(CONST_HEADER);

    for (spec.ops) |op| {
        switch (op) {
            .conv => |c| {
                try b.print(
                    "        tensor<fp16, [{d}, {d}, 1, 1]> {s} = const()[name = string(\"{s}\"), val = tensor<fp16, [{d}, {d}, 1, 1]>(BLOBFILE(path = string(\"{s}\"), offset = uint64({d})))];\n",
                    .{ c.cout, c.cin, c.w, c.w, c.cout, c.cin, c.file, c.blob_offset },
                );
                try b.print(
                    "        tensor<fp16, [1, {d}, 1, 1]> {s} = conv(dilations = c_dilations, groups = c_groups, pad = c_pad, pad_type = c_pad_type, strides = c_strides, weight = {s}, x = {s})[name = string(\"{s}\")];\n",
                    .{ c.cout, c.y, c.w, c.x, c.y },
                );
            },
            .add => |a| {
                try b.print(
                    "        tensor<fp16, [1, {d}, 1, 1]> {s} = add(x = {s}, y = {s})[name = string(\"{s}\")];\n",
                    .{ a.channels, a.y, a.a, a.b, a.y },
                );
            },
            .sigmoid => |s| {
                try b.print(
                    "        tensor<fp16, [1, {d}, 1, 1]> {s} = sigmoid(x = {s})[name = string(\"{s}\")];\n",
                    .{ s.channels, s.y, s.x, s.y },
                );
            },
            .mul => |m| {
                try b.print(
                    "        tensor<fp16, [1, {d}, 1, 1]> {s} = mul(x = {s}, y = {s})[name = string(\"{s}\")];\n",
                    .{ m.channels, m.y, m.a, m.b, m.y },
                );
            },
        }
    }

    try b.appendSlice("    } -> (");
    for (spec.outputs, 0..) |o, i| {
        if (i > 0) try b.appendSlice(", ");
        try b.appendSlice(o);
    }
    try b.appendSlice(");\n}\n");
    return b.toOwnedSlice();
}

test "single conv program is well-formed" {
    const allocator = std.testing.allocator;
    const mil = try build(allocator, .{
        .inputs = &.{.{ .name = "i0", .channels = 8 }},
        .ops = &.{.{ .conv = .{
            .x = "i0",
            .w = "w0",
            .y = "o0",
            .cin = 8,
            .cout = 4,
            .blob_offset = 64,
            .file = "@model_path/weights/w0.bin",
        } }},
        .outputs = &.{"o0"},
    });
    defer allocator.free(mil);
    try std.testing.expect(std.mem.indexOf(u8, mil, "program(1.3)") != null);
    try std.testing.expect(std.mem.indexOf(u8, mil, "tensor<fp16, [1, 8, 1, 1]> i0") != null);
    try std.testing.expect(std.mem.indexOf(u8, mil, "tensor<fp16, [4, 8, 1, 1]> w0") != null);
    try std.testing.expect(std.mem.indexOf(u8, mil, "offset = uint64(64)") != null);
    try std.testing.expect(std.mem.indexOf(u8, mil, "} -> (o0);") != null);
    // Every `{` must be balanced; a malformed program fails deep inside ANECCompile.
    var depth: i32 = 0;
    for (mil) |ch| {
        if (ch == '{') depth += 1;
        if (ch == '}') depth -= 1;
    }
    try std.testing.expectEqual(@as(i32, 0), depth);
}

test "multi-op program orders outputs" {
    const allocator = std.testing.allocator;
    const mil = try build(allocator, .{
        .inputs = &.{ .{ .name = "i0", .channels = 4 }, .{ .name = "i1", .channels = 4 } },
        .ops = &.{
            .{ .conv = .{ .x = "i0", .w = "w0", .y = "t0", .cin = 4, .cout = 4, .blob_offset = 64, .file = "@model_path/weights/w0.bin" } },
            .{ .conv = .{ .x = "i1", .w = "w1", .y = "t1", .cin = 4, .cout = 4, .blob_offset = 128, .file = "@model_path/weights/w0.bin" } },
            .{ .add = .{ .a = "t0", .b = "t1", .y = "o0", .channels = 4 } },
        },
        .outputs = &.{"o0"},
    });
    defer allocator.free(mil);
    try std.testing.expect(std.mem.indexOf(u8, mil, "tensor<fp16, [1, 4, 1, 1]> i0, tensor<fp16, [1, 4, 1, 1]> i1") != null);
    try std.testing.expect(std.mem.indexOf(u8, mil, "add(x = t0, y = t1)") != null);
}
