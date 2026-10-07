const std = @import("std");
const builtin = @import("builtin");
const Io = std.Io;

fn defaultIo() Io {
    return Io.Threaded.global_single_threaded.io();
}

test "json + io probe" {
    const alloc = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const io = std.testing.io;
    try tmp.dir.writeFile(io, .{ .sub_path = "a.json", .data = "{\"x\": 3, \"arr\": [1,2], \"s\": \"hi\", \"b\": true, \"n\": null, \"f\": 0.5}" });
    const bytes = try tmp.dir.readFileAlloc(io, "a.json", alloc, .unlimited);
    defer alloc.free(bytes);

    var arena = std.heap.ArenaAllocator.init(alloc);
    defer arena.deinit();
    const val = try std.json.parseFromSliceLeaky(std.json.Value, arena.allocator(), bytes, .{});
    try std.testing.expectEqual(@as(i64, 3), val.object.get("x").?.integer);
    try std.testing.expectEqual(@as(usize, 2), val.object.get("arr").?.array.items.len);
    try std.testing.expectEqualStrings("hi", val.object.get("s").?.string);
    try std.testing.expectEqual(true, val.object.get("b").?.bool);
    try std.testing.expect(val.object.get("n").? == .null);

    var it = val.object.iterator();
    var count: usize = 0;
    while (it.next()) |_| count += 1;
    try std.testing.expectEqual(@as(usize, 6), count);

    // directory iteration through a different Io instance
    try tmp.dir.writeFile(io, .{ .sub_path = "b.safetensors", .data = "xx" });
    var d = try tmp.dir.openDir(io, ".", .{ .iterate = true });
    defer d.close(io);
    var diter = d.iterate();
    var names: usize = 0;
    while (try diter.next(io)) |e| {
        if (e.kind == .file) names += 1;
    }
    try std.testing.expectEqual(@as(usize, 2), names);

    // read the same file with global_single_threaded Io
    const gio = defaultIo();
    const cwd = Io.Dir.cwd();
    const full = try std.fs.path.join(alloc, &.{ ".zig-cache", "tmp", &tmp.sub_path });
    defer alloc.free(full);
    const file_path = try std.fs.path.join(alloc, &.{ full, "a.json" });
    defer alloc.free(file_path);
    const bytes2 = try cwd.readFileAlloc(gio, file_path, alloc, .unlimited);
    defer alloc.free(bytes2);
    try std.testing.expectEqualStrings(bytes, bytes2);
    const st = try cwd.statFile(gio, file_path, .{});
    try std.testing.expect(st.size == bytes.len);

    // hash map of string -> struct, insertion order preserved
    var map: std.array_hash_map.String(u32) = .empty;
    defer map.deinit(alloc);
    try map.put(alloc, "one", 1);
    try map.put(alloc, "two", 2);
    try std.testing.expectEqual(@as(usize, 2), map.count());
    try std.testing.expectEqual(@as(u32, 2), map.get("two").?);
    try std.testing.expectEqualStrings("one", map.keys()[0]);

    // ArrayList (unmanaged alias in 0.17)
    var list: std.ArrayList(u8) = .empty;
    defer list.deinit(alloc);
    try list.append(alloc, 7);
    try std.testing.expectEqual(@as(usize, 1), list.items.len);

    // readInt
    const buf = [_]u8{ 0x01, 0x02, 0x03, 0x04 };
    try std.testing.expectEqual(@as(u32, 0x04030201), std.mem.readInt(u32, &buf, .little));
}
