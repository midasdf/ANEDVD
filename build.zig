const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    // Default to ReleaseFast: the engine spends a real amount of time in Zig
    // code (prompt batching, attention, dequantisation) where Debug is ~8x
    // slower. `zig build -Doptimize=Debug` is still available for development.
    const optimize = b.option(std.builtin.Optimize, "optimize", "Prioritize performance, safety, or binary size") orelse .fast;

    // The ANE shim is Objective-C with ARC. Zig's bundled clang crashes on it,
    // so compile it with the system toolchain and link the object file.
    const shim_cc = b.addSystemCommand(&.{ "/usr/bin/clang", "-fobjc-arc", "-O2", "-Wall", "-c" });
    shim_cc.addFileArg(b.path("src/ane/shim.m"));
    shim_cc.addArg("-o");
    const shim_obj = shim_cc.addOutputFileArg("shim.o");

    const mod = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    mod.addObjectFile(shim_obj);
    mod.addIncludePath(b.path("src/ane"));
    mod.linkFramework("Foundation", .{});
    mod.linkFramework("IOSurface", .{});

    const exe = b.addExecutable(.{
        .name = "anedvd",
        .root_module = mod,
    });
    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    run_cmd.addPassthruArgs();
    b.step("run", "Run anedvd").dependOn(&run_cmd.step);

    // Test build. Zig runs `test` blocks from the test ROOT and the files it
    // references, so rooting the test build at src/main.zig silently ran zero
    // tests; src/tests.zig references every module instead.
    const test_mod = b.createModule(.{
        .root_source_file = b.path("src/tests.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    test_mod.addObjectFile(shim_obj);
    test_mod.addIncludePath(b.path("src/ane"));
    test_mod.linkFramework("Foundation", .{});
    test_mod.linkFramework("IOSurface", .{});

    const unit_tests = b.addTest(.{ .root_module = test_mod });
    b.step("test", "Run unit tests").dependOn(&b.addRunArtifact(unit_tests).step);
}
