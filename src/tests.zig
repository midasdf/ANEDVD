// tests.zig — the test root.
//
// Zig only runs `test` blocks from the test root and the files it references,
// so a test build rooted at src/main.zig executed zero tests: every module's
// suite existed but was never analysed. Referencing each module here is what
// makes `zig build test` real.

test {
    _ = @import("buf.zig");
    _ = @import("sys.zig");
    _ = @import("cpu.zig");
    _ = @import("model.zig");
    _ = @import("engine.zig");
    _ = @import("generate.zig");
    _ = @import("gguf.zig");
    _ = @import("tokenizer.zig");
    _ = @import("safetensors.zig");
    _ = @import("hf.zig");
    _ = @import("load_gguf.zig");
    _ = @import("load_hf.zig");
    _ = @import("model_open.zig");
    _ = @import("http.zig");
    _ = @import("server.zig");
    _ = @import("ane/mil.zig");
    _ = @import("ane/runtime.zig");
    _ = @import("ane/weights.zig");
}
