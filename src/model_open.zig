// model_open.zig — open a model directory or GGUF file and hand back everything
// the engine needs, regardless of format.
//
//   *.gguf            → GGUF reader (llama-family RoPE permutation rules)
//   a directory       → HuggingFace safetensors + config.json + tokenizer.json
//
// The returned `Loaded` owns the file handle, the tokenizer and the runtime
// weights; the layer/head sources it exposes stay valid until `deinit`.

const std = @import("std");
const gguf = @import("gguf.zig");
const hf = @import("hf.zig");
const load_gguf = @import("load_gguf.zig");
const load_hf = @import("load_hf.zig");
const model = @import("model.zig");
const sys = @import("sys.zig");
const tokenizer_mod = @import("tokenizer.zig");

pub const Format = enum { gguf, hf };

pub const Options = struct {
    progress: bool = false,
    /// Force the RoPE convention (null = derive from the format/architecture).
    rope_hf: bool = false,
    rope_adjacent: bool = false,
};

pub const Loaded = struct {
    allocator: std.mem.Allocator,
    format: Format,
    config: model.Config,
    rt: model.Runtime,
    layers: model.LayerSource,
    head: model.HeadSource,
    tokenizer: tokenizer_mod.Tokenizer,

    gguf_ptr: ?*gguf.Gguf = null,
    shards_ptr: ?*load_hf.Shards = null,
    layers_ctx: ?*anyopaque = null,
    head_ctx: ?*anyopaque = null,

    /// Frees the file handle, tokenizer and layer/head sources. The runtime
    /// weights are *not* freed here: `Engine.init` takes ownership of them.
    pub fn deinit(self: *Loaded) void {
        self.tokenizer.deinit();
        switch (self.format) {
            .gguf => {
                if (self.layers_ctx) |p| self.allocator.destroy(@as(*load_gguf.GgufLayers, @ptrCast(@alignCast(p))));
                if (self.head_ctx) |p| self.allocator.destroy(@as(*load_gguf.GgufHead, @ptrCast(@alignCast(p))));
            },
            .hf => {
                if (self.layers_ctx) |p| self.allocator.destroy(@as(*load_hf.HfLayers, @ptrCast(@alignCast(p))));
                if (self.head_ctx) |p| self.allocator.destroy(@as(*load_hf.HfHead, @ptrCast(@alignCast(p))));
            },
        }
        if (self.gguf_ptr) |g| {
            g.deinit();
            self.allocator.destroy(g);
        }
        if (self.shards_ptr) |sh| {
            sh.deinit();
            self.allocator.destroy(sh);
        }
        self.* = undefined;
    }
};

/// A `.gguf` file, whatever the case of the extension.
///
///
/// Case-sensitive matching sent `Model.GGUF` down the HuggingFace-directory path,
/// where it failed with a bare `NotDir` — the kind of error that sends you looking
/// at the filesystem rather than at the extension.
pub fn isGguf(path: []const u8) bool {
    if (path.len < 5) return false;
    const ext = path[path.len - 5 ..];
    return std.ascii.eqlIgnoreCase(ext, ".gguf");
}

fn applyRopeOverrides(cfg: *model.Config, opts: Options) void {
    if (opts.rope_hf) cfg.rope_adjacent = false;
    if (opts.rope_adjacent) cfg.rope_adjacent = true;
}

pub fn open(allocator: std.mem.Allocator, path: []const u8, opts: Options) !Loaded {
    // Report a helpful reason rather than a bare error name: an unsupported
    // tokenizer is the most common way a real model fails to load, and the fix
    // ("this file is SentencePiece; use a GPT-2-vocabulary model") is not
    // guessable from `error.UnsupportedTokenizerModel`.
    return if (isGguf(path))
        openGguf(allocator, path, opts) catch |e| explain(path, e)
    else
        openHf(allocator, path, opts) catch |e| explain(path, e);
}

fn explain(path: []const u8, e: anyerror) anyerror {
    switch (e) {
        error.UnsupportedTokenizerModel, error.MissingTokens => {
            if (tokenizer_mod.Tokenizer.last_error_detail) |detail| {
                sys.eprint("cannot load {s}:\n  {s}\n", .{ path, detail });
            } else {
                sys.eprint("cannot load {s}: the tokenizer format is not supported.\n", .{path});
            }
        },
        error.UnsupportedArchitecture => {
            sys.eprint("cannot load {s}: unsupported architecture (see src/hf.zig for the list).\n", .{path});
        },
        else => {},
    }
    return e;
}

fn openGguf(allocator: std.mem.Allocator, path: []const u8, opts: Options) !Loaded {
    const g = try allocator.create(gguf.Gguf);
    errdefer allocator.destroy(g);
    g.* = try gguf.Gguf.load(allocator, path);

    var cfg = try load_gguf.loadConfig(g);
    applyRopeOverrides(&cfg, opts);

    var tok = try tokenizer_mod.Tokenizer.fromGguf(allocator, g);
    errdefer tok.deinit();

    var rt = try load_gguf.loadRuntime(allocator, g, cfg, opts.progress);
    errdefer rt.deinit();

    const lc = try allocator.create(load_gguf.GgufLayers);
    errdefer allocator.destroy(lc);
    lc.* = .{ .g = g, .cfg = rt.config };
    const hc = try allocator.create(load_gguf.GgufHead);
    errdefer allocator.destroy(hc);
    hc.* = .{ .g = g, .cfg = rt.config, .embed = rt.embed };

    return .{
        .allocator = allocator,
        .format = .gguf,
        .config = rt.config,
        .rt = rt,
        .layers = lc.source(),
        .head = hc.source(),
        .tokenizer = tok,
        .gguf_ptr = g,
        .layers_ctx = lc,
        .head_ctx = hc,
    };
}

fn openHf(allocator: std.mem.Allocator, dir: []const u8, opts: Options) !Loaded {
    const sh = try allocator.create(load_hf.Shards);
    errdefer allocator.destroy(sh);
    sh.* = try load_hf.Shards.open(allocator, dir);

    var hcfg = try hf.loadConfig(allocator, dir);
    defer hcfg.deinit(allocator);
    var cfg = try load_hf.toModelConfig(hcfg);
    applyRopeOverrides(&cfg, opts);

    const tok_path = try std.fs.path.join(allocator, &.{ dir, "tokenizer.json" });
    defer allocator.free(tok_path);
    var tok = try tokenizer_mod.Tokenizer.fromTokenizerJson(allocator, tok_path);
    errdefer tok.deinit();

    var rt = try load_hf.loadRuntime(allocator, sh, cfg, opts.progress);
    errdefer rt.deinit();

    const lc = try allocator.create(load_hf.HfLayers);
    errdefer allocator.destroy(lc);
    lc.* = .{ .shards = sh, .cfg = rt.config };
    const hc = try allocator.create(load_hf.HfHead);
    errdefer allocator.destroy(hc);
    hc.* = .{ .shards = sh, .cfg = rt.config, .embed = rt.embed };

    return .{
        .allocator = allocator,
        .format = .hf,
        .config = rt.config,
        .rt = rt,
        .layers = lc.source(),
        .head = hc.source(),
        .tokenizer = tok,
        .shards_ptr = sh,
        .layers_ctx = lc,
        .head_ctx = hc,
    };
}

/// Model id used by the API: the GGUF file stem, or the directory name.
pub fn modelName(path: []const u8) []const u8 {
    const base = std.fs.path.basename(path);
    if (isGguf(path)) {
        const ext = std.fs.path.extension(base);
        return base[0 .. base.len - ext.len];
    }
    return if (base.len == 0) path else base;
}

test "isGguf accepts any case and rejects near misses" {
    // Case-sensitive matching sent `Model.GGUF` down the HF-directory path, where it
    // failed with a bare `NotDir` instead of loading.
    try std.testing.expect(isGguf("m.gguf"));
    try std.testing.expect(isGguf("m.GGUF"));
    try std.testing.expect(isGguf("m.Gguf"));
    try std.testing.expect(isGguf("/a/b/model.GGUF"));
    try std.testing.expect(!isGguf("m.gguf.bak"));
    try std.testing.expect(!isGguf("gguf"));
    try std.testing.expect(!isGguf("m.ggu"));
    try std.testing.expect(!isGguf(""));
    try std.testing.expect(!isGguf("dir/"));
    // A directory that merely contains "gguf" is not a gguf file.
    try std.testing.expect(!isGguf("model-gguf"));
    try std.testing.expect(!isGguf("a.ggufs"));
}

test "model name derivation" {
    try std.testing.expectEqualStrings("smollm2-135m-q8_0", modelName("models/smollm2-135m-q8_0.gguf"));
    try std.testing.expectEqualStrings("SmolLM2-135M-Instruct", modelName("models/SmolLM2-135M-Instruct"));
}
