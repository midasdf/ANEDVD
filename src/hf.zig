//! HuggingFace model-directory reader: `config.json` plus safetensors shard
//! discovery. std-only; shard contents are read through `safetensors.zig`.
//!
//! Scope: the Llama-family architectures that share the same config keys
//! (`LlamaForCausalLM`, `Qwen2ForCausalLM`, `Qwen3ForCausalLM`, Mistral, ...).
//! Anything outside that family is rejected with `error.UnsupportedArchitecture`
//! and a log line naming the supported set.

const std = @import("std");
const Allocator = std.mem.Allocator;
const Io = std.Io;
const st = @import("safetensors.zig");

const log = std.log.scoped(.hf);

/// Architectures whose `config.json` uses the keys understood here.
pub const supported_architectures: []const []const u8 = &.{
    "LlamaForCausalLM",
    "Qwen2ForCausalLM",
    "Qwen3ForCausalLM",
    "Qwen2MoeForCausalLM",
    "Qwen3MoeForCausalLM",
    "MistralForCausalLM",
    "SmolLM3ForCausalLM",
};

/// `supported_architectures` rendered as a comma-separated list, at comptime.
pub const supported_architectures_list: []const u8 = blk: {
    var s: []const u8 = "";
    for (supported_architectures, 0..) |arch, i| {
        s = s ++ (if (i == 0) "" else ", ") ++ arch;
    }
    break :blk s;
};

/// `model_type` values mapped onto their architecture name, used when a config
/// has no `architectures` array.
const model_type_map = .{
    .{ "llama", "LlamaForCausalLM" },
    .{ "qwen2", "Qwen2ForCausalLM" },
    .{ "qwen2_moe", "Qwen2MoeForCausalLM" },
    .{ "qwen3", "Qwen3ForCausalLM" },
    .{ "qwen3_moe", "Qwen3MoeForCausalLM" },
    .{ "mistral", "MistralForCausalLM" },
    .{ "smollm3", "SmolLM3ForCausalLM" },
};

pub const Error = error{
    /// `config.json` is not valid JSON, or is not a JSON object.
    InvalidConfig,
    /// A field that has no sensible default (`hidden_size`,
    /// `num_hidden_layers`, `num_attention_heads`, `vocab_size`, `arch`) is
    /// absent or has the wrong type.
    MissingRequiredField,
    /// `architectures[0]` / `model_type` is outside `supported_architectures`.
    UnsupportedArchitecture,
    /// `num_attention_heads` is zero, so `head_dim` cannot be derived.
    InvalidAttentionGeometry,
    /// A shard index exists but is malformed (missing/empty `weight_map`).
    InvalidShardIndex,
    /// The model directory contains no `.safetensors` files.
    NoShardsFound,
};

pub const Config = struct {
    /// e.g. `"LlamaForCausalLM"`, `"Qwen2ForCausalLM"`.
    arch: []const u8,
    hidden_size: u32,
    num_hidden_layers: u32,
    num_attention_heads: u32,
    /// Defaults to `num_attention_heads` (i.e. no grouped-query attention).
    num_key_value_heads: u32,
    /// Defaults to `hidden_size / num_attention_heads`.
    head_dim: u32,
    /// `0` when the config omits it (e.g. some MoE configs).
    intermediate_size: u32,
    vocab_size: u32,
    /// Defaults to `1e-6`; Llama configs usually set `1e-5` explicitly.
    rms_norm_eps: f32,
    /// Defaults to `10000.0`; also read from `rope_parameters.rope_theta`
    /// and `rope_scaling.rope_theta`.
    rope_theta: f32,
    /// Defaults to `false`.
    tie_word_embeddings: bool,
    /// `0` when the config omits it (treated as "unspecified").
    max_position_embeddings: u32,
    bos_token_id: ?u32,
    eos_token_id: ?u32,

    /// Frees `arch`. All other fields are value types.
    pub fn deinit(self: *Config, allocator: Allocator) void {
        allocator.free(self.arch);
        self.* = undefined;
    }

    /// `head_dim * num_attention_heads`, the width of the query projection.
    pub fn queryDim(self: Config) u32 {
        return self.head_dim *% self.num_attention_heads;
    }

    /// `head_dim * num_key_value_heads`, the width of the K/V projections.
    pub fn kvDim(self: Config) u32 {
        return self.head_dim *% self.num_key_value_heads;
    }

    /// True when the embedding matrix is shared with the LM head.
    pub fn tiedEmbeddings(self: Config) bool {
        return self.tie_word_embeddings;
    }
};

/// True when `arch` is one of `supported_architectures`.
pub fn isSupportedArchitecture(arch: []const u8) bool {
    for (supported_architectures) |known| {
        if (std.mem.eql(u8, arch, known)) return true;
    }
    return false;
}

/// Reads `<dir>/config.json`.
///
/// Missing optional keys take their defaults; missing required keys and
/// unsupported architectures produce specific errors (see `Error`).
pub fn loadConfig(allocator: Allocator, dir: []const u8) !Config {
    const path = try joinPath(allocator, dir, "config.json");
    defer allocator.free(path);

    const bytes = try Io.Dir.cwd().readFileAlloc(st.ioInstance(), path, allocator, .unlimited);
    defer allocator.free(bytes);

    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const root = std.json.parseFromSliceLeaky(std.json.Value, arena.allocator(), bytes, .{}) catch
        return Error.InvalidConfig;
    if (root != .object) return Error.InvalidConfig;
    const obj = root.object;

    const arch_name = try resolveArch(allocator, obj);
    errdefer allocator.free(arch_name);

    const hidden_size = try requireU32(obj, &.{ "hidden_size", "n_embd" });
    const num_hidden_layers = try requireU32(obj, &.{ "num_hidden_layers", "n_layer" });
    const num_attention_heads = try requireU32(obj, &.{ "num_attention_heads", "n_head" });
    const vocab_size = try requireU32(obj, &.{"vocab_size"});

    const head_dim = getU32(obj, &.{"head_dim"}) orelse blk: {
        if (num_attention_heads == 0) return Error.InvalidAttentionGeometry;
        break :blk hidden_size / num_attention_heads;
    };

    return .{
        .arch = arch_name,
        .hidden_size = hidden_size,
        .num_hidden_layers = num_hidden_layers,
        .num_attention_heads = num_attention_heads,
        .num_key_value_heads = getU32(obj, &.{"num_key_value_heads"}) orelse num_attention_heads,
        .head_dim = head_dim,
        .intermediate_size = getU32(obj, &.{ "intermediate_size", "n_inner" }) orelse 0,
        .vocab_size = vocab_size,
        .rms_norm_eps = getF32(obj, &.{"rms_norm_eps"}) orelse 1e-6,
        .rope_theta = getF32(obj, &.{"rope_theta"}) orelse
            getNestedF32(obj, "rope_parameters", "rope_theta") orelse
            getNestedF32(obj, "rope_scaling", "rope_theta") orelse 10000.0,
        .tie_word_embeddings = getBool(obj, &.{"tie_word_embeddings"}) orelse false,
        .max_position_embeddings = getU32(obj, &.{
            "max_position_embeddings",
            "n_positions",
            "max_seq_len",
        }) orelse 0,
        .bos_token_id = getU32(obj, &.{"bos_token_id"}),
        .eos_token_id = getU32OrFirstOfArray(obj, "eos_token_id"),
    };
}

fn resolveArch(allocator: Allocator, obj: std.json.ObjectMap) ![]const u8 {
    var raw: ?[]const u8 = null;
    if (obj.get("architectures")) |value| {
        if (value == .array and value.array.items.len > 0) {
            const first = value.array.items[0];
            if (first == .string and first.string.len > 0) raw = first.string;
        }
    }
    if (raw == null) {
        const model_type = getString(obj, &.{"model_type"}) orelse return Error.MissingRequiredField;
        raw = modelTypeToArch(model_type) orelse model_type;
    }

    const arch = try allocator.dupe(u8, raw.?);
    if (!isSupportedArchitecture(arch)) {
        // Deliberately `warn`, not `err`: the Zig test runner turns any `err`
        // log into a failed test run, and callers already receive
        // `error.UnsupportedArchitecture`.
        log.warn("unsupported architecture '{s}'; supported architectures: {s}", .{
            arch,
            supported_architectures_list,
        });
        allocator.free(arch);
        return Error.UnsupportedArchitecture;
    }
    return arch;
}

fn modelTypeToArch(model_type: []const u8) ?[]const u8 {
    inline for (model_type_map) |entry| {
        if (std.mem.eql(u8, model_type, entry[0])) return entry[1];
    }
    return null;
}

/// Every `.safetensors` shard path in `dir`.
///
/// When `<dir>/model.safetensors.index.json` exists, its `weight_map` is the
/// source of truth and the union of its shards (deduplicated, sorted by name)
/// is returned. Otherwise every `*.safetensors` file in `dir` is returned,
/// sorted by name.
///
/// Caller owns both the strings and the outer slice; release them with
/// `freeShards`. Fails with `error.NoShardsFound` when nothing matches.
pub fn findShards(allocator: Allocator, dir: []const u8) ![][]const u8 {
    const index_path = try joinPath(allocator, dir, "model.safetensors.index.json");
    defer allocator.free(index_path);

    const io = st.ioInstance();
    const index_bytes: ?[]u8 = Io.Dir.cwd().readFileAlloc(io, index_path, allocator, .unlimited) catch |err| switch (err) {
        error.FileNotFound => null,
        else => return err,
    };
    if (index_bytes) |bytes| {
        defer allocator.free(bytes);
        return shardsFromIndex(allocator, dir, bytes);
    }

    var names: std.ArrayList([]const u8) = .empty;
    errdefer {
        for (names.items) |name| allocator.free(name);
        names.deinit(allocator);
    }

    var d = try Io.Dir.cwd().openDir(io, dir, .{ .iterate = true });
    defer d.close(io);
    var it = d.iterate();
    while (try it.next(io)) |entry| {
        if (entry.kind == .directory) continue;
        if (!std.mem.endsWith(u8, entry.name, ".safetensors")) continue;
        try names.append(allocator, try joinPath(allocator, dir, entry.name));
    }
    if (names.items.len == 0) return Error.NoShardsFound;

    std.mem.sort([]const u8, names.items, {}, lessThanStr);
    return names.toOwnedSlice(allocator);
}

fn shardsFromIndex(allocator: Allocator, dir: []const u8, bytes: []const u8) ![][]const u8 {
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const root = std.json.parseFromSliceLeaky(std.json.Value, arena.allocator(), bytes, .{}) catch
        return Error.InvalidShardIndex;
    if (root != .object) return Error.InvalidShardIndex;
    const weight_map = root.object.get("weight_map") orelse return Error.InvalidShardIndex;
    if (weight_map != .object) return Error.InvalidShardIndex;

    var names: std.ArrayList([]const u8) = .empty;
    errdefer {
        for (names.items) |name| allocator.free(name);
        names.deinit(allocator);
    }

    var it = weight_map.object.iterator();
    while (it.next()) |kv| {
        const value = kv.value_ptr.*;
        if (value != .string) return Error.InvalidShardIndex;
        const file = value.string;
        if (!std.mem.endsWith(u8, file, ".safetensors")) continue;
        var seen = false;
        for (names.items) |existing| {
            if (std.mem.eql(u8, std.fs.path.basename(existing), std.fs.path.basename(file))) {
                seen = true;
                break;
            }
        }
        if (seen) continue;
        try names.append(allocator, try joinPath(allocator, dir, file));
    }
    if (names.items.len == 0) return Error.NoShardsFound;

    std.mem.sort([]const u8, names.items, {}, lessThanStr);
    return names.toOwnedSlice(allocator);
}

/// Releases the result of `findShards` (each path, then the outer slice).
pub fn freeShards(allocator: Allocator, shards: []const []const u8) void {
    for (shards) |path| allocator.free(path);
    allocator.free(shards);
}

/// Loads the first shard that contains `name` and converts it to `f32`.
/// Returns null when no shard has that tensor. Shards that do not exist on disk
/// are skipped; other I/O and parse errors propagate.
pub fn findTensorInShards(
    allocator: Allocator,
    shard_paths: []const []const u8,
    name: []const u8,
) !?[]f32 {
    return findTensorInShardsAs(f32, allocator, shard_paths, name);
}

/// As `findTensorInShards`, but returns `f16` values.
pub fn findTensorInShardsF16(
    allocator: Allocator,
    shard_paths: []const []const u8,
    name: []const u8,
) !?[]f16 {
    return findTensorInShardsAs(f16, allocator, shard_paths, name);
}

fn findTensorInShardsAs(
    comptime T: type,
    allocator: Allocator,
    shard_paths: []const []const u8,
    name: []const u8,
) !?[]T {
    for (shard_paths) |path| {
        var shard = st.Safetensors.load(allocator, path) catch |err| switch (err) {
            error.FileNotFound => continue,
            else => return err,
        };
        if (!shard.has(name)) {
            shard.deinit();
            continue;
        }
        defer shard.deinit();
        const values: []T = switch (T) {
            f32 => try shard.readF32(allocator, name),
            f16 => try shard.readF16(allocator, name),
            else => @compileError("findTensorInShardsAs: unsupported element type"),
        };
        return values;
    }
    return null;
}

fn joinPath(allocator: Allocator, dir: []const u8, name: []const u8) ![]u8 {
    if (dir.len == 0) return allocator.dupe(u8, name);
    return std.fs.path.join(allocator, &.{ dir, name });
}

fn lessThanStr(_: void, a: []const u8, b: []const u8) bool {
    return std.mem.lessThan(u8, a, b);
}

// --- JSON helpers -----------------------------------------------------------

fn getValue(obj: std.json.ObjectMap, keys: []const []const u8) ?std.json.Value {
    for (keys) |key| {
        if (obj.get(key)) |value| {
            if (value != .null) return value;
        }
    }
    return null;
}

fn requireU32(obj: std.json.ObjectMap, keys: []const []const u8) !u32 {
    return getU32(obj, keys) orelse Error.MissingRequiredField;
}

fn getU32(obj: std.json.ObjectMap, keys: []const []const u8) ?u32 {
    const value = getValue(obj, keys) orelse return null;
    const n = toI64(value) orelse return null;
    if (n < 0) return null;
    return std.math.cast(u32, n);
}

/// Some configs write a single token id, others write a list of them; the first
/// entry wins.
fn getU32OrFirstOfArray(obj: std.json.ObjectMap, key: []const u8) ?u32 {
    const value = obj.get(key) orelse return null;
    switch (value) {
        .array => |arr| {
            if (arr.items.len == 0) return null;
            const n = toI64(arr.items[0]) orelse return null;
            if (n < 0) return null;
            return std.math.cast(u32, n);
        },
        else => return getU32(obj, &.{key}),
    }
}

/// Accepts JSON integers, floats (truncated toward zero) and numeric strings.
fn toI64(value: std.json.Value) ?i64 {
    return switch (value) {
        .integer => |i| i,
        .float => |f| blk: {
            if (!std.math.isFinite(f)) break :blk null;
            if (f < -9.2233720368547758e18 or f > 9.2233720368547758e18) break :blk null;
            break :blk @intFromFloat(f);
        },
        .number_string, .string => |s| std.fmt.parseInt(i64, s, 10) catch null,
        else => null,
    };
}

fn getF32(obj: std.json.ObjectMap, keys: []const []const u8) ?f32 {
    const value = getValue(obj, keys) orelse return null;
    return switch (value) {
        .integer => |i| @floatFromInt(i),
        .float => |f| @floatCast(f),
        .number_string, .string => |s| std.fmt.parseFloat(f32, s) catch null,
        else => null,
    };
}

fn getNestedF32(obj: std.json.ObjectMap, section: []const u8, key: []const u8) ?f32 {
    const value = obj.get(section) orelse return null;
    if (value != .object) return null;
    return getF32(value.object, &.{key});
}

fn getBool(obj: std.json.ObjectMap, keys: []const []const u8) ?bool {
    const value = getValue(obj, keys) orelse return null;
    return switch (value) {
        .bool => |b| b,
        .integer => |i| i != 0,
        .string => |s| std.mem.eql(u8, s, "true"),
        else => null,
    };
}

fn getString(obj: std.json.ObjectMap, keys: []const []const u8) ?[]const u8 {
    const value = getValue(obj, keys) orelse return null;
    return switch (value) {
        .string => |s| s,
        else => null,
    };
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

const testing = std.testing;

/// Writes `contents` as `<dir>/config.json` in a fresh tmp dir and returns the
/// cwd-relative directory path (caller frees).
fn configDir(allocator: Allocator, tmp: *testing.TmpDir, contents: []const u8) ![]u8 {
    try tmp.dir.writeFile(testing.io, .{ .sub_path = "config.json", .data = contents });
    return std.fs.path.join(allocator, &.{ ".zig-cache", "tmp", &tmp.sub_path });
}

fn tmpDirPath(allocator: Allocator, tmp: *testing.TmpDir) ![]u8 {
    return std.fs.path.join(allocator, &.{ ".zig-cache", "tmp", &tmp.sub_path });
}

const qwen2_config =
    \\{
    \\  "architectures": ["Qwen2ForCausalLM"],
    \\  "hidden_size": 896,
    \\  "num_hidden_layers": 24,
    \\  "num_attention_heads": 14,
    \\  "num_key_value_heads": 2,
    \\  "intermediate_size": 4864,
    \\  "vocab_size": 151936,
    \\  "rms_norm_eps": 1e-06,
    \\  "rope_parameters": {"rope_theta": 1000000.0, "rope_type": "default"},
    \\  "tie_word_embeddings": true,
    \\  "max_position_embeddings": 32768,
    \\  "bos_token_id": 151643,
    \\  "eos_token_id": 151645,
    \\  "model_type": "qwen2"
    \\}
;

test "loadConfig: Qwen2-style config with rope_parameters" {
    const allocator = testing.allocator;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try configDir(allocator, &tmp, qwen2_config);
    defer allocator.free(dir);

    var cfg = try loadConfig(allocator, dir);
    defer cfg.deinit(allocator);

    try testing.expectEqualStrings("Qwen2ForCausalLM", cfg.arch);
    try testing.expectEqual(@as(u32, 896), cfg.hidden_size);
    try testing.expectEqual(@as(u32, 24), cfg.num_hidden_layers);
    try testing.expectEqual(@as(u32, 14), cfg.num_attention_heads);
    try testing.expectEqual(@as(u32, 2), cfg.num_key_value_heads);
    try testing.expectEqual(@as(u32, 64), cfg.head_dim); // 896 / 14
    try testing.expectEqual(@as(u32, 4864), cfg.intermediate_size);
    try testing.expectEqual(@as(u32, 151936), cfg.vocab_size);
    try testing.expectEqual(@as(f32, 1e-6), cfg.rms_norm_eps);
    try testing.expectEqual(@as(f32, 1000000.0), cfg.rope_theta);
    try testing.expect(cfg.tie_word_embeddings);
    try testing.expectEqual(@as(u32, 32768), cfg.max_position_embeddings);
    try testing.expectEqual(@as(?u32, 151643), cfg.bos_token_id);
    try testing.expectEqual(@as(?u32, 151645), cfg.eos_token_id);
    try testing.expectEqual(@as(u32, 896), cfg.queryDim());
    try testing.expectEqual(@as(u32, 128), cfg.kvDim());
    try testing.expect(cfg.tiedEmbeddings());
}

test "loadConfig: defaults when optional keys are absent" {
    const allocator = testing.allocator;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try configDir(
        allocator,
        &tmp,
        \\{"architectures":["LlamaForCausalLM"],"hidden_size":64,"num_hidden_layers":2,
        \\ "num_attention_heads":8,"intermediate_size":128,"vocab_size":1000}
        ,
    );
    defer allocator.free(dir);

    var cfg = try loadConfig(allocator, dir);
    defer cfg.deinit(allocator);

    try testing.expectEqual(@as(u32, 8), cfg.num_key_value_heads); // == num_attention_heads
    try testing.expectEqual(@as(u32, 8), cfg.head_dim); // 64 / 8
    try testing.expectEqual(@as(f32, 1e-6), cfg.rms_norm_eps);
    try testing.expectEqual(@as(f32, 10000.0), cfg.rope_theta);
    try testing.expect(!cfg.tie_word_embeddings);
    try testing.expectEqual(@as(u32, 0), cfg.max_position_embeddings);
    try testing.expectEqual(@as(?u32, null), cfg.bos_token_id);
    try testing.expectEqual(@as(?u32, null), cfg.eos_token_id);
}

test "loadConfig: Llama style (model_type, eos list, rope_scaling, explicit head_dim)" {
    const allocator = testing.allocator;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try configDir(
        allocator,
        &tmp,
        \\{"model_type":"llama","hidden_size":128,"num_hidden_layers":3,
        \\ "num_attention_heads":4,"num_key_value_heads":4,"head_dim":32,
        \\ "intermediate_size":256,"vocab_size":2000,"rms_norm_eps":1e-5,
        \\ "rope_scaling":{"rope_theta":500000.0,"type":"linear"},
        \\ "tie_word_embeddings":false,"max_position_embeddings":2048,
        \\ "bos_token_id":1,"eos_token_id":[2,3]}
        ,
    );
    defer allocator.free(dir);

    var cfg = try loadConfig(allocator, dir);
    defer cfg.deinit(allocator);
    try testing.expectEqualStrings("LlamaForCausalLM", cfg.arch); // from model_type
    try testing.expectEqual(@as(u32, 32), cfg.head_dim); // explicit wins
    try testing.expectEqual(@as(f32, 1e-5), cfg.rms_norm_eps);
    try testing.expectEqual(@as(f32, 500000.0), cfg.rope_theta);
    try testing.expectEqual(@as(?u32, 2), cfg.eos_token_id); // first of list
    try testing.expectEqual(@as(u32, 2048), cfg.max_position_embeddings);
}

test "loadConfig: n_embd-style aliases and float-encoded integers" {
    const allocator = testing.allocator;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try configDir(
        allocator,
        &tmp,
        \\{"architectures":["MistralForCausalLM"],"n_embd":64.0,"n_layer":2,
        \\ "n_head":8,"n_inner":128,"vocab_size":100,"rms_norm_eps":"1e-05",
        \\ "rope_theta":"5000.0","tie_word_embeddings":1,"max_position_embeddings":512}
        ,
    );
    defer allocator.free(dir);

    var cfg = try loadConfig(allocator, dir);
    defer cfg.deinit(allocator);
    try testing.expectEqual(@as(u32, 64), cfg.hidden_size);
    try testing.expectEqual(@as(u32, 2), cfg.num_hidden_layers);
    try testing.expectEqual(@as(u32, 8), cfg.num_attention_heads);
    try testing.expectEqual(@as(u32, 8), cfg.num_key_value_heads);
    try testing.expectEqual(@as(u32, 8), cfg.head_dim);
    try testing.expectEqual(@as(u32, 128), cfg.intermediate_size);
    try testing.expectEqual(@as(f32, 1e-5), cfg.rms_norm_eps);
    try testing.expectEqual(@as(f32, 5000.0), cfg.rope_theta);
    try testing.expect(cfg.tie_word_embeddings); // 1 -> true
    try testing.expectEqual(@as(u32, 512), cfg.max_position_embeddings);
}

test "loadConfig: rejects unsupported architecture" {
    const allocator = testing.allocator;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try configDir(
        allocator,
        &tmp,
        \\{"architectures":["GPT2LMHeadModel"],"hidden_size":64,"num_hidden_layers":2,
        \\ "num_attention_heads":8,"intermediate_size":128,"vocab_size":100}
        ,
    );
    defer allocator.free(dir);

    try testing.expectError(Error.UnsupportedArchitecture, loadConfig(allocator, dir));
    try testing.expect(!isSupportedArchitecture("GPT2LMHeadModel"));
    try testing.expect(isSupportedArchitecture("LlamaForCausalLM"));
    try testing.expect(std.mem.indexOf(u8, supported_architectures_list, "Qwen2ForCausalLM") != null);
}

test "loadConfig: missing file, malformed JSON and missing required fields" {
    const allocator = testing.allocator;
    try testing.expectError(error.FileNotFound, loadConfig(allocator, "no/such/model/dir"));

    var tmp1 = testing.tmpDir(.{});
    defer tmp1.cleanup();
    const dir1 = try configDir(allocator, &tmp1, "[1,2,3]");
    defer allocator.free(dir1);
    try testing.expectError(Error.InvalidConfig, loadConfig(allocator, dir1));

    var tmp2 = testing.tmpDir(.{});
    defer tmp2.cleanup();
    const dir2 = try configDir(allocator, &tmp2, "{oops");
    defer allocator.free(dir2);
    try testing.expectError(Error.InvalidConfig, loadConfig(allocator, dir2));

    // no architectures and no model_type
    var tmp3 = testing.tmpDir(.{});
    defer tmp3.cleanup();
    const dir3 = try configDir(
        allocator,
        &tmp3,
        \\{"hidden_size":64,"num_hidden_layers":2,"num_attention_heads":8,"vocab_size":100}
        ,
    );
    defer allocator.free(dir3);
    try testing.expectError(Error.MissingRequiredField, loadConfig(allocator, dir3));

    // hidden_size absent
    var tmp4 = testing.tmpDir(.{});
    defer tmp4.cleanup();
    const dir4 = try configDir(
        allocator,
        &tmp4,
        \\{"architectures":["LlamaForCausalLM"],"num_hidden_layers":2,"num_attention_heads":8,
        \\ "intermediate_size":128,"vocab_size":100}
        ,
    );
    defer allocator.free(dir4);
    try testing.expectError(Error.MissingRequiredField, loadConfig(allocator, dir4));
}

test "loadConfig: zero attention heads cannot derive head_dim" {
    const allocator = testing.allocator;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try configDir(
        allocator,
        &tmp,
        \\{"architectures":["LlamaForCausalLM"],"hidden_size":64,"num_hidden_layers":2,
        \\ "num_attention_heads":0,"intermediate_size":128,"vocab_size":100}
        ,
    );
    defer allocator.free(dir);
    try testing.expectError(Error.InvalidAttentionGeometry, loadConfig(allocator, dir));
}

test "findShards: falls back to every *.safetensors file, sorted" {
    const allocator = testing.allocator;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();

    try tmp.dir.writeFile(testing.io, .{ .sub_path = "model-00002-of-00002.safetensors", .data = "" });
    try tmp.dir.writeFile(testing.io, .{ .sub_path = "model-00001-of-00002.safetensors", .data = "" });
    try tmp.dir.writeFile(testing.io, .{ .sub_path = "config.json", .data = "{}" });
    try tmp.dir.writeFile(testing.io, .{ .sub_path = "tokenizer.json", .data = "{}" });
    try tmp.dir.writeFile(testing.io, .{ .sub_path = "weights.gguf", .data = "" });

    const dir = try tmpDirPath(allocator, &tmp);
    defer allocator.free(dir);

    const shards = try findShards(allocator, dir);
    defer freeShards(allocator, shards);
    try testing.expectEqual(@as(usize, 2), shards.len);
    try testing.expect(std.mem.endsWith(u8, shards[0], "model-00001-of-00002.safetensors"));
    try testing.expect(std.mem.endsWith(u8, shards[1], "model-00002-of-00002.safetensors"));
}

test "findShards: prefers model.safetensors.index.json weight_map" {
    const allocator = testing.allocator;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();

    try tmp.dir.writeFile(testing.io, .{ .sub_path = "model-00001-of-00002.safetensors", .data = "" });
    try tmp.dir.writeFile(testing.io, .{ .sub_path = "model-00002-of-00002.safetensors", .data = "" });
    // A stray shard that the index does not mention must be excluded.
    try tmp.dir.writeFile(testing.io, .{ .sub_path = "orphan.safetensors", .data = "" });
    try tmp.dir.writeFile(testing.io, .{
        .sub_path = "model.safetensors.index.json",
        .data =
        \\{"metadata":{"total_size":2},
        \\ "weight_map":{"a":"model-00001-of-00002.safetensors",
        \\               "b":"model-00002-of-00002.safetensors",
        \\               "c":"model-00001-of-00002.safetensors"}}
        ,
    });

    const dir = try tmpDirPath(allocator, &tmp);
    defer allocator.free(dir);

    const shards = try findShards(allocator, dir);
    defer freeShards(allocator, shards);
    try testing.expectEqual(@as(usize, 2), shards.len);
    try testing.expect(std.mem.endsWith(u8, shards[0], "model-00001-of-00002.safetensors"));
    try testing.expect(std.mem.endsWith(u8, shards[1], "model-00002-of-00002.safetensors"));
}

test "findShards: errors when the directory has no shards or a broken index" {
    const allocator = testing.allocator;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(testing.io, .{ .sub_path = "config.json", .data = "{}" });
    const dir = try tmpDirPath(allocator, &tmp);
    defer allocator.free(dir);
    try testing.expectError(Error.NoShardsFound, findShards(allocator, dir));

    try tmp.dir.writeFile(testing.io, .{
        .sub_path = "model.safetensors.index.json",
        .data = "{\"metadata\":{}}",
    });
    try testing.expectError(Error.InvalidShardIndex, findShards(allocator, dir));

    try tmp.dir.writeFile(testing.io, .{
        .sub_path = "model.safetensors.index.json",
        .data = "{\"weight_map\":{\"a\":\"b.bin\"}}",
    });
    try testing.expectError(Error.NoShardsFound, findShards(allocator, dir));

    try testing.expectError(error.FileNotFound, findShards(allocator, "no/such/model/dir"));
}

test "findTensorInShards: locates a tensor across shards" {
    const allocator = testing.allocator;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();

    // shard 0 has unrelated weights, shard 1 has the tensor we look for.
    const shard0 = try buildShard(allocator, "other.weight", &.{ 9.0, 9.0 });
    defer allocator.free(shard0);
    const shard1 = try buildShard(allocator, "model.embed_tokens.weight", &.{ 1.5, -2.5, 3.25 });
    defer allocator.free(shard1);

    try tmp.dir.writeFile(testing.io, .{ .sub_path = "model-00001-of-00002.safetensors", .data = shard0 });
    try tmp.dir.writeFile(testing.io, .{ .sub_path = "model-00002-of-00002.safetensors", .data = shard1 });

    const dir = try tmpDirPath(allocator, &tmp);
    defer allocator.free(dir);
    const shards = try findShards(allocator, dir);
    defer freeShards(allocator, shards);
    try testing.expectEqual(@as(usize, 2), shards.len);

    const found = (try findTensorInShards(allocator, shards, "model.embed_tokens.weight")).?;
    defer allocator.free(found);
    try testing.expectEqualSlices(f32, &.{ 1.5, -2.5, 3.25 }, found);

    const as_f16 = (try findTensorInShardsF16(allocator, shards, "model.embed_tokens.weight")).?;
    defer allocator.free(as_f16);
    try testing.expectEqualSlices(f16, &.{ 1.5, -2.5, 3.25 }, as_f16);

    try testing.expectEqual(@as(?[]f32, null), try findTensorInShards(allocator, shards, "missing.weight"));

    // Missing shard files are skipped rather than failing the lookup.
    const with_missing = [_][]const u8{ "definitely/not/here.safetensors", shards[1] };
    const via_skip = (try findTensorInShards(allocator, &with_missing, "model.embed_tokens.weight")).?;
    defer allocator.free(via_skip);
    try testing.expectEqualSlices(f32, &.{ 1.5, -2.5, 3.25 }, via_skip);
}

/// Builds a one-tensor F32 safetensors image for the shard tests.
fn buildShard(allocator: Allocator, name: []const u8, values: []const f32) ![]u8 {
    const payload = try allocator.alloc(u8, values.len * 4);
    defer allocator.free(payload);
    for (values, 0..) |v, i| {
        std.mem.writeInt(u32, payload[i * 4 ..][0..4], @bitCast(v), .little);
    }
    return buildShardRaw(allocator, name, "F32", values.len, payload);
}

/// Builds a one-tensor BF16 safetensors image (values rounded to bf16).
fn buildBf16Shard(allocator: Allocator, name: []const u8, values: []const f32) ![]u8 {
    const payload = try allocator.alloc(u8, values.len * 2);
    defer allocator.free(payload);
    for (values, 0..) |v, i| {
        std.mem.writeInt(u16, payload[i * 2 ..][0..2], st.f32ToBf16(v), .little);
    }
    return buildShardRaw(allocator, name, "BF16", values.len, payload);
}

fn buildShardRaw(
    allocator: Allocator,
    name: []const u8,
    dtype: []const u8,
    numel: usize,
    payload: []const u8,
) ![]u8 {
    const header = try std.fmt.allocPrint(
        allocator,
        "{{\"{s}\":{{\"dtype\":\"{s}\",\"shape\":[{d}],\"data_offsets\":[0,{d}]}}}}",
        .{ name, dtype, numel, payload.len },
    );
    defer allocator.free(header);

    var image: std.ArrayList(u8) = .empty;
    errdefer image.deinit(allocator);
    try image.ensureTotalCapacity(allocator, 8 + header.len + payload.len);
    var len_bytes: [8]u8 = undefined;
    std.mem.writeInt(u64, &len_bytes, header.len, .little);
    image.appendSliceAssumeCapacity(&len_bytes);
    image.appendSliceAssumeCapacity(header);
    image.appendSliceAssumeCapacity(payload);
    return image.toOwnedSlice(allocator);
}

test "end-to-end: config.json + index.json + BF16/F16 shards" {
    const allocator = testing.allocator;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();

    try tmp.dir.writeFile(testing.io, .{
        .sub_path = "config.json",
        .data =
        \\{"architectures":["Qwen2ForCausalLM"],"hidden_size":64,"num_hidden_layers":2,
        \\ "num_attention_heads":8,"num_key_value_heads":2,"head_dim":8,
        \\ "intermediate_size":128,"vocab_size":1000,"rms_norm_eps":1e-06,
        \\ "rope_theta":10000.0,"tie_word_embeddings":false,"max_position_embeddings":512,
        \\ "bos_token_id":1,"eos_token_id":2}
        ,
    });

    // Shard 1 carries the embedding in BF16, shard 2 a q_proj weight in F16.
    // bf16 can represent these exactly: 1.0, -2.0, 0.5, 4.0
    const embed = try buildBf16Shard(allocator, "model.embed_tokens.weight", &.{ 1.0, -2.0, 0.5, 4.0 });
    defer allocator.free(embed);
    const q_proj = try buildShardRaw(
        allocator,
        "model.layers.0.self_attn.q_proj.weight",
        "F16",
        2,
        &.{ 0x00, 0x3C, 0x00, 0xBC }, // 1.0, -1.0 little-endian f16
    );
    defer allocator.free(q_proj);

    try tmp.dir.writeFile(testing.io, .{ .sub_path = "model-00001-of-00002.safetensors", .data = embed });
    try tmp.dir.writeFile(testing.io, .{ .sub_path = "model-00002-of-00002.safetensors", .data = q_proj });
    try tmp.dir.writeFile(testing.io, .{
        .sub_path = "model.safetensors.index.json",
        .data =
        \\{"metadata":{"total_size":12},
        \\ "weight_map":{"model.embed_tokens.weight":"model-00001-of-00002.safetensors",
        \\               "model.layers.0.self_attn.q_proj.weight":"model-00002-of-00002.safetensors"}}
        ,
    });

    const dir = try tmpDirPath(allocator, &tmp);
    defer allocator.free(dir);

    var cfg = try loadConfig(allocator, dir);
    defer cfg.deinit(allocator);
    try testing.expectEqual(@as(u32, 2), cfg.num_key_value_heads);
    try testing.expectEqual(@as(u32, 8), cfg.head_dim);
    try testing.expectEqual(@as(u32, 16), cfg.kvDim());

    const shards = try findShards(allocator, dir);
    defer freeShards(allocator, shards);
    try testing.expectEqual(@as(usize, 2), shards.len);

    const embed_values = (try findTensorInShards(allocator, shards, "model.embed_tokens.weight")).?;
    defer allocator.free(embed_values);
    try testing.expectEqualSlices(f32, &.{ 1.0, -2.0, 0.5, 4.0 }, embed_values);

    const q_values = (try findTensorInShards(allocator, shards, "model.layers.0.self_attn.q_proj.weight")).?;
    defer allocator.free(q_values);
    try testing.expectEqualSlices(f32, &.{ 1.0, -1.0 }, q_values);

    // A tensor that exists in no shard yields null, not an error.
    try testing.expectEqual(@as(?[]f32, null), try findTensorInShards(allocator, shards, "lm_head.weight"));
}
