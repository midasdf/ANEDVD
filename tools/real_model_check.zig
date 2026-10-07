//! One-off verification harness: reads a real HuggingFace model directory with
//! `src/hf.zig` / `src/safetensors.zig` and prints a deterministic dump that can
//! be diffed against an independent Python implementation of the same math.
//!
//! `zig run` cannot import files outside the module root, so run it through
//! `tools/real_model_check.sh <model-dir>` (it copies the two src files next to
//! this harness in a temp dir after verifying nothing else changed).

const std = @import("std");
const hf = @import("hf");
const safetensors = @import("safetensors");

const model_dir = "/tmp/hf_real/tiny-random-LlamaForCausalLM";

pub fn main() !void {
    var debug_alloc = std.heap.DebugAllocator(.{}){};
    defer _ = debug_alloc.deinit();
    const allocator = debug_alloc.allocator();

    const out = std.Io.File.stdout();
    var buffer: [4096]u8 = undefined;
    var writer = out.writer(safetensors.ioInstance(), &buffer);
    const w = &writer.interface;

    var cfg = try hf.loadConfig(allocator, model_dir);
    defer cfg.deinit(allocator);
    try w.print("config arch={s} hidden={d} layers={d} heads={d} kv_heads={d} head_dim={d} " ++
        "intermediate={d} vocab={d} eps={e} theta={e} tied={} max_pos={d} bos={?d} eos={?d}\n", .{
        cfg.arch,
        cfg.hidden_size,
        cfg.num_hidden_layers,
        cfg.num_attention_heads,
        cfg.num_key_value_heads,
        cfg.head_dim,
        cfg.intermediate_size,
        cfg.vocab_size,
        cfg.rms_norm_eps,
        cfg.rope_theta,
        cfg.tie_word_embeddings,
        cfg.max_position_embeddings,
        cfg.bos_token_id,
        cfg.eos_token_id,
    });

    const shards = try hf.findShards(allocator, model_dir);
    defer hf.freeShards(allocator, shards);
    try w.print("shards={d}\n", .{shards.len});
    for (shards) |path| try w.print("shard {s}\n", .{std.fs.path.basename(path)});

    for (shards) |path| {
        var shard = safetensors.Safetensors.load(allocator, path) catch |err| {
            // A real sharded repo may be represented by its index.json only.
            try w.print("shard {s} not readable: {s}\n", .{
                std.fs.path.basename(path),
                @errorName(err),
            });
            continue;
        };
        defer shard.deinit();
        const names = shard.names();
        try w.print("tensors={d}\n", .{names.len});
        // Stable order for diffing.
        const sorted = try allocator.dupe([]const u8, names);
        defer allocator.free(sorted);
        std.mem.sort([]const u8, sorted, {}, struct {
            fn lt(_: void, a: []const u8, b: []const u8) bool {
                return std.mem.lessThan(u8, a, b);
            }
        }.lt);

        for (sorted) |name| {
            const t = shard.tensor(name).?;
            const values = try shard.readF32(allocator, name);
            defer allocator.free(values);
            var sum: f64 = 0;
            for (values) |v| sum += v;
            const first_bits: u32 = if (values.len > 0) @bitCast(values[0]) else 0;
            try w.print("tensor {s} dtype={s} shape=[", .{ name, t.dtype.toString() });
            for (t.shape, 0..) |dim, i| {
                if (i != 0) try w.writeAll(",");
                try w.print("{d}", .{dim});
            }
            try w.print("] numel={d} first_bits={x:0>8} sum={d:.8}\n", .{
                values.len,
                first_bits,
                sum,
            });
        }
    }

    // Also exercise a by-name lookup through the shard list.
    const embed = (try hf.findTensorInShards(allocator, shards, "model.embed_tokens.weight")) orelse {
        try w.print("lookup model.embed_tokens.weight: absent\n", .{});
        try w.flush();
        return;
    };
    defer allocator.free(embed);
    var embed_sum: f64 = 0;
    for (embed) |v| embed_sum += v;
    try w.print("lookup model.embed_tokens.weight numel={d} sum={d:.8}\n", .{ embed.len, embed_sum });

    try w.flush();
}
