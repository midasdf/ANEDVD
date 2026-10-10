//! GPT-2-style byte-level BPE tokenizer (std-only), fed either from a GGUF
//! file (`tokenizer.ggml.*` metadata) or from an HF `tokenizer.json`.
//!
//! Encoding pipeline (matching the reference GPT-2 / llama.cpp behaviour):
//!   1. optional BOS
//!   2. special/added tokens are matched literally on the raw text
//!   3. the remaining spans are pre-tokenized with an approximation of the
//!      GPT-2 regex `'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`
//!   4. each pre-token's bytes are mapped through the GPT-2 byte->unicode
//!      table (so a space becomes `Ġ`, U+0120)
//!   5. BPE merges are applied by merge rank
//!   6. symbols are looked up in the vocabulary
//!
//! `decode` concatenates token strings and reverses the byte->unicode mapping,
//! skipping CONTROL tokens (BOS/EOS and other specials), so
//! `decode(encode(text)) == text` for ordinary text.
//!
//! Documented limitations:
//! * `tokenizer.ggml.pre` is ignored: the GPT-2 pre-tokenizer above is used
//!   for every model.  For "llama-bpe"/"qwen2" vocabularies the difference is
//!   that those split digit runs into single digits, so numbers may tokenize
//!   into fewer tokens than the reference implementation produces.
//! * `\p{L}`/`\p{N}`/`\s` are approximated by the ranges in `isLetter`,
//!   `isDigit` and `isWhitespace` (exact for ASCII, covering the major Unicode
//!   blocks otherwise).
//! * BPE merges are applied with the standard rank-ordered algorithm, which is
//!   quadratic in pre-token length; pre-tokens are word-sized, so this is fine
//!   for ordinary text.

const std = @import("std");
const Allocator = std.mem.Allocator;
const Io = std.Io;
const gguf = @import("gguf.zig");

/// `tokenizer.ggml.token_type` values.
pub const TokenType = enum(u32) {
    undefined = 0,
    normal = 1,
    unknown = 2,
    control = 3,
    user_defined = 4,
    unused = 5,
    byte = 6,
    _,

    /// Tokens that `decode` skips and that `encode` matches literally.
    pub fn isSpecial(self: TokenType) bool {
        return self == .control or self == .user_defined;
    }
};

/// GPT-2 byte -> unicode codepoint table: printable bytes map to themselves,
/// the remaining 68 bytes map to U+0100.. (so byte 32/space becomes U+0120 Ġ).
pub const byte_to_unicode: [256]u21 = blk: {
    var map: [256]u21 = @splat(0);
    var used: [256]bool = @splat(false);
    for (33..127) |b| {
        map[b] = @intCast(b);
        used[b] = true;
    }
    for (161..173) |b| {
        map[b] = @intCast(b);
        used[b] = true;
    }
    for (174..256) |b| {
        map[b] = @intCast(b);
        used[b] = true;
    }
    var next: u21 = 256;
    for (0..256) |b| {
        if (!used[b]) {
            map[b] = next;
            next += 1;
        }
    }
    break :blk map;
};

/// Largest codepoint produced by `byte_to_unicode` (256 + 67 = 323).
pub const max_mapped_codepoint: usize = blk: {
    var max: usize = 0;
    for (byte_to_unicode) |cp| max = @max(max, cp);
    break :blk max;
};

/// Reverse table: codepoint -> byte, -1 when the codepoint is not a mapped byte.
const unicode_to_byte: [max_mapped_codepoint + 1]i16 = blk: {
    var map: [max_mapped_codepoint + 1]i16 = @splat(-1);
    for (byte_to_unicode, 0..) |cp, b| map[cp] = @intCast(b);
    break :blk map;
};

/// Maps a byte-level codepoint back to its byte, or null if it is not one.
pub fn byteForCodepoint(cp: u21) ?u8 {
    if (cp >= unicode_to_byte.len) return null;
    const v = unicode_to_byte[cp];
    return if (v < 0) null else @intCast(v);
}

const decoded = struct { cp: u21, len: usize };

/// Decodes one codepoint; invalid UTF-8 is treated as a single raw byte so
/// that arbitrary byte strings survive a round trip.
fn decodeChar(s: []const u8) decoded {
    if (s.len == 0) return .{ .cp = 0, .len = 0 };
    const len = std.unicode.utf8ByteSequenceLength(s[0]) catch return .{ .cp = s[0], .len = 1 };
    if (len > s.len) return .{ .cp = s[0], .len = 1 };
    const cp = std.unicode.utf8Decode(s[0..len]) catch return .{ .cp = s[0], .len = 1 };
    return .{ .cp = cp, .len = len };
}

/// Approximation of Unicode `\p{L}` (documented, exact for ASCII): the major
/// letter blocks, so CJK/Greek/Cyrillic/accented Latin text is treated as
/// letters the way GPT-2's regex does.
pub fn isLetter(cp: u21) bool {
    return switch (cp) {
        'a'...'z', 'A'...'Z', 0xAA, 0xB5, 0xBA => true,
        0xC0...0xD6, 0xD8...0xF6, 0xF8...0x2FF => true, // Latin-1 letters, Latin Extended-A/B
        0x370...0x3FF, 0x400...0x52F => true, // Greek, Cyrillic
        0x531...0x58F => true, // Armenian
        0x5D0...0x5EA, 0x5EF...0x5F2 => true, // Hebrew
        0x620...0x64A, 0x66E...0x6D3 => true, // Arabic
        0x900...0x97F => true, // Devanagari
        0x1E00...0x1FFF => true, // Latin/Greek extended additional
        0x2C60...0x2C7F, 0xA720...0xA7FF => true,
        0x3040...0x30FF => true, // Hiragana, Katakana
        0x3400...0x4DBF, 0x4E00...0x9FFF => true, // CJK
        0xA960...0xA97F, 0xAC00...0xD7FF => true, // Hangul
        0xF900...0xFAFF => true, // CJK compatibility ideographs
        0x20000...0x3FFFF => true, // CJK extensions
        else => false,
    };
}

/// Approximation of Unicode `\p{N}`.
pub fn isDigit(cp: u21) bool {
    return switch (cp) {
        '0'...'9' => true,
        0x660...0x669, 0x6F0...0x6F9 => true, // Arabic-Indic
        0x966...0x96F, 0x9E6...0x9EF => true, // Devanagari, Bengali
        0xFF10...0xFF19 => true, // fullwidth
        else => false,
    };
}

/// Approximation of `\s` (ASCII plus the usual Unicode space separators).
pub fn isWhitespace(cp: u21) bool {
    return switch (cp) {
        ' ', '\t', '\n', '\r', 0x0B, 0x0C => true,
        0x85, 0xA0, 0x1680 => true,
        0x2000...0x200A, 0x2028, 0x2029, 0x202F, 0x205F, 0x3000 => true,
        else => false,
    };
}

const CharClass = enum { letter, digit, other };

fn classMatches(class: CharClass, cp: u21) bool {
    return switch (class) {
        .letter => isLetter(cp),
        .digit => isDigit(cp),
        .other => !isWhitespace(cp) and !isLetter(cp) and !isDigit(cp),
    };
}

/// The GPT-2 pre-tokenizer regex, implemented as a scanner.  Alternatives are
/// tried in the reference order, so e.g. " hello" (space + word) stays one
/// pre-token while "  hello" yields " " followed by " hello".
pub const PreTokenizer = struct {
    text: []const u8,
    pos: usize = 0,

    const contractions = [_][]const u8{ "'s", "'t", "'re", "'ve", "'m", "'ll", "'d" };

    pub fn next(self: *PreTokenizer) ?[]const u8 {
        if (self.pos >= self.text.len) return null;
        const start = self.pos;
        const rest = self.text[self.pos..];

        // 1. contractions ('s|'t|'re|'ve|'m|'ll|'d)
        for (contractions) |c| {
            if (std.mem.startsWith(u8, rest, c)) {
                self.pos += c.len;
                return self.text[start..self.pos];
            }
        }

        // 2.-4. optional space + run of letters / numbers / other
        inline for (.{ CharClass.letter, CharClass.digit, CharClass.other }) |class| {
            if (self.runEnd(start, class)) |end| {
                self.pos = end;
                return self.text[start..end];
            }
        }

        // 5. \s+(?!\S): all but the last whitespace character of a run that is
        //    followed by a non-whitespace character.  A single whitespace
        //    character cannot satisfy the lookahead, so it falls through to 6.
        const ws_end = self.whitespaceEnd(start);
        if (ws_end > start) {
            if (ws_end < self.text.len and ws_end - 1 > start) {
                self.pos = ws_end - 1;
            } else {
                self.pos = ws_end;
            }
            return self.text[start..self.pos];
        }

        // 6. \s+ (unreachable in practice: 5 already covers whitespace runs)
        const d = decodeChar(rest);
        self.pos += @max(d.len, 1);
        return self.text[start..self.pos];
    }

    fn runEnd(self: PreTokenizer, start: usize, class: CharClass) ?usize {
        var i = start;
        if (i < self.text.len and self.text[i] == ' ') i += 1;
        const body = i;
        while (i < self.text.len) {
            const d = decodeChar(self.text[i..]);
            if (!classMatches(class, d.cp)) break;
            i += d.len;
        }
        if (i == body) return null;
        return i;
    }

    fn whitespaceEnd(self: PreTokenizer, start: usize) usize {
        var i = start;
        while (i < self.text.len) {
            const d = decodeChar(self.text[i..]);
            if (!isWhitespace(d.cp)) break;
            i += d.len;
        }
        return i;
    }
};

/// Splits `text` with the GPT-2 pre-tokenizer.  Returned slices alias `text`;
/// only the outer slice is allocated.
pub fn pretokenize(allocator: Allocator, text: []const u8) ![][]const u8 {
    var list: std.ArrayList([]const u8) = .empty;
    defer list.deinit(allocator);
    var it = PreTokenizer{ .text = text };
    while (it.next()) |piece| try list.append(allocator, piece);
    return list.toOwnedSlice(allocator);
}

const Special = struct {
    /// The token's text with the byte-level mapping reversed (e.g. "<s>").
    text: []const u8,
    id: u32,
};

/// A byte-level BPE tokenizer.  Owns all of its data: it stays valid after the
/// `Gguf` it was built from is deinitialized.
pub const Tokenizer = struct {
    allocator: Allocator,
    arena: std.heap.ArenaAllocator,
    /// id -> token string (byte-level mapped form, e.g. "Ġhello").
    tokens: [][]const u8 = &.{},
    token_types: []TokenType = &.{},
    /// token string -> id (first definition wins, except for added tokens).
    index: std.StringHashMapUnmanaged(u32) = .empty,
    /// "left\x00right" -> merge rank (lower rank merges first).
    merge_ranks: std.StringHashMapUnmanaged(u32) = .empty,
    /// Special tokens matched literally by `encode`, longest first.
    specials: []const Special = &.{},
    bos: ?u32 = null,
    eos: ?u32 = null,
    /// True when the vocabulary is SentencePiece (`tokenizer.ggml.model =
    /// "llama"`): pieces use U+2581 for a leading space and the vocabulary has
    /// no `merges` list, so encoding works by longest-match rather than BPE.
    sentencepiece: bool = false,
    /// The GGUF `general.architecture`, when the tokenizer came from one.
    arch: []const u8 = "",
    /// The GGUF `tokenizer.chat_template`, when present. This is the only
    /// reliable way to pick a template for models whose markers (<|user|> etc.)
    /// are plain text rather than vocabulary entries.
    chat_template: []const u8 = "",
    /// SentencePiece scores, used to derive token classes when the file has no
    /// `tokenizer.ggml.token_type` (TinyLlama, Llama-2 era).
    scores: ?[]const f32 = null,
    /// `tokenizer.ggml.add_bos_token` (default true).
    add_bos: bool = true,

    /// Builds a tokenizer from GGUF metadata:
    /// `tokenizer.ggml.tokens`, `tokenizer.ggml.merges`,
    /// `tokenizer.ggml.token_type`, `tokenizer.ggml.bos_token_id`,
    /// `tokenizer.ggml.eos_token_id`, `tokenizer.ggml.add_bos_token`.
    ///
    /// Returns `error.UnsupportedTokenizerModel` for non-BPE models (e.g.
    /// SentencePiece `model = "llama"`).
    /// Diagnostics from the last fromGguf failure (shown by the CLI).
    pub var last_error_detail: ?[]const u8 = null;

    pub fn fromGguf(allocator: Allocator, g: *const gguf.Gguf) !Tokenizer {
        last_error_detail = null;
        var is_spm = false;
        if (g.getString("tokenizer.ggml.model")) |model| {
            // The Gemma family declares the SentencePiece variant it uses by name rather than
            // calling it "llama": Gemma 4's files say `tokenizer.ggml.model = "gemma4"`, and the
            // layout is the one already implemented (U+2581 for a leading space, the same
            // token_type codes). Treating it as unknown refused the whole model.
            if (std.mem.eql(u8, model, "llama") or std.mem.startsWith(u8, model, "gemma")) {
                is_spm = true;
            } else if (!std.mem.eql(u8, model, "gpt2") and !std.mem.eql(u8, model, "bpe")) {
                last_error_detail = "only byte-level BPE (\"gpt2\"/\"bpe\") and SentencePiece (\"llama\") vocabularies are implemented.";
                return error.UnsupportedTokenizerModel;
            }
        } else {
            last_error_detail = "the GGUF file has no tokenizer.ggml.model key, so the vocabulary type is unknown.";
            return error.UnsupportedTokenizerModel;
        }
        if (g.getStringArray("tokenizer.ggml.tokens") == null) {
            last_error_detail = "the GGUF file has no tokenizer.ggml.tokens array.";
            return error.MissingTokens;
        }
        const tokens = g.getStringArray("tokenizer.ggml.tokens") orelse return error.MissingTokens;

        var t = Tokenizer{
            .allocator = allocator,
            .arena = std.heap.ArenaAllocator.init(allocator),
        };
        errdefer t.deinit();

        try t.setTokens(tokens);
        t.sentencepiece = is_spm;
        if (g.arch()) |arch| t.arch = try t.arena.allocator().dupe(u8, arch);
        if (g.getString("tokenizer.chat_template")) |tmpl| {
            t.chat_template = try t.arena.allocator().dupe(u8, tmpl);
        }
        if (g.getValue("tokenizer.ggml.token_type")) |v| {
            try t.setTokenTypes(v);
        } else if (is_spm) {
            // No token_type array: derive the classes from the scores, which is
            // how llama.cpp does it for SentencePiece vocabularies.
            try t.deriveTokenTypesFromScores(g);
        }
        if (g.getStringArray("tokenizer.ggml.merges")) |merges| try t.addMerges(merges);
        t.bos = g.getU32("tokenizer.ggml.bos_token_id");
        t.eos = g.getU32("tokenizer.ggml.eos_token_id");
        if (g.getBool("tokenizer.ggml.add_bos_token")) |b| t.add_bos = b;
        try t.buildSpecials();
        return t;
    }

    /// Reads an HF `tokenizer.json` from disk (BPE models only).
    pub fn fromTokenizerJson(allocator: Allocator, path: []const u8) !Tokenizer {
        const text = try Io.Dir.cwd().readFileAlloc(
            Io.Threaded.global_single_threaded.io(),
            path,
            allocator,
            .limited(1 << 30),
        );
        defer allocator.free(text);
        return fromTokenizerJsonSlice(allocator, text);
    }

    /// Same as `fromTokenizerJson`, from an in-memory document.
    pub fn fromTokenizerJsonSlice(allocator: Allocator, text: []const u8) !Tokenizer {
        const parsed = try std.json.parseFromSlice(std.json.Value, allocator, text, .{});
        defer parsed.deinit();

        const root = parsed.value;
        if (root != .object) return error.InvalidTokenizerJson;
        const model = root.object.get("model") orelse return error.InvalidTokenizerJson;
        if (model != .object) return error.InvalidTokenizerJson;
        const model_type = model.object.get("type") orelse return error.InvalidTokenizerJson;
        if (model_type != .string or !std.mem.eql(u8, model_type.string, "BPE")) {
            return error.UnsupportedTokenizerModel;
        }

        var t = Tokenizer{
            .allocator = allocator,
            .arena = std.heap.ArenaAllocator.init(allocator),
        };
        errdefer t.deinit();
        const a = t.arena.allocator();

        // Pass 1: size the id table from the vocab.
        const vocab = model.object.get("vocab") orelse return error.InvalidTokenizerJson;
        if (vocab != .object) return error.InvalidTokenizerJson;
        var max_id: u32 = 0;
        {
            var it = vocab.object.iterator();
            while (it.next()) |entry| {
                const id = jsonId(entry.value_ptr.*) orelse return error.InvalidTokenizerJson;
                max_id = @max(max_id, id);
            }
        }
        if (root.object.get("added_tokens")) |added| {
            if (added == .array) {
                for (added.array.items) |item| {
                    if (item != .object) continue;
                    const id_value = item.object.get("id") orelse continue;
                    const id = jsonId(id_value) orelse continue;
                    max_id = @max(max_id, id);
                }
            }
        }

        const count: usize = @as(usize, max_id) + 1;
        const tokens = try a.alloc([]const u8, count);
        for (tokens) |*tok| tok.* = "";
        t.tokens = tokens;
        const types = try a.alloc(TokenType, count);
        @memset(types, .unused);
        t.token_types = types;
        try t.index.ensureTotalCapacity(a, @intCast(count));

        {
            var it = vocab.object.iterator();
            while (it.next()) |entry| {
                const id = jsonId(entry.value_ptr.*) orelse return error.InvalidTokenizerJson;
                const tok = try a.dupe(u8, entry.key_ptr.*);
                tokens[id] = tok;
                types[id] = .normal;
                if (tok.len != 0) {
                    const gop = t.index.getOrPutAssumeCapacity(tok);
                    if (!gop.found_existing) gop.value_ptr.* = id;
                }
            }
        }

        // added_tokens win over the vocab for their content.
        if (root.object.get("added_tokens")) |added| {
            if (added == .array) {
                for (added.array.items) |item| {
                    if (item != .object) continue;
                    const id_value = item.object.get("id") orelse continue;
                    const id = jsonId(id_value) orelse continue;
                    const content = item.object.get("content") orelse continue;
                    if (content != .string) continue;
                    const special = if (item.object.get("special")) |s| (s == .bool and s.bool) else false;
                    const tok = try a.dupe(u8, content.string);
                    tokens[id] = tok;
                    types[id] = if (special) .control else .normal;
                    if (tok.len != 0) try t.index.put(a, tok, id);
                }
            }
        }

        if (model.object.get("merges")) |merges| try t.addJsonMerges(merges);
        try t.buildSpecials();
        return t;
    }

    pub fn deinit(self: *Tokenizer) void {
        self.arena.deinit();
        self.* = undefined;
    }

    /// Number of token ids (including unused slots).
    pub fn vocabSize(self: *const Tokenizer) usize {
        return self.tokens.len;
    }

    pub fn bosId(self: *const Tokenizer) ?u32 {
        return self.bos;
    }

    pub fn eosId(self: *const Tokenizer) ?u32 {
        return self.eos;
    }

    /// Raw token string for `id` (byte-level mapped form, e.g. "Ġhello"),
    /// or "" when the id is out of range.
    pub fn tokenText(self: *const Tokenizer, id: u32) []const u8 {
        if (id >= self.tokens.len) return "";
        return self.tokens[id];
    }

    pub fn tokenType(self: *const Tokenizer, id: u32) TokenType {
        if (id >= self.token_types.len) return .undefined;
        return self.token_types[id];
    }

    /// Looks a token string up; the argument must be in byte-level form.
    pub fn tokenId(self: *const Tokenizer, token: []const u8) ?u32 {
        return self.index.get(token);
    }

    /// Encodes `text`.
    ///
    /// When `add_special` is true, BOS is prepended if the tokenizer knows one
    /// and `tokenizer.ggml.add_bos_token` was not explicitly false.
    /// The caller owns the returned slice.
    pub fn encode(self: *const Tokenizer, allocator: Allocator, text: []const u8, add_special: bool) ![]u32 {
        const a = allocator;
        var out: std.ArrayList(u32) = .empty;
        errdefer out.deinit(a);

        if (add_special and self.add_bos) {
            if (self.bos) |b| try out.append(a, b);
        }

        var i: usize = 0;
        var segment_start: usize = 0;
        while (i < text.len) {
            const spec = self.matchSpecial(text[i..]) orelse {
                i += 1;
                continue;
            };
            if (i > segment_start) try self.encodeSegment(a, text[segment_start..i], &out);
            try out.append(a, spec.id);
            i += spec.text.len;
            segment_start = i;
        }
        if (segment_start < text.len) try self.encodeSegment(a, text[segment_start..], &out);

        return out.toOwnedSlice(a);
    }

    /// Derive token classes from SentencePiece scores.
    ///
    /// SentencePiece vocabularies usually ship without `tokenizer.ggml.token_type`.
    /// llama.cpp falls back to the scores: control pieces (like <s>, </s>) have
    /// score 0, and the 256 byte pieces have the most negative scores. This
    /// mirrors that so `decode` skips the right pieces.
    fn deriveTokenTypesFromScores(self: *Tokenizer, g: *const gguf.Gguf) !void {
        const scores_val = g.getValue("tokenizer.ggml.scores") orelse return;
        if (scores_val != .array) return;
        const raw: []const f32 = switch (scores_val.array.data) {
            .f32 => |s| s,
            .f64 => |s| blk: {
                const a0 = self.arena.allocator();
                const conv = try a0.alloc(f32, s.len);
                for (s, 0..) |v, i| conv[i] = @floatCast(v);
                break :blk conv;
            },
            else => return,
        };
        if (raw.len < self.tokens.len) return;
        const a = self.arena.allocator();
        const owned = try a.dupe(f32, raw[0..self.tokens.len]);
        self.scores = owned;

        // The 256 single-byte pieces have the lowest scores; control tokens sit
        // at 0.0. Everything else is a normal piece.
        var lowest: f32 = 0;
        for (owned) |v| lowest = @min(lowest, v);

        for (self.tokens, 0..) |tok, i| {
            if (owned[i] <= lowest) {
                self.token_types[i] = .byte;
            } else if (owned[i] == 0 and tok.len > 1 and tok[0] == '<') {
                self.token_types[i] = .control;
            }
        }
    }

    /// SentencePiece encoding: split on whitespace, prefix each word with U+2581,
    /// then take the longest matching vocabulary piece at each position.
    ///
    /// Not BPE: a SentencePiece GGUF has no `merges` list, so the merge ranks the
    /// GPT-2 path relies on do not exist. Longest-match over the pieces is what
    /// llama.cpp's SPM path approximates.
    fn encodeSegmentSpm(self: *const Tokenizer, a: Allocator, text: []const u8, out: *std.ArrayList(u32)) !void {
        const SEP = "\u{2581}";
        var i: usize = 0;
        while (i < text.len) {
            // A literal space is encoded by the separator that prefixes the next
            // piece, so consume it here rather than emitting a <0x20> piece (which
            // would double the space on decode).
            if (text[i] == ' ') {
                i += 1;
                continue;
            }
            // At a word boundary, try the piece formed with the separator first.
            const at_boundary = blk: {
                if (i == 0) break :blk true;
                const prev = text[i - 1];
                break :blk prev == ' ' or prev == '\n' or prev == '\t';
            };
            var best_len: usize = 0;
            var best_id: u32 = 0;
            // Candidate pieces: longest match starting at i, with or without the
            // leading separator.
            var len: usize = text.len - i;
            while (len > 0) : (len -= 1) {
                const piece = text[i..][0..len];
                if (at_boundary) {
                    var buf: [256]u8 = undefined;
                    const with_sep = std.fmt.bufPrint(&buf, "{s}{s}", .{ SEP, piece }) catch null;
                    if (with_sep) |ws| {
                        if (self.index.get(ws)) |id| {
                            if (len > best_len) {
                                best_len = len;
                                best_id = id;
                            }
                        }
                    }
                }
                if (self.index.get(piece)) |id| {
                    if (len > best_len) {
                        best_len = len;
                        best_id = id;
                    }
                }
                if (len > 64) {
                    // Long pieces are rare; cap the scan so a pathological input
                    // cannot make this quadratic in the whole segment.
                    len = 64;
                    continue;
                }
            }
            if (best_len > 0) {
                try out.append(a, best_id);
                i += best_len;
                continue;
            }
            // No piece matched: fall back to a single byte. SentencePiece
            // vocabularies contain all 256 byte pieces, so this should not
            // normally happen.
            try out.append(a, self.byteFallbackId(text[i]) orelse return error.UnencodableText);
            i += 1;
        }
    }

    /// The vocabulary id whose piece is the raw byte `b` (SentencePiece byte
    /// pieces are usually `\u2581`+letter forms, so this is a best effort).
    fn byteFallbackId(self: *const Tokenizer, b: u8) ?u32 {
        var buf: [8]u8 = undefined;
        const s = std.fmt.bufPrint(&buf, "<0x{X:0>2}>", .{b}) catch return null;
        if (self.index.get(s)) |id| return id;
        return self.index.get(&[_]u8{b});
    }

    /// Decodes token ids back to bytes.  CONTROL tokens (BOS/EOS and other
    /// specials) are skipped; unknown control-ish slots decode to nothing.
    pub fn decode(self: *const Tokenizer, allocator: Allocator, ids: []const u32) ![]u8 {
        var out: std.ArrayList(u8) = .empty;
        errdefer out.deinit(allocator);
        for (ids) |id| {
            if (id >= self.tokens.len) return error.InvalidTokenId;
            if (self.token_types[id].isSpecial()) continue;
            if (self.sentencepiece) {
                try appendDecodedSpm(allocator, &out, self.tokens[id]);
            } else {
                try appendDecoded(allocator, &out, self.tokens[id]);
            }
        }
        return out.toOwnedSlice(allocator);
    }

    /// Decodes a single token to bytes (CONTROL tokens decode to "").
    pub fn tokenBytes(self: *const Tokenizer, allocator: Allocator, id: u32) ![]u8 {
        if (id >= self.tokens.len) return error.InvalidTokenId;
        var out: std.ArrayList(u8) = .empty;
        errdefer out.deinit(allocator);
        if (!self.token_types[id].isSpecial()) {
            // The same branch `decode` takes, so streamed output and `decode`
            // never disagree (this path emitted literal "\u2581" for a while).
            if (self.sentencepiece) {
                try appendDecodedSpm(allocator, &out, self.tokens[id]);
            } else {
                try appendDecoded(allocator, &out, self.tokens[id]);
            }
        }
        return out.toOwnedSlice(allocator);
    }

    // ------------------------------------------------------------ internals

    fn setTokens(self: *Tokenizer, tokens: []const []const u8) !void {
        const a = self.arena.allocator();
        const list = try a.alloc([]const u8, tokens.len);
        for (tokens, 0..) |tok, i| list[i] = try a.dupe(u8, tok);
        self.tokens = list;

        const types = try a.alloc(TokenType, tokens.len);
        @memset(types, .normal);
        self.token_types = types;

        try self.index.ensureTotalCapacity(a, @intCast(tokens.len));
        for (list, 0..) |tok, i| {
            if (tok.len == 0) continue;
            const gop = self.index.getOrPutAssumeCapacity(tok);
            if (!gop.found_existing) gop.value_ptr.* = @intCast(i);
        }
    }

    fn setTokenTypes(self: *Tokenizer, value: gguf.Value) !void {
        const data = value.array.data;
        const n = @min(data.len(), self.token_types.len);
        for (0..n) |i| {
            const raw: i64 = switch (data) {
                .i32 => |s| s[i],
                .u32 => |s| s[i],
                .i64 => |s| s[i],
                .u64 => |s| @intCast(s[i]),
                .i16 => |s| s[i],
                .u16 => |s| s[i],
                .i8 => |s| s[i],
                .u8 => |s| s[i],
                else => return,
            };
            if (raw < 0 or raw > 6) continue;
            self.token_types[i] = @fromBackingInt(@intCast(@as(u32, @intCast(raw))));
        }
    }

    fn addMerges(self: *Tokenizer, merges: []const []const u8) !void {
        try self.merge_ranks.ensureTotalCapacity(self.arena.allocator(), @intCast(merges.len));
        for (merges, 0..) |m, merge_rank| {
            const sep = std.mem.indexOfScalar(u8, m, ' ') orelse continue;
            const right = m[sep + 1 ..];
            if (right.len == 0) continue;
            try self.putMerge(m[0..sep], right, @intCast(merge_rank));
        }
    }

    fn addJsonMerges(self: *Tokenizer, merges: std.json.Value) !void {
        if (merges != .array) return;
        try self.merge_ranks.ensureTotalCapacity(self.arena.allocator(), @intCast(merges.array.items.len));
        for (merges.array.items, 0..) |item, merge_rank| {
            switch (item) {
                // Newer tokenizer.json writes merges as ["a", "b"] pairs.
                .array => |pair| {
                    if (pair.items.len != 2) continue;
                    if (pair.items[0] != .string or pair.items[1] != .string) continue;
                    try self.putMerge(pair.items[0].string, pair.items[1].string, @intCast(merge_rank));
                },
                // Older files write a single "a b" string.
                .string => |s| {
                    const sep = std.mem.indexOfScalar(u8, s, ' ') orelse continue;
                    const right = s[sep + 1 ..];
                    if (right.len == 0) continue;
                    try self.putMerge(s[0..sep], right, @intCast(merge_rank));
                },
                else => continue,
            }
        }
    }

    fn putMerge(self: *Tokenizer, left: []const u8, right: []const u8, merge_rank: u32) !void {
        const a = self.arena.allocator();
        const key = try std.mem.concat(a, u8, &.{ left, "\x00", right });
        const gop = try self.merge_ranks.getOrPut(a, key);
        if (!gop.found_existing) gop.value_ptr.* = merge_rank;
    }

    fn buildSpecials(self: *Tokenizer) !void {
        const a = self.arena.allocator();
        var list: std.ArrayList(Special) = .empty;
        for (self.tokens, 0..) |tok, i| {
            if (!self.token_types[i].isSpecial()) continue;
            var buf: std.ArrayList(u8) = .empty;
            try appendDecoded(a, &buf, tok);
            if (buf.items.len == 0) continue;
            try list.append(a, .{ .text = buf.items, .id = @intCast(i) });
        }
        // Longest first, so "<|endoftext|>" wins over a hypothetical "<|".
        std.mem.sort(Special, list.items, {}, struct {
            fn lessThan(_: void, x: Special, y: Special) bool {
                return x.text.len > y.text.len;
            }
        }.lessThan);
        self.specials = try list.toOwnedSlice(a);
    }

    fn matchSpecial(self: *const Tokenizer, rest: []const u8) ?Special {
        for (self.specials) |spec| {
            if (std.mem.startsWith(u8, rest, spec.text)) return spec;
        }
        return null;
    }

    /// Pre-tokenizes and BPE-merges one plain-text span, or takes the
    /// SentencePiece path for `llama`-style vocabularies.
    fn encodeSegment(self: *const Tokenizer, a: Allocator, text: []const u8, out: *std.ArrayList(u32)) !void {
        if (self.sentencepiece) return self.encodeSegmentSpm(a, text, out);
        var mapped: std.ArrayList(u8) = .empty;
        defer mapped.deinit(a);
        var symbols: std.ArrayList(Symbol) = .empty;
        defer symbols.deinit(a);
        var key_buf: std.ArrayList(u8) = .empty;
        defer key_buf.deinit(a);

        var it = PreTokenizer{ .text = text };
        while (it.next()) |piece| {
            mapped.clearRetainingCapacity();
            symbols.clearRetainingCapacity();

            // One symbol per byte, encoded through the byte->unicode table.
            for (piece) |b| {
                var cp_buf: [4]u8 = undefined;
                const n = std.unicode.utf8Encode(byte_to_unicode[b], &cp_buf) catch unreachable;
                try symbols.append(a, .{ .start = mapped.items.len, .len = n });
                try mapped.appendSlice(a, cp_buf[0..n]);
            }

            try self.merge(a, &symbols, mapped.items, &key_buf);

            for (symbols.items) |s| {
                const symbol = mapped.items[s.start .. s.start + s.len];
                if (self.index.get(symbol)) |id| {
                    try out.append(a, id);
                } else {
                    try self.emitByteFallback(a, symbol, out);
                }
            }
        }
    }

    /// Standard BPE: repeatedly find the lowest-rank adjacent pair and merge
    /// every occurrence of it.
    fn merge(self: *const Tokenizer, a: Allocator, symbols: *std.ArrayList(Symbol), mapped: []const u8, key_buf: *std.ArrayList(u8)) !void {
        while (symbols.items.len >= 2) {
            var best: u32 = std.math.maxInt(u32);
            var found = false;
            for (0..symbols.items.len - 1) |i| {
                const l = mapped[symbols.items[i].start..][0..symbols.items[i].len];
                const r = mapped[symbols.items[i + 1].start..][0..symbols.items[i + 1].len];
                if (try self.pairRank(a, key_buf, l, r)) |merge_rank| {
                    if (merge_rank < best) {
                        best = merge_rank;
                        found = true;
                    }
                }
            }
            if (!found) break;

            var i: usize = 0;
            while (i + 1 < symbols.items.len) {
                const l = mapped[symbols.items[i].start..][0..symbols.items[i].len];
                const r = mapped[symbols.items[i + 1].start..][0..symbols.items[i + 1].len];
                const merge_rank = try self.pairRank(a, key_buf, l, r);
                if (merge_rank != null and merge_rank.? == best) {
                    symbols.items[i].len += symbols.items[i + 1].len;
                    _ = symbols.orderedRemove(i + 1);
                } else {
                    i += 1;
                }
            }
        }
    }

    fn pairRank(self: *const Tokenizer, a: Allocator, key_buf: *std.ArrayList(u8), left: []const u8, right: []const u8) !?u32 {
        key_buf.clearRetainingCapacity();
        try key_buf.appendSlice(a, left);
        try key_buf.append(a, 0);
        try key_buf.appendSlice(a, right);
        return self.merge_ranks.get(key_buf.items);
    }

    /// Last resort for symbols that are not in the vocabulary: emit the
    /// individual byte tokens (always present in a byte-level BPE vocab).
    fn emitByteFallback(self: *const Tokenizer, a: Allocator, symbol: []const u8, out: *std.ArrayList(u32)) !void {
        var i: usize = 0;
        while (i < symbol.len) {
            const d = decodeChar(symbol[i..]);
            if (d.len == 0) break;
            const one = symbol[i .. i + d.len];
            const id = self.index.get(one) orelse return error.UnknownToken;
            try out.append(a, id);
            i += d.len;
        }
    }
};

const Symbol = struct { start: usize, len: usize };

/// Appends `token`'s text with the byte-level mapping reversed.
/// Append a SentencePiece piece in plain text form: U+2581 becomes a space and
/// `<0xXX>` becomes the raw byte.
pub fn appendDecodedSpm(allocator: Allocator, out: *std.ArrayList(u8), token: []const u8) !void {
    if (token.len == 6 and token[0] == '<' and token[1] == '0' and token[2] == 'x' and token[5] == '>') {
        const b = std.fmt.parseInt(u8, token[3..5], 16) catch {
            try out.appendSlice(allocator, token);
            return;
        };
        try out.append(allocator, b);
        return;
    }
    var i: usize = 0;
    while (i < token.len) {
        const len = std.unicode.utf8ByteSequenceLength(token[i]) catch 1;
        if (i + len <= token.len) {
            const cp = std.unicode.utf8Decode(token[i..][0..len]) catch 0;
            if (cp == 0x2581) {
                try out.append(allocator, ' ');
                i += len;
                continue;
            }
        }
        try out.append(allocator, token[i]);
        i += 1;
    }
}

fn appendDecoded(a: Allocator, out: *std.ArrayList(u8), token: []const u8) !void {
    var i: usize = 0;
    while (i < token.len) {
        const d = decodeChar(token[i..]);
        if (d.len == 0) break;
        if (byteForCodepoint(d.cp)) |_| {
            try out.append(a, byteForCodepoint(d.cp).?);
        } else {
            try out.appendSlice(a, token[i .. i + d.len]);
        }
        i += d.len;
    }
}

fn jsonId(value: std.json.Value) ?u32 {
    return switch (value) {
        .integer => |i| if (i >= 0 and i <= std.math.maxInt(u32)) @intCast(i) else null,
        .float => |f| if (f >= 0 and f <= std.math.maxInt(u32)) @intFromFloat(f) else null,
        else => null,
    };
}

// ===========================================================================
// Tests
// ===========================================================================

const testing = std.testing;

/// `<repo>/tests/fixtures/<rel>`, derived from this file's own path so tests do
/// not depend on the process working directory.
fn fixturePath(a: Allocator, rel: []const u8) ![]u8 {
    return std.fs.path.join(a, &.{ "tests/fixtures", rel });
}

/// The tiny gpt2 vocabulary generated by tools/make_fixtures.py: ids 0..255 are
/// the single-byte tokens (id == byte value), then the merged tokens, then the
/// two control tokens.
const merged_tokens = [_][]const u8{
    "he",            "hel",        "hell",        "hello",
    "\u{0120}w",     "\u{0120}wo", "\u{0120}wor", "\u{0120}worl",
    "\u{0120}world",
};
const merge_strings = [_][]const u8{
    "h e",            "he l",        "hel l",        "hell o",
    "\u{0120} w",     "\u{0120}w o", "\u{0120}wo r", "\u{0120}wor l",
    "\u{0120}worl d",
};
const bos_token = "<s>";
const eos_token = "</s>";
const eot_token = "<|endoftext|>";

/// Minimal GGUF v3 writer (metadata only) for tokenizer tests.
const KvBuilder = struct {
    a: Allocator,
    buf: std.ArrayList(u8) = .empty,

    fn int(self: *KvBuilder, comptime T: type, v: T) !void {
        var tmp: [@sizeOf(T)]u8 = undefined;
        std.mem.writeInt(T, &tmp, v, .little);
        try self.buf.appendSlice(self.a, &tmp);
    }

    fn str(self: *KvBuilder, s: []const u8) !void {
        try self.int(u64, s.len);
        try self.buf.appendSlice(self.a, s);
    }

    fn kvString(self: *KvBuilder, key: []const u8, v: []const u8) !void {
        try self.str(key);
        try self.int(u32, 8);
        try self.str(v);
    }

    fn kvU32(self: *KvBuilder, key: []const u8, v: u32) !void {
        try self.str(key);
        try self.int(u32, 4);
        try self.int(u32, v);
    }

    fn kvBool(self: *KvBuilder, key: []const u8, v: bool) !void {
        try self.str(key);
        try self.int(u32, 7);
        try self.int(u8, @intFromBool(v));
    }

    fn kvStringArray(self: *KvBuilder, key: []const u8, items: []const []const u8) !void {
        try self.str(key);
        try self.int(u32, 9);
        try self.int(u32, 8);
        try self.int(u64, items.len);
        for (items) |s| try self.str(s);
    }

    fn kvF32Array(self: *KvBuilder, key: []const u8, items: []const f32) !void {
        try self.str(key);
        try self.int(u32, 9); // array
        try self.int(u32, 6); // f32 elements
        try self.int(u64, items.len);
        for (items) |v| try self.int(u32, @bitCast(v));
    }

    fn kvI32Array(self: *KvBuilder, key: []const u8, items: []const i32) !void {
        try self.str(key);
        try self.int(u32, 9);
        try self.int(u32, 5);
        try self.int(u64, items.len);
        for (items) |v| try self.int(i32, v);
    }
};

/// Builds a tiny GGUF v3 containing only tokenizer metadata.
/// `add_bos_value == null` omits `tokenizer.ggml.add_bos_token` (so the
/// tokenizer's default applies).
fn buildTokenizerGguf(a: Allocator, add_bos_value: ?bool, with_eot: bool) ![]u8 {
    var tokens: std.ArrayList([]const u8) = .empty;
    defer {
        for (tokens.items) |t| a.free(t);
        tokens.deinit(a);
    }
    for (0..256) |b| {
        var cp_buf: [4]u8 = undefined;
        const n = try std.unicode.utf8Encode(byte_to_unicode[b], &cp_buf);
        try tokens.append(a, try a.dupe(u8, cp_buf[0..n]));
    }
    for (merged_tokens) |m| try tokens.append(a, try a.dupe(u8, m));
    try tokens.append(a, try a.dupe(u8, bos_token));
    try tokens.append(a, try a.dupe(u8, eos_token));
    if (with_eot) try tokens.append(a, try a.dupe(u8, eot_token));

    var types: std.ArrayList(i32) = .empty;
    defer types.deinit(a);
    for (0..tokens.items.len) |i| {
        const special = i >= 256 + merged_tokens.len;
        try types.append(a, if (special) 3 else 1);
    }

    const bos_id: u32 = @intCast(256 + merged_tokens.len);
    const eos_id: u32 = bos_id + 1;
    const kv_count: u32 = if (add_bos_value == null) 6 else 7;

    var b = KvBuilder{ .a = a };
    errdefer b.buf.deinit(a);
    try b.int(u32, 0x4655_4747); // GGUF
    try b.int(u32, 3);
    try b.int(u64, 0); // no tensors
    try b.int(u64, kv_count);
    try b.kvString("tokenizer.ggml.model", "gpt2");
    try b.kvStringArray("tokenizer.ggml.tokens", tokens.items);
    try b.kvStringArray("tokenizer.ggml.merges", &merge_strings);
    try b.kvI32Array("tokenizer.ggml.token_type", types.items);
    try b.kvU32("tokenizer.ggml.bos_token_id", bos_id);
    try b.kvU32("tokenizer.ggml.eos_token_id", eos_id);
    if (add_bos_value) |v| try b.kvBool("tokenizer.ggml.add_bos_token", v);
    // Real GGUF files pad up to `general.alignment` (default 32) even when the
    // tensor-data section is empty, so the reader can compute its offset.
    while (b.buf.items.len % 32 != 0) try b.buf.append(a, 0);
    return b.buf.toOwnedSlice(a);
}

fn expectPieces(text: []const u8, expected: []const []const u8) !void {
    const a = testing.allocator;
    const pieces = try pretokenize(a, text);
    defer a.free(pieces);
    try testing.expectEqual(expected.len, pieces.len);
    for (expected, pieces, 0..) |want, got, i| {
        testing.expectEqualStrings(want, got) catch |err| {
            std.debug.print("pre-token {d} of {s}: expected \"{s}\", got \"{s}\"\n", .{ i, text, want, got });
            return err;
        };
    }
}

test "byte-level: GPT-2 byte<->unicode mapping" {
    // Space maps to U+0120 (Ġ), the byte-level space marker.
    try testing.expectEqual(@as(u21, 0x0120), byte_to_unicode[32]);
    // The first unmapped byte (NUL) maps to U+0100.
    try testing.expectEqual(@as(u21, 0x0100), byte_to_unicode[0]);
    // Printable ASCII is the identity.
    try testing.expectEqual(@as(u21, 'A'), byte_to_unicode['A']);
    // 0xAD (soft hyphen) is the last unmapped byte: 256 + 67 = 323.
    try testing.expectEqual(@as(u21, 323), byte_to_unicode[0xAD]);
    // 0xA0 is the last of the 0x7F..0xA0 block: 256 + 66 = 322.
    try testing.expectEqual(@as(u21, 322), byte_to_unicode[0xA0]);
    try testing.expectEqual(@as(u21, 289), byte_to_unicode[0x7F]);

    // The mapping is a bijection over 0..=255 and round-trips through the
    // reverse table.
    var seen: [1024]bool = @splat(false);
    for (byte_to_unicode, 0..) |cp, b| {
        try testing.expect(cp < seen.len);
        try testing.expect(!seen[cp]);
        seen[cp] = true;
        try testing.expectEqual(@as(?u8, @intCast(b)), byteForCodepoint(cp));
    }
    // Codepoints outside the table are not byte-level markers.
    try testing.expect(byteForCodepoint(0x3042) == null); // あ
    try testing.expect(byteForCodepoint(0x0000) == null);
}

test "pre-tokenizer matches the GPT-2 regex on ASCII" {
    try expectPieces("hello world", &.{ "hello", " world" });
    try expectPieces("hello", &.{"hello"});
    try expectPieces("hello  world", &.{ "hello", " ", " world" });
    try expectPieces("  hi", &.{ " ", " hi" });
    try expectPieces(" leading", &.{" leading"});
    try expectPieces("trailing ", &.{ "trailing", " " });
    try expectPieces("  ", &.{"  "});
    try expectPieces("Hello", &.{"Hello"});
    try expectPieces("don't", &.{ "don", "'t" });
    try expectPieces("it's we're I'll", &.{ "it", "'s", " we", "'re", " I", "'ll" });
    try expectPieces("a1!?", &.{ "a", "1", "!?" });
    try expectPieces("hello, world!", &.{ "hello", ",", " world", "!" });
    try expectPieces("\n\nHello", &.{ "\n", "\n", "Hello" });
    try expectPieces("hello\nworld", &.{ "hello", "\n", "world" });
    try expectPieces("x", &.{"x"});

    // Non-ASCII letters stay in one run (documented approximation of \p{L}).
    try expectPieces("日本語", &.{"日本語"});
    try expectPieces("café", &.{"café"});
    try expectPieces("naïve café", &.{ "naïve", " café" });
}

test "encode/decode against the independent Python reference (tiny vocab)" {
    const a = testing.allocator;
    const bytes = try buildTokenizerGguf(a, null, true);
    defer a.free(bytes);
    var g = try gguf.Gguf.fromBytes(a, bytes);
    defer g.deinit();
    var tok = try Tokenizer.fromGguf(a, &g);
    defer tok.deinit();

    try testing.expectEqual(@as(usize, 268), tok.vocabSize());
    try testing.expectEqual(@as(?u32, 265), tok.bosId());
    try testing.expectEqual(@as(?u32, 266), tok.eosId());
    try testing.expectEqualStrings("hello", tok.tokenText(259));
    try testing.expectEqualStrings("\u{0120}", tok.tokenText(32));
    try testing.expectEqual(TokenType.control, tok.tokenType(265));
    try testing.expectEqual(TokenType.normal, tok.tokenType(259));
    try testing.expectEqualStrings("", tok.tokenText(9999));

    // Expected ids were produced by tools/reference_tokenizer.py, a separate
    // implementation of the published GPT-2 algorithm.
    const cases = [_]struct { text: []const u8, ids: []const u32 }{
        .{ .text = "hello world", .ids = &.{ 259, 264 } },
        .{ .text = "hello", .ids = &.{259} },
        .{ .text = "hello  world", .ids = &.{ 259, 32, 264 } },
        .{ .text = "  hi", .ids = &.{ 32, 32, 104, 105 } },
        .{ .text = "Hello", .ids = &.{ 72, 101, 108, 108, 111 } },
        .{ .text = "don't", .ids = &.{ 100, 111, 110, 39, 116 } },
        .{ .text = "a1!?", .ids = &.{ 97, 49, 33, 63 } },
        .{ .text = "it's we're I'll", .ids = &.{ 105, 116, 39, 115, 260, 101, 39, 114, 101, 32, 73, 39, 108, 108 } },
        .{ .text = "\n\nHello", .ids = &.{ 10, 10, 72, 101, 108, 108, 111 } },
        .{ .text = "hello\nworld", .ids = &.{ 259, 10, 119, 111, 114, 108, 100 } },
        .{ .text = "  ", .ids = &.{ 32, 32 } },
        .{ .text = "日本語", .ids = &.{ 230, 151, 165, 230, 156, 172, 232, 170, 158 } },
        .{ .text = "café", .ids = &.{ 99, 97, 102, 195, 169 } },
        .{ .text = "naïve café", .ids = &.{ 110, 97, 195, 175, 118, 101, 32, 99, 97, 102, 195, 169 } },
        .{ .text = "hello, world!", .ids = &.{ 259, 44, 264, 33 } },
        .{ .text = " leading", .ids = &.{ 32, 108, 101, 97, 100, 105, 110, 103 } },
    };
    for (cases) |c| {
        const ids = try tok.encode(a, c.text, false);
        defer a.free(ids);
        testing.expectEqualSlices(u32, c.ids, ids) catch |err| {
            std.debug.print("encode(\"{s}\") mismatch\n", .{c.text});
            return err;
        };
        const back = try tok.decode(a, ids);
        defer a.free(back);
        try testing.expectEqualStrings(c.text, back);
    }

    // add_special prepends BOS (the metadata key is absent here, default true).
    const with_bos = try tok.encode(a, "hello world", true);
    defer a.free(with_bos);
    try testing.expectEqualSlices(u32, &.{ 265, 259, 264 }, with_bos);

    // decode skips CONTROL tokens.
    const decoded_bos = try tok.decode(a, &.{ 265, 259, 264, 266 });
    defer a.free(decoded_bos);
    try testing.expectEqualStrings("hello world", decoded_bos);

    // Special tokens are matched literally on the raw text.
    const with_eot = try tok.encode(a, "a<|endoftext|>b", false);
    defer a.free(with_eot);
    try testing.expectEqualSlices(u32, &.{ 97, 267, 98 }, with_eot);
    const eot_back = try tok.decode(a, with_eot);
    defer a.free(eot_back);
    try testing.expectEqualStrings("ab", eot_back);

    // Empty input.
    const empty = try tok.encode(a, "", false);
    defer a.free(empty);
    try testing.expectEqual(@as(usize, 0), empty.len);
    const only_bos = try tok.encode(a, "", true);
    defer a.free(only_bos);
    try testing.expectEqualSlices(u32, &.{265}, only_bos);

    // Invalid ids and unknown symbols.
    try testing.expectError(error.InvalidTokenId, tok.decode(a, &.{9999}));
    try testing.expectEqual(@as(?u32, 259), tok.tokenId("hello"));
    try testing.expect(tok.tokenId("definitely not a token") == null);
}

test "fromGguf honors add_bos_token = false" {
    const a = testing.allocator;
    const bytes = try buildTokenizerGguf(a, false, false);
    defer a.free(bytes);
    var g = try gguf.Gguf.fromBytes(a, bytes);
    defer g.deinit();
    var tok = try Tokenizer.fromGguf(a, &g);
    defer tok.deinit();

    const ids = try tok.encode(a, "hello", true);
    defer a.free(ids);
    try testing.expectEqualSlices(u32, &.{259}, ids);
}

test "fromGguf accepts a llama (SentencePiece) vocabulary" {
    const a = testing.allocator;
    var b = KvBuilder{ .a = a };
    defer b.buf.deinit(a);
    try b.int(u32, 0x4655_4747);
    try b.int(u32, 3);
    try b.int(u64, 0);
    try b.int(u64, 3);
    try b.kvString("tokenizer.ggml.model", "llama");
    try b.kvStringArray("tokenizer.ggml.tokens", &.{ "a", "\u{2581}b" });
    try b.kvF32Array("tokenizer.ggml.scores", &.{ -1.0, -2.0 });
    while (b.buf.items.len % 32 != 0) try b.buf.append(a, 0);
    var g = try gguf.Gguf.fromBytes(a, b.buf.items);
    defer g.deinit();
    var tok = try Tokenizer.fromGguf(a, &g);
    defer tok.deinit();
    try testing.expect(tok.sentencepiece);
    try testing.expectEqual(@as(usize, 2), tok.vocabSize());
}

test "fromGguf rejects a vocabulary type it does not implement" {
    const a = testing.allocator;
    var b = KvBuilder{ .a = a };
    defer b.buf.deinit(a);
    try b.int(u32, 0x4655_4747);
    try b.int(u32, 3);
    try b.int(u64, 0);
    try b.int(u64, 2);
    try b.kvString("tokenizer.ggml.model", "wordpiece");
    try b.kvStringArray("tokenizer.ggml.tokens", &.{"a"});
    while (b.buf.items.len % 32 != 0) try b.buf.append(a, 0);
    var g = try gguf.Gguf.fromBytes(a, b.buf.items);
    defer g.deinit();
    try testing.expectError(error.UnsupportedTokenizerModel, Tokenizer.fromGguf(a, &g));

    var b2 = KvBuilder{ .a = a };
    defer b2.buf.deinit(a);
    try b2.int(u32, 0x4655_4747);
    try b2.int(u32, 3);
    try b2.int(u64, 0);
    try b2.int(u64, 1);
    try b2.kvString("tokenizer.ggml.model", "gpt2");
    while (b2.buf.items.len % 32 != 0) try b2.buf.append(a, 0);
    var g2 = try gguf.Gguf.fromBytes(a, b2.buf.items);
    defer g2.deinit();
    try testing.expectError(error.MissingTokens, Tokenizer.fromGguf(a, &g2));
}

test "fromTokenizerJson loads HF fixtures (string and array merges)" {
    const a = testing.allocator;

    inline for (.{ "tokenizer/tiny_tokenizer.json", "tokenizer/tiny_tokenizer_arrays.json" }) |rel| {
        const path = try fixturePath(a, rel);
        defer a.free(path);
        var tok = try Tokenizer.fromTokenizerJson(a, path);
        defer tok.deinit();

        try testing.expectEqual(@as(usize, 267), tok.vocabSize());
        try testing.expectEqual(@as(?u32, null), tok.bosId());
        try testing.expectEqual(@as(?u32, null), tok.eosId());

        const ids = try tok.encode(a, "hello world", false);
        defer a.free(ids);
        try testing.expectEqualSlices(u32, &.{ 259, 264 }, ids);
        const back = try tok.decode(a, ids);
        defer a.free(back);
        try testing.expectEqualStrings("hello world", back);

        // added_tokens are CONTROL tokens: decode skips them, encode matches.
        const with_special = try tok.encode(a, "<s>hello", false);
        defer a.free(with_special);
        try testing.expectEqualSlices(u32, &.{ 265, 259 }, with_special);
        const special_back = try tok.decode(a, with_special);
        defer a.free(special_back);
        try testing.expectEqualStrings("hello", special_back);

        const jap = try tok.encode(a, "日本語", false);
        defer a.free(jap);
        const jap_back = try tok.decode(a, jap);
        defer a.free(jap_back);
        try testing.expectEqualStrings("日本語", jap_back);
    }
}

test "spm decode reverses the separator and byte pieces" {
    const a = testing.allocator;
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(a);
    try appendDecodedSpm(a, &out, "\u{2581}Paris");
    try std.testing.expectEqualStrings(" Paris", out.items);
    out.clearRetainingCapacity();
    try appendDecodedSpm(a, &out, "<0x0A>");
    try std.testing.expectEqualStrings("\n", out.items);
    out.clearRetainingCapacity();
    // A plain piece with no separator passes through.
    try appendDecodedSpm(a, &out, ".");
    try std.testing.expectEqualStrings(".", out.items);
    out.clearRetainingCapacity();
    // A multi-byte character survives the byte walk.
    try appendDecodedSpm(a, &out, "\u{2581}\u{65e5}\u{672c}");
    try std.testing.expectEqualStrings(" \u{65e5}\u{672c}", out.items);
}

test "spm encode round-trips through decode" {
    const a = testing.allocator;
    // A miniature SentencePiece-style vocabulary built as a GGUF, so the pieces
    // can contain a literal U+2581 without JSON escaping games.
    var b = KvBuilder{ .a = a };
    defer b.buf.deinit(a);
    try b.int(u32, 0x4655_4747);
    try b.int(u32, 3);
    try b.int(u64, 0);
    try b.int(u64, 3);
    try b.kvString("tokenizer.ggml.model", "llama");
    try b.kvStringArray("tokenizer.ggml.tokens", &.{
        "\u{2581}the", "\u{2581}capital", "\u{2581}of", "\u{2581}France", "\u{2581}is", "\u{2581}Paris", ".",
    });
    try b.kvF32Array("tokenizer.ggml.scores", &.{ -1, -2, -3, -4, -5, -6, -7 });
    while (b.buf.items.len % 32 != 0) try b.buf.append(a, 0);
    var g = try gguf.Gguf.fromBytes(a, b.buf.items);
    defer g.deinit();
    var tok = try Tokenizer.fromGguf(a, &g);
    defer tok.deinit();
    try testing.expect(tok.sentencepiece);

    const ids = try tok.encode(a, "the capital of France is Paris.", false);
    defer a.free(ids);
    try testing.expectEqual(@as(usize, 7), ids.len);
    const back = try tok.decode(a, ids);
    defer a.free(back);
    try testing.expectEqualStrings(" the capital of France is Paris.", back);
}

test "fromTokenizerJsonSlice rejects non-BPE models" {
    const a = testing.allocator;
    const json =
        \\{"model":{"type":"WordPiece","vocab":{"a":0},"merges":[]}}
    ;
    try testing.expectError(error.UnsupportedTokenizerModel, Tokenizer.fromTokenizerJsonSlice(a, json));
    const broken =
        \\{"model":{"type":"BPE"}}
    ;
    try testing.expectError(error.InvalidTokenizerJson, Tokenizer.fromTokenizerJsonSlice(a, broken));
}

test "fromTokenizerJson round-trips the GGUF fixture vocabulary" {
    const a = testing.allocator;
    const path = try fixturePath(a, "gguf/tiny_v3.gguf");
    defer a.free(path);
    var g = try gguf.Gguf.loadWithIo(a, testing.io, path);
    defer g.deinit();
    var tok = try Tokenizer.fromGguf(a, &g);
    defer tok.deinit();

    try testing.expectEqual(@as(usize, 267), tok.vocabSize());
    try testing.expectEqual(@as(?u32, 265), tok.bosId());
    try testing.expectEqual(@as(?u32, 266), tok.eosId());
    // add_bos_token is explicitly true in the fixture.
    const ids = try tok.encode(a, "hello world", true);
    defer a.free(ids);
    try testing.expectEqualSlices(u32, &.{ 265, 259, 264 }, ids);
    const back = try tok.decode(a, ids);
    defer a.free(back);
    try testing.expectEqualStrings("hello world", back);
}

test "byte-level encode/decode round-trips arbitrary bytes" {
    const a = testing.allocator;
    const bytes = try buildTokenizerGguf(a, null, false);
    defer a.free(bytes);
    var g = try gguf.Gguf.fromBytes(a, bytes);
    defer g.deinit();
    var tok = try Tokenizer.fromGguf(a, &g);
    defer tok.deinit();

    // Including invalid UTF-8 and control bytes: a byte-level BPE tokenizer
    // must reproduce the input byte for byte.
    const cases = [_][]const u8{
        "\x00\x01\x02",
        "\x80\x81\xfe\xff",
        "\xc3\x28", // invalid UTF-8 sequence
        "tab\there\n",
        "trailing\x7f",
        "null\x00inside",
    };
    for (cases) |text| {
        const ids = try tok.encode(a, text, false);
        defer a.free(ids);
        const back = try tok.decode(a, ids);
        defer a.free(back);
        testing.expectEqualSlices(u8, text, back) catch |err| {
            std.debug.print("round trip failed for {d} bytes\n", .{text.len});
            return err;
        };
    }

    // The byte tokens are the identity mapping for printable ASCII and the
    // mapped codepoint for everything else.
    try testing.expectEqual(@as(?u32, 32), tok.tokenId("\u{0120}"));
    try testing.expectEqualStrings("\u{00ff}", tok.tokenText(0xFF));
    try testing.expectEqual(@as(?u32, 0xFF), tok.tokenId("\u{00ff}"));
}
