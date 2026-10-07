#!/usr/bin/env python3
"""Generate every fixture used by src/gguf.zig's and src/tokenizer.zig's tests.

Outputs (all deterministic; re-run to regenerate):
  tests/fixtures/dequant/<type>.bin    raw quantized blocks (deterministic PRNG)
  tests/fixtures/dequant/<type>.f32    expected binary32 output, from ggml_reference.py
  tests/fixtures/gguf/tiny_v3.gguf     a hand-written GGUF v3 file (f32/f16/q4_0/q8_0
                                       tensors, all 13 metadata value types, and a
                                       tiny gpt2 tokenizer for src/tokenizer.zig)
  tests/fixtures/tokenizer/tiny_tokenizer.json         HF tokenizer.json, string merges
  tests/fixtures/tokenizer/tiny_tokenizer_arrays.json  HF tokenizer.json, [a, b] merges

The .f32 expectations are additionally verified against llama.cpp's compiled
dequantize_row_* (tools/crosscheck_dequant.sh).
"""

import json
import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ggml_reference as ref  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --------------------------------------------------------------- PRNG

class Rng:
    """Deterministic xorshift64* so fixtures never depend on Python's random."""

    def __init__(self, seed):
        self.s = seed & 0xFFFFFFFFFFFFFFFF or 0x9E3779B97F4A7C15

    def next_u64(self):
        x = self.s
        x ^= (x >> 12)
        x ^= (x << 25) & 0xFFFFFFFFFFFFFFFF
        x ^= (x >> 27)
        self.s = x & 0xFFFFFFFFFFFFFFFF
        return (x * 0x2545F4914F6CDD1D) & 0xFFFFFFFFFFFFFFFF

    def byte(self):
        return self.next_u64() & 0xFF

    def bytes(self, n):
        return bytes(self.byte() for _ in range(n))


def safe_half(rng):
    """A finite, normal binary16: |v| in [2^-6, 1), no NaN/Inf/subnormal."""
    mant = rng.next_u64() % 1024                   # 10 fraction bits
    exp = 9 + (rng.next_u64() % 6)                 # unbiased 2^-6 .. 2^-1
    bits = (exp << 10) | mant
    v = struct.unpack("<e", struct.pack("<H", bits))[0]
    assert 0.015 <= abs(v) < 1.0, v
    return struct.pack("<H", bits)


def safe_f32(rng):
    mant = rng.next_u64() % (1 << 20)
    exp = 118 + (rng.next_u64() % 8)               # 2^-9 .. 2^-2
    bits = (rng.next_u64() & 0x80000000) | (exp << 23) | mant
    return struct.pack("<I", bits)


# scale-field offsets per type: name -> [(offset, kind)]
SCALE_FIELDS = {
    2: [(0, "h")],                    # q4_0: d
    3: [(0, "h"), (2, "h")],          # q4_1: d, m
    6: [(0, "h")],                    # q5_0: d
    7: [(0, "h"), (2, "h")],          # q5_1: d, m
    8: [(0, "h")],                    # q8_0: d
    9: [(0, "h"), (2, "h")],          # q8_1: d, s
    10: [(80, "h"), (82, "h")],       # q2_K: d, dmin
    11: [(108, "h")],                 # q3_K: d
    12: [(0, "h"), (2, "h")],         # q4_K: d, dmin
    13: [(0, "h"), (2, "h")],         # q5_K: d, dmin
    14: [(208, "h")],                 # q6_K: d
    15: [(0, "f")],                   # q8_K: d
}


def make_block(ttype, rng):
    epb, bb = ref.LAYOUT[ttype]
    buf = bytearray(rng.bytes(bb))
    for off, kind in SCALE_FIELDS.get(ttype, []):
        val = safe_half(rng) if kind == "h" else safe_f32(rng)
        buf[off:off + len(val)] = val
    return bytes(buf)


def gen_dequant_fixtures():
    out_dir = os.path.join(ROOT, "tests", "fixtures", "dequant")
    os.makedirs(out_dir, exist_ok=True)
    for ttype, name in sorted(ref.TYPE_NAMES.items()):
        epb, bb = ref.LAYOUT[ttype]
        nblocks = 2 if epb >= 32 else 16
        rng = Rng(0x5EED_0000 + ttype)
        data = b"".join(make_block(ttype, rng) for _ in range(nblocks))
        if ttype == 0:      # f32: keep values finite and exactly representable
            data = b"".join(safe_f32(rng) for _ in range(16))
        if ttype == 1:      # f16
            data = b"".join(safe_half(rng) for _ in range(16))
        if ttype == 30:     # bf16
            data = b"".join(struct.pack("<I", struct.unpack("<I", safe_f32(rng))[0])[0:2]
                            for _ in range(16))
        expected = ref.dequantize(ttype, data)
        with open(os.path.join(out_dir, name + ".bin"), "wb") as f:
            f.write(data)
        with open(os.path.join(out_dir, name + ".f32"), "wb") as f:
            f.write(b"".join(struct.pack("<f", v) for v in expected))
        print("  dequant %-5s %4d blocks, %5d elements" % (name, len(data) // bb, len(expected)))


# ------------------------------------------------------------ GGUF writer

GGUF_MAGIC = 0x46554747  # "GGUF" little-endian
T_U8, T_I8, T_U16, T_I16, T_U32, T_I32, T_F32, T_BOOL, T_STRING, T_ARRAY, T_U64, T_I64, T_F64 = range(13)

FMT = {T_U8: "<B", T_I8: "<b", T_U16: "<H", T_I16: "<h", T_U32: "<I", T_I32: "<i",
       T_F32: "<f", T_BOOL: "<B", T_U64: "<Q", T_I64: "<q", T_F64: "<d"}


def gguf_string(s):
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def gguf_value(vtype, v):
    if vtype == T_STRING:
        return struct.pack("<I", T_STRING) + gguf_string(v)
    if vtype == T_ARRAY:
        etype, items = v
        out = struct.pack("<I", T_ARRAY) + struct.pack("<I", etype) + struct.pack("<Q", len(items))
        for it in items:
            if etype == T_STRING:
                out += gguf_string(it)
            else:
                out += struct.pack(FMT[etype], it)
        return out
    return struct.pack("<I", vtype) + struct.pack(FMT[vtype], v)


def write_gguf(path, kvs, tensors, alignment=32):
    """kvs: [(key, vtype, value)], tensors: [(name, dims, ttype, data)]"""
    body = struct.pack("<IIQQ", GGUF_MAGIC, 3, len(tensors), len(kvs))
    for key, vtype, value in kvs:
        body += gguf_string(key) + gguf_value(vtype, value)

    data = bytearray()
    infos = bytearray()
    for name, dims, ttype, tdata in tensors:
        while len(data) % alignment != 0:
            data.append(0)
        offset = len(data)
        data += tdata
        infos += gguf_string(name)
        infos += struct.pack("<I", len(dims))
        for d in dims:
            infos += struct.pack("<Q", d)
        infos += struct.pack("<I", ttype) + struct.pack("<Q", offset)

    out = body + bytes(infos)
    while len(out) % alignment != 0:
        out += b"\x00"
    out += bytes(data)
    with open(path, "wb") as f:
        f.write(out)
    return len(out)


# --------------------------------------------------- tiny byte-level BPE

def byte_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + \
        list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: c for b, c in zip(bs, cs)}


B2U = byte_to_unicode()

MERGES = ["h e", "he l", "hel l", "hell o", "Ġ w", "Ġw o", "Ġwo r", "Ġwor l", "Ġworl d"]


def build_vocab():
    tokens = [chr(B2U[b]) for b in range(256)]
    for m in MERGES:
        tokens.append("".join(m.split(" ")))
    tokens.append("<s>")
    tokens.append("</s>")
    return tokens


def gen_tokenizer_fixtures():
    tokens = build_vocab()
    vocab = {t: i for i, t in enumerate(tokens)}
    bos, eos = vocab["<s>"], vocab["</s>"]

    out_dir = os.path.join(ROOT, "tests", "fixtures", "tokenizer")
    os.makedirs(out_dir, exist_ok=True)

    def tokenizer_json(merge_style):
        merges = MERGES if merge_style == "str" else [m.split(" ", 1) for m in MERGES]
        return {
            "version": "1.0",
            "truncation": None,
            "padding": None,
            "added_tokens": [
                {"id": bos, "content": "<s>", "single_word": False, "lstrip": False,
                 "rstrip": False, "normalized": False, "special": True},
                {"id": eos, "content": "</s>", "single_word": False, "lstrip": False,
                 "rstrip": False, "normalized": False, "special": True},
            ],
            "normalizer": None,
            "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": False,
                              "trim_offsets": True, "use_regex": True},
            "post_processor": None,
            "decoder": {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True,
                        "use_regex": True},
            "model": {
                "type": "BPE",
                "dropout": None,
                "unk_token": None,
                "continuing_subword_prefix": None,
                "end_of_word_suffix": None,
                "fuse_unk": False,
                "byte_fallback": False,
                "vocab": dict(sorted(vocab.items())),
                "merges": merges,
            },
        }

    with open(os.path.join(out_dir, "tiny_tokenizer.json"), "w", encoding="utf-8") as f:
        json.dump(tokenizer_json("str"), f, ensure_ascii=False, indent=1)
        f.write("\n")
    with open(os.path.join(out_dir, "tiny_tokenizer_arrays.json"), "w", encoding="utf-8") as f:
        json.dump(tokenizer_json("arr"), f, ensure_ascii=False, indent=1)
        f.write("\n")
    print("  tokenizer.json (%d tokens, %d merges, bos=%d eos=%d)" % (len(tokens), len(MERGES), bos, eos))
    return tokens, vocab, bos, eos


def gen_gguf_fixture(tokens, vocab, bos, eos):
    out_dir = os.path.join(ROOT, "tests", "fixtures", "gguf")
    os.makedirs(out_dir, exist_ok=True)

    # w.f32: [1.5, -2.0, 3.25, 0.5]
    f32_data = struct.pack("<4f", 1.5, -2.0, 3.25, 0.5)
    # w.f16: dims [2,3], dims[0] fastest varying
    f16_vals = [1.0, 2.5, -0.75, 4.0, -1.5, 0.25]
    f16_data = b"".join(struct.pack("<e", v) for v in f16_vals)
    # w.q4_0: [32], d=0.5, qs[0]=0x21, rest zero -> x[0]=(1-8)*0.5=-3.5, x[16]=(2-8)*0.5=-3.0
    q4_data = struct.pack("<e", 0.5) + bytes([0x21] + [0] * 15)
    # w.q8_0: [32], d=0.25, qs = -128..-97 (i.e. bytes 0x80..0x9F)
    q8_data = struct.pack("<e", 0.25) + bytes(range(0x80, 0xA0))

    kvs = [
        ("general.architecture", T_STRING, "llama"),
        ("general.name", T_STRING, "tiny-fixture"),
        ("general.file_type", T_U32, 0),
        ("general.alignment", T_U32, 32),
        ("llama.block_count", T_U32, 1),
        ("llama.attention.head_count", T_U32, 2),
        ("llama.rope.freq_base", T_F32, 10000.0),
        ("fixture.u8", T_U8, 200),
        ("fixture.i8", T_I8, -100),
        ("fixture.u16", T_U16, 60000),
        ("fixture.i16", T_I16, -30000),
        ("fixture.i32", T_I32, -7),
        ("fixture.u64", T_U64, 1234567890123),
        ("fixture.i64", T_I64, -1234567890123),
        ("fixture.f64", T_F64, 0.5),
        ("fixture.bool", T_BOOL, 1),
        ("fixture.u32_array", T_ARRAY, (T_U32, [1, 2, 3])),
        ("fixture.string_array", T_ARRAY, (T_STRING, ["a", "bb", "ccc"])),
        ("fixture.empty_array", T_ARRAY, (T_U32, [])),
        ("tokenizer.ggml.model", T_STRING, "gpt2"),
        ("tokenizer.ggml.tokens", T_ARRAY, (T_STRING, tokens)),
        ("tokenizer.ggml.merges", T_ARRAY, (T_STRING, MERGES)),
        ("tokenizer.ggml.token_type",
         T_ARRAY, (T_I32, [3 if i >= len(tokens) - 2 else 1 for i in range(len(tokens))])),
        ("tokenizer.ggml.bos_token_id", T_U32, bos),
        ("tokenizer.ggml.eos_token_id", T_U32, eos),
        ("tokenizer.ggml.add_bos_token", T_BOOL, 1),
    ]

    tensors = [
        ("w.f32", [4], 0, f32_data),
        ("w.f16", [2, 3], 1, f16_data),
        ("w.q4_0", [32], 2, q4_data),
        ("w.q8_0", [32], 8, q8_data),
    ]
    path = os.path.join(out_dir, "tiny_v3.gguf")
    size = write_gguf(path, kvs, tensors)
    print("  gguf tiny_v3.gguf (%d bytes, %d tensors, %d metadata keys)" % (size, len(tensors), len(kvs)))


def main():
    print("writing fixtures under %s" % os.path.relpath(os.path.join(ROOT, "tests", "fixtures"), ROOT))
    gen_dequant_fixtures()
    tokens, vocab, bos, eos = gen_tokenizer_fixtures()
    gen_gguf_fixture(tokens, vocab, bos, eos)
    print("done")


if __name__ == "__main__":
    main()
