#!/usr/bin/env python3
"""Bit-accurate reference dequantizers for ggml block formats.

Every function below is a line-by-line port of the corresponding
``dequantize_row_*`` function from llama.cpp's ``ggml/src/ggml-quants.c``
(retrieved 2026-10-08), together with the matching block layouts from
``ggml/src/ggml-common.h``.  ``GGML_FP16_TO_FP32`` is emulated with
``struct``'s ``e`` format (exact IEEE-754 binary16 -> binary32), and every
arithmetic operation is rounded to binary32 exactly like the C code, so the
outputs are bit-identical to the reference implementation (verified against
the C source compiled by clang via ``tools/crosscheck_dequant.sh``).

Used only to generate and verify the committed fixtures under
``tests/fixtures/dequant/``.
"""

import struct

QK_K = 256
K_SCALE_SIZE = 12

# ggml type id -> (elements per block, bytes per block); ids from
# `enum ggml_type` in ggml.h.
LAYOUT = {
    0: (1, 4),       # f32
    1: (1, 2),       # f16
    2: (32, 18),     # q4_0
    3: (32, 20),     # q4_1
    6: (32, 22),     # q5_0
    7: (32, 24),     # q5_1
    8: (32, 34),     # q8_0
    9: (32, 36),     # q8_1
    10: (256, 84),   # q2_K
    11: (256, 110),  # q3_K
    12: (256, 144),  # q4_K
    13: (256, 176),  # q5_K
    14: (256, 210),  # q6_K
    15: (256, 292),  # q8_K
    30: (1, 2),      # bf16
}

TYPE_NAMES = {
    0: "f32", 1: "f16", 2: "q4_0", 3: "q4_1", 6: "q5_0", 7: "q5_1",
    8: "q8_0", 9: "q8_1", 10: "q2_k", 11: "q3_k", 12: "q4_k", 13: "q5_k",
    14: "q6_k", 15: "q8_k", 30: "bf16",
}


def f32(x):
    """Round a Python float to the nearest binary32 (like a C float op)."""
    return struct.unpack("<f", struct.pack("<f", x))[0]


def f16(h):
    """binary16 (as a raw u16) -> Python float."""
    return struct.unpack("<e", struct.pack("<H", h))[0]


def u16(b, off):
    return struct.unpack_from("<H", b, off)[0]


def u32(b, off):
    return struct.unpack_from("<I", b, off)[0]


def i8(v):
    return v - 256 if v >= 128 else v


def i16(b, off):
    return struct.unpack_from("<h", b, off)[0]


# ---------------------------------------------------------------- raw types

def dequant_f32(b, n):
    return [struct.unpack_from("<f", b, 4 * i)[0] for i in range(n)]


def dequant_f16(b, n):
    return [f16(u16(b, 2 * i)) for i in range(n)]


def dequant_bf16(b, n):
    return [struct.unpack("<f", struct.pack("<I", u16(b, 2 * i) << 16))[0] for i in range(n)]


# ------------------------------------------------------------ legacy quants

def _nibble(block_bytes, has_min, minus, has_high_bit):
    """Closure implementing dequantize_row_q4_0/q4_1/q5_0/q5_1."""

    def run(b, n):
        qk = 32
        out = []
        for i in range(n // qk):
            base = i * block_bytes
            d = f16(u16(b, base))
            m = f16(u16(b, base + 2)) if has_min else 0.0
            if has_high_bit:
                qh_off = base + (4 if has_min else 2)
                qs_off = qh_off + 4
                qh = u32(b, qh_off)
            else:
                qs_off = base + (4 if has_min else 2)
                qh = 0
            lo = [0.0] * (qk // 2)
            hi = [0.0] * (qk // 2)
            for j in range(qk // 2):
                byte = b[qs_off + j]
                if has_high_bit:
                    xh_0 = ((qh >> (j + 0)) << 4) & 0x10
                    xh_1 = ((qh >> (j + 12))) & 0x10
                    x0 = ((byte & 0x0F) | xh_0) - minus
                    x1 = ((byte >> 4) | xh_1) - minus
                else:
                    x0 = (byte & 0x0F) - minus
                    x1 = (byte >> 4) - minus
                lo[j] = f32(f32(x0 * d) + m)
                hi[j] = f32(f32(x1 * d) + m)
            out.extend(lo)
            out.extend(hi)
        return out

    return run


dequant_q4_0 = _nibble(18, has_min=False, minus=8, has_high_bit=False)
dequant_q4_1 = _nibble(20, has_min=True, minus=0, has_high_bit=False)
dequant_q5_0 = _nibble(22, has_min=False, minus=16, has_high_bit=True)
dequant_q5_1 = _nibble(24, has_min=True, minus=0, has_high_bit=True)


def dequant_q8_0(b, n):
    qk = 32
    out = []
    for i in range(n // qk):
        base = i * 34
        d = f16(u16(b, base))
        for j in range(qk):
            out.append(f32(i8(b[base + 2 + j]) * d))
    return out


def dequant_q8_1(b, n):
    # NB: `s` is d*sum(qs), not a scale; dequantization only uses d.
    qk = 32
    out = []
    for i in range(n // qk):
        base = i * 36
        d = f16(u16(b, base))
        for j in range(qk):
            out.append(f32(i8(b[base + 4 + j]) * d))
    return out


# --------------------------------------------------------------- k-quants

def get_scale_min_k4(j, q, base):
    if j < 4:
        d = q[base + j] & 63
        m = q[base + j + 4] & 63
    else:
        d = (q[base + j + 4] & 0xF) | ((q[base + j - 4] >> 6) << 4)
        m = (q[base + j + 4] >> 4) | ((q[base + j - 0] >> 6) << 4)
    return d, m


def dequant_q2_k(b, n):
    out = []
    for i in range(n // QK_K):
        base = i * 84
        d = f16(u16(b, base + 80))
        mn = f16(u16(b, base + 82))
        scales = base
        q = base + 16
        is_ = 0
        for _ in range(0, QK_K, 128):
            shift = 0
            for _ in range(4):
                sc = b[scales + is_]; is_ += 1
                dl = f32(d * (sc & 0xF)); ml = f32(mn * (sc >> 4))
                for l in range(16):
                    out.append(f32(f32(dl * ((b[q + l] >> shift) & 3)) - ml))
                sc = b[scales + is_]; is_ += 1
                dl = f32(d * (sc & 0xF)); ml = f32(mn * (sc >> 4))
                for l in range(16):
                    out.append(f32(f32(dl * ((b[q + l + 16] >> shift) & 3)) - ml))
                shift += 2
            q += 32
    return out


def dequant_q3_k(b, n):
    kmask1 = 0x03030303
    kmask2 = 0x0F0F0F0F
    out = []
    for i in range(n // QK_K):
        base = i * 110
        d_all = f16(u16(b, base + 108))
        hm = base        # uint8_t hmask[QK_K/8]
        q = base + 32    # uint8_t qs[QK_K/4]
        sc_off = base + 96
        aux = [u32(b, sc_off), u32(b, sc_off + 4), u32(b, sc_off + 8), 0]
        tmp = aux[2]
        aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4)
        aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4)
        aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4)
        aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4)
        scales = [(aux[k // 4] >> (8 * (k % 4))) & 0xFF for k in range(16)]
        scales = [i8(s) for s in scales]
        is_ = 0
        m = 1
        qq = q
        for _ in range(0, QK_K, 128):
            shift = 0
            for _ in range(4):
                dl = f32(d_all * (scales[is_] - 32)); is_ += 1
                for l in range(16):
                    v = ((b[qq + l] >> shift) & 3) - (0 if (b[hm + l] & m) else 4)
                    out.append(f32(dl * v))
                dl = f32(d_all * (scales[is_] - 32)); is_ += 1
                for l in range(16):
                    v = ((b[qq + l + 16] >> shift) & 3) - (0 if (b[hm + l + 16] & m) else 4)
                    out.append(f32(dl * v))
                shift += 2
                m <<= 1
            qq += 32
    return out


def dequant_q4_k(b, n):
    out = []
    for i in range(n // QK_K):
        base = i * 144
        d = f16(u16(b, base))
        mn = f16(u16(b, base + 2))
        scales = base + 4
        q = base + 16
        is_ = 0
        for _ in range(0, QK_K, 64):
            sc, m = get_scale_min_k4(is_ + 0, b, scales)
            d1 = f32(d * sc); m1 = f32(mn * m)
            sc, m = get_scale_min_k4(is_ + 1, b, scales)
            d2 = f32(d * sc); m2 = f32(mn * m)
            for l in range(32):
                out.append(f32(f32(d1 * (b[q + l] & 0xF)) - m1))
            for l in range(32):
                out.append(f32(f32(d2 * (b[q + l] >> 4)) - m2))
            q += 32
            is_ += 2
    return out


def dequant_q5_k(b, n):
    out = []
    for i in range(n // QK_K):
        base = i * 176
        d = f16(u16(b, base))
        mn = f16(u16(b, base + 2))
        scales = base + 4
        qh = base + 16
        ql = base + 48
        is_ = 0
        u1, u2 = 1, 2
        for _ in range(0, QK_K, 64):
            sc, m = get_scale_min_k4(is_ + 0, b, scales)
            d1 = f32(d * sc); m1 = f32(mn * m)
            sc, m = get_scale_min_k4(is_ + 1, b, scales)
            d2 = f32(d * sc); m2 = f32(mn * m)
            for l in range(32):
                v = (b[ql + l] & 0xF) + (16 if (b[qh + l] & u1) else 0)
                out.append(f32(f32(d1 * v) - m1))
            for l in range(32):
                v = (b[ql + l] >> 4) + (16 if (b[qh + l] & u2) else 0)
                out.append(f32(f32(d2 * v) - m2))
            ql += 32
            is_ += 2
            u1 <<= 2
            u2 <<= 2
    return out


def dequant_q6_k(b, n):
    out = []
    for i in range(n // QK_K):
        base = i * 210
        d = f16(u16(b, base + 208))
        for nb in range(0, QK_K, 128):
            ql = base + (nb // 128) * 64
            qh = base + 128 + (nb // 128) * 32
            sc = base + 192 + (nb // 128) * 8
            y = [0.0] * 128
            for l in range(32):
                is_ = l // 16
                q1 = ((b[ql + l] & 0xF) | (((b[qh + l] >> 0) & 3) << 4)) - 32
                q2 = ((b[ql + l + 32] & 0xF) | (((b[qh + l] >> 2) & 3) << 4)) - 32
                q3 = ((b[ql + l] >> 4) | (((b[qh + l] >> 4) & 3) << 4)) - 32
                q4 = ((b[ql + l + 32] >> 4) | (((b[qh + l] >> 6) & 3) << 4)) - 32
                y[l + 0] = f32(f32(d * i8(b[sc + is_ + 0])) * q1)
                y[l + 32] = f32(f32(d * i8(b[sc + is_ + 2])) * q2)
                y[l + 64] = f32(f32(d * i8(b[sc + is_ + 4])) * q3)
                y[l + 96] = f32(f32(d * i8(b[sc + is_ + 6])) * q4)
            out.extend(y)
    return out


def dequant_q8_k(b, n):
    out = []
    for i in range(n // QK_K):
        base = i * 292
        d = struct.unpack_from("<f", b, base)[0]
        for j in range(QK_K):
            out.append(f32(i8(b[base + 4 + j]) * d))
    return out


DEQUANT = {
    0: dequant_f32,
    1: dequant_f16,
    2: dequant_q4_0,
    3: dequant_q4_1,
    6: dequant_q5_0,
    7: dequant_q5_1,
    8: dequant_q8_0,
    9: dequant_q8_1,
    10: dequant_q2_k,
    11: dequant_q3_k,
    12: dequant_q4_k,
    13: dequant_q5_k,
    14: dequant_q6_k,
    15: dequant_q8_k,
    30: dequant_bf16,
}


def dequantize(ttype, data, n_elems=None):
    """Dequantize `data` (a whole number of blocks) into a list of binary32."""
    elems_per_block, block_bytes = LAYOUT[ttype]
    if n_elems is None:
        n_elems = (len(data) // block_bytes) * elems_per_block
    return DEQUANT[ttype](data, n_elems)


if __name__ == "__main__":
    # Hand-checked smoke tests (repeated as explicit literals in src/gguf.zig).
    d = struct.pack("<H", struct.unpack("<H", struct.pack("<e", 0.5))[0])
    blk = d + bytes([0x21] + [0] * 15)
    got = dequantize(2, blk)
    assert got[0] == -3.5 and got[16] == -3.0, (got[0], got[16])

    blk = d + d + bytes([0x21] + [0] * 15)  # q4_1: d=0.5, m=0.5, nibbles
    got = dequantize(3, blk)
    assert got[0] == 1.0 and got[16] == 1.5, (got[0], got[16])

    print("ggml_reference self-check ok")
