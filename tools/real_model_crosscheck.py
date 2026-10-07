#!/usr/bin/env python3
"""Independent safetensors reader used to cross-check src/safetensors.zig.

Parses a safetensors file using only `json` + `struct` (no ML libraries) and
prints one deterministic line per tensor: name, dtype, shape, element count, the
exact bit pattern of the first element, and the f64 sum of all elements.

Diff its output against tools/real_model_check.sh output to verify the Zig
reader bit-for-bit:

    python3 tools/real_model_crosscheck.py <dir>/model.safetensors > py.txt
    tools/real_model_check.sh <dir> | grep -E '^tensor|^lookup|^tensors' > zig.txt
    diff py.txt zig.txt

Supports F32, F16 and BF16 (the dtypes used by real HuggingFace checkpoints).

Usage: python3 tools/real_model_crosscheck.py <model.safetensors> [tensor_name]
"""
import json, struct, sys

path = sys.argv[1]
raw = open(path,"rb").read()
n = struct.unpack("<Q", raw[:8])[0]
header = json.loads(raw[8:8+n].decode()); data = raw[8+n:]
rows=[]
for name, meta in header.items():
    if name=="__metadata__": continue
    b,e = meta["data_offsets"]; blob = data[b:e]
    numel=1
    for d in meta["shape"]: numel*=d
    dt = meta["dtype"]
    if dt == "BF16":
        u = struct.unpack("<%dH" % numel, blob)
        bits = (u[0] << 16) & 0xFFFFFFFF if numel else 0
        vals = [struct.unpack("<f", struct.pack("<I", v << 16))[0] for v in u]
    elif dt == "F32":
        u = struct.unpack("<%df" % numel, blob); vals = list(u)
        bits = struct.unpack("<I", blob[:4])[0] if numel else 0
    elif dt == "F16":
        u = struct.unpack("<%de" % numel, blob); vals = list(u)
        bits = struct.unpack("<H", blob[:2])[0] << 16 if numel else 0
    else:
        raise SystemExit("unhandled dtype " + dt)
    assert len(vals) == numel
    t=0.0
    for v in vals: t+=float(v)
    rows.append((name, dt, meta["shape"], numel, bits, t))
print("tensors=%d" % len(rows))
for name,dt,shape,numel,bits,t in sorted(rows):
    print("tensor %s dtype=%s shape=[%s] numel=%d first_bits=%08x sum=%.8f" % (name,dt,",".join(map(str,shape)),numel,bits,t))
lookup_name = sys.argv[2] if len(sys.argv) > 2 else "model.embed_tokens.weight"
emb = header[lookup_name]; b,e = emb["data_offsets"]
u = struct.unpack("<%dH" % ((e-b)//2), data[b:e])
vals=[struct.unpack("<f", struct.pack("<I", v << 16))[0] for v in u]
t=0.0
for v in vals: t+=float(v)
print("lookup %s numel=%d sum=%.8f" % (lookup_name, len(vals), t))
