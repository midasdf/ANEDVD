#!/usr/bin/env python3
"""Independent Qwen2MoE forward for the tiny test checkpoint, in plain Python.

Written from the reference implementation in transformers
(`Qwen2MoeSparseMoeBlock`), in f64, reading the safetensors file directly. It
shares no code with the Zig engine, so agreeing with it means the Zig MoE path
(router, top-k, experts, shared expert) is right rather than merely
self-consistent.

The checkpoint is `yujiepan/qwen1.5-moe-tiny-random`: hidden 4, 60 experts, top-4,
2 layers. Small enough that an unvectorised Python forward is instant.
"""
import json, math, struct, sys

def load(path):
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        hdr = json.loads(f.read(n))
        base = 8 + n
        out = {}
        for name, info in hdr.items():
            if name == '__metadata__':
                continue
            dt = info['dtype']
            shape = info['shape']
            numel = 1
            for s in shape:
                numel *= s
            f.seek(base + info['data_offsets'][0])
            raw = f.read(info['data_offsets'][1] - info['data_offsets'][0])
            if dt == 'F16':
                vals = list(struct.unpack('<%de' % numel, raw))
            elif dt == 'F32':
                vals = list(struct.unpack('<%df' % numel, raw))
            else:
                raise SystemExit('unhandled dtype ' + dt)
            out[name] = (shape, [float(v) for v in vals])
        return out

def matvec(w, x):
    """w is (shape, data) for [out][in]; returns [out]."""
    shape, d = w[0], w[1]
    out_dim = shape[0]
    in_dim = shape[1] if len(shape) > 1 else 1
    res = []
    for o in range(out_dim):
        acc = 0.0
        row = d[o * in_dim:(o + 1) * in_dim]
        for a, b in zip(row, x):
            acc += a * b
        res.append(acc)
    return res

def rmsnorm(x, w, eps):
    ss = sum(v * v for v in x) / len(x)
    inv = 1.0 / math.sqrt(ss + eps)
    return [v * inv * wi for v, wi in zip(x, w)]

def silu(v):
    return v / (1.0 + math.exp(-v))

def softmax(xs):
    m = max(xs)
    es = [math.exp(v - m) for v in xs]
    s = sum(es)
    return [e / s for e in es]

def rope_half(x, pos, theta, head_dim):
    """HF rotate_half convention: split the head in half and rotate."""
    out = list(x)
    half = head_dim // 2
    for d in range(half):
        inv = theta ** (-2.0 * d / head_dim)
        ang = pos * inv
        c, s = math.cos(ang), math.sin(ang)
        a = x[d]
        b = x[d + half]
        out[d] = a * c - b * s
        out[d + half] = b * c + a * s
    return out

def main():
    d = load('models/moe-tiny-hf/model.safetensors')
    cfg = json.load(open('models/moe-tiny-hf/config.json'))
    H = cfg['hidden_size']; L = cfg['num_hidden_layers']; NH = cfg['num_attention_heads']
    NKV = cfg['num_key_value_heads']; HD = cfg['head_dim'] if 'head_dim' in cfg else H // NH
    E = cfg['num_experts']; TOPK = cfg['num_experts_per_tok']
    MI = cfg['moe_intermediate_size']; SI = cfg['shared_expert_intermediate_size']
    eps = cfg['rms_norm_eps']; theta = cfg['rope_theta']
    norm_topk = cfg.get('norm_topk_prob', False)
    step = cfg.get('decoder_sparse_step', 1)
    mlp_only = set(cfg.get('mlp_only_layers', []))

    def g(name):
        return d[name]

    # Prompts given as token ids on the command line; default to a few ids.
    ids = [int(x) for x in sys.argv[1:]] or [1000, 2000, 3000]

    kcache = [[0.0] * (NKV * HD) for _ in range(L)]
    vcache = [[0.0] * (NKV * HD) for _ in range(L)]
    logits = None
    for pos, tok in enumerate(ids):
        # Embedding lookup: one row, so index instead of a matvec.
        eshape, edata = g('model.embed_tokens.weight')
        x = edata[tok * H:(tok + 1) * H]
        for li in range(L):
            p = 'model.layers.%d.' % li
            h = rmsnorm(x, g(p + 'input_layernorm.weight')[1], eps)
            q = matvec(g(p + 'self_attn.q_proj.weight'), h)
            q = [a + b for a, b in zip(q, g(p + 'self_attn.q_proj.bias')[1])]
            kk = matvec(g(p + 'self_attn.k_proj.weight'), h)
            kk = [a + b for a, b in zip(kk, g(p + 'self_attn.k_proj.bias')[1])]
            vv = matvec(g(p + 'self_attn.v_proj.weight'), h)
            vv = [a + b for a, b in zip(vv, g(p + 'self_attn.v_proj.bias')[1])]
            q = [rope_half(q[i * HD:(i + 1) * HD], pos, theta, HD) for i in range(NH)]
            q = [v for head in q for v in head]
            kk = [rope_half(kk[i * HD:(i + 1) * HD], pos, theta, HD) for i in range(NKV)]
            kk = [v for head in kk for v in head]
            kcache[li][pos * NKV * HD:pos * NKV * HD + NKV * HD] = kk
            vcache[li][pos * NKV * HD:pos * NKV * HD + NKV * HD] = vv
            group = NH // NKV
            attn = []
            for head in range(NH):
                kvi = head // group
                sc = []
                for t in range(pos + 1):
                    kt = kcache[li][t * NKV * HD + kvi * HD: t * NKV * HD + (kvi + 1) * HD]
                    s = sum(a * b for a, b in zip(q[head * HD:(head + 1) * HD], kt)) / math.sqrt(HD)
                    sc.append(s)
                pr = softmax(sc)
                o = [0.0] * HD
                for t, w in enumerate(pr):
                    vt = vcache[li][t * NKV * HD + kvi * HD: t * NKV * HD + (kvi + 1) * HD]
                    for j in range(HD):
                        o[j] += w * vt[j]
                attn.extend(o)
            proj = matvec(g(p + 'self_attn.o_proj.weight'), attn)
            x = [a + b for a, b in zip(x, proj)]

            h = rmsnorm(x, g(p + 'post_attention_layernorm.weight')[1], eps)
            if E > 0 and li not in mlp_only and (li + 1) % step == 0:
                rl = matvec(g(p + 'mlp.gate.weight'), h)
                pr = softmax(rl)
                order = sorted(range(E), key=lambda e: -pr[e])[:TOPK]
                ws = [pr[e] for e in order]
                if norm_topk:
                    s = sum(ws)
                    ws = [w / s for w in ws]
                acc = [0.0] * H
                for e, w in zip(order, ws):
                    ge = matvec(g(p + 'mlp.experts.%d.gate_proj.weight' % e), h)
                    ue = matvec(g(p + 'mlp.experts.%d.up_proj.weight' % e), h)
                    act = [silu(a) * b for a, b in zip(ge, ue)]
                    de = matvec(g(p + 'mlp.experts.%d.down_proj.weight' % e), act)
                    for j in range(H):
                        acc[j] += w * de[j]
                if SI > 0:
                    sg = matvec(g(p + 'mlp.shared_expert.gate_proj.weight'), h)
                    su = matvec(g(p + 'mlp.shared_expert.up_proj.weight'), h)
                    sact = [silu(a) * b for a, b in zip(sg, su)]
                    sd = matvec(g(p + 'mlp.shared_expert.down_proj.weight'), sact)
                    gl = matvec(g(p + 'mlp.shared_expert_gate.weight'), h)[0]
                    gs = 1.0 / (1.0 + math.exp(-gl))
                    for j in range(H):
                        acc[j] += gs * sd[j]
                x = [a + b for a, b in zip(x, acc)]
            else:
                gate = matvec(g(p + 'mlp.gate_proj.weight'), h)
                up = matvec(g(p + 'mlp.up_proj.weight'), h)
                act = [silu(a) * b for a, b in zip(gate, up)]
                x = [a + b for a, b in zip(x, matvec(g(p + 'mlp.down_proj.weight'), act))]
        h = rmsnorm(x, g('model.norm.weight')[1], eps)
        logits = matvec(g('lm_head.weight'), h)

    top = sorted(range(len(logits)), key=lambda i: -logits[i])[:5]
    print('args =', ' '.join(str(i) for i in ids))
    print('top5 ids:', ' '.join(str(i) for i in top))
    print('top5 logits:', ' '.join('%.6f' % logits[i] for i in top))
    print('sum_logits: %.6f' % sum(logits))

main()
