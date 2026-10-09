# MoE support: what the engine can and cannot do on the ANE

Researched 2026-10-09 against the Qwen2MoE reference implementation
(`transformers/models/qwen2_moe/modeling_qwen2_moe.py`) and llama.cpp's tensor
naming (`src/llama-arch.cpp`), while looking at how
[Strata](https://github.com/ExTV/strata-5090-4070) runs a 125B MoE on consumer
hardware.

## The arithmetic that decides the design

| model | experts x layers | expert weights (fp16) | read per token (top-k) |
|---|---|---|---|
| Qwen1.5-MoE-A2.7B | 60 x 24 | 49.8 GB | 3.3 GB |
| Qwen3-30B-A3B | 128 x 48 | 116 GB | 7.2 GB |

Two facts fall out:

1. **Experts cannot all be resident on the ANE.** Qwen1.5-MoE alone would need
   1440 loaded kernels; the ANE program pool is machine-wide and small (an
   earlier probe found only ~1 kernel per 0.13-0.15 ms of steady-state cost, and
   a second process already cannot get resources). Building a kernel per expert
   is not an option.
2. **Decode reads the *active* experts, not the whole model.** 138 MB/layer for
   top-4 out of 60, versus 2.1 GB/layer if every expert were read. That is the
   entire reason a 125B MoE runs on a gaming PC: the sparse path is ~15x less
   memory traffic than the dense equivalent.

So the ANE is the wrong place for MoE experts. It is the right place for the
attention, which is dense, small, and where the existing engine already wins.
The split that follows from the numbers:

- **Attention, norms, lm_head, router** -> ANE, unchanged from the dense path.
- **Experts -> CPU**, reading only the k selected experts per token, straight out
  of the mapped file (no copy of the whole model).
- **Shared expert** -> CPU, always, since every token uses it. It is dense and
  small (`shared_expert_intermediate_size`, usually a fraction of `intermediate`).

Strata's own patches corroborate the shape of this: it streams cold experts with
`pread` instead of faulting, keeps the rest page-locked in RAM, and batches expert
gathers for prefill. The analogue here is `pread`-per-active-expert and a
per-layer expert gather over the chunk.

## Qwen2MoE forward, exactly

```
h      = post_attention_layernorm(x)
router = softmax(h @ gate.T)                       # [num_experts]
topv, topi = topk(router, k)                      # k = num_experts_per_tok
if norm_topk_prob: topv /= topv.sum()             # qwen1.5 tiny: False
out = sum(topv[i] * expert[topi[i]](h))
shared = shared_expert(h)                          # dense MLP, shared_intermediate
out += sigmoid(shared_expert_gate(h)) * shared     # a 1-wide gate
```

Each expert is `down(silu(gate(h)) * up(h))` — an ordinary SwiGLU MLP.

Dense layers interleave: a layer uses the sparse block only when
`(layer_idx + 1) % decoder_sparse_step == 0` and `layer_idx not in
mlp_only_layers`. `Qwen2MoeDecoderLayer` chooses per layer, so the model is a
mixture of dense and sparse layers, not "MoE throughout".

## Tensor names

HF (safetensors), per layer:
```
mlp.gate.weight                    [num_experts, hidden]
mlp.experts.{e}.gate_proj.weight   [moe_inter, hidden]
mlp.experts.{e}.up_proj.weight     [moe_inter, hidden]
mlp.experts.{e}.down_proj.weight   [hidden, moe_inter]
mlp.shared_expert.{gate,up,down}_proj.weight
mlp.shared_expert_gate.weight      [1, hidden]
```

GGUF:
```
blk.N.ffn_gate_inp.weight          router
blk.N.ffn_gate_exps.weight         [hidden, moe_inter, num_experts]
blk.N.ffn_up_exps.weight
blk.N.ffn_down_exps.weight
blk.N.ffn_gate_shexp / ffn_up_shexp / ffn_down_shexp
blk.N.ffn_gate_inp_shexp.weight
```
Config keys: `<arch>.expert_count`, `.expert_used_count`,
`.expert_shared_feed_forward_length`, `.expert_shared_count`.

The GGUF expert tensors are 3-D, declared `{n_embd, n_ff, n_expert}` in llama.cpp's
`llama-model.cpp`. ggml's `ne[0]` varies fastest, so the expert axis is the
slowest-varying one: each expert's slice is CONTIGUOUS, and within a slice the
layout is out-major with `in` contiguous, i.e. exactly `[out][in]`. `loadExperts`
therefore requires the expert axis to be last and copies each slice whole; it
still checks the other two dims rather than assuming, because the wrong order
loads fine and produces plausible garbage.

## Why not just build a kernel per needed expert

Because the set changes per token and per layer. Over a few hundred tokens that
is hundreds of distinct experts x layers compiled and loaded, each compile costing
milliseconds. Streaming a few MB of fp16 expert weights into a CPU matvec is
simply faster than compiling a kernel for them.

## Where the CPU time actually goes (measured on the engine)

`anedvd run` prints the CPU MoE time, which the ANE node split does not show. Two
corrections to what earlier commits in this repo claimed:

**The number was wrong.** The first version of that line divided the whole run's MoE
time by the *decode* token count, folding the prompt pass into it. On a 10-token
prompt generating 1 token that inflated it 11x: it reported 11386 ms/token. The
corrected figure for the same run is

  CPU MoE experts: 57.7 ms/token

So decode is not 93% CPU experts; that was an artefact of the arithmetic. Reported
properly, the ANE attention remains the larger share of decode.

**The matmul was still worth vectorising.** Isolated, one layer's expert work for one
token (4 experts x gate/up/down, from a 60-expert layer):

| step | before | after |
|---|---|---|
| matrix multiply | 122 ms | 20.8 ms |
| dequantise (streaming path only) | 205 ms | 165 ms |

0.83 GMAC/token at one multiply-accumulate per instruction predicts ~10 s/token, so
the scalar loop was real; eight lanes cut it ~6x, and the reference still matches to
six decimals.

**Keeping experts in f32 is not the win it looks like.** Measured over one expert
(2048x1408), the conversion dominates the READ:

| step | rate |
|---|---|
| `dequantizeRange` to f32 | 1101 M elem/s |
| `f32ToF16Slice` alone | 666 M elem/s |
| `readExpertF16` (both, end to end) | 7377 us |
| `dequantizeRange` alone, f32 out | 2613 us |

so a path that dequantises straight to f32 and matmuls there looked 2.1x faster, and a
prototype that read AND matmul'd each expert agreed (9774 us against 4581 us).

**It was wrong.** Those microbenchmarks read the expert as part of the timing. Once the
weights are resident and only the MATMUL is timed — which is what a decoding token
does, reusing scratch across experts — f32 is 4% *slower*:

| weights | 4 experts, resident |
|---|---|
| fp16 | 25.8 ms |
| f32 | 26.8 ms |

The f32 scratch is 34.6 MB against 17.3 MB, so the doubled footprint costs more in
cache pressure than the skipped conversion saves. Interleaving the two in one process
was necessary to see this: separate runs of the same binary varied 1947-2789 ms, which
swamps the effect entirely.

The vectorisation is still a hardware rate and widening it does not help (8, 16, 32 and
64 lanes give 514, 555, 499 and 359 M elem/s).

## Batching experts per prefill chunk (implemented)

Prefill re-read an expert for every (token, expert) pair. Since a prompt routes to
most experts anyway — 54.9 of 60 per layer at ~300 tokens — the fix is to route the
whole chunk first and then run each expert once over all the columns that chose it.

| | before | after |
|---|---|---|
| prefill, 332 tokens | 148.9 / 149.6 s | 92.1 / 93.5 s |
| CPU experts | 424.7 / 423.0 ms/token | 251.3 / 255.2 ms/token |

1.61x, verified by A/B on the same prompt with two runs each. The engine's prefill now
agrees with the CPU reference rank for rank (220, 151645, 151643, 264, 1147).

Decode cannot use this: it processes one token at a time, so there is nothing to share
an expert read with. That is why decode stays at ~2.5 s/token while prefill improved.

## Expert cache: measured, rejected

The obvious next lever for decode is caching hot experts, since 84% of the per-token MoE
cost is reading and dequantising them from the mapping. Measured with `anedvd route` on a
32-token generation (24 layers x 4 experts per token):

    distinct experts per layer: min 35, max 57, mean 42.5 of 60
    a 4-slot LRU per layer hit 55% of selections

So the hit rate is real. A 4-slot cache across 24 layers costs 4 x 24 x 23.1 MB = **2.2 GB
resident** (one expert is hidden 2048 x moe_inter 5632 x 2 bytes x 3 tensors).

It would save 55% of the ~120 ms read per selection: 4 selections x 66 ms = **264 ms of
~2500 ms/token, about 11%**.

Rejected: 11% for 2.2 GB on an 8 GB machine, where an earlier eager-materialisation
mistake already cost 30 s/token through swap thrashing. The memory is worth more than the
11% here. If the machine had 32 GB the answer would be different, and with a much longer
context the hit rate would rise too — the figure above is for only 32 generated tokens.

## Status and verification

Implemented and verified for safetensors Qwen2MoE, against
`yujiepan/qwen1.5-moe-tiny-random` (hidden 4, 60 experts, top-4, 2 layers).

The check that matters: `tools/moe_reference.py` is a from-scratch Python forward
written from the transformers reference, in f64, reading the safetensors file
directly. On tokens 1000,2000,3000 the two implementations agree exactly:

```
Zig     prompt top5 ids: 103920 101469 145307 109789 147842
        prompt top5 logits: 0.166585 0.165961 0.164188 0.163087 0.160523
Python  top5 ids: 103920 101469 145307 109789 147842
        top5 logits: 0.166585 0.165961 0.164188 0.163087 0.160523
```

`anedvd cpu --model <hf-dir> --prompt-ids 1000,2000,3000` runs the Zig reference on
chosen token ids and prints this summary, so the comparison is numeric rather than
by eyeballing generated text.

## Measured limits on this machine (A18 Pro, 8 GB, macOS 27)

Qwen1.5-MoE-A2.7B Q4_K_M is 9.5 GB on disk. Dequantised to fp16 it is ~33 GB, so
the eager loader cannot run it here at all (26.6 GB for the experts alone, one
layer being 1.11 GB). `Gguf.readExpertF16` streams one expert out of the mapped
file instead, touching only its own bytes.

Measured on the real file, reading `blk.0.ffn_gate_exps` (60 experts of 2048x1408):

| | |
|---|---|
| all 60 experts, cold | 1.54 s, 225 MB/s |
| top-4 experts per token, warm | 105 ms/token, 219 MB/s |

So the rate is the same cold and warm, which says the bottleneck is **not** page
faults or disk: it is the scalar dequantiser. The GGUF dequantisers are bit-exact
against ggml (verified by the fixture cross-check) but they process one value at a
time.

That made the full model impractical as written: ~1.66 GB of fp16 expert output
per token across 24 layers, at ~220 MB/s, is **7.6 s/token**.

Three passes brought the streaming rate to **668 MB/s** (219 -> 406 -> 573 -> 668),
each step bit-exact against compiled ggml, and one token's experts now cost
**3.5 s/token**:

| change | fp16 output rate |
|---|---|
| scalar everywhere | 219 MB/s |
| Q4_K + Q8_0 vectorised | 406 MB/s |
| f32 -> f16 conversion vectorised | 573-668 MB/s |

**An intermediate diagnosis was wrong and is worth recording.** After the first
pass the streaming rate was 219 MB/s while a raw in-RAM Q4_K dequantise measured
**2931 MB/s**, a 13x gap — so the dequantiser was not the wall. Benchmarking the
same four experts repeatedly (573 MB/s, flat across rounds) ruled out page faults
too, since repeated reads of resident pages cost the same as the first. The gap was
the scalar `@floatCast` conversion pass after each chunk, which the third change
removes.

The honest ceiling now: 0.29 tok/s before a single matmul runs, so large-MoE decode
on this machine needs the algorithmic levers below rather than faster loops.

What would change it, in the order the numbers suggest:

1. **SIMD dequantisation.** 220 MB/s for scalar block dequantisation is the wall.
   This is the single highest-value change and it helps the dense GGUF path too.
2. **Keep hot experts resident.** This is exactly what Strata does — experts no card
   holds are page-locked in RAM and only cold ones are `pread`. A routing histogram
   over a few hundred tokens would say whether 1.5 GB of experts covers most of the
   traffic.
3. **Batch prefill.** One expert read then serves every token in a chunk that
   routes to it, so prompt processing amortises the read over 128 positions instead
   of one.

## Where the CPU time actually goes (measured on the engine)

`anedvd run` now prints the CPU MoE time, which the ANE node split does not show.
On Qwen1.5-MoE-A2.7B, one token with a 10-token prompt:

  CPU MoE experts: 11386 ms/token (93% of decode+prefill)

Broken down in isolation, per layer for one token (4 experts x 3 tensors):

| step | before | after |
|---|---|---|
| matrix multiply | 122 ms | 23 ms |
| dequantise | 205 ms | 165 ms |

The matmul was scalar — one multiply-accumulate per iteration — and 0.83 GMAC per
token at ~1 MAC/cycle predicts the ~10 s/token the engine measured. Vectorising it
to eight lanes cut that step 5.3x, and the MoE reference still matches exactly.

**End-to-end decode barely moved (11.4 -> 9.97 s/token), and the reason matters:**
the microbenchmark re-reads the same four experts, so their pages are hot, while the
engine reads whatever the router picked. On this randomly-routed checkpoint that is
a different, cold set nearly every layer. The dominant remaining cost is therefore
dequantisation volume, not arithmetic — the same conclusion the streaming
measurements reached from the other direction (219 MB/s cold and warm alike).

## Batching experts per prefill chunk (implemented)

Prefill re-read an expert for every (token, expert) pair. Since a prompt routes to
most experts anyway — 54.9 of 60 per layer at ~300 tokens — the fix is to route the
whole chunk first and then run each expert once over all the columns that chose it.

| | before | after |
|---|---|---|
| prefill, 332 tokens | 148.9 / 149.6 s | 92.1 / 93.5 s |
| CPU experts | 424.7 / 423.0 ms/token | 251.3 / 255.2 ms/token |

1.61x, verified by A/B on the same prompt with two runs each. The engine's prefill now
agrees with the CPU reference rank for rank (220, 151645, 151643, 264, 1147).

Decode cannot use this: it processes one token at a time, so there is nothing to share
an expert read with. That is why decode stays at ~2.5 s/token while prefill improved.

## Expert cache: measured, rejected

The obvious next lever for decode is caching hot experts, since 84% of the per-token MoE
cost is reading and dequantising them from the mapping. Measured with `anedvd route` on a
32-token generation (24 layers x 4 experts per token):

    distinct experts per layer: min 35, max 57, mean 42.5 of 60
    a 4-slot LRU per layer hit 55% of selections

So the hit rate is real. A 4-slot cache across 24 layers costs 4 x 24 x 23.1 MB = **2.2 GB
resident** (one expert is hidden 2048 x moe_inter 5632 x 2 bytes x 3 tensors).

It would save 55% of the ~120 ms read per selection: 4 selections x 66 ms = **264 ms of
~2500 ms/token, about 11%**.

Rejected: 11% for 2.2 GB on an 8 GB machine, where an earlier eager-materialisation
mistake already cost 30 s/token through swap thrashing. The memory is worth more than the
11% here. If the machine had 32 GB the answer would be different, and with a much longer
context the hit rate would rise too — the figure above is for only 32 generated tokens.

## Status

- **Verified**: safetensors Qwen2MoE, end to end, against an independent Python
  reference (identical top-5 ids, logits to six decimals).
- **Implemented, config verified against a real 9.5 GB file**: GGUF MoE metadata
  and expert tensors. Widths come from the tensor shapes, because the metadata
  omits them (see the commit for the two bugs that caused).
- **Implemented, measured, not yet fast**: streaming one expert at a time out of
  the mapping. The rate above is the honest state.
- **Not attempted**: running experts on the ANE. The arithmetic at the top of this
  file is why.
