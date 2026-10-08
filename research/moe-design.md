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

Not yet done: GGUF MoE (metadata is read, the 3-D expert tensors are not consumed),
and running the experts on the ANE at all — see the arithmetic above for why that
is the wrong target.
