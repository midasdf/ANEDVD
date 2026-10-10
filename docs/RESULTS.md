# ANEDVD — measured results

All numbers from the development machine:

* **Apple A18 Pro** (Mac17,5), 6 cores, 8 GB unified memory
* **macOS 27.0.1** (26A434), SIP enabled
* unsigned binary, **no entitlements**
* Zig 0.17.0, ReleaseFast (the Zig side matters: Debug is ~8x slower there)
* every number from repeated runs — a single run can be 40% off

## ANE matmul throughput

`anedvd bench` — a single 1x1 conv kernel, fp16, batch 1, weights baked in,
timed over 40-60 evaluations after warm-up. Best of three runs, because the ANE
itself is noisy (an identical run can come out 30-40% slower).

| size (in=out) | us/eval | GFLOP/s | weight read |
|---|---|---|---|
| 128 | 71 | 0.5 | launch-bound |
| 256 | 101 | 1.3 | launch-bound |
| 512 | 96 | 5.5 | |
| 1024 | 131 | 16.1 | 16 GB/s |
| 2048 | 289 | 29.1 | 29 GB/s |
| 4096 | 1161 | 28.9 | 29 GB/s |

Two regimes:

* **Below ~512** a ~70-100 us per-evaluation overhead dominates: the same kernel
  costs about the same for 1 column as for 128 (see the width sweep), so small
  projections are launch-bound but prompt tokens are nearly free.
* **Above ~1024** the kernel saturates at ~29 GB/s of fp16 weights, which is
  where the decode speed comes from: 32.5 MB of weights in 1.16 ms.

## Current numbers (all via `--repeat`, medians)

Single runs are not trustworthy: the ANE's own per-token time varied 16.9-32.1 ms
across identical invocations, so every number here comes from repeated runs.

| model | prefill (short) | decode (median of 9) | peak RSS |
|---|---|---|---|
| SmolLM2-135M GGUF Q8_0 | 488 tok/s | 52.2 tok/s | 304 MB |
| SmolLM2-135M HF safetensors F16 | 371 tok/s | 47.4 tok/s | ~450 MB |
| Qwen2.5-0.5B GGUF Q8_0 | 190 tok/s | 22.7 tok/s | 1.0 GB |
| Qwen3-0.6B GGUF Q8_0 | 146 tok/s | 17.1 tok/s | ~1.1 GB |
| Qwen2.5-1.5B GGUF Q4_K_M | 28 tok/s | 9.5 tok/s | 1.6 GB |
| TinyLlama-1.1B GGUF Q8_0 | ~150 tok/s | 17.0 tok/s | ~1.5 GB |

Decode ranges over 9 repeats are wide (SmolLM2: 32-57 tok/s, Qwen2.5-0.5B:
21-26) because the ANE's own timing is unstable, which is why every number here
is a repeated measurement.

Prefill is faster with longer prompts because the ANE takes a whole chunk at
once; Qwen2.5-0.5B:

| prompt tokens | prefill |
|---|---|
| 59 | 527 tok/s |
| 203 | 683 tok/s |
| 491 | 453 tok/s |
| 971 | 283 tok/s |

## Per-kernel bandwidth

`anedvd kernels <model>` times each layer kernel and reports achieved bandwidth.
Qwen2.5-0.5B, chunk 128:

| kernel | ms/1 eval | weights | GB/s |
|---|---|---|---|
| qkv | 0.200 | 2.1 MB | 10.3 |
| o | 0.176 | 1.6 MB | 9.1 |
| ffn (fused) | 0.902 | 26.2 MB | 29.0 |
| lm_head | 6.750 | 272 MB | 40.3 |

The small kernels run at lower bandwidth simply because a fixed per-eval cost
(~100 us) is a larger fraction of a shorter eval; the large ones reach 30-40 GB/s,
which is the practical ceiling. `lm_head` is 14% of decode time for Qwen2.5-0.5B
(17% for SmolLM2-135M at 7%) because the vocabulary is 151 936 wide.

## Model runs

### SmolLM2-135M-Instruct (Q8_0, 144 MB)

30 layers, hidden 576, 9/3 heads, head_dim 64, inter 1536, vocab 49152.

```
121 ANE kernels compiled in 9.6 s
prompt: <|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n
output: The capital of France is Paris. Paris is a city located in the northern
        part of the country, and it is known for its historical landmarks,
        cultural institutions, and cultural attractions. Paris is famous for
prefill: 16 tokens in 0.06 s (258 tok/s)     [was 25.6 tok/s before batching]
decode:  24 tokens in 0.77 s (31.3 tok/s)    [was 25.4]
```

ReleaseFast, `chunk = 64`. Peak RSS 318 MB.

### Qwen2.5-0.5B-Instruct (Q8_0, 531 MB)

24 layers, hidden 896, 14/2 heads (GQA), head_dim 64, inter 4864, vocab 151936,
QKV biases, RoPE θ = 10⁶, eps = 10⁻⁶.

```
97 ANE kernels compiled in 9.5 s
output: The ocean is a vast and mysterious body of water that covers
        approximately 71% of the Earth's surface, containing vast amounts of
        water, life, and energy.
prefill: 33 tokens in 0.12 s (277 tok/s)     [was ~17 tok/s before batching]
decode:  34 tokens in 1.65 s (20.6 tok/s)    [28-35 tok/s once warm]
ANE:     3395 evals for 35 tokens (97 kernels/token), 1243 ms, 35.5 ms/token
```

ReleaseFast, `chunk = 64`. Peak RSS 988 MB.

66.8 ms/token for ~494 M fp16 parameters is 988 MB of weights per token, i.e.
≈14.8 GB/s — again at the ANE's memory-bandwidth ceiling. Qwen2.5-0.5B at fp16
on this ANE is bandwidth-bound, so a quantised weight format is the only way to
go substantially faster.

## Numerical accuracy

### Whole-engine check (`anedvd selftest`)

Tiny random transformer (2 layers, hidden 64, heads 4/2, inter 128, vocab 256),
ANE engine vs a pure-CPU reference of the same model and weights, 5 decode steps:

| FFN path | max abs diff | relative error | evals/token |
|---|---|---|---|
| **split (default)** | 1.2e-3 | **4.3e-4** | 9 |
| fused (`--fuse`) | 6.4e-2 | 2.3e-2 | 7 |

The split path is at fp16 rounding level. The fused path — which expresses SiLU
inside MIL as `sigmoid` + `mul` — is ~50× worse even at these tiny shapes, and
catastrophically wrong at real sizes (see below).

### Per-kernel check (`anedvd check models/…`)

Every kernel of layer 0 compared against a CPU matmul with the same fp16
weights, on SmolLM2-135M:

```
qkv          max|ANE-CPU| = 2.99788e-3  rel = 3.55246e-4  -> ok
o            max|ANE-CPU| = 2.00653e-3  rel = 1.41123e-4  -> ok
gate         max|ANE-CPU| = 1.13177e-3  rel = 4.73259e-4  -> ok
up           max|ANE-CPU| = 1.01376e-3  rel = 4.65309e-4  -> ok
lm_head      max|ANE-CPU| = 6.44226e-2  rel = 3.66178e-4  -> ok
ffn (fused)  max|ANE-CPU| = 1.14019e+1  rel = 8.59726e-1  -> BROKEN
```

The same check on Qwen2.5-0.5B is identical in character, and the CPU reference
in `anedvd cpu` reproduces the ANE's text token-for-token — the ANE path is not
merely "close", it is the same computation.

## Memory and startup

Qwen2.5-0.5B-Instruct Q8_0 (494 M params), measured with `/usr/bin/time -l`:

| build | peak RSS | note |
|---|---|---|
| eager load | 1.39 GB | whole model as fp16 in RAM + all weight blobs |
| streaming layers + chunked weight files | 965 MB | |
| + weight files deleted after load | ~0.72–0.99 GB | run-to-run variance from mmap residency |
| **+ prompt batching, ReleaseFast (current)** | **~0.99 GB** | larger activation surfaces |

SmolLM2-135M: 299 MB (Q8_0) / 361 MB (F16). The F16 case is larger only because
its mmap is larger; its embedding is a zero-copy view into that mapping.

Startup for Qwen2.5-0.5B, split by phase (`anedvd check` prints this):

```
phase split: write 0.31 s, ANECCompile 9.33 s, load 0.38 s
all kernels compiled in 20.39 s     (the rest is Q8_0 -> fp16 conversion in Zig)
```

`ANECCompile` is ~10 s for 97 kernels (~100 ms each) and dominates startup.
The ANE daemon does not appear to reuse compiled programs across processes on
this machine, so every launch pays it.

Two operational findings from this work:

* **The weight dictionary passed to `modelWithMILText:weights:optionsPlist:` is
  not read by the compiler** — it only feeds the cache hash. Empty data compiles
  to bit-identical results. It must still carry a content digest, or two models
  with the same MIL shapes would share a cache entry and silently reuse the
  wrong weights.
* **The ANE program pool is machine-wide.** A second process loading a full
  model fails with `no ANE resources (transient; retry)` (status 0x5) while
  another holds its kernels. The shim now retries with exponential backoff.

## Model formats

The same model in both formats produces the same text on the ANE:

| format | file | prefill | decode | peak RSS |
|---|---|---|---|---|
| GGUF Q8_0 | `smollm2-135m-q8_0.gguf` (145 MB) | 258 tok/s | 31.3 tok/s | 318 MB |
| HF safetensors F16 | `SmolLM2-135M-Instruct/` (269 MB) | 239 tok/s | 31.2 tok/s | 449 MB |

F16 is faster because it skips the Q8_0 → fp16 conversion and can use the
embedding straight from the mapping.

## int8 weights: negative result

`constexpr_affine_dequantize` — the MIL op CoreML uses for quantised weights —
is rejected by `ANECCompile` with `InvalidMILProgram` in every configuration
tried: two container layouts (the fp16 chunk format and maderix's int8 header),
three `axis` values, and an fp16-input control that isolates the op from the
dtype. The control also fails, so the op is absent from this ANE's MIL dialect
rather than merely int8 being unsupported. This matches the prior art (Orion:
"quantization is not yet supported"; Espresso: "INT8/quantized weights:
unsupported"). Reproduce with `probe/ane_int8_probe.m`.

Consequence: weight traffic stays fp16, and the measured ~12–15 GB/s of weight
bandwidth remains the decode ceiling.

## Sampling

The sampler keeps candidates within `max_logit - 20*T`, sorts that reduced set,
and truncates by `top_k` then `top_p`. That is one pass over the vocabulary plus
a small sort, replacing the previous O(vocabulary x k) selection — with a 151 936
token vocabulary and k=40 that scan alone was several milliseconds per token.

Effect on Qwen2.5-0.5B, same prompt ("Tell me about the ocean."):

| settings | output |
|---|---|
| greedy | "The ocean is a vast and complex system of water bodies that cover approximately 71% of the Earth's surface. It is the largest body of water on Earth, covering about 367,000 square miles (950" |
| `--temp 0.7 --top-p 0.9 --repeat-penalty 1.3` | "The Earth's oceans are a complex network of salty water that covers approximately 71% of its surface area, forming an immense and dynamic system...\n\nKey features include:\n\n1. **Size**: The largest single..." |

## Fused FFN: root cause of the 0.86 relative error

The fused FFN (`gate→conv, up→conv, sigmoid, mul, mul, down→conv`) was wrong by
8.6e-1 relative error at hidden 576 / inter 1536 while every single-conv kernel
was within 5e-4. `probe/ane_fuse_probe.m` bisects it stage by stage and, with
the fix in place, reports:

```
  1 conv, chunk0               rel=0.002125 ok
  1 conv, chunk1               rel=0.017461 ok
  2 convs (baseline)           rel=0.014526 ok
  2 convs + sigmoid            rel=0.001784 ok
  full fused FFN               rel=0.591226 ok      (vs the f32 reference)
```

The remaining error is fp16 accumulation through a much longer chain; against a
CPU matmul with the same fp16 weights the engine reports 7.5e-3 (SmolLM2) and
4.9e-3 (Qwen), which is the number `anedvd check` gates on.

**Root cause:** the ANE weight file's per-chunk 64-byte header records the
absolute offset of *that chunk's* payload. Every chunk was written with the
constant 128. The published `ffn_blob_ref.bin` shows 128 for chunk 0 and 240 for
chunk 1; the parser then reads weights from the wrong place for every chunk after
the first. Single-tensor files — every kernel except the fused FFN — are
unaffected, which is why the engine looked correct for weeks.

Fix in `src/ane/shim.m` (and the same assumption in `weights.zig`'s packer),
pinned by tests that compare against the fixture layout.

### Effect

Decode, same prompt, temperature 0, two runs each:

| model | split FFN | fused FFN |
|---|---|---|
| SmolLM2-135M | 32.7 / 32.8 tok/s | **46.9 / 46.5 tok/s** |
| Qwen2.5-0.5B | 20.7 / 27.7 tok/s | **32.1 / 29.9 tok/s** |

Kernels per layer drop from 4 to 3 (121 → 91 for SmolLM2), and the intermediate
activation no longer round-trips through the CPU. The fused path is now the
default; `--split` restores the old one.

## Attention optimisation (measured, then fixed)

`anedvd attnbench --ctx N` splits the CPU attention cost; at ctx 1024 with
Qwen2.5-0.5B's shape (14 heads, 2 KV heads, head_dim 64) the original split was:

| | per layer | share |
|---|---|---|
| QK dot products | 636 us | 66% |
| softmax | 58 us | 6% |
| value accumulation | 273 us | 28% |
| **total** | **967 us** | |

The dots were slow because `dot += a*b` cannot be reassociated by LLVM without
fast-math, so the loop serialises on the add dependency chain (~1.4 GFLOP/s).
Explicit 8-lane @Vector accumulators, plus keeping the value accumulation in
registers instead of writing o_h back per position, and an fp16 KV cache:

| ctx | before | after | speedup |
|---|---|---|---|
| 1024 | 967 us | 131 us | 7.4x |
| 2048 | 2098 us | 266 us | 7.9x |

Prefill attention was worse: the per-position loop re-walked the whole KV cache
for every query, and softmax called libm expf once per (query, key, head, layer)
— 344k calls per token at 971 tokens, 78% of prefill. `cpu.attentionPrefill`
batches a whole chunk and `cpu.expApprox` replaces the libm call with a
range-reduced polynomial (worst case 4.1e-6 relative error):

| prompt | before | after |
|---|---|---|
| 203 tokens | 500 tok/s | **698** |
| 491 tokens | 337 tok/s | **453** |
| 971 tokens | 172 tok/s | **289** |

End to end on Qwen2.5-0.5B decode (tok/s): ctx 16 19.6 -> 30.2, ctx 128
19.2 -> 28.6, ctx 512 13.6 -> 26.8, ctx 1024 12.2 -> 19.4. The CPU share of
decode at ctx 1024 fell from 48% to 24%.

## The test suite was not running

`zig build test` reported success from the first commit while executing **zero**
tests: Zig runs `test` blocks from the test root and the files it references,
and the build rooted the test at `src/main.zig`, which references no module's
tests. Running the suite for real (76 tests) failed three of them immediately:

* a genuine bug — `presence_penalty` was applied once per *occurrence* instead
  of once per distinct token, so a token appearing twice in the window was
  penalised twice;
* two incorrect assertions in the weight-layout tests (an element-vs-byte size
  mix-up, and reading a chunk header's magic instead of its size field).

All 76 pass now, including an engine test that builds a tiny model and asserts
the batched-prefill logits equal the per-token ones.

## Whole-model verification

`anedvd check` validates single kernels against a CPU matmul, which is necessary
but not sufficient: it cannot see a mistake in the hand-off *between* kernels.
`anedvd verify <model.gguf>` fixes that by running the same prompt through the
ANE engine and through the pure-CPU reference and comparing the final logits.
All four GGUF models on this machine:

| model | prefill rel | decode rel | verdict |
|---|---|---|---|
| SmolLM2-135M GGUF Q8_0 | 1.71e-2 | 1.48e-2 | MATCH |
| SmolLM2-135M HF F16 | 1.70e-2 | 1.43e-2 | MATCH |
| Qwen2.5-0.5B Q8_0 | 1.32e-2 | 2.30e-2 | MATCH |
| Qwen3-0.6B Q8_0 | 1.15e-2 | 1.46e-2 | MATCH |
| Qwen2.5-1.5B Q4_K_M | 1.50e-2 | 1.97e-2 | MATCH |

A relative error of ~1.5% is fp16 accumulation over ~30 layers, not a layout
error (an actual layout error showed up as >100%).

## Sampling cost

Sampling looks cheap next to the ANE, but the top-k selection was doing far more
work than it needed. The cut is `max_logit - min_keep_delta * temperature`; at
temperature 1.0 that leaves ~31k candidates out of 151936, and the sampler then
sorted all of them to keep 40.

Measured in isolation (151936 candidates, top_k 40):

| step | cost |
|---|---|
| `std.mem.sort` (stable block sort) over all candidates | 18.4 ms |
| `exp()` over the same candidates | 1.1 ms |

So the sort was essentially the entire cost of sampling. Switching to
`sortUnstable` took the real number from **4.44 ms/token to 1.16 ms/token** at
temperature 1.0 (0.89 -> 0.37 at 0.8), which is about 10% of decode time at the
highest temperature the CLI accepts.

Two hand-written partial selections were attempted first and both were wrong —
a shifting sorted window that overwrote slots the scan had not reached yet, and
a size-k min-heap. The comparison test described below caught both, and the code
now uses the library rather than a third attempt.

## Silent prompt truncation

A prompt longer than the context window is truncated from the front — the right
choice, since the end of a conversation is what matters — but nothing said so. A
5010-token prompt came back `HTTP 200`, `prompt_tokens: 2043`, and a confident
answer; the only way to notice was to compare that number against what was sent.

Found by sending `"word " x 5000` on purpose. It now reports itself three ways: a
log line ("dropped 2967 leading prompt tokens (5010 sent, 2043 used) to fit the
2048-token context"), a `prompt_tokens_dropped` field in the OpenAI `usage` object
on both the streaming and non-streaming paths, and a status line in the WebUI.

## MoE support

Qwen2MoE runs end to end on the real 9.5 GB Qwen1.5-MoE-A2.7B-Chat Q4_K_M:

| prompt | output |
|---|---|
| The capital of France is | Paris |
| 2 + 2 = | 4 |
| The sun rises in the | east |

The routing, the top-k experts and the shared expert are verified against an
independent Python forward (`tools/moe_reference.py`) on the tiny-random checkpoint:
identical top-5 ids and logits matching to six decimals.

Three things were missing when this started, and each produced plausible rather than
broken output:

1. The routed experts never reached the engine — `LayerSource.load` returns
   `Matrices`, which had no field for them.
2. Prefill had no routing at all, so a prompt's routed-expert contribution was absent.
3. The shared expert's `sigmoid(gate . h)` scale was applied nowhere, because the ANE
   kernel cannot express a 1-wide projection plus a sigmoid.

Speed is the caveat and it is severe: ~35 s/token, 89% of it the CPU expert matmuls.
`research/moe-design.md` records the measurements.

## Server liveness during a long prefill

The mid-generation hook that keeps `/health` answerable only ran between decode
tokens, and prefill happens entirely before the first token — so the case the
hook exists for, a long job that must not make the server look dead, was the one
case it did not cover.

A 971-token prompt prefills in 6.4 s on Qwen2.5-1.5B. During that window `/health`
answered 1 of 3 requests, the others timing out; after adding `Engine.prefill_tick`
(one call per prefill chunk, ~8 chunks for that prompt) it answers 4 of 4, with
the generation itself unchanged at 6.6 s.

## Server bugs found by testing two requests instead of one

Every earlier test used a single request, which hid two bugs in the same code
path:

1. **A second identical request returned zero tokens.** `generate.Session`
   tracks which token ids the engine's KV cache holds, and that list included
   every *generated* token, not just the prompt. So `commonPrefix` matched the
   whole new prompt against the previous turn's prompt + output, reuse equalled
   the prompt length, nothing was prefilled, and decode then read `reuse + 1`
   positions — the extra entry being stale KV data from the model's own previous
   answer. The model saw its old output as context and emitted EOS immediately.
   Three identical requests: `+7 tok`, `+0 tok`, `+0 tok`, all HTTP 200.

   Reuse is now only allowed when the new prompt covers the entire cached prefix
   (`generate.reuseLength`), which keeps multi-turn chat working (turn 2 reused
   36 tokens) while a fresh prompt starts clean.

2. **A request needing the engine during a generation was silently dropped.**
   The mid-generation hook accepted the connection, read the request, and
   returned without answering if it was not a cheap GET — so the client saw a
   closed connection and no status line at all. It now returns 503 with
   `Retry-After: 1` and a JSON body.

Both were invisible because the server logged only streaming completions;
every completed generation is now logged with prompt/new/reused token counts.

## The recurring failure mode: checks that could not fail

Three separate bugs survived a green test run because the thing that was
supposed to catch them could not actually fail. Worth recording as a pattern:

1. **`zig build test` ran zero tests.** The test build was rooted at
   `src/main.zig`, and Zig only runs `test` blocks from the test root and the
   files it references. Every "tests pass" report was vacuous until
   `src/tests.zig` referenced the modules. Running the suite for real failed
   3 of 76 immediately.
2. **`run --ab` and the engine test compared a buffer with itself.** `forward()`
   and `prefill()` both return the engine's internal `logits` slice, so
   `for (seq, batch) |a, b|` diffed that one buffer against itself and always
   printed `max|diff| = 0`. Both now copy the first result out, and the test was
   verified to fail when a bug is deliberately reintroduced.
3. **`check` only ever validated layer 0, kernel by kernel.** It cannot see a
   mistake in the hand-off between kernels, which is exactly where the
   transposed prefill-attention layout lived.

`anedvd verify` exists because of (3), and (1) is why it took so long to notice
(2): with the suite actually running, the copy bug would have been caught by the
deliberate-mutation check from the start.

## Bugs found and fixed during development

These are worth recording because each one produced *plausible-looking* output
rather than an error:

1. **Planar tensor layout.** The ANE ignores nominal tensor dims and uses
   per-channel planes with a 64-byte minimum stride. Writing a `[1, 4, 1, 2]`
   tensor contiguously silently computed only the first channel's contribution.
2. **Weight-blob offsets.** `BLOBFILE(offset)` is the payload's absolute file
   offset minus 64, not the file offset. Getting it wrong compiled fine and
   produced wrong numbers.
3. **GGUF tensor index.** Tensor names were never inserted into the lookup map,
   so every weight lookup returned null — while metadata lookups (config) worked.
4. **`Ġ` space marker.** GGUF vocabularies spell a leading space as U+0120, not
   ASCII space; naive prompt tokenisation silently fed the wrong tokens.
5. **RoPE permutation.** Llama-family GGUFs are permuted to the adjacent-pair
   layout; Qwen2 GGUFs are not. Using the wrong convention gives *fluent but
   repetitive* text, which looks like a weak model rather than a bug.
6. **Qwen2 attention biases.** Qwen2 adds biases to Q/K/V; ignoring them
   produced multilingual garbage.
7. **Prefill dropped the rotated query.** The batched path applied RoPE to a
   gathered copy of q but never wrote it back, so the attention pass re-read the
   unrotated vector. Symptom: the model answered with EOS immediately for some
   prompts. The `--ab` check (sequential vs batched logits) now pins this at
   bit-identical.
8. **Prefill attention read the activations transposed.** cpu.attentionPrefill
   indexed its buffers as [position][head][dim] while the engine's activations
   are channel-major [channel * chunk + column]. Reading them transposed returns
   plausible numbers, so every per-kernel check passed; the damage only appeared
   once full layers were chained (Qwen2.5-1.5B produced "</>A, and with two, and
   with three"). Fixed, and `anedvd verify` now exists to catch this class.
9. **Two checks that could not fail.** `run --ab` and the engine test both
   compared the sequential and batched logits directly, but forward() and
   prefill() return the SAME internal buffer, so they compared a buffer with
   itself and always reported "max|diff| = 0". That is why bug 8 survived an
   "A/B is bit-identical" claim. Both now copy the first result out, and the test
   was verified to fail when the bug is deliberately reintroduced.
10. **The CPU reference skipped Qwen3's Q/K normalisation**, so `verify` blamed
    the ANE for a mismatch that was the reference's fault. LayerWeights now
    carries q_norm/k_norm and refForward applies them like the engine does.
11. **Debug builds made prefill look broken.** The Zig-side CPU work (batching,
   attention, dequantisation) is ~8× slower unoptimised: the same 33-token
   prefill took 0.59 s in Debug and 0.12 s in ReleaseFast. `zig build` now
   defaults to ReleaseFast.
12. **Gemma 2 emitted a run of "." because the `(1 + w)` norm offset was applied
    twice.** `GemmaRMSNorm` computes `x * (1 + w)` from the raw parameter, so an HF
    checkpoint needs the offset added — but the GGUF converter has already applied it.
    Measured on gemma-2-2b's `blk.0.attn_norm.weight`: mean **1.1927**, which is the
    effective factor for a converged model, where the raw HF parameter would be ~0.19.
    The flag is now set per source (`load_hf` yes, `load_gguf` no). With that one change,
    gemma-2-2b answers "The capital of France is **Paris**", "2 + 2 = 4" and "The sun
    rises in the **morning**".

    Four rounds of source-reading (transformers, llama.cpp's graph builder, the
    converter, whose model classes are not in the file the top-level script downloads)
    failed to settle it. One measurement of the stored norm values did.

    Along the way, Gemma also needed the tanh-GELU FFN (not SiLU — the two differ 40× at
    x = -3), the `sqrt(hidden)` embedding scale, both logit soft-caps, alternating
    sliding-window attention with **layer 0 sliding**, and `1/sqrt(n_embd/n_head)` as the
    attention scale (1/√288, not 1/√256 — 6% apart).

13. **`verify` reported MISMATCH for a model that was answering correctly.** Two causes,
    both in the checking code rather than the engine:
    * `refForward` never applied Gemma 2's sandwich norms (`post_attention_norm`,
      `post_ffw_norm`), so it was compared against a model two norms per layer short.
    * The comparison ran on **soft-capped** logits. With `softcap = 30` every value is
      compressed into [-30, 30] and tanh saturation destroys the ordering, so `rel` could
      not distinguish a working model from a broken one. `Engine.pre_softcap` now lets
      `verify` compare pre-cap logits, where the range is 146–292 rather than a flat 30.

    Same lesson as (10): a reference that implements less than the engine produces
    mismatches that look like engine bugs.

14. **`check` printed `RESULT: OK` regardless of what it found.** It listed
    `lm_head ... rel = 9.45e-1 -> BROKEN` and then contradicted it two lines later, so a
    script reading only the summary saw success. `reportKernel` now records failures and
    `check` prints FAIL and exits 1.
15. **MADV_RANDOM on the model mapping makes things worse, not better.** MoE expert access
    is scattered across a 9.5 GB file, which looks like the textbook case for disabling
    readahead. Measured over four interleaved pairs of single-token runs, counting minor
    page faults:

        MADV_RANDOM: 395442 383096 426033 386974   median 391208
        default:     394916 380089 385354 381300   median 383327
        wins for MADV_RANDOM: 0 of 4

    It loses every pair. The experts a generation revisits are evidently clustered by
    layer, so readahead is closer to right than random advice. `sys.zig` records this so it
    is not "fixed" again without measuring.

16. **A 4-slot expert LRU per layer is not worth its memory here.** `anedvd route` on a
    32-token generation measures 42.5 of 60 experts distinct per layer, and a 4-slot LRU
    hits 55% of selections. That saves ~264 ms of a ~2500 ms/token decode (**11%**) for
    4 x 24 x 23.1 MB = **2.2 GB** resident. On an 8 GB machine, where an earlier eager
    materialisation mistake already produced 30 s/token through swap thrashing, the memory
    is worth more than the 11%. See `research/moe-design.md`.
17. **The ANE has a ~103 µs fixed cost per `eval()`, and for small models that dominates
    decode.** `anedvd bench` sweeps a 1x1 conv over square sizes, timing only
    `kernel.eval()` (not the input write or output read):

        size    weight MB    us/eval    implied GB/s
         128      0.033      107.73        0.30
         256      0.131      106.63        1.23
         512      0.524      102.78        5.10
        1024      2.097      138.80       15.11
        2048      8.389      285.18       29.42
        4096     33.554     1206.42       27.81

    Time is **flat at 103-108 µs from size 128 to 512**, where the weights are 33 KB to
    524 KB — too small for bandwidth to matter. So ~103 µs is a floor on a single eval,
    independent of what it computes. Above that the marginal rate is 46-59 GB/s up to
    size 2048. Reproduced across runs (114.72 / 106.52 / 104.63 / 136.22 / 279.68 /
    1209.02 us).

    What that means for the models, using three projections per layer plus one head eval:

        model            evals/token   floor      decode       floor share
        SmolLM2-135M         91        9.4 ms     23.2 ms         40%
        Qwen2.5-0.5B         73        7.5 ms     30.4 ms         25%
        Qwen3-0.6B           85        8.8 ms     36.2 ms         24%
        Qwen2.5-1.5B         85        8.8 ms     88.0 ms         10%

    **A quarter to two-fifths of small-model decode is the ANE's per-eval floor**, and the
    only way to reduce it is fewer evals. A transformer decode cannot: qkv, attention
    output and the FFN are sequentially dependent through the CPU attention in between, so
    three evals per layer is the minimum. This is why SmolLM2's qkv runs at 6 GB/s while a
    2048-wide conv reaches 30 GB/s — the kernel is fine, the shape is simply below the
    floor's useful range. It also bounds priorities (1) and (2): the CPU-side sampler and
    attention are already a few per cent, and the ANE side is floor- plus bandwidth-bound.
18. **The API returned invalid UTF-8 for non-ASCII output.** A Japanese prompt through
    `/v1/chat/completions` produced a body that failed `bytes.decode("utf-8")`:

        b'\xe3\x81\x93\xe3\x82\x93\xe3\x81\xab\xe3\x81\xa1\xe3\x81'   <- ends mid-character

    Token boundaries fall inside multi-byte characters (the vocabulary splits by bytes), so
    concatenating token pieces can end in a truncated sequence, and `max_tokens` can cut a
    generation mid-character. Strict JSON clients reject such a body outright.

    Streaming now trims each emit slice back to a UTF-8 boundary so a partial sequence waits
    for the token that completes it, exactly as stop sequences already do; the final flush
    and the non-streaming route replace what cannot be completed with U+FFFD. Verified both
    routes: `VALID UTF-8`, content `こんにち<U+FFFD>`, and ASCII output unchanged.
19. **One stalled client made the whole server unresponsive.** Sending

        POST /v1/chat/completions HTTP/1.1
        Content-Length: 5000

        {"messages":[...]}          <- 40 bytes, then stop

    left `/health` timing out for **every** other client (3/3 attempts) for as long as the
    connection stayed open. No generation was running and no CPU was consumed; a malformed
    request was enough.

    Two causes. `readRequest`'s body loop is a blocking `recv` bounded only by `SO_RCVTIMEO`
    (30 s), and it sits on the server's single service path; the readiness check before it
    was `hasPendingInput`, which only means *some* bytes arrived — true here, because the
    headers had. And the connection-parking slot was singular, so `servicePending` closed
    newly accepted connections while it was occupied, blocking the accept path too.

    Fixed by requiring the whole request (`FIONREAD` + `MSG_PEEK` in `hasCompleteRequest`)
    before a connection is handed to `readRequest`, and by using four parking slots with a
    stalled connection — never a fresh arrival — evicted when they are full. Verified with
    three stalled connections parked: `/health` answered 4/4 instead of 3/3 timeouts.
20. **A generation failure looked like success to the client.** Two separate faults, both
    found by probing a live server:

    * `max_tokens` larger than the context failed inside `generate` with `PromptTooLong`,
      after the server had committed to a response. Streaming clients got `200 OK` with
      `Content-Type: text/event-stream` and **zero bytes**; non-streaming clients got **no
      reply at all** (`http=000 bytes=0`). `clampMaxTokens` now caps the request at what the
      engine can serve, keeping the prompt and leaving room for at least one token, and is
      applied to all three generation routes.
    * Every route propagated a `generate` error out of the handler, so any *other* failure
      had the same effect. Non-streaming routes now return a real 400; streaming routes
      write an error frame in their own dialect (`data: {"error":...}` for OpenAI,
      `event: error` for Anthropic), because the 200 is already on the wire by then.

    Verified by forcing the failure on all three APIs: 400 for non-streaming, an error frame
    for streaming. An oversized `max_tokens` now returns a real answer — 11 SSE frames,
    `finish_reason: "stop"`, `prompt_tokens_dropped: 0`.
21. **A wrong argument failed far from its cause.** `--chunk 0` compiled a width-zero kernel
    and reported `AneCompileFailed`; `--max-seq 0` reported `PromptTooLong`. Neither names
    the flag, and both send the reader to the wrong place. `argValueAtLeast` now rejects a
    value below a minimum with the argument named and exit code 2, applied where `run`,
    `serve`, `kernels` and `attnbench` read the flags. The minimums are measured: 0 is the
    only unusable chunk width, and a max-seq below 4 cannot hold a two-token prompt.

22. **Probed and found correct** (recorded so they are not re-tested):

    * **KV prefix reuse** — a conversation grown by one turn reuses 22 of 38 tokens
      (`[req] prompt 38 (22 reused, 16 new)`). Reuse correctly falls to 0 when the prompt is
      shorter than the cache or when a different conversation intervenes.
    * **Interleaved conversations do not corrupt each other** — A, B, A, B with temperature
      0 gives byte-identical output for both A and B.
    * **Malformed GGUFs are rejected safely**, with specific errors and exit 1: a random
      file and a text file give `NotGgufFile`, a truncated model `TruncatedFile`, an empty
      directory `NoShardsFound`, and a crafted header claiming 2^40 tensors or metadata
      entries `TooManyTensors` / `TooManyMetadataEntries`. No crash, no large allocation.
    * **HTTP edge cases** — `OPTIONS` 204, `HTTP/1.0`, a 5 KB URL (`404`), a 4 KB header
      value, `Host:x` with no space, lowercase method, pipelined requests, chunked encoding
      without a length (`400`), and invalid UTF-8 in the JSON body (`400`).
    * **The WebUI end to end** — `/health`, `/v1/models`, the page itself (24976 bytes) and
      its streaming chat call, whose frames assemble into valid UTF-8 content.
    * **Immediate EOS** — a degenerate prompt (1250 repetitions of `a`) makes SmolLM2 emit
      EOS as its first token. A cold request returns `finish_reason: "stop"` with empty
      content, and so does the CLI (`stop: stop`, 0 tokens), so this is the model, not a
      cache or API fault.
23. **Coverage added for guards and rules that only had comments.** Three paths were correct
    but untested, so nothing would have stopped a refactor from breaking them:

    * `TooManyTensors` / `TooManyMetadataEntries` — the guards that stop a crafted GGUF
      claiming billions of entries from being honoured. Confirmed the test fails when both
      are deleted (113 passed / 1 failed).
    * `parseAtLeast` — the argument minimum rule, reachable now that `main.zig` is in the
      test root. That module was the only one missing from `src/tests.zig`, so none of the
      CLI was analysed by the test build.
    * The MoE width derivation — `moe_inter` from `ffn_gate_exps` dims[1] and `num_experts`
      from dims[2], not from metadata. Confirmed the test fails when the derivation is
      replaced with the metadata default the code's own comment warns against
      (115 passed / 1 failed).

    Tests: 113 → 116. `main.zig`, `load_gguf.zig` and `gguf.zig` gained their first
    coverage of these paths.
24. **CPU attention becomes the dominant prefill cost at long context — correcting an
    earlier claim.** I recorded that "prefill CPU is only 8% of prefill time" from a
    20-token measurement. That is true only for short prompts. Measured on Qwen2.5-0.5B:

        prompt     prefill        attn        all CPU     attention share
          26 tok   0.17 s       0.24 ms/tok   0.99 ms/tok       ~14%
         194 tok   0.28 s       0.45 ms/tok   0.70 ms/tok       ~48%
         374 tok   0.65 s       0.79 ms/tok   1.06 ms/tok       **45% of prefill**

    `anedvd attnbench` at ctx 1024 breaks the cost down further: decode attention is
    135.3 µs, of which the Q·K dot is 64.2 (47%), softmax 21.4 (16%) and the A·V plus the
    rest 49.7 (37%); prefill is 171.5 µs per query at 896 past keys.

    That works out to 14-18 G element/s for the dot and the A·V step, so the inner loops are
    already running at a reasonable rate — the cost is the O(n²) work itself, not a bad
    inner loop. There *is* a structural redundancy worth knowing about: with grouped-query
    attention (14 heads over 2 KV heads) each K/V row is converted and read seven times, once
    per head in its group. Hoisting that would cut the conversions by 7x, but because the
    loops are already at 14-18 G/s rather than load-bound, the realistic gain is a fraction of
    the dot and A·V steps — not the 7x the redundancy suggests. Left unimplemented rather
    than changed on an estimate; the numbers above are what a real attempt should be judged
    against.
25. **The MoE batching now has the test the dense path always had.** The prefill-vs-decode
    equivalence test used a dense model, so the MoE expert batching — the code with the
    `moe_col_out[j..]` versus `[j * hd + c]` mis-indexing — was unverified. A two-layer,
    four-expert, top-2 MoE is now built in process and run through both paths on 16 prompt
    tokens, asserting the argmax matches and the numeric difference is bounded.

    Measured, which is where the thresholds come from:

        correct code   1.68e-1
        off-by-index   9.96e-1     (argmax differs too)

    The numeric bound is 4e-1, a 2.4x margin over the rounding and 2.5x under the bug —
    looser than ideal, and looser than the dense test's 1e-3 because the two paths sum the
    experts in a different order in fp16. The argmax assertion is the check that actually
    pins the behaviour. Worth noting for anyone reading a future measurement here: my first
    figure of 0.0078 was the *first element outside tolerance*, not the worst difference.
26. **The shared expert's `sigmoid(gate . h)` scale now has a test that can fail on it.** This
    was a real bug once — the scale was computed nowhere and the shared expert joined the
    residual at full weight — and neither of the existing checks could catch it: the
    prefill-vs-decode equivalence test would have both paths omit it and agree, and `verify`
    shares the code with the engine, so its CPU reference omits it too. The independent check
    was `tools/moe_reference.py`, run by hand.

    The new test is differential rather than numeric. With `shared_gate_lin` all zeros the
    pre-activation is exactly 0, so the scale is sigmoid(0) = 0.5 and the shared expert must
    contribute at half weight; leaving the tensor empty takes the other branch, where the
    scale is 1.0. Applied correctly the two runs differ; ignored they are byte-identical.
    Routed experts are zeroed so only the shared path contributes.

    Breaking each path in turn shows the pair now covers both:

        decode scale removed   -> the new test fails
        prefill scale removed  -> the MoE equivalence test fails

    Tests: 119 → 120. README's stale "76 tests" line is corrected too.
27. **An over-long prompt also starved the completion.** `clampMaxTokens` limited a request to
    `max_seq - prompt_len - 1`, which is correct while the prompt fits: the prompt keeps its
    place and generation gets the remainder. When the prompt did NOT fit it fell back to a
    single token, so a 154-token prompt with `max_seq` 128 asking for 20 tokens returned
    **1** with `finish_reason: "length"` — measured against a live server, and it made a
    growing conversation collapse to 1-token replies at turn 6.

    A prompt the engine must truncate anyway is no reason to starve the completion, because
    the context is freed by dropping prompt tokens regardless. The clamp now honours the
    request up to `max_seq - 2` in that case, which still leaves at least one prompt token:

        before:  completion=1,  dropped=28
        after:   completion=20, dropped=47     (what the client asked for)

    The drop is still reported in `usage.prompt_tokens_dropped` and in the server log.
28. **The CLI truncated prompts silently.** The server reports a cut prompt in
    `usage.prompt_tokens_dropped` and logs a warning; `chat` and `run` said nothing. Growing a
    conversation past `--max-seq 96` in `chat` showed the prompt capping at 87 tokens with
    `reused from cache` falling to **0** and nothing explaining either. Two consequences the
    user could not diagnose: the conversation silently forgot its opening turns, and every
    later turn re-prefilled the whole prompt (the truncated prompt is a suffix, so
    `reuseLength` correctly returns 0 and the prefix benefit is lost).

    Both now print a note naming the context size and the number of tokens dropped, only when
    something was dropped. `chat` also suggests `--max-seq N` or `/reset`, and `/reset` was
    checked to exist and to recover (the next prompt returns to 12 tokens from 87).
29. **Both KV caches overflowed above 1024 tokens, silently.** `anedvd verify` with a
    1121-token prompt reported all-zero logits and a rel of exactly 1.0:

        prefill: max|ANE - CPU| = 3.10734e3   max|logit| = 3.107e3   rel = 1.00000e0
        decode:  max|ANE - CPU| = 0.00000e0   max|logit| = 0.000e0

    The engine's cache is `max_seq` rows and the CPU reference's was a hardcoded
    `1024 * cfg.kvDim()`, so a longer prompt wrote past both. Neither crashed, which is why
    this went unnoticed — it returned meaningless numbers instead, and `verify` reported
    MISMATCH rather than saying the comparison could not be made.

    `Engine.prefill`/`forward` now return `error.ContextOverflow`; `verify` encodes the prompt
    before building the engine and sizes both contexts to fit; the `cpu` command sizes its
    reference from the prompt plus the completion. The same 1121-token prompt then matches at
    rel 1.20e-2 prefill and 1.53e-2 decode with equal argmax.

    The consequence worth stating plainly: `verify` is the ground truth behind every numerical
    claim in this project, and above 1024 tokens it had been comparing garbage with garbage.
30. **The KV-cache overflow was heap corruption, not just a wrong number.** Removing the two
    `ContextOverflow` guards to check the new test does not merely fail it:

        expected error.ContextOverflow, found { 1.51, -0.36, ... }
        thread panic: free of invalid memory [addr: 100a04040, len: 256]
                      or corrupted metadata

    The panic confirms it was memory unsafety rather than a numerical slip, and explains why
    `verify` returned all-zero logits with `rel = 1.0`: it had corrupted the allocator's
    metadata rather than crashing outright. The test covers all four boundaries — exactly
    filling the context, one token past it, a late start position (`prefill(ids, 6)` with four
    tokens needs row ten of an eight-row cache), and `forward` at `pos == max_seq`.
31. **`--ab` named nothing when it ran out of context.** It prefills the raw prompt, unlike
    the generation path which truncates, so a prompt longer than `--max-seq` hit the new guard
    and printed a bare `error: ContextOverflow`. It now says

        prompt is 330 tokens but the context is 64; raise it with --max-seq N

    and exits 2. The overflow surfaces in the sequential `forward` loop rather than at
    `prefill`, which is where the catch had to go — the `prefill` catch written first was dead
    code.

32. **Probed this round and found correct:** `anedvd selftest` (rel 9.68e-4, PASS), `anedvd
    probe` (rel 3.60e-4, CORRECT), and the sharded-HF path, which turns out to have four
    dedicated tests (fallback to every `*.safetensors` sorted, preference for
    `model.safetensors.index.json`'s `weight_map`, a broken index, and a tensor located across
    shards) plus an end-to-end one. `--repeat` was verified in the previous round: identical
    output every iteration, since `session.reset()` runs before each.
33. **The grouped-query "7x redundant K/V reads" cost nothing, measured.** With 14 heads over
    2 KV heads each K/V row is converted and read once per head in its group. Hoisting that
    conversion looked like the last optimisation left in attention, but the estimate was never
    checked. `anedvd attnbench` varies the head counts directly, so it can be:

        kv-heads=2  (group 7)    decode 208.4 us   dot 108.0   prefill 267.90 us/query
        kv-heads=14 (group 1)    decode 220.3 us   dot 127.9   prefill 268.67 us/query

    The grouped case is **faster** on decode and identical on prefill. The reason is that
    grouping shrinks the cache it re-reads: kv-heads=2 gives a 128-element row and a 262 KB
    cache against 896 elements and 1.8 MB for kv-heads=14, so the redundant reads are all
    served from a resident cache while the ungrouped case streams seven times more data.

    So there is no headroom here and the change should not be made. My earlier note said
    hoisting "would not give the 7x it suggests" — the measurement says it gives nothing at
    all, because the redundancy and the smaller cache cancel in the grouped case's favour.
34. **Mistral was listed as supported while its sliding window was ignored.** `known_architectures`
    names mistral; the sliding-window plumbing only ever ran for Gemma. `load_gguf` read
    `attention.sliding_window` inside the gemma branch and nowhere else, so a Mistral GGUF
    attended **globally** rather than within its window. The HF side read the window but left
    `swa_pattern` at 2, applying **Gemma 2's alternating pattern** and windowing only half of
    Mistral's layers.

    The rules really do differ, per llama.cpp: `set_swa_pattern` is
    `is_swa[il] = n_pattern == 0 || ...`, so `n_pattern == 0` means every layer slides
    (Mistral), where Gemma 2 alternates. `Config.swa_all` now carries it.

    Bounded impact — the window only binds past its size, and Mistral's is 4096 against a 2048
    default context — so this was latent. It is the difference between supporting a model and
    claiming to.
35. **Both sliding-window rules now have tests on both loaders.** `load_hf`'s mapper and
    `load_gguf`'s metadata reader were the two places the Mistral fix touched and neither had
    coverage. The tests assert a Mistral config windows every layer while a Gemma 2 config with
    the same window alternates (layer 0 slides, layer 1 does not), that a config with no window
    slides nowhere, and — for the crafted GGUF — that Mistral keeps the llama-family defaults
    (adjacent RoPE, no GELU, no norm offset) so a future edit cannot pull it into Gemma's flags.
    Both fail on the original behaviour. Tests: 122 → 124.
36. **The sampler is candidate-bound, not pass-bound — correcting my earlier guess.** I had
    assumed the two full-vocabulary walks (max, then collect) were the fixed cost and that
    there was nothing worth doing in the sampler. Measuring across temperatures on
    Qwen2.5-0.5B shows the opposite:

        temp 1.0   45,226 candidates avg (max 130,865)   1.33 ms/token
        temp 0.3       33 candidates avg (max 458)       0.13 ms/token

    The cost tracks the **candidate count**, so 0.13 ms is the two full-vocab walks over
    151,936 logits and **1.20 ms — 90% of sampling at temperature 1.0 — is candidate
    processing**. Priority (1) is therefore not closed: cutting the candidate work is worth
    up to ~1.2 ms/token, about 4-5% of decode on the small models.

    The shape of the fix is visible in the code: pass 2 materialises every candidate into an
    array (45k of them, ~360 KB of writes) and `selectTopK` then walks that array to keep 40,
    when the top-k could be maintained during the first walk instead. It is not implemented
    here — the semantics need care, `top_k = 0` bypasses the selection entirely and must keep
    its current behaviour, and this project's two previous attempts at hand-rolled partial
    selection were both wrong and had to be reverted.
37. **The top-k selection no longer sorts everything: 1.33 -> 0.47 ms/token.** `selectTopK`
    called `sortUnstable` on the whole candidate array to keep the first 40, and at
    temperature 1.0 that array holds ~45k of 151,936 logits. Nothing below the top `k` is ever
    read, so the rest does not need ordering. Keeping the k largest in `items[0..k]` and
    leaving the tail alone:

        temperature 1.0   1.33 -> 0.47 ms/token   (2.8x, candidates unchanged at 45,226)
        temperature 0.3   0.13 -> 0.13 ms/token   (nothing to gain, 33 candidates)

    `top_k = 0` and `k >= items.len` keep the old path.

    The existing test `selectTopK agrees with a full sort` caught two bugs in my first two
    attempts, each of which would have silently dropped candidates — writing past the element
    just read while still filling, and leaving the inserted value in the array twice when it
    came from the tail. AGENTS.md warns that two earlier hand-rolled partial selections were
    wrong and had to be reverted; this time the test that made it safe was already there.
38. **The sampling change is now checked at the draw, not only at the selection.** Replacing
    the full sort with a bounded selection made `selectTopK agrees with a full sort` the guard;
    that test sees a wrong array. The new test sees the consequence: with `top_k` set, 50 draws
    over 40 trials on 4000 logits (small spread, so ties are common) must never return a token
    below the k-th largest logit. Breaking the running-minimum comparison fails three tests
    including this one, so it is not redundant.

    One thing worth recording: the first break I tried was logically equivalent to the original
    (`items[k-1].logit > c.logit` against `c.logit <= items[k-1].logit`) and correctly passed.
    A deliberate break has to actually break something. Tests: 124 -> 125.
39. **The server path re-checked after the sampler change.** The sampler is shared by the CLI,
    the HTTP API and the WebUI, so changing `selectTopK` lands in all three at once. A live
    smoke test on SmolLM2-135M:

        greedy, non-streaming   "The capital of France is Paris"
        temperature 1.0, top_k 40   "The sea. It's the unsung hero of our l..."  finish "length"
        streaming               9 SSE chunks, terminated by [DONE]
        a second request        answered

    Streaming and the top_k path are the two that matter here: the first goes through the SSE
    holdback, and the second is the code the bounded selection replaced.
40. **The sampler's candidate-array writes are cheaper than they look — tried, measured, and
    reverted.** Pass 2 materialises ~45k candidates (~360 KB) so `selectTopK` can pick 40, and
    maintaining the top-k during that walk instead is provably equivalent (the kept set is
    `{v >= cut}` intersected with the top k, which is order-independent). Implemented, it
    passes all 125 tests and leaves greedy output unchanged, but:

        before   0.47 ms/token
        after    0.40, 0.43, 0.43, 0.43 ms/token      (four runs)

    About 0.05 ms — 0.2% of decode. Streaming writes to a 360 KB array are much cheaper than
    the write count suggests. The interleaved A/B that would have confirmed it never ran: my
    shell chain used `zig build | grep -c "error:" && ...`, and `grep -c` exits 1 when the
    count is zero, so everything after the first build was skipped. **A pipeline whose exit
    status drives `&&` must not be `grep -c`.** The tree was left holding the older source by
    that same break, which is how the mistake was noticed.

    Reverted: 0.2% does not pay for an extra branch, a second comparator and an order-restoring
    sort. The design stays in AGENTS.md in case a future change makes those writes matter.
41. **Two claims of support were separated from evidence of support.** The README listed
    "Verified end to end: Llama, Qwen2, Qwen3, Mistral, SmolLM2/3, the TinyLlama-era Llama
    layout, Qwen2-MoE and Gemma 2". Mistral and SmolLM3 were never run on this machine, and
    Mistral's sliding window had in fact been ignored entirely until the previous rounds — so
    the claim was not merely optimistic. The README now splits *run on a real model here*
    (Llama both formats, TinyLlama, Qwen2 0.5B/1.5B, Qwen3-0.6B, Qwen2-MoE, Gemma 2) from
    *recognised but not run* (Mistral, SmolLM3), and `known_architectures`' comment says that
    being on the list is not a statement that the architecture was exercised — nineteen are
    listed, six were run.
42. **Gemma 3's sliding pattern was assumed rather than read.** `swa_pattern` was hard-coded to
    2 — correct for Gemma 2's 1:1 alternation — and `sliding_window_pattern`, the key that would
    change it, was read nowhere. Gemma 3 repeats over six (five sliding layers, then one
    global), and the reference rule `sliding if (i + 1) % pattern` is the shape `layerIsSliding`
    already implements, so only the constant was wrong. One round after fixing Mistral's ignored
    window, the same class of fault: recognised in `known_architectures`, wrong in the mask.

    Both loaders now carry the pattern, with an explicit pattern overriding `swa_all`, and
    `model.test` pins 6.

    Two things I had to correct from the round that recorded this:

    * I wrote that an HF Gemma 3 fails loudly. It does not — `load_hf` maps every `Gemma*` to
      `"gemma2"` and every unrecognised architecture to `"llama"` by prefix match, with no
      validation. The safe direction I claimed was not there.
    * My first test called `layerIsSliding` with a hand-built config. That pins the rule but
      not the reading of the key, and deleting the plumbing failed nothing — a check that could
      not fail, in the round after writing about them. The test now goes through
      `toModelConfig`, and removing the plumbing fails it (126 -> 125 passed).
43. **The README's sampling section had drifted twice over.** It said the sampler "sorts that
    much smaller set" and "a single pass over the vocabulary plus a small sort". Both were true
    when written and neither is now: there are two passes, the cut leaves ~45 000 candidates at
    temperature 1.0 rather than something small, and the selection stopped sorting them in
    round 34. Replaced with the measured picture — 0.13 ms for the two walks, 1.33 -> 0.47 ms
    for the candidate processing, and the cost tracking the candidate count rather than the
    vocabulary.

    This is the fifth documentation correction in five rounds, all of the same kind: a change
    lands and the prose describing the old behaviour stays. The pattern is worth naming —
    **code and the sentences about it need to be changed in the same commit**, because the
    sentence is what a reader trusts and it fails silently.
44. **`--max-seq` was undocumented while every truncation message recommends it.** The command
    table listed `run`, `chat` and `serve` without it, and the notes added in round 22 say "use
    `--max-seq N` for a longer context" — so a reader following that advice had nothing to read.
    The flag is now on all three lines and `verify` gained `--layers`, with a paragraph in Long
    context covering the default (2048), which commands take it, that an over-long prompt is cut
    from the front and reported rather than silent, and that the KV cache is sized to it in both
    the engine and the CPU reference.

    Checked while doing it that a small context does not break generation: `--max-seq 64` with a
    6-token prompt still prefills and decodes 4 tokens, the same as the 2048 default.
45. **`run` prefills more tokens than the prompt contains, and it was not a bug.** For
    "The capital of France is" it reports **14** tokens where `anedvd cpu` and `anedvd layers`
    report **5**. `run` always applies `formatChatFor` — there is no raw path — so it sends

        <|im_start|>user
        The capital of France is<|im_end|>
        <|im_start|>assistant

    and the two commands are counting different strings. Not a tokenizer disagreement.

    The part worth recording is how long this took: `run` **prints** the formatted prompt and its
    count — `prompt (14 tokens): <|im_start|>user ...` — on the line immediately above the
    `prefill:` line I was reading. I had the answer on screen and grepped past it twice, then
    wrote it up as an open question. When two numbers in one project disagree, read the output
    around them before theorising about which is wrong.
46. **Gemma 4 E2B/E4B: what it actually needs, read from the official GGUF header.** Asked about
    running them, so rather than guess I fetched the first 8 MB of
    `google/gemma-4-E2B-it-qat-q4_0-gguf/gemma-4-E2B_q4_0-it.gguf` with a Range request and
    parsed the metadata directly. The architecture is `gemma4`, 35 layers, and it differs from
    Gemma 2/3 in five ways that cannot be guessed:

        gemma4.attention.key_length = 512        key_length_swa = 256
        gemma4.attention.value_length = 512      value_length_swa = 256
        gemma4.attention.sliding_window_pattern = [true x4, false, true x4, false, ...]
        gemma4.attention.shared_kv_layers = 20
        gemma4.embedding_length_per_layer_input = 256
        gemma4.feed_forward_length = [6144 x15, 12288 x20, ...]
        gemma4.rope.freq_base = 1e6              freq_base_swa = 1e4
        tokenizer.ggml.tokens = 262144 entries

    So: two head dimensions (512 on global layers, 256 on sliding), a per-layer sliding/full
    **list** rather than a modulo, per-layer FFN widths, K/V shared by the last 20 layers, and
    per-layer input embeddings (PLE). `head_count_kv = 1`, so it is MQA.

    Consequences for this codebase: `Config.head_dim` is a single number and the KV cache is
    allocated uniformly as `max_seq * kvDim`; the ANE kernel set is built per layer but the
    engine assumes one activation width and one KV width throughout; and the whole forward pass
    is text-only while the checkpoint is `Gemma4ForConditionalGeneration` (audio + vision +
    text, with the text config nested under `text_config`). This is a new architecture, not a
    flag.

    What was done now: `gemma4` is **refused with a message naming those five things** instead of
    falling through to the "treat it like llama" path, which would have run and produced fluent
    nonsense. A crafted Gemma 4 GGUF must return `error.UnsupportedArchitecture`, and the test
    also checks that `gemma2` still loads normally, so it cannot pass by refusing everything
    beginning with "gemma". The misleading "see src/hf.zig for the list" on a GGUF is fixed too.

    Not done, in order of least to most work: per-layer `layer_types` (the `swa_pattern` field is
    a modulo — an explicit list is a small change), per-layer FFN widths (kernels are already
    built per layer), per-layer head dims (touches the KV allocation), KV sharing (touches the
    attention path), PLE (a new embedding table plus a projection per layer), and the
    multimodal wrapper.
47. **Gemma 4 support, steps 1-2 of 6 done.** Working from the metadata read out of the official
    QAT GGUF header (finding 46), in the order recorded there:

    * **Step 1, the per-layer sliding list.** `layer_types` is a list, not a repeating length
      (E2B reads `[sliding x4, full, ...]`, and it repeats over five), so `Config.swa_pattern`
      could not express it. `Config` now carries `swa_explicit` plus a 128-bit `swa_layers` mask,
      and `layerIsSliding` consults it before `swa_all` and `swa_pattern`. `load_gguf` accepts
      `attention.sliding_window_pattern` as either an array of bools (Gemma 4) or the scalar
      length it already handled; `hf.Config` parses `layer_types` into the same mask, as bits
      rather than strings because the JSON arena does not outlive `loadConfig`.

    * **Step 2, per-layer FFN widths.** Gemma 4's layers are `[6144 x15, 12288 x20, ...]` and it
      declares that as an array, which the scalar read returned null on — so the load failed on
      a missing key before anything else could go wrong. `interFromMetadata` now takes the max
      (what the scratch buffers need) and each layer's width comes from its own tensor via
      `tensorOutDim`, the same "shapes are authoritative" rule already used for MoE experts. The
      engine takes `lw.gate.len / hidden` for the FFN kernel, which is the layer's width for a
      dense layer and the shared expert's for a sparse one, so the MoE branch is unchanged.

    Steps 3-5 remain and are the structural ones: per-layer head dims (512 on global layers, 256
    on sliding — the KV cache is allocated uniformly as `max_seq * kvDim` today), KV sharing
    across 20 layers, and PLE. `gemma4` is still refused at load, with the message now naming
    only those three.
48. **`anedvd check` on gemma-2-2b is not reproducible on this machine, and I mistook that for a
    regression.** While testing step 3 of the Gemma 4 work I saw `check` report
    `lm_head BROKEN` (rel 1.217e1). The sequence, all on the same model:

        committed code, before I touched anything this round   -> OK    (lm_head rel 2.07e-4)
        my working tree                                        -> BROKEN
        committed code again (stashed my changes, rebuilt)      -> OK
        committed code (my changes reverted, rebuilt)           -> BROKEN, then BROKEN again

    The last two are the same source, so it is not the change. The head input is identical and
    deterministic across runs (`|x| = 163.5754`), and the small models are stable throughout —
    SmolLM2-135M and Qwen2.5-0.5B both report OK right now. So this is machine state: the ANE
    program pool is machine-wide and a 2B model needs 27 kernels' worth of it, and the run that
    passed came after ~3.7 hours of no activity, where the failures come after a stretch of
    heavy model loading.

    Two consequences. **`check` on the 2B model is not a usable pass/fail gate after heavy use**;
    `verify` and the small-model `check` are, and they stayed green. And the reverted diff for
    step 3's engine half compiles and is saved at `/tmp/item3-engine.patch` — it deserves to be
    re-tested on an idle machine before being believed or blamed.

    What would settle it: run `check` on gemma-2-2b as the first ANE work after a fresh boot,
    and again after loading three or four models, and compare. That is one command each.
49. **Gemma 4 support, step 3 of 6 done: per-layer head dimensions.** The last structural piece
    before the two that remain. Sliding layers are 256 and global layers 512, so neither the KV
    cache (`max_seq * kvDim`, uniform) nor the qkv/o projection widths could stay single-valued.

    The engine now binds `l_hd`/`kv_dim`/`q_dim` inside both layer loops, `ropeAndCache` takes
    the head dimension instead of reading `cfg.head_dim`, `Engine.init` allocates each layer's
    K/V rows from `cfg.layerKvDim(i)`, the one-layer-at-a-time buffers are sized to the `max*`
    widths, and `buildLayerKernels` takes the layer index. `diagnose` takes the layer it
    describes rather than assuming layer 0.

    Both loaders carry it, under names that disagree: GGUF's `key_length` is the global 512 and
    `key_length_swa` the sliding 256, while HF's `global_head_dim` is the global 512 and
    `head_dim` the *sliding* 256 — the reverse. Two tests drive the mapping, and removing both
    reads fails them (132 -> 130).

    Two things worth keeping from the round this nearly went wrong in:

    * The engine half was parked for a round because `check` on gemma-2-2b said `lm_head
      BROKEN`. It was not the change: the same source passed and then failed, and the failure
      signature — `rel = 1.21667e1`, `head input |x| = 163.5754` — is byte-for-byte what the
      unchanged committed code produced. **`check` on the 2B model is not a pass/fail gate on
      this machine**; the small models and `verify` are, and they stayed green throughout.
    * Re-testing on the next round with the right gates took one command. Parked work plus a
      written-down verification plan beats shipping a change whose evidence is a flaky check.
50. **Gemma 4 step 4, config half: K/V sharing, measured from the file.** The last 20 of E2B's
    35 layers do not compute K/V. The reference takes the **last non-shared layer of the same
    attention type** as the donor — not the layer `num_shared` back — because sliding and global
    layers have different head dimensions (256 against 512). `Config.kv_shared_layers` plus
    `firstKvSharedLayer`/`isKvShared`/`kvDonor` now express it, tested over E2B's
    `[sliding x4, full]` pattern.

    Rather than take that from the source alone, range requests against the official QAT GGUF
    settled it: `blk.14.attn_k.weight` and `blk.14.attn_v.weight` are present, `blk.15.*` for both
    is **absent**, and `attn_q.weight` is present on every layer including 34. The shared layers
    have no K/V weights at all, so the loader must not ask for them and their qkv is Q-only.

    The same fetch showed the shared layers carrying `inp_gate`, `proj` and `layer_output_scale`
    — which is step 5's PLE, plus a per-layer output scale the reference models as a buffer.

    Engine half still to do: a Q-only qkv kernel for shared layers and attention pointed at the
    donor's cache. Tests: 132 -> 133.
51. **Gemma 4 step 4 done: K/V sharing.** The last 20 of E2B's 35 layers borrow an earlier
    layer's keys and values. The loader builds a Q-only qkv matrix for them (the checkpoint has
    no K/V weights — `blk.15.attn_k.weight` is absent where `blk.14.attn_k.weight` is present),
    `LayerKernels.qkv_rows` carries what the kernel really produces, `ropeQueryOnly` normalises
    and rotates the layer's own queries and nothing else, and attention reads
    `cfg.kvDonor(li)`'s cache in both paths.

    **A second memory bug in this feature, and it was mine.** The first version kept the
    original `defer allocator.free(q)` and added an explicit free — a double free that corrupted
    the heap on the first model through the path, which `verify` reported as MISMATCH for
    SmolLM2-135M. Ownership is now stated explicitly instead of implied by a defer. Worth
    noting that the small models caught it at once; gemma-2-2b's `check` was flaky throughout and
    would not have.

    Remaining: PLE (step 5) and the multimodal wrapper (step 6).
52. **Gemma 4 step 5 (PLE): the complete tensor inventory, read from the official GGUF.** Names
    collected by range request, so this is what the file has rather than what the architecture
    paper implies.

    Outside the layers:

        per_layer_token_embd.weight     the per-layer embedding table
        per_layer_model_proj.weight     hidden -> num_layers * ple_dim
        per_layer_proj_norm.weight      normalises the projected per-layer inputs
        token_embd.weight, output_norm.weight, rope_freqs.weight

    On EVERY layer, shared or not (layer 0 shown; the shared ones carry the same set minus
    attn_k/attn_v):

        blk.0.attn_norm, ffn_norm, post_attention_norm, post_ffw_norm, post_norm   <- FIVE norms
        blk.0.attn_q_norm, attn_k_norm
        blk.0.inp_gate.weight        PLE: gates this layer's per-layer input
        blk.0.proj.weight            PLE: projects it
        blk.0.layer_output_scale.weight

    **Correction to what I first wrote here.** I called `post_norm` "a fifth norm per layer,
    which no other architecture here has", inferring it from the name. It is not a residual norm:
    the reference calls it `post_per_layer_input_norm`, and it takes PLE's **projected per-layer
    vector** (width `hidden`) inside the PLE block. So the per-layer tensor names map as

        blk.N.inp_gate.weight         per_layer_input_gate      [ple_dim][hidden]
        blk.N.proj.weight             per_layer_projection     [hidden][ple_dim]
        blk.N.post_norm.weight        post_per_layer_input_norm[hidden]
        blk.N.layer_output_scale.weight  layer_scalar          a scalar buffer, ones initially

    which is a correction of exactly the kind this document keeps recording: reading a name and
    concluding a role.`layer_output_scale` IS a genuine extra — every layer ends with
    `hidden_states *= self.layer_scalar`. The reference's flow for the per-layer input is
    project -> scale by hidden^-0.5 -> reshape to (tokens, layers, ple_dim) -> normalise with
    `per_layer_projection_norm` -> `(projection + per_layer_input) * 1/sqrt(2)`, and the layer's
    `inp_gate` gates it before it joins the residual.

    Also from the reference and not yet in this engine, all of which the text path needs: the
    attention scale is **1.0** (Gemma 4 relies on its Q/K norms, unlike Gemma 2/3's
    `query_pre_attn_scalar`), V has a norm with **no learned weight**, and the RoPE base differs
    per layer type (1e6 global against 1e4 sliding — the GGUF carries `rope.freq_base` and
    `rope.freq_base_swa`, which are read but not yet used per layer).
53. **Gemma 4 step 5, in pieces: what is read, what is applied.** The layer set carries two things
    no other architecture here has — a fifth norm (`post_norm`) and a per-layer
    `layer_output_scale` — plus PLE's tables and a per-layer RoPE base. Done so far:

    * **Config and both loaders** carry `ple_dim`/`ple_vocab`, `layer_output_scale`, `post_norm`,
      `rope_theta_swa` and `attn_scale = 1.0`. The two flags are read by *looking for the
      tensors* (`blk.0.layer_output_scale.weight`, `blk.0.post_norm.weight`) rather than
      inferred from the architecture name.
    * **The per-layer RoPE base is applied**: `ropeAndCache` and `ropeQueryOnly` call
      `cfg.layerRopeTheta(li)`, so sliding layers rotate at 1e4 and global ones at 1e6. Nothing
      moves for any other model because `rope_theta_swa == 0` there.

    Not applied yet: the PLE lookup and its `inp_gate`/`proj`, the fifth norm, and the output
    scale. Those are the rest of step 5.

    Two process notes, both mine. The PLE assertions first landed inside the Mistral/Gemma test,
    where they ran and passed under a name that did not mention them — coverage that cannot be
    found is coverage that will be lost, so that test is renamed for what it checks. And the
    commit message for the RoPE change went through an **unquoted** heredoc, so the shell ate
    every backticked identifier and left a sentence with holes in it; a quoted delimiter is what
    prevents that.
54. **Gemma 4 step 5: the exact PLE algorithm, from the reference.** Written down because it is
    the part that cannot be guessed and every constant matters.

    Once per prompt, for all layers together:

        per_layer_inputs = embed_tokens_per_layer(input_ids)      # [T][layers * ple_dim]
                                 * sqrt(ple_dim)                  # the table's embed_scale
        per_layer_inputs = reshape(per_layer_inputs, T, layers, ple_dim)

        proj = per_layer_model_projection(inputs_embeds) * hidden^-0.5   # [T][layers * ple_dim]
        proj = reshape(proj, T, layers, ple_dim)
        proj = per_layer_projection_norm(proj)                    # RMSNorm over ple_dim

        per_layer_inputs = (proj + per_layer_inputs) * 2^-0.5

    Then each layer, at the very end of its block:

        residual = h
        h = inp_gate(h)            # hidden -> ple_dim
        h = gelu_tanh(h)
        h = h * per_layer_inputs[:, layer]      # elementwise, width ple_dim
        h = proj(h)                # ple_dim -> hidden
        h = post_norm(h)           # i.e. post_per_layer_input_norm, width hidden
        h = residual + h
        h = h * layer_output_scale # the per-layer scalar

    Two constants that would be easy to get wrong: the per-layer table is scaled by
    `sqrt(ple_dim)` (256 -> 16) and the combined vector by `2^-0.5`, and the model projection by
    `hidden^-0.5`. The main embedding is scaled by `sqrt(hidden)` as Gemma 2/3 are, which this
    engine already does through `embed_scale`.
55. **Gemma 4 step 5, progress so far.** Applied: the per-layer RoPE base (finding 53) and the
    per-layer output scale. Loaded but not applied: PLE's `inp_gate`/`proj`/`post_norm` (finding
    54 has the exact algorithm and its three constants).

    `layer_output_scale` is a scalar every layer ends with — `hidden_states *= self.layer_scalar`
    in the reference, where it is a buffer. `LayerKernels.out_scale` carries it and both loops
    apply it at the end of the layer: decode to its one column, prefill to every column of the
    chunk, the same split the residual already uses. It defaults to 1.0 and no other model here
    has the tensor, so the multiply is skipped and nothing moves.

    What is left for step 5 is PLE itself: the per-layer table lookup (23.5e9 parameters in a
    `[262144][35*256]` table, so a row at a time, not all of it), `per_layer_model_proj` with its
    `hidden^-0.5` scale, `per_layer_proj_norm`, the `2^-0.5` combination, and then the per-layer
    block — `inp_gate` -> `gelu_tanh` -> times the per-layer vector -> `proj` -> `post_norm` ->
    residual. `gemma4` stays refused until all of that is in, so a partly-applied path cannot
    produce output.
56. **Gemma 4: all six steps implemented.** The order recorded in finding 46 is done:

    1. the per-layer sliding list (`swa_explicit` + a 128-bit mask, from GGUF bool arrays and HF
       `layer_types`)
    2. per-layer FFN widths (from each layer's own tensor; the metadata array gives the max for
       scratch)
    3. per-layer head dimensions (256 sliding, 512 global; KV rows allocated per layer)
    4. K/V sharing (Q-only qkv for the shared tail, attention on the donor's cache,
       `ropeQueryOnly`)
    5. PLE (table row read on demand, `per_layer_model_proj`, `plePrepare`/`pleBlock` with the
       reference's three constants, plus the per-layer RoPE base and output scale)
    6. the multimodal wrapper's nested `text_config`

    137 tests pass, up from 76 when this work started, and every existing model is unchanged
    throughout — SmolLM2 both formats, Qwen2.5, gemma2-tiny MATCH; SmolLM2 and Qwen2.5 `check` OK;
    `run --ab` bit-identical; the tiny MoE's numbers and the 9.5 GB Qwen1.5-MoE load intact.

    **None of it has run on a real Gemma 4 checkpoint.** The design comes from the official QAT
    GGUF's metadata and tensor names (read by range request), the reference implementation, and
    unit tests of the arithmetic — not from a working forward pass. `gemma4` is still refused at
    load, which is what has kept that safe: nothing half-applied can produce output. The 3.35 GB
    checkpoint is downloading now; the first real test will be `anedvd check` and `verify` against
    the CPU reference, which is what has caught every previous mistake in this project.
57. **Gemma 4 on a real checkpoint: four faults, and what stopped the run.**

    Downloading the official QAT GGUF (3.35 GB) and running it found four things that the
    metadata, the reference implementation and 137 unit tests could not:

    1. `known_architectures` had no "gemma4" — the very list I added `gemma4` to in the *comment*
       and forgot in the array.
    2. The tokenizer refused the file: `tokenizer.ggml.model = "gemma4"`, and only
       `llama`/`gpt2`/`bpe` were accepted. The Gemma family names its SentencePiece variant rather
       than calling it "llama".
    3. The shared-KV tail has **no `attn_k_norm.weight`** either — layers 15-34 omit
       `attn_k`, `attn_v` *and* `attn_k_norm`, so the loader's unconditional read failed with a
       bare `TensorNotFound`.
    4. **`loadLayer` had none of the per-layer work.** There are two GGUF weight loaders:
       `loadWeights` (eager; `verify`, `cpu`) and `loadLayer` (streaming; `layers`, `run`, `chat`,
       `serve`). Every change for per-layer head dimensions, Q-only shared layers and PLE went
       into the first. The first real load therefore failed with
       `attn_q.weight has 3145728 elements, expected 6291456 (1536x4096)` — layer 0 is sliding at
       256 a head and the streaming path still asked for the global 512.

    The general lesson: **I changed the loader I was looking at, not every loader there is**, and
    the tests all exercise the one I had already done.

    The run then stopped for a different reason: after the process was killed during the load, the
    machine's ANE program pool was left degraded. `selftest` (a two-layer synthetic model) still
    passes, but any real model — 30 or 35 layers, tens of kernels — now stalls at the first layer's
    kernel compilation, with no error and no stray process to kill. Memory is fine: 66% free, swap
    500 MB. This is the machine-wide pool noted in AGENTS.md, and the remedy is a reboot rather
    than anything this code can do.

    So: the file now parses, the tokenizer accepts it and the load gets past configuration — but a
    complete forward pass has still not run.
58. **Running Gemma 4 through the CPU path found two more faults, with no ANE needed.** After the
    ANE pool degraded, `anedvd cpu` was the one path still usable — it is the pure-CPU reference
    and compiles no kernels. It found:

    * `loadWeights` still used `cfg.qDim()` for the **o projection** while the streaming loader had
      been fixed: layer 0 is a sliding layer at 8*256 = 2048 against the global 8*512 = 4096.
      This is the mirror image of the previous round's mistake — first the streaming loader had
      none of the per-layer work, then the eager one had all of it except `o`. **When two
      loaders diverge once, compare them tensor family by tensor family rather than fixing the
      one that failed.**
    * `neox_archs` listed gemma, gemma2 and gemma3 but not **gemma4**, so its RoPE came out
      adjacent where the reference uses `is_neox_style=True` (half-split). That convention is
      listed in AGENTS.md as producing fluent repetition rather than garbage.

    With both fixed the CPU reference loads the whole model and starts the forward pass:
    `gemma4 hidden=1536 layers=35 heads=8/1 rope_adjacent=false`. It then exited with no output;
    35 layers of scalar code on a 2B model is the likely cause but that has not been established,
    so it is recorded as an open question rather than a conclusion.

    Regression checked on the path that works: 137 tests pass, `anedvd cpu` still answers "Paris"
    on SmolLM2 and Qwen2.5, tiny MoE numbers unchanged.
59. **The CPU reference segfaults on Gemma 4, and why.** `anedvd cpu` on the real checkpoint exits
    with **139 (SIGSEGV)**, not an out-of-memory kill as I had assumed:

        cpu reference: gemma4 hidden=1536 layers=35 heads=8/1 rope_adjacent=false
        prompt (2 tokens):  hi
        ---
        Segmentation fault: 11

    The cause is visible in the code. `refForward` in `src/main.zig` still takes its geometry from
    the config-wide values — `cfg.qDim()` and `cfg.kvDim()` at lines 347-348, `cfg.head_dim` at
    350 and then at 400, 406, 411-415 and 421 — and the reference's KV cache is allocated from
    `cfg.kvDim()` as well. On Gemma 4 layer 0 is a SLIDING layer with a 256-wide head
    (q_dim 2048) while `cfg.head_dim` is the global 512 (q_dim 4096), so the reference walks past
    the end of the layer's q. The engine was given per-layer geometry; the reference beside it was
    not.

    That is the third place this same mistake has surfaced — the streaming loader, the eager
    loader's `o` projection, and now the CPU reference — which is worth stating plainly: **the
    per-layer head dimension touched four things and I updated them one failure at a time.**

    Not fixed here. `refForward` is ~80 lines and verifying the fix needs a Gemma 4 CPU run that
    takes minutes, which is not something to start at the end of a round. The line numbers above
    are where it needs to change.
60. **A debug build found the Gemma 4 segfault that reading the code did not.** `zig build
    -Doptimize=debug` with bounds checks, run against the real checkpoint, named the line
    immediately: `main.zig:488`, the gate matmul in `refForward`. The reference used `inter` from
    the config — the MAXIMUM across layers (12288 on E2B) — for every layer, so layer 0 at 6144
    wide had `matmulF16` asked for twice the rows its matrix has. Fixed by binding
    `l_inter = lw.gate.len / hidden` per layer.

    That is the **fourth** instance of the same mistake in this feature:

        1. `loadLayer` (streaming) had none of the per-layer work
        2. `loadWeights` (eager) had it except the `o` projection
        3. `refForward` used cfg.head_dim / cfg.qDim() / cfg.kvDim() throughout
        4. `refForward` used cfg.inter for every layer's FFN

    Each was found by a separate failure, and three of the four by *running* something rather than
    reading it. The lesson is not "read more carefully" — I did read these — it is that a config
    full of per-layer fields is a standing invitation to keep using the config-wide ones, and a
    debug build catches it in one run where reading catches it in four.

    The debug run then got past that and stopped in `loadNorm` -> `g.readF32` while reading
    `blk.N.post_ffw_norm.weight`, so one more fault remains in `loadWeights`. Recorded with the
    call chain, not diagnosed.
61. **Gemma 4 E2B now runs end to end on the CPU path — and its output is wrong.** The FFN-width
    fix was the last crash. `anedvd cpu --model gemma-4-E2B-q4_0.gguf --prompt "The capital of
    France is" --max-tokens 4` completes with **exit 0**:

        cpu reference: gemma4 hidden=1536 layers=35 heads=8/1 rope_adjacent=false
        prefill 15.46 s, decode 6.83 s

    The text it produces is multilingual noise (`み気の थकान tỷ`), which is what a wrong forward
    pass looks like, and there is an obvious candidate: **`refForward` does not implement PLE at
    all.** The engine does — `preparePle` and `cpu.pleBlock` are wired into both its loops — so the
    reference beside it is computing a different model. That is exactly the shape of the Gemma 2
    sandwich-norm bug, where `verify` reported MISMATCH for a working model because the reference
    skipped a term.

    Two other things this round established:

    * The "segfault" in the debug build was **not a crash**: the run completed, printed its top-5
      and its timings, and the trace I had been reading was Zig's testing allocator reporting a
      **leak** at exit. `loadNorm` leaks the f32 buffer `readF32` returns when the tensor's length
      does not match the expected `n` — it returns `error.DimensionMismatch` without freeing `v`.
      On the real file that path is taken somewhere, which is itself worth knowing.
    * The release binary had never been re-run after the FFN fix; the last release attempt predated
      it, which is why I still believed there was a crash. **Re-run the thing you just fixed
      before concluding anything from a log.**
62. **Gemma 4 E2B answers correctly, not once but three times.** A single " Paris." could be luck;
    three factual completions from one implementation is evidence:

        The capital of France is   ->   Paris.
        The capital of Japan is    ->   Tokyo.
        The sun rises in the       ->   east, the sun rises in

    via `anedvd cpu --model gemma-4-E2B-q4_0.gguf` on the real 3.35 GB QAT checkpoint, about 20 s
    per run (prefill ~13 s, decode ~8 s). Every one of the six steps is in and the load-time
    refusal is gone, so the file loads as itself rather than under an escape hatch.

    What is NOT verified: the ANE path. `check`, `verify` and `run` all build kernels, and this
    machine's ANE program pool has been degraded since a Gemma 4 load was killed mid-way —
    `selftest` (two layers) passes, `check` on a 30-layer model stalls at the first layer's kernel
    compilation, no stray process holds it, and memory is fine. That is the machine-wide pool
    AGENTS.md describes, and a reboot is the remedy. So the claim is precise: **Gemma 4's text
    inference works through the CPU reference; whether the ANE kernels agree with it is untested.**

    Counting the whole feature: 76 tests became 137, nine real faults were found — four of them
    only by running the real file, and one only by a debug build — and every existing model is
    unchanged throughout.
63. **E4B cannot be verified on this machine, and the reason is arithmetic rather than code.** Two
    attempts:

    * `unsloth/gemma-4-E4B-it-qat-UD-Q2_K_XL.gguf` (3.22 GB) loads and then fails with
      `error: UnsupportedType` — an Unsloth "dynamic" quant mixes quantisation types, and
      `gguf.zig` implements fifteen of them, not all. The file was deleted rather than kept.
    * A standard Q4_0 E4B exists (`ggml-org/gemma-4-E4B-it-GGUF`, 4.59 GB), but it would not run
      here either. The CPU reference loads **every layer eagerly**, and E4B's per-layer weights
      are:

          inter = 10240   qkv 26.2 + o 21.0 + ffn 157.3 MB/layer ->  8.6 GB for 42 layers
          inter = 20480   qkv 26.2 + o 21.0 + ffn 314.6 MB/layer -> 15.2 GB for 42 layers

      against 8 GB of RAM. E4B's layers are 10240 or 20480 wide (`use_double_wide_mlp`), so it is
      8.6 GB at best.

    The streaming engine path would fit — that is exactly what it is for — but it builds ANE
    kernels, and this machine's ANE program pool has been degraded since a Gemma 4 load was killed
    mid-way. So E4B is blocked on a reboot either way, and the honest statement is that **E2B's
    text inference is verified and E4B's is not**.

    Worth noting for whoever picks this up: none of the nine faults found in this feature were
    E2B-specific. They were all in the shared loader, the shared reference or the shared engine,
    so E4B should need the same code — the only untested thing about it is its own geometry.
