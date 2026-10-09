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
