# ANEDVD — measured results

All numbers from the development machine:

* **Apple A18 Pro** (Mac17,5), 6 cores, 8 GB unified memory
* **macOS 27.0.1** (26A434), SIP enabled
* unsigned binary, **no entitlements**
* Zig 0.17.0, `-O Debug` (the Zig side is not the bottleneck; the ANE is)

## ANE matmul throughput

`anedvd bench` — a single 1×1 conv kernel, fp16, batch 1, weights baked in,
timed over 50 evaluations after warm-up.

| size (in=out) | µs/eval | GFLOP/s | implied weight read |
|---|---|---|---|
| 128 | 80.9 | 0.4 | launch-bound |
| 256 | 86.6 | 1.5 | launch-bound |
| 512 | 156.5 | 3.4 | |
| 1024 | 191.0 | 11.0 | 5.5 GB/s |
| 2048 | 395.2 | 21.2 | 10.6 GB/s |
| 4096 | 1409.8 | 23.8 | 11.9 GB/s |

Two regimes:

* **Below ~256×256** the ~80–90 µs per-evaluation launch overhead dominates.
* **Above ~1024** the kernel is memory-bandwidth-bound at ≈12 GB/s of fp16
  weights. A 4096×4096 fp16 matrix is 33.5 MB, and 33.5 MB / 1.41 ms ≈ 23.8
  GB/s of traffic counting the read+write of the output; the ANE's share of DRAM
  bandwidth is the limit, not the MAC array.

That is why the engine fuses projections as aggressively as the data
dependencies allow (4 kernels per layer instead of 6) and why int8 weight
quantisation is the most promising future optimisation: it would halve the bytes
per token.

## Prompt batching (activation width)

`anedvd width` — one 1x1 conv, fp16, 2048→2048, weights read once per eval:

| width | µs/eval | vs width 1 | GFLOP/s |
|---|---|---|---|
| 1 | 304.97 | 1.00× | 27.5 |
| 8 | 286.93 | 0.94× | 233.9 |
| 32 | 305.47 | 1.00× | 878.8 |
| 64 | 290.23 | 0.95× | 1849.8 |
| 128 | 321.40 | 1.05× | 3340.8 |

Flat within 5% across a 128× range of work: the kernel is reading weights, not
computing. So every kernel is compiled for `chunk` columns (default 64), a decode
step fills column 0 only, and a prefill step fills up to 64 — one kernel set, no
extra compiles, no decode penalty.

`anedvd run --ab` runs the prompt through both paths and compares:

```
A/B sequential vs batched: max|diff| = 0.000000e0, argmax 504 vs 504
```

Bit-identical, which is also how the bug below was caught.

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
| 1024 | 967 us | 234 us | 4.1x |
| 2048 | 2098 us | 483 us | 4.3x |

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
8. **Debug builds made prefill look broken.** The Zig-side CPU work (batching,
   attention, dequantisation) is ~8× slower unoptimised: the same 33-token
   prefill took 0.59 s in Debug and 0.12 s in ReleaseFast. `zig build` now
   defaults to ReleaseFast.
