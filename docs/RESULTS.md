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

## Model runs

### SmolLM2-135M-Instruct (Q8_0, 144 MB)

30 layers, hidden 576, 9/3 heads, head_dim 64, inter 1536, vocab 49152.

```
121 ANE kernels compiled in 9.6 s
prompt: <|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n
output: The capital of France is Paris. Paris is a city located in the northern
        part of the country, and it is known for its historical landmarks,
        cultural institutions, and cultural attractions. Paris is famous for
prefill: 16 tokens in 0.62 s (25.6 tok/s)
decode:  40 tokens in 1.57 s (25.4 tok/s)
ANE:     6776 evals for 56 tokens (121 kernels/token), 1293 ms, 23.09 ms/token
ANE share of wall time: 82%
```

### Qwen2.5-0.5B-Instruct (Q8_0, 531 MB)

24 layers, hidden 896, 14/2 heads (GQA), head_dim 64, inter 4864, vocab 151936,
QKV biases, RoPE θ = 10⁶, eps = 10⁻⁶.

```
97 ANE kernels compiled in 9.5 s
output: The ocean is a vast and mysterious body of water that covers
        approximately 71% of the Earth's surface, containing vast amounts of
        water, life, and energy.
prefill: 15 tokens in 1.66 s (9.1 tok/s)
decode:  34 tokens in 3.37 s (10.1 tok/s)
ANE:     4753 evals for 49 tokens (97 kernels/token), 3273 ms, 66.80 ms/token
ANE share of wall time: 65%
```

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
| **+ weight files deleted after load (current)** | **~0.72 GB** | |

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

| format | file | decode |
|---|---|---|
| GGUF Q8_0 | `smollm2-135m-q8_0.gguf` (145 MB) | 25–30 tok/s |
| HF safetensors F16 | `SmolLM2-135M-Instruct/` (269 MB) | 28–32 tok/s |

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
