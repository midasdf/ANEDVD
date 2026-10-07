# Orion / ANE Constraints Reference — for writing valid MIL programs

**Purpose:** self-contained technical reference distilled from the Orion paper and its
companion open-source repository, for a CoreML-free ANE inference engine (Zig, driving
`_ANEClient` / `_ANEInMemoryModel` / MIL on Apple Silicon).

**Retrieved:** 2026-10-08 (Asia/Tokyo).

## Sources

| # | Source | URL | Notes |
|---|---|---|---|
| S1 | Orion paper, HTML v1 | https://arxiv.org/html/2603.06728v1 | arXiv:2603.06728v1 [cs.LG], 06 Mar 2026, CC BY 4.0 |
| S2 | Orion paper, LaTeX source (authoritative) | https://arxiv.org/src/2603.06728 | 24,830-byte tarball → `main.tex` (1,237 lines) + `references.bib`. **All verbatim paper quotes below come from this file.** |
| S3 | Orion paper, PDF | https://arxiv.org/pdf/2603.06728 | downloaded HTML instead; PDF not separately mined |
| S4 | Orion open-source repo | https://github.com/mechramc/Orion | MIT. Repo HEAD cloned 2026-10-08 |
| S5 | `docs/ane_constraints.md` | https://github.com/mechramc/Orion/blob/HEAD/docs/ane_constraints.md | **Superset of the paper's catalog** — 17 entries with symptoms/workarounds |
| S6 | `docs/ane_api_reference.md` | https://github.com/mechramc/Orion/blob/HEAD/docs/ane_api_reference.md | **The only place the full API sequence + MIL text grammar is written down** |
| S7 | `experiments/hello_mil.m` | https://github.com/mechramc/Orion/blob/HEAD/experiments/hello_mil.m | Complete, minimal, working MIL program + 7-step API sequence |
| S8 | `core/mil_builder.m` | https://github.com/mechramc/Orion/blob/HEAD/core/mil_builder.m | Real MIL emission code: header, conv1x1, layernorm, rmsnorm, gelu, silu, causal attention, BLOBFILE writer |
| S9 | `compiler/pass_ane_validate.c` | https://github.com/mechramc/Orion/blob/HEAD/compiler/pass_ane_validate.c | Machine-checkable constraint list (hardcoded `ANE_MIN_TENSOR_BYTES 49152`) |
| S10 | `README.md`, `RESULTS.md` | https://github.com/mechramc/Orion/blob/HEAD/RESULTS.md | 23 constraints claimed in repo README (20 paper + 3 community) |

> **Trust ordering used here:** S2 (paper LaTeX) > S6/S7/S8 (repo code) > S5 (repo prose).
> Where the paper and repo disagree, both are shown. "UNCERTAIN" marks anything not
> directly stated by a source or where sources conflict.

---

## 0. Read this first — four hard caveats

1. **The paper contains NO MIL snippet and NO code listing.** `main.tex` has zero
   `lstlisting` / `verbatim` / `minted` environments. The only API-level artifacts in the
   paper are the private-class table (Table 2) and Algorithm 1 (3 lines of pseudo-code).
   Every concrete MIL string in this document comes from the **repo** (S6–S8), not the paper.
2. **The paper says NOTHING about entitlements, SIP, sandboxing, code signing, or root.**
   A case-insensitive search for `entitlement|SIP|system integrity|sandbox|permission|
   TCC|root priv|sudo|codesign|AMFI|notariz` over the full LaTeX source returns **zero
   matches**. The same search over the entire cloned repo returns **exactly one** hit:
   `docs/ane_api_reference.md:12` — *"Returns `NULL` if framework not available
   (non-Apple-Silicon, SIP issues)."* That is the total extent of what these sources say.
   See §10.
3. **The paper's 20 constraints and the repo's 17 constraints are numbered differently and
   are NOT the same set.** The repo's `#12–#14` are the paper's `#18–#20`; the repo's
   `#15–#17` (weight-budget limits) do **not appear in the paper at all**; the paper's
   `#12` (matmul named consts) and `#14` (output liveness) are not in the repo's numbered
   list. Do not cross-reference by number — see the mapping table in §4.
4. **All characterization is on M4 Max (paper) / M4 (community constraints), macOS 15 and
   26.5.2.** Nothing here is verified on A18 Pro / macOS 27.0.1. The paper's own limitation
   section: *"the system has been validated on M4 Max only (other Apple silicon variants may
   have different ANE configurations)"*. Treat every numeric limit as generation-specific
   and re-probe on the target device. **UNCERTAIN on A18 Pro.**

---

## 1. Hardware characterization (M4 Max, H16)

Verbatim from S2, Table 1 (`tab:ane_hw`) and its two footnotes:

| Property | Value |
|---|---|
| Generation | H16 |
| Neural Engine cores | 16 |
| Peak throughput (INT8, Apple spec) | 38 TOPS |
| Peak throughput (fp16, measured) | ∼19 TFLOPS |
| On-chip SRAM | 32 MB |
| Evaluation queue depth | 127 |
| Dispatch overhead (XPC+IOKit) | ∼0.095 ms |
| Idle power | Zero (hard power gating) |

Footnotes, verbatim:
> "Apple specifies 38 TOPS (INT8), but the ANE dequantizes INT8 to fp16 before computation.
> Actual measured peak is ~19 TFLOPS fp16 (maderix); **INT8 saves only memory bandwidth,
> not compute cycles.**"

> "Performance drops ~30% when working sets exceed the 32 MB SRAM budget, forcing spills to DRAM."

Related verbatim findings attributed to maderix (S2 §2.3), independently confirmed by Orion
on M4 Max:

> "(4) showing that 1×1 convolutions deliver 3× better throughput than equivalent `matmul`
> operations; (5) discovering that deep operation graphs (16–64 ops) achieve 94% ANE
> utilization versus ~30% for single operations; and (6) identifying the ~119
> compilation-per-process limit."

Additional microbenchmark latencies (S2, Fig. 7, M4 Max, log scale):

| Operation | Latency |
|---|---|
| Single-token dispatch (bare XPC+IOKit) | 0.03 ms |
| Weight swap eval | 0.15 ms |
| ANE decode / token | 5.78 ms |
| Weight swap compile | 11.3 ms |
| ANE prefill cached | 95 ms |

> Note the conflict with the Table-1 dispatch figure of "~0.095 ms" vs the Fig-7
> "Single-token dispatch 0.03 ms". The caption says *"Single-token dispatch shows the bare
> XPC+IOKit overhead (~0.03 ms)"*. **UNCERTAIN which is the representative number** — both
> appear in the same paper.

---

## 2. The paper's catalog of 20 constraints (verbatim, S2 Table 3)

Caption, verbatim:
> "ANE constraint catalog. Source: P = prior work (maderix; hollance), O = discovered during
> Orion development, \*confirmed by maderix/ANEgpt codebases. Prior-work constraints were
> independently confirmed on M4 Max."

And the discovery-process claim (S2 §3):
> "The remaining 14 constraints were discovered during Orion development through 161
> engineering tasks spanning 18 sessions, primarily involving MIL IR compilation failures,
> evaluation errors, and silent numerical corruption..."

| # | Constraint | Symptom | Workaround | Src |
|---|---|---|---|---|
| 1 | `concat` MIL op rejected by ANE compiler | Compile failure | Split into separate programs | O |
| 2 | Multi-output buffers must have uniform sizes | `0x1d` at eval | Pad outputs to max size | O |
| 3 | Multi-output surfaces ordered alphabetically | Silent wrong data | Name outputs in sorted order | O |
| 4 | Minimum ∼49 KB IOSurface for eval | `0x1d` at eval | Pad seq dim ≥ 16 | O |
| 5 | ∼119 compilations per process limit | Silent fail / crash | `exec()` restart | P |
| 6 | SDPA causal masks silently ignored | Wrong attention | Manual causal masking | P\* |
| 7 | Weights baked at compile time | Stale weights | Recompile after update | P |
| 8 | BLOBFILE offset is `uint64(64)`, not 128 | Garbage weights | Correct offset in MIL ref | O |
| 9 | MIL text must be `NSData*`, not `NSString*` | Immediate crash | Encode to UTF-8 data | O |
| 10 | `gelu` is not a valid MIL activation | Compile failure | Tanh approximation | O |
| 11 | Weight dict must be `@{}`, not `nil` | Immediate crash | Pass empty dictionary | O |
| 12 | `matmul` transpose flags need named consts | MIL rejection | Emit `const` nodes | O |
| 13 | `conv` does not support `bias=` param | MIL rejection | Separate add operation | O |
| 14 | Output vars must ref live (post-opt) nodes | Invalid program | Update refs after DCE | O |
| 15 | `exec()` restart overhead ∼50 ms | Latency cost | Batch steps per process | P |
| 16 | 32K-channel convolutions rejected | Compile failure | CPU fallback or chunking | O |
| 17 | Conv 1×1 is 3× faster than `matmul` | Performance gap | Prefer conv formulation | P |
| 18 | Multi-input surfaces must have uniform alloc sizes | `0x1d` at eval | Allocate all inputs at max size | O |
| 19 | Multi-input surfaces ordered alphabetically | Silent wrong data | Name inputs in sorted order | O |
| 20 | ANE reads flat buffer as packed `[1,C,1,S]` | Silent wrong data | Write packed data at buffer start | O |

### 2.1 Category prose, verbatim (S2 §3)

**MIL IR Restrictions (#1, 6, 10, 12, 13, 16):**
> "The ANE compiler accepts a subset of MIL operations. Several operations that are valid in
> CoreML's MIL specification are silently rejected or produce incorrect results on the ANE.
> Most critically, the `concat` operation (#1) causes immediate compilation failure, requiring
> all multi-tensor operations to be decomposed into separate programs. The `gelu` activation
> (#10) must be replaced with its tanh approximation:
> GELU(x) ≈ 0.5x(1 + tanh[√(2/π)(x + 0.044715x³)])."

**Memory and I/O Constraints (#2, 3, 4, 8, 9, 11, 18, 19, 20):**
> "Multi-output programs require all output buffers to have identical byte sizes (#2), with
> outputs ordered alphabetically by their MIL variable names (#3). Symmetrically, multi-input
> programs require all input IOSurfaces to have the same allocation size (#18), with inputs
> also ordered alphabetically by MIL parameter name (#19). When input surfaces are
> over-allocated (padded to uniform size), the ANE reads the flat buffer as packed `[1,C,1,S]`
> data starting from byte 0, ignoring the surface's nominal dimensions (#20)."

> "There is a minimum IOSurface size of approximately 49 KB (#4), meaning single-token tensors
> with shape `[1, 768, 1, 1]` (3,072 bytes in fp16) must be padded to at least
> `[1, 768, 1, 16]` (24,576 bytes)."

> "The BLOBFILE weight format uses an offset of 64 bytes from the chunk header (#8), not from
> the file start --- an undocumented detail that causes silent weight corruption if incorrect."

**Compilation Limits (#5, 7, 14, 15):**
> "The ANE compiler maintains internal state that limits each process to approximately 119
> compilations before subsequent compilations silently fail (#5). Since weights are baked at
> compile time (#7), every training step requires recompilation of weight-bearing kernels.
> Orion v1.0 addressed this with an `exec()` restart strategy: after each training step, the
> process re-executes itself with updated checkpoint state, resetting the compilation counter
> at a cost of ~50 ms (#15)."

**Performance Characteristics (#16, 17):**
> "the ANE's convolution engine delivers ~3× better throughput for 1×1 convolutions compared
> to equivalent `matmul` operations (#17) ... However, convolutions with very large channel
> counts (e.g., 32,000 for vocabulary projection) are rejected (#16), a new finding that
> requires CPU fallback for classifier layers."

### 2.2 ⚠️ The paper's byte figures are internally inconsistent — resolved below

The paper writes `[1, 768, 1, 1]` = "3,072 bytes in fp16" and `[1, 768, 1, 16]` = "24,576 bytes".
Check:

| Shape | fp16 (2 B/elem) | fp32 (4 B/elem) |
|---|---|---|
| `[1,768,1,1]` | 1,536 B | **3,072 B** ← paper's number |
| `[1,768,1,16]` | **24,576 B** ← paper's number | **49,152 B** ← repo's "~49 KB" |
| `[1,256,1,64]` | 32,768 B | **65,536 B** (hello_mil working shape) |

**Consistent reading:** the paper's byte counts alternate between fp16 and fp32 without
saying so, but both match the repo's rule that **IOSurface allocations are sized in fp32
(4 bytes/element) while ANE compute is fp16.** The repo states this explicitly (S6):
*"fp32 I/O, fp16 compute: Use `cast()` between fp32 IOSurface data and fp16 internal
operations"* and *"ANE IOSurface buffers are sized for fp32 (4 bytes/elem)."*

**Therefore the operative minimum is 49,152 bytes of *allocated* IOSurface**, not 24,576.
Corroboration: `compiler/pass_ane_validate.c` hardcodes
`#define ANE_MIN_TENSOR_BYTES 49152` with the error string
`"node '%s': tensor size %lld bytes < 49KB minimum"`. And `hello_mil.m` uses the
known-good shape `[1,256,1,64]` = 65,536 B in fp32.

> **UNCERTAIN:** whether 49,152 B is a hard *allocation*-size floor or a
> *declared-tensor-bytes* floor. S5 says *"Minimum working size: `[768, 16]` = 49,152 bytes
> (48KB). ANE internally uses a stride of 16 for the seq dimension in padded surfaces."*
> S9 compares `orion_node_tensor_bytes(n)` (graph bytes). Probe locally.

**Practical rule for decode:** seq dimension floor is 16. From S5:
> "Use `ORION_DECODE_SEQ = 16` as the minimum decode bucket. Place token data at seq position
> 0 and zero-pad positions 1-15."

Corroborated by S6: *"Minimum tensor size: Very small tensors (e.g., [1,4,1,4]) fail at
evaluation. Use at least [1,256,1,64]."* And the paper's inference pipeline: *"Each
subsequent token is processed through the full model on ANE with a minimum sequence
dimension of 16 (to satisfy constraint #4)."*

---

## 3. Repo constraint doc (S5) — superset with symptoms and workarounds

The repo's `docs/ane_constraints.md` is the **operationally richest** source. Its 17 entries,
with the concrete numbers verbatim.

### 3.1 MIL-op-level

| # | Constraint | Exact behaviour / error |
|---|---|---|
| 1 | No `concat` | `"ANECCompile() FAILED: err=() for any program using concat(axis=1, values=(...))"`. *"All 7 training kernels using concat failed; qkvBwd (single summed output, no concat) was the only one that compiled."* Workaround: multi-output MIL programs. |
| 10 | No `gelu` | *"Using `gelu(x)` in MIL text causes `ANECCompile() FAILED`."* Decompose: `gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`. *"Implemented as `orion_mil_gelu` using `tanh`, `mul`, `add`, and `pow` MIL ops which are all supported."* Note: **`silu` also requires decomposition.** |
| 17 | No `rsqrt` | *"Any program containing `rsqrt` fails to compile, regardless of weight count — including a program with zero weights."* Workaround: `pow(x, -0.5)`. Verified: `15 conv + rsqrt -> FAILED`, `0 conv + rsqrt -> FAILED`. |

Also from repo README (S10): *"ANE MIL requires named const refs for matmul `transpose_x`/`transpose_y` — inline `true`/`false` rejected"* and *"ANE MIL `conv` does NOT support `bias=` — bias must be a separate `add` op"* and *"Output variable names must reference live nodes — dead names in return tuples cause `InvalidMILProgram`"*.

### 3.2 ⭐ Weight-budget constraints — NOT in the paper (S5 #15–#17, from @tyrauber)

These are arguably the **single most important limit for kernel fusion** and are absent from
the paper entirely.

**#15 — Maximum 16 BLOBFILE weight tensors per program.**
> "A program with 17 or more BLOBFILE-backed weight tensors fails to compile with
> `InvalidMILProgram`. 16 compiles fine."

> "**The budget counts tensors, not bytes, and not just conv weights.**"

Verbatim measured table (a — count budget, not byte budget):
```
C     bytes/weight   ceiling   total at ceiling
16    512            16        0.01 MB
64    8192           16        0.12 MB
256   131072         16        2.00 MB
768   1179648        16        18.00 MB
```
(b — shape variety is irrelevant):
```
uniform 64            -> 16    (0.12 MB)
alternating 64/256    -> 16    (0.50 MB)
cycling 64/128/256    -> 16    (0.56 MB)
FFN-like 768/3072     -> 16    (72.00 MB)
lopsided 32/1024      -> 16    (1.00 MB)
```
(c — every BLOBFILE tensor costs a full slot regardless of size):
```
conv only         -> 16 conv  (16 blobs)
conv + bias each  ->  8 conv  (16 blobs)
```
Verification runs: `14 conv -> SUCCESS`, `15 conv -> SUCCESS`, `16 conv -> SUCCESS`,
`17 conv -> FAILED`, `18 conv -> FAILED`.

> "**Practical impact:** this is the binding limit on mega-kernel fusion... A linear layer
> *with a bias* costs **two** slots, not one. Budget in blobs, not in conceptual weights:
> count every `const()` that references a BLOBFILE, including biases and norm weight
> vectors."

> "Inline constants — conv `strides`/`pad`/`dilations` attributes and similar — do not count
> against this budget. See #16 for the one case where an inline scalar does cost a slot."

**#16 — Scalar-operand ops cost one weight slot.**
> "A program containing `pow()`, `add()`, or `mul()` against an *inline scalar constant* drops
> the blob ceiling from 16 to 15. At 16 blobs plus any such op, compilation fails."

> "`mul(scalar)` carrying the same penalty as `add(scalar)` is worth noting: it means the cost
> attaches to feeding an inline scalar into an elementwise op, not to a specific opcode."

> "The penalty does **not** stack — a program using several of these still compiles at 15
> blobs. Unary elementwise ops with no constant operand (`sqrt`, `tanh`, `sigmoid`, `exp`)
> carry no penalty and compile fine at 16."

Measured:
```
16 conv + sigmoid      -> SUCCESS     16 conv + mul(scalar) -> FAILED
16 conv + tanh         -> SUCCESS     16 conv + add(scalar) -> FAILED
16 conv + sqrt         -> SUCCESS     16 conv + pow(const)  -> FAILED
16 conv + exp          -> SUCCESS
15 conv + add(scalar)  -> SUCCESS
15 conv + pow(const)   -> SUCCESS
15 conv + rsqrt        -> FAILED
16 conv + add + pow    -> FAILED     15 conv + add + pow   -> SUCCESS
conv + bias        -> 8 conv (16 blobs)
conv + bias + pow  -> 7 conv (14 blobs)
```
> "**Why RMSNorm appears to break things:** RMSNorm is built on `pow(x, -0.5)`, so any program
> containing one silently inherits the -1 penalty. This originally looked like a rule about
> 'mixing norm and linear weight types'; the real cause is the `pow()` op. There is no
> weight-type mixing rule."

> "**Consequence for activation lowering:** how an activation is expressed decides whether it
> costs budget. `SiLU(x) = x * sigmoid(x)` uses no scalar constant and stays at 16. Rewriting
> it via the identity `sigmoid(x) = 0.5*(tanh(0.5x)+1)` introduces scalar `mul` and `add`,
> dropping the ceiling to 15." → `SiLU via sigmoid -> 16`, `SiLU via tanh -> 15`.

> "**Workaround:** budget 15 blobs for any program containing a norm, or restructure to avoid
> `pow()` where an unpenalized op will do."

Provenance: *"Discovered: @tyrauber in issue #3 on M4 Max / macOS 15. Independently reproduced
on M4 / macOS 26.5.2"*, verified by `experiments/ane_weight_limit_probe.m`.

> **UNCERTAIN on A18 Pro:** whether the ceiling is 16 there is unverified. Make it a runtime
> probe, not a compile-time constant.

### 3.3 Memory / IOSurface constraints, with exact numbers

| # | Repo # | Constraint | Exact detail (verbatim) |
|---|---|---|---|
| 2 | 2 | Uniform output alloc sizes | `"ANEProgramProcessRequestDirect() Failed with status=0x1d : Program Inference error"`. *"Pad all multi-output IOSurfaces to the max channel size across all outputs."* |
| 3 | 3 | Alphabetical **output** order | *"Output surfaces arrive in alphabetical order by their MIL variable name, NOT by their position in the MIL return tuple."* Example: *"MIL returns `(q32, k32, v32)` but actual output order is `k32, q32, v32`. You must provide output surfaces as `{ioK, ioQ, ioV}`."* |
| 4 | 4 | Min IOSurface | `status=0x1d` at eval; *"Programs compile fine but fail at eval when IOSurface allocations are too small. seq=1 tensors with 768 channels = 3072 bytes -- too small."* |
| 8 | 8 | BLOBFILE offset | *"The BLOBFILE format has a 128-byte header, but the MIL `const()` weight reference offset points to the chunk header at byte 64, not the start of the file (byte 0) or end of the full header (byte 128)."* |
| 9 | 9 | `milText` type | Passing `NSString*` *"causes a crash or silent failure. The API expects raw UTF-8 bytes."* |
| 11 | 11 | Empty weight dict | *"Passing `nil` as the weight dictionary to `ANEProgramProcessRequestDirect()` causes a crash, even for programs that have no weights."* |
| 18 | 12 | Uniform **input** alloc sizes | *"even though the MIL declares them with different shapes."* Example: *"LoRA program with `x [1,768,1,32]` and `lora_A [1,768,1,16]`: allocating surfaces as `orion_tensor_create(768, 32)` and `orion_tensor_create(768, 16)` fails at eval. Both surfaces must be allocated with `orion_tensor_create(768, 32)`."* |
| 19 | 13 | Alphabetical **input** order | *"MIL declares `func main(x, lora_A, lora_B)` — declaration order. Inputs must be provided as `{lora_A, lora_B, x}` — alphabetical order by parameter name."* *"Tested all 6 permutations of 3 inputs; only alphabetical order produces correct output."* |
| 20 | 14 | Packed flat reads | *"When a MIL parameter declares shape `[1,C,1,S]` but the IOSurface is larger (e.g., allocated for `[1,C,1,S_max]`), ANE reads the first `C*S` contiguous fp16 values from the flat buffer. It does NOT use stride or padding — the data must be packed."* *"If you write data with stride-32 layout... ANE reads the wrong values."* *"Stride-padded data produced ~260x smaller LoRA contribution than expected."* |

> **Critical interaction (#18/#20 with #2/#3):** because all surfaces must be allocated at the
> *same* size but each MIL parameter declares its *own* shape, an oversized surface for a
> small tensor must be written **packed from byte 0** and the ANE will read exactly
> `C*S` elements — no stride honouring. This is the exact opposite of a padded/strided
> layout. Getting it wrong is *silent*.

### 3.4 Compilation limits

**#5 — ~119 compiles per process.**
> "After approximately 119 calls to `ANECCompile()`, the compiler silently fails or the
> process becomes unstable. **The ANE compiler leaks internal state that cannot be
> reclaimed.**"

Workaround: `exec()` restart, *"overhead is negligible (~50ms)"*. Paper §5: *"consuming 83.9%
of wall time"* was compile in v1.0.

**#7 — Weights baked at compile time.**
> "Overwriting BLOBFILE weight files on disk and reloading does NOT change model outputs. The
> weights are embedded into the compiled ANE program binary at compile time. There is no way
> to update weights without recompiling."

> ⚠️ **This repo statement contradicts the paper's delta-compilation result.** The paper
> (§5) shows that an *unload → patch BLOBFILEs on disk → reload* cycle **does** pick up new
> weights. See §5 below; the reconciliation is that the repo doc's #7 was written before
> delta compilation and describes only the "reload without unload" case.

---

## 4. ⚠️ Constraint-number mapping between paper and repo

Do not conflate these two catalogs.

| Paper # (S2) | Repo # (S5) | Same constraint? |
|---|---|---|
| 1 concat | 1 | ✅ |
| 2 uniform outputs | 2 | ✅ |
| 3 alphabetical outputs | 3 | ✅ |
| 4 min ~49 KB | 4 | ✅ |
| 5 ~119 compiles | 5 | ✅ |
| 6 SDPA masks ignored | 6 | ✅ |
| 7 weights baked | 7 | ✅ |
| 8 BLOBFILE offset 64 | 8 | ✅ |
| 9 milText NSData | 9 | ✅ |
| 10 no gelu | 10 | ✅ |
| 11 weight dict `@{}` | 11 | ✅ |
| **12 matmul named consts** | — | ❌ paper only |
| **13 conv no `bias=`** | — | ❌ paper only |
| **14 output var liveness** | — | ❌ paper only |
| 15 `exec()` overhead | — | paper only (repo mentions exec but not as numbered constraint) |
| 16 32K-channel conv rejected | — | ❌ paper only |
| 17 conv1×1 3× faster than matmul | — | paper only (repo has it as guidance, not numbered) |
| 18 uniform inputs | **12** | ✅ same, different number |
| 19 alphabetical inputs | **13** | ✅ same, different number |
| 20 packed flat reads | **14** | ✅ same, different number |
| — | **15 max 16 BLOBFILE weights** | ❌ repo/community only |
| — | **16 scalar ops cost a slot** | ❌ repo/community only |
| — | **17 no `rsqrt`** | ❌ repo/community only |

Repo README (S10) counts **23** total: *"23 constraints discovered (6 from upstream, 14 newly
documented by Orion, 3 contributed by the community)."* 20 (paper) + 3 (community) = 23 ≈ the
union. Also note the repo README lists matmul-named-consts, conv-no-bias and output-liveness
under a *"Compiler-level"* heading rather than in the numbered list.

---

## 5. Compile-time weight baking, delta compilation, program caching

### 5.1 Weight baking

S2 §2.1, verbatim:
> "Critically, the ANE *bakes weights at compile time*: weight tensors are embedded in the
> compiled program and cannot be mutated post-compilation."

### 5.2 Delta compilation (the paper's headline technique) — S2 §5

Key insight, verbatim:
> "Compiled ANE programs are managed by `_ANEModel` objects that expose `unloadWithQoS:` and
> `loadWithQoS:` methods. When a model is unloaded, its backing weight files (BLOBFILEs) on
> disk can be modified. Reloading the model picks up the new weights *without invoking
> `ANECCompile()`* --- the E5 microcode and MIL text are unchanged; only the weight data is
> refreshed. Crucially, when the MIL text and weight dictionary keys are identical, the ANE
> assigns the same `hexStringIdentifier` (a composite of three SHA-256 hashes), so the
> program's internal identity is preserved across reloads."

Algorithm 1, verbatim:
```
Require: Compiled program P with model handle M, new weight dict W'
  M.unloadWithQoS(21)                        ▷ Remove from ANE
  for each weight file path p in W' do
      Write W'[p] to disk at M.tmpDir/p      ▷ Update BLOBFILE
  end for
  M.loadWithQoS(21)                          ▷ Reload with new weights
```

> "This replaces the full compilation path: no `_ANEInMemoryModelDescriptor` creation, no MIL
> parsing, no `ANECCompile()` invocation. The implementation
> (`orion_program_reload_weights` in `core/ane_runtime.m`) handles ownership transfer of the
> temporary directory between old and new program states."

**Ownership trap, verbatim (§5.3):**
> "when the new and old programs share the same `hexStringIdentifier` (which they do, since
> the MIL text is unchanged), they share the same temporary directory on disk. The old
> program's ownership of this directory must be transferred before release, or the
> `orion_release_program` destructor will delete the shared directory, causing the new
> program to fail on its next reload."

**Where the 8.5× comes from, verbatim (§5.3):**
> "The 8.5× recompile speedup comes from avoiding three expensive operations in the full
> compilation path: (1) creating new `_ANEInMemoryModelDescriptor` objects (~3 ms/kernel for
> MIL parsing), (2) invoking `ANECCompile()` (~30–80 ms/kernel), and (3) loading a new model
> identity (~30 ms/kernel). Delta reload replaces all three with a single
> unload–write–reload cycle (~8 ms/kernel)."

Per-kernel costs from Fig. 3: v1.0 `~70 ms/kernel` compile, `~3 ms/kernel` MIL parse;
v2.0 `~9 ms/kernel` reload.

### 5.3 Program caching — S2 §4.2, verbatim

> "**Program cache.** Compiled programs are cached with composite keys (model name, layer
> index, sequence length, weight version). Cache hits skip the ~11 ms compilation overhead
> per program."

(Note: 11 ms here vs 30–80 ms/kernel for `ANECCompile()` in §5.3 and 17.1 ms observed in
`hello_mil.m`. **UNCERTAIN / workload-dependent.**)

---

## 6. Numerical behaviour

### 6.1 fp16 is the compute type

S2 §2.1: *"The ANE is a fixed-function accelerator optimized for convolution and
matrix-multiply workloads in fp16 precision."*
S2 §2.1: *"All tensor I/O uses IOSurface-backed shared memory in a fixed `[1, C, 1, S]` layout
(fp16), enabling zero-copy data transfer between the CPU and ANE."*

### 6.2 fp16 range and the overflow cascade (S2 §7.2)

> "**Root cause.** The ANE operates natively in fp16 (±65,504 dynamic range). Large
> intermediate activations overflowed to ±∞, which propagated through softmax and
> cross-entropy to produce NaN loss values."

> "**Fix: Activation clamping.** Before softmax and layer normalization, activations are
> clamped to `[-65504, +65504]`"

> "This prevents overflow without affecting well-behaved activations (which are orders of
> magnitude smaller than the fp16 limit)."

Implemented in `kernels/training/stories_train.m` as `if (v > 65504.0f) v = 65504.0f; else if
(v < -65504.0f) v = -65504.0f;` plus NaN→0.

### 6.3 Two other NaN-inducing bugs (S2 §7)

- **Bug 1 — stale programs on resume:** *"ANE programs were compiled before checkpoint
  weights were loaded."* Fix: *"Programs are now compiled after checkpoint loading... Each
  process compiles exactly once with the correct weights."* → **ordering rule: bake weights
  only after the checkpoint is final.**
- **Bug 3 — corrupted BLOBFILE weights:** *"The BLOBFILE writer produced corrupted weight data
  when checkpoint tensor layouts did not match the expected MIL weight dictionary format.
  This caused silent numerical corruption --- weights loaded without error but contained
  garbage values."* Fix: sanitize gradients before writing (`NaN → 0`, `±∞ → ±65504`) and
  validate after load by scanning for NaN/Inf.

### 6.4 Silent-wrongness is the dominant failure mode

S2 §3: *"primarily involving MIL IR compilation failures, evaluation errors, and **silent
numerical corruption**"*. Constraints #3, #6, #7, #8, #19, #20 all produce **silently wrong
data with no error**. Any new MIL program needs a numerical A/B test against a CPU reference.

### 6.5 Numerical fidelity achieved

S2 §8.1, verbatim:
> "**CPU–ANE parity.** The ANE and CPU inference paths produce *identical* token sequences.
> For the prompt "The meaning of life is," both backends generate the exact same 64-token
> greedy continuation with 100% token-level agreement."

> "ANE's ANE full-forward path achieves 170 tokens/s in decode mode, with 100% top-1 argmax
> agreement against a CPU fp32 baseline (maximum logit error: 0.073 across 12 layers)."

### 6.6 Quantization: **not supported by Orion**

S2 §9 Limitations, verbatim:
> "(5) quantization (INT8/INT4) is not yet supported"

There is **no palettization, no int4, no int8 weight path** described anywhere in the paper or
repo. The only INT8 statement is the hardware footnote: *"the ANE dequantizes INT8 to fp16
before computation... INT8 saves only memory bandwidth, not compute cycles."*

> **UNCERTAIN:** whether MIL `const` with `int8`/`int4` BLOBFILE + `dequantize`/`constexpr_lut_to_dense`
> palettization ops are accepted by `_ANECompiler` has **not** been characterized by either
> source. This is an open question for our project — needs its own probe.

---

## 7. Concrete MIL — everything needed to emit valid programs

> **Source: S6/S7/S8 (repo), not the paper.** The paper contains no MIL text.

### 7.1 Program grammar (S6, verbatim)

```
program(1.3)
[buildInfo = dict<string, string>({
    {"coremlc-component-MIL", "3510.2.1"},
    {"coremlc-version", "3505.4.1"},
    {"coremltools-component-milinternal", ""},
    {"coremltools-version", "9.0"}
})]
{
    func main<ios18>(tensor<dtype, [shape]> input_name, ...) {
        // operations
        type var = op(args...)[name = string("unique_id")];
    } -> (output_var);
}
```

Every op carries a mandatory `[name = string("...")]` attribute. Variable names are **SSA**
(assign once) and, per #3/#19, their **alphabetical order determines I/O binding order**.

Multi-output form (S8 `orion_mil_program_multi`): the return tuple is
`} -> (out_a, out_b);` and output surfaces must be supplied in **alphabetical order of the
variable names**, not tuple order.

### 7.2 Tensor layout

S6, verbatim: *"ANE tensors are always 4D: `[batch, channels, height, spatial]` → `[1, C, 1, S]`"*

S2 §4.2: *"The runtime handles the transpose between CPU-native `[seq, d_model]` and
ANE-native `[1, d_model, 1, seq]` layouts."*

### 7.3 Supported dtypes (S6, verbatim table)

| MIL Type | Bytes | Notes |
|---|---|---|
| `fp32` | 4 | IOSurface I/O type |
| `fp16` | 2 | Internal ANE compute type |
| `int32` | 4 | Shape constants, indices |
| `bool` | 1 | Flags |
| `string` | - | Op parameters |

No int8/int4/palette types appear.

### 7.4 A complete, verified minimal program (S7 `hello_mil.m`)

`z = x + y` on `[1,256,1,64]`, fp32 in → fp16 compute → fp32 out. Verbatim emitted text:

```
program(1.3)
[buildInfo = dict<string, string>({{"coremlc-component-MIL", "3510.2.1"}, {"coremlc-version", "3505.4.1"}, {"coremltools-component-milinternal", ""}, {"coremltools-version", "9.0"}})]
{
    func main<ios18>(tensor<fp32, [1, 256, 1, 64]> x, tensor<fp32, [1, 256, 1, 64]> y) {
        string to16 = const()[name = string("to16"), val = string("fp16")];
        tensor<fp16, [1, 256, 1, 64]> x16 = cast(dtype = to16, x = x)[name = string("cx")];
        tensor<fp16, [1, 256, 1, 64]> y16 = cast(dtype = to16, x = y)[name = string("cy")];
        tensor<fp16, [1, 256, 1, 64]> z16 = add(x = x16, y = y16)[name = string("add_op")];
        string to32 = const()[name = string("to32"), val = string("fp32")];
        tensor<fp32, [1, 256, 1, 64]> z = cast(dtype = to32, x = z16)[name = string("out")];
    } -> (z);
}
```

Measured on this program (M4 Max, macOS 15, `docs/ane_api_reference.md`):
*"z = x + y on [1,256,1,64] — PASS (compile 17.1ms, eval 0.223ms)"*.

Note the idiom: **declare fp32 inputs, `cast` to fp16 for compute, `cast` back to fp32 for
outputs.** IOSurfaces are fp32-sized.

### 7.5 1×1 convolution as a linear layer (S8 `orion_mil_linear`, verbatim emitted lines)

```
        string {p}_pt = const()[name=string("{p}_pt"), val=string("valid")];
        tensor<int32, [2]> {p}_st = const()[name=string("{p}_st"), val=tensor<int32, [2]>([1,1])];
        tensor<int32, [4]> {p}_pd = const()[name=string("{p}_pd"), val=tensor<int32, [4]>([0,0,0,0])];
        tensor<int32, [2]> {p}_dl = const()[name=string("{p}_dl"), val=tensor<int32, [2]>([1,1])];
        int32 {p}_gr = const()[name=string("{p}_gr"), val=int32(1)];

        tensor<fp16, [OUT,IN,1,1]> {p}_W = const()[name=string("{p}_W"),
            val=tensor<fp16, [OUT,IN,1,1]>(BLOBFILE(path=string("@model_path/weights/weight.bin"), offset=uint64(64)))];

        tensor<fp16, [1,OUT,1,SEQ]> {p}_conv = conv(
            dilations={p}_dl, groups={p}_gr, pad={p}_pd, pad_type={p}_pt, strides={p}_st,
            weight={p}_W, x={inp})[name=string("{p}_conv")];

        // bias, if present — separate add, NEVER a conv attr:
        tensor<fp16, [1,OUT,1,1]> {p}_b = const()[name=string("{p}_b"),
            val=tensor<fp16, [1,OUT,1,1]>(BLOBFILE(path=string("{bias}"), offset=uint64(64)))];
        tensor<fp16, [1,OUT,1,SEQ]> {p}_out = add(x={p}_conv, y={p}_b)[name=string("{p}_out")];

        // no bias → alias through identity:
        tensor<fp16, [1,OUT,1,SEQ]> {p}_out = identity(x={p}_conv)[name=string("{p}_out")];
```

Key structural facts:
- All `conv` attributes are **pre-declared named consts** (`strides`, `pad`, `dilations`,
  `groups`, `pad_type`) — never inline literals. This mirrors the matmul rule (#12).
- Weight tensor shape for 1×1 conv is `[out_dim, in_dim, 1, 1]`; bias is `[1, out_dim, 1, 1]`.
- `conv` has **no `bias=` parameter** (paper #13) — bias is a separate `add`.
- `pad_type` is the string `"valid"`; `pad` is a 4-element int32 `[0,0,0,0]`.
- `groups` is a scalar `int32(...)`, declared as type `int32` (not a tensor).

### 7.6 Manual SDPA decomposition (S8 `orion_mil_causal_attention`)

Required because **SDPA masks are silently ignored** (#6). Structure, verbatim:

```
        tensor<int32, [4]> {p}_rsh = const()[name=string("{p}_rsh"), val=tensor<int32, [4]>([1,{n_head},{head_dim},{seq}])];
        tensor<int32, [4]> {p}_pm  = const()[name=string("{p}_pm"),  val=tensor<int32, [4]>([0,1,3,2])];

        tensor<fp16, [1,H,D,S]> {p}_qr = reshape(shape={p}_rsh, x={q})[name=string("{p}_qr")];
        tensor<fp16, [1,H,S,D]> {p}_q  = transpose(perm={p}_pm, x={p}_qr)[name=string("{p}_q")];
        // ... same for K, V ...

        bool {p}_txf = const()[name=string("{p}_txf"), val=bool(false)];
        bool {p}_txt = const()[name=string("{p}_txt"), val=bool(true)];
        tensor<fp16, [1,H,S,S]> {p}_sc = matmul(transpose_x={p}_txf, transpose_y={p}_txt, x={p}_q, y={p}_k)[name=string("{p}_sc")];

        fp16 {p}_scv = const()[name=string("{p}_scv"), val=fp16(0.088388)];   // 1/sqrt(head_dim)
        tensor<fp16, [1,H,S,S]> {p}_scs = mul(x={p}_sc, y={p}_scv)[name=string("{p}_scs")];

        tensor<fp16, [1,1,S,S]> {p}_mask = const()[name=string("{p}_mask"),
            val=tensor<fp16, [1,1,S,S]>(BLOBFILE(path=string("@model_path/masks/causal_{seq}.bin"), offset=uint64(64)))];
        tensor<fp16, [1,H,S,S]> {p}_masked = add(x={p}_scs, y={p}_mask)[name=string("{p}_masked")];

        int32 {p}_sax = const()[name=string("{p}_sax"), val=int32(-1)];
        tensor<fp16, [1,H,S,S]> {p}_attn = softmax(axis={p}_sax, x={p}_masked)[name=string("{p}_attn")];

        tensor<fp16, [1,H,S,D]> {p}_ctx = matmul(transpose_x={p}_txf, transpose_y={p}_txf, x={p}_attn, y={p}_v)[name=string("{p}_ctx")];
        tensor<fp16, [1,H,D,S]> {p}_ctxt = transpose(perm={p}_pm, x={p}_ctx)[name=string("{p}_ctxt")];
        tensor<fp16, [1,DMODEL,1,S]> {p}_out = reshape(shape={p}_osh, x={p}_ctxt)[name=string("{p}_out")];
```

- `transpose_x` / `transpose_y` are **named `bool` consts**, not inline `true`/`false` (#12).
- Causal mask is an **additive** `[1,1,S,S]` fp16 tensor: `0` for `j <= i`, `-1e4` for `j > i`
  (not `-inf`, not a `bool` mask), added to scaled scores before `softmax`.
- `softmax` axis is a named `int32` const, `-1` here.
- Attention uses `matmul` (not SDPA) — consistent with SDPA being unusable.

### 7.7 BLOBFILE format (S6, verbatim header table)

128-byte header + fp16 weight data:

| Offset | Size | Value | Description |
|---|---|---|---|
| 0 | 1 | `0x01` | Magic byte 0 |
| 4 | 1 | `0x02` | Magic byte 4 |
| 64 | 4 | `0xDEADBEEF` | Chunk magic (LE: `EF BE AD DE`) |
| 68 | 1 | `0x01` | Version |
| 72 | 4 | varies | Data size in bytes |
| 80 | 4 | `128` | Data offset from file start |
| 128+ | N×2 | fp16 | Weight data (`_Float16`) |

Cross-checked against the writer in S8 (`orion_make_causal_mask_blob`), verbatim:
```c
int total = 128 + data_bytes;
buf[0] = 1; buf[4] = 2;
buf[64] = 0xEF; buf[65] = 0xBE; buf[66] = 0xAD; buf[67] = 0xDE;
buf[68] = 1;
*(uint32_t *)(buf + 72) = data_bytes;
*(uint32_t *)(buf + 80) = 128;
_Float16 *fp16 = (_Float16 *)(buf + 128);
```
**Two distinct offsets — do not confuse them:**
- MIL `const()` reference: `offset=uint64(64)` → points at the **chunk header** (`0xDEADBEEF`).
- Weight-dict entry `@"offset"`: `@(0)` → points at the **start of blob data**.
- Actual fp16 payload begins at **byte 128** from file start.

S6 weight-dict format, verbatim:
```objc
@{
    @"@model_path/weights/weight.bin": @{
        @"offset": @(0),      // byte offset into blob
        @"data": weightBlob   // NSData* containing the full blob
    }
}
```

### 7.8 MIL op signatures (S6, verbatim table)

| Op | Signature | Notes |
|---|---|---|
| `add` | `add(x=a, y=b)` | Element-wise add |
| `mul` | `mul(x=a, y=b)` | Element-wise multiply |
| `cast` | `cast(dtype=dt, x=a)` | Type conversion |
| `conv` | `conv(weight=W, x=a, ...)` | 1×1 conv (3× faster than matmul on ANE) |
| `matmul` | `matmul(x=a, y=b, ...)` | Matrix multiply |
| `softmax` | `softmax(axis=ax, x=a)` | Softmax (WARNING: ignores causal masks) |
| `reshape` | `reshape(shape=s, x=a)` | Reshape tensor |
| `transpose` | `transpose(perm=p, x=a)` | Permute dimensions |
| `reduce_sum` | `reduce_sum(x=a, axes=ax, keep_dims=kd)` | Sum reduction |
| `pow` | `pow(x=a, y=b)` | Element-wise power |
| `sigmoid` | `sigmoid(x=a)` | Sigmoid activation |
| `concat` | `concat(axis=ax, interleave=b, values=(a,b,...))` | **REJECTED by ANE** |
| `const` | `const()[name=..., val=...]` | Constant declaration |

Additional ops actually emitted by S8 `core/mil_builder.m` (thus proven accepted by
`_ANECompiler`): `identity`, `sub`, `reduce_mean`, `tanh`, `exp`, `sqrt`, `pow`, `mul`, `add`,
`cast`, `reshape`, `transpose`, `matmul`, `conv`, `softmax`, `const`. (`neg`, `relu`,
`reduce_max`, `split`, `pad`, `slice` are declared in the graph IR but were not observed being
emitted as MIL text by the builders read here — **UNCERTAIN whether ANE accepts them as-is.**)
Ops documented by the graph-IR table (S2 Table 4, 27 ops):
> Data: `input`, `const`, `identity` · Linear: `conv1x1`, `matmul` · Elementwise: `add`,
> `sub`, `mul`, `neg` · Activation: `relu`, `tanh`, `sigmoid` · Math: `exp`, `pow`, `sqrt`,
> `rsqrt` · Reduction: `reduce_sum`, `reduce_mean`, `reduce_max` · Shape: `reshape`,
> `transpose`, `split`, `pad`, `slice` · Other: `cast`, `softmax`, `concat_banned`

> Note the graph IR lists `rsqrt` but the ANE rejects it (#S5-17) — the IR op exists and is
> lowered to `pow(x,-0.5)`. The IR also names the banned op literally `concat_banned`.

### 7.9 Anti-patterns to avoid when emitting MIL

1. `concat(...)` → split into multiple programs / multi-output programs.
2. `gelu(x)` → expand to tanh approximation (`tanh`, `mul`, `add`, `pow`).
3. `rsqrt(x)` → `pow(x, -0.5)` (note: costs a weight slot, #16).
4. `conv(..., bias=...)` → separate `add`.
5. `matmul(transpose_x=true, ...)` → declare `bool` consts and reference them.
6. Inline scalar into `add`/`mul`/`pow` → **costs one of the 16 blob slots.**
7. Returning a variable that DCE removed → update the return tuple after optimization.
8. Small tensors (< ~49 KB allocated) → pad seq to ≥ 16.
9. Any multi-input/multi-output program where the MIL function signature order is assumed to
   be the binding order → it is **alphabetical**.
10. Writing into an oversized IOSurface with per-channel stride → ANE reads packed from byte 0.

---

## 8. Full private-API call sequence (S6 verbatim; class list from S2 Table 2)

S2 Table 2, verbatim:

| Class | Role |
|---|---|
| `_ANEClient` | Singleton connection to ANE daemon |
| `_ANECompiler` | MIL → E5 microcode compilation |
| `_ANEInMemoryModel` | In-memory model (no filesystem) |
| `_ANEInMemoryModelDescriptor` | Accepts MIL + weight blobs |
| `_ANEModel` | Holds compiled program handle |
| `_ANERequest` | Evaluation request specification |
| `_ANEIOSurfaceObject` | IOSurface wrapper for tensor I/O |

S2 §2.2, verbatim:
> "These are loaded at runtime via `dlopen()` and `objc_getClass()` from
> `/System/Library/PrivateFrameworks/AppleNeuralEngine.framework`."

(Note: the repo uses `NSClassFromString()` after `dlopen()`, not `objc_getClass()` — a minor
discrepancy between sources.)

### Step 0 — framework load
```objc
dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);
Class Desc = NSClassFromString(@"_ANEInMemoryModelDescriptor");
Class IMM  = NSClassFromString(@"_ANEInMemoryModel");
Class AR   = NSClassFromString(@"_ANERequest");
Class AIO  = NSClassFromString(@"_ANEIOSurfaceObject");
```
> "All four must be non-nil for the ANE pipeline to work."

### Step 1 — descriptor
```objc
id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(
    Desc, @selector(modelWithMILText:weights:optionsPlist:),
    milData,      // NSData* — MIL text as UTF-8 bytes (NOT NSString*)
    weightsDict,  // NSDictionary* — use @{} if no weights
    nil);         // options (unused, pass nil)
```

### Step 2 — in-memory model
```objc
id model = ((id(*)(Class,SEL,id))objc_msgSend)(
    IMM, @selector(inMemoryModelWithDescriptor:), desc);
```

### Step 3 — pre-populate the temp dir (mandatory)
> "The ANE compiler reads MIL text and weight files from a temp directory derived from the
> model's hex identifier. This directory **must** be pre-created before compilation."
```objc
id hexId = ((id(*)(id,SEL))objc_msgSend)(model, @selector(hexStringIdentifier));
NSString *tmpDir = [NSTemporaryDirectory() stringByAppendingPathComponent:hexId];
[fm createDirectoryAtPath:[tmpDir stringByAppendingPathComponent:@"weights"]
    withIntermediateDirectories:YES attributes:nil error:nil];
[milData writeToFile:[tmpDir stringByAppendingPathComponent:@"model.mil"] atomically:YES];
[weightBlob writeToFile:[tmpDir stringByAppendingPathComponent:@"weights/weight.bin"] atomically:YES];
```

### Step 4 — compile
```objc
BOOL ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
    model, @selector(compileWithQoS:options:error:), 21, @{}, &e);
```
> "**QoS 21** = high priority (always use this)"

### Step 5 — load
```objc
ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
    model, @selector(loadWithQoS:options:error:), 21, @{}, &e);
```

### Step 6 — IOSurfaces (fp32-sized!)
```objc
size_t bytes = channels * spatial * sizeof(float); // fp32: 4 bytes/elem
IOSurfaceRef surface = IOSurfaceCreate((__bridge CFDictionaryRef)@{
    (id)kIOSurfaceWidth: @(bytes),
    (id)kIOSurfaceHeight: @1,
    (id)kIOSurfaceBytesPerElement: @1,
    (id)kIOSurfaceBytesPerRow: @(bytes),
    (id)kIOSurfaceAllocSize: @(bytes),
    (id)kIOSurfacePixelFormat: @0
});
```
Note this is a **1×N linear IOSurface**, not a 2D image (`Height=1`, `BytesPerElement=1`,
`BytesPerRow=Width=AllocSize`, `PixelFormat=0`).
```objc
IOSurfaceLock(surface, 0, NULL);
memcpy(IOSurfaceGetBaseAddress(surface), data, bytes);
IOSurfaceUnlock(surface, 0, NULL);
// read: IOSurfaceLock(surface, kIOSurfaceLockReadOnly, NULL); ... IOSurfaceUnlock(...)
```

### Step 7 — request
```objc
id wIn  = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(AIO, @selector(objectWithIOSurface:), ioIn);
id req = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(
    AR,
    @selector(requestWithInputs:inputIndices:outputs:outputIndices:
              weightsBuffer:perfStats:procedureIndex:),
    @[wIn1, wIn2],   // NSArray of _ANEIOSurfaceObject inputs
    @[@0, @1],       // NSArray of NSNumber input indices (match MIL func arg order)
    @[wOut],         // NSArray of _ANEIOSurfaceObject outputs
    @[@0],           // NSArray of NSNumber output indices
    nil,             // weightsBuffer (nil for in-memory models)
    nil,             // perfStats (nil unless profiling)
    @0);             // procedureIndex (always 0 for single-function programs)
```
> ⚠️ The comment *"match MIL func arg order"* is contradicted by constraint #19 (alphabetical
> order). Trust #19 — the repo's own `ane_constraints.md` says declaration order is wrong and
> only alphabetical produced correct output across all 6 permutations tested.

### Step 8 — evaluate
```objc
ok = ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
    model, @selector(evaluateWithQoS:options:request:error:), 21, @{}, req, &e);
```

### Step 9 — unload & cleanup
```objc
((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(
    model, @selector(unloadWithQoS:error:), 21, &e);
CFRelease(ioIn); CFRelease(ioOut);
[[NSFileManager defaultManager] removeItemAtPath:tmpDir error:nil];
```
> "**Always call unload**: Failing to unload causes ANE resource leaks."

Build line used by the repo's PoC:
```
xcrun clang -O2 -fobjc-arc -framework Foundation -framework IOSurface -ldl hello_mil.m -o hello_mil
```
No special linker flag or entitlement is needed for the PoC — plain `dlopen` of the private
framework.

---

## 9. Performance numbers

### 9.1 GPT-2 124M inference (M4 Max) — S2 Table 5

| Configuration | Throughput | Latency (p50) |
|---|---|---|
| CPU decode | 283 tok/s | 3.48 ms/tok |
| ANE full forward (decode) | 170 tok/s | 5.76 ms/tok |
| ANE prefill (first call) | 12 tok/s | — |
| ANE prefill (cached) | 165 tok/s | — |

Verbatim:
> "The CPU decode path outperforms ANE decode due to the ~2.3 ms IOSurface round-trip
> overhead per ANE dispatch. This overhead is amortized during prefill (longer sequences) but
> dominates for single-token decode. ANE compilation adds a one-time cost of ~1015 ms for 24
> programs, after which cached programs achieve 165 tok/s prefill throughput."

> "First-call latency includes ANE compilation (~1015 ms for 24 programs); subsequent calls
> use cached programs."

Prefill seq-length buckets used: **32, 64, 128, 256, 512, 1024** (S2 §4.4).

> ⚠️ **Discrepancy:** `RESULTS.md` says *"ANE prefill (first call) | 1399 ms (83% compile
> time)"* while the paper says ~1015 ms and Table 5 says 12 tok/s first call. Different runs /
> metrics; treat as approximate.

### 9.2 Training — Stories110M, TinyStories, M4 Max

S2 Table 7, lr=3e-4, grad_accum=4, verbatim:

| Metric | v1.0 | v2.0 | Speedup |
|---|---|---|---|
| Train time (compute) | 908 ms | 849 ms | ~1× |
| Recompile / reload time | 4,200 ms | 494 ms | 8.5× |
| Total step time | 5,108 ms | 1,345 ms | 3.8× |
| Recompile % of step | 83.9% | 36.8% | −47.1 pp |
| 1000-step wall time | ~85 min | 22.4 min | 3.8× |
| Process model | 1 step/process | Single process | — |
| Compiles during training | 72/step | 0 | Eliminated |

S2 Table 8: throughput 0.612 TFLOPS (v1.0) → 0.656 TFLOPS (v2.0), 0 NaN / 1,000 both.

Program counts: **72 ANE programs** compiled once at startup (60 weight-bearing + 12 static
SDPA backward kernels, 6 per layer). S2 §4.5, verbatim:
> "At startup, 72 ANE programs are compiled once (60 weight-bearing + 12 static SDPA backward
> kernels, 6 per layer). This is the only compilation in the entire training run."

### 9.3 Per-operation ANE vs CPU (Stories110M) — S2 Table 9

| Operation | CPU | ANE | Speedup |
|---|---|---|---|
| Classifier fwd (embed × x) | 10.77 ms | 1.06 ms | 10.2× |
| Softmax (vocab=32000) | 81.11 ms | 2.40 ms | 33.8× |
| RMSNorm backward | 0.18 ms | 0.21 ms | ~1× |

### 9.4 CPU/ANE division of labour — S2 Table 4 (what does NOT go on the ANE)

| Operation | Device | Reason |
|---|---|---|
| Transformer fwd/bwd (dx) | ANE | Compute-bound convolutions |
| Token sampling | CPU | Sequential, branching logic |
| Adam optimizer | CPU | Weights immutable on ANE |
| ∇W accumulation | CPU | `cblas_sgemm` via GCD |
| NLL loss + gradient | CPU | `gather` not in MIL |
| Classifier backward | CPU | 32K channels rejected |
| Embedding lookup | CPU | Table indexing |

### 9.5 Stability stress test (S2 Table 6)

25 total steps, 5 chains × 5 steps, each in a fresh process:
resume chains 5, NaN/Inf **0/25**, loss monotonically decreasing 5/5,
step time **913 ± 30 ms**, throughput **0.612 TFLOPS**, `exec()` restart success **25/25**,
step-1 loss **13.975 ± 0.003**, step-5 loss **13.913 ± 0.007**.

### 9.6 LoRA adapter-as-input (S2 §6)

> `Y = XW_base + α·(XA)B`, with `W_base ∈ R^{d×d}` baked as BLOBFILE and `A ∈ R^{d×r}`,
> `B ∈ R^{r×d}` passed as **IOSurface inputs**. Hot-swap requires only changing input surface
> data — *"zero recompilation, zero program cache invalidation."*
> `orion_frontend_lora_attention` takes **8 adapter matrices** as IOSurface inputs.

Limitation, verbatim: *"LoRA inference integration is implemented for the compiler frontends
and adapter loader but not yet wired into the full Stories110M inference pipeline."*

---

## 10. Entitlements, SIP, permissions — what the sources actually say

**The paper: nothing.** Not one sentence, table entry, footnote, or reference. Confirmed by
regex over all 1,237 lines of `main.tex`.

**The repo: one parenthetical.** `docs/ane_api_reference.md` line 12, in full:
> "Returns `NULL` if framework not available (non-Apple-Silicon, SIP issues)."

That is the complete extent. Specifically **not** stated by either source:
- whether `dlopen()` of `AppleNeuralEngine.framework` requires any entitlement;
- whether `com.apple.ane.*` / `com.apple.private.*` entitlements are needed;
- whether SIP must be disabled or any binary must be ad-hoc/development-signed;
- whether `task_for_pid`, a helper daemon, or root is required;
- whether the ANE daemon (`aned`) must be reachable, or how;
- whether AMFI / hardened runtime / library validation interferes.

**What can be inferred (marked UNCERTAIN):**
- The reference implementation is a **plain CLI tool** built with
  `xcrun clang -O2 -fobjc-arc -framework Foundation -framework IOSurface -ldl` and run
  directly (`./hello_mil`), with no signing step, no entitlement plist, and no `sudo` shown
  anywhere in the build or run instructions (S7 header comment, S6, S10 "Requirements"
  section: *"macOS 15+ (Sequoia) on Apple Silicon (M1 or later) / Xcode Command Line Tools"*
  — nothing else). This strongly suggests **no entitlement is required on the tested
  configuration** (M4 Max, macOS 15). **UNCERTAIN for macOS 27.0.1 / A18 Pro / SIP enabled.**
- The `dlopen` failure mode is documented as returning `NULL`, i.e. an ordinary runtime
  failure the caller must handle — not a crash — which is consistent with a policy check
  rather than a hard AMFI block. **UNCERTAIN (inference from one sentence).**
- The paper's *declared* limitations do not mention permissions at all; limitation (1) is
  merely *"it uses Apple's private APIs, which may change without notice."*
- The repo's `Requirements` does not mention disabling SIP, so the tested path apparently
  ran with SIP at its default (enabled) state. **UNCERTAIN — the repo never states the SIP
  state of the test machine.**

> **Action item for our project:** the exact entitlement/SIP question is unresolved by the
> literature. It must be answered empirically on A18 Pro / macOS 27.0.1 by attempting the
> `dlopen` + class-resolution probe (Step 0 above) and recording `dlerror()` output. Treat a
> `NULL` return as the documented failure mode.

---

## 11. Applicability gaps for our target (A18 Pro, macOS 27.0.1)

| Item | Paper/repo basis | Risk for A18 Pro |
|---|---|---|
| 16-core ANE, H16 | M4 Max | A18 Pro is a different generation; core count/INT8 TOPS differ. Zero ANE data for A18 Pro in these sources. **UNCERTAIN** |
| 32 MB SRAM cliff | M4 Max | **UNCERTAIN**; must re-measure |
| ~19 TFLOPS fp16 | M4 Max | **UNCERTAIN** |
| Eval queue depth 127 | M4 Max | **UNCERTAIN** |
| ~119 compile limit | M4 Max, macOS 15 | Likely per-process compiler state, not per-SoC — but **UNCERTAIN** on macOS 27 |
| Max 16 BLOBFILE/tensor budget | M4 Max/macOS 15 + M4/macOS 26.5.2 | **UNCERTAIN**; probe with `ane_weight_limit_probe.m` pattern |
| 49 KB min IOSurface / seq≥16 | M4 Max | **UNCERTAIN** |
| `concat`, `gelu`, `rsqrt` rejection | M4 Max | Compiler-level, probably stable across generations, but **UNCERTAIN** |
| Alphabetical I/O ordering | M4 Max | Host-side API behaviour; **likely stable, UNCERTAIN** |
| BLOBFILE header layout | M4 Max | On-disk format; **likely stable, UNCERTAIN** |
| MIL grammar version `program(1.3)`, `func main<ios18>` | M4 Max, coremlc 3510.2.1 / coremltools 9.0 | **On macOS 27 the expected `buildInfo` versions and the `<ios18>` target string may need bumping. UNCERTAIN — highest-priority thing to verify by compiling a hello-world MIL.** |

---

## 12. Recommended verification order for our Zig engine

1. **Hello-MIL bring-up** — emit exactly the `z = x + y` program from §7.4 on `[1,256,1,64]`,
   `dlopen` the framework, resolve the 4 classes, compile with QoS 21, eval, compare to 3.0.
   This validates MIL grammar version, `buildInfo` strings, `<ios18>` target, IOSurface
   plumbing, and the entitlement/SIP question in one shot.
2. **Probe the numeric limits on A18 Pro** rather than trusting the table: min IOSurface size
   (bisect 1 KB → 64 KB), seq floor (1, 2, 4, 8, 16), BLOBFILE slot ceiling (12…20 convs,
   with and without `pow`), compile-count ceiling (~119?).
3. **Probe the rejected-op set** directly: `concat`, `gelu`, `rsqrt`, `conv(bias=)`,
   `matmul(transpose_x=true)` inline, and (open question) int8/int4 BLOBFILE + palettization
   ops.
4. **Only then** build the transformer frontends, keeping the 1×1-conv-over-matmul and
   15-blob-budget rules in mind.
5. Every new kernel gets a **CPU fp32 reference diff**, because 6 of the 20 constraints fail
   silently rather than erroring.

---

## 13. Quick-reference card

```
HARD LIMITS (M4 Max; re-probe on A18 Pro)
  ANE SRAM ................ 32 MB (30% perf cliff beyond)
  fp16 range .............. ±65504  (clamp before softmax / norm)
  eval queue depth ........ 127
  compiles per process .... ~119   (silent failure after; exec() restart ≈ 50 ms)
  min IOSurface alloc ..... ~49152 B (fp32-sized alloc; pad seq >= 16)
  BLOBFILE weights/program  16     (count, not bytes; 15 if any pow/add/mul/scalar)
  conv1x1 vs matmul ....... 3x faster -> prefer conv

ALWAYS
  cast fp32 <-> fp16 at the boundary (IOSurface is fp32-sized)
  named consts for conv strides/pad/dilations/groups/pad_type and matmul transpose_x/y
  bias via separate add (conv has no bias=)
  gelu -> 0.5x(1+tanh(sqrt(2/pi)(x+0.044715x^3)))
  rsqrt(x) -> pow(x,-0.5)          [costs a weight slot]
  manual causal attention: matmul -> +mask(0/-1e4) -> softmax -> matmul
  unload every model; pass @{} not nil; milText as NSData* UTF-8
  QoS 21 on compile / load / evaluate
  alphabetical binding order for BOTH inputs and outputs
  uniform IOSurface alloc size for BOTH inputs and outputs (pad to max)
  write packed data from byte 0 into oversized surfaces
  BLOBFILE: MIL ref offset=uint64(64); weight-dict offset=@(0); payload at byte 128

NEVER
  concat / gelu / rsqrt in MIL
  inline scalar into add/mul/pow (costs a blob slot)
  inline true/false for matmul transpose flags
  rely on SDPA causal masks
  assume tuple order == surface order
  forget to unload or reuse a shared tmpDir without transferring ownership
```

---

### Change log

- 2026-10-08 — initial extraction. Paper via LaTeX source (S2); repo at HEAD (S4–S10).
  No claim in this document about entitlements, SIP, int4 palettization, or A18 Pro is
  sourced — those are explicitly flagged UNCERTAIN or absent.
