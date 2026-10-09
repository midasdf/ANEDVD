# Workspace notes — ANEDVD

**Apple Neural Engine Decoupled Vandal Driver** — CoreML-free LLM inference on
the Apple Neural Engine, in Zig. Started
2026-10-08 on an A18 Pro / macOS 27.0.1 machine. Read `README.md` first; the
measurements live in `docs/RESULTS.md`.

## Project

| File | Role |
|---|---|
| `src/main.zig` | CLI: `info`, `probe`, `bench`, `width`, `attnbench`, `kernels`, `verify`, `layers`, `selftest`, `check`, `run`, `chat`, `cpu`, `serve` |
| `src/tests.zig` | the test root; references every module so `zig build test` is real |
| `src/ane/shim.m` / `shim.h` | the only Objective-C + private-API code (`_ANEInMemoryModel`, IOSurfaces) |
| `src/ane/runtime.zig` | Zig kernel wrapper, planar fp16 scatter/gather |
| `src/ane/mil.zig` | MIL program generator (`conv`, `add`, `sigmoid`, `mul`) |
| `src/ane/weights.zig` | ANE weight-blob packer |
| `src/engine.zig` | transformer decode loop + kernel construction |
| `src/generate.zig` | sampling loop + prompt formatting (shared by CLI and server) |
| `src/http.zig` | minimal HTTP/1.1 server on libc sockets (std.http sits behind the new Io) |
| `src/server.zig` | OpenAI + Anthropic routes, SSE framing, stop-sequence holdback |
| `src/webui.html` | built-in chat UI, embedded with `@embedFile` |
| `src/cpu.zig` | RMSNorm, RoPE (adjacent + half-split), attention, SwiGLU, sampling |
| `src/model.zig` | format-independent `Config` / `LayerWeights` / `ModelWeights` |
| `src/load_gguf.zig`, `src/load_hf.zig` | format → runtime weights + per-layer matrices |
| `src/model_open.zig` | `--model` accepts a .gguf file or an HF directory |
| `src/gguf.zig`, `src/tokenizer.zig` | GGUF reader + 15 ggml dequantisers; byte-level BPE |
| `src/safetensors.zig`, `src/hf.zig` | HF checkpoint + config.json readers |
| `src/sys.zig`, `src/buf.zig` | libc-based file IO + LE readers; growable byte buffer |
| `probe/*.m` | standalone Objective-C probes that established the ANE facts |
| `research/*.md` | extracted Orion constraints, Espresso API notes, entitlement analysis |
| `tests/fixtures/`, `tools/` | dequant/tokenizer fixtures and the independent Python cross-checks |

```sh
zig build            # zig-out/bin/anedvd (ReleaseFast by default; -Doptimize=debug to develop)
zig build test       # unit tests (gguf/tokenizer fixtures need cwd = repo root)
zig build run -- info
zig fmt src build.zig
```

## Things that will bite you

* **Zig 0.17 churn.** `std.ArrayList` is the unmanaged flavour (`.empty` +
  `append(gpa, …)`); `std.fs`/`std.Io` need an `Io` (this project uses libc via
  `std.c` instead, in `src/sys.zig`); the `**` array-repeat operator is gone
  (`@splat`); `@src().file` is only a basename; `std.meta.intToEnum` is
  `std.enums.fromInt`. `main` takes `std.process.Init` for args.
* **Zig's bundled clang crashes on the ARC Objective-C shim**, so `build.zig`
  compiles `src/ane/shim.m` with `/usr/bin/clang` and links the object file.
* **Kernels are compiled for a fixed activation width** (`Options.chunk`, default
  128): decode fills column 0, prefill fills a whole chunk. The ANE costs the
  same either way (weights are read once), so do not "fix" this by building
  width-1 kernels — that just makes prefill 10x slower.
* **The ANE weight file needs a per-chunk `data_off`** (128, 240, …), not a
  constant. It only shows up in multi-tensor files (the fused FFN); single-conv
  kernels cannot catch it. `weights.zig` tests pin the fixture layout.
* **`zig build test` needs `src/tests.zig`.** Zig only runs tests from the test
  root and the files it references; rooting the test build at `src/main.zig`
  executed zero tests and still reported success. Add new modules to
  `src/tests.zig`.
* **One sampler for everything** (`cpu.sample`, driven by `generate.Session`):
  CLI, HTTP API and WebUI all go through it, so a sampling change lands
  everywhere at once.
* **Use `run --ab`** after touching the prefill path: it asserts the batched and
  per-token logits are bit-identical.
* **`check` only validates layer 0, kernel by kernel.** It cannot see a bug in
  the hand-off between kernels (a transposed activation layout did exactly that
  once). Use `anedvd verify <model.gguf>` for a whole-model ANE-vs-CPU compare,
  and `anedvd layers --upto N` to bisect by depth.
* **Two tokenizer families are supported** (byte-level BPE and SentencePiece),
  and the chat template comes from the GGUF `tokenizer.chat_template` metadata —
  not from probing the vocabulary, because models like TinyLlama spell their
  `<|user|>` markers as plain text in the template rather than as vocabulary
  entries.
* **MoE experts run on the CPU, not the ANE.** A 60-expert model would need
  thousands of loaded kernels and the ANE pool holds a few dozen. Attention stays on
  the ANE; the experts stream from the mapping. `research/moe-design.md` has the
  arithmetic, the verified forward pass and the measured (bad) speed.
* **Sampling is not negligible.** At temperature 1.0 the candidate cut leaves
  ~31k of 151936 logits, and ordering them was ~10% of decode (`sortUnstable`,
  not `sort`, for the top-k selection). `anedvd run` prints the cost per token.
* **Never `git checkout <path>` with uncommitted work you want to keep.** A bare
  `git checkout src` during this session silently destroyed ~2 hours of Gemma support
  (sandwich norms, logit soft-capping, sliding-window attention, chat template) that had
  been measured working on a real 2B model. Commit or stash first; there is no undo for
  an uncommitted revert.
* **Beware checks that cannot fail.** Three bugs here survived a green run: a
  test build that ran zero tests, an A/B check that compared a buffer with
  itself, and a per-kernel check blind to the hand-off between kernels. When
  adding a check, prove it fails on a deliberately broken input.
* **`Tokenizer.fromGguf` and `load_gguf.viewF16` alias the mapped GGUF file.**
  The `Gguf` must outlive them; returning either from a helper that closes the
  map is a segfault waiting for first use.
* **The ANE fails silently.** A wrong weight-blob offset, a transposed weight or
  a mis-strided IOSurface all *compile and run* and return plausible numbers.
  Always validate a new kernel against a CPU matmul (`anedvd check`).
* **Tensor layout is planar**, 64-byte minimum plane stride, and the authoritative
  strides come back from `modelAttributes` — never assume them.
* **RoPE convention is per-architecture** (llama = adjacent, qwen2 = half-split).
  The wrong one produces fluent repetition, not garbage.
* **Test the server with two requests, not one.** Both server bugs found so
  far were invisible to single-request tests: a stale-KV bug that returned
  zero tokens on the second identical request, and a request silently
  dropped with no HTTP status while a generation was running. A second
  completion is now refused with 503 + Retry-After rather than queued.
* The HTTP server handles **one request at a time** on purpose: one ANE engine,
  one KV cache. Do not add a thread pool without giving each thread its own
  engine.
* **A `anedvd cpu` run on a 2B model costs ~850 MB of swap while it runs.** A long
  reference run started in the background and then abandoned held that much for a long
  time. `anedvd cpu` is scalar and takes many minutes at that size, so put a timeout on
  it and check for leftover processes (`pgrep -fl anedvd`) before blaming the machine for
  pressure.
* **The ANE program pool is machine-wide**, so only one process can hold a full
  model's kernels. Killing a stale `anedvd serve` is usually what fixes a
  sudden "no ANE resources" failure.
* Weights are streamed per layer and the ANE weight files are deleted after
  load; `ANEDVD_KEEP_ANE_FILES=1` keeps them for debugging.
* Models are not committed (`models/` is gitignored); re-download a small GGUF
  (SmolLM2-135M-Q8_0, Qwen2.5-0.5B-Q8_0) to run end to end.

## Toolchain

| Tool | Version / path |
|---|---|
| Zig | 0.17.0, `/opt/homebrew/bin/zig` |
| zls | 0.17.0-dev, `/opt/homebrew/bin/zls` |
| clang | 21.0.0, `/usr/bin/clang` (CommandLineTools) |
| python3 | 3.9.6 (fixture generators, cross-checks) |

## DSH dev environment

`~/.dsh/profiles/desktop/cordis.patch.yml` adds one `insert:` block with six
rows: `dev-lsp`, `dev-lsp-stdio`, `dev-tool-lsp`, `dev-terminal`,
`dev-terminal-bash`, `dev-tool-terminal`. The four packages behind them
(`@deepseek-ai/dsh-lsp`, `-lsp-stdio`, `-tool-lsp`, `-tool-terminal`, all
`0.2.0-rc.2`) live in `~/.dsh/profiles/desktop/node_modules`.

* `lsp` — read-only `goToDefinition` / `findReferences` / `goToImplementation` /
  `hover`, one-based line and UTF-16 column, workspace files only. `.zig`/`.zon`
  → zls, `.c/.h/.cc/.cpp/.hpp/.m` → clangd, `.rs` → rust-analyzer, `.go` →
  gopls, `.py/.pyi` → pyright, `.ts/.tsx/.js/.jsx/.mjs/.cjs` → tsserver.
* `terminal_open/send/read/signal/close/list` — a real PTY that survives across
  calls (`run_in_background` for long commands).
* `plugin_manager` — list/install/remove plugins and bundles for this profile;
  every action needs `danger-full-access`. The profile patch carries a
  last-write `disabled: false` override for `tool-plugin-manager`.

Caveats: `dev-lsp-stdio` resolves every configured `command` at load, so one
missing executable stops the whole provider; the profile patch is the last
layer, so overrides here win; new plugins need an `insert:` patch entry (a bare
`- id:` only overrides an existing row); changes reach newly created
agents/sessions only.

## Gemma support: implemented, still producing wrong text

All of the pieces below are implemented and committed (stages 1-3). What is NOT
working is the generated text: gemma-2-2b-it loads, checks OK and its prompt encodes
correctly (`<bos><start_of_turn>user\n...`), but generation emits a run of "." instead
of an answer. Treat the model as unsupported until that is fixed.

Implemented and verified to the extent stated:

* `Config.norm_unit_offset` — Gemma's RMSNorm is `(1 + w)`. Confirmed against
  `GemmaRMSNorm.forward`; baked into the weights in `loadNormFor` so no compute path
  changed. A test pins the equivalence.
* `Config.embed_scale` — `sqrt(hidden_size)`, applied at both embedding reads. The GGUF
  stores UNSCALED weights (measured mean|w| ~ 1e-5 against an HF std of ~0.01), so
  llama.cpp does not pre-apply it.
* `Config.attn_logit_softcap` / `final_logit_softcap` — `softcap * tanh(x / softcap)`,
  read from the file (50.0 and 30.0).
* `Config.sliding_window` / `layerIsSliding` — alternating global/windowed attention,
  read from the file (4096).
* `Norm.post_attn` / `Norm.post_ffw` — the sandwich norms, applied at all four residual
  points (decode and prefill, attention and FFN).
* A `<start_of_turn>` chat template selected from `tokenizer.chat_template`, and Gemma
  added to the HF architecture list.

**Where it stands now.** The `WrongShape` is fixed (prefill was reading the whole
vocabulary from one head chunk). All four of gemma-2-2b's kernels check OK against the CPU
reference (qkv 2.0e-4, o 4.2e-4, ffn 4.1e-3, lm_head 2.1e-4), its prompt encodes correctly,
and the tiny-random Gemma 2 checks OK.

`anedvd run` on gemma-2-2b still emits a run of "." rather than an answer. The defect is
**reachable in a single layer**:

    anedvd verify models/gemma-2-2b-it-q4_k_m.gguf --layers 1
      prefill: max|ANE - CPU| = 4.26e1  argmax 2 "<bos>" vs 603 "▁is"   MISMATCH

so it is not something that accumulates over 26 layers, and `--layers 1` is a
two-minute reproduction rather than a full run. Note the engine and `anedvd cpu` agree
with each other on the tiny Gemma 2 but not on the real one, so the divergence is
size-dependent: `head_dim` 256 against 32, `q_dim` 2048 against `hidden` 2304, and a
4096 sliding window that neither prompt actually reaches.

**Correction to an earlier claim in this file.** I wrote that `anedvd cpu` also produced
wrong text, making the bug shared. That conclusion came from a hand-built prompt containing
a literal `<bos>` string, which is not a fair test of the reference. On a plain prompt the
reference gives `603 476 575 573 919` — " is a to in", a plausible continuation — while the
engine gives `235269 235265 ...` — ", ." — which is not. So the reference looks right and
the engine wrong, and the bug is engine-side after all. The lesson is the same one this
file keeps recording: check what a test actually exercises before concluding from it.

**Where the bug is, narrowed to one subsystem.** `check`/`diagnose` never apply RoPE
or attention — it runs embed -> attn_norm -> qkv -> o -> ffn_norm -> ffn -> head and
nothing else. Yet `verify --layers 1` on the one-token prompt "The" already disagrees with
the reference (`9` against `714`). Since every projection kernel passes while the chain
fails, and the only things the chain adds are RoPE, the KV cache and attention, the defect
is in one of those three. That is a much smaller surface than the whole model, and
gemma-2's rotate_half RoPE and half-split convention both check out on inspection, so
attention itself is the first place to look.

Established by measurement, so they can be skipped:

* `head_dim` is 256 for gemma-2 and 32 for the tiny model. The prefill/decode attention
  equivalence tests now cover 256 and pass, so attention is consistent at that size.
* Disabling the sliding window changes nothing (neither prompt reaches 4096).
* The o projection is built `cin = qDim (2048) -> cout = hidden (2304)`, correct for a
  model whose q_dim is smaller than its hidden.
* Every Gemma flag loads: sandwich, unit-offset norm, gelu, both soft-caps, the window,
  and layer 0's four norms all arrive at hidden = 2304.
* Q6_K, the type of Gemma's embedding tensor, is bit-exact against llama.cpp.

Open questions for whoever picks this up, in the order worth trying:

0. The `WrongShape` above. It is confined to `runHead`'s chunk loop: the kernels all
   check OK, so it is the read, not the arithmetic.
1. `query_pre_attn_scalar` is 256 for gemma-2-2b, equal to `head_dim`, so the existing
   `head_dim**-0.5` scaling matches. Check this for other Gemma sizes.
2. `q_dim` (8 x 256 = 2048) is smaller than `hidden` (2304). Nothing has been verified
   about the `o_proj` shape beyond it loading.
3. The "sandwich" ordering was applied as `norm(attn_out) + residual`, which is what the
   reference does, but it has never been checked numerically against a reference
   implementation for this model — `anedvd cpu` and the engine share the code, so they
   agree with each other and cannot validate it.
4. Prefill attention masks the window only within the chunk; whether a chunk boundary
   needs care when `pos` wraps the window has not been tested.
