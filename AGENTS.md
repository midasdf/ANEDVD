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
* **"CPU is a small part of prefill" is only true for short prompts.** At 374 tokens CPU
  attention alone is 0.79 ms/token against a 1.74 ms/token prefill — roughly 45%, up from
  ~14% at 26 tokens, because attention is O(n²) while the ANE part is per-layer. Measure at
  the context length you care about; `anedvd attnbench` isolates it.
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
* **A KV cache is `max_seq` rows and nothing checked that.** `verify` used a fixed 1024-row
  context and its CPU reference a hardcoded `1024 * cfg.kvDim()`, so a 1121-token prompt
  wrote past both — and returned all-zero logits with `rel = 1.0` rather than crashing.
  `Engine.prefill`/`forward` now return `error.ContextOverflow`, and every caller that takes
  an arbitrary prompt sizes its context from it (`verify`, `layers`); `generate` truncates.
  If you add a command that calls the engine directly with a user-supplied prompt, size the
  context to it. `verify` is the ground truth for every numerical claim here, so a silent
  overflow above its context made those claims unverifiable rather than obviously wrong.
* **The ANE fails silently.** A wrong weight-blob offset, a transposed weight or
  a mis-strided IOSurface all *compile and run* and return plausible numbers.
  Always validate a new kernel against a CPU matmul (`anedvd check`).
* **Tensor layout is planar**, 64-byte minimum plane stride, and the authoritative
  strides come back from `modelAttributes` — never assume them.
* **RoPE convention is per-architecture** (llama = adjacent, qwen2 = half-split).
  The wrong one produces fluent repetition, not garbage.
* **A readiness check must mean "the whole request arrived", not "some bytes did".** The
  server used `hasPendingInput` (poll for POLLIN) before handing a connection to
  `readRequest`, whose body loop is a blocking `recv` bounded only by `SO_RCVTIMEO`
  (30 s). A client that sent headers claiming `Content-Length: 5000` and then 40 bytes
  therefore blocked the single service path and made `/health` time out for **every**
  other client until the timeout expired. `hasCompleteRequest` (FIONREAD + MSG_PEEK) now
  gates it, and there are four parking slots rather than one. Probe this class of bug with
  a raw socket: claim more than you send, then check that other clients still get answers.
* **A failure after the response has started has no status left to use.** Headers go out
  before prefill so a long prompt never looks like a dead server, which means a later
  `generate` error can only be reported in-band. Every route used to propagate it out of
  the handler: non-streaming closed with no reply at all, streaming left a `200 OK` with an
  empty body that a client cannot tell from success. Both now report properly. Also clamp
  what you accept — `max_tokens` above the context produced exactly that silent 200.
  Probe by forcing the failure path, not by reading the code.
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

## Gemma 2 works (rounds 8-10)

Verified end to end: "The capital of France is **Paris**", "2 + 2 = 4", "The sun rises in
the **morning**", and the chat template answers correctly too.

What it needed, all implemented and committed:

* `Config.norm_unit_offset` — Gemma's RMSNorm is `(1 + w)`. **Per-file, not
  per-architecture**: an HF checkpoint stores the raw parameter (mean ~0.19 on
  gemma-2-2b) and needs the offset added; the GGUF converter has already applied it
  (measured mean 1.1927 on `blk.0.attn_norm.weight`, the effective factor). Applying it
  twice turns "Paris" into a run of dots. `load_hf` sets it, `load_gguf` clears it.
* `Config.embed_scale` — `sqrt(hidden_size)`, applied at all three embedding reads.
* `Config.use_gelu` — Gemma's FFN is `gelu_pytorch_tanh`, not SiLU. At x = -3 the two
  differ 40x.
* `Config.attn_logit_softcap` / `final_logit_softcap` — `softcap * tanh(x / softcap)`.
* `Config.attn_scale` — `1/sqrt(n_embd/n_head)`, NOT `1/sqrt(head_dim)`: 1/sqrt(288)
  against 1/sqrt(256) for gemma-2-2b, a 6% difference (llama.cpp's gemma2 graph).
* `Config.sliding_window` / `layerIsSliding` — the reference builds `layer_types` as
  `sliding if (i + 1) % 2`, so **layer 0 slides**.
* `Norm.post_attn` / `Norm.post_ffw` — the sandwich norms, in the engine AND the CPU
  reference (which was missing them entirely).
* A `<start_of_turn>` chat template, and Gemma in the HF architecture list.

Two lessons worth keeping:

* **A wrong reference is worse than no reference.** `verify` reported MISMATCH for a
  working model twice over: the CPU path skipped the sandwich norms, and the comparison
  ran on soft-capped logits where tanh saturation hides everything. `Engine.pre_softcap`
  now gives `verify` the pre-cap values, where the numbers mean something.
* **Measure the file, do not infer the convention.** Four rounds went into reading
  transformers, llama.cpp's graph builder and the converter (whose model classes are not
  in the file that script downloads). One measurement — the stored norm mean is 1.19, not
  0.19 — settled it in a minute.

Gemma does not yet pass `verify` (rel 0.45 prefill / 0.61 decode against a 10% bar), so a
numerical difference between the two paths remains. What is known about it, so the next
attempt does not repeat the search:

* The engine is **internally consistent**: `run --ab` reports `max|diff| = 0.000000e0`
  between prefill and decode, so this is engine-vs-reference, not prefill-vs-decode.
* **Every kernel is correct at layer 0 on a real prompt token**: `check` now diagnoses
  token 603 (from "The capital of France is") instead of the hardcoded id 1, and reports
  qkv 2.7e-4, o 4.0e-4, ffn 4.2e-3, lm_head 2.1e-4 — OK, exit 0. Each of the three head
  chunks is fine individually too.
* A depth sweep says the error is largest with **zero** layers and one layer *reduces* it:

      --layers 0   rel 0.4536   (no layer runs at all)
      --layers 1   rel 0.0469   (passes)
      --layers 2   rel 0.1190
      --layers 3   rel 0.1395

  So it originates in the path before any layer — embed -> final_norm -> head — rather than
  accumulating through the stack. `check` exercises that same path and passes, so look for
  what differs between `diagnose` and `Engine.prefill` for the last column.
* The tiny Gemma 2 (hidden 8) passes the same comparison at rel 4.1e-3, so whatever it is
  scales with model size.
* **Correction to the head-input comparison I recorded earlier.** I compared the engine's
  `head_in` (232.19) against the reference's 227.34 and called it fp16 rounding. Those are
  different tokens: `verify` runs prefill, then a decode step, then the reference loop, so the
  engine's buffer holds the DECODE step's input, whose matching reference line is 228.37.
  Corrected, the difference is 1.67%, and fp16 storage can account for only ~0.024% (10-bit
  mantissa, values around 4.8). **So the head input genuinely differs and the cause is
  upstream of it** — which rules out "the engine stores the same numbers less precisely" and
  points at the embedding or the final-norm application.
* The depth sweep reproduces exactly with the current binary (0.45360 vs 0.46882e-1 for
  `--layers 0` and `1`), so the earlier figures were not affected by the KV-cache overflow
  fixed in rounds 23-25: those runs used a 6-token prompt, far below any context limit.
