# Workspace notes — ANEDVD

**Apple Neural Engine Decoupled Vandal Driver** — CoreML-free LLM inference on
the Apple Neural Engine, in Zig. Started
2026-10-08 on an A18 Pro / macOS 27.0.1 machine. Read `README.md` first; the
measurements live in `docs/RESULTS.md`.

## Response language

**Answer the user in Japanese.** Everything the user reads — explanations,
plans, progress notes, questions, and final answers — is written in Japanese,
whatever language the code, commits or repository docs use. Code, identifiers,
file contents and command output stay as they are (English), unless the user
asks for something else.

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
* **Grouped-query attention's redundant K/V reads are free — measured, do not "optimise"
  them.** Each K/V row is converted once per head in its group, which looks wasteful, but
  `attnbench --kv-heads 2` against `--kv-heads 14` shows the grouped case is *faster* on
  decode (208.4 against 220.3 us) and identical on prefill (267.90 against 268.67 us/query):
  grouping shrinks the cache being re-read (262 KB against 1.8 MB), so the redundancy is
  served from a resident cache while the ungrouped case streams seven times more data.
* **Sampling is not negligible, and the cost tracks the CANDIDATE count, not the vocabulary.**
  At temperature 1.0 the cut leaves ~45k of 151936 logits and sampling was 1.33 ms/token,
  against 0.13 ms when it leaves 33 — so the fixed two-pass walk over the vocabulary is 0.13 ms
  and everything else was candidate processing. `selectTopK` used to `sortUnstable` the whole
  candidate array to keep 40; it now keeps the k largest in `items[0..k]` and leaves the rest
  unordered, which is 0.47 ms/token. `anedvd run` prints the cost per token. **When changing
  this, run `selectTopK agrees with a full sort` — it caught two candidate-dropping bugs in the
  first two attempts at the bounded selection.**
* **The sampler's remaining cost, and the design for it (not implemented).** After the bounded
  selection, sampling is 0.47 ms/token at temperature 1.0 of which 0.13 ms is the two
  full-vocabulary walks; the rest is pass 2 materialising every candidate (`scratch[n] = ...`
  for ~45k of them, ~360 KB of writes) so that `selectTopK` can then pick 40. The top-k could
  be maintained during pass 2 instead, writing at most `k` entries. The kept set is provably
  the same either way — it is `{v >= cut}` intersected with the top `k` of all logits, and that
  intersection does not depend on the order the two are applied in.
  Three details to get right, which is why it is deferred rather than done:
  1. `top_k == 0` disables the selection and must keep materialising every candidate.
  2. When `n <= top_k` the old code leaves the candidates in **index order**, not logit order,
     and the sampler walks them cumulatively — so reordering changes which token a given seed
     draws even though the distribution is unchanged. Match the old layout: maintain the top-k
     by logit, then re-sort by index when `n <= top_k` (n is at most `top_k`, so it is cheap).
  3. Ties: `selectTopK` orders equal logits by its insertion rules. Equal logits have equal
     probability, so the distribution is unaffected, but a fixed-seed A/B will differ. Compare
     distributions, not token ids, when validating this.
  **Tried and reverted (round 38).** The implementation above passes all 125 tests and leaves
  greedy output unchanged, but the gain is only 0.47 -> 0.40-0.43 ms/token across four runs —
  about 0.05 ms, 0.2% of decode. The 45k writes were cheaper than I assumed; streaming stores
  do not cost what a counted array write suggests. The interleaved A/B that would have
  confirmed it failed to run (my shell chain broke on `grep -c` returning 1 for zero matches),
  so the gain is measured but not confirmed to this project's standard, and 0.2% does not buy
  the extra branch, comparator and order-restoring sort.
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
* **`swa_pattern` was hard-coded to 2 and the key that changes it was read nowhere — fixed in
  round 42; what follows is the record of the fault.**
  Gemma 2 alternates 1:1, so 2 is right for it — but Gemma 3 uses
  `sliding_window_pattern = 6`, i.e. five sliding layers then one global, and the HF
  reference computes `layer_types` as `sliding if (i + 1) % pattern` — the same shape my
  `layerIsSliding` implements, just with a different constant. A Gemma 3 GGUF therefore gets
  Gemma 2's pattern. This is the same fault as Mistral's window: recognised in
  `known_architectures`, wrong in the attention mask.
  To fix, four small pieces:
  1. `hf.Config.sliding_window_pattern: u32 = 0` plus its JSON read, and
     `.swa_pattern = if (c.sliding_window_pattern > 0) c.sliding_window_pattern else 2` in
     `toModelConfig`.
  2. `load_gguf`: read `attention.sliding_window_pattern` and assign `cfg.swa_pattern`.
  3. **Nothing to do, and I got this wrong twice.** `Gemma3ForCausalLM` is already in
     `hf.supported_architectures`, and `loadConfig` rejects anything not on that list with
     `error.UnsupportedArchitecture`, so an HF Gemma 3 loads legitimately. `toModelConfig` then
     maps every `Gemma*` to `"gemma2"` (a prefix match) — which is exactly why the pattern had
     to come from the config, and it now does. My predecessor notes said this failed loudly and
     that gemma3 was missing from the list; both claims were false, and I only checked by
     reading the list rather than reasoning about the prefix match.
  4. A test mirroring "Mistral windows every layer; Gemma 2 alternates": with pattern 6, layers
     0-4 slide, layer 5 does not, layer 6 slides.
  **All four are resolved** (round 42): the pattern is parsed from both formats, `toModelConfig`
  uses it with 2 as the Gemma 2 default, `load_gguf` clears `swa_all` when an explicit pattern
  is present, and `load_hf.test` pins pattern 6 through `toModelConfig` (126 tests). Piece 3
  needed nothing, as above.
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

## REA (reverse engineering)

REA is installed as a **CLI only** — `rea` (rea-agents 6.3.0, pinned at
`/opt/homebrew/bin/rea`), with its workflow skill in `.agents/skills/`:

* `reverse-engineer-anything` — copied from the installed package (version-matched).
* `rea-on-dsh` — how to drive REA from here: the MCP-tool → CLI-command mapping,
  `--snapshot` instead of an open session, output-size flags, and an instruction
  to skip `rea setup --client <other agent>` (DSH is not one of REA's clients).
  Read it before any REA work.

The MCP server is deliberately **not** registered: 6.3.0 advertises 139 tools and
~573 KB of model-visible input schemas, and `dsh-mcp-client` has no allowlist, so
every request would carry all of them. Apple static analysis needs no provider
(`rea inspect-macho`, `inspect-signature` — it returns `entitlements`,
`trace-dylib-resolution`, `demangle-swift`); `rea decompile` / `rea function`
need Hopper, Ghidra or IDA, and none is installed — check `rea providers` first.

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
* **Ruled out by reading the code, so a next attempt can start further along.** At
  `--layers 0` the whole path is embed -> final_norm -> head, and both sides should be
  identical apart from the head input being f16 in the engine:
  * `viewF16` rejects anything that is not f16 (`if (t.ttype != .f16) return null`), and
    gemma-2-2b's embedding is Q6_K, so the engine falls back to `loadLinear` exactly as the
    reference does. Same dequantiser, same values.
  * `rmsnormColumn` (engine) and `cpu.rmsnorm` (reference) compute the same expression —
    same `acc`, same `inv`, same `x * inv * w` — the only difference being the strided read
    and the f16 output, which is 0.024%.
  * Both `final_norm` vectors go through `loadNormFor` with a config derived from the same
    file, so the weights match.
  Since f16 storage accounts for 0.024% and 1.67% is observed, one of these three
  assumptions about what each side computes at `--layers 0` must still be wrong.
* **Printed the pre-norm `x` from both sides, and the embedding read is the suspect.**
  Temporary prints in `Engine.forward` and `refForward` after the embedding loop, run with
  `verify --layers 0`, gave for the same token:

      engine  x[0]  8.876953e-1   x[1] -3.944092e-1   x[2] 3.058594e0   x[3] -2.244242e1
      ref     x[0]  8.876953e-1   x[1] -3.944092e-1   x[2] 3.058594e0   x[3]  9.865723e-1

  The first three channels agree to the last printed digit and the fourth does not, and no
  norm runs before this point, so the difference is in the embedding write rather than the
  final-norm application.

  **The print indices are correct**, checked afterwards: `self.x` is allocated
  `hidden * ch` (`src/engine.zig`), so `self.x[3 * ch]` really is channel 3 of column 0, and
  the reference's `x[3]` is the same element. So the per-channel comparison stands: channels
  0-2 identical, channel 3 different.

  **What is left, and why this was deprioritised.** Both sides load the embedding with the
  same call — `viewF16` rejects gemma's Q6_K tensor so the engine falls back to
  `loadLinear(allocator, g, "token_embd.weight", cfg.hidden, cfg.vocab)`, which is exactly
  what `loadWeights` does — so the values should be identical and channel 3 should match.
  Something in that chain is still not what it looks like, and the next step is to compare the
  embedding rows directly (print `embed[token * hidden + 0..4]` from both sides before any
  scaling) rather than through `x`.

  This has no user-visible effect: gemma-2-2b answers correctly ("The capital of France is
  **Paris**", "2 + 2 = 4", the chat template), `check` passes on a real prompt token, and the
  engine's prefill and decode agree exactly. It is a numerical discrepancy with the CPU
  reference in a model that works. After seven rounds on it, the remaining value is in the
  record above rather than in more searching.
