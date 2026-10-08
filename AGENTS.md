# Workspace notes — ANEDVD

**Apple Neural Engine Decoupled Vandal Driver** — CoreML-free LLM inference on
the Apple Neural Engine, in Zig. Started
2026-10-08 on an A18 Pro / macOS 27.0.1 machine. Read `README.md` first; the
measurements live in `docs/RESULTS.md`.

## Project

| File | Role |
|---|---|
| `src/main.zig` | CLI: `info`, `probe`, `bench`, `width`, `attnbench`, `selftest`, `check`, `run`, `chat`, `cpu`, `serve` |
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
  64): decode fills column 0, prefill fills a whole chunk. The ANE costs the
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
* **The ANE fails silently.** A wrong weight-blob offset, a transposed weight or
  a mis-strided IOSurface all *compile and run* and return plausible numbers.
  Always validate a new kernel against a CPU matmul (`anedvd check`).
* **Tensor layout is planar**, 64-byte minimum plane stride, and the authoritative
  strides come back from `modelAttributes` — never assume them.
* **RoPE convention is per-architecture** (llama = adjacent, qwen2 = half-split).
  The wrong one produces fluent repetition, not garbage.
* The HTTP server handles **one request at a time** on purpose: one ANE engine,
  one KV cache. Do not add a thread pool without giving each thread its own
  engine.
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
