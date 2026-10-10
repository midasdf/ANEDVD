---
name: rea-on-dsh
description: Run REA (rea-agents) from DeepSeek Harness through its global CLI instead of MCP. Load alongside reverse-engineer-anything when a task needs shipped-artifact, Mach-O/Apple bundle, JavaScript/Electron, or decompilation evidence.
whenToUse: The task needs REA (binary, app, or website reverse engineering) and no mcp__rea__* tools exist in this session, which is the normal state in this profile.
metadata:
  version: "1"
---

# REA on DSH: CLI, not MCP

`rea` (rea-agents 6.3.0, version-pinned) is installed globally and on PATH.
This profile deliberately does **not** register REA's MCP server: at 6.3.0 it
advertises 139 tools whose model-visible definitions (name + description +
input schema) total ~573 KB, and DSH's MCP client has no allowlist, so all of
them would be attached to every request.

## How to use the workflow skill with the CLI

* Read `reverse-engineer-anything` for the workflow, the evidence discipline,
  and the per-target guides; then run the CLI command instead of the MCP tool.
* **Skip its "Connect only when needed" section.** `rea doctor --client codex`
  and `rea setup --client ...` write other agents' JSON/TOML registrations.
  DSH is not one of REA's supported clients, and the CLI needs no registration.
* Every CLI invocation is a separate process with no open session, so
  session-state MCP tools (`binary_session`, `goto_address`,
  `current_procedure`, `list_procedures`, `close_binary`) have no equivalent:
  pass the target path to each command and pass `--snapshot <file>` to reuse
  analysis between calls.
* `open_binary` has no command either — the path is an argument of every
  analysis command.

## Name mapping (snake_case tool -> kebab-case command)

| REA MCP tool | CLI |
|---|---|
| `inspect_macho` | `rea inspect-macho <path>` |
| `inspect_signature` | `rea inspect-signature <path>` |
| `inspect_plist`, `inspect_keyed_archive`, `inspect_asset_catalog`, `decode_interface_builder` | same kebab-case name |
| `trace_dylib_resolution` | `rea trace-dylib-resolution <path>` |
| `analyze_javascript_application` | `rea analyze-javascript-application <path>` |
| `open_binary` + `inspect_analysis_view` | `rea inspect <path>` or `rea analyze <path>` |
| `procedure_pseudo_code` | `rea decompile <path> <address-or-name>` |
| `analyze_function` | `rea function <path> <address-or-name>` |
| `xrefs`, `find_xrefs_to_name` | `rea xrefs <path> <address>` |
| `trace_feature`, `trace_application_feature` | `rea trace <path> <query>` |
| `search_strings`, `search_procedures` | `rea search <path> <pattern> [--kind strings\|procedures]` |
| `export_evidence_bundle`, `import_evidence_bundle` | `rea evidence-export`, `rea evidence-import` |
| `binary_session` availability | `rea capabilities`, `rea providers` |

## CLI tips that matter for an agent

* `rea --llms` prints an LLM-readable command manifest; `rea <command> --schema`
  prints that command's exact input schema. Use them instead of guessing flags.
* Control output size instead of dumping it: `--format json` (default),
  `--token-limit <n>`, `--token-count`, `--filter-output a.b,c[0,3]`.
* Deep native analysis runs through **Ghidra 12.1.4** (installed 2026-10-10,
  `brew install ghidra`). `rea` on PATH in this profile is a wrapper at
  `~/.local/bin/rea` that supplies the two variables REA requires —
  `GHIDRA_INSTALL_DIR=/opt/homebrew/opt/ghidra/libexec` and the keg-only
  `JAVA_HOME=/opt/homebrew/opt/openjdk@21/libexec/openjdk.jdk/Contents/Home` —
  so `rea decompile`, `rea function` and `rea search --kind procedures` work
  without any setup. Hopper and IDA are still absent; `rea doctor --provider
  ghidra` is the readiness check.
* **Mach-O procedure names keep the leading underscore** in Ghidra: pass
  `_ane_helper`, not `ane_helper` (a bare name fails with `invalid_request`).
  Use `rea search <path> <pattern> --kind procedures` to get the exact name or
  address first.
* **A universal (fat) Mach-O is refused** with `architecture_unsupported`;
  "analyze a thinned Mach-O slice instead". Get the exact slice names with
  `lipo -archs <file>` (`aned` is `x86_64 arm64e arm64e.x1` — no plain `arm64`)
  and thin first: `lipo -thin arm64e <fat> -output <thin>`.
* Every CLI call is a new process, so Ghidra re-imports and re-analyzes the
  target each time: a 1.4 MB Mach-O costs ~60 s per query. `--snapshot <file>`
  retains results for later commands that take Evidence input, but it does **not**
  avoid that re-analysis — measured 64.8 s against 60.6 s for the same query with
  the same snapshot. Budget for the wall clock or ask several questions per run.
* These work with no provider at all: `inspect-macho`, `inspect-signature`,
  `inspect-plist`, `trace-dylib-resolution`, `decode-interface-builder`,
  `inspect-keyed-archive`, `inspect-asset-catalog`,
  `analyze-javascript-application`, `inspect-binary-layout`.
* `rea observe-native-calls` (LLDB) records function entries with integer
  arguments and backtraces, but its results are **racy**: on one test binary 6 of
  8 runs recorded every expected hit, one recorded 5, and one recorded nothing
  while leaving the target stopped for the whole window. Re-run before
  concluding a function is never called, and read `coverage.status` — a
  zero-event result is inconclusive, not negative evidence.
* Results are Evidence records with an `evidence_id`, a subject digest, and
  explicit unknowns. Keep them in files and cite the IDs; the CLI does not
  retain them between calls unless `--snapshot` is used.

## Install layout and updating

* CLI: `npm install --global rea-agents@6.3.0` installs `/opt/homebrew/bin/rea`.
  `~/.local/bin/rea` (first in PATH) is a **wrapper** that adds
  `GHIDRA_INSTALL_DIR` and `JAVA_HOME` and execs it; explicit variables still win.
  Remove the wrapper to get the bare CLI back.
* Providers: Ghidra 12.1.4 via `brew install ghidra` (pulls the keg-only
  `openjdk@21`); Hopper/IDA are not installed. Nothing else is needed for the
  provider-free operations.
* Skills live in **two roots** and must stay in step: this repository's
  `.agents/skills/` (project root, wins for sessions opened here) and
  `~/.dsh/skills/` (user root, makes them visible in every other workspace).
  Both were copied from the installed package's `skills/` directory so the
  instructions match the CLI.
* REA releases almost daily. Update deliberately: install the new exact version,
  then re-copy both skills from the new package —
  `cp -R "$(npm root -g)/rea-agents/skills/reverse-engineer-anything" .agents/skills/`
  and the same into `~/.dsh/skills/` — and re-check the tool count before
  considering MCP again.
