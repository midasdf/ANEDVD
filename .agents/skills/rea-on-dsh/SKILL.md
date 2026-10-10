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
* Deep native analysis needs Hopper, Ghidra, or IDA. **None is installed on
  this machine**, so check `rea providers` before promising pseudocode. These
  work with no provider: `inspect-macho`, `inspect-signature`, `inspect-plist`,
  `demangle-swift`, `trace-dylib-resolution`, `decode-interface-builder`,
  `inspect-keyed-archive`, `inspect-asset-catalog`,
  `analyze-javascript-application`, `inspect-binary-layout`.
* Results are Evidence records with an `evidence_id`, a subject digest, and
  explicit unknowns. Keep them in files and cite the IDs; the CLI does not
  retain them between calls unless `--snapshot` is used.

## Install layout and updating

* CLI: `npm install --global rea-agents@6.3.0` (`/opt/homebrew/bin/rea`).
* Workflow skill: `.agents/skills/reverse-engineer-anything/`, copied from the
  installed package's `skills/` directory so the instructions match the CLI.
* REA releases almost daily. Update deliberately: install the new exact
  version, then re-copy the skill from the new package —
  `cp -R "$(npm root -g)/rea-agents/skills/reverse-engineer-anything" .agents/skills/` —
  and re-check the tool count before considering MCP again.
