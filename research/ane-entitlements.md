# ANE entitlements & unprivileged access — research reference

**Question:** can an unprivileged, non-Apple-signed process drive the Apple Neural Engine
directly (no CoreML) on modern macOS — specifically macOS 27.0.1 (26A434) on Apple A18 Pro,
SIP enabled, ad-hoc/unsigned binaries only?

**Status:** answered, with **direct measurement on the target machine** plus a literature review.
Compiled 2026-10-08 (Asia/Tokyo).

---

## 0. Bottom line

1. **YES — it works.** On this machine (macOS 27.0.1 build 26A434, Apple A18 Pro, SIP **enabled**,
   euid 501, main executable **ad-hoc signed with zero entitlements**) I compiled, loaded and
   **evaluated** a hand-written MIL program on the ANE end-to-end:
   `compileWithQoS:options:error:` → `True` (28 ms), `loadWithQoS:options:error:` → `True` (18 ms),
   `evaluateWithQoS:options:request:error:` → `True` (0.3 ms), correct `relu` output on 4096 fp16
   values (0 mismatches). See §7 for raw output.
2. **The kernel entitlement is real but you never need to hold it.** `com.apple.ane.iokit-user-access`
   gates opening the `H11ANEIn` user client. Only `aned` and `aneuserd` hold it (verified in their
   launchd plists on this machine). The daemon performs the privileged open **on your behalf** and
   returns a program handle; your process talks to the daemon over XPC (Mach service
   `com.apple.appleneuralengine.private`) and dispatches on the returned program.
3. **You cannot claim that entitlement yourself.** Ad-hoc signing a binary with
   `com.apple.aned.private.allow` / `com.apple.ane.iokit-user-access` embeds them in the signature
   but the process is **SIGKILLed at launch** (measured: exit 137, `Killed: 9`). SIP disable is
   neither required nor attempted here.
4. **SIP stays enabled; no entitlement, no sudo, no special signing.** Locally built / ad-hoc signed
   binaries are all that is needed. (On Apple Silicon every arm64 binary must carry *at least* an
   ad-hoc signature to exec at all — "completely unsigned" is not a reachable state.)
5. **Practical path for the Zig engine:** `dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine")`
   → `_ANEInMemoryModelDescriptor` → `_ANEInMemoryModel` → compile/load/eval. §8 has the exact
   call sequence, QoS value, and the pitfalls to design around.

---

## 1. Machine context and method

| Fact | Value |
|---|---|
| OS | macOS 27.0.1, build `26A434` (`sw_vers`) |
| Kernel | `Darwin 27.0.0 … RELEASE_ARM64_T8140 arm64` |
| SoC | Apple A18 Pro (A-series, **not** M-series) |
| SIP | `csrutil status` → **enabled** |
| Test process | `euid 501`, non-root |
| Test binary | copy of CLT Python 3.9.6, re-signed **ad-hoc** (`codesign -s - --force`), `TeamIdentifier=not set`, **no entitlements** |
| Probe mechanism | Python `ctypes` + `libobjc` (`objc_getClass` / `objc_msgSend`) — no compiler, no Zig, no workspace writes |

Method: (a) read the primary literature, (b) inspect the local system (frameworks, daemons,
launchd plists, code signatures, daemon logs), (c) run a minimal end-to-end probe on the target
machine. Everything in §7 is *measured here*; everything else is *cited*.

Probe scripts (kept outside the workspace, re-runnable):
`/tmp/ane-research/probe.py` (compile+load), `/tmp/ane-research/probe2.py` (compile+load+eval with
IOSurface I/O), `/tmp/ane-research/pyfw/Versions/3.9/bin/python3.9` (the ad-hoc interpreter).

---

## 2. Q1 — the arXiv paper: *Apple Neural Engine: Architecture, Programming, and Performance*

Source: **arXiv:2606.22283v1**, Spencer H. Bryngelson, submitted 21 Jun 2026, 302 pp.
<https://arxiv.org/abs/2606.22283> · full text <https://arxiv.org/html/2606.22283v1>
(measurements in the paper are on **M1/H13** and **M5**; quotes below are verbatim from the
HTML v1 text, retrieved 2026-10-08).

### 2.1 The kernel gate (H11ANEIn / ANEServices user client)

> "The client open checks the two kernel entitlements of Listing 61, the hard device-open gate and
> the resident data-chaining gate.
> `/* checked at H11ANEInUserClient::init via copyClientEntitlement: */`
> `"com.apple.ane.iokit-user-access"    /* the hard device-open gate */`
> `"com.apple.ane.allow-dataChaining-access"  /* resident data-chaining gate */`
> **A single kernel entitlement gates opening either user client.** The check runs once at client
> construction and is a boolean on the client object, not re-checked per selector. Across the whole
> system, exactly two binaries hold `com.apple.ane.iokit-user-access`: the system broker daemon and
> its per-user sibling. No application process opens the device." (§27.7)

> "The device has one user-space entry point. The kernel driver denies opening the user client
> without the entitlement `com.apple.ane.iokit-user-access`, and exactly two binaries on the system
> hold it: the daemon and its per-user sibling. Every other process proves itself to the daemon over
> the interprocess channel instead, gated by a `com.apple.aned.private.*` entitlement family that
> the daemon checks per connection and per method." (§5.5)

> "A developer-signed binary cannot assert the kernel gate, since it names a restricted user-client
> class that ad-hoc and development signing cannot claim. The direct route thus reaches the engine
> the same way a sanctioned application does, through the daemon. It authors its work at the model
> and program layer rather than at the kernel interface." (§5.5)

**Local confirmation (this machine):** `/System/Library/LaunchDaemons/com.apple.aned.plist` and
`com.apple.aneuserd.plist` each contain `"com.apple.ane.iokit-user-access" => true` at top level
(i.e. held by the daemon itself), and their Mach services are:

| Daemon | MachServices | Runs as |
|---|---|---|
| `/usr/libexec/aned` | `com.apple.appleneuralengine`, `com.apple.appleneuralengine.private` | root |
| `/usr/libexec/aneuserd` | `com.apple.aneuserd` | `_neuralengine` |

### 2.2 The entitlement family that gates the *daemon* (not the kernel)

Paper Table 5.4 (verbatim):

| Entitlement | What it authorizes | Holders (in the paper's build) |
|---|---|---|
| `com.apple.ane.iokit-user-access` | the hard kernel gate: privileged device open, compile, cache | 2 |
| `com.apple.aned.private.allow` | baseline: compile, load, and instantiate models through the daemon | 18 |
| `com.apple.aned.private.ANEAccess.allow` | the inference-client access grant | 14 |
| `com.apple.aned.private.adapterWeight.allow` | stream adapter weights onto a shared resident base model | 5 |
| `com.apple.aned.private.processModelShare.allow` | share one resident model across processes | 4 |
| `com.apple.aned.private.secondaryANECompilerServiceAccess.allow` | use the longer-duration secondary compiler service | 1 |

> "A privileged subset of 27 binaries also holds a sandbox exception for the class
> `H11ANEInDirectPathClient`, which lets a latency-sensitive client open the low-latency user client
> and drive per-inference submission on its own connection … **The exception grants no device access
> by itself: the daemon still performs the privileged open and returns the program handle.**" (§5.5)

**These are the entitlements you cannot have.** They are Apple-private / restricted; §6(b) shows
what happens if you try to claim them.

### 2.3 Does the XPC path avoid the kernel entitlement? — YES, structurally

> "Per-inference submit is `IOConnectCallAsyncMethod` selector 2, the `ANE_ProgramSendRequest`
> handler in the H11ANE dispatch table, and **the unentitled client issues it directly**: a freshly
> compiled program issued selector 2 exactly once per execute, observed in the client process and
> never in the daemon. The daemon issues only the lifecycle selectors over its own connection during
> the same compile and prepare … the daemon compiles the network and creates, prepares, and destroys
> the program object over IOKit, **while the client submits each inference over selector 2 itself**." (§5.2.2)

> "A system daemon mediates access to the engine. Only that daemon and its per-user sibling hold the
> kernel gate that opens the device, so every other process reaches the engine through it. A client
> proves itself to the daemon over the interprocess channel, and the daemon performs the privileged
> device open on the client's behalf, returning a program handle the client then drives." (§5.4)

> "A request from a developer-signed program and a request from a system dispatcher arrive at the
> same broker, and the same queue arbitrates both." (§5.4)

**Reading:** the client process never calls `IOServiceOpen` on `H11ANEIn`; the daemon does. The
client gets a connection/program handle and issues the per-inference selector directly. That is why
an unentitled process can still dispatch at full speed.

### 2.4 Compilation is a separate gated service (but you don't talk to it)

> "Compilation does not happen in the calling process. The service that turns a network into the
> engine's program format is a separate sandboxed interprocess service, `com.apple.ANECompilerService`,
> reached through an `NSXPCConnection` named for it. That service vends a single entry point, the
> method `compileModelAt:csIdentity:sandboxExtension:options:tempDirectory:…:withReply:`, and admits a
> connection only through an entitlement gate. The service's `listener:shouldAcceptNewConnection:`
> delegate calls `valueForEntitlement:` on the connection against the string returned by
> `+compilerServiceAccessEntitlement` … The service holds two entitlements an ordinary process does
> not: it writes the system-protected compile cache under `rootless.storage.ane_model_cache`, and it
> decrypts under `coreml.decypt_allowed`. **A caller thus hands its network to the service and
> receives the compiled bundle back.**" (§5.6)

**Local confirmation:** the XPC bundle
`…/AppleNeuralEngine.framework/XPCServices/ANECompilerService.xpc/Contents/MacOS/ANECompilerService`
contains the symbols/strings `compilerServiceAccessEntitlement`,
`largeModelCompilerServiceAccessEntitlement`, `processModelShareAccessEntitlement`,
`valueForEntitlement:`, `%@: client(%d) : has entitlement(%@)`,
`%@: client(%d) : missing entitlement(%@)`. `aned` holds `com.apple.ANECompilerService.allow`
(visible in its own entitlements), which is how *your* compile succeeds without you holding anything.

### 2.5 The hard limit below everything: program signature + trustcache

> "**One hard limit is below the compute and defines the whole access model: a hand-built or
> self-compiled program cannot be loaded onto the engine.** The kernel driver verifies every submitted
> program before it reaches the firmware, by a corecrypto signature check over the program bytes and a
> trustcache check on the backing file's vnode, and rejects a program that fails either check at load
> with error `0xe00002e2`. **The only program the kernel will load is one the system daemon compiled
> and signed in place**, so a caller cannot author a network binary by hand and submit it." (§8.5)

> "The path has three steps. First, the client authors the network as intermediate language rather
> than a loadable binary … Second, it hands that intermediate language to the system daemon, the one
> process able to sign: the daemon compiles and signs it in place, and the returned program carries
> the signature and trustcache trust the kernel load check requires. Third, the client drives the
> returned signed program over the kernel interface directly, where the corecrypto signature and
> trustcache checks pass because it is daemon-signed and the program loads onto the engine." (§8.6, "Unentitled dispatch path")

### 2.6 Which features are entitlement-gated (and which are not)

> "The operations that on-device perception and numerics networks are built from compile and run on
> the direct route **without any entitlement**. Two-dimensional convolution and its transpose, matrix
> multiply, fused attention, the normalizations, the activation tables, pooling, elementwise
> arithmetic, and the data-movement operations all run … **an entitlement gates none of them.**" (§8.1)

Four features are *not* reachable on the direct route, and **no entitlement fixes them** because the
gate is elsewhere (missing backend lowering / missing silicon primitive):

| Feature | Gate | Direct path |
|---|---|---|
| 3-D convolution | no backend lowering on any device mask | compile: "Not implemented" |
| Native stateful types (state, ring buffer) | counter-and-event DMA engine stubbed out of the M1 descriptor | compile fails |
| bf16 program I/O | bf16 absent from the 11 `ANECIRDataType` codes | compile: "Unsupported function output dtype bf16" |
| Flexible/symbolic shapes | runtime path (symbolic-shape master gate) | parses, then fails to lower |
| Direct image-format (`pixel_buffer_to_tensor`) input | lowering needs the framework route | parses, does not lower |

> "The unentitled direct path is performance-complete for dispatch: it reaches the same throughput as
> any path with more privilege." (§6.4)

### 2.7 Error codes the paper names

| Code | Meaning per paper | Local SDK confirmation (MacOSX27.0.sdk, `Kernel.framework/Headers/IOKit/IOReturn.h`) |
|---|---|---|
| `0xe00002e2` | program-load rejection (corecrypto signature + trustcache) | `kIOReturnNotPermitted` (line 138) |
| `0xe00002c7` | **entitlement-rejection** return code for higher-tier gated features | `kIOReturnUnsupported` (line 109) |
| `0xe00002c2` | (paper: size/null check failures) | `kIOReturnBadArgument` (line 102) |
| `0xe00002c5` | (paper: closed-or-closing client state) | — |

> "The entitlement check the direct route never trips returns a distinct code from the program-load
> rejection. The program-load check returns `0xe00002e2` from the signature and trustcache check,
> while the entitlement gate that guards the higher inference-tier features returns `0xe00002c7`,
> `kIOReturnUnsupported`, the code a client gets when a gated feature's entitlement is absent." (§8.7)

---

## 3. Q2 — `maderix/ANE` (README.md, training/README.md, bridge/)

Sources: <https://github.com/maderix/ANE> · <https://raw.githubusercontent.com/maderix/ANE/main/README.md> ·
<https://raw.githubusercontent.com/maderix/ANE/main/training/README.md> ·
<https://raw.githubusercontent.com/maderix/ANE/main/bridge/ane_bridge.m> ·
<https://raw.githubusercontent.com/maderix/ANE/main/training/ane_runtime.h>

What it actually says about entitlements / sudo / SIP / macOS version / bridge loading:

* **Entitlements: nothing.** The words "entitlement", "codesign", "SIP" do not appear in the main
  README or `training/README.md`. The disclaimer only says the private APIs
  "may change or break with any macOS update".
* **Requirements:** "Requires macOS 15+ on Apple Silicon (tested on M4)". No minimum OS beyond that,
  no signing instructions, no SIP instructions.
* **Bridge loading:** plain `dlopen` of the framework, classes fetched at runtime:
  ```objc
  // training/ane_runtime.h
  dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);
  // then objc_msgSend on _ANEInMemoryModelDescriptor / _ANEInMemoryModel / _ANERequest / _ANEIOSurfaceObject
  ```
  The README's build lines are ordinary `make` / `xcrun clang -O2 -fobjc-arc -framework Foundation -framework IOSurface -ldl`
  — i.e. locally built, ad-hoc signed binaries. **No `sudo`, no entitlements, no SIP changes.**
* **`sudo` appears only** for the optional live dashboard (`sudo python3 dashboard.py`) because it
  reads power counters — unrelated to ANE access.
* **Documented workarounds worth knowing:** ~119 ANE compile limit per process (worked around with
  `exec()` restart); multi-input ANE requests cause a `0x1d` error, so inputs are packed into the
  spatial dimension; `matmul` is expressed as convolution.

**Interpretation:** this project demonstrably ran ANE training on macOS 15-era systems as an ordinary
local binary. It is the strongest "it works" precedent, and it matches what I measured here.

---

## 4. Q3 — `christopherkarani/Espresso`

Source: <https://raw.githubusercontent.com/christopherkarani/Espresso/main/README.md>

* "Espresso compiles MIL programs straight to ANE silicon through reverse-engineered private APIs
  (`_ANEClient`, `_ANEInMemoryModel`). **No CoreML in the hot path.**"
* Architecture table: "**ANEInterop** | `dlopen` bridge to `_ANEClient` and `_ANEInMemoryModel`."
* Requirements: "Hardware: Apple Silicon (M1+) with Neural Engine; **macOS 15.0+**; Swift 6.0+;
  Dependencies: None required".
* Platform table: "**Apple A-series (iOS) | ✅ | ⚠️ Requires entitlement; not App Store safe**" —
  note this row is explicitly about **iOS**, and the same table marks Intel Macs ❌. macOS rows for
  M1–M4 are ✅ with no entitlement caveat.
* "macOS 15+ required. **iOS / tvOS not supported out of the box (private API entitlements differ per platform).**"
* No mention of SIP, sudo, or codesigning anywhere in the README.
* Disclaimer: "Apps using private ANE APIs (`_ANEClient`, `_ANEInMemoryModel`) will be rejected [by the
  App Store]. Everywhere else: Internal tools, research, sideloaded apps, enterprise distribution — all fine."

**Interpretation:** on macOS the project claims it needs no entitlements; on iOS it explicitly says
entitlements are required. **The A-series-on-macOS case (this machine) is not covered by either
project's docs** — hence the direct measurement in §7.

---

## 5. Q4 — other reports

### 5.1 `mdaiter/ane` — the "requires entitlements" counter-report

Source: <https://raw.githubusercontent.com/mdaiter/ane/master/README.md>

> "| **Entitlement Bypass** | Struct init functions work without signing | Can probe all layer descriptor layouts |"
> "| **Silent Failures** | `compileModel:` returns NULL without error | Operations fail silently without entitlements |"

> "| Operation | Works? | Notes |
> | Load ANECompiler.framework | Yes | All frameworks load |
> | Create `_ANEClient` | Yes | Object created but… |
> | Call `compileModel:` | No | Returns NULL silently |
> | Call `loadModel:` | No | Returns NULL silently |
> | ANE inference | No | Requires entitlements |
> | CoreML with ANE | Yes | Use `MLComputeUnitsAll` - working path! |
> | **XPC to aned** | Yes | **Connection succeeds, ops need entitlements** |"

Its entitlement table lists `com.apple.aned.private.allow` ("Primary ANE access — compile, load,
evaluate"), `com.apple.aned.private.adapterWeight.allow`, `com.apple.aned.private.aggressivePowerSaving.allow`,
`com.apple.ANECompilerService.allow`, `com.apple.aned.private.processModelShare.allow`,
`com.apple.ane.memoryUnwiringOptOutAccess.allow`, `com.apple.private.modelPurgeInAllPartitions.allow`,
`com.apple.aned.private.secondaryANECompilerServiceAccess.allow`, `com.apple.private.ANEStorageMaintainer.allow`.
It also notes boot-arg bypasses are **Apple-internal-build only** (`_ANEDeviceInfo.isInternalBuild`
checks `/AppleInternal`, `os_variant_has_internal_content` — all false on consumer macOS).

**⚠️ Conflict:** this README says `compileModel:` fails silently without entitlements, while
maderix/Espresso/zenn and my own measurement show the ANE path working unentitled.
**Most likely explanation:** `mdaiter` is describing (i) the **file-based** `_ANEClient compileModel:`
/`loadModel:` path (a compiled `.mlmodelc` on disk, a different code path from the in-memory MIL path)
and/or (ii) the direct `ANEServices`/IOKit path, which genuinely is entitlement-gated. It is **not**
describing `_ANEInMemoryModel` MIL compilation, which I verified works. This distinction matters:
**use the in-memory MIL path, not the file-based/ANEServices path.**

Local support for the daemon-side split: `aned` contains `_ANEXPCServiceHelper` with
`_restricted`, `_unrestricted`, `_unrestrictedUser` properties and logs
`Ready to accept restricted and unrestricted XPC connections` / `User daemon is ready to accept
unrestrictedUser XPC connections`; restricted access is granted by
`allowRestrictedAccessFor:entitlementString:` and `valueForEntitlement:` (per-method gates such as
`adapterWeightsAccessEntitlement`, `processModelShareAccessEntitlement`,
`secondaryANECompilerServiceAccessEntitlement`, `aggressivePowerSavingEntitlement`,
`modelPurgeInAllPartitionsEntitlement`). **The compile/load/dispatch baseline is on the
unrestricted side — measured.**

### 5.2 Japanese engineering report (SALESCORE, 2026-03-16) — works on M4 Max, no entitlements

Source: <https://zenn.dev/salescore/articles/776dff7a85f781>

Describes exactly this route (`dlopen` `AppleNeuralEngine.framework`, ObjC runtime, `_ANEInMemoryModelDescriptor`,
`_ANEInMemoryModel`, `_ANERequest`, `_ANEIOSurfaceObject`) on an **M4 Max**, with **no mention of
entitlements, codesigning or SIP**. Independent corroboration that the direct route works on
consumer macOS without privileges. Useful extra facts from it:

* ANE I/O is **IOSurface, channel-first `[1, C, 1, S]`**, fp16 internally (fp32 I/O via MIL `cast`).
* `matmul` fails at runtime with **`status=0x1d`** → use `conv` 1×1 instead.
* Compile budget **~115 per process** (memory leak) → lazy compile / shape buckets.
* `D=4096` (pure power of two) suffers an SRAM bank conflict; padding to `D=4160` (= 64×65) fixed a
  3–5× slowdown.
* Multi-output mapping is **alphabetical by variable name**, not declaration order.
* ANE ignores `attn_mask` in SDPA; causal masking must be decomposed.

### 5.3 Other leads checked

* **tinygrad** `ane/` (HWX format, `AppleH11ANEInterface` IOKit path) — the older, lower-level
  reverse-engineering route; not needed here. <https://github.com/tinygrad/tinygrad>
* **Orion** (arXiv:2603.06728) — I fetched the HTML and found **no occurrences of "entitlement",
  "aned", "H11ANEIn", "SIP"** in the retrieved text, so it contributes nothing on this question.
  **UNCERTAIN** whether a fuller/different version discusses it.
* **Espresso CI** runs hardware ANE tests on self-hosted macOS runners (README "ANE Matrix" workflow)
  — more evidence that unentitled CI machines can run ANE code.
* No public report specifically about **macOS 27 / A18 Pro** was found; §7 is the only evidence for
  that combination that I could obtain.

---

## 6. Q5 — direct answers

### (a) Exact entitlement string(s)

**Kernel / device-open gate (you must NOT hold this; the daemon holds it):**
* `com.apple.ane.iokit-user-access` — hard gate for opening `H11ANEInUserClient` **or**
  `H11ANEInDirectPathClient` (checked once in `H11ANEInUserClient::init` via `copyClientEntitlement:`).
* `com.apple.ane.allow-dataChaining-access` — resident data-chaining gate (same place).

**Daemon-side family (Apple-private; higher-tier features only):**
`com.apple.aned.private.allow`, `com.apple.aned.private.ANEAccess.allow`,
`com.apple.aned.private.adapterWeight.allow`, `com.apple.aned.private.processModelShare.allow`,
`com.apple.aned.private.secondaryANECompilerServiceAccess.allow`, plus (mdaiter + local strings)
`com.apple.aned.private.aggressivePowerSaving.allow`, `com.apple.ane.memoryUnwiringOptOutAccess.allow`,
`com.apple.private.modelPurgeInAllPartitions.allow`, `com.apple.private.ANEStorageMaintainer.allow`,
`com.apple.ANECompilerService.allow`, `com.apple.ANELargeModelCompilerService.allow`,
`com.apple.private.coreml.decypt_allowed`.

**For the in-memory MIL path you are building: NONE of these are needed.** Verified in §7.

### (b) Can codesigning with entitlements (no SIP disable) work? — **NO** (measured)

* `codesign -s - --force --entitlements ent.plist <binary>` **succeeds** and the entitlements are
  embedded (`codesign -d --entitlements -` shows them).
* Launching that binary → **`Killed: 9`, exit code 137** (SIGKILL by AMFI at exec). Measured twice
  with `com.apple.aned.private.allow` + `com.apple.ane.iokit-user-access`.
* Control: the same binary signed with a **non-restricted** entitlement (`com.apple.security.get-task-allow`)
  launches normally (exit 0). So it is the *restricted/private* nature of the ANE entitlements that
  kills the process, not entitlements per se.
* Therefore: entitlements are a dead end on SIP-enabled consumer macOS; **do not spend time on them,
  and do not disable SIP** (not needed, and out of scope for this project).

### (c) Does XPC-based `_ANEClient` work unsigned? — **YES** (measured)

Ad-hoc signed, `TeamIdentifier=not set`, **zero entitlements**, euid 501, SIP enabled:

```
compileWithQoS:options:error: -> True   (28 ms)   NSError <nil>
loadWithQoS:options:error:    -> True   (18 ms)   NSError <nil>
model.state = 3
evaluateWithQoS:options:request:error: -> True (0.3 ms)  NSError <nil>
relu over 4096 fp16 values: 0 mismatches
unloadWithQoS:error: -> True
```

Daemon log for the same run (`log show --predicate 'process == "aned"'`) shows the brokered path:

```
aned: [com.apple.xpc:connection] activating connection: … name=com.apple.appleneuralengine.peer[54468]…
aned: ANED QoS: model.string_id=0 clientQos=21 … proc=_uuidANECompilerServiceRegular
aned: [com.apple.xpc:connection] activating connection: … name=com.apple.ANECompilerService
aned: (ANEServices) Found matching service: ANEDriver
aned: (ANEServices) Found matching service: H11ANEIn
aned: (ANEServices) Total num of devices 2
aned: (ANEServices) (Single-ANE System with ANEDriver) Selected ANEDriver device
aned: (ANEServices) ANEServicesDevice::ANEServicesDeviceOpen, usage type: 2
aned: (ANEServices) ANEDriver Device Open succeeded with usage type: 2
aned: (ANEServices) ANEServicesDevice::ANE_ProgramCreate, input buffer count: 1, output buffer count: 1
```

**No entitlement rejection appears anywhere.** The privileged device open happens **inside aned**,
exactly as the paper describes. Note the A18 Pro-specific detail: the selected device is
**`ANEDriver`** (not `H11ANEIn`), described by the daemon as "Single-ANE System with ANEDriver".

### (d) What to expect if blocked

| Symptom | Code / text | When |
|---|---|---|
| `NSError` from compile/load, or `NO` return | `0xe00002e2` (`kIOReturnNotPermitted`) | program fails the corecrypto-signature / trustcache load check (e.g. you hand-build a `.e5` binary instead of letting the daemon sign it) |
| `NSError` / `NO` on a higher-tier feature | `0xe00002c7` (`kIOReturnUnsupported`) | a gated inference-tier feature without its entitlement |
| Silent `NULL` / `nil` from `compileModel:` (file-based path) | no error object | reported by mdaiter for the file-based `_ANEClient` path |
| `status=0x1d` at eval time | `0x1d` | unsupported op/config — e.g. `matmul` (use 1×1 `conv`), or multi-input requests (pack inputs into the spatial dim) |
| XPC connection rejected | `aned` log: `Rejecting XPC connection requests from client(%d) : missing entitlement(%@)` / `missing signingIdentifier` | only for the *restricted* method set; not seen for the in-memory path |
| Compile hangs / stalls after failures | — | paper: "repeated failed compiles in quick succession can stall it, so pace compiles after a failure by about 15 seconds" |
| Process killed before `main` | SIGKILL, exit 137 | you signed with a restricted entitlement (§6b) |

---

## 7. Measured evidence on this machine (2026-10-08)

### 7.1 Frameworks exist but are dyld-cache-only

`/System/Library/PrivateFrameworks/{AppleNeuralEngine,ANEServices,ANECompiler}.framework` contain
only `Resources/`, `_CodeSignature/` and (for AppleNeuralEngine) `XPCServices/` — **the Mach-O
binary is not on disk**:

```
$ test -e …/AppleNeuralEngine.framework/AppleNeuralEngine && echo EXISTS || echo MISSING
MISSING
```

Nevertheless **absolute-path `dlopen` works** (dyld falls back to the shared cache):

```
OK dlopen: /System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine
OK dlopen: /System/Library/PrivateFrameworks/AppleNeuralEngine.framework/Versions/A/AppleNeuralEngine
FAIL dlopen: AppleNeuralEngine.framework/AppleNeuralEngine   (bare name — not in dyld cache)
```

→ **Always dlopen the absolute path**; a bare framework name fails. Also note: `strings`/`nm` on the
on-disk path yields nothing — inspect via the shared cache or at runtime if you need symbols.

### 7.2 Classes available after dlopen

`_ANEClient`, `_ANEInMemoryModel`, `_ANEInMemoryModelDescriptor`, `_ANEModel`,
`_ANEDeviceController` all resolve via `objc_getClass` once the framework is loaded (none before).
`ANEServices.framework` and `ANECompiler.framework` load too (their device classes are C++, not ObjC,
so `objc_getClass("ANEServicesDevice")` returns nil — expected).

### 7.3 Signature facts of the test process

```
$ codesign -dv pyfw/Versions/3.9/bin/python3.9
Identifier=python3-55554944…   Signature=adhoc   TeamIdentifier=not set
$ codesign -d --entitlements - pyfw/Versions/3.9/bin/python3.9
(no entitlements)
```

### 7.4 The kernel/daemon facts confirmed locally

* `csrutil status` → `System Integrity Protection status: enabled.`
* `aned` entitlements include `com.apple.ane.iokit-user-access`, `com.apple.ANECompilerService.allow`,
  `com.apple.ANELargeModelCompilerService.allow`, `platform-application`, `seatbelt-profiles: [aned]`.
* `aneuserd` entitlements include `com.apple.ane.iokit-user-access`, `com.apple.aneuserd.private.allow`.
* Both are Apple platform binaries (`Platform identifier=26`, `TeamIdentifier=not set`).

---

## 8. Concrete recipe for the Zig engine

**Do this (verified working, 2026-10-08):**

1. `dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW)`.
   Load Foundation (and CoreFoundation/IOSurface) first.
2. Fetch classes with `objc_getClass`: `_ANEInMemoryModelDescriptor`, `_ANEInMemoryModel`,
   `_ANERequest`, `_ANEIOSurfaceObject` (optionally `_ANEClient` for connection-level calls).
3. Build MIL text (header from maderix's generators):
   ```
   program(1.3)
   [buildInfo = dict<string, string>({{"coremlc-component-MIL", "3510.2.1"}, {"coremlc-version", "3505.4.1"},
     {"coremltools-component-milinternal", ""}, {"coremltools-version", "9.0"}})]
   {
       func main<ios18>(tensor<fp16, [1, C, 1, S]> x) {
           tensor<fp16, [1, C, 1, S]> y = relu(x=x)[name=string("y")];
       } -> (y);
   }
   ```
   (note the `} -> (y);` return syntax).
4. `desc = [_ANEInMemoryModelDescriptor modelWithMILText:weights:optionsPlist:]`
   (`weights` = `@{}` for weightless programs, otherwise a dict keyed
   `"@model_path/weights/weight.bin"` → `{offset, data}`).
5. `model = [_ANEInMemoryModel inMemoryModelWithDescriptor:desc]`.
6. Read `[model hexStringIdentifier]`, `mkdir $TMPDIR/<hex>` (and `<hex>/weights`), write `model.mil`
   there — this pre-population is part of the known-good recipe (the framework/daemon read the MIL
   from that directory).
7. `[model compileWithQoS:21 options:@{} error:&e]` → `BOOL`.
8. `[model loadWithQoS:21 options:@{} error:&e]` → `BOOL`; `[model state]` becomes `3`.
9. Build IOSurfaces (`IOSurfaceCreate` with Width=nbytes, Height=1, BytesPerElement=1,
   BytesPerRow=nbytes, AllocSize=nbytes, PixelFormat=0; data layout `[1,C,1,S]` fp16) →
   `[_ANEIOSurfaceObject objectWithIOSurface:]` → `[_ANERequest requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:]`.
10. `[model evaluateWithQoS:21 options:@{} request:req error:&e]` per token/step; read the output
    IOSurface with `IOSurfaceLock`/`IOSurfaceGetBaseAddress`/`IOSurfaceUnlock`.
11. Clean up: `unloadWithQoS:error:`.

**Design constraints to respect from the start:**

* **Never** sign with ANE entitlements; **never** touch SIP. Ad-hoc signing (default for `zig build`
  output on Apple Silicon) is sufficient.
* Compile is expensive and budgeted: ~115–119 compiles per process (memory-leak workaround: restart
  the process, or compile once and reuse). Cache compiled models yourself; the daemon also keeps a
  content-hashed cache (`cache/com.apple.e5rt.e5bundlecache/<os-build>/<hash>/`).
* Express matmuls as 1×1 convolutions (`matmul` can fail with `0x1d`); avoid multi-input requests;
  one input surface with everything packed into the spatial dimension is the proven pattern.
* IOSurface is the only I/O path; channel-first `[1, C, 1, S]`; fp16 I/O is ~37% faster than fp32
  (maderix) — but fp32 I/O is the well-trodden default in their examples.
* Avoid `D = 4096`-style power-of-two strides (SRAM bank conflict, zenn report) — pad to e.g. 4160.
* Causal masking: ANE ignores `attn_mask`; decompose SDPA.
* This is **unsupported and version-fragile**: an OS update can break it, and there is no App Store path.

---

## 9. UNCERTAIN / open questions

* **UNCERTAIN — A-series specifics.** All measurements here are on A18 Pro (H18-class). The paper's
  decompilation is M1/H13 and M5; maderix is M4; Espresso is M1–M4; zenn is M4 Max. Only a trivial
  `relu` kernel was verified end-to-end here — no conv/matmul/fused-attention kernel was tested on
  A18 Pro. The daemon does report `Single-ANE System with ANEDriver`, i.e. the device node differs
  from M-series naming.
* **UNCERTAIN — iOS vs macOS A-series.** Espresso's README says A-series **on iOS** requires
  entitlements. This machine shows A-series **on macOS** does not. Do not extrapolate to iOS.
* **UNCERTAIN — file-based `_ANEClient` path.** I verified `_ANEInMemoryModel` (MIL, in-memory).
  `[_ANEClient compileModel:options:qos:error:]` on an on-disk `.mlmodelc` was **not** tested;
  mdaiter reports silent `NULL` there. Prefer the in-memory path.
* **UNCERTAIN — per-method daemon gates.** The daemon has restricted/unrestricted helper classes; I
  verified the baseline compile/load/evaluate flow is unrestricted, but I did not enumerate which
  *other* selectors (`processModelShare`, adapter weights, aggressive power saving, secondary
  compiler service) are gated in practice.
* **UNCERTAIN — Orion paper.** Retrieved HTML contained no entitlement discussion; possibly a
  different version discusses it.
* **UNCERTAIN — longevity.** The paper calls the route "undocumented, unsupported, and
  version-fragile across operating-system updates"; nothing here changes that.
* **Not attempted (out of scope):** disabling SIP, boot-args, Apple-internal-build bypasses
  (`/AppleInternal`, `os_variant_*` — all false on consumer macOS).

---

## 10. Sources

* Bryngelson, *Apple Neural Engine: Architecture, Programming, and Performance*, arXiv:2606.22283v1
  (21 Jun 2026) — <https://arxiv.org/abs/2606.22283> · <https://arxiv.org/html/2606.22283v1>
  (PDF: <https://arxiv.org/pdf/2606.22283v1>) — §5.2.2, §5.4, §5.5, §5.6, §6.3, §6.4, §8.1–8.7, §27.7
* maderix/ANE — <https://github.com/maderix/ANE> · README, `training/README.md`,
  `bridge/ane_bridge.m`, `training/ane_runtime.h`, `training/training_dynamic/mil_dynamic.h`
* christopherkarani/Espresso — <https://github.com/christopherkarani/Espresso> · README.md
* mdaiter/ane — <https://github.com/mdaiter/ane> · README.md (entitlement table, "what works
  without entitlements", boot-args, internal-build detection)
* SALESCORE / muramoto, *Apple Neural Engine の Private API を叩いて LLM 推論を高速化しようとした話*
  (2026-03-16) — <https://zenn.dev/salescore/articles/776dff7a85f781> (`0x1d`, compile budget,
  bank conflict, channel-first layout)
* Orion: *Characterizing and Programming Apple's Neural Engine for LLM Training and Inference*,
  arXiv:2603.06728 — <https://arxiv.org/abs/2603.06728> (no entitlement content found in HTML)
* tinygrad ANE work — <https://github.com/tinygrad/tinygrad>
* Local macOS 27.0.1 (26A434) / A18 Pro inspection: `sw_vers`, `csrutil status`, launchd plists,
  `codesign -d --entitlements`, `strings` on `/usr/libexec/aned` and `/usr/libexec/aneuserd`,
  `log show --predicate 'process == "aned"'`, `MacOSX27.0.sdk/…/Kernel.framework/Headers/IOKit/IOReturn.h`
