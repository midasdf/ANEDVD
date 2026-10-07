# Espresso ANE private-API reference (extracted for a CoreML-free Zig ANE engine)

**Source:** `christopherkarani/Espresso` (MIT), branch `main`, commit
`05ea6a3da91be642abd3d4741c0341a941ce1064` (committed 2026-09-12).
All file paths below are relative to that repo root; raw URL prefix is
`https://raw.githubusercontent.com/christopherkarani/Espresso/05ea6a3da91be642abd3d4741c0341a941ce1064/`.

**Method:** the files were downloaded verbatim and read; everything in this document is
quoted from, or directly derived from, those files. Anything not directly evidenced is
marked **UNCERTAIN**.

---

## 0. TL;DR — the exact working call sequence

Names are exactly as they appear in `Sources/ANEInterop/ane_interop.m`. `objc_msgSend` uses
the casts shown because the private selectors are not declared anywhere. QoS argument is
**21** everywhere in this codebase.

1. **Load the private framework** →
   `dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW)`
   then `NSClassFromString` for `_ANEInMemoryModelDescriptor`, `_ANEInMemoryModel`,
   `_ANERequest`, `_ANEIOSurfaceObject`.
2. **Describe the program (in memory)** →
   `[_ANEInMemoryModelDescriptor modelWithMILText:weights:optionsPlist:](milData, weightsDictOr@{}, nil)`;
   `weightsDict` = `NSMutableDictionary<NSString* path, @{@"offset": @0, @"data": NSData*}>`,
   keys **must** start with `@model_path/` (e.g. `@model_path/weights/weight.bin`).
3. **Create the in-memory model** →
   `[_ANEInMemoryModel inMemoryModelWithDescriptor:desc]`.
4. **Materialise the model tree on disk** (the compiler reads from disk, not from the dict):
   `id hx = [mdl hexStringIdentifier]`;
   `td = NSTemporaryDirectory()/hx`; write `td/model.mil` (exact MIL UTF-8 bytes) and every
   weight blob to `td/<sanitised @model_path/… relative path>`.
5. **(Optional) set options** → `[mdl setPerfStatsMask:(unsigned int)]`, build an options
   `NSDictionary` (`kANEFPerformanceStatsMask`, `kANEFKeepModelMemoryWiredKey`, …), or pass `@{}`.
6. **Compile** → `[mdl compileWithQoS:21 options:finalOptions error:&e]` (BOOL). On failure and
   non-empty options → retry with `@{}`. *No compile counter increment on the reload paths.*
7. **Load** → `[mdl loadWithQoS:21 options:finalOptions error:&e]` (BOOL); on failure retry with `@{}`.
8. **Grab the client + model objects** →
   `client = [mdl sharedConnection]` (`_ANEClient`), `clientModel = [mdl model]` (`_ANEModel`).
   *This is the only way the code obtains an ANE client — there is no class-level client
   singleton call in the working path.*
9. **Allocate IOSurfaces** → `IOSurfaceCreate` with `{Width: bytes, Height: 1,
   BytesPerElement: 1, BytesPerRow: bytes, AllocSize: bytes, PixelFormat: 0}` — one per MIL
   input and one per MIL output, sized in **bytes** (`C*S*2` for fp16 `[C,S]`).
10. **Wrap surfaces** → `[_ANEIOSurfaceObject objectWithIOSurface:surf]` for each input/output.
11. **Build the request** →
    `[_ANERequest requestWithInputs:inputIndices:outputs:outputIndices:procedureIndex:](wIns, iIdx, wOuts, oIdx, @0)`
    (indices are `NSNumber`s `0..n-1`; `@0` = procedure index 0).
12. **Evaluate** → `[mdl evaluateWithQoS:21 options:options request:req error:&e]` (BOOL).
    Client-path alternatives: `[client evaluateWithModel:clientModel options:options request:req qos:21 error:&e]`
    or `[client doEvaluateDirectWithModel:clientModel options:options request:req qos:21 error:&e]`
    (used when `ANE_EVAL_PATH=clientDirect`, or whenever perf stats are requested and the
    selector exists); on failure it **falls back** to `[mdl evaluateWithQoS:…]`.
13. **Read/write surface memory** → `IOSurfaceLock(surf, 0, NULL)` for write,
    `IOSurfaceLock(surf, kIOSurfaceLockReadOnly, NULL)` for read, `IOSurfaceGetBaseAddress`,
    index as `_Float16 *base` with `idx = channel * spatial + position` (channel-first `[C,S]`,
    tightly packed, `bytesPerRow == allocSize`), then `IOSurfaceUnlock` with the same flags.
14. **Teardown** → `[mdl unloadWithQoS:21 error:&e]` (guarded by `respondsToSelector:`), then
    release surfaces, request, `client`, `clientModel`, `model` (ARC `CFBridgingRetain` in
    Espresso; a Zig implementation must retain/release explicitly), and remove the tmp dir
    unless `ANE_KEEP_TMPDIR` is set.

Everything above is proven by the code in §3–§8. The only paths that are **not** usable are the
`_ANEVirtualClient` family (§9) — every instantiation returned `nil` on real hardware
("classic IOKit entitlement gate").

---

## 1. What Espresso is / what it claims

From `README.md`:

> Espresso compiles MIL programs straight to ANE silicon through reverse-engineered private APIs
> (`_ANEClient`, `_ANEInMemoryModel`). No CoreML in the hot path. No per-token recompilation.
> IOSurface buffers and fused multi-layer kernels for Apple Silicon.

* Architecture (`README.md` "Architecture"): `ANEInterop (ObjC/C — private API bridge)` →
  `ANETypes` → `MILGenerator` → `ANERuntime` → `Espresso`/`RealModelInference`.
* `MILGenerator` "Generates MIL text for forward, backward, decode, and fused kernels."
* `ANERuntime` "Compiles MIL to ANE E5 binaries. Manages IOSurface buffers and compile budget."
* Zero third-party deps; links only Foundation/Accelerate/IOSurface/Metal/CoreML
  (`Package.swift`: `ANEInterop` target links `Foundation`, `CoreML`, `IOSurface`, `Accelerate`, `libdl`;
  C flags `-fobjc-arc -O2`). **CoreML is linked but not used in the hot path.**
* Private-API disclaimer: "Apps using private ANE APIs (`_ANEClient`, `_ANEInMemoryModel`) will be
  rejected [from the App Store]. Everywhere else: Internal tools, research, sideloaded apps,
  enterprise distribution — all fine."

### 1.1 Platform requirements (as stated by the project)

| Requirement | Value | Source |
|---|---|---|
| Hardware | Apple Silicon M1+ with Neural Engine (A-series iOS "requires entitlement; not App Store safe") | `README.md` |
| macOS | 15.0+ (`.macOS(.v15)` in `Package.swift`) | `README.md`, `Package.swift` |
| Swift | 6.0+ (6.2 recommended); binary uses ARC | `README.md`, `Package.swift` |
| Tested SoCs | M1/M2 (16-core ANE), M3 (18-core, reference), M4 (38-core) | `README.md` |
| Intel Mac | unsupported (no ANE) | `README.md` |
| iOS/tvOS | not supported out of the box (entitlements differ) | `README.md` |
| Entitlements | **none needed for the in-memory macOS path** — no `codesign`/entitlement step exists anywhere in the repo (scripts, CI, `espresso` CLI) | repo-wide search |
| SIP | **no evidence that SIP must be disabled**, and no `csrutil`/`nvram`/`xattr`/`spctl` step anywhere in the repo | repo-wide search |
| Hardened runtime / signing | not mentioned for the ANE path; only `_ANEVirtualClient` is described as entitlement-gated (and unusable) | `docs/reverse-engineering-apple-neural-engine.html` |

**UNCERTAIN / caveats to carry into our project:**
* The project targets `macOS 15+` and never mentions macOS 26/27 or A18. Espresso's code is
  version-defensive (every private selector is `respondsToSelector:`-guarded), so it should
  be treated as "works where the selectors exist", not "verified on macOS 27.0.1 / A18 Pro".
  The very first thing our Zig engine must do is the same selector-existence probe.
* No `dlopen` of `ANECompilerService.framework` appears in the shipped code; the blog
  (`docs/reverse-engineering-…html`) shows an older `dlopen("/System/Library/PrivateFrameworks/ANECompilerService.framework/Versions/A/ANECompilerService", RTLD_NOW|RTLD_LOCAL)`
  plus `dlsym`. **The shipped `ane_interop.m` only dlopens `AppleNeuralEngine.framework`**
  and resolves classes by name. Prefer the shipped path.
* The blog's IOSurface recipe (`BytesPerElement: 2`, `kCVPixelFormatType_OneComponent16Half`,
  2-D width/height) does **not** match the shipped `ane_interop_create_surface` (§6.1). The
  shipped code is authoritative; the blog is illustrative.

### 1.2 Performance numbers claimed (`README.md`, `benchmarks/results/latest.json`, M3 Max, macOS 15.0, Espresso 1.1.0)

| Backend | ms/token | tok/s | Notes |
|---|---|---|---|
| **Espresso ANE** (recurrent fused, 6-layer) | **1.93** | **519** | fused 3-layer recurrent decode + ANE classifier |
| Espresso ANE (direct transformer, 6-layer) | 6.56 | 153 | no recurrent fusion |
| CoreML `.cpuAndNeuralEngine` | 6.58 | 152 | Apple's standard path |
| Speedup vs CoreML | | **3.41×** | fused recurrent path |

Workload: 6-layer local artifact, `dim=768`, 12 heads, 32k vocab, `seqLen=256`.
Other latency facts stated in the docs: first **cold compile of a 6-layer recurrent model =
131 s**; subsequent runs after OS cache warm-up = **350 ms**; Qwen2.5-1.5B hybrid path puts
Q/K/V + SwiGLU FFN on ANE and RoPE/attention/467 MB LM head on CPU. Model formats:
`.esp` (canonical portable bundle), `.espc` (host-local compiled cache), plus raw MIL+weights.

---

## 2. Public API surface exposed by `Sources/ANEInterop/include/ane_interop.h`

Full header is 560 lines; the ANE-driving subset is below (verbatim signatures). All of the
`io_*`, `neon_*`, `bnns_*`, probe and chaining functions are *auxiliary* (CPU I/O, probes,
experiments) — a Zig port does not need them to run a model.

```c
typedef struct ANEHandle ANEHandle;

void ane_interop_init(void);
IOSurfaceRef ane_interop_create_surface(size_t bytes) CF_RETURNS_RETAINED;

#define ANE_INTEROP_COMPILE_ERROR_NONE 0
#define ANE_INTEROP_COMPILE_ERROR_INVALID_ARGUMENTS 1
#define ANE_INTEROP_COMPILE_ERROR_DUPLICATE_WEIGHT_PATH 2
#define ANE_INTEROP_COMPILE_ERROR_SURFACE_ALLOCATION_FAILED 3
#define ANE_INTEROP_COMPILE_ERROR_COMPILER_FAILURE 4

ANEHandle *ane_interop_compile(const uint8_t *milText, size_t milLen,
                               const char **weightPaths, const uint8_t **weightDatas,
                               const size_t *weightLens, int weightCount,
                               int nInputs, const size_t *inputSizes,
                               int nOutputs, const size_t *outputSizes);

bool ane_interop_eval(ANEHandle *handle);
IOSurfaceRef ane_interop_get_input(ANEHandle *handle, int index) CF_RETURNS_NOT_RETAINED;
IOSurfaceRef ane_interop_get_output(ANEHandle *handle, int index) CF_RETURNS_NOT_RETAINED;
IOSurfaceRef ane_interop_copy_input(ANEHandle *handle, int index) CF_RETURNS_RETAINED;
IOSurfaceRef ane_interop_copy_output(ANEHandle *handle, int index) CF_RETURNS_RETAINED;
void ane_interop_free(ANEHandle *handle);

int  ane_interop_compile_count(void);
void ane_interop_set_compile_count(int value);
int  ane_interop_last_compile_error(void);
void ane_interop_set_force_eval_failure(bool value);
int  ane_interop_live_handle_count(void);
uint64_t ane_interop_last_hw_execution_time_ns(ANEHandle *handle);
bool ane_interop_has_perf_stats(ANEHandle *handle);
bool ane_interop_cached_donor_net_plist_exists(const char *hexId);
bool ane_interop_should_try_cached_load(const char *hexId, bool compiledExists);

/// Replace an input surface and rebuild the ANE request.
bool ane_interop_rebind_input(ANEHandle *handle, int index, IOSurfaceRef newSurface);

ANEHandle *ane_interop_compile_with_id(const uint8_t *milText, size_t milLen,
                                       const char **weightPaths, const uint8_t **weightDatas,
                                       const size_t *weightLens, int weightCount,
                                       int nInputs, const size_t *inputSizes,
                                       int nOutputs, const size_t *outputSizes,
                                       char *outHexId, size_t hexIdBufLen);

ANEHandle *ane_interop_delta_reload(const uint8_t *milText, size_t milLen,
                                    const char **weightPaths, const uint8_t **weightDatas,
                                    const size_t *weightLens, int weightCount,
                                    int nInputs, const size_t *inputSizes,
                                    int nOutputs, const size_t *outputSizes,
                                    const char *donorHexId);

bool ane_interop_fast_reload(ANEHandle *handle,
                             const char **weightPaths, const uint8_t **weightDatas,
                             const size_t *weightLens, int weightCount);

bool ane_interop_get_hex_id(ANEHandle *handle, char *outHexId, size_t bufLen);

// fp16 conversion + CPU-side surface I/O helpers
void ane_interop_cvt_f32_to_f16(void *dst, const float *src, int count);
void ane_interop_cvt_f16_to_f32(float *dst, const void *src, int count);
int  ane_interop_fp16_gemv_argmax(const void *weights_f16, const float *input, int vocab_size, int dim);
bool ane_interop_io_copy(IOSurfaceRef dst, int dst_ch_off, IOSurfaceRef src, int src_ch_off, int channels, int spatial);
bool ane_interop_io_write_fp16(IOSurfaceRef surface, const float *data, int channels, int spatial);
bool ane_interop_io_read_fp16(IOSurfaceRef surface, int ch_off, float *data, int channels, int spatial);
bool ane_interop_io_write_fp16_spatial_slice(IOSurfaceRef, int ch_off, int spatial_index, int spatial, const float*, int channels);
bool ane_interop_io_read_fp16_spatial_slice(IOSurfaceRef, int ch_off, int spatial_index, int spatial, float*, int channels);
bool ane_interop_io_argmax_fp16_spatial_slice(IOSurfaceRef, int ch_off, int spatial_index, int spatial, int channels, int *out_index, float *out_value);
bool ane_interop_io_argmax_fp16_spatial_slice_with_hint(IOSurfaceRef, int ch_off, int spatial_index, int spatial, int channels, IOSurfaceRef hint_surface, int hint_spatial_index, int hint_spatial, int *out_index, float *out_value);
bool ane_interop_io_lock_write(IOSurfaceRef surface);
bool ane_interop_io_unlock_write(IOSurfaceRef surface);
bool ane_interop_io_lock_read(IOSurfaceRef surface);
bool ane_interop_io_unlock_read(IOSurfaceRef surface);
bool ane_interop_io_write_fp16_unlocked(IOSurfaceRef, const float*, int channels, int spatial);
bool ane_interop_io_read_fp16_unlocked(IOSurfaceRef, int ch_off, float*, int channels, int spatial);
```

Key semantics (from `Sources/ANERuntime/ANEKernel.swift` docs): `inputSizes`/`outputSizes` are
**byte sizes per IOSurface** ("typically 1 element for single-input kernels"). For a `[C,S]`
fp16 tensor that is `C*S*2`; tests use `probeChannels * probeSpatial * MemoryLayout<UInt16>.stride`.

---

## 3. Private classes, selectors and argument shapes

| Class | Selector | Kind | Args / return | Where used |
|---|---|---|---|---|
| `_ANEInMemoryModelDescriptor` | `modelWithMILText:weights:optionsPlist:` | class | `(NSData* mil, NSDictionary* weights, id plist) -> id desc` | compile, delta reload |
| `_ANEInMemoryModel` | `inMemoryModelWithDescriptor:` | class | `(id desc) -> id mdl` | compile, delta reload |
| `_ANEInMemoryModel` | `hexStringIdentifier` | instance | `-> NSString*` (used as tmp dir name + cache key) | compile, delta reload, `ane_interop_get_hex_id` |
| `_ANEInMemoryModel` | `compileWithQoS:options:error:` | instance | `(unsigned int qos, NSDictionary* opts, NSError** e) -> BOOL` | compile |
| `_ANEInMemoryModel` | `loadWithQoS:options:error:` | instance | `(unsigned int qos, NSDictionary* opts, NSError** e) -> BOOL` | compile, delta/fast reload |
| `_ANEInMemoryModel` | `unloadWithQoS:error:` | instance | `(unsigned int qos, NSError** e) -> BOOL`, `respondsToSelector:`-guarded | teardown, reload |
| `_ANEInMemoryModel` | `evaluateWithQoS:options:request:error:` | instance | `(unsigned int qos, NSDictionary* opts, id req, NSError** e) -> BOOL` | eval (primary + fallback) |
| `_ANEInMemoryModel` | `sharedConnection` | instance | `-> id` (`_ANEClient`) | get client |
| `_ANEInMemoryModel` | `model` | instance | `-> id` (`_ANEModel`) | get client model handle |
| `_ANEInMemoryModel` | `compiledModelExists` | instance | `-> BOOL` | cache policy |
| `_ANEInMemoryModel` | `purgeCompiledModel` | instance | `-> void` | cache policy `forceCold` |
| `_ANEInMemoryModel` | `setQueueDepth:` | instance | `(char)` clamped to `<=127`, env `ANE_QUEUE_DEPTH` | tuning |
| `_ANEInMemoryModel` | `setPerfStatsMask:` | instance | `(unsigned int)` | perf stats |
| `_ANEInMemoryModel` | `compilerOptionsWithOptions:isCompiledModelCached:` | instance | `(NSDictionary*, BOOL) -> NSDictionary*` | opt-in via `ANE_USE_COMPILER_OPTIONS` |
| `_ANERequest` | `requestWithInputs:inputIndices:outputs:outputIndices:procedureIndex:` | class | `(NSArray<_ANEIOSurfaceObject*>, NSArray<NSNumber*>, NSArray, NSArray, NSNumber*) -> id req` | request build |
| `_ANERequest` | `requestWithInputs:inputIndices:outputs:outputIndices:perfStats:procedureIndex:` | class | `(…, id perfStats, NSNumber*)` | perf-stats variant |
| `_ANERequest` | `requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:` | class | `(…, id weightsBuffer /*nil*/, id perfStats, NSNumber*)` | fallback variant |
| `_ANERequest` | `requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:procedureIndex:` | class | `(…, nil, NSNumber*)` | last-resort variant |
| `_ANERequest` | `requestWithInputs:inputIndices:outputs:outputIndices:perfStats:perfStatsMask:procedureIndex:sharedEvents:` | class | 8-arg | shared-events probe only |
| `_ANERequest` | `setSharedEvents:` / `setCompletionHandler:` / `perfStats` / `perfStatsArray` | instance | probes / stats read-back | probes |
| `_ANEIOSurfaceObject` | `objectWithIOSurface:` | class | `(IOSurfaceRef) -> id` | surface wrapping |
| `_ANEClient` | `evaluateWithModel:options:request:qos:error:` | instance | `(id model, NSDictionary*, id req, unsigned int qos, NSError**) -> BOOL` | eval `ANE_EVAL_PATH=client` |
| `_ANEClient` | `doEvaluateDirectWithModel:options:request:qos:error:` | instance | same shape | eval `clientDirect` / perf-stats |
| `_ANEClient` | `evaluateRealTimeWithModel:options:request:error:` | instance | `(id model, id opts, id req, NSError**) -> BOOL` | realtime path |
| `_ANEClient` | `beginRealTimeTask` / `endRealTimeTask` | instance | `-> BOOL` | realtime | 
| `_ANEClient` | `loadRealTimeModel:options:qos:error:` / `unloadRealTimeModel:options:qos:error:` | instance | `(id model, id opts, unsigned int qos, NSError**) -> BOOL` | realtime |
| `_ANEClient` | `virtualClient` | instance | `-> id` (**returns nil in practice**) | VC probe |
| `_ANEClient` | `prepareChainingWithModel:options:chainingReq:qos:error:` | instance | chaining | probe |
| `_ANEChainingRequest` | `chainingRequestWithInputs:outputSets:lbInputSymbolId:lbOutputSymbolId:procedureIndex:signalEvents:transactionHandle:fwEnqueueDelay:memoryPoolId:` | class | chaining | probe |
| `_ANEPerformanceStats` | `statsWithRequestPerformanceBuffer:statsBufferSize:` | class | `(void**, unsigned int*) -> id` | perf stats |
| `_ANEPerformanceStats` | `driverMaskForANEFMask:` | class | `(unsigned int) -> unsigned int` | perf mask translation |
| `_ANEPerformanceStats` | `hwExecutionTime` | instance | `-> uint64_t` ns | timing |
| `_ANEVirtualClient` | see §9 | — | **all instantiation paths return nil** | dead end |

Class objects are fetched once in `ane_interop_init()` (verbatim):

```objc
void ane_interop_init(void) {
    if (g_ane_loaded) return;
    dispatch_once(&g_ane_once, ^{
        dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);
        g_ANEDesc = NSClassFromString(@"_ANEInMemoryModelDescriptor");
        g_ANEInMem = NSClassFromString(@"_ANEInMemoryModel");
        g_ANEReq = NSClassFromString(@"_ANERequest");
        g_ANEIO = NSClassFromString(@"_ANEIOSurfaceObject");
        g_ane_loaded = true;
    });
}
```
Source: `Sources/ANEInterop/ane_interop.m#L1111-L1121`.

The `ANEHandle` struct (what a Zig port must mirror):

```objc
struct ANEHandle {
    void *model;               // CFBridgingRetain'd _ANEInMemoryModel
    void *client;              // CFBridgingRetain'd _ANEClient (optional)
    void *clientModel;         // CFBridgingRetain'd _ANEModel (optional)
    IOSurfaceRef *ioInputs;
    IOSurfaceRef *ioOutputs;
    void *request;             // CFBridgingRetain'd _ANERequest
    void *perfStats;           // CFBridgingRetain'd _ANEPerformanceStats (optional)
    bool perfStatsRequested;
    unsigned int perfStatsMask;
    void *evalOptions;         // CFBridgingRetain'd NSDictionary (optional)
    bool realtimeLoaded;
    void *tmpDir;              // CFBridgingRetain'd NSString
    int nInputs, nOutputs;
    size_t *inputBytes;
    size_t *outputBytes;
    bool liveHandleCounted;
    uint64_t lastHwExecutionTimeNS;
};
```
Source: `Sources/ANEInterop/ane_interop.m#L13-L31`.

---

## 4. (b) Creating an in-memory model from MIL + weights — verbatim

```objc
NSData *milData = [NSData dataWithBytesNoCopy:(void *)milText length:milLen freeWhenDone:NO];

NSMutableDictionary *weights = [NSMutableDictionary dictionaryWithCapacity:(NSUInteger)weightCount];
for (int i = 0; i < weightCount; i++) {
    NSString *path = [NSString stringWithUTF8String:weightPaths[i]];
    NSData *wd = [NSData dataWithBytesNoCopy:(void *)weightDatas[i] length:weightLens[i] freeWhenDone:NO];
    weights[path] = @{@"offset": @0, @"data": wd};   // <- weight dictionary shape
}

id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(
    g_ANEDesc, @selector(modelWithMILText:weights:optionsPlist:),
    milData, (id)(weightCount ? weights : @{}), nil);

id mdl = ((id(*)(Class,SEL,id))objc_msgSend)(
    g_ANEInMem, @selector(inMemoryModelWithDescriptor:), desc);
```
Source: `Sources/ANEInterop/ane_interop.m#L1879-L1929` (and L745-L769, L1885-L1918).

Notes:
* `milText` is copied into an `NSData` **without copying** (`freeWhenDone:NO`) → the caller's
  buffer must outlive compilation.
* `weightDatas[i]` may be `NULL` when `weightLens[i] == 0` (`[NSData data]` is used in that case
  on the reload path, `ane_interop.m#L1194-L1201`).
* Duplicate weight paths are rejected up-front (`ANE_INTEROP_COMPILE_ERROR_DUPLICATE_WEIGHT_PATH`).
* Weight paths **must** have the `@model_path/` prefix; `ane_interop_sanitized_relative_weight_path`
  strips it and rejects absolute paths and `.`/`..` components. Anything else →
  `ANE_INTEROP_COMPILE_ERROR_INVALID_ARGUMENTS` and (in compile) the stderr message
  `"ANE compile failed: invalid weight path '%s'"` / `"ANE compile failed: escaped tmp dir for weight path '%s'"`.
* The `@{@"offset": @0, @"data": NSData}` dictionary is what the descriptor factory wants; the
  **same bytes are separately written to disk** because the compiler reads the model directory
  (§5).

---

## 5. (c) Loading / unloading, the on-disk model tree, caching, retry logic

### 5.1 On-disk model tree (required — the compiler reads files)

```objc
id hx = ((id(*)(id,SEL))objc_msgSend)(mdl, @selector(hexStringIdentifier));  // NSString
NSString *td = [NSTemporaryDirectory() stringByAppendingPathComponent:hx];
// createDirectory td/weights
// write td/model.mil            <- exact MIL text, atomically
// for each weight path: write td/<rel>   (rel = path minus "@model_path/")
```
Source: `ane_interop.m#L2028-L2097` and `ane_interop_write_model_tree` `#L148-L191`.
The directory is removed on failure/teardown unless `ANE_KEEP_TMPDIR` is set
(`ane_interop_remove_tmpdir`, `#L1822-L1826`).

So the layout on disk is:
```
$TMPDIR/<hexStringIdentifier>/
    model.mil
    weights/<name>.bin ...
```
and the MIL `BLOBFILE(path=string("@model_path/weights/weight.bin"), offset=...)` resolves
`@model_path/` to that directory (the `@model_path/weights/…` key is also exactly the dictionary key).

### 5.2 Compile + load (verbatim core)

```objc
NSError *e = nil;
if (!((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
        mdl, @selector(compileWithQoS:options:error:), 21, finalOptions, &e)) {
    // Retry without options (some host builds reject unknown keys).
    if ([finalOptions count] > 0) {
        if (strictOptions) { /* ANE_STRICT_OPTIONS=1 -> fail hard */ ... }
        e = nil;
        if (!((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
                mdl, @selector(compileWithQoS:options:error:), 21, @{}, &e)) {
            fprintf(stderr, "ANE compile failed: %s\n", e ? [[e description] UTF8String] : "no error");
            /* ANE_INTEROP_COMPILE_ERROR_COMPILER_FAILURE; remove tmpdir; return NULL */
        }
        finalOptions = @{};
    } else { /* fail */ }
}

NSDictionary *loadOptions = ane_interop_reload_with_fallback_options(mdl, finalOptions, strictOptions, &e);
if (!loadOptions) {
    fprintf(stderr, "ANE load failed: %s\n", e ? [[e description] UTF8String] : "no error");
    /* fail */
}
```
Source: `ane_interop.m#L2118-L2161`; fallback helper `#L243-L266`:

```objc
static NSDictionary *ane_interop_reload_with_fallback_options(id mdl, NSDictionary *preferredOptions,
                                                              bool strictOptions, NSError **outError) {
    NSError *loadError = nil;
    NSDictionary *loadOptions = preferredOptions ?: @{};
    if (objc_msgSend(mdl, @selector(loadWithQoS:options:error:), 21, loadOptions, &loadError)) return loadOptions;
    if ([loadOptions count] > 0 && !strictOptions) {
        loadError = nil;
        if (objc_msgSend(mdl, @selector(loadWithQoS:options:error:), 21, @{}, &loadError)) return @{};
    }
    *outError = loadError; return nil;
}
```

### 5.3 Unload (guarded — selector may be missing on some OS builds)

```objc
/// Guarded unload: the private `unloadWithQoS:error:` selector is not guaranteed
/// to exist on every OS build. Returns false when the selector is missing or the send fails.
static bool ane_interop_unload_model(id mdl) {
    if (!mdl) return false;
    SEL sel = @selector(unloadWithQoS:error:);
    if (![mdl respondsToSelector:sel]) return false;
    NSError *e = nil;
    return ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(mdl, sel, 21, &e);
}
```
Source: `ane_interop.m#L198-L207`.

### 5.4 Compile-cache policies (env `ANE_COMPILE_CACHE_POLICY`)

* `auto` (default) → always `compileWithQoS:` + `loadWithQoS:`.
* `preferCached`/`prefer_cached` → if `[mdl compiledModelExists]` (or a cached donor `net.plist`
  exists), copy the donor `net.plist` into the model dir and call **only** `loadWithQoS:`
  (skips compile). Cache root: `ANE_INTEROP_CACHE_ROOT` or `~/Library/Caches/<…>`, guarded by
  `NSDistributedLock` per hex id.
* `forceCold`/`force_cold` → call `[mdl purgeCompiledModel]` first, then compile.
Source: `ane_interop.m#L1080-L1086`, `#L1951-L1959`, `#L2102-L2116`, `#L268-L412`.

### 5.5 Reload paths (weights swapped without recompiling)

* `ane_interop_delta_reload(..., donorHexId)` — new descriptor/model from the same MIL + new
  weights, writes the tree, copies the donor's compiled `net.plist`, then `loadWithQoS:`
  (no `compileWithQoS:`); **does not** increment the compile counter.
* `ane_interop_fast_reload(handle, …)` — `unloadWithQoS:` → stage new weight files → move
  directories → `loadWithQoS:`; preserves `model.mil` and `net.plist`; backs out and restores
  on failure. This is the "swap weights, same program" mechanism (the docs claim weight swapping
  is not supported by the compiler; this is the workaround, at directory level).

### 5.6 Model load options (only built when needed)

Keys used (all set conditionally, never required):
`kANEFEnablePowerSavingKey` (`ANE_DISABLE_POWER_SAVING=1` → `@NO`),
`kANEFKeepModelMemoryWiredKey` (`ANE_KEEP_MODEL_WIRED=1` → `@YES`),
`kANEFEnableLateLatchKey` (`ANE_ENABLE_LATE_LATCH=1`),
`kANEFSkipPreparePhaseKey` (`ANE_SKIP_PREPARE=1`),
`kANEFEnableFWToFWSignal` (`ANE_ENABLE_FW_TO_FW_SIGNAL=1`),
`kANEFDisableIOFencesUseSharedEventsKey` (`ANE_DISABLE_IO_FENCES=1`),
`kANEFMemoryPoolIDKey` (`ANE_MEMORY_POOL_ID=<n>`),
`kANEFPerformanceStatsMask` + `kANEFModelLoadPerformanceStats` (with `ANE_PERF_STATS=1`).
Source: `ane_interop.m#L413-L493` and `#L1962-L2017`.

### 5.7 Realtime path (optional; may need entitlements)

```objc
[client beginRealTimeTask];
BOOL loaded = [client loadRealTimeModel:clientModel options:finalOptions qos:21 error:&rtErr];
if (!loaded) [client endRealTimeTask];
// eval:
[client evaluateRealTimeWithModel:modelObj options:options request:req error:&e];
// teardown:
[client unloadRealTimeModel:modelObj options:options qos:21 error:&rtErr];
[client endRealTimeTask];
```
Source: `ane_interop.m#L1415-L1442`, `#L2441-L2454`, `#L2362-L2365`, `#L3275-L3283`.
Selector-discovery test asserts all five selectors exist on Apple Silicon macOS 15+
(`Tests/ANERuntimeTests/RealTimeEvalProbeTests.swift`), but the repo itself notes
"loadRealTimeModel failed (may require entitlements)". Needed only when
`ANE_EVAL_PATH=realtime`; otherwise `handle->realtimeLoaded == false`.

---

## 6. (d)/(f) IOSurfaces and the weight blob format

### 6.1 Surface creation (verbatim)

```objc
IOSurfaceRef ane_interop_create_surface(size_t bytes) {
    return IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth: @(bytes),
        (id)kIOSurfaceHeight: @1,
        (id)kIOSurfaceBytesPerElement: @1,
        (id)kIOSurfaceBytesPerRow: @(bytes),
        (id)kIOSurfaceAllocSize: @(bytes),
        (id)kIOSurfacePixelFormat: @0
    });
}
```
Source: `Sources/ANEInterop/ane_interop.m#L1811-L1820`.

Note the flat 1-D model: width = height = **alloc size in bytes**, `bytesPerRow == allocSize`,
`pixelFormat = 0`. (The blog's `BytesPerElement: 2` + `OneComponent16Half` + 2-D shape is a
different, illustrative recipe — **UNCERTAIN** whether both are accepted; the shipped one is
what the hardware tests use.)

### 6.2 Locking and memory layout

* Write: `IOSurfaceLock(surface, 0, NULL)` (flags `0`), then `IOSurfaceGetBaseAddress`.
* Read: `IOSurfaceLock(surface, kIOSurfaceLockReadOnly, NULL)`.
* Unlock with the identical flag argument: `IOSurfaceUnlock(surface, 0, NULL)` /
  `IOSurfaceUnlock(surface, kIOSurfaceLockReadOnly, NULL)`.
* Every read/write helper checks `bytes <= IOSurfaceGetAllocSize(surface)` first.

Layout: surfaces are treated as a **tightly packed channel-first fp16 tensor `[C, S]`**:

```c
_Float16 *base = (_Float16 *)IOSurfaceGetBaseAddress(surface);
size_t baseIdx = (size_t)ch_off * spatial + (size_t)spatial_index;   // channel-major
// element index = channel * spatial + position; no row padding (bytesPerRow == C*S*2)
```
Source: `Sources/ANEInterop/surface_io.c#L57-L91`, `#L300-L330`, `#L950-L975`.
MIL sees the tensor as `[1, C, 1, S]` — batch 1, channel axis = feature dim, height 1,
width = spatial/sequence positions (blog + `GenericMIL.conv`).

### 6.3 (f) Weight blob / file format

`Sources/ANETypes/WeightBlob.swift` (verbatim, whole file is 93 lines):

```swift
public enum WeightBlob {
    private static func writeHeader(_ raw: UnsafeMutableRawBufferPointer, payloadBytes: Int) {
        precondition(payloadBytes >= 0 && payloadBytes <= Int(UInt32.max))
        let base = raw.baseAddress!.assumingMemoryBound(to: UInt8.self)
        base[0] = 1
        base[4] = 2
        base[64] = 0xEF
        base[65] = 0xBE
        base[66] = 0xAD
        base[67] = 0xDE
        base[68] = 1
        raw.storeBytes(of: UInt32(payloadBytes).littleEndian, toByteOffset: 72, as: UInt32.self)
        raw.storeBytes(of: UInt32(128).littleEndian, toByteOffset: 80, as: UInt32.self)
    }

    public static func build(from weights: UnsafeBufferPointer<Float>, rows: Int, cols: Int) -> Data {
        let weightBytes = weights.count * 2
        let total = 128 + weightBytes
        var data = Data(count: total)
        data.withUnsafeMutableBytes { raw in
            writeHeader(raw, payloadBytes: weightBytes)
            let payload = raw.baseAddress!.advanced(by: 128)
            ane_interop_cvt_f32_to_f16(payload, src, Int32(weights.count))
        }
        return data
    }
    // buildTransposed(from:rows:cols:)  -> same header, fp16 payload written column-major
    // buildFP16(from:)                  -> same header, raw UInt16 payload copied
}
```

Decoded single-blob layout (offsets relative to start of file):

| Bytes | Meaning |
|---|---|
| 0 | `1` |
| 4 | `2` |
| 64..67 | magic `EF BE AD DE` (i.e. `0xDEADBEEF` little-endian) |
| 68 | version `1` |
| 72 | `u32 LE data_size` = fp16 payload byte count |
| 80 | `u32 LE data_offset` = `128` (absolute file offset where payload starts) |
| 128.. | fp16 payload, row-major `[rows, cols]` (built with `buildTransposed` for MIL `W* t` constants) |

This is exactly consistent with the MIL constants: `BLOBFILE(path = string("@model_path/weights/weight.bin"), offset = uint64(64))`
— `offset = 64` is where the **64-byte chunk header** begins (bytes 0..63 are a file-level
header: `[0]=1`, `[4]=2`), and the chunk's own `data_offset` field (=128) points at the payload.

**Fused multi-weight blobs** put one 64-byte header per weight; the fixture test proves it
(`Tests/MILGeneratorTests/MILGeneratorTests.swift#L242-L273`):
chunk `i` starts at `64 + i*(64 + payloadBytes)`; at `chunkStart+0` magic `EF BE AD DE`,
`+4` = `0x01`, `+8` = `u32 data_size`, `+16` = `u32 data_offset` = `chunkStart + 64`; payload
begins at `chunkStart + 64`. Example: `fused_ffn.mil` uses `W3` at
`offset = 64 + (64 + 6*4*2) = 176`, matching the fixture.

### 6.4 How weights are associated with MIL symbols

There is **no separate symbol table**: the association is by the MIL `path` string, which must
equal the weight-dictionary key (and the on-disk path):

```objc
weights[path] = @{@"offset": @0, @"data": wd};   // path == "@model_path/weights/weight.bin"
```
and in MIL:
```
tensor<fp16, [6, 4, 1, 1]> W = const()[name = string("W"), val = tensor<fp16, [6, 4, 1, 1]>(
    BLOBFILE(path = string("@model_path/weights/weight.bin"), offset = uint64(64)))];
```
The dict value's `@"offset"` is always `@0` in the shipped code; the real byte offset lives in
the MIL `BLOBFILE(...offset=…)`, i.e. multiple MIL symbols may share one weight file with
different offsets (this is how fused QKV/FFN kernels pack Wq/Wk/Wv or W1/W3 into one `.bin`).

---

## 7. (e) Evaluating — `doEvaluateDirectWithModel:` and friends

### 7.1 The eval function (verbatim, the whole decision ladder)

```objc
bool ane_interop_eval(ANEHandle *handle) {
    if (!handle) return false;
    if (__sync_fetch_and_add(&g_force_eval_failure, 0) != 0) return false;
    id mdl = (__bridge id)handle->model;
    id req = (__bridge id)handle->request;
    NSError *e = nil;
    NSDictionary *options = handle->evalOptions ? (__bridge NSDictionary *)handle->evalOptions : @{};

    BOOL ok = NO;
    ANEEvalPath evalPath = ane_interop_eval_path();
    if (evalPath == ANE_EVAL_REALTIME && !handle->realtimeLoaded) {
        // Real-time path may be unavailable on public builds; fall back to standard in-memory eval.
        evalPath = ANE_EVAL_INMEM;
    }

    const bool shouldTryClient = (evalPath != ANE_EVAL_INMEM) || handle->perfStatsRequested;
    if (shouldTryClient && handle->client && handle->clientModel) {
        id client = (__bridge id)handle->client;
        id modelObj = (__bridge id)handle->clientModel;
        if (evalPath == ANE_EVAL_REALTIME && handle->realtimeLoaded &&
            [client respondsToSelector:@selector(evaluateRealTimeWithModel:options:request:error:)]) {
            ok = ((BOOL(*)(id,SEL,id,id,id,NSError**))objc_msgSend)(
                client, @selector(evaluateRealTimeWithModel:options:request:error:), modelObj, options, req, &e);
        } else {
            SEL sel = @selector(evaluateWithModel:options:request:qos:error:);
            if (evalPath == ANE_EVAL_CLIENT_DIRECT || handle->perfStatsRequested) {
                SEL directSel = @selector(doEvaluateDirectWithModel:options:request:qos:error:);
                if ([client respondsToSelector:directSel]) {
                    sel = directSel;
                }
            }
            ok = ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(
                client, sel, modelObj, options, req, 21, &e);
        }
        if (!ok && ane_interop_trace_enabled()) {
            fprintf(stderr, "ANE client eval failed (will fallback): %s\n", e ? [[e description] UTF8String] : "no error");
        }
    }
    if (!ok) {
        e = nil;
        ok = ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
            mdl, @selector(evaluateWithQoS:options:request:error:), 21, options, req, &e);
    }
    if (!ok) {
        fprintf(stderr, "ANE eval failed: %s\n", e ? [[e description] UTF8String] : "no error");
        handle->lastHwExecutionTimeNS = 0;
    } else if (handle->perfStatsRequested) {
        id ps = nil;
        if ([req respondsToSelector:@selector(perfStats)]) ps = objc_msgSend(req, @selector(perfStats));
        if (!ps && [req respondsToSelector:@selector(perfStatsArray)]) {
            id arr = objc_msgSend(req, @selector(perfStatsArray));
            if ([arr isKindOfClass:[NSArray class]] && [arr count] > 0) ps = [arr objectAtIndex:0];
        }
        if (!ps) ps = handle->perfStats ? (__bridge id)handle->perfStats : nil;
        handle->lastHwExecutionTimeNS = ps ? objc_msgSend(ps, @selector(hwExecutionTime)) : 0;
    }
    return ok;
}
```
Source: `Sources/ANEInterop/ane_interop.m#L2342-L2410` (lightly reflowed; semantics unchanged).

Key facts:
* **Default path is in-memory-model eval**: `[mdl evaluateWithQoS:21 options:options request:req error:&e]`.
* The `_ANEClient` path (with `doEvaluateDirectWithModel:…` preferred) is only tried when
  `ANE_EVAL_PATH` is `client`/`clientDirect`/`realtime` **or** perf stats are requested; it is
  always followed by a fallback to `evaluateWithQoS:` on failure.
* QoS is the literal `21` in every call site. **UNCERTAIN** what 21 encodes (likely a
  QoS/priority class); the code never explains it, it is simply the constant that works.
* Eval is synchronous (no wait/poll loop); results are read straight from the output IOSurface
  after the call returns.

### 7.2 `ane_interop_rebind_input` (swap an input surface, rebuild request)

Checks `IOSurfaceGetAllocSize(newSurface) >= handle->inputBytes[index]`, rebuilds the wrapped
arrays + `_ANERequest`, retains the new surface, releases the old one and old request.
Source: `ane_interop.m#L2522-L2592`.

### 7.3 Shared events / completion handler (probes; not proven on hardware)

* Request factories with shared events:
  `requestWithInputs:inputIndices:outputs:outputIndices:perfStats:perfStatsMask:procedureIndex:sharedEvents:` (8-arg)
  and the 9-arg variant with `transactionHandle:`.
* `_ANESharedSignalEvent.signalEventWithValue:symbolIndex:eventType:sharedEvent:`
  (types: `unsigned long long value, unsigned int symbolIndex, long long eventType, id sharedEvent`).
* `_ANESharedWaitEvent.waitEventWithValue:sharedEvent:eventType:` (or the 3-arg form).
* `_ANESharedEvents.sharedEventsWithSignalEvents:waitEvents:(NSArray*, NSArray*)`.
* `[req setSharedEvents:container]`, `[req setCompletionHandler:^{…}]`, then
  `[mdl evaluateWithQoS:21 options:request:error:]`, with a 5 s semaphore wait, and
  `setCompletionHandler:` / `setSharedEvents:` set back to `nil` afterwards.
* `MTLSharedEvent` obtained by `dlopen("/System/Library/Frameworks/Metal.framework/Metal")` →
  `MTLCreateSystemDefaultDevice()` → `[device newSharedEvent]` → `[ev signaledValue]`.
Source: `ane_interop.m#L1044-L1072`, `#L3132-L3271`.
Project's own conclusion (blog): "The standard eval path hung on hardware before the completion
callback fired. Metal-based event signaling doesn't bridge to the ANE scheduler the way it does
to the GPU scheduler." → **Do not depend on this.**

---

## 8. Error codes, messages, retry logic

Compile error codes (`ane_interop.h#L18-L22`, returned by `ane_interop_last_compile_error()`):

| Code | Name | Trigger |
|---|---|---|
| 0 | `ANE_INTEROP_COMPILE_ERROR_NONE` | success |
| 1 | `ANE_INTEROP_COMPILE_ERROR_INVALID_ARGUMENTS` | NULL/zero args, bad weight path, path escape, duplicate/NaN shapes |
| 2 | `ANE_INTEROP_COMPILE_ERROR_DUPLICATE_WEIGHT_PATH` | same `path` key twice |
| 3 | `ANE_INTEROP_COMPILE_ERROR_SURFACE_ALLOCATION_FAILED` | `IOSurfaceCreate` returned NULL / `_ANEIOSurfaceObject` returned nil |
| 4 | `ANE_INTEROP_COMPILE_ERROR_COMPILER_FAILURE` | `modelWithMILText:`, `inMemoryModelWithDescriptor:`, `compileWithQoS:`, `loadWithQoS:`, `_ANERequest` factory, `hexStringIdentifier`, OOM |

Exact stderr strings emitted (useful for matching against a Zig port's logs):
`"ANE compile failed: %s\n"`, `"ANE load failed: %s\n"`, `"ANE eval failed: %s\n"`,
`"ANE client eval failed (will fallback): %s\n"`,
`"ANE compile retrying without options...\n"`,
`"ANE compile failed with strict options (no fallback): %s\n"`,
`"ANE compile cachePolicy=%d compiledExists=%d donorExists=%d options=%lu\n"`,
`"ANE compile: id=%s tmpdir=%s milLen=%zu weights=%d inputs=%d outputs=%d\n"`,
`"  weight: %s (%zu bytes)\n"`,
`"ANE compile failed: IOSurfaceCreate returned NULL (input %d)\n"`,
`"ANE compile failed: _ANEIOSurfaceObject returned nil (input %d)\n"`,
`"ANE compile failed: _ANERequest returned nil\n"`,
`"ANE compile failed: OOM allocating ANEHandle\n"`,
`"ANE realtime load failed: %s\n"`, `"ANE hwExecutionTime: %llu ns\n"`,
`"ANE perfStatsMask: 0x%08X\n"`. All are gated on `ANE_INTEROP_TRACE` except the failure ones.

Retry ladder (in order): compile with options → compile with `@{}` → (cache policy) load from
donor `net.plist` instead of compiling → load with options → load with `@{}` → hard failure.
Eval: client path → `evaluateWithQoS:` fallback. Never more than one level of fallback; every
step is `respondsToSelector:`-guarded, so a missing private selector degrades instead of crashing.

### 8.1 Environment variables (complete list read by `ane_interop.m`)

| Var | Effect |
|---|---|
| `ANE_INTEROP_TRACE` | verbose stderr tracing of every step |
| `ANE_EVAL_PATH` | `inmem` (default), `client`, `clientDirect`, `realtime` |
| `ANE_COMPILE_CACHE_POLICY` | `auto` (default), `preferCached`, `forceCold` |
| `ANE_INTEROP_CACHE_ROOT` | override compile-cache root |
| `ANE_QUEUE_DEPTH` | `setQueueDepth:` (char, clamped to 127) |
| `ANE_KEEP_TMPDIR` | keep the model directory for inspection |
| `ANE_PERF_STATS` | `1` → build `_ANEPerformanceStats` + set mask |
| `ANE_PERF_STATS_MASK` | ANEF mask, default `0xF` (M3 Max rejects bits outside `{1,2,4,8}`) |
| `ANE_STRICT_OPTIONS` | `1` → no options fallback retry |
| `ANE_DISABLE_POWER_SAVING`, `ANE_KEEP_MODEL_WIRED`, `ANE_ENABLE_LATE_LATCH`, `ANE_SKIP_PREPARE`, `ANE_ENABLE_FW_TO_FW_SIGNAL`, `ANE_DISABLE_IO_FENCES`, `ANE_MEMORY_POOL_ID`, `ANE_USE_COMPILER_OPTIONS` | load-option toggles (§5.6) |
| `ANE_INTEROP_CHAINING_PROBE_STATS_SURFACE` | `null` / `output0` / `scratch` (probe only) |

---

## 9. Dead ends (do not build on these)

1. **`_ANEVirtualClient`** — `+sharedConnection`, `-initWithSingletonAccess`, `-connect`,
   `-hasANE`, `+new`, `alloc/init`: every path returned `nil`.
   "classic IOKit entitlement gate. This API requires kernel-level entitlements that only Apple's
   own processes have." Its selectors do exist and are probed:
   `doEvaluateWithModel:options:request:qos:completionEvent:error:`,
   `doMapIOSurfacesWithModel:request:cacheInference:error:`,
   `loadModel:options:qos:error:`, `+getCodeSigningIdentity`, `+setCodeSigningIdentity:`
   (the setter is tried with `@"com.apple.coreml"`, `@"com.apple.appleNeuralEngine"`, `@"*"` —
   it crashes internally for plain strings; the file warns about `__setObject:forKey:`).
   Source: `ane_interop.m#L2594-L3128`, blog.
2. **Async eval via `_ANESharedEvents` + `MTLSharedEvent`** — standard eval path hung on hardware
   before the completion callback fired.
3. **Metal/ANE hybrid attention** — measured 1.8 ms/token *slower* than pure ANE.
4. **Multi-layer KV cache channel packing** — `InvalidMILProgram` at compile time (M3 Max, macOS 15).
5. **Chaining** (`_ANEChainingRequest`, `prepareChainingWithModel:options:chainingReq:qos:error:`)
   — implemented as a probe with ~26 stage codes; no evidence it works on hardware. Treat as
   experimental.

---

## 10. MIL: real, valid programs + dialect rules

### 10.1 Header and dialect

Every generator emits:

```
program(1.3)
[buildInfo = dict<string, string>({{"coremlc-component-MIL", "3510.2.1"}, {"coremlc-version", "3505.4.1"}, {"coremltools-component-milinternal", ""}, {"coremltools-version", "9.0"}})]
{
    func main<ios18>(...) {
        ...
    } -> (...);
}
```
Source: `Sources/MILGenerator/MILBuilder.swift#L28-L52`. `defaultDeploymentTarget = "ios18"`,
overridable at runtime via `ESPRESSO_MIL_DEPLOYMENT_TARGET`. (So even on macOS the function tag
is `ios18`.) `ANECodegen.swift` emits the same header text for its own programs.

### 10.2 `conv.mil` — complete fixture, verbatim

`Tests/MILGeneratorTests/Fixtures/conv.mil` (generated by `GenericMIL.conv(inCh: 4, outCh: 6, spatial: 2)`):

```
program(1.3)
[buildInfo = dict<string, string>({{"coremlc-component-MIL", "3510.2.1"}, {"coremlc-version", "3505.4.1"}, {"coremltools-component-milinternal", ""}, {"coremltools-version", "9.0"}})]
{
    func main<ios18>(tensor<fp32, [1, 4, 1, 2]> x) {
        string c_pad_type = const()[name = string("c_pad_type"), val = string("valid")];
        tensor<int32, [2]> c_strides = const()[name = string("c_strides"), val = tensor<int32, [2]>([1, 1])];
        tensor<int32, [4]> c_pad = const()[name = string("c_pad"), val = tensor<int32, [4]>([0, 0, 0, 0])];
        tensor<int32, [2]> c_dilations = const()[name = string("c_dilations"), val = tensor<int32, [2]>([1, 1])];
        int32 c_groups = const()[name = string("c_groups"), val = int32(1)];
        string to_fp16 = const()[name = string("to_fp16"), val = string("fp16")];
        tensor<fp16, [1, 4, 1, 2]> x16 = cast(dtype = to_fp16, x = x)[name = string("cast_in")];
        tensor<fp16, [6, 4, 1, 1]> W = const()[name = string("W"), val = tensor<fp16, [6, 4, 1, 1]>(BLOBFILE(path = string("@model_path/weights/weight.bin"), offset = uint64(64)))];
        tensor<fp16, [1, 6, 1, 2]> y16 = conv(dilations = c_dilations, groups = c_groups, pad = c_pad, pad_type = c_pad_type, strides = c_strides, weight = W, x = x16)[name = string("conv")];
        string to_fp32 = const()[name = string("to_fp32"), val = string("fp32")];
        tensor<fp32, [1, 6, 1, 2]> y = cast(dtype = to_fp32, x = y16)[name = string("cast_out")];
    } -> (y);
}
```

### 10.3 `fused_ffn.mil` — complete fixture, verbatim (two weights + one blob, offsets 64 and 176)

`Tests/MILGeneratorTests/Fixtures/fused_ffn.mil` (generated by `GenericMIL.fusedFFNUp(dim: 4, hiddenDim: 6, spatial: 2)`):

```
program(1.3)
[buildInfo = dict<string, string>({{"coremlc-component-MIL", "3510.2.1"}, {"coremlc-version", "3505.4.1"}, {"coremltools-component-milinternal", ""}, {"coremltools-version", "9.0"}})]
{
    func main<ios18>(tensor<fp32, [1, 4, 1, 2]> x) {
        string c_pad_type = const()[name = string("c_pad_type"), val = string("valid")];
        tensor<int32, [2]> c_strides = const()[name = string("c_strides"), val = tensor<int32, [2]>([1, 1])];
        tensor<int32, [4]> c_pad = const()[name = string("c_pad"), val = tensor<int32, [4]>([0, 0, 0, 0])];
        tensor<int32, [2]> c_dilations = const()[name = string("c_dilations"), val = tensor<int32, [2]>([1, 1])];
        int32 c_groups = const()[name = string("c_groups"), val = int32(1)];
        string to_fp16 = const()[name = string("to_fp16"), val = string("fp16")];
        tensor<fp16, [1, 4, 1, 2]> x16 = cast(dtype = to_fp16, x = x)[name = string("cast_in")];
        tensor<fp16, [6, 4, 1, 1]> W1 = const()[name = string("W1"), val = tensor<fp16, [6, 4, 1, 1]>(BLOBFILE(path = string("@model_path/weights/weight.bin"), offset = uint64(64)))];
        tensor<fp16, [6, 4, 1, 1]> W3 = const()[name = string("W3"), val = tensor<fp16, [6, 4, 1, 1]>(BLOBFILE(path = string("@model_path/weights/weight.bin"), offset = uint64(176)))];
        tensor<fp16, [1, 6, 1, 2]> h1 = conv(dilations = c_dilations, groups = c_groups, pad = c_pad, pad_type = c_pad_type, strides = c_strides, weight = W1, x = x16)[name = string("conv_w1")];
        tensor<fp16, [1, 6, 1, 2]> h3 = conv(dilations = c_dilations, groups = c_groups, pad = c_pad, pad_type = c_pad_type, strides = c_strides, weight = W3, x = x16)[name = string("conv_w3")];
        string to_fp32 = const()[name = string("to_fp32"), val = string("fp32")];
        tensor<fp32, [1, 6, 1, 2]> out1 = cast(dtype = to_fp32, x = h1)[name = string("cast_h1")];
        tensor<fp32, [1, 6, 1, 2]> out3 = cast(dtype = to_fp32, x = h3)[name = string("cast_h3")];
    } -> (out1, out3);
}
```
(Note: two outputs → two output IOSurfaces, `outputIndices = [0, 1]`.)

### 10.4 `matmul.mil` — third example, verbatim (weights + activations as *inputs*, no BLOBFILE)

`Tests/MILGeneratorTests/Fixtures/matmul.mil` (992 bytes, generated by `GenericMIL.matmul(inCh: 4, outCh: 6, spatial: 2)`):

```
program(1.3)
[buildInfo = dict<string, string>({{"coremlc-component-MIL", "3510.2.1"}, {"coremlc-version", "3505.4.1"}, {"coremltools-component-milinternal", ""}, {"coremltools-version", "9.0"}})]
{
    func main<ios18>(tensor<fp32, [1, 4, 2]> x, tensor<fp32, [1, 6, 4]> W) {
        string to_fp16 = const()[name = string("to_fp16"), val = string("fp16")];
        tensor<fp16, [1, 4, 2]> x16 = cast(dtype = to_fp16, x = x)[name = string("cast_x")];
        tensor<fp16, [1, 6, 4]> W16 = cast(dtype = to_fp16, x = W)[name = string("cast_W")];
        bool tx = const()[name = string("tx"), val = bool(false)];
        bool ty = const()[name = string("ty"), val = bool(false)];
        tensor<fp16, [1, 6, 2]> y16 = matmul(transpose_x = tx, transpose_y = ty, x = W16, y = x16)[name = string("mm")];
        string to_fp32 = const()[name = string("to_fp32"), val = string("fp32")];
        tensor<fp32, [1, 6, 2]> y = cast(dtype = to_fp32, x = y16)[name = string("cast_out")];
    } -> (y);
}
```

Other small fixtures available in the same directory: `fused_qkv.mil` (2460 B), `qkvb.mil`
(2466 B), `ffn_bwd.mil` (3538 B), `ffn_fwd_taps.mil` (3197 B), `sdpa_fwd_taps.mil` (5129 B),
`sdpa_bwd1.mil`, `sdpa_bwd2.mil`, `ffn_bwd.mil`. Binary reference blobs:
`ffn_blob_ref.bin` (288 B = 128 + 6*4*2 + 64… per the fused layout), `qkv_blob_ref.bin` (352 B).

### 10.5 MIL dialect gotchas (from `docs/reverse-engineering-apple-neural-engine.html`)

| Op | Status | Note |
|---|---|---|
| `conv` | works | ANE's native linear op — use for all matmuls (often faster than `matmul`) |
| `matmul` | works | |
| `softmax` | works* | power-of-2 dims only; 257 → fail, 256 → ok (fails at **eval** time) |
| `sigmoid`, `mul`, `add`, `reshape`, `transpose`, `reduce_sum`, `reduce_max`, `pow`, `concat` (12+ inputs) | works | |
| `rsqrt` | **broken** | use `pow(x, fp16(-0.5))` |
| `reduce_mean` | **missing** | use `reduce_sum` + divide |
| `cast(uint8→fp16)`, `constexpr_affine_dequantize` | **broken** | fp16 weights only |
| `slice_by_index` on function inputs + complex graph | **broken** | pre-shape outside the program |
| INT8/quantized weights | unsupported | |

Other stated rules: keep channel dims multiples of 64 / 64-byte alignment where possible;
`[1, C, 1, S]` with sequence along width; weights are baked at compile time; start from a
single-op identity kernel and grow one op at a time (InvalidMILProgram often comes from op
*combinations*); cold compile of a 6-layer recurrent model took 131 s, warm ~350 ms.

---

## 11. Surface-I/O helper semantics worth copying into Zig

* `ane_interop_io_copy(dst, dst_ch_off, src, src_ch_off, channels, spatial)` — copies
  `channels*spatial` fp16 elements, source index `src_ch_off*spatial + i`.
* `…_fp16_spatial_slice` — copies one token's `channels` values from `src_spatial_index` to
  `dst_spatial_index` (stride = `spatial`), i.e. KV-cache lane writes.
* `…_argmax_fp16_spatial_slice(+_with_hint)` — argmax over `channels` at one spatial position
  (token id read), optional "hint" surface carrying a `reduce_max` result for early exit;
  first-max semantic (lowest index wins) documented in the header.
* `ane_interop_io_lock_write/read` + `…_unlocked` variants — amortise lock cost across
  write→eval→read cycles. Locked write is `IOSurfaceLock(surf, 0, NULL)`; locked read is
  `IOSurfaceLock(surf, kIOSurfaceReadOnly, NULL)`; guard with `IOSurfaceGetAllocSize`.
* The header also exposes NEON fp16↔fp32 conversion, strided gather/scatter, a streaming
  fp16 GEMV argmax, and BNNS fp32 GEMV/GEMM with a `n_threads` argument — all pure CPU, but
  these are the functions the project actually uses for the LM head / attention on CPU.

---

## 12. Source URLs (pinned to commit `05ea6a3`)

* README — <https://raw.githubusercontent.com/christopherkarani/Espresso/05ea6a3da91be642abd3d4741c0341a941ce1064/README.md>
* `ane_interop.h` — <https://raw.githubusercontent.com/christopherkarani/Espresso/05ea6a3da91be642abd3d4741c0341a941ce1064/Sources/ANEInterop/include/ane_interop.h>
* `ane_interop.m` — <https://raw.githubusercontent.com/christopherkarani/Espresso/05ea6a3da91be642abd3d4741c0341a941ce1064/Sources/ANEInterop/ane_interop.m>
* `surface_io.c` — <https://raw.githubusercontent.com/christopherkarani/Espresso/05ea6a3da91be642abd3d4741c0341a941ce1064/Sources/ANEInterop/surface_io.c>
* `WeightBlob.swift` — <https://raw.githubusercontent.com/christopherkarani/Espresso/05ea6a3da91be642abd3d4741c0341a941ce1064/Sources/ANETypes/WeightBlob.swift>
* `ANEKernel.swift` — <https://raw.githubusercontent.com/christopherkarani/Espresso/05ea6a3da91be642abd3d4741c0341a941ce1064/Sources/ANERuntime/ANEKernel.swift>
* `GenericMIL.swift` — <https://raw.githubusercontent.com/christopherkarani/Espresso/05ea6a3da91be642abd3d4741c0341a941ce1064/Sources/MILGenerator/GenericMIL.swift>
* `MILBuilder.swift` — <https://raw.githubusercontent.com/christopherkarani/Espresso/05ea6a3da91be642abd3d4741c0341a941ce1064/Sources/MILGenerator/MILBuilder.swift>
* Fixtures — <https://raw.githubusercontent.com/christopherkarani/Espresso/05ea6a3da91be642abd3d4741c0341a941ce1064/Tests/MILGeneratorTests/Fixtures/> (`conv.mil`, `fused_ffn.mil`, `matmul.mil`, …)
* `Tests/ANERuntimeTests/Fixtures/README.md` — <https://raw.githubusercontent.com/christopherkarani/Espresso/05ea6a3da91be642abd3d4741c0341a941ce1064/Tests/ANERuntimeTests/Fixtures/README.md>
* Reverse-engineering write-up — <https://github.com/christopherkarani/Espresso/blob/main/docs/reverse-engineering-apple-neural-engine.html>

### `Tests/ANERuntimeTests/Fixtures/README.md` (verbatim, complete)

```
# ObjC Golden Fixtures (Phase 6b)

These files are required by ANERuntime cross-validation tests when
`OBJC_CROSS_VALIDATION=1` is enabled.

Expected files (raw little-endian `Float32` vectors):
- `fwd_attn_oOut_seq256_f32le.bin`
- `fwd_ffn_y_seq256_f32le.bin`
- `ffn_bwd_dx_seq256_f32le.bin`

Each fixture stores exactly `ModelConfig.dim * ModelConfig.seqLen` float values.

If fixtures are missing, cross-validation tests skip with an explicit message.
```

---

## 13. Open questions for the Zig port (marked UNCERTAIN)

1. Whether `_ANEInMemoryModel`/`_ANEClient` selectors above still exist unchanged on **macOS 27.0.1**.
   Mitigation: replicate `respondsToSelector:` probing (and `ANE_INTEROP_TRACE`-style class dumping)
   as the first runtime step.
2. Whether A18 Pro (iPhone-class ANE via macOS-on-A-series — only relevant if this runs on a
   Mac with an A-series SoC, which does not exist) needs the iOS entitlement the README mentions.
   The README's "Apple A-series (iOS) requires entitlement" row is about iOS targets, not macOS.
3. Meaning of the QoS literal `21`; whether other values are valid.
4. Why `pixelFormat = 0` works while the blog specifies `OneComponent16Half` — possibly the ANE
   only uses width/bytesPerRow/allocSize and ignores pixel format for non-display surfaces.
5. Whether the file-level weight header bytes `[0]=1`, `[4]=2` are load-bearing or incidental
   (they are never explained; only the chunk header at offset 64 is interpreted).
6. Whether `_ANEVirtualClient`/chaining ever works on macOS 27 with an Apple-internal signature —
   no evidence; assume no.
7. No source in the repo proves the A18 Pro ANE accepts the same MIL `program(1.3)` /
   `func main<ios18>` dialect; that dialect string is chosen for macOS 15 + M-series.
