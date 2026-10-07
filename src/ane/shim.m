// ane_shim.m — Objective-C implementation of the ANE bridge.
//
// Verified working on A18 Pro / macOS 27.0.1 with SIP enabled, unsigned binary,
// no entitlements: dlopen of AppleNeuralEngine.framework succeeds, _ANEClient
// resolves, ANECCompile() and load succeed, and fp16 conv kernels produce
// numerically correct results against a CPU reference.
//
// Build: clang -fobjc-arc -O2 -c shim.m   (linked with -framework Foundation -framework IOSurface)

#import <Foundation/Foundation.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <IOSurface/IOSurface.h>
#import <mach/mach_time.h>
#include <CommonCrypto/CommonDigest.h>
#include "shim.h"

// ---------------------------------------------------------------- globals

static Class g_Desc = nil;   // _ANEInMemoryModelDescriptor
static Class g_IMM = nil;    // _ANEInMemoryModel
static Class g_Req = nil;    // _ANERequest
static Class g_IO = nil;     // _ANEIOSurfaceObject
static char kInInfoKey;      // associated-object keys (stable addresses)
static char kOutInfoKey;
static bool g_ready = false;
static mach_timebase_info_data_t g_tb;
static uint64_t g_write_ns = 0, g_compile_ns = 0, g_load_ns = 0;

static uint64_t ns_since(uint64_t t0) {
    uint64_t dt = mach_absolute_time() - t0;
    return (uint64_t)((double)dt * g_tb.numer / g_tb.denom);
}
uint64_t ane_shim_write_ns(void) { return g_write_ns; }
uint64_t ane_shim_compile_ns(void) { return g_compile_ns; }
uint64_t ane_shim_load_ns(void) { return g_load_ns; }
static int g_compile_count = 0;
static char g_err[4096];

static void set_err(NSString *msg) {
    if (!msg) msg = @"(no message)";
    snprintf(g_err, sizeof(g_err), "%s", [msg UTF8String]);
}

const char *ane_shim_last_error(void) { return g_err; }
int ane_shim_compile_count(void) { return g_compile_count; }
int ane_shim_ready(void) { return g_ready ? 1 : 0; }

int ane_shim_init(void) {
    if (g_ready) return 0;
    void *h = dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);
    if (!h) {
        set_err([NSString stringWithFormat:@"dlopen AppleNeuralEngine failed: %s", dlerror()]);
        return -1;
    }
    g_Desc = NSClassFromString(@"_ANEInMemoryModelDescriptor");
    g_IMM  = NSClassFromString(@"_ANEInMemoryModel");
    g_Req  = NSClassFromString(@"_ANERequest");
    g_IO   = NSClassFromString(@"_ANEIOSurfaceObject");
    if (!g_Desc || !g_IMM || !g_Req || !g_IO) {
        set_err(@"failed to resolve _ANEInMemoryModelDescriptor/_ANEInMemoryModel/_ANERequest/_ANEIOSurfaceObject");
        return -1;
    }
    mach_timebase_info(&g_tb);
    g_ready = true;
    g_err[0] = '\0';
    return 0;
}

// ---------------------------------------------------------------- kernel

@interface ANEKernelImpl : NSObject
@property (nonatomic, strong) id model;
@property (nonatomic, strong) id request;
@property (nonatomic, strong) NSString *tmpDir;
@property (nonatomic, strong) NSMutableArray *inSurfaces;   // bridged IOSurfaceRef
@property (nonatomic, strong) NSMutableArray *outSurfaces;
@property (nonatomic, strong) NSMutableArray *inObjects;    // _ANEIOSurfaceObject
@property (nonatomic, strong) NSMutableArray *outObjects;
@property (nonatomic, assign) int nInputs;
@property (nonatomic, assign) int nOutputs;
@property (nonatomic, assign) size_t *inCaps;
@property (nonatomic, assign) size_t *outCaps;
@property (nonatomic, assign) uint64_t lastEvalNs;
@end

@implementation ANEKernelImpl
- (void)dealloc {
    free(_inCaps);
    free(_outCaps);
    if (_tmpDir) [[NSFileManager defaultManager] removeItemAtPath:_tmpDir error:nil];
}
@end

// Minimum IOSurface allocation. The ANE daemon on some OS versions rejects
// surfaces below 49152 bytes; oversizing is harmless because the ANE always
// uses its declared planar layout regardless of surface capacity (verified).
#define ANE_MIN_SURFACE_BYTES 49152

static IOSurfaceRef create_surface(size_t bytes) {
    return IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth: @(bytes),
        (id)kIOSurfaceHeight: @1,
        (id)kIOSurfaceBytesPerElement: @1,
        (id)kIOSurfaceBytesPerRow: @(bytes),
        (id)kIOSurfaceAllocSize: @(bytes),
        (id)kIOSurfacePixelFormat: @0
    });
}

// Pull LiveInputList/LiveOutputList out of modelAttributes:
//   modelAttributes = { ANEFModelDescription = {...}, NetworkStatusList = ( { LiveInputList = ( {...}, ... ), LiveOutputList = (...), Name = main }, ... ) }
static int extract_live(NSDictionary *attrs, NSString *key, ANETensorInfo *out, int max) {
    NSArray *statusList = attrs[@"NetworkStatusList"];
    if (![statusList isKindOfClass:[NSArray class]] || statusList.count == 0) return 0;
    NSDictionary *status = statusList[0];
    if (![status isKindOfClass:[NSDictionary class]]) return 0;
    NSArray *list = status[key];
    if (![list isKindOfClass:[NSArray class]]) return 0;

    int n = 0;
    for (id e in list) {
        if (![e isKindOfClass:[NSDictionary class]] || n >= max) continue;
        ANETensorInfo t;
        memset(&t, 0, sizeof(t));
        NSString *name = e[@"Name"];
        if (name) snprintf(t.name, sizeof(t.name), "%s", [[name description] UTF8String]);
        NSString *type = e[@"Type"];
        t.dtype = ANE_DTYPE_UNKNOWN;
        if (type) {
            if ([type containsString:@"Float16"]) t.dtype = ANE_DTYPE_FP16;
            else if ([type containsString:@"Float32"]) t.dtype = ANE_DTYPE_FP32;
        }
        t.batches      = [e[@"Batches"] intValue];
        t.channels     = [e[@"Channels"] intValue];
        t.height       = [e[@"Height"] intValue];
        t.width        = [e[@"Width"] intValue];
        t.plane_stride = (size_t)[e[@"PlaneStride"] longLongValue];
        t.row_stride   = (size_t)[e[@"RowStride"] longLongValue];
        t.depth_stride = (size_t)[e[@"DepthStride"] longLongValue];
        t.batch_stride = (size_t)[e[@"BatchStride"] longLongValue];
        if (t.batch_stride == 0) t.batch_stride = (size_t)t.channels * t.plane_stride;
        if (t.batches == 0) t.batches = 1;
        t.nbytes = t.batch_stride * (size_t)t.batches;
        out[n++] = t;
    }
    return n;
}

static BOOL write_all_fd(int fd, const uint8_t *bytes, size_t len) {
    size_t off = 0;
    while (off < len) {
        ssize_t n = write(fd, bytes + off, len - off);
        if (n <= 0) return NO;
        off += (size_t)n;
    }
    return YES;
}

/// Writes one ANE weight file: 64-byte file header, then per chunk a 64-byte
/// chunk header followed by the fp16 payload. Streamed straight from the
/// caller's buffers so no full-file copy is needed.
static BOOL write_weight_file(NSString *path, const ANEWeightFile *wf) {
    int fd = open([path fileSystemRepresentation], O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) return NO;
    uint8_t header[64];
    memset(header, 0, sizeof(header));
    header[0] = 0x01;
    header[4] = 0x02;
    BOOL ok = write_all_fd(fd, header, sizeof(header));
    // Each chunk header records the ABSOLUTE offset of its own payload, not a
    // constant: the published ffn_blob_ref.bin has 128 for the first chunk and
    // 240 for the second. Writing a constant here compiles fine and then
    // silently feeds the wrong weights to every chunk after the first.
    size_t pos = sizeof(header);
    for (int i = 0; ok && i < wf->n_chunks; i++) {
        uint8_t chunk[64];
        memset(chunk, 0, sizeof(chunk));
        chunk[0] = 0xEF; chunk[1] = 0xBE; chunk[2] = 0xAD; chunk[3] = 0xDE;
        chunk[4] = 0x01;
        uint32_t size = (uint32_t)wf->chunk_sizes[i];
        memcpy(chunk + 8, &size, 4);
        uint32_t data_off = (uint32_t)(pos + sizeof(chunk));
        memcpy(chunk + 16, &data_off, 4);
        ok = write_all_fd(fd, chunk, sizeof(chunk)) &&
             write_all_fd(fd, wf->chunk_data[i], wf->chunk_sizes[i]);
        pos += sizeof(chunk) + wf->chunk_sizes[i];
    }
    close(fd);
    return ok;
}

ANEKernel *ane_shim_kernel_create(const char *mil, size_t mil_len,
                                  const ANEWeightFile *files, int n_files) {
    @autoreleasepool {
        if (ane_shim_init() != 0) return NULL;
        NSError *e = nil;

        NSData *milData = [NSData dataWithBytes:mil length:mil_len];

        // The compiler reads the weights from $TMPDIR, not from this dictionary
        // (verified: empty data compiles to identical results). The dictionary
        // *does* feed the model hash, though, so it must carry a content digest:
        // passing empty data would make two different models with the same MIL
        // shapes share a cache entry and silently reuse the wrong weights.
        NSMutableDictionary *wdict = [NSMutableDictionary dictionary];
        for (int i = 0; i < n_files; i++) {
            NSString *name = [NSString stringWithUTF8String:files[i].name];
            CC_SHA256_CTX ctx;
            CC_SHA256_Init(&ctx);
            for (int c = 0; c < files[i].n_chunks; c++) {
                CC_SHA256_Update(&ctx, files[i].chunk_data[c], (CC_LONG)files[i].chunk_sizes[c]);
            }
            unsigned char md[CC_SHA256_DIGEST_LENGTH];
            CC_SHA256_Final(md, &ctx);
            wdict[name] = @{@"offset": @0, @"data": [NSData dataWithBytes:md length:sizeof(md)]};
        }

        id desc = ((id(*)(Class, SEL, id, id, id))objc_msgSend)(
            g_Desc, @selector(modelWithMILText:weights:optionsPlist:),
            milData, wdict.count ? wdict : @{}, nil);
        if (!desc) { set_err(@"modelWithMILText:weights:optionsPlist: returned nil"); return NULL; }

        id mdl = ((id(*)(Class, SEL, id))objc_msgSend)(g_IMM, @selector(inMemoryModelWithDescriptor:), desc);
        if (!mdl) { set_err(@"inMemoryModelWithDescriptor: returned nil"); return NULL; }

        // The compiler reads MIL + weights from $TMPDIR/<hexStringIdentifier>/.
        NSString *hex = ((id(*)(id, SEL))objc_msgSend)(mdl, @selector(hexStringIdentifier));
        NSString *td = [NSTemporaryDirectory() stringByAppendingPathComponent:hex];
        NSFileManager *fm = [NSFileManager defaultManager];
        [fm createDirectoryAtPath:[td stringByAppendingPathComponent:@"weights"]
      withIntermediateDirectories:YES attributes:nil error:nil];
        if (![milData writeToFile:[td stringByAppendingPathComponent:@"model.mil"] atomically:YES]) {
            set_err(@"could not write model.mil to the ANE temp directory");
            return NULL;
        }
        for (int i = 0; i < n_files; i++) {
            NSString *name = [NSString stringWithUTF8String:files[i].name];
            NSString *rel = name;
            if ([name hasPrefix:@"@model_path/"]) rel = [name substringFromIndex:12];
            NSString *full = [td stringByAppendingPathComponent:rel];
            [fm createDirectoryAtPath:[full stringByDeletingLastPathComponent]
          withIntermediateDirectories:YES attributes:nil error:nil];
            uint64_t tw0 = mach_absolute_time();
            BOOL wrote = write_weight_file(full, &files[i]);
            g_write_ns += ns_since(tw0);
            if (!wrote) {
                set_err([NSString stringWithFormat:@"could not write weight file %@", rel]);
                return NULL;
            }
        }

        // Compile (weights are baked in here) then load.
        uint64_t tc0 = mach_absolute_time();
        BOOL compiled = ((BOOL(*)(id, SEL, unsigned int, id, NSError **))objc_msgSend)(
            mdl, @selector(compileWithQoS:options:error:), 21, @{}, &e);
        g_compile_ns += ns_since(tc0);
        if (!compiled) {
            set_err([NSString stringWithFormat:@"ANECCompile failed: %@", e ? [e description] : @"unknown"]);
            [fm removeItemAtPath:td error:nil];
            return NULL;
        }
        g_compile_count++;

        // Loading can fail with "no ANE resources (transient; retry)" when the
        // program pool is busy — notably when another process is already
        // serving a model, because the pool is shared machine-wide. Back off
        // and retry before giving up.
        uint64_t tl0 = mach_absolute_time();
        BOOL loaded = NO;
        NSError *last_err = nil;
        const useconds_t backoff_us[6] = { 100000, 200000, 400000, 800000, 1600000, 3200000 };
        for (int attempt = 0; attempt < 7 && !loaded; attempt++) {
            if (attempt > 0) usleep(backoff_us[attempt - 1]);
            e = nil;
            loaded = ((BOOL(*)(id, SEL, unsigned int, id, NSError **))objc_msgSend)(
                mdl, @selector(loadWithQoS:options:error:), 21, @{}, &e);
            if (!loaded) last_err = e;
        }
        g_load_ns += ns_since(tl0);
        if (!loaded) {
            set_err([NSString stringWithFormat:@"ANE load failed after retries: %@", last_err ? [last_err description] : @"unknown"]);
            [fm removeItemAtPath:td error:nil];
            return NULL;
        }

        // The compiled program lives in the ANE daemon's own cache; the MIL text
        // and weight blobs in $TMPDIR are only inputs to ANECCompile(). They can
        // be several hundred MB per model, so drop them as soon as the model is
        // loaded. ANEDVD_KEEP_ANE_FILES=1 keeps them for debugging.
        const char *keep_files = getenv("ANEDVD_KEEP_ANE_FILES");
        if (!keep_files || keep_files[0] == '0' || keep_files[0] == '\0') {
            [fm removeItemAtPath:td error:nil];
        }

        // ---- declared layout ----
        ANEKernelImpl *k = [ANEKernelImpl new];
        k.model = mdl;
        k.tmpDir = td;
        k.lastEvalNs = 0;

        id attrs = ((id(*)(id, SEL))objc_msgSend)(mdl, @selector(modelAttributes));
        ANETensorInfo inInfo[16], outInfo[16];
        int ni = 0, no = 0;
        if ([attrs isKindOfClass:[NSDictionary class]]) {
            ni = extract_live(attrs, @"LiveInputList", inInfo, 16);
            no = extract_live(attrs, @"LiveOutputList", outInfo, 16);
        }
        if (ni <= 0 || no <= 0) {
            set_err(@"could not read LiveInputList/LiveOutputList from modelAttributes");
            return NULL;
        }
        k.nInputs = ni;
        k.nOutputs = no;

        // All input surfaces share one capacity; likewise for outputs.
        size_t inCap = ANE_MIN_SURFACE_BYTES, outCap = ANE_MIN_SURFACE_BYTES;
        for (int i = 0; i < ni; i++) if (inInfo[i].nbytes > inCap) inCap = inInfo[i].nbytes;
        for (int i = 0; i < no; i++) if (outInfo[i].nbytes > outCap) outCap = outInfo[i].nbytes;

        k.inCaps = (size_t *)calloc(ni, sizeof(size_t));
        k.outCaps = (size_t *)calloc(no, sizeof(size_t));
        k.inSurfaces = [NSMutableArray arrayWithCapacity:ni];
        k.outSurfaces = [NSMutableArray arrayWithCapacity:no];
        k.inObjects = [NSMutableArray arrayWithCapacity:ni];
        k.outObjects = [NSMutableArray arrayWithCapacity:no];

        NSMutableArray *inIdx = [NSMutableArray arrayWithCapacity:ni];
        NSMutableArray *outIdx = [NSMutableArray arrayWithCapacity:no];
        for (int i = 0; i < ni; i++) {
            IOSurfaceRef s = create_surface(inCap);
            if (!s) { set_err(@"IOSurfaceCreate failed for an input"); return NULL; }
            k.inCaps[i] = inCap;
            [k.inSurfaces addObject:(__bridge id)s];
            CFRelease(s);
            id obj = ((id(*)(Class, SEL, IOSurfaceRef))objc_msgSend)(g_IO, @selector(objectWithIOSurface:), s);
            [k.inObjects addObject:obj];
            [inIdx addObject:@(i)];
        }
        for (int i = 0; i < no; i++) {
            IOSurfaceRef s = create_surface(outCap);
            if (!s) { set_err(@"IOSurfaceCreate failed for an output"); return NULL; }
            k.outCaps[i] = outCap;
            [k.outSurfaces addObject:(__bridge id)s];
            CFRelease(s);
            id obj = ((id(*)(Class, SEL, IOSurfaceRef))objc_msgSend)(g_IO, @selector(objectWithIOSurface:), s);
            [k.outObjects addObject:obj];
            [outIdx addObject:@(i)];
        }

        k.request = ((id(*)(Class, SEL, id, id, id, id, id, id, id))objc_msgSend)(
            g_Req,
            @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
            k.inObjects, inIdx, k.outObjects, outIdx, nil, nil, @0);
        if (!k.request) { set_err(@"_ANERequest creation failed"); return NULL; }

        // Stash the declared layout inside the object for later queries.
        objc_setAssociatedObject(k, &kInInfoKey, [NSData dataWithBytes:inInfo length:sizeof(inInfo)], OBJC_ASSOCIATION_RETAIN);
        objc_setAssociatedObject(k, &kOutInfoKey, [NSData dataWithBytes:outInfo length:sizeof(outInfo)], OBJC_ASSOCIATION_RETAIN);

        g_err[0] = '\0';
        return (__bridge_retained void *)k;
    }
}

static ANEKernelImpl *impl(ANEKernel *k) { return (__bridge ANEKernelImpl *)k; }

void ane_shim_kernel_free(ANEKernel *k) {
    @autoreleasepool {
        if (!k) return;
        ANEKernelImpl *ki = impl(k);
        if (ki.model) {
            NSError *e = nil;
            SEL unload = @selector(unloadWithQoS:error:);
            if ([ki.model respondsToSelector:unload])
                ((BOOL(*)(id, SEL, unsigned int, NSError **))objc_msgSend)(ki.model, unload, 21, &e);
        }
        // Consume the +1 reference handed out by __bridge_retained in create().
        id released = (__bridge_transfer id)(void *)k;
        (void)released;
    }
}

int ane_shim_kernel_eval(ANEKernel *k) {
    @autoreleasepool {
        if (!k) { set_err(@"eval on NULL kernel"); return 0; }
        ANEKernelImpl *ki = impl(k);
        NSError *e = nil;
        uint64_t t0 = mach_absolute_time();
        BOOL ok = ((BOOL(*)(id, SEL, unsigned int, id, id, NSError **))objc_msgSend)(
            ki.model, @selector(evaluateWithQoS:options:request:error:), 21, @{}, ki.request, &e);
        uint64_t t1 = mach_absolute_time();
        ki.lastEvalNs = t1 - t0;
        if (!ok) {
            set_err([NSString stringWithFormat:@"evaluate failed: %@", e ? [e description] : @"unknown"]);
            return 0;
        }
        return 1;
    }
}

int ane_shim_input_count(const ANEKernel *k) { return k ? impl((ANEKernel *)k).nInputs : 0; }
int ane_shim_output_count(const ANEKernel *k) { return k ? impl((ANEKernel *)k).nOutputs : 0; }

static int info_for(const ANEKernel *k, int idx, int isOut, ANETensorInfo *out) {
    if (!k || !out) return 0;
    ANEKernelImpl *ki = impl((ANEKernel *)k);
    NSData *d = objc_getAssociatedObject(ki, isOut ? &kOutInfoKey : &kInInfoKey);
    int n = isOut ? ki.nOutputs : ki.nInputs;
    if (!d || idx < 0 || idx >= n || n > 16) return 0;
    const ANETensorInfo *arr = (const ANETensorInfo *)[d bytes];
    *out = arr[idx];
    return 1;
}

int ane_shim_input_info(const ANEKernel *k, int idx, ANETensorInfo *out) { return info_for(k, idx, 0, out); }
int ane_shim_output_info(const ANEKernel *k, int idx, ANETensorInfo *out) { return info_for(k, idx, 1, out); }

static IOSurfaceRef surface_at(const ANEKernel *k, int idx, int isOut) {
    if (!k) return NULL;
    ANEKernelImpl *ki = impl((ANEKernel *)k);
    NSArray *a = isOut ? ki.outSurfaces : ki.inSurfaces;
    if (idx < 0 || idx >= (int)a.count) return NULL;
    return (__bridge IOSurfaceRef)a[idx];
}

void *ane_shim_input_base(const ANEKernel *k, int idx) {
    IOSurfaceRef s = surface_at(k, idx, 0);
    return s ? IOSurfaceGetBaseAddress(s) : NULL;
}
void *ane_shim_output_base(const ANEKernel *k, int idx) {
    IOSurfaceRef s = surface_at(k, idx, 1);
    return s ? IOSurfaceGetBaseAddress(s) : NULL;
}
int ane_shim_input_lock(const ANEKernel *k, int idx) {
    IOSurfaceRef s = surface_at(k, idx, 0);
    return s ? (IOSurfaceLock(s, 0, NULL) == 0) : 0;
}
int ane_shim_input_unlock(const ANEKernel *k, int idx) {
    IOSurfaceRef s = surface_at(k, idx, 0);
    return s ? (IOSurfaceUnlock(s, 0, NULL) == 0) : 0;
}
int ane_shim_output_lock(const ANEKernel *k, int idx) {
    IOSurfaceRef s = surface_at(k, idx, 1);
    return s ? (IOSurfaceLock(s, kIOSurfaceLockReadOnly, NULL) == 0) : 0;
}
int ane_shim_output_unlock(const ANEKernel *k, int idx) {
    IOSurfaceRef s = surface_at(k, idx, 1);
    return s ? (IOSurfaceUnlock(s, kIOSurfaceLockReadOnly, NULL) == 0) : 0;
}
size_t ane_shim_input_capacity(const ANEKernel *k, int idx) {
    if (!k) return 0;
    ANEKernelImpl *ki = impl((ANEKernel *)k);
    return (idx >= 0 && idx < ki.nInputs) ? ki.inCaps[idx] : 0;
}
size_t ane_shim_output_capacity(const ANEKernel *k, int idx) {
    if (!k) return 0;
    ANEKernelImpl *ki = impl((ANEKernel *)k);
    return (idx >= 0 && idx < ki.nOutputs) ? ki.outCaps[idx] : 0;
}
uint64_t ane_shim_last_eval_ns(const ANEKernel *k) { return k ? impl((ANEKernel *)k).lastEvalNs : 0; }
