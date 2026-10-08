// ane_slots_probe.m — does per-evaluation cost grow with the number of LOADED
// kernels?
//
// The engine's per-kernel timings are strange: a 1x1 conv with 2 MB of weights
// takes 6 ms (0.3 GB/s) while the lm_head with 272 MB takes 6.7 ms (40 GB/s).
// The standalone probe times the same small conv at ~300 us. The difference is
// that the engine keeps 60-90 kernels loaded at once.
//
// This probe loads K identical-shaped kernels and times one of them, so the
// answer is a number rather than a theory.
//
// Build: clang -fobjc-arc -O2 -framework Foundation -framework IOSurface -o ane_slots_probe ane_slots_probe.m

#import <Foundation/Foundation.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <IOSurface/IOSurface.h>
#import <mach/mach_time.h>

static Class g_Desc, g_IMM, g_Req, g_IO;
static mach_timebase_info_data_t g_tb;

static double ms_since(uint64_t t0) {
    return (double)(mach_absolute_time() - t0) * g_tb.numer / g_tb.denom / 1e6;
}

static IOSurfaceRef createSurface(size_t bytes) {
    return IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth: @(bytes), (id)kIOSurfaceHeight: @1,
        (id)kIOSurfaceBytesPerElement: @1, (id)kIOSurfaceBytesPerRow: @(bytes),
        (id)kIOSurfaceAllocSize: @(bytes), (id)kIOSurfacePixelFormat: @0});
}

// One weight file with a single fp16 tensor of `n` values.
static NSData *blobOne(const _Float16 *w, int n, uint32_t tag) {
    size_t total = 64 + 64 + (size_t)n * 2;
    uint8_t *buf = calloc(total, 1);
    buf[0] = 0x01; buf[4] = 0x02;
    buf[64] = 0xEF; buf[65] = 0xBE; buf[66] = 0xAD; buf[67] = 0xDE;
    buf[68] = 0x01;
    uint32_t sz = (uint32_t)n * 2, off = 128;
    memcpy(buf + 72, &sz, 4);
    memcpy(buf + 80, &off, 4);
    memcpy(buf + 128, w, (size_t)n * 2);
    (void)tag;
    return [NSData dataWithBytes:buf length:total];
}

// A kernel is a compiled+loaded model with one input and one output surface.
typedef struct {
    id model;
    IOSurfaceRef ioIn, ioOut;
    id req;
} Kernel;

static int g_width = 1;

static BOOL buildKernel(int cin, int cout, const _Float16 *w, uint32_t tag, Kernel *out) {
    NSString *mil = [NSString stringWithFormat:
        @"program(1.3)\n"
         "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, {\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, {\"coremltools-version\", \"9.0\"}})]\n"
         "{\n"
         "    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> i0) {\n"
         "        string c_pad_type = const()[name = string(\"c_pad_type\"), val = string(\"valid\")];\n"
         "        tensor<int32, [2]> c_strides = const()[name = string(\"c_strides\"), val = tensor<int32, [2]>([1, 1])];\n"
         "        tensor<int32, [4]> c_pad = const()[name = string(\"c_pad\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n"
         "        tensor<int32, [2]> c_dilations = const()[name = string(\"c_dilations\"), val = tensor<int32, [2]>([1, 1])];\n"
         "        int32 c_groups = const()[name = string(\"c_groups\"), val = int32(1)];\n"
         "        tensor<fp16, [%d, %d, 1, 1]> wa = const()[name = string(\"wa\"), val = tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/w.bin\"), offset = uint64(64)))];\n"
         "        tensor<fp16, [1, %d, 1, %d]> o0 = conv(dilations = c_dilations, groups = c_groups, pad = c_pad, pad_type = c_pad_type, strides = c_strides, weight = wa, x = i0)[name = string(\"o0\")];\n"
         "    } -> (o0);\n"
         "}\n", cin, g_width, cout, cin, cout, cin, cout, g_width];

    NSData *milData = [mil dataUsingEncoding:NSUTF8StringEncoding];
    NSData *blob = blobOne(w, cin * cout, tag);
    NSError *e = nil;
    NSDictionary *wdict = @{@"@model_path/weights/w.bin": @{@"offset": @0, @"data": [NSData data]}};
    id desc = ((id(*)(Class, SEL, id, id, id))objc_msgSend)(g_Desc, @selector(modelWithMILText:weights:optionsPlist:), milData, wdict, nil);
    id model = ((id(*)(Class, SEL, id))objc_msgSend)(g_IMM, @selector(inMemoryModelWithDescriptor:), desc);
    NSString *hex = ((id(*)(id, SEL))objc_msgSend)(model, @selector(hexStringIdentifier));
    NSString *td = [NSTemporaryDirectory() stringByAppendingPathComponent:hex];
    NSFileManager *fm = [NSFileManager defaultManager];
    [fm createDirectoryAtPath:[td stringByAppendingPathComponent:@"weights"] withIntermediateDirectories:YES attributes:nil error:nil];
    [milData writeToFile:[td stringByAppendingPathComponent:@"model.mil"] atomically:YES];
    [blob writeToFile:[td stringByAppendingPathComponent:@"weights/w.bin"] atomically:YES];

    if (!((BOOL(*)(id, SEL, unsigned int, id, NSError **))objc_msgSend)(model, @selector(compileWithQoS:options:error:), 21, @{}, &e)) {
        printf("  compile failed: %s\n", e ? [[e description] UTF8String] : "?");
        return NO;
    }
    BOOL loaded = NO;
    for (int attempt = 0; attempt < 7 && !loaded; attempt++) {
        if (attempt) usleep(200000 * attempt);
        e = nil;
        loaded = ((BOOL(*)(id, SEL, unsigned int, id, NSError **))objc_msgSend)(model, @selector(loadWithQoS:options:error:), 21, @{}, &e);
    }
    if (!loaded) {
        printf("  load failed: %s\n", e ? [[e description] UTF8String] : "?");
        return NO;
    }
    [fm removeItemAtPath:td error:nil];

    id attrs = ((id(*)(id, SEL))objc_msgSend)(model, @selector(modelAttributes));
    NSArray *statusList = attrs[@"NetworkStatusList"];
    NSDictionary *status = statusList.count ? statusList[0] : nil;
    size_t inBytes = (size_t)[[status[@"LiveInputList"] firstObject][@"BatchStride"] longLongValue];
    size_t outBytes = (size_t)[[status[@"LiveOutputList"] firstObject][@"BatchStride"] longLongValue];
    IOSurfaceRef ioIn = createSurface(inBytes), ioOut = createSurface(outBytes);
    id wIn = ((id(*)(Class, SEL, IOSurfaceRef))objc_msgSend)(g_IO, @selector(objectWithIOSurface:), ioIn);
    id wOut = ((id(*)(Class, SEL, IOSurfaceRef))objc_msgSend)(g_IO, @selector(objectWithIOSurface:), ioOut);
    id req = ((id(*)(Class, SEL, id, id, id, id, id, id, id))objc_msgSend)(g_Req,
        @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
        @[wIn], @[@0], @[wOut], @[@0], nil, nil, @0);
    out->model = model;
    out->ioIn = ioIn;
    out->ioOut = ioOut;
    out->req = req;
    return YES;
}

static double timeEval(Kernel *k, int iters) {
    NSError *e = nil;
    for (int i = 0; i < 3; i++)
        ((BOOL(*)(id, SEL, unsigned int, id, id, NSError **))objc_msgSend)(k->model, @selector(evaluateWithQoS:options:request:error:), 21, @{}, k->req, &e);
    uint64_t t0 = mach_absolute_time();
    for (int i = 0; i < iters; i++)
        ((BOOL(*)(id, SEL, unsigned int, id, id, NSError **))objc_msgSend)(k->model, @selector(evaluateWithQoS:options:request:error:), 21, @{}, k->req, &e);
    return ms_since(t0) / iters;
}

/// Time reading one column of the output surface (what the engine does) vs
/// reading the whole thing.
static double timeColumnRead(Kernel *k, int cout, int width) {
    IOSurfaceLock(k->ioOut, kIOSurfaceLockReadOnly, NULL);
    uint8_t *base = (uint8_t *)IOSurfaceGetBaseAddress(k->ioOut);
    uint64_t t0 = mach_absolute_time();
    volatile float sink = 0;
    for (int rep = 0; rep < 200; rep++) {
        for (int c = 0; c < cout; c++) {
            sink += (float)((_Float16 *)(base + (size_t)c * 64))[0];
        }
    }
    double ms = ms_since(t0) / 200.0;
    IOSurfaceUnlock(k->ioOut, kIOSurfaceLockReadOnly, NULL);
    (void)width;
    return ms;
}

static double timeFullRead(Kernel *k, int cout, int width) {
    IOSurfaceLock(k->ioOut, kIOSurfaceLockReadOnly, NULL);
    uint8_t *base = (uint8_t *)IOSurfaceGetBaseAddress(k->ioOut);
    size_t total = (size_t)cout * width * 2;
    uint64_t t0 = mach_absolute_time();
    volatile float sink = 0;
    for (int rep = 0; rep < 200; rep++) {
        for (size_t i = 0; i < total / 2; i++) sink += (float)((_Float16 *)base)[i];
    }
    double ms = ms_since(t0) / 200.0;
    IOSurfaceUnlock(k->ioOut, kIOSurfaceLockReadOnly, NULL);
    (void)width;
    return ms;
}

static int g_probe_mode = 0;   // 0 = width sweep, 1 = io mode compare

int main(int argc, char **argv) {
    @autoreleasepool {
        mach_timebase_info(&g_tb);
        dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);
        g_Desc = NSClassFromString(@"_ANEInMemoryModelDescriptor");
        g_IMM = NSClassFromString(@"_ANEInMemoryModel");
        g_Req = NSClassFromString(@"_ANERequest");
        g_IO = NSClassFromString(@"_ANIEOsurfaceObject");
        if (!g_IO) g_IO = NSClassFromString(@"_ANEIOSurfaceObject");

        const int cin = 896, cout = 1152;   // Qwen2.5-0.5B qkv shape
        _Float16 *w = malloc((size_t)cin * cout * 2);
        for (int i = 0; i < cin * cout; i++) w[i] = (_Float16)(0.01f * ((i % 97) - 48));

        if (argc > 1 && strcmp(argv[1], "io") == 0) g_probe_mode = 1;

        if (g_probe_mode == 1) {
            printf("eval + I/O cost by activation width (conv %dx%d, fp16)\n", cin, cout);
            printf("  width   eval ms   GB/s(w)   col-read ms   full-read ms\n");
            for (int width = 1; width <= 64; width *= 4) {
                g_width = width;
                Kernel k;
                if (!buildKernel(cin, cout, w, 0, &k)) continue;
                double ems = timeEval(&k, 50);
                double col = timeColumnRead(&k, cout, width);
                double full = timeFullRead(&k, cout, width);
                printf("  %5d   %7.3f   %7.2f   %11.3f   %11.3f\n", width, ems,
                       (double)cin * cout * 2 / (ems / 1000.0) / 1e9, col, full);
                NSError *e = nil;
                ((BOOL(*)(id, SEL, unsigned int, NSError **))objc_msgSend)(k.model, @selector(unloadWithQoS:error:), 21, &e);
                CFRelease(k.ioIn);
                CFRelease(k.ioOut);
            }
            return 0;
        }

        printf("per-eval cost vs number of loaded kernels (conv %dx%d, fp16)\n", cin, cout);
        printf("%s\n", "  K  ms/eval");
        for (int K = 1; K <= 32; K *= 2) {
            Kernel ks[32];
            int built = 0;
            for (int i = 0; i < K; i++) {
                if (!buildKernel(cin, cout, w, (uint32_t)i, &ks[i])) break;
                built++;
            }
            if (built < K) { printf("  %2d  (only %d built)\n", K, built); }
            double ms = timeEval(&ks[0], 50);
            printf("  %2d  %7.3f   (%.2f GB/s of weights)\n", built, ms,
                   (double)cin * cout * 2 / (ms / 1000.0) / 1e9);
            for (int i = 0; i < built; i++) {
                NSError *e = nil;
                ((BOOL(*)(id, SEL, unsigned int, NSError **))objc_msgSend)(ks[i].model, @selector(unloadWithQoS:error:), 21, &e);
                CFRelease(ks[i].ioIn);
                CFRelease(ks[i].ioOut);
            }
        }
        return 0;
    }
}
