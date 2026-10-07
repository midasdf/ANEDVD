// ane_probe.m — minimal CoreML-free ANE probe.
//
// Verifies on this machine:
//   1. AppleNeuralEngine.framework loads and the private classes resolve.
//   2. A hand-written MIL program (conv with a BLOBFILE weight) compiles via
//      _ANEInMemoryModelDescriptor / _ANEInMemoryModel (no CoreML anywhere).
//   3. The model loads and evaluates against IOSurface-backed I/O.
//   4. Which (BLOBFILE offset, weights-dict offset) combination yields
//      numerically correct output vs. a CPU reference conv.
//
// Build: clang -fobjc-arc -O2 -framework Foundation -framework IOSurface -o ane_probe ane_probe.m

#import <Foundation/Foundation.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <IOSurface/IOSurface.h>
#import <mach/mach_time.h>

static Class g_Desc, g_IMM, g_Req, g_IO, g_Client;
static mach_timebase_info_data_t g_tb;

// ---------------- MIL ----------------

static NSString *milText(int blobOffset) {
    return [NSString stringWithFormat:
        @"program(1.3)\n"
         "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, "
         "{\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, "
         "{\"coremltools-version\", \"9.0\"}})]\n"
         "{\n"
         "    func main<ios18>(tensor<fp32, [1, 4, 1, 2]> x) {\n"
         "        string c_pad_type = const()[name = string(\"c_pad_type\"), val = string(\"valid\")];\n"
         "        tensor<int32, [2]> c_strides = const()[name = string(\"c_strides\"), val = tensor<int32, [2]>([1, 1])];\n"
         "        tensor<int32, [4]> c_pad = const()[name = string(\"c_pad\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n"
         "        tensor<int32, [2]> c_dilations = const()[name = string(\"c_dilations\"), val = tensor<int32, [2]>([1, 1])];\n"
         "        int32 c_groups = const()[name = string(\"c_groups\"), val = int32(1)];\n"
         "        string to_fp16 = const()[name = string(\"to_fp16\"), val = string(\"fp16\")];\n"
         "        tensor<fp16, [1, 4, 1, 2]> x16 = cast(dtype = to_fp16, x = x)[name = string(\"cast_in\")];\n"
         "        tensor<fp16, [6, 4, 1, 1]> W = const()[name = string(\"W\"), val = tensor<fp16, [6, 4, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64(%d)))];\n"
         "        tensor<fp16, [1, 6, 1, 2]> y16 = conv(dilations = c_dilations, groups = c_groups, pad = c_pad, pad_type = c_pad_type, strides = c_strides, weight = W, x = x16)[name = string(\"conv\")];\n"
         "        string to_fp32 = const()[name = string(\"to_fp32\"), val = string(\"fp32\")];\n"
         "        tensor<fp32, [1, 6, 1, 2]> y = cast(dtype = to_fp32, x = y16)[name = string(\"cast_out\")];\n"
         "    } -> (y);\n"
         "}\n", blobOffset];
}

// ---------------- weights ----------------
//
// Layout used by maderix/ANE's builder:
//   [0..63]    file header  (buf[0]=0x01, buf[4]=0x02)
//   [64..127]  tensor header (0xEF 0xBE 0xAD 0xDE, elem marker, size @72, data offset @80)
//   [128..]    fp16 payload
#define WEIGHT_DATA_OFF 128

static NSData *buildWeightBlob(const float *w, int n) {
    size_t wsize = (size_t)n * 2;
    size_t total = WEIGHT_DATA_OFF + wsize;
    uint8_t *buf = (uint8_t *)calloc(total, 1);
    buf[0] = 0x01; buf[4] = 0x02;
    buf[64] = 0xEF; buf[65] = 0xBE; buf[66] = 0xAD; buf[67] = 0xDE;
    buf[68] = 0x01;
    *(uint32_t *)(buf + 72) = (uint32_t)wsize;
    *(uint32_t *)(buf + 80) = WEIGHT_DATA_OFF;
    _Float16 *h = (_Float16 *)(buf + WEIGHT_DATA_OFF);
    for (int i = 0; i < n; i++) h[i] = (_Float16)w[i];
    return [NSData dataWithBytes:buf length:total];
}

static IOSurfaceRef createSurface(size_t bytes) {
    return IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth: @(bytes),
        (id)kIOSurfaceHeight: @1,
        (id)kIOSurfaceBytesPerElement: @1,
        (id)kIOSurfaceBytesPerRow: @(bytes),
        (id)kIOSurfaceAllocSize: @(bytes),
        (id)kIOSurfacePixelFormat: @0
    });
}

// ---------------- probe ----------------

static int g_fail = 0;

static void attempt(int milOffset, int dictOffset, const float *w, const float *x, const float *ref) {
    @autoreleasepool {
        printf("\n=== attempt: BLOBFILE offset=%d, weights-dict offset=%d ===\n", milOffset, dictOffset);
        NSError *e = nil;
        NSData *mil = [milText(milOffset) dataUsingEncoding:NSUTF8StringEncoding];
        NSData *blob = buildWeightBlob(w, 24);

        NSDictionary *wdict = @{@"@model_path/weights/weight.bin": @{@"offset": @(dictOffset), @"data": blob}};

        id desc = ((id(*)(Class, SEL, id, id, id))objc_msgSend)(
            g_Desc, @selector(modelWithMILText:weights:optionsPlist:), mil, wdict, nil);
        if (!desc) { printf("  desc: FAILED\n"); g_fail++; return; }
        id model = ((id(*)(Class, SEL, id))objc_msgSend)(g_IMM, @selector(inMemoryModelWithDescriptor:), desc);
        if (!model) { printf("  model: FAILED\n"); g_fail++; return; }

        // Pre-populate $TMPDIR/<hexId>/ so the ANE daemon finds MIL + weights.
        NSString *hex = ((id(*)(id, SEL))objc_msgSend)(model, @selector(hexStringIdentifier));
        NSString *td = [NSTemporaryDirectory() stringByAppendingPathComponent:hex];
        NSFileManager *fm = [NSFileManager defaultManager];
        [fm createDirectoryAtPath:[td stringByAppendingPathComponent:@"weights"]
      withIntermediateDirectories:YES attributes:nil error:nil];
        [mil writeToFile:[td stringByAppendingPathComponent:@"model.mil"] atomically:YES];
        [blob writeToFile:[td stringByAppendingPathComponent:@"weights/weight.bin"] atomically:YES];
        printf("  hexId=%s\n", [hex UTF8String]);

        uint64_t t0 = mach_absolute_time();
        BOOL ok = ((BOOL(*)(id, SEL, unsigned int, id, NSError **))objc_msgSend)(
            model, @selector(compileWithQoS:options:error:), 21, @{}, &e);
        double cms = (double)(mach_absolute_time() - t0) * g_tb.numer / g_tb.denom / 1e6;
        printf("  compile: %s (%.1f ms)\n", ok ? "YES" : "NO", cms);
        if (e) { printf("    err: %s\n", [[e description] UTF8String]); e = nil; }
        if (!ok) { g_fail++; [fm removeItemAtPath:td error:nil]; return; }

        ok = ((BOOL(*)(id, SEL, unsigned int, id, NSError **))objc_msgSend)(
            model, @selector(loadWithQoS:options:error:), 21, @{}, &e);
        printf("  load: %s\n", ok ? "YES" : "NO");
        if (e) { printf("    err: %s\n", [[e description] UTF8String]); e = nil; }
        if (!ok) { g_fail++; [fm removeItemAtPath:td error:nil]; return; }

        printf("  state: %lu\n", (unsigned long)((NSUInteger(*)(id, SEL))objc_msgSend)(model, @selector(state)));

        size_t inBytes = 8 * 4, outBytes = 12 * 4;
        IOSurfaceRef ioIn = createSurface(inBytes), ioOut = createSurface(outBytes);
        if (!ioIn || !ioOut) { printf("  IOSurface: FAILED\n"); g_fail++; return; }
        IOSurfaceLock(ioIn, 0, NULL);
        memcpy(IOSurfaceGetBaseAddress(ioIn), x, inBytes);
        IOSurfaceUnlock(ioIn, 0, NULL);

        id wIn = ((id(*)(Class, SEL, IOSurfaceRef))objc_msgSend)(g_IO, @selector(objectWithIOSurface:), ioIn);
        id wOut = ((id(*)(Class, SEL, IOSurfaceRef))objc_msgSend)(g_IO, @selector(objectWithIOSurface:), ioOut);
        id req = ((id(*)(Class, SEL, id, id, id, id, id, id, id))objc_msgSend)(g_Req,
            @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
            @[wIn], @[@0], @[wOut], @[@0], nil, nil, @0);

        ok = ((BOOL(*)(id, SEL, unsigned int, id, id, NSError **))objc_msgSend)(
            model, @selector(evaluateWithQoS:options:request:error:), 21, @{}, req, &e);
        printf("  evaluate: %s\n", ok ? "YES" : "NO");
        if (e) { printf("    err: %s\n", [[e description] UTF8String]); e = nil; }
        if (!ok) { g_fail++; return; }

        float out[12];
        IOSurfaceLock(ioOut, kIOSurfaceLockReadOnly, NULL);
        memcpy(out, IOSurfaceGetBaseAddress(ioOut), outBytes);
        IOSurfaceUnlock(ioOut, kIOSurfaceLockReadOnly, NULL);

        float maxErr = 0;
        printf("  out[0..5] = ");
        for (int i = 0; i < 6; i++) printf("%8.4f ", out[i]);
        printf("\n  ref[0..5] = ");
        for (int i = 0; i < 6; i++) printf("%8.4f ", ref[i]);
        printf("\n");
        for (int i = 0; i < 12; i++) {
            float d = fabsf(out[i] - ref[i]);
            if (d > maxErr) maxErr = d;
        }
        printf("  max|err| = %.6f  -> %s\n", maxErr, maxErr < 1e-2 ? "CORRECT" : "MISMATCH");
        if (maxErr >= 1e-2) g_fail++;

        ((BOOL(*)(id, SEL, unsigned int, NSError **))objc_msgSend)(
            model, @selector(unloadWithQoS:error:), 21, &e);
        CFRelease(ioIn); CFRelease(ioOut);
        [fm removeItemAtPath:td error:nil];
    }
}

int main(int argc, char **argv) {
    @autoreleasepool {
        mach_timebase_info(&g_tb);
        printf("ANE probe — CoreML-free private-framework path\n");
        printf("macOS/tmpdir: %s\n", [NSTemporaryDirectory() UTF8String]);

        void *h = dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);
        printf("dlopen AppleNeuralEngine: %s\n", h ? "OK" : dlerror());
        if (!h) return 1;

        g_Desc   = NSClassFromString(@"_ANEInMemoryModelDescriptor");
        g_IMM    = NSClassFromString(@"_ANEInMemoryModel");
        g_Req    = NSClassFromString(@"_ANERequest");
        g_IO     = NSClassFromString(@"_ANEIOSurfaceObject");
        g_Client = NSClassFromString(@"_ANEClient");
        printf("classes: desc=%d imm=%d req=%d io=%d client=%d\n",
               g_Desc != nil, g_IMM != nil, g_Req != nil, g_IO != nil, g_Client != nil);
        if (!g_Desc || !g_IMM || !g_Req || !g_IO) return 1;

        // Probe the ANE client connection (cheap availability check).
        @try {
            id conn = ((id(*)(Class, SEL))objc_msgSend)(g_Client, @selector(sharedConnection));
            printf("_ANEClient.sharedConnection: %s\n", conn ? "OK" : "nil");
        } @catch (NSException *ex) {
            printf("_ANEClient.sharedConnection raised: %s\n", [[ex description] UTF8String]);
        }

        // Reference conv: y[o][j] = sum_c W[o][c] * x[c][j];  x is [1,4,1,2], W is [6,4,1,1].
        float x[8], w[24], ref[12];
        for (int c = 0; c < 4; c++)
            for (int j = 0; j < 2; j++)
                x[c * 2 + j] = 0.25f * (float)(c * 2 + j + 1);
        for (int o = 0; o < 6; o++)
            for (int c = 0; c < 4; c++)
                w[o * 4 + c] = 0.5f * (float)(o + 1) - 0.125f * (float)c;
        for (int o = 0; o < 6; o++)
            for (int j = 0; j < 2; j++) {
                float s = 0;
                for (int c = 0; c < 4; c++) s += w[o * 4 + c] * x[c * 2 + j];
                ref[o * 2 + j] = s;
            }

        // Try the plausible (BLOBFILE offset, dict offset) conventions.
        attempt(WEIGHT_DATA_OFF, 0, w, x, ref);  // data at 128, dict offset 0
        attempt(64, 0, w, x, ref);               // data "at 64" (relative to file header)
        attempt(WEIGHT_DATA_OFF, 64, w, x, ref);
        attempt(64, 64, w, x, ref);

        printf("\nRESULT: %s (%d failed attempts)\n", g_fail == 0 ? "PASS" : "PARTIAL", g_fail);
        return g_fail == 0 ? 0 : 2;
    }
}
