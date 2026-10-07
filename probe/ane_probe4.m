// ane_probe4.m — read the ANE's declared tensor layout from modelAttributes, verify numerics.
//
// Build: clang -fobjc-arc -O2 -framework Foundation -framework IOSurface -o ane_probe4 ane_probe4.m

#import <Foundation/Foundation.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <IOSurface/IOSurface.h>
#import <mach/mach_time.h>

static Class g_Desc, g_IMM, g_Req, g_IO;
static mach_timebase_info_data_t g_tb;
#define WEIGHT_DATA_OFF 128

static NSData *buildWeightBlob(const float *w, int n) {
    size_t wsize = (size_t)n * 2, total = WEIGHT_DATA_OFF + wsize;
    uint8_t *buf = (uint8_t *)calloc(total, 1);
    buf[0] = 0x01; buf[4] = 0x02;
    buf[64] = 0xEF; buf[65] = 0xBE; buf[66] = 0xAD; buf[67] = 0xDE; buf[68] = 0x01;
    *(uint32_t *)(buf + 72) = (uint32_t)wsize;
    *(uint32_t *)(buf + 80) = WEIGHT_DATA_OFF;
    _Float16 *h = (_Float16 *)(buf + WEIGHT_DATA_OFF);
    for (int i = 0; i < n; i++) h[i] = (_Float16)w[i];
    return [NSData dataWithBytes:buf length:total];
}

static IOSurfaceRef createSurface(size_t bytes) {
    return IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth: @(bytes), (id)kIOSurfaceHeight: @1,
        (id)kIOSurfaceBytesPerElement: @1, (id)kIOSurfaceBytesPerRow: @(bytes),
        (id)kIOSurfaceAllocSize: @(bytes), (id)kIOSurfacePixelFormat: @0});
}

static NSString *matmulMIL(int cin, int cout, const char *dtype) {
    return [NSString stringWithFormat:
        @"program(1.3)\n"
         "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, "
         "{\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, "
         "{\"coremltools-version\", \"9.0\"}})]\n"
         "{\n"
         "    func main<ios18>(tensor<%s, [1, %d, 1, 1]> x) {\n"
         "        string c_pad_type = const()[name = string(\"c_pad_type\"), val = string(\"valid\")];\n"
         "        tensor<int32, [2]> c_strides = const()[name = string(\"c_strides\"), val = tensor<int32, [2]>([1, 1])];\n"
         "        tensor<int32, [4]> c_pad = const()[name = string(\"c_pad\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n"
         "        tensor<int32, [2]> c_dilations = const()[name = string(\"c_dilations\"), val = tensor<int32, [2]>([1, 1])];\n"
         "        int32 c_groups = const()[name = string(\"c_groups\"), val = int32(1)];\n"
         "        tensor<%s, [%d, %d, 1, 1]> W = const()[name = string(\"W\"), val = tensor<%s, [%d, %d, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64(64)))];\n"
         "        tensor<%s, [1, %d, 1, 1]> y = conv(dilations = c_dilations, groups = c_groups, pad = c_pad, pad_type = c_pad_type, strides = c_strides, weight = W, x = x)[name = string(\"conv\")];\n"
         "    } -> (y);\n"
         "}\n", dtype, cin, dtype, cout, cin, dtype, cout, cin, dtype, cout];
}

typedef struct { int batches, channels, height, width, planeCount, interleave; size_t planeStride, rowStride, batchStride, depthStride; char name[64], type[32]; } Live;

static int extractLive(NSArray *arr, Live *out, int max) {
    int n = 0;
    for (id e in arr) {
        if (![e isKindOfClass:[NSDictionary class]] || n >= max) continue;
        Live t = {0};
        id v;
        if ((v = e[@"Name"])) snprintf(t.name, sizeof(t.name), "%s", [[v description] UTF8String]);
        if ((v = e[@"Type"])) snprintf(t.type, sizeof(t.type), "%s", [[v description] UTF8String]);
        t.batches = [e[@"Batches"] intValue];
        t.channels = [e[@"Channels"] intValue];
        t.height = [e[@"Height"] intValue];
        t.width = [e[@"Width"] intValue];
        t.planeCount = [e[@"PlaneCount"] intValue];
        t.interleave = [e[@"Interleave"] intValue];
        t.planeStride = (size_t)[e[@"PlaneStride"] longLongValue];
        t.rowStride = (size_t)[e[@"RowStride"] longLongValue];
        t.batchStride = (size_t)[e[@"BatchStride"] longLongValue];
        t.depthStride = (size_t)[e[@"DepthStride"] longLongValue];
        out[n++] = t;
    }
    return n;
}

static int runCase(int cin, int cout, const char *dtype, size_t planeElems, int verbose) {
    @autoreleasepool {
        printf("\n===== conv 1x1 %s %d -> %d =====\n", dtype, cin, cout);
        NSError *e = nil;
        NSData *mil = [matmulMIL(cin, cout, dtype) dataUsingEncoding:NSUTF8StringEncoding];
        float *w = (float *)malloc(sizeof(float) * cin * cout);
        float *x = (float *)malloc(sizeof(float) * cin);
        for (int o = 0; o < cout; o++)
            for (int c = 0; c < cin; c++)
                w[o * cin + c] = 0.05f * (float)((o % 7) - 3) + 0.01f * (float)(c % 5) - 0.02f;
        for (int c = 0; c < cin; c++) x[c] = 0.1f * (float)((c % 11) - 5);
        float *ref = (float *)malloc(sizeof(float) * cout);
        for (int o = 0; o < cout; o++) { float s = 0; for (int c = 0; c < cin; c++) s += w[o * cin + c] * x[c]; ref[o] = s; }

        NSData *blob = buildWeightBlob(w, cin * cout);
        NSDictionary *wdict = @{@"@model_path/weights/weight.bin": @{@"offset": @0, @"data": blob}};
        id desc = ((id(*)(Class, SEL, id, id, id))objc_msgSend)(g_Desc, @selector(modelWithMILText:weights:optionsPlist:), mil, wdict, nil);
        id model = ((id(*)(Class, SEL, id))objc_msgSend)(g_IMM, @selector(inMemoryModelWithDescriptor:), desc);
        NSString *hex = ((id(*)(id, SEL))objc_msgSend)(model, @selector(hexStringIdentifier));
        NSString *td = [NSTemporaryDirectory() stringByAppendingPathComponent:hex];
        NSFileManager *fm = [NSFileManager defaultManager];
        [fm createDirectoryAtPath:[td stringByAppendingPathComponent:@"weights"] withIntermediateDirectories:YES attributes:nil error:nil];
        [mil writeToFile:[td stringByAppendingPathComponent:@"model.mil"] atomically:YES];
        [blob writeToFile:[td stringByAppendingPathComponent:@"weights/weight.bin"] atomically:YES];

        if (!((BOOL(*)(id, SEL, unsigned int, id, NSError **))objc_msgSend)(model, @selector(compileWithQoS:options:error:), 21, @{}, &e)) {
            printf("  compile FAILED: %s\n", e ? [[e description] UTF8String] : "?"); return 1;
        }
        if (!((BOOL(*)(id, SEL, unsigned int, id, NSError **))objc_msgSend)(model, @selector(loadWithQoS:options:error:), 21, @{}, &e)) {
            printf("  load FAILED: %s\n", e ? [[e description] UTF8String] : "?"); return 1;
        }

        id attrs = ((id(*)(id, SEL))objc_msgSend)(model, @selector(modelAttributes));
        if (verbose) {
            printf("  modelAttributes keys: %s\n", [[[attrs allKeys] description] UTF8String]);
            for (NSString *k in attrs) {
                id v = attrs[k];
                printf("    %s : %s = %s\n", [k UTF8String], class_getName([v class]),
                       [[v description] UTF8String]);
            }
        }
        Live ins[8], outs[8];
        NSArray *statusList = attrs[@"NetworkStatusList"];
        NSDictionary *status = [statusList isKindOfClass:[NSArray class]] && [statusList count] > 0 ? statusList[0] : nil;
        int ni = extractLive(status[@"LiveInputList"], ins, 8);
        int no = extractLive(status[@"LiveOutputList"], outs, 8);
        for (int i = 0; i < ni; i++)
            printf("  IN  %-8s %-8s B=%d C=%d H=%d W=%d planeCount=%d interleave=%d plane=%zu row=%zu depth=%zu batch=%zu\n",
                   ins[i].name, ins[i].type, ins[i].batches, ins[i].channels, ins[i].height, ins[i].width,
                   ins[i].planeCount, ins[i].interleave, ins[i].planeStride, ins[i].rowStride, ins[i].depthStride, ins[i].batchStride);
        for (int i = 0; i < no; i++)
            printf("  OUT %-8s %-8s B=%d C=%d H=%d W=%d planeCount=%d interleave=%d plane=%zu row=%zu depth=%zu batch=%zu\n",
                   outs[i].name, outs[i].type, outs[i].batches, outs[i].channels, outs[i].height, outs[i].width,
                   outs[i].planeCount, outs[i].interleave, outs[i].planeStride, outs[i].rowStride, outs[i].depthStride, outs[i].batchStride);
        if (ni < 1 || no < 1) { printf("  no Live lists\n"); return 1; }

        size_t inBytes = ins[0].batchStride ? ins[0].batchStride : ins[0].channels * ins[0].planeStride;
        size_t outBytes = outs[0].batchStride ? outs[0].batchStride : outs[0].channels * outs[0].planeStride;
        printf("  sizes: in=%zu out=%zu\n", inBytes, outBytes);

        IOSurfaceRef ioIn = createSurface(inBytes), ioOut = createSurface(outBytes);
        IOSurfaceLock(ioIn, 0, NULL);
        uint8_t *base = (uint8_t *)IOSurfaceGetBaseAddress(ioIn);
        memset(base, 0, inBytes);
        for (int c = 0; c < cin; c++) {
            size_t off = (size_t)c * ins[0].planeStride;
            if (strcmp(dtype, "fp16") == 0) ((_Float16 *)(base + off))[0] = (_Float16)x[c];
            else ((float *)(base + off))[0] = x[c];
        }
        IOSurfaceUnlock(ioIn, 0, NULL);

        id wIn = ((id(*)(Class, SEL, IOSurfaceRef))objc_msgSend)(g_IO, @selector(objectWithIOSurface:), ioIn);
        id wOut = ((id(*)(Class, SEL, IOSurfaceRef))objc_msgSend)(g_IO, @selector(objectWithIOSurface:), ioOut);
        id req = ((id(*)(Class, SEL, id, id, id, id, id, id, id))objc_msgSend)(g_Req,
            @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
            @[wIn], @[@0], @[wOut], @[@0], nil, nil, @0);

        if (!((BOOL(*)(id, SEL, unsigned int, id, id, NSError **))objc_msgSend)(model, @selector(evaluateWithQoS:options:request:error:), 21, @{}, req, &e)) {
            printf("  evaluate FAILED: %s\n", e ? [[e description] UTF8String] : "?"); return 1;
        }
        for (int i = 0; i < 10; i++)
            ((BOOL(*)(id, SEL, unsigned int, id, id, NSError **))objc_msgSend)(model, @selector(evaluateWithQoS:options:request:error:), 21, @{}, req, &e);
        int iters = 300;
        uint64_t t0 = mach_absolute_time();
        for (int i = 0; i < iters; i++)
            ((BOOL(*)(id, SEL, unsigned int, id, id, NSError **))objc_msgSend)(model, @selector(evaluateWithQoS:options:request:error:), 21, @{}, req, &e);
        double us = (double)(mach_absolute_time() - t0) * g_tb.numer / g_tb.denom / 1e3 / iters;
        printf("  %.2f us/eval  %.1f GFLOP/s (2*%d*%d)\n", us, 2.0 * cin * cout / 1e9 / (us / 1e6), cin, cout);

        float *got = (float *)malloc(sizeof(float) * cout);
        IOSurfaceLock(ioOut, kIOSurfaceLockReadOnly, NULL);
        uint8_t *ob = (uint8_t *)IOSurfaceGetBaseAddress(ioOut);
        for (int c = 0; c < cout; c++) {
            size_t off = (size_t)c * outs[0].planeStride;
            got[c] = strcmp(dtype, "fp16") == 0 ? (float)((_Float16 *)(ob + off))[0] : ((float *)(ob + off))[0];
        }
        IOSurfaceUnlock(ioOut, kIOSurfaceLockReadOnly, NULL);

        float maxErr = 0, maxMag = 0;
        for (int c = 0; c < cout; c++) {
            float d = fabsf(got[c] - ref[c]); if (d > maxErr) maxErr = d;
            if (fabsf(ref[c]) > maxMag) maxMag = fabsf(ref[c]);
        }
        float rel = maxMag > 0 ? maxErr / maxMag : maxErr;
        printf("  got[0..5]: "); for (int i = 0; i < 6 && i < cout; i++) printf("%9.5f ", got[i]); printf("\n");
        printf("  ref[0..5]: "); for (int i = 0; i < 6 && i < cout; i++) printf("%9.5f ", ref[i]); printf("\n");
        printf("  max|err|=%.6f rel=%.5f -> %s\n", maxErr, rel, rel < 0.02f ? "CORRECT" : "MISMATCH");

        ((BOOL(*)(id, SEL, unsigned int, NSError **))objc_msgSend)(model, @selector(unloadWithQoS:error:), 21, &e);
        CFRelease(ioIn); CFRelease(ioOut);
        [fm removeItemAtPath:td error:nil];
        free(w); free(x); free(ref); free(got);
        return rel < 0.02f ? 0 : 2;
    }
}

int main(void) {
    @autoreleasepool {
        mach_timebase_info(&g_tb);
        dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);
        g_Desc = NSClassFromString(@"_ANEInMemoryModelDescriptor");
        g_IMM = NSClassFromString(@"_ANEInMemoryModel");
        g_Req = NSClassFromString(@"_ANERequest");
        g_IO = NSClassFromString(@"_ANEIOSurfaceObject");

        int rc = 0;
        rc |= runCase(64, 64, "fp16", 0, 1);
        rc |= runCase(256, 256, "fp16", 0, 0);
        rc |= runCase(1024, 1024, "fp16", 0, 0);
        rc |= runCase(4096, 4096, "fp16", 0, 0);
        printf("\nRESULT: %s\n", rc == 0 ? "ALL CORRECT" : "SOME FAILED");
        return rc;
    }
}
