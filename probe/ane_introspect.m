// ane_introspect.m — dump the private ANE API surface and find the padding rule.
//
// 1. Prints every selector/ivar of the private classes (authoritative API list).
// 2. Compiles+loads the same conv MIL as ane_probe.m, then asks the loaded model
//    for its expected buffer sizes via candidate selectors.
// 3. Evaluates with deliberately oversized IOSurfaces to confirm execution and
//    numerical correctness, then bisects the true required size.
//
// Build: clang -fobjc-arc -O2 -framework Foundation -framework IOSurface -o ane_introspect ane_introspect.m

#import <Foundation/Foundation.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <IOSurface/IOSurface.h>
#import <mach/mach_time.h>

static Class g_Desc, g_IMM, g_Req, g_IO, g_Client;

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

#define WEIGHT_DATA_OFF 128

static NSData *buildWeightBlob(const float *w, int n) {
    size_t wsize = (size_t)n * 2, total = WEIGHT_DATA_OFF + wsize;
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
        (id)kIOSurfaceWidth: @(bytes), (id)kIOSurfaceHeight: @1,
        (id)kIOSurfaceBytesPerElement: @1, (id)kIOSurfaceBytesPerRow: @(bytes),
        (id)kIOSurfaceAllocSize: @(bytes), (id)kIOSurfacePixelFormat: @0
    });
}

static void dumpClass(const char *name) {
    Class c = NSClassFromString([NSString stringWithUTF8String:name]);
    printf("\n--- %s (%s) ---\n", name, c ? "found" : "MISSING");
    if (!c) return;
    Class cur = c;
    int depth = 0;
    while (cur && cur != [NSObject class] && depth < 4) {
        unsigned n = 0;
        Method *ms = class_copyMethodList(cur, &n);
        printf("  [%s] %u methods:\n", class_getName(cur), n);
        for (unsigned i = 0; i < n; i++)
            printf("    - %s\n", sel_getName(method_getName(ms[i])));
        free(ms);
        unsigned ivn = 0;
        Ivar *ivs = class_copyIvarList(cur, &ivn);
        if (ivn) printf("  [%s] ivars:", class_getName(cur));
        for (unsigned i = 0; i < ivn; i++) printf(" %s", ivar_getName(ivs[i]));
        if (ivn) printf("\n");
        free(ivs);
        cur = class_getSuperclass(cur);
        depth++;
    }
}

int main(void) {
    @autoreleasepool {
        dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);
        g_Desc = NSClassFromString(@"_ANEInMemoryModelDescriptor");
        g_IMM = NSClassFromString(@"_ANEInMemoryModel");
        g_Req = NSClassFromString(@"_ANERequest");
        g_IO = NSClassFromString(@"_ANEIOSurfaceObject");
        g_Client = NSClassFromString(@"_ANEClient");

        dumpClass("_ANEInMemoryModel");
        dumpClass("_ANEInMemoryModelDescriptor");
        dumpClass("_ANERequest");
        dumpClass("_ANEIOSurfaceObject");
        dumpClass("_ANEClient");

        // ---- compile + load ----
        NSError *e = nil;
        NSData *mil = [milText(64) dataUsingEncoding:NSUTF8StringEncoding];
        float x[8], w[24], ref[12];
        for (int c = 0; c < 4; c++) for (int j = 0; j < 2; j++) x[c * 2 + j] = 0.25f * (float)(c * 2 + j + 1);
        for (int o = 0; o < 6; o++) for (int c = 0; c < 4; c++) w[o * 4 + c] = 0.5f * (float)(o + 1) - 0.125f * (float)c;
        for (int o = 0; o < 6; o++) for (int j = 0; j < 2; j++) {
            float s = 0; for (int c = 0; c < 4; c++) s += w[o * 4 + c] * x[c * 2 + j];
            ref[o * 2 + j] = s;
        }
        NSData *blob = buildWeightBlob(w, 24);
        NSDictionary *wdict = @{@"@model_path/weights/weight.bin": @{@"offset": @0, @"data": blob}};
        id desc = ((id(*)(Class, SEL, id, id, id))objc_msgSend)(g_Desc, @selector(modelWithMILText:weights:optionsPlist:), mil, wdict, nil);
        id model = ((id(*)(Class, SEL, id))objc_msgSend)(g_IMM, @selector(inMemoryModelWithDescriptor:), desc);
        NSString *hex = ((id(*)(id, SEL))objc_msgSend)(model, @selector(hexStringIdentifier));
        NSString *td = [NSTemporaryDirectory() stringByAppendingPathComponent:hex];
        NSFileManager *fm = [NSFileManager defaultManager];
        [fm createDirectoryAtPath:[td stringByAppendingPathComponent:@"weights"] withIntermediateDirectories:YES attributes:nil error:nil];
        [mil writeToFile:[td stringByAppendingPathComponent:@"model.mil"] atomically:YES];
        [blob writeToFile:[td stringByAppendingPathComponent:@"weights/weight.bin"] atomically:YES];

        printf("\n--- load reply probing ---\n");
        BOOL ok = ((BOOL(*)(id, SEL, unsigned int, id, NSError **))objc_msgSend)(model, @selector(compileWithQoS:options:error:), 21, @{}, &e);
        printf("compile: %s\n", ok ? "YES" : "NO");
        if (!ok && e) printf("  err: %s\n", [[e description] UTF8String]);

        // Try the "reply" variants of load, if they exist.
        SEL loadReply = NSSelectorFromString(@"loadWithQoS:options:error:reply:");
        if ([model respondsToSelector:loadReply]) {
            id reply = nil;
            ok = ((BOOL(*)(id, SEL, unsigned int, id, NSError **, id *))objc_msgSend)(model, loadReply, 21, @{}, &e, &reply);
            printf("loadWithQoS:options:error:reply: -> %s reply=%s\n", ok ? "YES" : "NO", reply ? [[reply description] UTF8String] : "(nil)");
        }
        ok = ((BOOL(*)(id, SEL, unsigned int, id, NSError **))objc_msgSend)(model, @selector(loadWithQoS:options:error:), 21, @{}, &e);
        printf("loadWithQoS:options:error: -> %s\n", ok ? "YES" : "NO");
        if (e) { printf("  err: %s\n", [[e description] UTF8String]); e = nil; }

        // Ask for expected sizes via candidate selectors.
        const char *sizeSels[] = {"inputBufferSize", "outputBufferSize", "inputBufferSizes", "outputBufferSizes",
                                  "bufferSizes", "modelAttributes", "inputSizes", "outputSizes", "program"};
        for (int i = 0; i < 9; i++) {
            SEL s = NSSelectorFromString([NSString stringWithUTF8String:sizeSels[i]]);
            if ([model respondsToSelector:s]) {
                id v = ((id(*)(id, SEL))objc_msgSend)(model, s);
                printf("  model.%s = %s\n", sizeSels[i], v ? [[v description] UTF8String] : "(nil)");
            }
        }
        // Also try on the descriptor and on _ANEClient.
        for (int i = 0; i < 4; i++) {
            SEL s = NSSelectorFromString([NSString stringWithUTF8String:sizeSels[i]]);
            if ([desc respondsToSelector:s]) {
                id v = ((id(*)(id, SEL))objc_msgSend)(desc, s);
                printf("  desc.%s = %s\n", sizeSels[i], v ? [[v description] UTF8String] : "(nil)");
            }
        }

        // ---- evaluate with oversized surfaces ----
        printf("\n--- oversized-surface evaluation ---\n");
        size_t big = 1024 * 1024;
        IOSurfaceRef ioIn = createSurface(big), ioOut = createSurface(big);
        IOSurfaceLock(ioIn, 0, NULL);
        memcpy(IOSurfaceGetBaseAddress(ioIn), x, 32);
        IOSurfaceUnlock(ioIn, 0, NULL);
        id wIn = ((id(*)(Class, SEL, IOSurfaceRef))objc_msgSend)(g_IO, @selector(objectWithIOSurface:), ioIn);
        id wOut = ((id(*)(Class, SEL, IOSurfaceRef))objc_msgSend)(g_IO, @selector(objectWithIOSurface:), ioOut);
        id req = ((id(*)(Class, SEL, id, id, id, id, id, id, id))objc_msgSend)(g_Req,
            @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
            @[wIn], @[@0], @[wOut], @[@0], nil, nil, @0);
        ok = ((BOOL(*)(id, SEL, unsigned int, id, id, NSError **))objc_msgSend)(model, @selector(evaluateWithQoS:options:request:error:), 21, @{}, req, &e);
        printf("evaluate(1MB surfaces): %s\n", ok ? "YES" : "NO");
        if (e) { printf("  err: %s\n", [[e description] UTF8String]); e = nil; }
        if (ok) {
            float out[12];
            IOSurfaceLock(ioOut, kIOSurfaceLockReadOnly, NULL);
            memcpy(out, IOSurfaceGetBaseAddress(ioOut), 48);
            IOSurfaceUnlock(ioOut, kIOSurfaceLockReadOnly, NULL);
            printf("  out: "); for (int i = 0; i < 12; i++) printf("%.4f ", out[i]); printf("\n");
            printf("  ref: "); for (int i = 0; i < 12; i++) printf("%.4f ", ref[i]); printf("\n");
            float me = 0; for (int i = 0; i < 12; i++) { float d = fabsf(out[i] - ref[i]); if (d > me) me = d; }
            printf("  max|err| = %.6f -> %s\n", me, me < 1e-2 ? "CORRECT" : "MISMATCH");
        }
        CFRelease(ioIn); CFRelease(ioOut);
        [fm removeItemAtPath:td error:nil];
    }
    return 0;
}
