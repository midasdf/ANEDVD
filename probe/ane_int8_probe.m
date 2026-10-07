// ane_int8_probe.m — does ANECCompile accept int8 weights via
// constexpr_affine_dequantize, and do they produce correct numbers?
//
// Both reference projects (Orion, Espresso) report int8 as unsupported/broken,
// but Apple's INT8 path is documented to dequantize to fp16 before compute,
// which would halve the weight-bandwidth-bound decode cost. This probe tries
// several MIL/blob layouts and reports which combination compiles.
//
// Build: clang -fobjc-arc -O2 -framework Foundation -framework IOSurface -o ane_int8_probe ane_int8_probe.m

#import <Foundation/Foundation.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <IOSurface/IOSurface.h>
#import <mach/mach_time.h>

static Class g_Desc, g_IMM, g_Req, g_IO;

static IOSurfaceRef createSurface(size_t bytes) {
    return IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth: @(bytes), (id)kIOSurfaceHeight: @1,
        (id)kIOSurfaceBytesPerElement: @1, (id)kIOSurfaceBytesPerRow: @(bytes),
        (id)kIOSurfaceAllocSize: @(bytes), (id)kIOSurfacePixelFormat: @0});
}

// ---------------------------------------------------------------- blob layouts

// Layout A: same container as the fp16 path (64-byte file header, then per
// tensor a 64-byte chunk header + payload), with the 8-bit marker at byte 10.
static NSData *blobA(const int8_t *wq, int nq, const _Float16 *scale, int ns, const int8_t *zp, int nz,
                     size_t *off_q, size_t *off_s, size_t *off_z) {
    size_t total = 64 + (64 + nq) + (64 + ns * 2) + (64 + nz);
    uint8_t *buf = (uint8_t *)calloc(total, 1);
    buf[0] = 0x01; buf[4] = 0x02;
    size_t cursor = 64;
    // int8 chunk
    buf[cursor + 0] = 0xEF; buf[cursor + 1] = 0xBE; buf[cursor + 2] = 0xAD; buf[cursor + 3] = 0xDE;
    buf[cursor + 4] = 0x01; buf[cursor + 10] = 0x08;
    *(uint32_t *)(buf + cursor + 8) = (uint32_t)nq;
    memcpy(buf + cursor + 64, wq, nq);
    *off_q = cursor; cursor += 64 + nq;
    // fp16 scale chunk
    buf[cursor + 0] = 0xEF; buf[cursor + 1] = 0xBE; buf[cursor + 2] = 0xAD; buf[cursor + 3] = 0xDE;
    buf[cursor + 4] = 0x01;
    *(uint32_t *)(buf + cursor + 8) = (uint32_t)(ns * 2);
    memcpy(buf + cursor + 64, scale, ns * 2);
    *off_s = cursor; cursor += 64 + ns * 2;
    // int8 zero point chunk
    buf[cursor + 0] = 0xEF; buf[cursor + 1] = 0xBE; buf[cursor + 2] = 0xAD; buf[cursor + 3] = 0xDE;
    buf[cursor + 4] = 0x01; buf[cursor + 10] = 0x08;
    *(uint32_t *)(buf + cursor + 8) = (uint32_t)nz;
    memcpy(buf + cursor + 64, zp, nz);
    *off_z = cursor;
    return [NSData dataWithBytes:buf length:total];
}

// Layout B: maderix-style — a single 64-byte header then the int8 payload, and
// the BLOBFILE offset points at byte 64.
static NSData *blobB(const int8_t *wq, int nq, const _Float16 *scale, int ns, const int8_t *zp, int nz,
                     size_t *off_q, size_t *off_s, size_t *off_z) {
    size_t total = 64 + nq + 64 + ns * 2 + 64 + nz;
    uint8_t *buf = (uint8_t *)calloc(total, 1);
    size_t cursor = 0;
    buf[cursor + 0] = 0xEF; buf[cursor + 1] = 0xBE; buf[cursor + 2] = 0xAD; buf[cursor + 3] = 0xDE;
    buf[cursor + 4] = 0x01; buf[cursor + 10] = 0x08;
    memcpy(buf + cursor + 64, wq, nq);
    *off_q = cursor + 64; cursor += 64 + nq;
    buf[cursor + 0] = 0xEF; buf[cursor + 1] = 0xBE; buf[cursor + 2] = 0xAD; buf[cursor + 3] = 0xDE;
    buf[cursor + 4] = 0x01;
    memcpy(buf + cursor + 64, scale, ns * 2);
    *off_s = cursor + 64; cursor += 64 + ns * 2;
    buf[cursor + 0] = 0xEF; buf[cursor + 1] = 0xBE; buf[cursor + 2] = 0xAD; buf[cursor + 3] = 0xDE;
    buf[cursor + 4] = 0x01; buf[cursor + 10] = 0x08;
    memcpy(buf + cursor + 64, zp, nz);
    *off_z = cursor + 64;
    return [NSData dataWithBytes:buf length:total];
}

// ---------------------------------------------------------------- MIL

static NSString *int8MIL(int cin, int cout, size_t oq, size_t os, size_t oz, int variant) {
    NSString *deq;
    if (variant == 0) {
        // per-channel affine dequantize, axis 0
        deq = [NSString stringWithFormat:
            @"        tensor<fp16, [%d, %d, 1, 1]> W = constexpr_affine_dequantize(input = Wq, scale = sc, zero_point = zp, axis = int32(0))[name = string(\"W\")];\n",
            cout, cin];
    } else if (variant == 1) {
        // scale only, zero_point omitted
        deq = [NSString stringWithFormat:
            @"        tensor<fp16, [%d, %d, 1, 1]> W = constexpr_affine_dequantize(input = Wq, scale = sc, zero_point = zp, axis = int32(1))[name = string(\"W\")];\n",
            cout, cin];
    } else if (variant == 3) {
        // same op but with fp16 input: isolates "op unsupported" from "int8 unsupported"
        deq = [NSString stringWithFormat:
            @"        tensor<fp16, [%d, %d, 1, 1]> W = constexpr_affine_dequantize(input = Wf, scale = sc, zero_point = zp, axis = int32(0))[name = string(\"W\")];\n",
            cout, cin];
    } else {
        // per-tensor (axis = 3, the last dim)
        deq = [NSString stringWithFormat:
            @"        tensor<fp16, [%d, %d, 1, 1]> W = constexpr_affine_dequantize(input = Wq, scale = sc, zero_point = zp, axis = int32(3))[name = string(\"W\")];\n",
            cout, cin];
    }
    return [NSString stringWithFormat:
        @"program(1.3)\n"
         "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, {\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, {\"coremltools-version\", \"9.0\"}})]\n"
         "{\n"
         "    func main<ios18>(tensor<fp16, [1, %d, 1, 1]> i0) {\n"
         "        string c_pad_type = const()[name = string(\"c_pad_type\"), val = string(\"valid\")];\n"
         "        tensor<int32, [2]> c_strides = const()[name = string(\"c_strides\"), val = tensor<int32, [2]>([1, 1])];\n"
         "        tensor<int32, [4]> c_pad = const()[name = string(\"c_pad\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n"
         "        tensor<int32, [2]> c_dilations = const()[name = string(\"c_dilations\"), val = tensor<int32, [2]>([1, 1])];\n"
         "        int32 c_groups = const()[name = string(\"c_groups\"), val = int32(1)];\n"
         "        tensor<int8, [%d, %d, 1, 1]> Wq = const()[name = string(\"Wq\"), val = tensor<int8, [%d, %d, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/w.bin\"), offset = uint64(%zu)))];\n"
         "        tensor<fp16, [%d, %d, 1, 1]> Wf = const()[name = string(\"Wf\"), val = tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/w.bin\"), offset = uint64(%zu)))];\n"
         "        tensor<fp16, [%d]> sc = const()[name = string(\"sc\"), val = tensor<fp16, [%d]>(BLOBFILE(path = string(\"@model_path/weights/w.bin\"), offset = uint64(%zu)))];\n"
         "        tensor<int8, [%d]> zp = const()[name = string(\"zp\"), val = tensor<int8, [%d]>(BLOBFILE(path = string(\"@model_path/weights/w.bin\"), offset = uint64(%zu)))];\n"
         "%@"
         "        tensor<fp16, [1, %d, 1, 1]> y = conv(dilations = c_dilations, groups = c_groups, pad = c_pad, pad_type = c_pad_type, strides = c_strides, weight = W, x = i0)[name = string(\"conv\")];\n"
         "    } -> (y);\n"
         "}\n",
        cin, cout, cin, cout, cin, oq, cout, cin, cout, cin, oq, cout, cout, os, cout, cout, oz, deq, cout];
}

// ---------------------------------------------------------------- runner

static int tryVariant(const char *label, int cin, int cout, const int8_t *wq, const _Float16 *sc, const int8_t *zp,
                      const float *ref, int layout, int milVariant) {
    @autoreleasepool {
        size_t oq, os, oz;
        NSData *blob = (layout == 0)
            ? blobA(wq, cin * cout, sc, cout, zp, cout, &oq, &os, &oz)
            : blobB(wq, cin * cout, sc, cout, zp, cout, &oq, &os, &oz);
        NSString *mil = int8MIL(cin, cout, oq, os, oz, milVariant);
        NSData *milData = [mil dataUsingEncoding:NSUTF8StringEncoding];

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

        printf("\n=== %s (layout %d, axis variant %d) ===\n", label, layout, milVariant);
        if (!((BOOL(*)(id, SEL, unsigned int, id, NSError **))objc_msgSend)(model, @selector(compileWithQoS:options:error:), 21, @{}, &e)) {
            printf("  COMPILE FAILED: %s\n", e ? [[e description] UTF8String] : "?");
            [fm removeItemAtPath:td error:nil];
            return 0;
        }
        if (!((BOOL(*)(id, SEL, unsigned int, id, NSError **))objc_msgSend)(model, @selector(loadWithQoS:options:error:), 21, @{}, &e)) {
            printf("  LOAD FAILED: %s\n", e ? [[e description] UTF8String] : "?");
            [fm removeItemAtPath:td error:nil];
            return 0;
        }
        printf("  compile+load OK\n");

        id attrs = ((id(*)(id, SEL))objc_msgSend)(model, @selector(modelAttributes));
        NSArray *statusList = attrs[@"NetworkStatusList"];
        NSDictionary *status = [statusList isKindOfClass:[NSArray class]] && statusList.count ? statusList[0] : nil;
        NSDictionary *inInfo = [status[@"LiveInputList"] firstObject];
        NSDictionary *outInfo = [status[@"LiveOutputList"] firstObject];
        size_t inBytes = (size_t)[inInfo[@"BatchStride"] longLongValue];
        size_t outBytes = (size_t)[outInfo[@"BatchStride"] longLongValue];
        size_t inPlane = (size_t)[inInfo[@"PlaneStride"] longLongValue];
        size_t outPlane = (size_t)[outInfo[@"PlaneStride"] longLongValue];

        IOSurfaceRef ioIn = createSurface(inBytes), ioOut = createSurface(outBytes);
        IOSurfaceLock(ioIn, 0, NULL);
        uint8_t *base = (uint8_t *)IOSurfaceGetBaseAddress(ioIn);
        memset(base, 0, inBytes);
        float x[256];
        for (int c = 0; c < cin; c++) {
            x[c] = 0.05f * (float)((c % 7) - 3);
            ((_Float16 *)(base + (size_t)c * inPlane))[0] = (_Float16)x[c];
        }
        IOSurfaceUnlock(ioIn, 0, NULL);

        id wIn = ((id(*)(Class, SEL, IOSurfaceRef))objc_msgSend)(g_IO, @selector(objectWithIOSurface:), ioIn);
        id wOut = ((id(*)(Class, SEL, IOSurfaceRef))objc_msgSend)(g_IO, @selector(objectWithIOSurface:), ioOut);
        id req = ((id(*)(Class, SEL, id, id, id, id, id, id, id))objc_msgSend)(g_Req,
            @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
            @[wIn], @[@0], @[wOut], @[@0], nil, nil, @0);
        if (!((BOOL(*)(id, SEL, unsigned int, id, id, NSError **))objc_msgSend)(model, @selector(evaluateWithQoS:options:request:error:), 21, @{}, req, &e)) {
            printf("  EVAL FAILED: %s\n", e ? [[e description] UTF8String] : "?");
            [fm removeItemAtPath:td error:nil];
            return 0;
        }
        float maxErr = 0, maxMag = 0;
        IOSurfaceLock(ioOut, kIOSurfaceLockReadOnly, NULL);
        uint8_t *ob = (uint8_t *)IOSurfaceGetBaseAddress(ioOut);
        for (int c = 0; c < cout; c++) {
            float got = (float)((_Float16 *)(ob + (size_t)c * outPlane))[0];
            float d = fabsf(got - ref[c]);
            if (d > maxErr) maxErr = d;
            if (fabsf(ref[c]) > maxMag) maxMag = fabsf(ref[c]);
        }
        IOSurfaceUnlock(ioOut, kIOSurfaceLockReadOnly, NULL);
        float rel = maxMag > 0 ? maxErr / maxMag : maxErr;
        printf("  EVAL OK  max|err|=%.6f rel=%.5f -> %s\n", maxErr, rel, rel < 0.05f ? "CORRECT" : "MISMATCH");
        ((BOOL(*)(id, SEL, unsigned int, NSError **))objc_msgSend)(model, @selector(unloadWithQoS:error:), 21, &e);
        CFRelease(ioIn); CFRelease(ioOut);
        [fm removeItemAtPath:td error:nil];
        return rel < 0.05f ? 2 : 1;
    }
}

int main(void) {
    @autoreleasepool {
        dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);
        g_Desc = NSClassFromString(@"_ANEInMemoryModelDescriptor");
        g_IMM = NSClassFromString(@"_ANEInMemoryModel");
        g_Req = NSClassFromString(@"_ANERequest");
        g_IO = NSClassFromString(@"_ANEIOSurfaceObject");

        const int cin = 64, cout = 64;
        int8_t *wq = malloc(cin * cout);
        _Float16 *sc = malloc(cout * 2);
        int8_t *zp = calloc(cout, 1);
        float *w = malloc(sizeof(float) * cin * cout);
        float *x = malloc(sizeof(float) * cin);
        float *ref = malloc(sizeof(float) * cout);
        for (int o = 0; o < cout; o++) {
            float mx = 0;
            for (int i = 0; i < cin; i++) {
                float v = 0.05f * (float)((o % 7) - 3) + 0.01f * (float)(i % 5) - 0.02f;
                w[o * cin + i] = v;
                if (fabsf(v) > mx) mx = fabsf(v);
            }
            float scale = mx / 127.0f;
            if (scale == 0) scale = 1;
            sc[o] = (_Float16)scale;
            for (int i = 0; i < cin; i++) {
                float q = w[o * cin + i] / scale;
                if (q > 127) q = 127;
                if (q < -128) q = -128;
                wq[o * cin + i] = (int8_t)lrintf(q);
            }
        }
        for (int i = 0; i < cin; i++) x[i] = 0.05f * (float)((i % 7) - 3);
        for (int o = 0; o < cout; o++) {
            float s = 0;
            for (int i = 0; i < cin; i++) s += w[o * cin + i] * x[i];
            ref[o] = s;
        }

        printf("int8 weight probe: does ANEC accept constexpr_affine_dequantize?\n");
        int r = 0;
        r |= tryVariant("container layout A", cin, cout, wq, sc, zp, ref, 0, 0);
        r |= tryVariant("container layout A, axis=1", cin, cout, wq, sc, zp, ref, 0, 1);
        r |= tryVariant("maderix layout B", cin, cout, wq, sc, zp, ref, 1, 0);
        r |= tryVariant("maderix layout B, axis=3", cin, cout, wq, sc, zp, ref, 1, 2);
        r |= tryVariant("op existence test (fp16 input)", cin, cout, wq, sc, zp, ref, 0, 3);
        printf("\nRESULT: %s\n", r == 8 ? "INT8 WORKS" : "int8 not usable with these layouts");
        return 0;
    }
}
