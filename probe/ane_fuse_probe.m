// ane_fuse_probe.m — bisect the fused-FFN numeric bug.
//
// The engine's experimental fused FFN expresses SiLU inside MIL as
// sigmoid + mul + mul between two convs and produces wrong numbers at real
// model sizes. This probe builds progressively larger programs at the real
// shapes (hidden 576, inter 1536) and compares every stage against a CPU
// reference, so the failing op is identifiable.
//
// Build: clang -fobjc-arc -O2 -framework Foundation -framework IOSurface -o ane_fuse_probe ane_fuse_probe.m

#import <Foundation/Foundation.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <IOSurface/IOSurface.h>

static Class g_Desc, g_IMM, g_Req, g_IO;

#define HIDDEN 576
#define INTER  1536

static IOSurfaceRef createSurface(size_t bytes) {
    return IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth: @(bytes), (id)kIOSurfaceHeight: @1,
        (id)kIOSurfaceBytesPerElement: @1, (id)kIOSurfaceBytesPerRow: @(bytes),
        (id)kIOSurfaceAllocSize: @(bytes), (id)kIOSurfacePixelFormat: @0});
}

/// Builds the weight file: 64-byte file header, then per tensor a 64-byte chunk
/// header plus fp16 payload. Returns the BLOBFILE offsets.
static NSData *buildBlob(NSArray<NSData *> *tensors, NSMutableArray<NSNumber *> *offsets) {
    size_t total = 64;
    for (NSData *t in tensors) total += 64 + t.length;
    uint8_t *buf = calloc(total, 1);
    buf[0] = 0x01; buf[4] = 0x02;
    size_t cursor = 64;
    for (NSData *t in tensors) {
        buf[cursor + 0] = 0xEF; buf[cursor + 1] = 0xBE; buf[cursor + 2] = 0xAD; buf[cursor + 3] = 0xDE;
        buf[cursor + 4] = 0x01;
        uint32_t sz = (uint32_t)t.length;
        memcpy(buf + cursor + 8, &sz, 4);
        uint32_t off = (uint32_t)(cursor + 64);   // absolute payload offset
        memcpy(buf + cursor + 16, &off, 4);
        memcpy(buf + cursor + 64, t.bytes, t.length);
        [offsets addObject:@(cursor)];
        cursor += 64 + t.length;
    }
    return [NSData dataWithBytes:buf length:total];
}

static NSData *f16Data(const float *v, int n) {
    _Float16 *h = malloc(n * 2);
    for (int i = 0; i < n; i++) h[i] = (_Float16)v[i];
    NSData *d = [NSData dataWithBytes:h length:n * 2];
    free(h);
    return d;
}

/// Runs one program and returns the max relative error against `ref`.
static float runProgram(NSString *label, NSString *opLines, NSArray<NSData *> *tensors,
                        const float *x, int cin, const float *ref, int cout) {
    @autoreleasepool {
        NSMutableArray<NSNumber *> *offsets = [NSMutableArray array];
        NSData *blob = buildBlob(tensors, offsets);
        NSString *mil = [NSString stringWithFormat:
            @"program(1.3)\n"
             "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, {\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, {\"coremltools-version\", \"9.0\"}})]\n"
             "{\n"
             "    func main<ios18>(tensor<fp16, [1, %d, 1, 1]> i0) {\n"
             "        string c_pad_type = const()[name = string(\"c_pad_type\"), val = string(\"valid\")];\n"
             "        tensor<int32, [2]> c_strides = const()[name = string(\"c_strides\"), val = tensor<int32, [2]>([1, 1])];\n"
             "        tensor<int32, [4]> c_pad = const()[name = string(\"c_pad\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n"
             "        tensor<int32, [2]> c_dilations = const()[name = string(\"c_dilations\"), val = tensor<int32, [2]>([1, 1])];\n"
             "        int32 c_groups = const()[name = string(\"c_groups\"), val = int32(1)];\n"
             "%@"
             "    } -> (o0);\n"
             "}\n", cin, opLines];

        NSError *e = nil;
        NSData *milData = [mil dataUsingEncoding:NSUTF8StringEncoding];
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
            if (getenv("ANE_FUSE_DUMP")) printf("---- MIL ----\n%s\n------------\n", [mil UTF8String]);
            printf("  %-28s COMPILE FAILED: %s\n", [label UTF8String], e ? [[e description] UTF8String] : "?");
            [fm removeItemAtPath:td error:nil];
            return -1;
        }
        if (!((BOOL(*)(id, SEL, unsigned int, id, NSError **))objc_msgSend)(model, @selector(loadWithQoS:options:error:), 21, @{}, &e)) {
            printf("  %-28s LOAD FAILED: %s\n", [label UTF8String], e ? [[e description] UTF8String] : "?");
            [fm removeItemAtPath:td error:nil];
            return -1;
        }

        id attrs = ((id(*)(id, SEL))objc_msgSend)(model, @selector(modelAttributes));
        NSArray *statusList = attrs[@"NetworkStatusList"];
        NSDictionary *status = [statusList isKindOfClass:[NSArray class]] && statusList.count ? statusList[0] : nil;
        size_t inBytes = (size_t)[[status[@"LiveInputList"] firstObject][@"BatchStride"] longLongValue];
        size_t inPlane = (size_t)[[status[@"LiveInputList"] firstObject][@"PlaneStride"] longLongValue];
        size_t outBytes = (size_t)[[status[@"LiveOutputList"] firstObject][@"BatchStride"] longLongValue];
        size_t outPlane = (size_t)[[status[@"LiveOutputList"] firstObject][@"PlaneStride"] longLongValue];

        IOSurfaceRef ioIn = createSurface(inBytes), ioOut = createSurface(outBytes);
        IOSurfaceLock(ioIn, 0, NULL);
        uint8_t *base = (uint8_t *)IOSurfaceGetBaseAddress(ioIn);
        memset(base, 0, inBytes);
        for (int c = 0; c < cin; c++) ((_Float16 *)(base + (size_t)c * inPlane))[0] = (_Float16)x[c];
        IOSurfaceUnlock(ioIn, 0, NULL);

        id wIn = ((id(*)(Class, SEL, IOSurfaceRef))objc_msgSend)(g_IO, @selector(objectWithIOSurface:), ioIn);
        id wOut = ((id(*)(Class, SEL, IOSurfaceRef))objc_msgSend)(g_IO, @selector(objectWithIOSurface:), ioOut);
        id req = ((id(*)(Class, SEL, id, id, id, id, id, id, id))objc_msgSend)(g_Req,
            @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
            @[wIn], @[@0], @[wOut], @[@0], nil, nil, @0);
        if (!((BOOL(*)(id, SEL, unsigned int, id, id, NSError **))objc_msgSend)(model, @selector(evaluateWithQoS:options:request:error:), 21, @{}, req, &e)) {
            printf("  %-28s EVAL FAILED: %s\n", [label UTF8String], e ? [[e description] UTF8String] : "?");
            [fm removeItemAtPath:td error:nil];
            return -1;
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
        if (getenv("ANE_FUSE_VERBOSE")) {
            IOSurfaceLock(ioOut, kIOSurfaceLockReadOnly, NULL);
            printf("    got[0..3]: ");
            for (int c = 0; c < 3; c++) printf("%.6f ", (float)((_Float16 *)(ob + (size_t)c * outPlane))[0]);
            printf("\n    ref[0..3]: ");
            for (int c = 0; c < 3; c++) printf("%.6f ", ref[c]);
            printf("\n    maxMag=%.6f\n", maxMag);
            IOSurfaceUnlock(ioOut, kIOSurfaceLockReadOnly, NULL);
        }
        printf("  %-28s rel=%.6f %s\n", [label UTF8String], rel, rel < 0.01 ? "ok" : "BROKEN");
        ((BOOL(*)(id, SEL, unsigned int, NSError **))objc_msgSend)(model, @selector(unloadWithQoS:error:), 21, &e);
        CFRelease(ioIn); CFRelease(ioOut);
        [fm removeItemAtPath:td error:nil];
        return rel;
    }
}

int main(void) {
    @autoreleasepool {
        dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);
        g_Desc = NSClassFromString(@"_ANEInMemoryModelDescriptor");
        g_IMM = NSClassFromString(@"_ANEInMemoryModel");
        g_Req = NSClassFromString(@"_ANERequest");
        g_IO = NSClassFromString(@"_ANEIOSurfaceObject");

        const int cin = HIDDEN, inter = INTER, cout = HIDDEN;
        const size_t off0 = 64;                                    // first payload
        const size_t off1 = 128 + (size_t)inter * cin * 2;          // second payload
        const size_t off2 = 192 + (size_t)inter * cin * 4;          // third payload

        float *w1 = malloc(sizeof(float) * inter * cin);
        float *w2 = malloc(sizeof(float) * inter * cin);
        float *w3 = malloc(sizeof(float) * cout * inter);
        float *x = malloc(sizeof(float) * cin);
        float *g = malloc(sizeof(float) * inter);
        float *u = malloc(sizeof(float) * inter);
        float *a = malloc(sizeof(float) * inter);
        float *tmp = malloc(sizeof(float) * inter);
        float *ref_chain = malloc(sizeof(float) * cout);
        float *ref_sigmoid = malloc(sizeof(float) * cout);
        float *ref_mul = malloc(sizeof(float) * cout);
        float *ref_silu = malloc(sizeof(float) * cout);
        // Pseudo-random weights: a rank-1 / resonant pattern makes the CPU
        // reference tiny and the relative error meaningless.
        unsigned int seed = 12345u;
        for (int i = 0; i < inter * cin; i++) {
            seed = seed * 1664525u + 1013904223u;
            w1[i] = 0.03f * ((float)(seed >> 8) / 16777216.0f - 0.5f);
        }
        for (int i = 0; i < inter * cin; i++) {
            seed = seed * 1664525u + 1013904223u;
            w2[i] = 0.03f * ((float)(seed >> 8) / 16777216.0f - 0.5f);
        }
        for (int i = 0; i < cout * inter; i++) {
            seed = seed * 1664525u + 1013904223u;
            w3[i] = 0.02f * ((float)(seed >> 8) / 16777216.0f - 0.5f);
        }
        for (int i = 0; i < cin; i++) {
            seed = seed * 1664525u + 1013904223u;
            x[i] = 1.0f * ((float)(seed >> 8) / 16777216.0f - 0.5f);
        }

        for (int o = 0; o < inter; o++) {
            float sg = 0, su = 0;
            for (int i = 0; i < cin; i++) {
                sg += w1[o * cin + i] * x[i];
                su += w2[o * cin + i] * x[i];
            }
            g[o] = sg;
            u[o] = su;
        }
        NSArray<NSData *> *one = @[ f16Data(w1, inter * cin), f16Data(w3, cout * inter) ];
        NSArray<NSData *> *two = @[ f16Data(w1, inter * cin), f16Data(w2, inter * cin), f16Data(w3, cout * inter) ];

        for (int o = 0; o < cout; o++) { float s = 0; for (int i = 0; i < inter; i++) s += w3[o * inter + i] * g[i]; ref_chain[o] = s; }
        for (int i = 0; i < inter; i++) tmp[i] = 1.0f / (1.0f + expf(-g[i]));
        for (int o = 0; o < cout; o++) { float s = 0; for (int i = 0; i < inter; i++) s += w3[o * inter + i] * tmp[i]; ref_sigmoid[o] = s; }
        for (int i = 0; i < inter; i++) tmp[i] = g[i] * g[i];
        for (int o = 0; o < cout; o++) { float s = 0; for (int i = 0; i < inter; i++) s += w3[o * inter + i] * tmp[i]; ref_mul[o] = s; }
        for (int i = 0; i < inter; i++) a[i] = g[i] / (1.0f + expf(-g[i])) * u[i];
        for (int o = 0; o < cout; o++) { float s = 0; for (int i = 0; i < inter; i++) s += w3[o * inter + i] * a[i]; ref_silu[o] = s; }

        printf("fused-FFN bisect at hidden=%d inter=%d\n", cin, inter);

        // 0a. single conv reading chunk 0 (offset 64) — the engine's normal case
        {
            float *ref1 = malloc(sizeof(float) * inter);
            for (int o = 0; o < inter; o++) {
                float s = 0;
                for (int i = 0; i < cin; i++) s += w1[o * cin + i] * x[i];
                ref1[o] = s;
            }
            NSMutableString *m = [NSMutableString string];
            [m appendFormat:@"        tensor<fp16, [%d, %d, 1, 1]> wa = const()[name = string(\"wa\"), val = tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/w.bin\"), offset = uint64(%zu)))];\n", inter, cin, inter, cin, off0];
            [m appendFormat:@"        tensor<fp16, [1, %d, 1, 1]> o0 = conv(dilations = c_dilations, groups = c_groups, pad = c_pad, pad_type = c_pad_type, strides = c_strides, weight = wa, x = i0)[name = string(\"o0\")];\n", inter];
            runProgram(@"1 conv, chunk0", m, one, x, cin, ref1, inter);
        }
        // 0b. single conv reading chunk 1 (offset 128 + size0) — tests chunk offsets
        {
            float *ref2 = malloc(sizeof(float) * cout);
            for (int o = 0; o < cout; o++) {
                float s = 0;
                for (int i = 0; i < inter; i++) s += w3[o * inter + i] * g[i];
                ref2[o] = s;
            }
            // input is g (inter values), so the surface must hold inter channels
            NSMutableString *m = [NSMutableString string];
            [m appendFormat:@"        tensor<fp16, [%d, %d, 1, 1]> wc = const()[name = string(\"wc\"), val = tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/w.bin\"), offset = uint64(%zu)))];\n", cout, inter, cout, inter, off1];
            [m appendFormat:@"        tensor<fp16, [1, %d, 1, 1]> o0 = conv(dilations = c_dilations, groups = c_groups, pad = c_pad, pad_type = c_pad_type, strides = c_strides, weight = wc, x = i0)[name = string(\"o0\")];\n", cout];
            runProgram(@"1 conv, chunk1", m, one, g, inter, ref2, cout);
        }

        // 1. baseline: two convs back to back, no activation
        {
            NSMutableString *m = [NSMutableString string];
            [m appendFormat:@"        tensor<fp16, [%d, %d, 1, 1]> wa = const()[name = string(\"wa\"), val = tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/w.bin\"), offset = uint64(%zu)))];\n", inter, cin, inter, cin, off0];
            [m appendFormat:@"        tensor<fp16, [1, %d, 1, 1]> t0 = conv(dilations = c_dilations, groups = c_groups, pad = c_pad, pad_type = c_pad_type, strides = c_strides, weight = wa, x = i0)[name = string(\"t0\")];\n", inter];
            [m appendFormat:@"        tensor<fp16, [%d, %d, 1, 1]> wc = const()[name = string(\"wc\"), val = tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/w.bin\"), offset = uint64(%zu)))];\n", cout, inter, cout, inter, off1];
            [m appendFormat:@"        tensor<fp16, [1, %d, 1, 1]> o0 = conv(dilations = c_dilations, groups = c_groups, pad = c_pad, pad_type = c_pad_type, strides = c_strides, weight = wc, x = t0)[name = string(\"o0\")];\n", cout];
            runProgram(@"2 convs (baseline)", m, one, x, cin, ref_chain, cout);
        }
        // 2. sigmoid between the convs
        {
            NSMutableString *m = [NSMutableString string];
            [m appendFormat:@"        tensor<fp16, [%d, %d, 1, 1]> wa = const()[name = string(\"wa\"), val = tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/w.bin\"), offset = uint64(%zu)))];\n", inter, cin, inter, cin, off0];
            [m appendFormat:@"        tensor<fp16, [1, %d, 1, 1]> t0 = conv(dilations = c_dilations, groups = c_groups, pad = c_pad, pad_type = c_pad_type, strides = c_strides, weight = wa, x = i0)[name = string(\"t0\")];\n", inter];
            [m appendFormat:@"        tensor<fp16, [1, %d, 1, 1]> t1 = sigmoid(x = t0)[name = string(\"t1\")];\n", inter];
            [m appendFormat:@"        tensor<fp16, [%d, %d, 1, 1]> wc = const()[name = string(\"wc\"), val = tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/w.bin\"), offset = uint64(%zu)))];\n", cout, inter, cout, inter, off1];
            [m appendFormat:@"        tensor<fp16, [1, %d, 1, 1]> o0 = conv(dilations = c_dilations, groups = c_groups, pad = c_pad, pad_type = c_pad_type, strides = c_strides, weight = wc, x = t1)[name = string(\"o0\")];\n", cout];
            runProgram(@"2 convs + sigmoid", m, one, x, cin, ref_sigmoid, cout);
        }
        // 3. mul of a tensor with itself
        {
            NSMutableString *m = [NSMutableString string];
            [m appendFormat:@"        tensor<fp16, [%d, %d, 1, 1]> wa = const()[name = string(\"wa\"), val = tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/w.bin\"), offset = uint64(%zu)))];\n", inter, cin, inter, cin, off0];
            [m appendFormat:@"        tensor<fp16, [1, %d, 1, 1]> t0 = conv(dilations = c_dilations, groups = c_groups, pad = c_pad, pad_type = c_pad_type, strides = c_strides, weight = wa, x = i0)[name = string(\"t0\")];\n", inter];
            [m appendFormat:@"        tensor<fp16, [1, %d, 1, 1]> t1 = mul(x = t0, y = t0)[name = string(\"t1\")];\n", inter];
            [m appendFormat:@"        tensor<fp16, [%d, %d, 1, 1]> wc = const()[name = string(\"wc\"), val = tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/w.bin\"), offset = uint64(%zu)))];\n", cout, inter, cout, inter, off1];
            [m appendFormat:@"        tensor<fp16, [1, %d, 1, 1]> o0 = conv(dilations = c_dilations, groups = c_groups, pad = c_pad, pad_type = c_pad_type, strides = c_strides, weight = wc, x = t1)[name = string(\"o0\")];\n", cout];
            runProgram(@"2 convs + mul(t,t)", m, one, x, cin, ref_mul, cout);
        }
        // 4. the full fused FFN
        {
            NSMutableString *m = [NSMutableString string];
            [m appendFormat:@"        tensor<fp16, [%d, %d, 1, 1]> wg = const()[name = string(\"wg\"), val = tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/w.bin\"), offset = uint64(%zu)))];\n", inter, cin, inter, cin, off0];
            [m appendFormat:@"        tensor<fp16, [1, %d, 1, 1]> t0 = conv(dilations = c_dilations, groups = c_groups, pad = c_pad, pad_type = c_pad_type, strides = c_strides, weight = wg, x = i0)[name = string(\"t0\")];\n", inter];
            [m appendFormat:@"        tensor<fp16, [%d, %d, 1, 1]> wu = const()[name = string(\"wu\"), val = tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/w.bin\"), offset = uint64(%zu)))];\n", inter, cin, inter, cin, off1];
            [m appendFormat:@"        tensor<fp16, [1, %d, 1, 1]> t1 = conv(dilations = c_dilations, groups = c_groups, pad = c_pad, pad_type = c_pad_type, strides = c_strides, weight = wu, x = i0)[name = string(\"t1\")];\n", inter];
            [m appendFormat:@"        tensor<fp16, [1, %d, 1, 1]> t2 = sigmoid(x = t0)[name = string(\"t2\")];\n", inter];
            [m appendFormat:@"        tensor<fp16, [1, %d, 1, 1]> t3 = mul(x = t0, y = t2)[name = string(\"t3\")];\n", inter];
            [m appendFormat:@"        tensor<fp16, [1, %d, 1, 1]> t4 = mul(x = t3, y = t1)[name = string(\"t4\")];\n", inter];
            [m appendFormat:@"        tensor<fp16, [%d, %d, 1, 1]> wd = const()[name = string(\"wd\"), val = tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/w.bin\"), offset = uint64(%zu)))];\n", cout, inter, cout, inter, off2];
            [m appendFormat:@"        tensor<fp16, [1, %d, 1, 1]> o0 = conv(dilations = c_dilations, groups = c_groups, pad = c_pad, pad_type = c_pad_type, strides = c_strides, weight = wd, x = t4)[name = string(\"o0\")];\n", cout];
            runProgram(@"full fused FFN", m, two, x, cin, ref_silu, cout);
        }
        return 0;
    }
}
