// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// FP8 Load Test - 验证FP8数据从GM加载到L1是否正确

#include "acl/acl.h"
#include "test_common.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

using namespace PtoTestCommon;

// Kernel declarations
void Launchfp8_load_32x64(uint8_t *input, float *output, void *stream);
void Launchfp8_load_64x64(uint8_t *input, float *output, void *stream);
void Launchfp8_load_128x64(uint8_t *input, float *output, void *stream);

using LaunchFn = void (*)(uint8_t *, float *, void *);

struct TestCase {
    const char *name;
    size_t m;
    size_t n;
    LaunchFn launch;
};

static const TestCase kCases[] = {
    {"fp8_load_32x64", 32, 64, Launchfp8_load_32x64},
    {"fp8_load_64x64", 64, 64, Launchfp8_load_64x64},
    {"fp8_load_128x64", 128, 64, Launchfp8_load_128x64},
};
static constexpr size_t kNumCases = sizeof(kCases) / sizeof(kCases[0]);

static int RunCase(const TestCase &tc, int deviceId, aclrtStream stream) {
    (void)deviceId;
    int rc = 0;
    const size_t input_elems = tc.m * tc.n;
    const size_t input_bytes = input_elems * sizeof(uint8_t);
    const size_t output_elems = tc.m * tc.n;
    const size_t output_bytes = output_elems * sizeof(float);

    std::printf("[INFO] === case: %s (m=%zu, n=%zu) ===\n", tc.name, tc.m, tc.n);

    std::string caseDir = std::string("./") + tc.name;

    void *inputHost = nullptr, *outputHost = nullptr;
    void *inputDevice = nullptr, *outputDevice = nullptr;

    aclrtMallocHost(&inputHost, input_bytes);
    aclrtMallocHost(&outputHost, output_bytes);
    aclrtMalloc(&inputDevice, input_bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&outputDevice, output_bytes, ACL_MEM_MALLOC_HUGE_FIRST);

    size_t inputFileSize = input_bytes;
    if (!ReadFile((caseDir + "/input.bin").c_str(), inputFileSize, inputHost, input_bytes)) {
        std::fprintf(stderr, "[ERROR] read input failed\n");
        rc = 1;
    }

    if (rc == 0) {
        aclrtMemcpy(inputDevice, input_bytes, inputHost, input_bytes, ACL_MEMCPY_HOST_TO_DEVICE);
        aclrtMemset(outputDevice, output_bytes, 0, output_bytes);

        tc.launch(static_cast<uint8_t *>(inputDevice), static_cast<float *>(outputDevice), stream);
        aclrtSynchronizeStream(stream);

        aclrtMemcpy(outputHost, output_bytes, outputDevice, output_bytes, ACL_MEMCPY_DEVICE_TO_HOST);
    }

    if (rc == 0 && !WriteFile((caseDir + "/output.bin").c_str(), outputHost, output_bytes)) {
        std::fprintf(stderr, "[ERROR] write output failed\n");
        rc = 1;
    }

    if (inputDevice) aclrtFree(inputDevice);
    if (outputDevice) aclrtFree(outputDevice);
    if (inputHost) aclrtFreeHost(inputHost);
    if (outputHost) aclrtFreeHost(outputHost);

    if (rc == 0) std::printf("[INFO] case %s done\n", tc.name);
    return rc;
}

int main(int argc, char *argv[]) {
    const char *caseFilter = (argc > 1) ? argv[1] : nullptr;
    int rc = 0;
    int deviceId = 0;
    aclrtStream stream = nullptr;

    aclInit(nullptr);
    if (const char *envDevice = std::getenv("ACL_DEVICE_ID")) deviceId = std::atoi(envDevice);
    aclrtSetDevice(deviceId);
    aclrtCreateStream(&stream);

    for (size_t i = 0; i < kNumCases; ++i) {
        if (caseFilter != nullptr && std::strcmp(kCases[i].name, caseFilter) != 0) continue;
        int ret = RunCase(kCases[i], deviceId, stream);
        if (ret != 0) {
            std::fprintf(stderr, "[ERROR] case %s failed\n", kCases[i].name);
            rc = 1;
            break;
        }
    }

    if (stream) aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return rc;
}
