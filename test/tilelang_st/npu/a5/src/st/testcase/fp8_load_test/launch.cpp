// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// FP8 Load Test - Launch functions

#include "acl/acl.h"
#include <cstdint>

// External kernel functions
extern "C" void fp8_load_32x64(void *stream, uint32_t blockDim, void *gm_input, void *gm_output);
extern "C" void fp8_load_64x64(void *stream, uint32_t blockDim, void *gm_input, void *gm_output);
extern "C" void fp8_load_128x64(void *stream, uint32_t blockDim, void *gm_input, void *gm_output);

void Launchfp8_load_32x64(uint8_t *input, float *output, void *stream) {
    fp8_load_32x64(stream, 1, input, output);
}

void Launchfp8_load_64x64(uint8_t *input, float *output, void *stream) {
    fp8_load_64x64(stream, 1, input, output);
}

void Launchfp8_load_128x64(uint8_t *input, float *output, void *stream) {
    fp8_load_128x64(stream, 1, input, output);
}
