// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "test_common.h"
#include "acl/acl.h"

#include <cstdio>
#include <cstdlib>
#include <cstdint>

using namespace PtoTestCommon;

#define ACL_CHECK(expr)                                                        \
  do {                                                                         \
    const aclError result = (expr);                                            \
    if (result != ACL_SUCCESS) {                                               \
      std::fprintf(stderr, "[ERROR] %s failed: %d (%s:%d)\\n", #expr,         \
                   static_cast<int>(result), __FILE__, __LINE__);              \
      rc = 1;                                                                  \
      goto cleanup;                                                            \
    }                                                                          \
  } while (0)

#define FILE_CHECK(expr, path)                                                 \
  do {                                                                         \
    if (!(expr)) {                                                             \
      std::fprintf(stderr, "[ERROR] file operation failed: %s (%s:%d)\\n",    \
                   path, __FILE__, __LINE__);                                  \
      rc = 1;                                                                  \
      goto cleanup;                                                            \
    }                                                                          \
  } while (0)

void LaunchMad_mx_e8m0_stage_abi_kernel(uint8_t *a, uint8_t *b,
                                         uint16_t *aScale, uint16_t *bScale,
                                         float *c, void *stream);

int main() {
  constexpr size_t kM = 256;
  constexpr size_t kN = 256;
  constexpr size_t kK = 256;
  constexpr size_t kScaleSrcStride = 8;
  constexpr size_t kScaleBytes = kM * kScaleSrcStride * sizeof(uint16_t);
  constexpr size_t kASize = kM * kK * sizeof(uint8_t);
  constexpr size_t kBSize = kK * kN * sizeof(uint8_t);
  constexpr size_t kCSize = kM * kN * sizeof(float);

  uint8_t *aHost = nullptr;
  uint8_t *bHost = nullptr;
  uint16_t *aScaleHost = nullptr;
  uint16_t *bScaleHost = nullptr;
  float *cHost = nullptr;
  uint8_t *aDevice = nullptr;
  uint8_t *bDevice = nullptr;
  uint16_t *aScaleDevice = nullptr;
  uint16_t *bScaleDevice = nullptr;
  float *cDevice = nullptr;
  aclrtStream stream = nullptr;
  bool initialized = false;
  bool deviceSet = false;
  int rc = 0;
  int deviceId = 0;
  size_t inputSize = 0;

  ACL_CHECK(aclInit(nullptr));
  initialized = true;
  if (const char *envDevice = std::getenv("ACL_DEVICE_ID"))
    deviceId = std::atoi(envDevice);
  ACL_CHECK(aclrtSetDevice(deviceId));
  deviceSet = true;
  ACL_CHECK(aclrtCreateStream(&stream));

  ACL_CHECK(aclrtMallocHost(reinterpret_cast<void **>(&aHost), kASize));
  ACL_CHECK(aclrtMallocHost(reinterpret_cast<void **>(&bHost), kBSize));
  ACL_CHECK(aclrtMallocHost(reinterpret_cast<void **>(&aScaleHost), kScaleBytes));
  ACL_CHECK(aclrtMallocHost(reinterpret_cast<void **>(&bScaleHost), kScaleBytes));
  ACL_CHECK(aclrtMallocHost(reinterpret_cast<void **>(&cHost), kCSize));
  ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&aDevice), kASize,
                         ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&bDevice), kBSize,
                         ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&aScaleDevice), kScaleBytes,
                         ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&bScaleDevice), kScaleBytes,
                         ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&cDevice), kCSize,
                         ACL_MEM_MALLOC_HUGE_FIRST));

  inputSize = kASize;
  FILE_CHECK(ReadFile("./v1.bin", inputSize, aHost, kASize) && inputSize == kASize,
             "./v1.bin");
  inputSize = kBSize;
  FILE_CHECK(ReadFile("./v2.bin", inputSize, bHost, kBSize) && inputSize == kBSize,
             "./v2.bin");
  inputSize = kCSize;
  FILE_CHECK(ReadFile("./v3.bin", inputSize, cHost, kCSize) && inputSize == kCSize,
             "./v3.bin");
  inputSize = kScaleBytes;
  FILE_CHECK(ReadFile("./v4.bin", inputSize, aScaleHost, kScaleBytes) &&
                 inputSize == kScaleBytes, "./v4.bin");
  inputSize = kScaleBytes;
  FILE_CHECK(ReadFile("./v5.bin", inputSize, bScaleHost, kScaleBytes) &&
                 inputSize == kScaleBytes, "./v5.bin");

  ACL_CHECK(aclrtMemcpy(aDevice, kASize, aHost, kASize, ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(bDevice, kBSize, bHost, kBSize, ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(aScaleDevice, kScaleBytes, aScaleHost, kScaleBytes,
                        ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(bScaleDevice, kScaleBytes, bScaleHost, kScaleBytes,
                        ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(cDevice, kCSize, cHost, kCSize, ACL_MEMCPY_HOST_TO_DEVICE));

  LaunchMad_mx_e8m0_stage_abi_kernel(aDevice, bDevice, aScaleDevice,
                                      bScaleDevice, cDevice, stream);
  ACL_CHECK(aclrtSynchronizeStream(stream));
  ACL_CHECK(aclrtMemcpy(cHost, kCSize, cDevice, kCSize, ACL_MEMCPY_DEVICE_TO_HOST));
  FILE_CHECK(WriteFile("./v3.bin", cHost, kCSize), "./v3.bin");

cleanup:
  if (cDevice != nullptr)
    aclrtFree(cDevice);
  if (bScaleDevice != nullptr)
    aclrtFree(bScaleDevice);
  if (aScaleDevice != nullptr)
    aclrtFree(aScaleDevice);
  if (bDevice != nullptr)
    aclrtFree(bDevice);
  if (aDevice != nullptr)
    aclrtFree(aDevice);
  if (cHost != nullptr)
    aclrtFreeHost(cHost);
  if (bScaleHost != nullptr)
    aclrtFreeHost(bScaleHost);
  if (aScaleHost != nullptr)
    aclrtFreeHost(aScaleHost);
  if (bHost != nullptr)
    aclrtFreeHost(bHost);
  if (aHost != nullptr)
    aclrtFreeHost(aHost);
  if (stream != nullptr)
    aclrtDestroyStream(stream);
  if (deviceSet)
    aclrtResetDevice(deviceId);
  if (initialized)
    aclFinalize();
  return rc;
}
