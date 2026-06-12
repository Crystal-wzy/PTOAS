# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# coding=utf-8

import os
import sys
import numpy as np

from cases import CASES
from st_common import result_cmp, style_fail, style_pass


def bf16_to_f32(raw_bytes):
    """Convert raw bf16 bytes (2 bytes each) to f32 numpy array."""
    u16 = np.frombuffer(raw_bytes, dtype=np.uint16)
    u32 = u16.astype(np.uint32) << 16
    return u32.view(np.float32)


def nz_to_nd_16x16_f32(raw_bytes):
    """Convert 16x16 f32 NZ layout (N1=2, M1=1, M0=16, N0=8) to ND row-major."""
    nz = np.frombuffer(raw_bytes, dtype=np.float32).reshape(2, 1, 16, 8)
    return nz.transpose(1, 2, 0, 3).reshape(16, 16)


def nz_to_nd_32x32_f16(raw_bytes):
    """Convert 32x32 f16 NZ layout (K1=2, M1=2, M0=16, K0=16) to ND row-major.

    Physical layout: K1 x M1 x M0 x K0 where K1=N/K0 (outer K), M1=M/M0 (outer M).
    For output [M=N=32]: K1=2, M1=2.
    We need to convert to ND [M, N] = [M1, M0, K1, K0] ordering.
    """
    nz = np.frombuffer(raw_bytes, dtype=np.float16).reshape(2, 2, 16, 16)
    return nz.transpose(1, 2, 0, 3).reshape(32, 32)


def main():
    case_filter = sys.argv[1] if len(sys.argv) > 1 else None

    all_passed = True
    for case in CASES:
        if case_filter is not None and case["name"] != case_filter:
            continue

        case_dir = case["name"]
        m, n = case["m"], case["n"]
        dtype_out = case["dtype_out"]

        if not case.get("has_output", False):
            print(style_pass(f"[INFO] {case['name']}: compile-only (no output comparison)"))
            continue

        golden_path = os.path.join(case_dir, "golden.bin")
        output_path = os.path.join(case_dir, "output.bin")

        if not os.path.exists(output_path):
            print(style_fail(f"[ERROR] {case['name']}: output.bin not found"))
            all_passed = False
            continue

        golden = np.fromfile(golden_path, dtype=dtype_out)

        if case.get("out_bf16"):
            with open(output_path, "rb") as f:
                output = bf16_to_f32(f.read())
        else:
            out_dtype_map = {np.dtype(np.float16): np.float16, np.dtype(np.float32): np.float32}
            out_dtype = out_dtype_map.get(np.dtype(dtype_out), np.float32)
            output = np.fromfile(output_path, dtype=out_dtype)
            output = output.astype(np.float32)

        golden_f32 = golden.astype(np.float32)

        if case.get("nz_layout"):
            if m == 32 and n == 32:
                output = nz_to_nd_32x32_f16(output.astype(np.float16).tobytes()).astype(np.float32)
            else:
                print(style_fail(f"[ERROR] {case['name']}: unsupported nz_layout dims {m}x{n}"))
                all_passed = False
                continue

        if case.get("nz_layout_f32"):
            if m == 16 and n == 16:
                output = nz_to_nd_16x16_f32(output.astype(np.float32).tobytes())
            else:
                print(style_fail(f"[ERROR] {case['name']}: unsupported nz_layout_f32 dims {m}x{n}"))
                all_passed = False
                continue

        if golden_f32.shape != output.shape:
            print(style_fail(
                f"[ERROR] {case['name']}: shape mismatch golden={golden_f32.shape} output={output.shape}"
            ))
            all_passed = False
            continue

        ok = result_cmp(golden_f32.reshape(m, n), output, case["eps"])
        if ok:
            print(style_pass(f"[INFO] {case['name']}: compare passed"))
        else:
            print(style_fail(f"[ERROR] {case['name']}: compare failed"))
            all_passed = False

    if not all_passed:
        sys.exit(2)
    print(style_pass("[INFO] all cases passed"))


if __name__ == "__main__":
    main()
