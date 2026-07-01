#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""
Int64 stride vectorized load example.

Demonstrates the correct offset/alignment contract for ``pto.ldg`` with
packed vector types and dynamic ``i64`` stride:

1. The ``offset`` operand of ``pto.ldg`` / ``pto.stg`` is an **element**
   offset, not a byte offset.  For ``!pto.ptr<vector<2xf32>, gm>``,
   offset 1 means "advance by one vector<2xf32>", i.e. 8 bytes.

2. When the original stride is expressed in scalar (f32) units, do **not**
   pass the scalar offset directly to a vector pointer — that would
   multiply the stride by sizeof(vector) instead of sizeof(scalar).
   Instead, divide the stride by the vector element count, or use the
   ``addptr`` + ``pointer_cast`` pattern to advance a scalar pointer
   first, then cast to a vector pointer and load with offset 0.

3. ``dynamic i64`` offsets must be explicitly cast to ``index`` via
   ``scalar.index_cast(value)`` before passing to ``pto.ldg`` / ``pto.stg``.

4. The effective address for ``vector<2xf32>`` must satisfy 8-byte
   alignment (enforced by the upstream call-site contract).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ptodsl"))

from ptodsl import pto, scalar


# Number of f32 elements per f32x2 vector
_ELEMS_PER_VEC = 2


@pto.simt
def int64_stride_vectorize_load_kernel(
    base: pto.ptr(pto.f32x2, "gm"),    # vector-typed base pointer
    dst: pto.ptr(pto.f32x2, "gm"),     # vector-typed destination
    scalar_stride: pto.i64,            # stride in *scalar* (f32) units
    nelem: pto.i32,
):
    """
    Vectorized load with dynamic i64 stride.

    Each work-item loads one f32x2 vector from ``base`` at offset
    ``tid * (scalar_stride // 2)`` (stride scaled to vector units).
    """
    tid = pto.get_tid_x()
    tid_i64 = scalar.convert(scalar.index_cast(tid), pto.i64)
    # Scale stride from scalar units to vector units
    vec_stride = tid_i64 * (scalar_stride // _ELEMS_PER_VEC)
    idx = scalar.index_cast(vec_stride)  # i64 → index

    if tid < nelem:
        value = pto.ldg(base, idx, l1cache="cache", l2cache="nmfv")
        pto.stg(value, dst, idx, l1cache="uncache", l2cache="wtsred")


@pto.jit(target="a5")
def int64_stride_vectorize_load(
    base: pto.ptr(pto.f32x2, "gm"),
    dst: pto.ptr(pto.f32x2, "gm"),
    scalar_stride: pto.i64,
    nelem: pto.i32,
):
    """Launch with 32 work-items, each accessing at tid × (stride/2) vector offset."""
    int64_stride_vectorize_load_kernel[32, 1, 1](base, dst, scalar_stride, nelem)
