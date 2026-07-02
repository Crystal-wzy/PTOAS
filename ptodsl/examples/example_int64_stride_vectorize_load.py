#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""
Int64 stride vectorized load example with the **addptr + castptr + ldg** contract.

Offset / alignment contract
----------------------------

1. ``pto.ldg`` / ``pto.stg`` offset is an **element** offset (not bytes).
   For ``!pto.ptr<vector<2xf32>, gm>``, offset 1 advances by 8 bytes
   (2 × sizeof(f32)).

2. When the original stride is expressed in scalar-f32 units, do **not**
   pass the scalar offset directly to a vector pointer — that would multiply
   the stride by ``sizeof(vector<2xf32>)`` instead of ``sizeof(f32)``,
   effectively doubling the advancement.  Instead, use the canonical pattern:

   .. code-block:: mlir

       %scalar_addr = pto.addptr %base, %scalar_offset
         : !pto.ptr<f32, gm> -> !pto.ptr<f32, gm>
       %vector_addr = pto.castptr %scalar_addr
         : !pto.ptr<f32, gm> -> !pto.ptr<vector<2xf32>, gm>
       %value = pto.ldg %vector_addr[%c0]
         : !pto.ptr<vector<2xf32>, gm> -> vector<2xf32>

   This guarantees the stride is applied in f32 units, not vector units.

3. The effective address for ``vector<2xf32>`` must satisfy 8-byte
   alignment.  The caller is responsible for ensuring the stride is a
   multiple of 2 f32 elements.  The op itself does not carry an alignment
   operand (first version; enforced by upstream ``T.assume``).

4. Dynamic ``i64`` scalar offsets must be explicitly cast to ``index`` via
   ``scalar.index_cast(value)`` before passing to ``pto.addptr``.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ptodsl"))

from ptodsl import pto, scalar


@pto.simt
def int64_stride_vectorize_load_kernel(
    base: pto.ptr(pto.f32, "gm"),       # scalar f32 base pointer
    dst: pto.ptr(pto.f32x2, "gm"),      # vector<2xf32> destination
    scalar_stride: pto.i64,             # stride in scalar f32 units
    nelem: pto.i32,
):
    """
    Stride-based vectorized load: each work-item loads one f32x2 vector
    at ``base[tid * scalar_stride]`` using the addptr + castptr contract.
    """
    tid = pto.get_tid_x()
    # Convert tid to i64 for stride arithmetic, then to index for addptr.
    tid_i64 = scalar.index_cast(pto.i64, scalar.index_cast(tid))
    offset_i64 = tid_i64 * scalar_stride
    scalar_offset = scalar.index_cast(offset_i64)       # i64 -> index

    # Contract step 1: advance the scalar f32 pointer by the scalar offset.
    scalar_addr = pto.addptr(base, scalar_offset)
    # Contract step 2: reinterpret as a vector<2xf32> pointer.
    vector_addr = pto.castptr(scalar_addr, pto.ptr(pto.f32x2, "gm"))
    # Contract step 3: load with offset 0 (addr already at target).
    # The scalar stride must be a multiple of 2 to satisfy 8-byte alignment.

    if tid < nelem:
        value = pto.ldg(vector_addr, 0, l1cache="cache", l2cache="nmfv")
        pto.stg(value, dst, 0, l1cache="uncache", l2cache="wtsred")


@pto.jit(target="a5")
def int64_stride_vectorize_load(
    base: pto.ptr(pto.f32, "gm"),
    dst: pto.ptr(pto.f32x2, "gm"),
    scalar_stride: pto.i64,
    nelem: pto.i32,
):
    """Launch with 32 work-items."""
    int64_stride_vectorize_load_kernel[32, 1, 1](base, dst, scalar_stride, nelem)


if __name__ == "__main__":
    compiled = int64_stride_vectorize_load.compile()
    print(compiled.mlir_text())
