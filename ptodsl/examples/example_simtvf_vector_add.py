#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""
SIMTVF vector example: GM vectorized load/store with ``pto.f32x2``.

Demonstrates loading packed ``vector<2xf32>`` values from GM via
``pto.ldg`` and storing them back via ``pto.stg``.  Each work-item
processes one packed 64-bit vector using a single GM load/store.

For ``!pto.ptr<vector<2xf32>, gm>`` the offset is in *vector* elements,
so offset 1 advances by 8 bytes (2 × sizeof(f32)).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ptodsl"))

from ptodsl import pto, scalar


@pto.simt
def simtvf_vector_copy_kernel(
    src: pto.ptr(pto.f32x2, "gm"),
    dst: pto.ptr(pto.f32x2, "gm"),
    nelem: pto.i32,
):
    """Each work-item copies one f32x2 vector from src to dst."""
    tid = pto.get_tid_x()
    idx = scalar.index_cast(tid)
    if tid < nelem:
        value = pto.ldg(src, idx, l1cache="cache", l2cache="nmfv")
        pto.stg(value, dst, idx, l1cache="uncache", l2cache="wtsred")


@pto.jit(target="a5")
def simtvf_vector_copy(
    src: pto.ptr(pto.f32x2, "gm"),
    dst: pto.ptr(pto.f32x2, "gm"),
    nelem: pto.i32,
):
    """Launch with 32 work-items (one warp)."""
    simtvf_vector_copy_kernel[32, 1, 1](src, dst, nelem)
