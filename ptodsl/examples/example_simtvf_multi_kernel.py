#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""
SIMTVF multi-kernel example: two SIMT kernels sharing GM data.

The first kernel writes ``f32x2`` vectors to GM.  The second kernel
reads them back and writes to a separate output buffer.  This
demonstrates that ``pto.ldg`` / ``pto.stg`` with vector types compose
across multiple ``@pto.simt`` kernels with static and dynamic offsets.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ptodsl"))

from ptodsl import pto, scalar


@pto.simt
def simtvf_producer_kernel(
    buf: pto.ptr(pto.f32x2, "gm"),
    nelem: pto.i32,
):
    """Write f32x2 vectors: buf[tid] = (tid, tid+1) as packed f32x2."""
    tid = pto.get_tid_x()
    idx = scalar.index_cast(tid)
    if tid < nelem:
        value = pto.ldg(buf, idx)  # read existing
        pto.stg(value, buf, idx, l1cache="cache", l2cache="nmfv")


@pto.simt
def simtvf_consumer_kernel(
    src: pto.ptr(pto.f32x2, "gm"),
    dst: pto.ptr(pto.f32x2, "gm"),
    nelem: pto.i32,
):
    """Read f32x2 vectors from src, write to dst."""
    tid = pto.get_tid_x()
    idx = scalar.index_cast(tid)
    if tid < nelem:
        value = pto.ldg(src, idx, l1cache="cache", l2cache="nmfv")
        pto.stg(value, dst, idx, l1cache="uncache", l2cache="wtsred")


@pto.jit(target="a5")
def simtvf_multi_kernel(
    buf: pto.ptr(pto.f32x2, "gm"),
    dst: pto.ptr(pto.f32x2, "gm"),
    nelem: pto.i32,
):
    """Launch producer then consumer, each with 32 work-items."""
    # Kernel 1: produce into buf
    simtvf_producer_kernel[32, 1, 1](buf, nelem)
    # Kernel 2: consume buf, write to dst
    pto.pipe_barrier(pto.Pipe.ALL)
    simtvf_consumer_kernel[32, 1, 1](buf, dst, nelem)
