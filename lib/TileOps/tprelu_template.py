# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""TileLang DSL template for pto.tprelu (Parametric ReLU)"""

import tilelang_dsl as pto


@pto.vkernel(
    target="a5",
    op="pto.tprelu",
    dtypes=[(pto.f16, pto.f16, pto.f16, pto.f16), (pto.f32, pto.f32, pto.f32, pto.f32),
            (pto.f16, pto.f16, pto.i8, pto.f16), (pto.f32, pto.f32, pto.i8, pto.f32)],
    advanced=True
)
def template_tprelu(src0: pto.Tile, src1: pto.Tile, tmp: pto.Tile, dst: pto.Tile):
    """Parametric ReLU: dst = src0 > 0 ? src0 : src0 * src1.
    
    Semantics:
    For each element (i, j):
        dst[i, j] = src0[i, j] > 0 ? src0[i, j] : src0[i, j] * src1[i, j]
    
    Supported data types: f16, f32
    A5 keeps the tmp operand in the ABI for cross-arch compatibility, but this
    implementation does not read or write it.
    """
    dtype = dst.element_type
    valid_rows, valid_cols = dst.valid_shape
    
    lanes = pto.get_lanes(dtype)
    
    for row in range(0, valid_rows, 1):
        remained = valid_cols
        for col in range(0, valid_cols, lanes):
            mask, remained = pto.make_mask(dtype, remained)
            vec0 = pto.vlds(src0[row, col:])
            vec1 = pto.vlds(src1[row, col:])
            result = pto.vprelu(vec0, vec1, mask)
            pto.vsts(result, dst[row, col:], mask)
    return
