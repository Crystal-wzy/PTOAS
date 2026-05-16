# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""TileLang DSL template for pto.tcolexpanddiv with IEEE 754 high-precision support

Divide each column of src0 by a per-column scalar from src1[0, col].
Semantics: dst[row, col] = src0[row, col] / src1[0, col]
"""

import sys
from pathlib import Path
import tilelang_dsl as pto

# Import shared high-precision division algorithms
from div_hp import _div_ieee754_f32_impl, _div_ieee754_f16_impl


@pto.vkernel(
    target="a5",
    op="pto.tcolexpanddiv"
)
def template_tcolexpanddiv(src0: pto.Tile, src1: pto.Tile, dst: pto.Tile):
    """Template for pto.tcolexpanddiv with optional high-precision mode."""
    dtype = dst.element_type
    valid_rows, valid_cols = dst.valid_shape

    precision_type = pto.get_op_attr("precisionType", "default")
    if pto.constexpr(precision_type == "high_precision"):
        for row in range(0, valid_rows, 1):
            remained = valid_cols
            for col in range(0, valid_cols, pto.get_lanes(dtype)):
                mask, remained = pto.make_mask(dtype, remained)
                lhs = pto.vlds(src0[row, col:])
                rhs = pto.vlds(src1[0, col:])
                if pto.constexpr(dtype == pto.f32):
                    result = _div_ieee754_f32_impl(lhs, rhs, mask)
                else:  # dtype == pto.f16 (guaranteed by MLIR validation)
                    result = _div_ieee754_f16_impl(lhs, rhs, mask)
                pto.vsts(result, dst[row, col:], mask)
    else:
        for row in range(0, valid_rows, 1):
            remained = valid_cols
            for col in range(0, valid_cols, pto.get_lanes(dtype)):
                mask, remained = pto.make_mask(dtype, remained)
                lhs = pto.vlds(src0[row, col:])
                rhs = pto.vlds(src1[0, col:])
                result = pto.vdiv(lhs, rhs, mask)
                pto.vsts(result, dst[row, col:], mask)
    return