# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""TileLang DSL template for pto.tdivs with IEEE 754 high-precision support

Supports two operand orders:
  1. tdivs(src_tile, scalar, dst) -> src / scalar
  2. tdivs(scalar, src_tile, dst) -> scalar / src

High-precision mode uses IEEE 754 compliant division algorithms from div_hp module
for improved accuracy with precision-sensitive, subnormal, and overflow boundary cases.
"""

import sys
from pathlib import Path
import tilelang_dsl as pto

# Import shared high-precision division algorithms
from div_hp import _div_ieee754_f32_impl, _div_ieee754_f16_impl


@pto.vkernel(
    target="a5",
    op="pto.tdivs",
)
def template_tdivs_tile_scalar(src: pto.Tile, scalar: pto.AnyType, dst: pto.Tile):
    """src / scalar with optional high-precision mode"""
    dtype = src.element_type
    valid_rows, valid_cols = src.valid_shape

    precision_type = pto.get_op_attr("precisionType", "default")
    if pto.constexpr(precision_type == "high_precision"):
        for row in range(0, valid_rows, 1):
            remained = valid_cols
            for col in range(0, valid_cols, pto.get_lanes(dtype)):
                mask, remained = pto.make_mask(dtype, remained)
                vec = pto.vlds(src[row, col:])
                scalar_vec = pto.vbr(scalar)
                if pto.constexpr(dtype == pto.f32):
                    result = _div_ieee754_f32_impl(vec, scalar_vec, mask)
                else:  # dtype == pto.f16 (guaranteed by MLIR validation)
                    result = _div_ieee754_f16_impl(vec, scalar_vec, mask)
                pto.vsts(result, dst[row, col:], mask)
    else:
        for row in range(0, valid_rows, 1):
            remained = valid_cols
            for col in range(0, valid_cols, pto.get_lanes(dtype)):
                mask, remained = pto.make_mask(dtype, remained)
                vec = pto.vlds(src[row, col:])
                scalar_vec = pto.vbr(scalar)
                result = pto.vdiv(vec, scalar_vec, mask)
                pto.vsts(result, dst[row, col:], mask)
    return


@pto.vkernel(
    target="a5",
    op="pto.tdivs",
)
def template_tdivs_scalar_tile(scalar: pto.AnyType, src: pto.Tile, dst: pto.Tile):
    """scalar / src with optional high-precision mode"""
    dtype = src.element_type
    valid_rows, valid_cols = src.valid_shape

    precision_type = pto.get_op_attr("precisionType", "default")
    if pto.constexpr(precision_type == "high_precision"):
        for row in range(0, valid_rows, 1):
            remained = valid_cols
            for col in range(0, valid_cols, pto.get_lanes(dtype)):
                mask, remained = pto.make_mask(dtype, remained)
                vec = pto.vlds(src[row, col:])
                scalar_vec = pto.vbr(scalar)
                if pto.constexpr(dtype == pto.f32):
                    result = _div_ieee754_f32_impl(scalar_vec, vec, mask)
                else:  # dtype == pto.f16 (guaranteed by MLIR validation)
                    result = _div_ieee754_f16_impl(scalar_vec, vec, mask)
                pto.vsts(result, dst[row, col:], mask)
    else:
        for row in range(0, valid_rows, 1):
            remained = valid_cols
            for col in range(0, valid_cols, pto.get_lanes(dtype)):
                mask, remained = pto.make_mask(dtype, remained)
                vec = pto.vlds(src[row, col:])
                scalar_vec = pto.vbr(scalar)
                result = pto.vdiv(scalar_vec, vec, mask)
                pto.vsts(result, dst[row, col:], mask)
    return