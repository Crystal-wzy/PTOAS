# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""TileLang DSL template for pto.trowexpanddiv with IEEE 754 high-precision support

Divide each row of src0 by a per-row scalar from src1[row, 0].
Semantics: dst[row, col] = src0[row, col] / src1[row, 0]
"""

import sys
from pathlib import Path
import tilelang_dsl as pto

# Import shared high-precision division algorithms
from div_hp import _div_ieee754_f32_impl, _div_ieee754_f16_impl


def _constraint_trowexpanddiv_row_major(src0: pto.Tile, src1: pto.Tile, dst: pto.Tile) -> bool:
    """Constraint for RowMajor layout trowexpanddiv template."""
    # All tiles must be RowMajor layout
    src0_row_major = src0.config.b_layout == pto.BLayout.ROW_MAJOR
    src1_row_major = src1.config.b_layout == pto.BLayout.ROW_MAJOR
    dst_row_major = dst.config.b_layout == pto.BLayout.ROW_MAJOR
    return src0_row_major and src1_row_major and dst_row_major


@pto.vkernel(
    target="a5",
    op="pto.trowexpanddiv",
    dtypes=[(pto.f32, pto.f32, pto.f32)],
    constraints=[_constraint_trowexpanddiv_row_major],
)
def template_trowexpanddiv_f32(src0: pto.Tile, src1: pto.Tile, dst: pto.Tile):
    """Template for pto.trowexpanddiv with f32 dtype and optional high-precision mode."""
    dtype = dst.element_type
    valid_rows, valid_cols = dst.valid_shape

    precision_type = pto.get_op_attr("precisionType", "default")
    if pto.constexpr(precision_type == "high_precision"):
        for row in range(0, valid_rows, 1):
            remained = valid_cols
            for col in range(0, valid_cols, pto.get_lanes(dtype)):
                mask, remained = pto.make_mask(dtype, remained)
                # Load the scalar vector from src1[row, :]
                # For row-major src1, valid_shape[1] is 32/sizeof(dtype) (e.g., 8 for f32)
                # vdup broadcasts the first element to the full vector width
                scalar_vec = pto.vlds(src1[row, :])
                broadcasted = pto.vdup(scalar_vec, mask)
                lhs = pto.vlds(src0[row, col:])
                result = _div_ieee754_f32_impl(lhs, broadcasted, mask)
                pto.vsts(result, dst[row, col:], mask)
    else:
        for row in range(0, valid_rows, 1):
            remained = valid_cols
            for col in range(0, valid_cols, pto.get_lanes(dtype)):
                mask, remained = pto.make_mask(dtype, remained)
                # Load the scalar vector from src1[row, :]
                scalar_vec = pto.vlds(src1[row, :])
                broadcasted = pto.vdup(scalar_vec, mask)
                lhs = pto.vlds(src0[row, col:])
                result = pto.vdiv(lhs, broadcasted, mask)
                pto.vsts(result, dst[row, col:], mask)
    return


@pto.vkernel(
    target="a5",
    op="pto.trowexpanddiv",
    dtypes=[(pto.f16, pto.f16, pto.f16)],
    constraints=[_constraint_trowexpanddiv_row_major],
)
def template_trowexpanddiv_f16(src0: pto.Tile, src1: pto.Tile, dst: pto.Tile):
    """Template for pto.trowexpanddiv with f16 dtype and optional high-precision mode."""
    dtype = dst.element_type
    valid_rows, valid_cols = dst.valid_shape

    precision_type = pto.get_op_attr("precisionType", "default")
    if pto.constexpr(precision_type == "high_precision"):
        for row in range(0, valid_rows, 1):
            remained = valid_cols
            for col in range(0, valid_cols, pto.get_lanes(dtype)):
                mask, remained = pto.make_mask(dtype, remained)
                # Load the scalar vector from src1[row, :]
                # For row-major src1, valid_shape[1] is 32/sizeof(dtype) (e.g., 16 for f16)
                # vdup broadcasts the first element to the full vector width
                scalar_vec = pto.vlds(src1[row, :])
                broadcasted = pto.vdup(scalar_vec, mask)
                lhs = pto.vlds(src0[row, col:])
                result = _div_ieee754_f16_impl(lhs, broadcasted, mask)
                pto.vsts(result, dst[row, col:], mask)
    else:
        for row in range(0, valid_rows, 1):
            remained = valid_cols
            for col in range(0, valid_cols, pto.get_lanes(dtype)):
                mask, remained = pto.make_mask(dtype, remained)
                # Load the scalar vector from src1[row, :]
                scalar_vec = pto.vlds(src1[row, :])
                broadcasted = pto.vdup(scalar_vec, mask)
                lhs = pto.vlds(src0[row, col:])
                result = pto.vdiv(lhs, broadcasted, mask)
                pto.vsts(result, dst[row, col:], mask)
    return