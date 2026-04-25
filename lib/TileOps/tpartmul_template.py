# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""TileLang DSL template for pto.tpartmul"""

import tilelang_dsl as pto


@pto.inline_proc
def tpart_op_instr(dst: pto.Tile, src0: pto.Tile, src1: pto.Tile, valid_rows, valid_cols):
    dtype = dst.element_type
    for row in range(0, valid_rows, 1):
        remained = valid_cols
        for col in range(0, valid_cols, pto.get_lanes(dtype)):
            mask, remained = pto.make_mask(dtype, remained)
            lhs = pto.vlds(src0[row, col:])
            rhs = pto.vlds(src1[row, col:])
            summed = pto.vmul(lhs, rhs, mask)
            pto.vsts(summed, dst[row, col:], mask)
    return None

@pto.inline_proc
def tpart_copy_instr(dst: pto.Tile, src: pto.Tile, valid_rows, valid_cols, start_row):
    dtype = dst.element_type
    for row in range(start_row, valid_rows, 1):
        remained = valid_cols
        for col in range(0, valid_cols, pto.get_lanes(dtype)):
            mask, remained = pto.make_mask(dtype, remained)
            val = pto.vlds(src[row, col:])
            pto.vsts(val, dst[row, col:], mask)
    return None

@pto.inline_proc
def tpart_op(dst: pto.Tile, src0: pto.Tile, src1: pto.Tile,
             dst_valid_rows, dst_valid_cols,
             src1_valid_rows, src1_valid_cols):

    src1_eq_dst = (src1_valid_rows == dst_valid_rows and src1_valid_cols == dst_valid_cols)
    src1_row_lt_dst = (src1_valid_rows < dst_valid_rows and src1_valid_cols == dst_valid_cols)
    src1_col_lt_dst = (src1_valid_rows <= dst_valid_rows and src1_valid_cols < dst_valid_cols)

    if src1_eq_dst:
        tpart_op_instr(dst, src0, src1, dst_valid_rows, dst_valid_cols)
    elif src1_col_lt_dst:
        tpart_copy_instr(dst, src0, dst_valid_rows, dst_valid_cols, 0)
        if src1_valid_cols > 0:
            tpart_op_instr(dst, src0, src1, src1_valid_rows, src1_valid_cols)
    elif src1_row_lt_dst:
        if src1_valid_cols > 0:
            tpart_op_instr(dst, src0, src1, src1_valid_rows, src1_valid_cols)
        tpart_copy_instr(dst, src0, dst_valid_rows, dst_valid_cols, src1_valid_rows)

    return

@pto.vkernel(
    target="a5",
    op="pto.tpartmul"
)
def template_tpartmul(src0: pto.Tile, src1: pto.Tile, dst: pto.Tile):
    dst_valid_rows, dst_valid_cols = dst.valid_shape
    src0_valid_rows, src0_valid_cols = src0.valid_shape
    src1_valid_rows, src1_valid_cols = src1.valid_shape

    src0_eq_dst = (src0_valid_rows == dst_valid_rows and src0_valid_cols == dst_valid_cols)
    src1_eq_dst = (src1_valid_rows == dst_valid_rows and src1_valid_cols == dst_valid_cols)

    if src0_eq_dst or src1_eq_dst:
        if src0_eq_dst:
            tpart_op(dst, src0, src1, dst_valid_rows, dst_valid_cols, src1_valid_rows, src1_valid_cols)
        elif src1_eq_dst:
            tpart_op(dst, src1, src0, dst_valid_rows, dst_valid_cols, src0_valid_rows, src0_valid_cols)
    # TODO: raise an error later

    return