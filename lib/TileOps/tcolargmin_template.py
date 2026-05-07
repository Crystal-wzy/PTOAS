# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import sys
from pathlib import Path
import tilelang_dsl as pto


def _validate_tcolargmin(
    src_shape=(),
    src_valid_shape=(),
    tmp_shape=(),
    tmp_valid_shape=(),
    dst_shape=(),
    dst_valid_shape=(),
    src_config=None,
    tmp_config=None,
    dst_config=None,
    src_dtype=None,
    tmp_dtype=None,
    dst_dtype=None,
):
    if src_config is None or tmp_config is None or dst_config is None:
        return False
    if src_config.b_layout != pto.BLayout.ROW_MAJOR:
        return False
    if tmp_config.b_layout != pto.BLayout.ROW_MAJOR:
        return False
    if dst_config.b_layout != pto.BLayout.ROW_MAJOR:
        return False
    if src_config.s_layout != pto.SLayout.NONE_BOX:
        return False
    if tmp_config.s_layout != pto.SLayout.NONE_BOX:
        return False
    if dst_config.s_layout != pto.SLayout.NONE_BOX:
        return False
    if dst_valid_shape[0] != 1:
        return False
    if src_dtype != tmp_dtype:
        return False
    return True


@pto.vkernel(
    target="a5",
    op="pto.tcolargmin",
    dtypes=[
        (pto.ui8, pto.ui8, pto.i32),
        (pto.i8, pto.i8, pto.i32),
    ],
    constraints=[_validate_tcolargmin],
    advanced=True,
)
def template_tcolargmin_i8_to_i32(src: pto.Tile, tmp: pto.Tile, dst: pto.Tile):
    src_valid_rows, src_valid_cols = src.valid_shape
    src_dtype = src.element_type
    lanes_i8 = pto.get_lanes(src_dtype)
    lanes_i32 = pto.get_lanes(pto.i32)

    if pto.constexpr(src_dtype == pto.ui8):
        intermediate_dtype = pto.ui16
        final_cvt_dtype = pto.ui32
    else:
        intermediate_dtype = pto.i16
        final_cvt_dtype = pto.i32

    lanes_intermediate = pto.get_lanes(intermediate_dtype)

    with pto.vecscope():
        all_mask_b8 = pto.make_mask(src_dtype, pto.PAT.ALL)
        all_mask_intermediate = pto.make_mask(intermediate_dtype, pto.PAT.ALL)
        
        for col in range(0, src_valid_cols, lanes_i8):
            remained = src_valid_cols - col
            mask_i32_0, remained = pto.make_mask(pto.i32, remained)
            mask_i32_1, remained = pto.make_mask(pto.i32, remained)
            mask_i32_2, remained = pto.make_mask(pto.i32, remained)
            mask_i32_3, _ = pto.make_mask(pto.i32, remained)

            index_old_even = pto.vdup(intermediate_dtype(0), all_mask_intermediate)
            index_old_odd = pto.vdup(intermediate_dtype(0), all_mask_intermediate)
            index_new_even = pto.vdup(intermediate_dtype(0), all_mask_intermediate)
            index_new_odd = pto.vdup(intermediate_dtype(0), all_mask_intermediate)

            vreg_old = pto.vlds(src[0, col:])
            vreg_old_even = pto.vcvt(vreg_old, intermediate_dtype, all_mask_b8, part=pto.VcvtPartMode.EVEN)
            vreg_old_odd = pto.vcvt(vreg_old, intermediate_dtype, all_mask_b8, part=pto.VcvtPartMode.ODD)

            for row in range(1, src_valid_rows, 1):
                index_new_even = pto.vadds(index_new_even, intermediate_dtype(1), all_mask_intermediate)
                index_new_odd = pto.vadds(index_new_odd, intermediate_dtype(1), all_mask_intermediate)
                vreg_new = pto.vlds(src[row, col:])
                vreg_new_even = pto.vcvt(vreg_new, intermediate_dtype, all_mask_b8, part=pto.VcvtPartMode.EVEN)
                vreg_new_odd = pto.vcvt(vreg_new, intermediate_dtype, all_mask_b8, part=pto.VcvtPartMode.ODD)

                select_even = pto.vcmp(vreg_new_even, vreg_old_even, all_mask_intermediate, "lt")
                select_odd = pto.vcmp(vreg_new_odd, vreg_old_odd, all_mask_intermediate, "lt")

                index_old_even = pto.vsel(index_new_even, index_old_even, select_even)
                index_old_odd = pto.vsel(index_new_odd, index_old_odd, select_odd)

                vreg_old_even = pto.vmin(vreg_old_even, vreg_new_even, all_mask_intermediate)
                vreg_old_odd = pto.vmin(vreg_old_odd, vreg_new_odd, all_mask_intermediate)

            index_output_0, index_output_1 = pto.vintlv(index_old_even, index_old_odd)
            output_even = pto.vcvt(index_output_0, final_cvt_dtype, all_mask_intermediate, part=pto.VcvtPartMode.EVEN)
            output_odd = pto.vcvt(index_output_0, final_cvt_dtype, all_mask_intermediate, part=pto.VcvtPartMode.ODD)
            output_0, output_1 = pto.vintlv(output_even, output_odd)

            output_0 = pto.vbitcast(output_0, pto.i32)
            output_1 = pto.vbitcast(output_1, pto.i32)

            pto.vsts(output_0, dst[0, col:], mask_i32_0)
            pto.vsts(output_1, dst[0, col + lanes_i32:], mask_i32_1)

            output_even = pto.vcvt(index_output_1, final_cvt_dtype, all_mask_intermediate, part=pto.VcvtPartMode.EVEN)
            output_odd = pto.vcvt(index_output_1, final_cvt_dtype, all_mask_intermediate, part=pto.VcvtPartMode.ODD)
            output_0, output_1 = pto.vintlv(output_even, output_odd)

            output_0 = pto.vbitcast(output_0, pto.i32)
            output_1 = pto.vbitcast(output_1, pto.i32)

            pto.vsts(output_0, dst[0, col + 2 * lanes_i32:], mask_i32_2)
            pto.vsts(output_1, dst[0, col + 3 * lanes_i32:], mask_i32_3)

    return


@pto.vkernel(
    target="a5",
    op="pto.tcolargmin",
    dtypes=[
        (pto.f16, pto.f16, pto.i32),
        (pto.ui16, pto.ui16, pto.i32),
    ],
    constraints=[_validate_tcolargmin],
    advanced=True,
)
def template_tcolargmin_f16_to_i32(src: pto.Tile, tmp: pto.Tile, dst: pto.Tile):
    src_valid_rows, src_valid_cols = src.valid_shape
    lanes_f16 = pto.get_lanes(pto.f16)
    lanes_i32 = pto.get_lanes(pto.i32)

    with pto.vecscope():
        all_mask = pto.make_mask(pto.f16, pto.PAT.ALL)
        for col in range(0, src_valid_cols, lanes_f16):
            remained = src_valid_cols - col
            
            mask_f16, _ = pto.make_mask(pto.f16, remained)

            mask_i32_0, remained = pto.make_mask(pto.i32, remained)
            mask_i32_1, _ = pto.make_mask(pto.i32, remained)

            index_old = pto.vdup(pto.i16(0), mask_f16)
            index_new = pto.vdup(pto.i16(0), mask_f16)
            min_vals = pto.vlds(src[0, col:])

            for row in range(1, src_valid_rows, 1):
                index_new = pto.vadds(index_new, pto.i16(1), mask_f16)
                new_vals = pto.vlds(src[row, col:])
                lt_mask = pto.vcmp(new_vals, min_vals, all_mask, "lt")
                index_old = pto.vsel(index_new, index_old, lt_mask)
                min_vals = pto.vmin(min_vals, new_vals, mask_f16)

            index_even = pto.vcvt(index_old, pto.i32, all_mask, part=pto.VcvtPartMode.EVEN)
            index_odd = pto.vcvt(index_old, pto.i32, all_mask, part=pto.VcvtPartMode.ODD)
            index_lo, index_hi = pto.vintlv(index_even, index_odd)

            pto.vsts(index_lo, dst[0, col:], mask_i32_0)
            pto.vsts(index_hi, dst[0, col + lanes_i32:], mask_i32_1)

    return


@pto.vkernel(
    target="a5",
    op="pto.tcolargmin",
    dtypes=[
        (pto.f32, pto.f32, pto.i32),
        (pto.ui32, pto.ui32, pto.i32),
    ],
    constraints=[_validate_tcolargmin],
    advanced=True,
)
def template_tcolargmin_f32_to_i32(src: pto.Tile, tmp: pto.Tile, dst: pto.Tile):
    src_valid_rows, src_valid_cols = src.valid_shape
    lanes = pto.get_lanes(pto.f32)

    with pto.vecscope():
        remained = src_valid_cols
        for col in range(0, src_valid_cols, lanes):
            all_mask = pto.make_mask(pto.f32, pto.PAT.ALL)
            mask, remained = pto.make_mask(pto.f32, remained)

            index_old = pto.vdup(pto.i32(0), mask)
            index_new = pto.vdup(pto.i32(0), mask)
            min_vals = pto.vlds(src[0, col:])

            for row in range(1, src_valid_rows, 1):
                index_new = pto.vadds(index_new, pto.i32(1), mask)
                new_vals = pto.vlds(src[row, col:])
                lt_mask = pto.vcmp(new_vals, min_vals, all_mask, "lt")
                index_old = pto.vsel(index_new, index_old, lt_mask)
                min_vals = pto.vmin(min_vals, new_vals, mask)

            pto.vsts(index_old, dst[0, col:], mask)

    return
