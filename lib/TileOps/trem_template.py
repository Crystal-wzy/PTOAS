# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""TileLang DSL template for pto.trem"""

import tilelang_dsl as pto


@pto.vkernel(
    target="a5",
    op="pto.trem",
    dtypes=[
        (pto.f32, pto.f32, pto.f32, pto.f32),
        (pto.f16, pto.f16, pto.f16, pto.f16),
        (pto.i32, pto.i32, pto.i32, pto.i32),
    ],
    advanced=True,
)
def template_trem(src0: pto.Tile, src1: pto.Tile, tmp: pto.Tile, dst: pto.Tile):
    # A5 keeps tmp in the ABI for compatibility with A2/A3, but the template
    # computes directly in registers and never touches it.
    dtype = dst.element_type
    valid_rows, valid_cols = dst.valid_shape

    for row in range(0, valid_rows, 1):
        remained = valid_cols
        for col in range(0, valid_cols, pto.get_lanes(dtype)):
            mask, remained = pto.make_mask(dtype, remained)
            lhs = pto.vlds(src0[row, col:])
            rhs = pto.vlds(src1[row, col:])
            if pto.constexpr(dtype == pto.f32):
                quotient = pto.vdiv(lhs, rhs, mask)
                quotient = pto.vtrc(quotient, mask, rnd=pto.VcvtRoundMode.F)
                floored_mul = pto.vmul(quotient, rhs, mask)
                result = pto.vsub(lhs, floored_mul, mask)
                sign_diff_mask = pto.vcmps(pto.vmul(rhs, result, mask), 0.0, mask, pto.CmpMode.LT)
                corrected = pto.vadd(result, rhs, sign_diff_mask)
                result = pto.vsel(corrected, result, sign_diff_mask)
            elif pto.constexpr(dtype == pto.f16):
                lhs_even = pto.vcvt(lhs, pto.f32, mask, part=pto.VcvtPartMode.EVEN)
                rhs_even = pto.vcvt(rhs, pto.f32, mask, part=pto.VcvtPartMode.EVEN)
                lhs_odd = pto.vcvt(lhs, pto.f32, mask, part=pto.VcvtPartMode.ODD)
                rhs_odd = pto.vcvt(rhs, pto.f32, mask, part=pto.VcvtPartMode.ODD)
                q_even = pto.vdiv(lhs_even, rhs_even, mask)
                q_odd = pto.vdiv(lhs_odd, rhs_odd, mask)
                q_even = pto.vtrc(q_even, mask, rnd=pto.VcvtRoundMode.F)
                q_odd = pto.vtrc(q_odd, mask, rnd=pto.VcvtRoundMode.F)
                fm_even = pto.vmul(q_even, rhs_even, mask)
                fm_odd = pto.vmul(q_odd, rhs_odd, mask)
                r_even = pto.vsub(lhs_even, fm_even, mask)
                r_odd = pto.vsub(lhs_odd, fm_odd, mask)
                dst_even = pto.vcvt(r_even, pto.f16, mask, rnd=pto.VcvtRoundMode.Z, sat=pto.VcvtSatMode.RS_ENABLE, part=pto.VcvtPartMode.EVEN)
                dst_odd = pto.vcvt(r_odd, pto.f16, mask, rnd=pto.VcvtRoundMode.Z, sat=pto.VcvtSatMode.RS_ENABLE, part=pto.VcvtPartMode.ODD)
                result = pto.vor(dst_even, dst_odd, mask)
                sign_diff_mask = pto.vcmps(pto.vmul(rhs, result, mask), 0.0, mask, pto.CmpMode.LT)
                corrected = pto.vadd(result, rhs, sign_diff_mask)
                result = pto.vsel(corrected, result, sign_diff_mask)
            elif pto.constexpr(dtype == pto.i32):
                result = pto.vmod(lhs, rhs, mask)
            pto.vsts(result, dst[row, col:], mask)
    return
