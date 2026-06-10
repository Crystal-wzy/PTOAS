# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""TileLang DSL template for pto.tcvt."""

import tilelang_dsl as pto


def _config_value(config, name):
    if config is None:
        return None
    if isinstance(config, dict):
        return config.get(name)
    return getattr(config, name, None)


def _matches_layout(value, expected, expected_name):
    if value is None:
        return False
    return value == expected or value == expected_name or str(value).lower().endswith(expected_name)


def _supports_basic_rowwise_tcvt(
    src=None,
    dst=None,
    src_shape=(),
    src_valid_shape=(),
    dst_shape=(),
    dst_valid_shape=(),
):
    if tuple(src_shape) != tuple(dst_shape):
        return False
    if tuple(src_valid_shape) != tuple(dst_valid_shape):
        return False
    if len(src_shape) != 2 or len(dst_shape) != 2:
        return False
    if src is None or dst is None:
        return False
    src_config = src.config
    dst_config = dst.config
    if src_config is None or dst_config is None:
        return False
    if not _matches_layout(_config_value(src_config, "b_layout"), pto.BLayout.ROW_MAJOR, "row_major"):
        return False
    if not _matches_layout(_config_value(dst_config, "b_layout"), pto.BLayout.ROW_MAJOR, "row_major"):
        return False
    if not _matches_layout(_config_value(src_config, "s_layout"), pto.SLayout.NONE_BOX, "none_box"):
        return False
    if not _matches_layout(_config_value(dst_config, "s_layout"), pto.SLayout.NONE_BOX, "none_box"):
        return False
    return True


def _supports_bf16_to_fp4_rowwise_tcvt(
    src=None,
    dst=None,
    src_shape=(),
    src_valid_shape=(),
    dst_shape=(),
    dst_valid_shape=(),
):
    if len(src_shape) != 2 or len(dst_shape) != 2:
        return False
    if src_shape[0] != dst_shape[0] or src_shape[1] != dst_shape[1] * 2:
        return False
    if src_valid_shape[0] != dst_valid_shape[0] or src_valid_shape[1] != dst_valid_shape[1] * 2:
        return False
    if src is None or dst is None:
        return False
    src_config = src.config
    dst_config = dst.config
    if src_config is None or dst_config is None:
        return False
    if not _matches_layout(_config_value(src_config, "b_layout"), pto.BLayout.ROW_MAJOR, "row_major"):
        return False
    if not _matches_layout(_config_value(dst_config, "b_layout"), pto.BLayout.ROW_MAJOR, "row_major"):
        return False
    if not _matches_layout(_config_value(src_config, "s_layout"), pto.SLayout.NONE_BOX, "none_box"):
        return False
    if not _matches_layout(_config_value(dst_config, "s_layout"), pto.SLayout.NONE_BOX, "none_box"):
        return False
    return True


@pto.vkernel(
    target="a5",
    op="pto.tcvt",
    dtypes=[
        (pto.f32, pto.f16),
    ],
    constraints=[_supports_basic_rowwise_tcvt],
)
def template_tcvt_f32_to_f16(src: pto.Tile, dst: pto.Tile):
    valid_rows, valid_cols = dst.valid_shape
    round_mode = pto.get_op_attr("round_mode", "RINT")
    rnd = pto.VcvtRoundMode.R
    if pto.constexpr(round_mode == "ROUND"):
        rnd = pto.VcvtRoundMode.A
    elif pto.constexpr(round_mode == "FLOOR"):
        rnd = pto.VcvtRoundMode.F
    elif pto.constexpr(round_mode == "CEIL"):
        rnd = pto.VcvtRoundMode.C
    elif pto.constexpr(round_mode == "TRUNC"):
        rnd = pto.VcvtRoundMode.Z
    elif pto.constexpr(round_mode == "ODD"):
        rnd = pto.VcvtRoundMode.O

    full_mask = pto.make_mask(pto.f32, pto.PAT.ALL)
    for row in range(0, valid_rows, 1):
        remained = valid_cols
        for col in range(0, valid_cols, pto.get_lanes(pto.f32)):
            store_mask, remained = pto.make_mask(pto.f32, remained)
            vec = pto.vlds(src[row, col:])
            converted = pto.vcvt(
                vec,
                pto.f16,
                full_mask,
                rnd=rnd,
                sat=pto.VcvtSatMode.SAT,
                part=pto.VcvtPartMode.EVEN,
            )
            pto.vsts(converted, dst[row, col:], store_mask, dist=pto.VStoreDist.PK_B32)
    return


@pto.vkernel(
    target="a5",
    op="pto.tcvt",
    dtypes=[
        (pto.f32, pto.i32),
    ],
    constraints=[_supports_basic_rowwise_tcvt],
)
def template_tcvt_f32_to_i32(src: pto.Tile, dst: pto.Tile):
    dst_dtype = dst.element_type
    valid_rows, valid_cols = dst.valid_shape
    round_mode = pto.get_op_attr("round_mode", "RINT")
    rnd = pto.VcvtRoundMode.R
    if pto.constexpr(round_mode == "ROUND"):
        rnd = pto.VcvtRoundMode.A
    elif pto.constexpr(round_mode == "FLOOR"):
        rnd = pto.VcvtRoundMode.F
    elif pto.constexpr(round_mode == "CEIL"):
        rnd = pto.VcvtRoundMode.C
    elif pto.constexpr(round_mode == "TRUNC"):
        rnd = pto.VcvtRoundMode.Z
    elif pto.constexpr(round_mode == "ODD"):
        rnd = pto.VcvtRoundMode.O

    for row in range(0, valid_rows, 1):
        remained = valid_cols
        for col in range(0, valid_cols, pto.get_lanes(dst_dtype)):
            mask, remained = pto.make_mask(dst_dtype, remained)
            vec = pto.vlds(src[row, col:])
            converted = pto.vcvt(
                vec,
                dst_dtype,
                mask,
                rnd=rnd,
                sat=pto.VcvtSatMode.SAT,
            )
            pto.vsts(converted, dst[row, col:], mask)
    return


@pto.vkernel(
    target="a5",
    op="pto.tcvt",
    dtypes=[
        (pto.i32, pto.f32),
    ],
    constraints=[_supports_basic_rowwise_tcvt],
)
def template_tcvt_i32_to_f32(src: pto.Tile, dst: pto.Tile):
    dst_dtype = dst.element_type
    valid_rows, valid_cols = dst.valid_shape
    round_mode = pto.get_op_attr("round_mode", "RINT")
    rnd = pto.VcvtRoundMode.R
    if pto.constexpr(round_mode == "ROUND"):
        rnd = pto.VcvtRoundMode.A
    elif pto.constexpr(round_mode == "FLOOR"):
        rnd = pto.VcvtRoundMode.F
    elif pto.constexpr(round_mode == "CEIL"):
        rnd = pto.VcvtRoundMode.C
    elif pto.constexpr(round_mode == "TRUNC"):
        rnd = pto.VcvtRoundMode.Z
    elif pto.constexpr(round_mode == "ODD"):
        rnd = pto.VcvtRoundMode.O

    for row in range(0, valid_rows, 1):
        remained = valid_cols
        for col in range(0, valid_cols, pto.get_lanes(dst_dtype)):
            mask, remained = pto.make_mask(dst_dtype, remained)
            vec = pto.vlds(src[row, col:])
            converted = pto.vcvt(
                vec,
                dst_dtype,
                mask,
                rnd=rnd,
            )
            pto.vsts(converted, dst[row, col:], mask)
    return


@pto.vkernel(
    target="a5",
    op="pto.tcvt",
    dtypes=[
        (pto.f16, pto.f32),
        (pto.bf16, pto.f32),
        (pto.i16, pto.f32),
        (pto.i16, pto.i32),
        (pto.i16, pto.ui32),
    ],
    constraints=[_supports_basic_rowwise_tcvt],
)
def template_tcvt_16_to_32(src: pto.Tile, dst: pto.Tile):
    src_dtype = src.element_type
    dst_dtype = dst.element_type
    valid_rows, valid_cols = dst.valid_shape
    full_mask = pto.make_mask(src_dtype, pto.PAT.ALL)
    for row in range(0, valid_rows, 1):
        remained = valid_cols
        for col in range(0, valid_cols, pto.get_lanes(dst_dtype)):
            store_mask, remained = pto.make_mask(dst_dtype, remained)
            vec = pto.vlds(src[row, col:], dist=pto.VLoadDist.UNPK_B16)
            converted = pto.vcvt(
                vec,
                dst_dtype,
                full_mask,
                part=pto.VcvtPartMode.EVEN,
            )
            pto.vsts(converted, dst[row, col:], store_mask)
    return


@pto.vkernel(
    target="a5",
    op="pto.tcvt",
    dtypes=[
        (pto.i32, pto.i64),
    ],
    constraints=[_supports_basic_rowwise_tcvt],
)
def template_tcvt_i32_to_i64(src: pto.Tile, dst: pto.Tile):
    valid_rows, valid_cols = dst.valid_shape
    full_mask = pto.make_mask(pto.i32, pto.PAT.ALL)
    for row in range(0, valid_rows, 1):
        remained = valid_cols * 2  # i64 requires double the mask
        for col in range(0, valid_cols, pto.get_lanes(pto.i64)):
            store_mask, remained = pto.make_mask(pto.i64, remained)
            vec = pto.vlds(src[row, col:], dist=pto.VLoadDist.UNPK_B32)
            converted = pto.vcvt(
                vec,
                pto.i64,
                full_mask,
                part=pto.VcvtPartMode.EVEN,
            )
            pto.vsts(converted, dst[row, col:], store_mask, dist=pto.VStoreDist.NORM_B32)
    return


@pto.vkernel(
    target="a5",
    op="pto.tcvt",
    dtypes=[
        (pto.ui8, pto.f16),
    ],
    constraints=[_supports_basic_rowwise_tcvt],
)
def template_tcvt_ui8_to_f16(src: pto.Tile, dst: pto.Tile):
    valid_rows, valid_cols = dst.valid_shape
    full_mask = pto.make_mask(pto.ui8, pto.PAT.ALL)
    for row in range(0, valid_rows, 1):
        remained = valid_cols
        for col in range(0, valid_cols, pto.get_lanes(pto.f16)):
            store_mask, remained = pto.make_mask(pto.f16, remained)
            vec = pto.vlds(src[row, col:], dist=pto.VLoadDist.UNPK_B8)
            converted = pto.vcvt(
                vec,
                pto.f16,
                full_mask,
                part=pto.VcvtPartMode.EVEN,
            )
            pto.vsts(converted, dst[row, col:], store_mask)
    return


@pto.vkernel(
    target="a5",
    op="pto.tcvt",
    dtypes=[
        (pto.ui8, pto.ui16),
    ],
    constraints=[_supports_basic_rowwise_tcvt],
)
def template_tcvt_ui8_to_ui16(src: pto.Tile, dst: pto.Tile):
    valid_rows, valid_cols = dst.valid_shape
    full_mask = pto.make_mask(pto.ui8, pto.PAT.ALL)
    for row in range(0, valid_rows, 1):
        remained = valid_cols
        for col in range(0, valid_cols, pto.get_lanes(pto.ui16)):
            store_mask, remained = pto.make_mask(pto.ui16, remained)
            vec = pto.vlds(src[row, col:], dist=pto.VLoadDist.UNPK_B8)
            converted = pto.vcvt(
                vec,
                pto.ui16,
                full_mask,
                part=pto.VcvtPartMode.EVEN,
            )
            pto.vsts(converted, dst[row, col:], store_mask)
    return


@pto.vkernel(
    target="a5",
    op="pto.tcvt",
    dtypes=[
        (pto.si8, pto.f16),
    ],
    constraints=[_supports_basic_rowwise_tcvt],
)
def template_tcvt_si8_to_f16(src: pto.Tile, dst: pto.Tile):
    valid_rows, valid_cols = dst.valid_shape
    full_mask = pto.make_mask(pto.si8, pto.PAT.ALL)
    for row in range(0, valid_rows, 1):
        remained = valid_cols
        for col in range(0, valid_cols, pto.get_lanes(pto.f16)):
            store_mask, remained = pto.make_mask(pto.f16, remained)
            vec = pto.vlds(src[row, col:], dist=pto.VLoadDist.UNPK_B8)
            converted = pto.vcvt(
                vec,
                pto.f16,
                full_mask,
                part=pto.VcvtPartMode.EVEN,
            )
            pto.vsts(converted, dst[row, col:], store_mask)
    return


@pto.vkernel(
    target="a5",
    op="pto.tcvt",
    dtypes=[
        (pto.si8, pto.si16),
    ],
    constraints=[_supports_basic_rowwise_tcvt],
)
def template_tcvt_si8_to_si16(src: pto.Tile, dst: pto.Tile):
    valid_rows, valid_cols = dst.valid_shape
    full_mask = pto.make_mask(pto.si8, pto.PAT.ALL)
    for row in range(0, valid_rows, 1):
        remained = valid_cols
        for col in range(0, valid_cols, pto.get_lanes(pto.si16)):
            store_mask, remained = pto.make_mask(pto.si16, remained)
            vec = pto.vlds(src[row, col:], dist=pto.VLoadDist.UNPK_B8)
            converted = pto.vcvt(
                vec,
                pto.si16,
                full_mask,
                part=pto.VcvtPartMode.EVEN,
            )
            pto.vsts(converted, dst[row, col:], store_mask, dist=pto.VStoreDist.NORM_B16)
    return


@pto.vkernel(
    target="a5",
    op="pto.tcvt",
    dtypes=[
        (pto.si8, pto.i32),
    ],
    constraints=[_supports_basic_rowwise_tcvt],
    advanced=True,
)
def template_tcvt_si8_to_i32(src: pto.Tile, dst: pto.Tile):
    valid_rows, valid_cols = dst.valid_shape
    b8_mask = pto.make_mask(pto.ui8, pto.PAT.ALL)
    v_zero = pto.vdup(pto.ui8(0), b8_mask)
    lanes_i32 = pto.get_lanes(pto.i32)
    for row in range(0, valid_rows, 1):
        remained = valid_cols
        next_remained = 0
        if valid_cols > lanes_i32:
            next_remained = valid_cols - lanes_i32
        for col in range(0, valid_cols, pto.get_lanes(pto.i16)):
            mask_b16_cur, remained = pto.make_mask(pto.i16, remained)
            mask_b16_next, next_remained = pto.make_mask(pto.i16, next_remained)
            mask_b32_cur = pto.punpack(mask_b16_cur, pto.PredicatePart.LOWER)
            mask_b32_next = pto.punpack(mask_b16_next, pto.PredicatePart.LOWER)
            vec_si8_0 = pto.vlds(src[row, col:], dist=pto.VLoadDist.UNPK_B8)
            vec_ui8_0 = pto.vbitcast(vec_si8_0, pto.ui8)
            vec_ui8_1, vec_ui8_2 = pto.vintlv(vec_ui8_0, v_zero)
            vec_si8_1 = pto.vbitcast(vec_ui8_1, pto.si8)
            vec_si8_2 = pto.vbitcast(vec_ui8_2, pto.si8)
            output_0 = pto.vcvt(vec_si8_1, pto.i32, b8_mask, part=pto.VcvtPartMode.P0)
            output_1 = pto.vcvt(vec_si8_2, pto.i32, b8_mask, part=pto.VcvtPartMode.P0)
            pto.vsts(output_0, dst[row, col:], mask_b32_cur, dist=pto.VStoreDist.NORM_B32)
            pto.vsts(output_1, dst[row, col + lanes_i32:], mask_b32_next, dist=pto.VStoreDist.NORM_B32)
    return


@pto.vkernel(
    target="a5",
    op="pto.tcvt",
    dtypes=[
        (pto.f32, pto.f32),
    ],
    constraints=[_supports_basic_rowwise_tcvt],
)
def template_tcvt_f32_to_f32(src: pto.Tile, dst: pto.Tile):
    valid_rows, valid_cols = dst.valid_shape
    round_mode = pto.get_op_attr("round_mode", "RINT")
    rnd = pto.VcvtRoundMode.R
    if pto.constexpr(round_mode == "ROUND"):
        rnd = pto.VcvtRoundMode.A
    elif pto.constexpr(round_mode == "FLOOR"):
        rnd = pto.VcvtRoundMode.F
    elif pto.constexpr(round_mode == "CEIL"):
        rnd = pto.VcvtRoundMode.C
    elif pto.constexpr(round_mode == "TRUNC"):
        rnd = pto.VcvtRoundMode.Z

    for row in range(0, valid_rows, 1):
        remained = valid_cols
        for col in range(0, valid_cols, pto.get_lanes(pto.f32)):
            mask, remained = pto.make_mask(pto.f32, remained)
            vec = pto.vlds(src[row, col:])
            converted = pto.vtrc(vec, mask, rnd=rnd)
            pto.vsts(converted, dst[row, col:], mask)
    return


@pto.vkernel(
    target="a5",
    op="pto.tcvt",
    dtypes=[
        (pto.f16, pto.i32),
    ],
    constraints=[_supports_basic_rowwise_tcvt],
)
def template_tcvt_f16_to_i32(src: pto.Tile, dst: pto.Tile):
    valid_rows, valid_cols = dst.valid_shape
    round_mode = pto.get_op_attr("round_mode", "RINT")
    rnd = pto.VcvtRoundMode.R
    if pto.constexpr(round_mode == "ROUND"):
        rnd = pto.VcvtRoundMode.A
    elif pto.constexpr(round_mode == "FLOOR"):
        rnd = pto.VcvtRoundMode.F
    elif pto.constexpr(round_mode == "CEIL"):
        rnd = pto.VcvtRoundMode.C
    elif pto.constexpr(round_mode == "TRUNC"):
        rnd = pto.VcvtRoundMode.Z
    elif pto.constexpr(round_mode == "ODD"):
        rnd = pto.VcvtRoundMode.O

    full_mask = pto.make_mask(pto.f16, pto.PAT.ALL)
    for row in range(0, valid_rows, 1):
        remained = valid_cols
        for col in range(0, valid_cols, pto.get_lanes(pto.i32)):
            store_mask, remained = pto.make_mask(pto.i32, remained)
            vec = pto.vlds(src[row, col:], dist=pto.VLoadDist.UNPK_B16)
            converted = pto.vcvt(
                vec,
                pto.i32,
                full_mask,
                rnd=rnd,
                part=pto.VcvtPartMode.EVEN,
            )
            pto.vsts(converted, dst[row, col:], store_mask)
    return


@pto.vkernel(
    target="a5",
    op="pto.tcvt",
    dtypes=[
        (pto.i16, pto.f16),
    ],
    constraints=[_supports_basic_rowwise_tcvt],
)
def template_tcvt_i16_to_f16(src: pto.Tile, dst: pto.Tile):
    valid_rows, valid_cols = dst.valid_shape
    round_mode = pto.get_op_attr("round_mode", "RINT")
    rnd = pto.VcvtRoundMode.R
    if pto.constexpr(round_mode == "ROUND"):
        rnd = pto.VcvtRoundMode.A
    elif pto.constexpr(round_mode == "FLOOR"):
        rnd = pto.VcvtRoundMode.F
    elif pto.constexpr(round_mode == "CEIL"):
        rnd = pto.VcvtRoundMode.C
    elif pto.constexpr(round_mode == "TRUNC"):
        rnd = pto.VcvtRoundMode.Z
    elif pto.constexpr(round_mode == "ODD"):
        rnd = pto.VcvtRoundMode.O

    full_mask = pto.make_mask(pto.i16, pto.PAT.ALL)
    for row in range(0, valid_rows, 1):
        remained = valid_cols
        for col in range(0, valid_cols, pto.get_lanes(pto.i16)):
            store_mask, remained = pto.make_mask(pto.f16, remained)
            vec = pto.vlds(src[row, col:])
            converted = pto.vcvt(
                vec,
                pto.f16,
                full_mask,
                rnd=rnd,
            )
            pto.vsts(converted, dst[row, col:], store_mask)
    return


@pto.vkernel(
    target="a5",
    op="pto.tcvt",
    dtypes=[
        (pto.i64, pto.f32),
    ],
    constraints=[_supports_basic_rowwise_tcvt],
)
def template_tcvt_i64_to_f32(src: pto.Tile, dst: pto.Tile):
    valid_rows, valid_cols = dst.valid_shape
    round_mode = pto.get_op_attr("round_mode", "RINT")
    rnd = pto.VcvtRoundMode.R
    if pto.constexpr(round_mode == "ROUND"):
        rnd = pto.VcvtRoundMode.A
    elif pto.constexpr(round_mode == "FLOOR"):
        rnd = pto.VcvtRoundMode.F
    elif pto.constexpr(round_mode == "CEIL"):
        rnd = pto.VcvtRoundMode.C
    elif pto.constexpr(round_mode == "TRUNC"):
        rnd = pto.VcvtRoundMode.Z
    elif pto.constexpr(round_mode == "ODD"):
        rnd = pto.VcvtRoundMode.O

    for row in range(0, valid_rows, 1):
        remained = valid_cols * 2  # i64 requires double the mask
        full_mask, _ = pto.make_mask(pto.i64, remained)
        for col in range(0, valid_cols, pto.get_lanes(pto.i64)):
            store_mask, remained = pto.make_mask(pto.f32, remained)
            vec = pto.vlds(src[row, col:])
            converted = pto.vcvt(
                vec,
                pto.f32,
                full_mask,
                rnd=rnd,
                part=pto.VcvtPartMode.EVEN,
            )
            pto.vsts(converted, dst[row, col:], store_mask, dist=pto.VStoreDist.PK_B64)
    return


@pto.vkernel(
    target="a5",
    op="pto.tcvt",
    dtypes=[
        (pto.i16, pto.ui8),
    ],
    constraints=[_supports_basic_rowwise_tcvt],
)
def template_tcvt_i16_to_ui8(src: pto.Tile, dst: pto.Tile):
    valid_rows, valid_cols = dst.valid_shape
    full_mask = pto.make_mask(pto.i16, pto.PAT.ALL)
    for row in range(0, valid_rows, 1):
        remained = valid_cols
        for col in range(0, valid_cols, pto.get_lanes(pto.i16)):
            store_mask, remained = pto.make_mask(pto.i16, remained)
            vec = pto.vlds(src[row, col:])
            converted = pto.vcvt(
                vec,
                pto.ui8,
                full_mask,
                sat=pto.VcvtSatMode.SAT,
                part=pto.VcvtPartMode.EVEN,
            )
            pto.vsts(converted, dst[row, col:], store_mask, dist=pto.VStoreDist.PK_B16)
    return


@pto.vkernel(
    target="a5",
    op="pto.tcvt",
    dtypes=[
        (pto.i32, pto.i16),
    ],
    constraints=[_supports_basic_rowwise_tcvt],
)
def template_tcvt_i32_to_i16(src: pto.Tile, dst: pto.Tile):
    valid_rows, valid_cols = dst.valid_shape
    full_mask = pto.make_mask(pto.i32, pto.PAT.ALL)
    for row in range(0, valid_rows, 1):
        remained = valid_cols
        for col in range(0, valid_cols, pto.get_lanes(pto.i32)):
            store_mask, remained = pto.make_mask(pto.i32, remained)
            vec = pto.vlds(src[row, col:])
            converted = pto.vcvt(
                vec,
                pto.i16,
                full_mask,
                sat=pto.VcvtSatMode.NOSAT,
                part=pto.VcvtPartMode.EVEN,
            )
            pto.vsts(converted, dst[row, col:], store_mask, dist=pto.VStoreDist.PK_B32)
    return


@pto.vkernel(
    target="a5",
    op="pto.tcvt",
    dtypes=[
        (pto.i32, pto.ui16),
    ],
    constraints=[_supports_basic_rowwise_tcvt],
)
def template_tcvt_i32_to_ui16(src: pto.Tile, dst: pto.Tile):
    valid_rows, valid_cols = dst.valid_shape
    full_mask = pto.make_mask(pto.i32, pto.PAT.ALL)
    for row in range(0, valid_rows, 1):
        remained = valid_cols
        for col in range(0, valid_cols, pto.get_lanes(pto.i32)):
            store_mask, remained = pto.make_mask(pto.i32, remained)
            vec = pto.vlds(src[row, col:])
            converted = pto.vcvt(
                vec,
                pto.ui16,
                full_mask,
                sat=pto.VcvtSatMode.SAT,
                part=pto.VcvtPartMode.EVEN,
            )
            pto.vsts(converted, dst[row, col:], store_mask, dist=pto.VStoreDist.PK_B32)
    return


@pto.vkernel(
    target="a5",
    op="pto.tcvt",
    dtypes=[
        (pto.i32, pto.ui8),
    ],
    constraints=[_supports_basic_rowwise_tcvt],
    advanced=True,
)
def template_tcvt_i32_to_ui8(src: pto.Tile, dst: pto.Tile):
    valid_rows, valid_cols = dst.valid_shape
    full_mask = pto.make_mask(pto.i32, pto.PAT.ALL)
    idx_mask_b8 = pto.pset_b8(pto.PAT.ALL)
    idx_mask_b16 = pto.pbitcast(idx_mask_b8, pto.mask_b16)
    lanes_i32 = pto.get_lanes(pto.i32)
    v_idx = pto.vci(pto.i8(0), pto.OrderMode.ASC)
    v_idx_i16 = pto.vbitcast(v_idx, pto.i16)
    v_idx_i16 = pto.vmuls(v_idx_i16, pto.i16(4), idx_mask_b16)
    v_idx_ui8 = pto.vbitcast(v_idx_i16, pto.ui8)
    for row in range(0, valid_rows, 1):
        mask_len_tail = valid_cols % lanes_i32
        if valid_cols % lanes_i32 == 0:
            mask_len_tail = lanes_i32
        for col in range(0, valid_cols, lanes_i32):
            mask_len = lanes_i32
            if valid_cols < lanes_i32 or col == valid_cols - lanes_i32:
                mask_len = mask_len_tail
            store_mask, _ = pto.make_mask(pto.ui8, mask_len)
            vec = pto.vlds(src[row, col:])
            converted = pto.vcvt(
                vec,
                pto.ui8,
                full_mask,
                sat=pto.VcvtSatMode.SAT,
                part=pto.VcvtPartMode.P0,
            )
            result = pto.vselr(converted, v_idx_ui8)
            pto.mem_bar(pto.BarrierType.VST_VST)
            pto.vsts(result, dst[row, col:], store_mask, dist=pto.VStoreDist.NORM_B8)
    return


@pto.vkernel(
    target="a5",
    op="pto.tcvt",
    dtypes=[
        (pto.ui32, pto.i16),
    ],
    constraints=[_supports_basic_rowwise_tcvt],
)
def template_tcvt_ui32_to_i16(src: pto.Tile, dst: pto.Tile):
    valid_rows, valid_cols = dst.valid_shape
    full_mask = pto.make_mask(pto.ui32, pto.PAT.ALL)
    for row in range(0, valid_rows, 1):
        remained = valid_cols
        for col in range(0, valid_cols, pto.get_lanes(pto.ui32)):
            store_mask, remained = pto.make_mask(pto.ui32, remained)
            vec = pto.vlds(src[row, col:])
            converted = pto.vcvt(
                vec,
                pto.i16,
                full_mask,
                sat=pto.VcvtSatMode.SAT,
                part=pto.VcvtPartMode.EVEN,
            )
            pto.vsts(converted, dst[row, col:], store_mask, dist=pto.VStoreDist.PK_B32)
    return


@pto.vkernel(
    target="a5",
    op="pto.tcvt",
    dtypes=[
        (pto.ui32, pto.ui16),
    ],
    constraints=[_supports_basic_rowwise_tcvt],
)
def template_tcvt_ui32_to_ui16(src: pto.Tile, dst: pto.Tile):
    valid_rows, valid_cols = dst.valid_shape
    full_mask = pto.make_mask(pto.ui32, pto.PAT.ALL)
    for row in range(0, valid_rows, 1):
        remained = valid_cols
        for col in range(0, valid_cols, pto.get_lanes(pto.ui32)):
            store_mask, remained = pto.make_mask(pto.ui32, remained)
            vec = pto.vlds(src[row, col:])
            converted = pto.vcvt(
                vec,
                pto.ui16,
                full_mask,
                sat=pto.VcvtSatMode.SAT,
                part=pto.VcvtPartMode.EVEN,
            )
            pto.vsts(converted, dst[row, col:], store_mask, dist=pto.VStoreDist.PK_B32)
    return


@pto.vkernel(
    target="a5",
    op="pto.tcvt",
    dtypes=[
        (pto.ui32, pto.ui8),
    ],
    constraints=[_supports_basic_rowwise_tcvt],
    advanced=True,
)
def template_tcvt_ui32_to_ui8(src: pto.Tile, dst: pto.Tile):
    valid_rows, valid_cols = dst.valid_shape
    full_mask = pto.make_mask(pto.ui32, pto.PAT.ALL)
    idx_mask_b8 = pto.pset_b8(pto.PAT.ALL)
    idx_mask_b16 = pto.pbitcast(idx_mask_b8, pto.mask_b16)
    lanes_ui32 = pto.get_lanes(pto.ui32)
    v_idx = pto.vci(pto.i8(0), pto.OrderMode.ASC)
    v_idx_i16 = pto.vbitcast(v_idx, pto.i16)
    v_idx_i16 = pto.vmuls(v_idx_i16, pto.i16(4), idx_mask_b16)
    v_idx_ui8 = pto.vbitcast(v_idx_i16, pto.ui8)
    for row in range(0, valid_rows, 1):
        mask_len_tail = valid_cols % lanes_ui32
        if valid_cols % lanes_ui32 == 0:
            mask_len_tail = lanes_ui32
        for col in range(0, valid_cols, lanes_ui32):
            mask_len = lanes_ui32
            if valid_cols < lanes_ui32 or col == valid_cols - lanes_ui32:
                mask_len = mask_len_tail
            store_mask, _ = pto.make_mask(pto.ui8, mask_len)
            vec = pto.vlds(src[row, col:])
            converted = pto.vcvt(
                vec,
                pto.ui8,
                full_mask,
                sat=pto.VcvtSatMode.SAT,
                part=pto.VcvtPartMode.P0,
            )
            result = pto.vselr(converted, v_idx_ui8)
            pto.mem_bar(pto.BarrierType.VST_VST)
            pto.vsts(result, dst[row, col:], store_mask, dist=pto.VStoreDist.NORM_B8)
    return


@pto.vkernel(
    target="a5",
    op="pto.tcvt",
    dtypes=[
        (pto.i64, pto.i32),
    ],
    constraints=[_supports_basic_rowwise_tcvt],
)
def template_tcvt_i64_to_i32(src: pto.Tile, dst: pto.Tile):
    valid_rows, valid_cols = dst.valid_shape
    for row in range(0, valid_rows, 1):
        remained = valid_cols * 2  # i64 requires double the mask
        full_mask, _ = pto.make_mask(pto.i64, remained)
        for col in range(0, valid_cols, pto.get_lanes(pto.i64)):
            store_mask, remained = pto.make_mask(pto.i32, remained)
            vec = pto.vlds(src[row, col:])
            converted = pto.vcvt(
                vec,
                pto.i32,
                full_mask,
                sat=pto.VcvtSatMode.NOSAT,
                part=pto.VcvtPartMode.EVEN,
            )
            pto.vsts(converted, dst[row, col:], store_mask, dist=pto.VStoreDist.PK_B64)
    return


@pto.vkernel(
    target="a5",
    op="pto.tcvt",
    dtypes=[
        (pto.f32, pto.bf16),
    ],
    constraints=[_supports_basic_rowwise_tcvt],
)
def template_tcvt_f32_to_bf16(src: pto.Tile, dst: pto.Tile):
    valid_rows, valid_cols = dst.valid_shape
    round_mode = pto.get_op_attr("round_mode", "RINT")
    rnd = pto.VcvtRoundMode.R
    if pto.constexpr(round_mode == "ROUND"):
        rnd = pto.VcvtRoundMode.A
    elif pto.constexpr(round_mode == "FLOOR"):
        rnd = pto.VcvtRoundMode.F
    elif pto.constexpr(round_mode == "CEIL"):
        rnd = pto.VcvtRoundMode.C
    elif pto.constexpr(round_mode == "TRUNC"):
        rnd = pto.VcvtRoundMode.Z
    elif pto.constexpr(round_mode == "ODD"):
        rnd = pto.VcvtRoundMode.O

    full_mask = pto.make_mask(pto.f32, pto.PAT.ALL)
    for row in range(0, valid_rows, 1):
        remained = valid_cols
        for col in range(0, valid_cols, pto.get_lanes(pto.f32)):
            store_mask, remained = pto.make_mask(pto.f32, remained)
            vec = pto.vlds(src[row, col:])
            converted = pto.vcvt(
                vec,
                pto.bf16,
                full_mask,
                rnd=rnd,
                sat=pto.VcvtSatMode.SAT,
                part=pto.VcvtPartMode.EVEN,
            )
            pto.vsts(converted, dst[row, col:], store_mask, dist=pto.VStoreDist.PK_B32)
    return


@pto.vkernel(
    target="a5",
    op="pto.tcvt",
    dtypes=[
        (pto.f32, pto.i64),
    ],
    constraints=[_supports_basic_rowwise_tcvt],
)
def template_tcvt_f32_to_i64(src: pto.Tile, dst: pto.Tile):
    valid_rows, valid_cols = dst.valid_shape
    round_mode = pto.get_op_attr("round_mode", "RINT")
    rnd = pto.VcvtRoundMode.R
    if pto.constexpr(round_mode == "ROUND"):
        rnd = pto.VcvtRoundMode.A
    elif pto.constexpr(round_mode == "FLOOR"):
        rnd = pto.VcvtRoundMode.F
    elif pto.constexpr(round_mode == "CEIL"):
        rnd = pto.VcvtRoundMode.C
    elif pto.constexpr(round_mode == "TRUNC"):
        rnd = pto.VcvtRoundMode.Z
    elif pto.constexpr(round_mode == "ODD"):
        rnd = pto.VcvtRoundMode.O

    full_mask = pto.make_mask(pto.f32, pto.PAT.ALL)
    for row in range(0, valid_rows, 1):
        remained = valid_cols * 2  # i64 requires double the mask
        for col in range(0, valid_cols, pto.get_lanes(pto.i64)):
            store_mask, remained = pto.make_mask(pto.i64, remained)
            vec = pto.vlds(src[row, col:], dist=pto.VLoadDist.UNPK_B32)
            converted = pto.vcvt(
                vec,
                pto.i64,
                full_mask,
                rnd=rnd,
                sat=pto.VcvtSatMode.SAT,
                part=pto.VcvtPartMode.EVEN,
            )
            pto.vsts(converted, dst[row, col:], store_mask, dist=pto.VStoreDist.NORM_B32)
    return


@pto.vkernel(
    target="a5",
    op="pto.tcvt",
    dtypes=[
        (pto.f16, pto.ui8),
    ],
    constraints=[_supports_basic_rowwise_tcvt],
)
def template_tcvt_f16_to_ui8(src: pto.Tile, dst: pto.Tile):
    valid_rows, valid_cols = dst.valid_shape
    round_mode = pto.get_op_attr("round_mode", "RINT")
    rnd = pto.VcvtRoundMode.R
    if pto.constexpr(round_mode == "ROUND"):
        rnd = pto.VcvtRoundMode.A
    elif pto.constexpr(round_mode == "FLOOR"):
        rnd = pto.VcvtRoundMode.F
    elif pto.constexpr(round_mode == "CEIL"):
        rnd = pto.VcvtRoundMode.C
    elif pto.constexpr(round_mode == "TRUNC"):
        rnd = pto.VcvtRoundMode.Z
    elif pto.constexpr(round_mode == "ODD"):
        rnd = pto.VcvtRoundMode.O

    full_mask = pto.make_mask(pto.f16, pto.PAT.ALL)
    for row in range(0, valid_rows, 1):
        remained = valid_cols
        for col in range(0, valid_cols, pto.get_lanes(pto.f16)):
            store_mask, remained = pto.make_mask(pto.f16, remained)
            vec = pto.vlds(src[row, col:])
            converted = pto.vcvt(
                vec,
                pto.ui8,
                full_mask,
                rnd=rnd,
                sat=pto.VcvtSatMode.NOSAT,
                part=pto.VcvtPartMode.EVEN,
            )
            pto.vsts(converted, dst[row, col:], store_mask, dist=pto.VStoreDist.PK_B16)
    return


@pto.vkernel(
    target="a5",
    op="pto.tcvt",
    dtypes=[
        (pto.bf16, pto.i32),
    ],
    constraints=[_supports_basic_rowwise_tcvt],
)
def template_tcvt_bf16_to_i32(src: pto.Tile, dst: pto.Tile):
    valid_rows, valid_cols = dst.valid_shape
    round_mode = pto.get_op_attr("round_mode", "RINT")
    rnd = pto.VcvtRoundMode.R
    if pto.constexpr(round_mode == "ROUND"):
        rnd = pto.VcvtRoundMode.A
    elif pto.constexpr(round_mode == "FLOOR"):
        rnd = pto.VcvtRoundMode.F
    elif pto.constexpr(round_mode == "CEIL"):
        rnd = pto.VcvtRoundMode.C
    elif pto.constexpr(round_mode == "TRUNC"):
        rnd = pto.VcvtRoundMode.Z
    elif pto.constexpr(round_mode == "ODD"):
        rnd = pto.VcvtRoundMode.O

    full_mask = pto.make_mask(pto.bf16, pto.PAT.ALL)
    for row in range(0, valid_rows, 1):
        remained = valid_cols
        for col in range(0, valid_cols, pto.get_lanes(pto.i32)):
            store_mask, remained = pto.make_mask(pto.i32, remained)
            vec = pto.vlds(src[row, col:], dist=pto.VLoadDist.UNPK_B16)
            converted = pto.vcvt(
                vec,
                pto.i32,
                full_mask,
                rnd=rnd,
                sat=pto.VcvtSatMode.SAT,
                part=pto.VcvtPartMode.EVEN,
            )
            pto.vsts(converted, dst[row, col:], store_mask)
    return


@pto.vkernel(
    target="a5",
    op="pto.tcvt",
    dtypes=[
        (pto.bf16, pto.f16),
    ],
    constraints=[_supports_basic_rowwise_tcvt],
)
def template_tcvt_bf16_to_f16(src: pto.Tile, dst: pto.Tile):
    valid_rows, valid_cols = dst.valid_shape
    round_mode = pto.get_op_attr("round_mode", "RINT")
    rnd = pto.VcvtRoundMode.R
    if pto.constexpr(round_mode == "ROUND"):
        rnd = pto.VcvtRoundMode.A
    elif pto.constexpr(round_mode == "FLOOR"):
        rnd = pto.VcvtRoundMode.F
    elif pto.constexpr(round_mode == "CEIL"):
        rnd = pto.VcvtRoundMode.C
    elif pto.constexpr(round_mode == "TRUNC"):
        rnd = pto.VcvtRoundMode.Z
    elif pto.constexpr(round_mode == "ODD"):
        rnd = pto.VcvtRoundMode.O

    full_mask = pto.make_mask(pto.bf16, pto.PAT.ALL)
    for row in range(0, valid_rows, 1):
        remained = valid_cols
        for col in range(0, valid_cols, pto.get_lanes(pto.bf16)):
            store_mask, remained = pto.make_mask(pto.f16, remained)
            vec = pto.vlds(src[row, col:])
            converted = pto.vcvt(
                vec,
                pto.f16,
                full_mask,
                sat=pto.VcvtSatMode.SAT,
                rnd=rnd,
            )
            pto.vsts(converted, dst[row, col:], store_mask)
    return


@pto.vkernel(
    target="a5",
    op="pto.tcvt",
    dtypes=[
        (pto.f32, pto.i16),
    ],
    constraints=[_supports_basic_rowwise_tcvt],
)
def template_tcvt_f32_to_i16(src: pto.Tile, dst: pto.Tile):
    valid_rows, valid_cols = dst.valid_shape
    round_mode = pto.get_op_attr("round_mode", "RINT")
    rnd = pto.VcvtRoundMode.R
    if pto.constexpr(round_mode == "ROUND"):
        rnd = pto.VcvtRoundMode.A
    elif pto.constexpr(round_mode == "FLOOR"):
        rnd = pto.VcvtRoundMode.F
    elif pto.constexpr(round_mode == "CEIL"):
        rnd = pto.VcvtRoundMode.C
    elif pto.constexpr(round_mode == "TRUNC"):
        rnd = pto.VcvtRoundMode.Z
    elif pto.constexpr(round_mode == "ODD"):
        rnd = pto.VcvtRoundMode.O

    full_mask = pto.make_mask(pto.f32, pto.PAT.ALL)
    for row in range(0, valid_rows, 1):
        remained = valid_cols
        for col in range(0, valid_cols, pto.get_lanes(pto.f32)):
            store_mask, remained = pto.make_mask(pto.f32, remained)
            vec_f32 = pto.vlds(src[row, col:])
            # sat=OFF NonSatTorch
            vec_i32 = pto.vcvt(
                vec_f32,
                pto.i32,
                full_mask,
                rnd=rnd,
                sat=pto.VcvtSatMode.NOSAT,
            )
            vec_i16 = pto.vcvt(
                vec_i32,
                pto.i16,
                full_mask,
                sat=pto.VcvtSatMode.NOSAT,
                part=pto.VcvtPartMode.EVEN,
            )
            pto.vsts(vec_i16, dst[row, col:], store_mask, dist=pto.VStoreDist.PK_B32)
    return


@pto.vkernel(
    target="a5",
    op="pto.tcvt",
    dtypes=[
        (pto.f16, pto.i16),
    ],
    constraints=[_supports_basic_rowwise_tcvt],
)
def template_tcvt_f16_to_i16(src: pto.Tile, dst: pto.Tile):
    valid_rows, valid_cols = dst.valid_shape
    round_mode = pto.get_op_attr("round_mode", "RINT")
    rnd = pto.VcvtRoundMode.R
    if pto.constexpr(round_mode == "ROUND"):
        rnd = pto.VcvtRoundMode.A
    elif pto.constexpr(round_mode == "FLOOR"):
        rnd = pto.VcvtRoundMode.F
    elif pto.constexpr(round_mode == "CEIL"):
        rnd = pto.VcvtRoundMode.C
    elif pto.constexpr(round_mode == "TRUNC"):
        rnd = pto.VcvtRoundMode.Z
    elif pto.constexpr(round_mode == "ODD"):
        rnd = pto.VcvtRoundMode.O

    full_mask_b16 = pto.make_mask(pto.f16, pto.PAT.ALL)
    full_mask_b32 = pto.make_mask(pto.i32, pto.PAT.ALL)
    for row in range(0, valid_rows, 1):
        remained = valid_cols
        for col in range(0, valid_cols, pto.get_lanes(pto.f32)):
            store_mask, remained = pto.make_mask(pto.i32, remained)
            vec_f16 = pto.vlds(src[row, col:], dist=pto.VLoadDist.UNPK_B16)
            # sat=OFF NonSatTorch
            vec_i32 = pto.vcvt(
                vec_f16,
                pto.i32,
                full_mask_b16,
                rnd=rnd,
                part=pto.VcvtPartMode.EVEN,
            )
            vec_i16 = pto.vcvt(
                vec_i32,
                pto.i16,
                full_mask_b32,
                sat=pto.VcvtSatMode.NOSAT,
                part=pto.VcvtPartMode.EVEN,
            )
            pto.vsts(vec_i16, dst[row, col:], store_mask, dist=pto.VStoreDist.PK_B32)
    return


@pto.vkernel(
    target="a5",
    op="pto.tcvt",
    dtypes=[
        (pto.f16, pto.si8),
    ],
    constraints=[_supports_basic_rowwise_tcvt],
)
def template_tcvt_f16_to_si8(src: pto.Tile, dst: pto.Tile):
    valid_rows, valid_cols = dst.valid_shape
    round_mode = pto.get_op_attr("round_mode", "RINT")
    rnd = pto.VcvtRoundMode.R
    if pto.constexpr(round_mode == "ROUND"):
        rnd = pto.VcvtRoundMode.A
    elif pto.constexpr(round_mode == "FLOOR"):
        rnd = pto.VcvtRoundMode.F
    elif pto.constexpr(round_mode == "CEIL"):
        rnd = pto.VcvtRoundMode.C
    elif pto.constexpr(round_mode == "TRUNC"):
        rnd = pto.VcvtRoundMode.Z
    elif pto.constexpr(round_mode == "ODD"):
        rnd = pto.VcvtRoundMode.O

    lanes_f16 = pto.get_lanes(pto.f16)
    pg = pto.make_mask(pto.f16, pto.PAT.ALL)
    for row in range(0, valid_rows, 1):
        remained = valid_cols
        for col in range(0, valid_cols, lanes_f16):
            full_mask, _ = pto.make_mask(pto.f16, lanes_f16)
            store_mask, remained = pto.make_mask(pto.f16, remained)
            vec_f16 = pto.vlds(src[row, col:])
            # sat=OFF NonSatTorch
            vec_i16 = pto.vcvt(
                vec_f16,
                pto.i16,
                full_mask,
                rnd=rnd,
                sat=pto.VcvtSatMode.NOSAT,
            )
            v_mask = pto.vdup(pto.i16(255), pg)
            vec_i16_and = pto.vand(vec_i16, v_mask, store_mask)
            vec_f16_temp = pto.vcvt(
                vec_i16_and,
                pto.f16,
                full_mask,
                rnd=rnd,
            )
            vec_si8 = pto.vcvt(
                vec_f16_temp,
                pto.si8,
                full_mask,
                rnd=rnd,
                sat=pto.VcvtSatMode.NOSAT,
                part=pto.VcvtPartMode.EVEN,
            )
            pto.vsts(vec_si8, dst[row, col:], store_mask, dist=pto.VStoreDist.PK_B16)
    return


@pto.vkernel(
    target="a5",
    op="pto.tcvt",
    dtypes=[
        (pto.f32, pto.f8e4m3),
        (pto.f32, pto.f8e5m2),
    ],
    constraints=[_supports_basic_rowwise_tcvt],
    advanced=True,
)
def template_tcvt_f32_to_fp8(src: pto.Tile, dst: pto.Tile):
    dst_dtype = dst.element_type
    valid_rows, valid_cols = dst.valid_shape
    round_mode = pto.get_op_attr("round_mode", "RINT")
    rnd = pto.VcvtRoundMode.R
    if pto.constexpr(round_mode == "ROUND"):
        rnd = pto.VcvtRoundMode.A
    elif pto.constexpr(round_mode == "FLOOR"):
        rnd = pto.VcvtRoundMode.F
    elif pto.constexpr(round_mode == "CEIL"):
        rnd = pto.VcvtRoundMode.C
    elif pto.constexpr(round_mode == "TRUNC"):
        rnd = pto.VcvtRoundMode.Z
    elif pto.constexpr(round_mode == "ODD"):
        rnd = pto.VcvtRoundMode.O
    lanes_f32 = pto.get_lanes(pto.f32)
    src_mask = pto.make_mask(pto.f32, pto.PAT.ALL)
    idx_mask_b8 = pto.pset_b8(pto.PAT.ALL)
    idx_mask_b16 = pto.pbitcast(idx_mask_b8, pto.mask_b16)
    v_idx = pto.vci(pto.i8(0), pto.OrderMode.ASC)
    v_idx_i16 = pto.vbitcast(v_idx, pto.i16)
    v_idx_i16 = pto.vmuls(v_idx_i16, pto.i16(4), idx_mask_b16)
    v_idx_ui8 = pto.vbitcast(v_idx_i16, pto.ui8)
    for row in range(0, valid_rows, 1):
        for col in range(0, valid_cols, lanes_f32):
            mask_len = valid_cols - col
            if mask_len > lanes_f32:
                mask_len = lanes_f32
            dst_mask, _ = pto.make_mask(dst_dtype, mask_len)
            vec = pto.vlds(src[row, col:])
            converted = pto.vcvt(
                vec,
                dst_dtype,
                src_mask,
                rnd=rnd,
                sat=pto.VcvtSatMode.SAT,
                part=pto.VcvtPartMode.P0,
            )
            result = pto.vselr(converted, v_idx_ui8)
            pto.mem_bar(pto.BarrierType.VST_VST)
            pto.vsts(result, dst[row, col:], dst_mask, dist=pto.VStoreDist.NORM_B8)
    return


@pto.vkernel(
    target="a5",
    op="pto.tcvt",
    dtypes=[
        (pto.f32, pto.hif8),
    ],
    constraints=[_supports_basic_rowwise_tcvt],
    advanced=True,
)
def template_tcvt_f32_to_hif8(src: pto.Tile, dst: pto.Tile):
    dst_dtype = dst.element_type
    valid_rows, valid_cols = dst.valid_shape
    round_mode = pto.get_op_attr("round_mode", "RINT")
    rnd = pto.VcvtRoundMode.A
    if pto.constexpr(round_mode == "ROUND"):
        rnd = pto.VcvtRoundMode.A
    lanes_f32 = pto.get_lanes(pto.f32)
    src_mask = pto.make_mask(pto.f32, pto.PAT.ALL)
    idx_mask_b8 = pto.pset_b8(pto.PAT.ALL)
    idx_mask_b16 = pto.pbitcast(idx_mask_b8, pto.mask_b16)
    v_idx = pto.vci(pto.i8(0), pto.OrderMode.ASC)
    v_idx_i16 = pto.vbitcast(v_idx, pto.i16)
    v_idx_i16 = pto.vmuls(v_idx_i16, pto.i16(4), idx_mask_b16)
    v_idx_ui8 = pto.vbitcast(v_idx_i16, pto.ui8)
    for row in range(0, valid_rows, 1):
        for col in range(0, valid_cols, lanes_f32):
            mask_len = valid_cols - col
            if mask_len > lanes_f32:
                mask_len = lanes_f32
            dst_mask, _ = pto.make_mask(dst_dtype, mask_len)
            vec = pto.vlds(src[row, col:])
            converted = pto.vcvt(
                vec,
                dst_dtype,
                src_mask,
                rnd=rnd,
                sat=pto.VcvtSatMode.NOSAT,
                part=pto.VcvtPartMode.P0,
            )
            result = pto.vselr(converted, v_idx_ui8)
            pto.mem_bar(pto.BarrierType.VST_VST)
            pto.vsts(result, dst[row, col:], dst_mask, dist=pto.VStoreDist.NORM_B8)
    return


@pto.vkernel(
    target="a5",
    op="pto.tcvt",
    dtypes=[
        (pto.f16, pto.hif8),
    ],
    constraints=[_supports_basic_rowwise_tcvt],
)
def template_tcvt_f16_to_hif8(src: pto.Tile, dst: pto.Tile):
    dst_dtype = dst.element_type
    valid_rows, valid_cols = dst.valid_shape
    round_mode = pto.get_op_attr("round_mode", "RINT")
    rnd = pto.VcvtRoundMode.A
    if pto.constexpr(round_mode == "ROUND"):
        rnd = pto.VcvtRoundMode.A
    src_mask = pto.make_mask(pto.f16, pto.PAT.ALL)
    for row in range(0, valid_rows, 1):
        remained = valid_cols
        for col in range(0, valid_cols, pto.get_lanes(pto.f16)):
            dst_mask, remained = pto.make_mask(pto.f16, remained)
            vec = pto.vlds(src[row, col:])
            converted = pto.vcvt(
                vec,
                dst_dtype,
                src_mask,
                rnd=rnd,
                sat=pto.VcvtSatMode.NOSAT,
                part=pto.VcvtPartMode.EVEN,
            )
            pto.vsts(converted, dst[row, col:], dst_mask, dist=pto.VStoreDist.PK_B16)
    return


@pto.vkernel(
    target="a5",
    op="pto.tcvt",
    dtypes=[
        (pto.bf16, pto.f4e1m2x2),
        (pto.bf16, pto.f4e2m1x2),
    ],
    constraints=[_supports_bf16_to_fp4_rowwise_tcvt],
    advanced=True,
)
def template_tcvt_bf16_to_fp4(src: pto.Tile, dst: pto.Tile):
    dst_dtype = dst.element_type
    valid_rows, valid_cols = dst.valid_shape
    round_mode = pto.get_op_attr("round_mode", "RINT")
    rnd = pto.VcvtRoundMode.R
    if pto.constexpr(round_mode == "ROUND"):
        rnd = pto.VcvtRoundMode.A
    elif pto.constexpr(round_mode == "FLOOR"):
        rnd = pto.VcvtRoundMode.F
    elif pto.constexpr(round_mode == "CEIL"):
        rnd = pto.VcvtRoundMode.C
    elif pto.constexpr(round_mode == "TRUNC"):
        rnd = pto.VcvtRoundMode.Z
    elif pto.constexpr(round_mode == "ODD"):
        rnd = pto.VcvtRoundMode.O
    lanes_bf16 = pto.get_lanes(pto.bf16)
    dst_chunk_cols = lanes_bf16 // 2
    idx_mask_b8 = pto.pset_b8(pto.PAT.ALL)
    idx_mask_b16 = pto.pbitcast(idx_mask_b8, pto.mask_b16)
    v_idx = pto.vci(pto.i8(0), pto.OrderMode.ASC)
    v_idx_i16 = pto.vbitcast(v_idx, pto.i16)
    v_idx_i16 = pto.vmuls(v_idx_i16, pto.i16(4), idx_mask_b16)
    v_idx_ui8 = pto.vbitcast(v_idx_i16, pto.ui8)
    for row in range(0, valid_rows, 1):
        remained = valid_cols
        for col in range(0, valid_cols, dst_chunk_cols):
            dst_mask, remained = pto.make_mask(dst_dtype, remained)
            store_len = valid_cols - col
            if store_len > dst_chunk_cols:
                store_len = dst_chunk_cols
            src_mask, _ = pto.make_mask(pto.bf16, store_len * 2)
            vec = pto.vlds(src[row, col * 2:])
            converted = pto.vcvt(
                vec,
                dst_dtype,
                src_mask,
                rnd=rnd,
                part=pto.VcvtPartMode.P0,
            )
            result = pto.vselr(converted, v_idx_ui8)
            pto.mem_bar(pto.BarrierType.VST_VST)
            pto.vsts(result, dst[row, col:], dst_mask, dist=pto.VStoreDist.NORM_B8)
    return
