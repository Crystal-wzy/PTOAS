# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""PTODSL TileLib templates for row-wise ``pto.tcvt`` paths."""

from ptodsl import pto
import ptodsl.tilelib as tilelib


def _rowwise(src_shape, src_valid_shape, dst_shape, dst_valid_shape, src_config, dst_config, **_):
    return (
        tuple(src_shape) == tuple(dst_shape)
        and tuple(src_valid_shape) == tuple(dst_valid_shape)
        and src_config.b_layout == "row_major"
        and dst_config.b_layout == "row_major"
        and src_config.s_layout == "none_box"
        and dst_config.s_layout == "none_box"
    )


def _round_mode():
    round_mode = pto.get_op_attr("round_mode", "RINT")
    if round_mode == "ROUND":
        return pto.VcvtRoundMode.A
    if round_mode == "FLOOR":
        return pto.VcvtRoundMode.F
    if round_mode == "CEIL":
        return pto.VcvtRoundMode.C
    if round_mode == "TRUNC":
        return pto.VcvtRoundMode.Z
    if round_mode == "ODD":
        return pto.VcvtRoundMode.O
    return pto.VcvtRoundMode.R


def _sat_mode(token):
    if token == "nosat":
        return pto.VcvtSatMode.NOSAT
    if token == "sat":
        return pto.VcvtSatMode.SAT
    return None


def _part_mode(token):
    if token == "even":
        return pto.VcvtPartMode.EVEN
    if token == "p0":
        return pto.VcvtPartMode.P0
    return None


def _render_tcvt(
    src,
    dst,
    *,
    rnd=False,
    sat=None,
    part=None,
    load_dist=None,
    store_dist=None,
    mask_dtype="dst",
    convert_mask="store",
):
    valid_rows, valid_cols = dst.valid_shape
    dtype = dst.dtype
    loop_dtype = src.dtype if mask_dtype == "src" else dtype
    lanes = pto.elements_per_vreg(loop_dtype)
    with pto.for_(0, valid_rows, step=1) as row:
        remained = valid_cols
        col_loop = pto.for_(0, valid_cols, step=lanes).carry(remained=remained)
        with col_loop:
            col = col_loop.iv
            mask, remained = pto.make_mask(loop_dtype, remained)
            convert_mask_value = mask
            if convert_mask == "src_full":
                convert_mask_value = pto.make_mask(src.dtype, pto.PAT.ALL)
            vec = pto.vlds(src[row, col:], dist=load_dist) if load_dist else pto.vlds(src[row, col:])
            kwargs = {}
            if rnd:
                kwargs["rnd"] = _round_mode()
            sat_mode = _sat_mode(sat)
            if sat_mode is not None:
                kwargs["sat"] = sat_mode
            part_mode = _part_mode(part)
            if part_mode is not None:
                kwargs["part"] = part_mode
            converted = pto.vcvt(vec, dtype, convert_mask_value, **kwargs)
            if store_dist:
                pto.vsts(converted, dst[row, col:], mask, dist=store_dist)
            else:
                pto.vsts(converted, dst[row, col:], mask)
            col_loop.update(remained=remained)


def _register_tcvt(
    *,
    name,
    dtypes,
    idx,
    rnd,
    sat=None,
    part=None,
    load_dist=None,
    store_dist=None,
    mask_dtype="dst",
    convert_mask="store",
):
    @tilelib.tile_template(
        op="pto.tcvt",
        target="a5",
        name=name,
        dtypes=[dtypes],
        constraints=[_rowwise],
        id=idx,
        loop_depth=2,
        is_post_update=False,
        tags=("convert", "rowwise"),
    )
    def template(src: pto.Tile, dst: pto.Tile):
        _render_tcvt(
            src,
            dst,
            rnd=rnd,
            sat=sat,
            part=part,
            load_dist=load_dist,
            store_dist=store_dist,
            mask_dtype=mask_dtype,
            convert_mask=convert_mask,
        )

    return template


template_tcvt_f32_to_i32 = _register_tcvt(
    name="template_tcvt_f32_to_i32",
    dtypes=("f32", "i32"),
    idx=0,
    rnd=True,
    sat="sat",
)

template_tcvt_i32_to_f32 = _register_tcvt(
    name="template_tcvt_i32_to_f32",
    dtypes=("i32", "f32"),
    idx=1,
    rnd=True,
)

template_tcvt_i16_to_f16 = _register_tcvt(
    name="template_tcvt_i16_to_f16",
    dtypes=("i16", "f16"),
    idx=2,
    rnd=True,
)

template_tcvt_f16_to_i16 = _register_tcvt(
    name="template_tcvt_f16_to_i16",
    dtypes=("f16", "i16"),
    idx=3,
    rnd=True,
    sat="sat",
)

template_tcvt_bf16_to_f16 = _register_tcvt(
    name="template_tcvt_bf16_to_f16",
    dtypes=("bf16", "f16"),
    idx=4,
    rnd=True,
    sat="sat",
)

template_tcvt_f32_to_f16 = _register_tcvt(
    name="template_tcvt_f32_to_f16",
    dtypes=("f32", "f16"),
    idx=5,
    rnd=True,
    sat="sat",
    part="even",
    store_dist=pto.VStoreDist.PK_B32,
    mask_dtype="src",
)

template_tcvt_f32_to_bf16 = _register_tcvt(
    name="template_tcvt_f32_to_bf16",
    dtypes=("f32", "bf16"),
    idx=6,
    rnd=True,
    sat="sat",
    part="even",
    store_dist=pto.VStoreDist.PK_B32,
    mask_dtype="src",
)

template_tcvt_f16_to_i32 = _register_tcvt(
    name="template_tcvt_f16_to_i32",
    dtypes=("f16", "i32"),
    idx=7,
    rnd=True,
    part="even",
    load_dist="UNPK_B16",
    convert_mask="src_full",
)
