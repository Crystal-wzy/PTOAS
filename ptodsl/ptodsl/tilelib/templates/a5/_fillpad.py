# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Shared basic fill-pad helpers."""

from ptodsl import pto
import ptodsl.tilelib as tilelib


_DTYPES = [
    ("f16", "f16"),
    ("bf16", "bf16"),
    ("f32", "f32"),
    ("ui8", "ui8"),
    ("si8", "si8"),
    ("i8", "i8"),
    ("ui16", "ui16"),
    ("si16", "si16"),
    ("i16", "i16"),
    ("ui32", "ui32"),
    ("si32", "si32"),
    ("i32", "i32"),
]


def _row_major(src_config, dst_config, src_dtype, dst_dtype, **_):
    return (
        src_config.b_layout == "row_major"
        and dst_config.b_layout == "row_major"
        and src_config.s_layout == "none_box"
        and dst_config.s_layout == "none_box"
        and src_dtype == dst_dtype
    )


def _zero(dtype):
    name = str(dtype)
    if name == "f32":
        return pto.f32(0.0)
    if name == "f16":
        return pto.f16(0.0)
    if name == "bf16":
        return pto.bf16(0.0)
    if name in {"ui32", "si32", "i32"}:
        return pto.i32(0)
    if name in {"ui16", "si16", "i16"}:
        return pto.i16(0)
    return pto.i8(0)


def _fill_scalar(dst):
    dtype = dst.dtype
    if str(dtype) == "f32" and str(getattr(dst, "pad_value", "Null")).lower() == "zero":
        return pto.f32(-1.0)
    return _zero(dtype)


def _copy_valid(src, dst):
    dtype = dst.dtype
    valid_rows, valid_cols = src.valid_shape
    lanes = pto.elements_per_vreg(dtype)
    with pto.for_(0, valid_rows, step=1) as row:
        remained = valid_cols
        col_loop = pto.for_(0, valid_cols, step=lanes).carry(remained=remained)
        with col_loop:
            col = col_loop.iv
            mask, remained = pto.make_mask(dtype, remained)
            data = pto.vlds(src[row, col:])
            pto.vsts(data, dst[row, col:], mask)
            col_loop.update(remained=remained)


def _fill(dst, row_start, row_stop, col_start, col_stop):
    dtype = dst.dtype
    lanes = pto.elements_per_vreg(dtype)
    fill_scalar = _fill_scalar(dst)
    with pto.for_(row_start, row_stop, step=1) as row:
        remained = col_stop - col_start
        col_loop = pto.for_(col_start, col_stop, step=lanes).carry(remained=remained)
        with col_loop:
            col = col_loop.iv
            mask, remained = pto.make_mask(dtype, remained)
            vec = pto.vdup(fill_scalar, mask)
            pto.vsts(vec, dst[row, col:], mask)
            col_loop.update(remained=remained)


def register_fillpad(*, op, name, copy):
    @tilelib.tile_template(
        op=op,
        target="a5",
        name=name,
        dtypes=_DTYPES,
        constraints=[_row_major],
        id=0,
        loop_depth=2,
        is_post_update=False,
        tags=("fillpad",),
    )
    def template(src: pto.Tile, dst: pto.Tile):
        src_valid_rows, src_valid_cols = src.valid_shape
        dst_valid_rows, dst_valid_cols = dst.valid_shape
        if copy:
            _copy_valid(src, dst)
        _fill(dst, 0, src_valid_rows, src_valid_cols, dst_valid_cols)
        _fill(dst, src_valid_rows, dst_valid_rows, 0, dst_valid_cols)

    return template


__all__ = ["register_fillpad"]
