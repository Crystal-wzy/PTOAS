# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""PTODSL TileLib template for the UB row-major ``pto.tinsert`` path."""

from ptodsl import pto
import ptodsl.tilelib as tilelib

from ._common import NUMERIC_DTYPES


def _vec_to_vec_nd(src_memory_space, dst_memory_space, src_config, dst_config, src_valid_shape, src_dtype, dst_dtype, **_):
    return (
        src_memory_space == "ub"
        and dst_memory_space == "ub"
        and src_config.b_layout == "row_major"
        and src_config.s_layout == "none_box"
        and dst_config.b_layout == "row_major"
        and dst_config.s_layout == "none_box"
        and src_dtype == dst_dtype
        and src_valid_shape != (1, 1)
    )


def _vec_to_vec_nd_scalar(src_memory_space, dst_memory_space, src_config, dst_config, src_valid_shape, src_dtype, dst_dtype, **_):
    return (
        src_memory_space == "ub"
        and dst_memory_space == "ub"
        and src_config.b_layout == "row_major"
        and src_config.s_layout == "none_box"
        and dst_config.b_layout == "row_major"
        and dst_config.s_layout == "none_box"
        and src_dtype == dst_dtype
        and src_valid_shape == (1, 1)
    )


_DTYPES = [(dtype, "i32", "i32", dtype) for dtype in NUMERIC_DTYPES]


@tilelib.tile_template(
    op="pto.tinsert",
    target="a5",
    name="template_tinsert_vec_to_vec_nd_basic",
    dtypes=_DTYPES,
    constraints=[_vec_to_vec_nd],
    id=0,
    loop_depth=2,
    is_post_update=False,
    tags=("move", "insert", "ub"),
)
def template_tinsert_vec_to_vec_nd_basic(
    src: pto.Tile,
    index_row: pto.i32,
    index_col: pto.i32,
    dst: pto.Tile,
):
    dtype = dst.dtype
    valid_rows, valid_cols = src.valid_shape
    lanes = pto.elements_per_vreg(dtype)
    for row in range(0, valid_rows, 1):
        remained = valid_cols
        for col in range(0, valid_cols, lanes):
            mask, remained = pto.make_mask(dtype, remained)
            data = pto.vlds(src[row, col:])
            pto.vsts(data, dst[index_row + row, index_col + col:], mask)


@tilelib.tile_template(
    op="pto.tinsert",
    target="a5",
    name="template_tinsert_vec_to_vec_nd_scalar_basic",
    dtypes=_DTYPES,
    constraints=[_vec_to_vec_nd_scalar],
    priority=1,
    id=1,
    loop_depth=0,
    is_post_update=False,
    tags=("move", "insert", "ub", "scalar"),
)
def template_tinsert_vec_to_vec_nd_scalar_basic(
    src: pto.Tile,
    index_row: pto.i32,
    index_col: pto.i32,
    dst: pto.Tile,
):
    value = pto.load_scalar(src.as_ptr(), 0)
    pto.store_scalar(dst.as_ptr(), index_row * dst.shape[1] + index_col, value)
