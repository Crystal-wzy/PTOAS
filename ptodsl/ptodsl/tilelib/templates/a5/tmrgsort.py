# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""PTODSL TileLib templates for ``pto.tmrgsort``."""

from ptodsl import pto
import ptodsl.tilelib as tilelib


STRUCT_SIZE = 8
STRUCT_SIZE_SHIFT = 3
BLOCK_NUM = 4


def _structures(valid_cols, dtype):
    if pto.bytewidth(dtype) == 4:
        return valid_cols // 2
    return valid_cols // 4


def _copy_tmp_to_dst(tmp, dst):
    dtype = dst.dtype
    valid_cols = dst.shape[1]
    lanes = pto.elements_per_vreg(dtype)
    for col in range(0, valid_cols, lanes):
        remained = valid_cols - col
        mask, _ = pto.make_mask(dtype, remained)
        data = pto.vlds(tmp[0, col:])
        pto.vsts(data, dst[0, col:], mask)


@tilelib.tile_template(
    op="pto.tmrgsort",
    target="a5",
    name="template_tmrgsort_single_list",
    dtypes=[("f32", "i32", "f32"), ("f16", "i32", "f16")],
    memory_spaces=("ub",),
    layouts=("row_major",),
    id=0,
    loop_depth=0,
    is_post_update=False,
    tags=("sort", "merge", "single-list"),
)
def template_tmrgsort_single_list(src: pto.Tile, block_len: pto.i32, dst: pto.Tile):
    num_structures = block_len * pto.bytewidth(src.dtype) >> STRUCT_SIZE_SHIFT
    repeat_times = src.shape[1] // (block_len * BLOCK_NUM)
    count = num_structures | (num_structures << 16) | (num_structures << 32) | (num_structures << 48)
    offset = num_structures * STRUCT_SIZE // pto.bytewidth(dst.dtype)
    config = repeat_times | (0b1111 << 8)
    src_ptr = src.as_ptr()
    pto.vmrgsort4(
        dst.as_ptr(),
        src_ptr,
        pto.addptr(src_ptr, offset),
        pto.addptr(src_ptr, offset * 2),
        pto.addptr(src_ptr, offset * 3),
        count,
        config,
    )


@tilelib.tile_template(
    op="pto.tmrgsort",
    target="a5",
    name="template_tmrgsort_multi_list2",
    dtypes=[("f32", "f32", "f32", "f32", "i32"), ("f16", "f16", "f16", "f16", "i32")],
    memory_spaces=("ub",),
    layouts=("row_major",),
    id=1,
    loop_depth=1,
    is_post_update=False,
    tags=("sort", "merge", "multi-list"),
)
def template_tmrgsort_multi_list2(src0: pto.Tile, src1: pto.Tile, tmp: pto.Tile, dst: pto.Tile, ex_vec: pto.i32):
    _ = ex_vec
    src0_structures = _structures(src0.shape[1], dst.dtype)
    src1_structures = _structures(src1.shape[1], dst.dtype)
    count = src0_structures | (src1_structures << 16)
    exhausted = int(pto.get_op_attr("exhausted", "0"))
    config = 1 | (0b0011 << 8) | (exhausted << 12)
    pto.vmrgsort4(tmp.as_ptr(), src0.as_ptr(), src1.as_ptr(), src0.as_ptr(), src0.as_ptr(), count, config)
    _copy_tmp_to_dst(tmp, dst)


@tilelib.tile_template(
    op="pto.tmrgsort",
    target="a5",
    name="template_tmrgsort_multi_list3",
    dtypes=[("f32", "f32", "f32", "f32", "f32", "i32"), ("f16", "f16", "f16", "f16", "f16", "i32")],
    memory_spaces=("ub",),
    layouts=("row_major",),
    id=2,
    loop_depth=1,
    is_post_update=False,
    tags=("sort", "merge", "multi-list"),
)
def template_tmrgsort_multi_list3(
    src0: pto.Tile,
    src1: pto.Tile,
    src2: pto.Tile,
    tmp: pto.Tile,
    dst: pto.Tile,
    ex_vec: pto.i32,
):
    _ = ex_vec
    src0_structures = _structures(src0.shape[1], dst.dtype)
    src1_structures = _structures(src1.shape[1], dst.dtype)
    src2_structures = _structures(src2.shape[1], dst.dtype)
    count = src0_structures | (src1_structures << 16) | (src2_structures << 32)
    exhausted = int(pto.get_op_attr("exhausted", "0"))
    config = 1 | (0b0111 << 8) | (exhausted << 12)
    pto.vmrgsort4(tmp.as_ptr(), src0.as_ptr(), src1.as_ptr(), src2.as_ptr(), src0.as_ptr(), count, config)
    _copy_tmp_to_dst(tmp, dst)


@tilelib.tile_template(
    op="pto.tmrgsort",
    target="a5",
    name="template_tmrgsort_multi_list4",
    dtypes=[
        ("f32", "f32", "f32", "f32", "f32", "f32", "i32"),
        ("f16", "f16", "f16", "f16", "f16", "f16", "i32"),
    ],
    memory_spaces=("ub",),
    layouts=("row_major",),
    id=3,
    loop_depth=1,
    is_post_update=False,
    tags=("sort", "merge", "multi-list"),
)
def template_tmrgsort_multi_list4(
    src0: pto.Tile,
    src1: pto.Tile,
    src2: pto.Tile,
    src3: pto.Tile,
    tmp: pto.Tile,
    dst: pto.Tile,
    ex_vec: pto.i32,
):
    _ = ex_vec
    src0_structures = _structures(src0.shape[1], dst.dtype)
    src1_structures = _structures(src1.shape[1], dst.dtype)
    src2_structures = _structures(src2.shape[1], dst.dtype)
    src3_structures = _structures(src3.shape[1], dst.dtype)
    count = (
        src0_structures
        | (src1_structures << 16)
        | (src2_structures << 32)
        | (src3_structures << 48)
    )
    exhausted = int(pto.get_op_attr("exhausted", "0"))
    config = 1 | (0b1111 << 8) | (exhausted << 12)
    pto.vmrgsort4(tmp.as_ptr(), src0.as_ptr(), src1.as_ptr(), src2.as_ptr(), src3.as_ptr(), count, config)
    _copy_tmp_to_dst(tmp, dst)
