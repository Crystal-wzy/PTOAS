# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""PTODSL TileLib template for pto.tsub (ported from lib/TileOps/tsub_template.py)."""

import ptodsl.tilelib as pto


@pto.tile_template(
    op="pto.tsub",
    target="a5",
    name="template_tsub",
    dtypes=[("f32", "f32", "f32")],
    layouts=["row_major"],
    memory_spaces=["ub"],
    priority=0,
)
def template_tsub(src0: pto.Tile, src1: pto.Tile, dst: pto.Tile):
    dtype = dst.element_type
    valid_rows, valid_cols = dst.valid_shape

    for row in range(0, valid_rows, 1):
        remained = valid_cols
        for col in range(0, valid_cols, pto.get_lanes(dtype)):
            mask, remained = pto.make_mask(dtype, remained)
            lhs = pto.vlds(src0[row, col:])
            rhs = pto.vlds(src1[row, col:])
            subtracted = pto.vsub(lhs, rhs, mask)
            pto.vsts(subtracted, dst[row, col:], mask)
