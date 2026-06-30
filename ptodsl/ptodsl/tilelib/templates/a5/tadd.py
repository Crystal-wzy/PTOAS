# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""PTODSL TileLib templates for pto.tadd."""

import ptodsl.tilelib as pto

from . import tbinop


class AddOp:
    @staticmethod
    def BinInstr(reg_src0, reg_src1, preg):
        return pto.vadd(reg_src0, reg_src1, preg)


def TAdd(dst: pto.Tile, src0: pto.Tile, src1: pto.Tile, version):
    tbinop.BinaryInstr(dst, src0, src1, AddOp, version)


@pto.tile_template(
    op="pto.tadd",
    target="a5",
    name="template_tadd_2d_no_post_update",
    id=0,
    constraints=[
        pto.check_type(("f32", "f32", "f32")),
        pto.check_memory_space("ub"),
        pto.check_layout("row_major"),
        pto.require_contiguous(False),
    ],
    priority=0,
    loop_depth=2,
    Tail=tbinop.has_tail,
    is_post_update=False,
    tags=["binop", "2d", "no_post_update"],
)
def template_tadd_2d_no_post_update(src0: pto.Tile, src1: pto.Tile, dst: pto.Tile):
    TAdd(dst, src0, src1, tbinop.VFIMPL_2D_NO_POST_UPDATE)


@pto.tile_template(
    op="pto.tadd",
    target="a5",
    name="template_tadd_1d_no_post_update",
    id=1,
    constraints=[
        pto.check_type(("f32", "f32", "f32")),
        pto.check_memory_space("ub"),
        pto.check_layout("row_major"),
        pto.require_contiguous(True),
    ],
    priority=0,
    loop_depth=1,
    Tail=tbinop.has_tail,
    is_post_update=False,
    tags=["binop", "1d", "no_post_update"],
)
def template_tadd_1d_no_post_update(src0: pto.Tile, src1: pto.Tile, dst: pto.Tile):
    TAdd(dst, src0, src1, tbinop.VFIMPL_1D_NO_POST_UPDATE)


@pto.tile_template(
    op="pto.tadd",
    target="a5",
    name="template_tadd_2d_post_update",
    id=2,
    constraints=[
        pto.check_type(("f32", "f32", "f32")),
        pto.check_memory_space("ub"),
        pto.check_layout("row_major"),
        pto.require_contiguous(False),
    ],
    priority=0,
    loop_depth=2,
    Tail=tbinop.has_tail,
    is_post_update=True,
    tags=["binop", "2d", "post_update"],
)
def template_tadd_2d_post_update(src0: pto.Tile, src1: pto.Tile, dst: pto.Tile):
    TAdd(dst, src0, src1, tbinop.VFIMPL_2D_POST_UPDATE)


@pto.tile_template(
    op="pto.tadd",
    target="a5",
    name="template_tadd_1d_post_update",
    id=3,
    constraints=[
        pto.check_type(("f32", "f32", "f32")),
        pto.check_memory_space("ub"),
        pto.check_layout("row_major"),
        pto.require_contiguous(True),
    ],
    priority=0,
    loop_depth=1,
    Tail=tbinop.has_tail,
    is_post_update=True,
    tags=["binop", "1d", "post_update"],
)
def template_tadd_1d_post_update(src0: pto.Tile, src1: pto.Tile, dst: pto.Tile):
    TAdd(dst, src0, src1, tbinop.VFIMPL_1D_POST_UPDATE)


# Compatibility alias for tests and examples that import the original tadd template.
template_tadd = template_tadd_2d_no_post_update
