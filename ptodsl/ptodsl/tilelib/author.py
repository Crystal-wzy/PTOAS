# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Author-facing TileLib body surface.

A template imports this namespace as ``pto`` and writes a tilelang-style body. Control
flow uses the **engine's** AST-rewrite surface (``for x in range(...)`` is rewritten to
``pto.for_(...).carry(...)`` at trace time — see ``_render_runtime`` / ``_ast_rewrite``),
so ``for_``/``static_range``/``if_``/``yield_`` are re-exported from ``_control_flow``
rather than reimplemented. The body ops route to the existing ``_ops`` engine.
"""

from __future__ import annotations

# Engine control-flow surface (target of the AST rewrite).
from .._control_flow import const_expr, for_, if_, static_range, vecscope, yield_
from .._surface_types import PostUpdate, Tile
from .. import _ops
from .._types import _resolve


def get_lanes(dtype) -> int:
    """Vector lanes for *dtype* (256-byte vreg). Returns a Python int used as a loop step."""
    return _ops._elements_per_vreg(_resolve(dtype))


def make_mask(dtype, value):
    """``pto.plt_b{8,16,32}`` predicate mask; returns a ``(mask, remained)``-unpackable value."""
    return _ops.make_mask(dtype, value)


def addptr(base_ptr, index_offset):
    """``pto.addptr`` – advance a pointer by an element offset."""
    return _ops.addptr(base_ptr, index_offset)


def vlds(src_ptr, offset=None, result_vreg_type=None, *, dist=None, post_update=PostUpdate.OFF):
    """``pto.vlds`` from a tile slice or pointer."""
    return _ops.vlds(
        src_ptr,
        offset,
        result_vreg_type,
        dist=dist,
        post_update=post_update,
    )


def vadd(lhs, rhs, mask):
    """``pto.vadd`` element-wise add."""
    return _ops.vadd(lhs, rhs, mask)


def vsub(lhs, rhs, mask):
    """``pto.vsub`` element-wise subtract."""
    return _ops.vsub(lhs, rhs, mask)


def vmul(lhs, rhs, mask):
    """``pto.vmul`` element-wise multiply."""
    return _ops.vmul(lhs, rhs, mask)


def vmax(lhs, rhs, mask):
    """``pto.vmax`` element-wise maximum."""
    return _ops.vmax(lhs, rhs, mask)


def vmin(lhs, rhs, mask):
    """``pto.vmin`` element-wise minimum."""
    return _ops.vmin(lhs, rhs, mask)


def vdiv(lhs, rhs, mask):
    """``pto.vdiv`` element-wise divide (default precision)."""
    return _ops.vdiv(lhs, rhs, mask)


def vsts(vec, dst_ptr, offset, mask=None, *, dist=None, post_update=PostUpdate.OFF):
    """``pto.vsts`` to a tile slice or pointer."""
    return _ops.vsts(vec, dst_ptr, offset, mask, dist=dist, post_update=post_update)


__all__ = [
    "Tile",
    "for_",
    "static_range",
    "if_",
    "yield_",
    "const_expr",
    "vecscope",
    "get_lanes",
    "make_mask",
    "addptr",
    "PostUpdate",
    "vlds",
    "vadd",
    "vsub",
    "vmul",
    "vmax",
    "vmin",
    "vdiv",
    "vsts",
]
