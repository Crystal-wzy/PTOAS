# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Constraint predicates + evaluation for TileLib version selection.

A template may declare ``constraints=[predicate, ...]`` (legality rules, e.g. "row-major
layout and a 1-row output"). During selection we build a per-operand context and call each
predicate by **name-matching its parameters** against that context — the same introspection
convention as tilelang-dsl's `_evaluate_constraints`, so predicates port verbatim. The
predicate receives keys like ``src_shape`` / ``dst_valid_shape`` / ``src_config`` and returns
a truthy value when legal.

``BLayout`` / ``SLayout`` mirror tilelang's enums so a copied predicate's
``cfg.b_layout != pto.BLayout.ROW_MAJOR`` comparison works unchanged (str enums compare equal
to the raw layout strings carried in operand specs).
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from enum import Enum


class BLayout(str, Enum):
    ROW_MAJOR = "row_major"
    COL_MAJOR = "col_major"


class SLayout(str, Enum):
    NONE_BOX = "none_box"
    ROW_MAJOR = "row_major"
    COL_MAJOR = "col_major"


@dataclass(frozen=True)
class _ConfigView:
    """The ``{name}_config`` object a constraint sees (``.b_layout`` / ``.s_layout`` strings,
    which compare equal to the BLayout/SLayout str-enums)."""

    b_layout: str
    s_layout: str


def build_context(tile_specs: dict, target: str, op: str) -> dict:
    """Build the flat name-keyed context predicates are matched against."""
    context: dict = {"target": target, "op": op}
    operand_dtypes = []
    operand_memory_spaces = []
    operand_rows = []
    operand_cols = []
    operand_sizes = []
    operand_valid_cols = []
    operand_b_layouts = []
    for name, spec in tile_specs.items():
        shape = tuple(spec.shape)
        valid = tuple(spec.valid_shape) if getattr(spec, "valid_shape", None) else shape
        dtype = spec.dtype.name
        memory_space = getattr(spec, "memory_space", "ub")
        b_layout = getattr(spec, "b_layout", "row_major")
        s_layout = getattr(spec, "s_layout", "none_box")
        operand_dtypes.append(dtype)
        operand_memory_spaces.append(memory_space)
        operand_sizes.append(_shape_size(shape))
        operand_b_layouts.append(b_layout)
        context[f"{name}_shape"] = shape
        context[f"{name}_valid_shape"] = valid
        context[f"{name}_dtype"] = dtype
        context[f"{name}_memory_space"] = memory_space
        context[f"{name}_config"] = _ConfigView(
            b_layout=b_layout,
            s_layout=s_layout,
        )
        if len(shape) == 2:
            context[f"{name}_rows"], context[f"{name}_cols"] = shape
            context[f"{name}_valid_rows"], context[f"{name}_valid_cols"] = valid
            operand_rows.append(shape[0])
            operand_cols.append(shape[1])
            operand_valid_cols.append(valid[1])
    context["operand_dtypes"] = tuple(operand_dtypes)
    context["operand_memory_spaces"] = tuple(operand_memory_spaces)
    context["operand_rows"] = tuple(operand_rows)
    context["operand_cols"] = tuple(operand_cols)
    context["operand_sizes"] = tuple(operand_sizes)
    context["operand_valid_cols"] = tuple(operand_valid_cols)
    context["operand_b_layouts"] = tuple(operand_b_layouts)
    return context


def _shape_size(shape):
    size = 1
    for dim in shape:
        size *= dim
    return size


def check_type(expected):
    expected = tuple(expected)

    def _check_type(operand_dtypes, **_):
        return tuple(operand_dtypes) == expected

    return _check_type


def check_memory_space(expected):
    def _check_memory_space(operand_memory_spaces, **_):
        return all(space == expected for space in operand_memory_spaces)

    return _check_memory_space


def check_layout(expected):
    def _check_layout(operand_b_layouts, **_):
        return all(layout == expected for layout in operand_b_layouts)

    return _check_layout


def require_contiguous(required=True):
    def _require_contiguous(operand_rows, operand_cols, operand_valid_cols, **_):
        if not required:
            return True
        full_cols = all(valid == cols for valid, cols in zip(operand_valid_cols, operand_cols))
        single_row = all(rows == 1 for rows in operand_rows)
        return full_cols or single_row

    return _require_contiguous


def passes(predicates, context: dict) -> bool:
    """Return True iff every predicate is satisfied for *context* (legality filter)."""
    for predicate in predicates:
        try:
            signature = inspect.signature(predicate)
        except (TypeError, ValueError):
            return False
        kwargs: dict = {}
        for parameter in signature.parameters.values():
            if parameter.kind == inspect.Parameter.VAR_KEYWORD:
                for key, value in context.items():
                    kwargs.setdefault(key, value)
                continue
            if parameter.kind == inspect.Parameter.VAR_POSITIONAL:
                continue
            if parameter.name in context:
                kwargs[parameter.name] = context[parameter.name]
            elif parameter.default is not inspect.Parameter.empty:
                continue
            else:
                # A required parameter we can't supply -> treat as not satisfiable.
                return False
        try:
            if not predicate(**kwargs):
                return False
        except Exception:
            return False
    return True


__all__ = [
    "BLayout",
    "SLayout",
    "build_context",
    "check_layout",
    "check_memory_space",
    "check_type",
    "passes",
    "require_contiguous",
]
