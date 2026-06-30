# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Lazy loader for ported, per-architecture TileLib templates."""

from functools import lru_cache
from importlib import import_module


_TEMPLATE_MODULES = {
    ("a5", "pto.tadd"): ".a5.tadd",
    ("a5", "pto.tcolmax"): ".a5.tcolmax",
    ("a5", "pto.tdiv"): ".a5.tdiv",
    ("a5", "pto.tmax"): ".a5.tmax",
    ("a5", "pto.tmin"): ".a5.tmin",
    ("a5", "pto.tmul"): ".a5.tmul",
    ("a5", "pto.tsub"): ".a5.tsub",
}


@lru_cache(maxsize=None)
def load_template(op: str, target: str) -> bool:
    """Import and register only the template module for ``(target, op)``.

    Both this cache and Python's module cache make repeated requests no-ops.
    Returns ``False`` when this TileLib has no module for the requested pair.
    """
    module_name = _TEMPLATE_MODULES.get((target, op))
    if module_name is None:
        return False
    import_module(module_name, package=__name__)
    return True


__all__ = ["load_template"]
