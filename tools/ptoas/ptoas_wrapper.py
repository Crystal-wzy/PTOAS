#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Add a CMake tree's Python root, then run the regular PTOAS CLI module."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path


def _resolve_wrapper_path(argv0: str | None = None) -> Path:
    candidate = argv0 if argv0 is not None else sys.argv[0]
    wrapper = Path(candidate)
    if wrapper.exists():
        return wrapper.resolve()

    found = shutil.which(wrapper.name or "ptoas")
    if found:
        return Path(found).resolve()

    raise SystemExit(f"unable to locate the installed ptoas wrapper: {candidate}")


def _is_build_tree_wrapper(wrapper: Path) -> bool:
    return len(wrapper.parents) >= 2 and wrapper.parents[1].name == "tools"


def _require_python_root(python_root: Path, *, context: str) -> Path:
    helper = python_root / "ptoas" / "_cli.py"
    if helper.is_file():
        return python_root.resolve()
    raise SystemExit(
        f"unable to locate the {context} ptoas Python package root: "
        f"expected {python_root}/ptoas/_cli.py"
    )


def _bootstrap_python_path(wrapper: Path) -> Path:
    if _is_build_tree_wrapper(wrapper):
        if len(wrapper.parents) < 3:
            raise SystemExit("unable to locate the build-tree root for ptoas")
        python_root = _require_python_root(
            wrapper.parents[2] / "python", context="build-tree"
        )
    else:
        if len(wrapper.parents) < 2:
            raise SystemExit("unable to locate the install-tree root for ptoas")
        python_root = _require_python_root(
            wrapper.parents[1], context="install-tree"
        )

    python_root_text = str(python_root)
    if python_root_text not in sys.path:
        sys.path.insert(0, python_root_text)
    return python_root


def main() -> None:
    wrapper = _resolve_wrapper_path()
    _bootstrap_python_path(wrapper)
    from ptoas import _cli

    raise SystemExit(_cli.launch(sys.argv[1:], wrapper=wrapper))


if __name__ == "__main__":
    main()
