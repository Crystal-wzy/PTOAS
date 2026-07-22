// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

# PTOAS Python Launcher Layout Design

## Purpose

This document records the internal launcher contract for the Python-backed
`ptoas` command. The user-facing README should describe how to install and run
PTOAS without exposing these implementation details.

## Entry Model

The wheel, build-tree, and install-tree launchers use the standard Python
console-script and native-extension model:

```text
console script or CMake tree wrapper
  -> ptoas._cli.main()
       -> import ptoas._native
       -> ptoas._native.main(argv)
```

`ptoas._native` is built with `pybind11_add_module`. CMake and Python therefore
own the platform and ABI-specific filename. Launcher and packaging code refer to
the import name and never construct `.so`, `.dylib`, or `.pyd` paths.

The LLVM-based driver implementation is compiled as an object library so it
can use LLVM's RTTI and exception settings independently from pybind11. Those
objects are linked into the Python extension; no separate native executable is
produced. Wheel and standalone-archive entrypoints therefore use the same
extension and CLI module.

## Wheel Layout

Wheel console scripts point directly to `ptoas._cli:main`. The CLI imports the
native extension through Python's normal package machinery and resolves TileOps
package data relative to the installed extension. It does not re-execute itself,
load modules by file path, rewrite `PYTHONPATH`, or override Python's standard
package precedence rules.

Editable installs use scikit-build-core's redirect mode. The backend maps the
Python sources to the checkout and the CMake-installed native extension to the
editable build output without package-local path manipulation.

Auditwheel and delocate discover the native extension through the standard
wheel binary scan, bundle its dependencies, and rewrite its runtime search
paths. The launcher does not preload or enumerate those libraries.

Wheel and editable builds use `scikit-build-core` directly as the PEP 517/660
backend. Project metadata and the console entry point live in `pyproject.toml`;
wheel tags, metadata, RECORD generation, CMake configure/build/install, and
editable redirects are owned by the backend rather than repository scripts.

The main `pyproject.toml` has the static distribution name `ptoas`.
`packaging/ptoas-vmi/pyproject.toml` is a second static PEP 621 project for the
mutually exclusive `ptoas-vmi` distribution. Both projects build the same CMake
source and install the same `ptoas` import package. This keeps project names
standards-compliant because PEP 621 does not permit a dynamic `project.name`.

CMake's `PTOAS_Python` install component contains only the generated/native
wheel payload. Python source packages are declared through
`tool.scikit-build.wheel.packages`, which also lets editable installs redirect
imports to the source tree without namespace-package or custom `.pth` logic.

## Build-Tree Layout

The build-tree wrapper resolves only its own generated outputs:

- wrapper: `<build>/tools/ptoas/ptoas`
- Python root: `<build>/python`
- native module: importable as `ptoas._native` from the Python root
- TileOps: `<build>/python/ptoas/_runtime/share/ptoas/TileOps`

Missing Python packages or TileOps resources are hard layout errors. The
wrapper only adds `<build>/python` to `sys.path`; `ptoas._cli` owns the common
runtime-resource resolution and native invocation.

## Install-Tree Layout

The install-tree wrapper resolves only files under the same prefix:

- wrapper: `<prefix>/bin/ptoas`
- Python root: `<prefix>`
- native module: importable as `ptoas._native` from the Python root
- TileOps: `<prefix>/ptoas/_runtime/share/ptoas/TileOps`

The install-tree wrapper only adds `<prefix>` to `sys.path`, then delegates to
the same `ptoas._cli` module used by wheels.

## Standalone Archive Layout

Standalone compiler archives contain the installed Python wrapper and package:

```text
bin/ptoas
ptoas/_cli.py
ptoas/_native.<abi>.so
ptoas/_runtime/share/ptoas/TileOps
lib/<native dependencies>
tilelang_dsl/
```

The archive requires a Python interpreter compatible with the packaged
extension ABI. `bin/ptoas` adds the archive root to `sys.path`, then uses the
same `ptoas._cli -> ptoas._native` path as the install tree. Linux archives set
the extension runtime path to `$ORIGIN/../lib`; macOS archives rewrite the
extension's install names relative to its package directory.
