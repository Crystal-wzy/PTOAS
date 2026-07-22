#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# For quick development, build and install ptoas and its python bindings 
# on top of Docker image https://github.com/learning-chip/agent_docker_npu/pull/8
# assume MLIR is already installed to save time
#
# Optional env:
#   LLVM_BUILD_DIR   - default: ${LLVM_SOURCE_DIR:-/llvm-workspace/llvm-project}/build-shared
#   PTO_INSTALL_DIR  - default: <repo>/install

set -euo pipefail

PTO_SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PTO_INSTALL_DIR="${PTO_INSTALL_DIR:-${PTO_SOURCE_DIR}/install}"

LLVM_SOURCE_DIR="${LLVM_SOURCE_DIR:-/llvm-workspace/llvm-project}"
LLVM_BUILD_DIR="${LLVM_BUILD_DIR:-${LLVM_SOURCE_DIR}/build-shared}"

for d in "$LLVM_BUILD_DIR/lib/cmake/llvm" "$LLVM_BUILD_DIR/lib/cmake/mlir"; do
  test -d "$d" || { echo "error: missing $d (set LLVM_BUILD_DIR?)" >&2; exit 1; }
done

MLIR_PY_PKG="${LLVM_BUILD_DIR}/tools/mlir/python_packages/mlir_core"
test -d "$MLIR_PY_PKG" || { echo "error: MLIR python package dir missing: $MLIR_PY_PKG" >&2; exit 1; }

PTOAS_VERSION="${PTOAS_VERSION:-$(python "${PTO_SOURCE_DIR}/.github/scripts/compute_ptoas_version.py" --cmake-file "${PTO_SOURCE_DIR}/CMakeLists.txt" --mode dev)}"

cd "$PTO_SOURCE_DIR"

PTO_WHEEL_DIR="${PTO_SOURCE_DIR}/build/wheel-dist"
rm -rf "${PTO_WHEEL_DIR}"
mkdir -p "${PTO_WHEEL_DIR}"
LLVM_BUILD_DIR="${LLVM_BUILD_DIR}" \
PTOAS_RELEASE_VERSION_OVERRIDE="${PTOAS_VERSION}" \
SKBUILD_BUILD_DIR="${PTO_SOURCE_DIR}/build" \
  python -m pip wheel . --no-deps --wheel-dir "${PTO_WHEEL_DIR}"

cmake --install "${PTO_SOURCE_DIR}/build" --prefix "${PTO_INSTALL_DIR}"

shopt -s nullglob
wheels=("${PTO_WHEEL_DIR}/ptoas-"*.whl)
shopt -u nullglob
((${#wheels[@]} > 0)) || { echo "error: no ptoas-*.whl under ${PTO_WHEEL_DIR}" >&2; exit 1; }
pip install --force-reinstall "${wheels[0]}"

export PATH="${PTO_SOURCE_DIR}/build/tools/ptoas:${PATH}"
export LD_LIBRARY_PATH="${LLVM_BUILD_DIR}/lib:${PTO_INSTALL_DIR}/lib:${LD_LIBRARY_PATH:-}"

python -c "import mlir.ir"
python -c "from mlir.dialects import pto"

which ptoas

PTOAS_ENV_TMP="${PTO_SOURCE_DIR}/tmp/set_ptoas_env"
mkdir -p "${PTOAS_ENV_TMP}/MatMul" "${PTOAS_ENV_TMP}/Abs"
(cd "${PTO_SOURCE_DIR}/test/samples/MatMul" && python ./tmatmulk.py > "${PTOAS_ENV_TMP}/MatMul/tmatmulk.pto" && ptoas "${PTOAS_ENV_TMP}/MatMul/tmatmulk.pto" -o "${PTOAS_ENV_TMP}/MatMul/tmatmulk.cpp")
(cd "${PTO_SOURCE_DIR}/test/samples/Abs" && python ./abs.py > "${PTOAS_ENV_TMP}/Abs/abs.pto" && ptoas --enable-insert-sync "${PTOAS_ENV_TMP}/Abs/abs.pto" -o "${PTOAS_ENV_TMP}/Abs/abs.cpp")

echo "quick_install.sh: OK"
