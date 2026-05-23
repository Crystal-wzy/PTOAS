#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

usage() {
  cat <<'EOF'
Run a PTODSL JIT example under `msprof op simulator`.

Usage:
  scripts/sim_dsl.sh [options] <example.py> [-- <example args...>]

Options:
  --output <dir>        Override msprof output directory.
  --soc-version <soc>   Override simulator soc version. Default: Ascend950PR_9599
  -h, --help            Show this help.

Examples:
  scripts/sim_dsl.sh ptodsl/examples/jit/tadd_launch.py
  scripts/sim_dsl.sh \
    --output "$PWD/build/msprof_res/flash_softmax" \
    ptodsl/examples/jit/flash_attention_softmax_launch.py
EOF
}

die() {
  echo "error: $*" >&2
  exit 1
}

SOC_VERSION="Ascend950PR_9599"
OUTPUT_DIR=""
EXAMPLE_PATH=""
EXAMPLE_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output)
      [[ $# -ge 2 ]] || die "--output requires a value"
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --soc-version)
      [[ $# -ge 2 ]] || die "--soc-version requires a value"
      SOC_VERSION="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      EXAMPLE_ARGS=("$@")
      break
      ;;
    -*)
      die "unknown option: $1"
      ;;
    *)
      if [[ -z "${EXAMPLE_PATH}" ]]; then
        EXAMPLE_PATH="$1"
      else
        EXAMPLE_ARGS+=("$1")
      fi
      shift
      ;;
  esac
done

[[ -n "${EXAMPLE_PATH}" ]] || die "missing <example.py>"

if [[ "${EXAMPLE_PATH}" != /* ]]; then
  EXAMPLE_PATH="${REPO_ROOT}/${EXAMPLE_PATH}"
fi
[[ -f "${EXAMPLE_PATH}" ]] || die "example script not found: ${EXAMPLE_PATH}"

if [[ -z "${ASCEND_HOME_PATH:-}" ]]; then
  die "ASCEND_HOME_PATH is not set; source CANN setenv or export it first"
fi

if [[ -z "${OUTPUT_DIR}" ]]; then
  EXAMPLE_STEM="$(basename -- "${EXAMPLE_PATH}" .py)"
  OUTPUT_DIR="${REPO_ROOT}/build/msprof_res/${EXAMPLE_STEM}"
fi

SIM_LIB_DIR="${ASCEND_HOME_PATH}/tools/simulator/${SOC_VERSION}/lib"
[[ -d "${SIM_LIB_DIR}" ]] || die "simulator library directory not found: ${SIM_LIB_DIR}"

mkdir -p "${OUTPUT_DIR}"

source "${ASCEND_HOME_PATH}/bin/setenv.bash"
source "${REPO_ROOT}/set_ptoas_env.sh"
export LD_LIBRARY_PATH="${SIM_LIB_DIR}:${LD_LIBRARY_PATH:-}"
ulimit -n 65535

# msprof rejects group/other-writable working directories, so always launch
# from a private directory and use an absolute path for the example script.
cd "${HOME}"

exec msprof op simulator \
  --soc-version="${SOC_VERSION}" \
  --output="${OUTPUT_DIR}" \
  python3 "${EXAMPLE_PATH}" "${EXAMPLE_ARGS[@]}"
