#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DOCKERFILE = REPO_ROOT / "docker" / "Dockerfile"
COLLECT_DIST_SCRIPT = REPO_ROOT / "docker" / "collect_ptoas_dist.sh"
COLLECT_DIST_MAC_SCRIPT = REPO_ROOT / "docker" / "collect_ptoas_dist_mac.sh"
PTOAS_CMAKE = REPO_ROOT / "tools" / "ptoas" / "CMakeLists.txt"
PTOAS_CLI_TEST = REPO_ROOT / "docker" / "test_ptoas_cli.sh"


class DockerRuntimePackagingTests(unittest.TestCase):
    def test_runtime_uses_standard_python_extension_without_native_cli(self):
        cmake = PTOAS_CMAKE.read_text(encoding="utf-8")

        self.assertIn(
            "pybind11_add_module(ptoas_runtime MODULE",
            cmake,
        )
        self.assertIn('OUTPUT_NAME "_native"', cmake)
        self.assertIn('BUILD_RPATH "${PTO_LLVM_BUILD_LIBRARY_DIR}"', cmake)
        self.assertIn('INSTALL_RPATH "${PTO_LLVM_BUILD_LIBRARY_DIR}"', cmake)
        self.assertNotIn("add_executable(ptoas_native_cli", cmake)
        self.assertNotIn('OUTPUT_NAME "ptoas-native"', cmake)

    def test_runtime_image_uses_wheel_entrypoint_instead_of_copied_wrapper(self):
        dockerfile = DOCKERFILE.read_text(encoding="utf-8")

        self.assertIn("COPY --from=builder /wheelhouse/ptoas*.whl /tmp/", dockerfile)
        self.assertIn(
            "RUN pip install --no-cache-dir /tmp/ptoas*.whl && rm /tmp/ptoas*.whl",
            dockerfile,
        )
        self.assertNotIn(
            "COPY --from=builder /llvm-workspace/PTOAS/build-release/tools/ptoas/ptoas /usr/local/bin/ptoas",
            dockerfile,
        )
        self.assertNotIn("/usr/local/lib/ptoas", dockerfile)

    def test_docker_builds_wheel_through_standard_pep517_backend(self):
        dockerfile = DOCKERFILE.read_text(encoding="utf-8")

        self.assertIn("python -m pip wheel .", dockerfile)
        self.assertIn("SKBUILD_BUILD_DIR=$PTO_BUILD_DIR", dockerfile)
        self.assertNotIn("create_wheel.sh", dockerfile)
        self.assertNotIn("_ptoas_build_backend", dockerfile)

    def test_linux_dist_packages_python_wrapper_and_native_extension(self):
        script = COLLECT_DIST_SCRIPT.read_text(encoding="utf-8")

        self.assertIn('PTOAS_BIN="${PTO_INSTALL_DIR}/bin/ptoas"', script)
        self.assertIn('PTOAS_PACKAGE_SRC_DIR="${PTO_INSTALL_DIR}/ptoas"', script)
        self.assertIn('PTOAS_NATIVE_MODULE="$(find "${PTOAS_PACKAGE_DIST_DIR}"', script)
        self.assertIn("patchelf --set-rpath '$ORIGIN/../lib' \"${PTOAS_NATIVE_MODULE}\"", script)
        self.assertIn('"${PTOAS_DIST_DIR}/bin/ptoas" --version', script)
        self.assertNotIn("ptoas-native", script)
        self.assertNotIn("ptoas.so", script)

    def test_macos_dist_packages_python_wrapper_and_native_extension(self):
        script = COLLECT_DIST_MAC_SCRIPT.read_text(encoding="utf-8")

        self.assertIn('PTOAS_BIN="${PTO_INSTALL_DIR}/bin/ptoas"', script)
        self.assertIn('PTOAS_PACKAGE_SRC_DIR="${PTO_INSTALL_DIR}/ptoas"', script)
        self.assertIn('PTOAS_NATIVE_MODULE="$(find "${PTOAS_PACKAGE_DIST_DIR}"', script)
        self.assertIn('collect_dylibs "${PTOAS_NATIVE_MODULE}"', script)
        self.assertIn('"${PTOAS_DIST_DIR}/bin/ptoas" --version', script)
        self.assertNotIn("ptoas-native", script)
        self.assertNotIn("ptoas.so", script)

    def test_wheel_import_smoke_imports_native_extension(self):
        script = (REPO_ROOT / "docker" / "test_wheel_imports.sh").read_text(encoding="utf-8")

        self.assertIn("from ptoas import _native", script)
        self.assertNotIn("POLLUTED_PTOAS_VERSION_OUTPUT", script)

    def test_build_tree_cli_test_receives_explicit_test_environment(self):
        script = PTOAS_CLI_TEST.read_text(encoding="utf-8")

        self.assertIn("for var in PTO_SOURCE_DIR PTOAS_BIN", script)
        self.assertIn("python ./tmatmulk.py", script)
        self.assertIn("python ./abs.py", script)
        self.assertNotIn("PYTHON_BIN", script)
        self.assertNotIn('command -v ptoas', script)
        self.assertNotIn('export PATH="${PTO_BUILD_DIR}', script)
        self.assertNotIn('export LD_LIBRARY_PATH="${LLVM_BUILD_DIR}', script)

if __name__ == "__main__":
    unittest.main()
