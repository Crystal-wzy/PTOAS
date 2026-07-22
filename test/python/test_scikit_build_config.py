#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import json
import subprocess
import sys
import tomllib
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PTOAS_PYPROJECT = REPO_ROOT / "pyproject.toml"
VMI_PROJECT_DIR = REPO_ROOT / "packaging" / "ptoas-vmi"
VMI_PYPROJECT = VMI_PROJECT_DIR / "pyproject.toml"
PTOAS_PACKAGE_INIT = REPO_ROOT / "ptodsl" / "ptoas" / "__init__.py"


def _read_pyproject(path: Path) -> dict:
    with path.open("rb") as file:
        return tomllib.load(file)


def _project_table(project_dir: Path) -> dict:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scikit_build_core.build",
            "project-table",
        ],
        cwd=project_dir,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise AssertionError(result.stderr or result.stdout)
    return json.loads(result.stdout)


class ScikitBuildConfigTests(unittest.TestCase):
    def test_root_project_uses_standard_backend_and_console_script(self):
        config = _read_pyproject(PTOAS_PYPROJECT)

        self.assertEqual(
            config["build-system"]["build-backend"],
            "scikit_build_core.build",
        )
        self.assertIn("scikit-build-core>=0.12.2,<2", config["build-system"]["requires"])
        self.assertEqual(config["project"]["name"], "ptoas")
        self.assertEqual(config["project"]["scripts"]["ptoas"], "ptoas._cli:main")
        self.assertEqual(
            config["tool"]["scikit-build"]["install"]["components"],
            ["PTOAS_Python"],
        )

    def test_static_projects_resolve_expected_distribution_metadata(self):
        ptoas = _project_table(REPO_ROOT)
        vmi = _project_table(VMI_PROJECT_DIR)

        self.assertEqual((ptoas["name"], ptoas["version"]), ("ptoas", "0.51"))
        self.assertEqual((vmi["name"], vmi["version"]), ("ptoas-vmi", "0.1.3"))
        self.assertEqual(ptoas["scripts"]["ptoas"], "ptoas._cli:main")
        self.assertEqual(vmi["scripts"]["ptoas"], "ptoas._cli:main")

    def test_projects_share_python_package_sources(self):
        ptoas = _read_pyproject(PTOAS_PYPROJECT)
        vmi = _read_pyproject(VMI_PYPROJECT)

        self.assertEqual(
            ptoas["tool"]["scikit-build"]["wheel"]["packages"],
            {
                "ptodsl": "ptodsl/ptodsl",
                "ptoas": "ptodsl/ptoas",
                "tilelang_dsl": "tilelang-dsl/python/tilelang_dsl",
            },
        )
        self.assertEqual(
            vmi["tool"]["scikit-build"]["cmake"]["source-dir"],
            "../..",
        )

    def test_editable_install_uses_backend_redirect_without_package_path_hacks(self):
        ptoas = _read_pyproject(PTOAS_PYPROJECT)
        package_init = PTOAS_PACKAGE_INIT.read_text(encoding="utf-8")

        self.assertEqual(
            ptoas["tool"]["scikit-build"]["editable"]["mode"],
            "redirect",
        )
        self.assertNotIn("extend_path", package_init)

    def test_cmake_python_payload_uses_dedicated_install_component(self):
        cmake_files = [
            REPO_ROOT / "CMakeLists.txt",
            REPO_ROOT / "lib" / "Bindings" / "Python" / "CMakeLists.txt",
            REPO_ROOT / "tools" / "ptoas" / "CMakeLists.txt",
        ]
        combined = "\n".join(path.read_text(encoding="utf-8") for path in cmake_files)

        self.assertIn("COMPONENT PTOAS_Python", combined)
        self.assertIn("if(SKBUILD)", combined)
        self.assertIn('DIRECTORY "${MLIR_PYTHON_PACKAGE_DIR}/mlir"', combined)

    def test_legacy_wheel_builders_are_removed(self):
        self.assertFalse((REPO_ROOT / "_ptoas_build_backend.py").exists())
        self.assertFalse((REPO_ROOT / "docker" / "create_wheel.sh").exists())
        self.assertFalse((REPO_ROOT / "docker" / "setup.py").exists())
        self.assertFalse((REPO_ROOT / "docker" / "setup_mac.py").exists())


if __name__ == "__main__":
    unittest.main()
