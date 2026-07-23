#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import subprocess
import tempfile
import unittest
import zipfile
import importlib.machinery
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
VALIDATOR = REPO_ROOT / "docker" / "validate_wheel_payload.py"
WHEEL_IMPORTS = REPO_ROOT / "docker" / "test_wheel_imports.sh"


class ValidateWheelPayloadTests(unittest.TestCase):
    def _make_wheel(
        self,
        root: Path,
        *,
        include_native_module: bool,
        include_cli: bool = True,
        include_tileops: bool = True,
        entry_points_text: str = "[console_scripts]\nptoas=ptoas._cli:main\n",
        wheel_stem: str = "ptoas",
        dist_info_stem: str = "ptoas",
    ) -> Path:
        wheel = root / f"{wheel_stem}-1.2.3-cp311-cp311-linux_x86_64.whl"
        with zipfile.ZipFile(wheel, "w") as zf:
            zf.writestr("ptoas/__init__.py", "")
            if include_cli:
                zf.writestr("ptoas/_cli.py", "")
            if include_tileops:
                zf.writestr(
                    "ptoas/_runtime/share/ptoas/TileOps/__init__.py", ""
                )
            if include_native_module:
                suffix = importlib.machinery.EXTENSION_SUFFIXES[0]
                zf.writestr(f"ptoas/_native{suffix}", "fake")
            zf.writestr(
                f"{dist_info_stem}-1.2.3.dist-info/entry_points.txt",
                entry_points_text,
            )
        return wheel

    def test_validator_accepts_current_runtime_payload_layout(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            wheel = self._make_wheel(Path(temp_dir), include_native_module=True)
            result = subprocess.run(
                ["python3", str(VALIDATOR), str(wheel)],
                capture_output=True,
                text=True,
                check=False,
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("validated wheel payload and launcher contract", result.stdout)

    def test_validator_rejects_missing_native_module(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            wheel = self._make_wheel(Path(temp_dir), include_native_module=False)
            result = subprocess.run(
                ["python3", str(VALIDATOR), str(wheel)],
                capture_output=True,
                text=True,
                check=False,
            )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("ptoas._native", result.stderr)

    def test_validator_rejects_missing_cli_module(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            wheel = self._make_wheel(
                Path(temp_dir),
                include_native_module=True,
                include_cli=False,
            )
            result = subprocess.run(
                ["python3", str(VALIDATOR), str(wheel)],
                capture_output=True,
                text=True,
                check=False,
            )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("ptoas/_cli.py", result.stderr)

    def test_validator_rejects_missing_packaged_tileops(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            wheel = self._make_wheel(
                Path(temp_dir),
                include_native_module=True,
                include_tileops=False,
            )
            result = subprocess.run(
                ["python3", str(VALIDATOR), str(wheel)],
                capture_output=True,
                text=True,
                check=False,
            )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("TileOps/__init__.py", result.stderr)

    def test_validator_rejects_legacy_bootstrap_entrypoint(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            wheel = self._make_wheel(
                Path(temp_dir),
                include_native_module=True,
                entry_points_text="[console_scripts]\nptoas=ptoas_wheel_bootstrap:main\n",
            )
            result = subprocess.run(
                ["python3", str(VALIDATOR), str(wheel)],
                capture_output=True,
                text=True,
                check=False,
            )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("ptoas._cli:main", result.stderr)

    def test_validator_accepts_normalized_entrypoint_spacing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            wheel = self._make_wheel(
                Path(temp_dir),
                include_native_module=True,
                entry_points_text="[console_scripts]\nptoas = ptoas._cli:main\n",
            )
            result = subprocess.run(
                ["python3", str(VALIDATOR), str(wheel)],
                capture_output=True,
                text=True,
                check=False,
            )

        self.assertEqual(result.returncode, 0, result.stderr)

    def test_validator_accepts_vmi_distribution_name(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            wheel = self._make_wheel(
                Path(temp_dir),
                include_native_module=True,
                wheel_stem="ptoas_vmi",
                dist_info_stem="ptoas_vmi",
            )
            result = subprocess.run(
                ["python3", str(VALIDATOR), str(wheel)],
                capture_output=True,
                text=True,
                check=False,
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("validated wheel payload and launcher contract", result.stdout)

    def test_shell_probe_reuses_shared_validator(self):
        self.assertIn(
            '"${PYTHON_BIN}" "${REPO_ROOT}/docker/validate_wheel_payload.py" "${TEST_WHEEL}"',
            WHEEL_IMPORTS.read_text(encoding="utf-8"),
        )

    def test_wheel_imports_script_keeps_clean_env_ptoas_smoke(self):
        script = WHEEL_IMPORTS.read_text(encoding="utf-8")

        self.assertIn('EXPECTED_PTOAS_CLI_VERSION="${PTOAS_CLI_VERSION:-${PTOAS_VERSION:-}}"', script)
        self.assertIn('env -i \\', script)
        self.assertIn('CLEAN_ENV_PTO="${CLEAN_ENV_PTO}" \\', script)
        self.assertIn('CLEAN_ENV_LOG="${CLEAN_ENV_DIR}/wheel-clean-env-probe.log"', script)
        self.assertIn('CLEAN_ENV_PTO_IR="${CLEAN_ENV_DIR}/wheel-clean-env-probe.pto.ir"', script)
        self.assertIn('def wheel_clean_env_probe():', script)
        self.assertIn('pto.alloc_tile(shape=[1, 16], dtype=pto.f32, addr=0)', script)
        self.assertIn('pto.castptr(pto.const(0, dtype=pto.ui64), pto.ptr(pto.f32, "gm"))', script)
        self.assertIn('pto.tile.load(a_view, a_tile)', script)
        self.assertIn('pto.tile.store(o_tile, o_view)', script)
        self.assertIn('--emit-pto-ir "${CLEAN_ENV_PTO}" -o "${CLEAN_ENV_PTO_IR}"', script)
        self.assertIn('def wheel_clean_env_probe(', script)
        self.assertIn('"${PTOAS_ENTRYPOINT}" --pto-arch=a5 --pto-backend=vpto --pto-level=level3 --enable-tile-op-expand --emit-pto-ir "${CLEAN_ENV_PTO}" -o "${CLEAN_ENV_PTO_IR}"', script)
        self.assertIn('>"${CLEAN_ENV_LOG}" 2>&1', script)
        self.assertIn('grep -q "wheel_clean_env_probe" "${CLEAN_ENV_PTO_IR}"', script)
        self.assertIn('grep -q "pto.tload" "${CLEAN_ENV_PTO_IR}"', script)
        self.assertIn('grep -q "pto.tstore" "${CLEAN_ENV_PTO_IR}"', script)
        self.assertIn('grep -q "candidates = " "${CLEAN_ENV_PTO_IR}"', script)
        self.assertIn('grep -q "TileLib daemon started successfully" "${CLEAN_ENV_LOG}"', script)
        self.assertIn('grep -q "TileLib daemon stopped" "${CLEAN_ENV_LOG}"', script)

if __name__ == "__main__":
    unittest.main()
