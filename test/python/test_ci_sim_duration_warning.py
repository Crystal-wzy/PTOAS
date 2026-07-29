# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest


_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO_ROOT / ".github" / "scripts" / "ci_sim_duration_warning.py"
_SPEC = importlib.util.spec_from_file_location("ci_sim_duration_warning", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
warning = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = warning
_SPEC.loader.exec_module(warning)


class FakeGitHubClient:
    def __init__(self, *, elapsed_seconds=3600, pr_state="open", current_sha="a" * 40, comments=None):
        self.elapsed_seconds = elapsed_seconds
        self.pr_state = pr_state
        self.current_sha = current_sha
        self.comments = list(comments or [])
        self.writes = []

    def get(self, path, query=None):
        if "/pulls/" in path:
            return {
                "state": self.pr_state,
                "head": {"sha": self.current_sha},
                "user": {"login": "test-author"},
            }
        if path.endswith("/jobs"):
            return {
                "jobs": [
                    {
                        "name": warning.JOB_NAME,
                        "started_at": "2026-07-28T10:00:00Z",
                        "completed_at": _timestamp_after(self.elapsed_seconds),
                        "conclusion": "success",
                    }
                ]
            }
        raise AssertionError(f"Unexpected GET {path} {query}")

    def list_all(self, path):
        self.comment_path = path
        return self.comments

    def post(self, path, payload):
        self.writes.append(("POST", path, payload))

    def patch(self, path, payload):
        self.writes.append(("PATCH", path, payload))


def _timestamp_after(seconds):
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"2026-07-28T{10 + hours:02d}:{minutes:02d}:{seconds:02d}Z"


class DurationWarningTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        context_dir = Path(self.temp_dir.name)
        (context_dir / "pr-number").write_text("42\n", encoding="utf-8")
        (context_dir / "pr-head-sha").write_text(f"{'a' * 40}\n", encoding="utf-8")
        self.config = warning.RunConfig(
            repo="test/repo",
            run_id=99,
            run_head_sha="a" * 40,
            run_url="https://example.invalid/run/99",
            soft_timeout_minutes=90,
            context_dir=context_dir,
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_within_budget_does_not_comment(self):
        client = FakeGitHubClient(elapsed_seconds=90 * 60)
        result = warning.observe_run(client, self.config)
        self.assertIn("within the 1h 30m budget", result)
        self.assertEqual(client.writes, [])

    def test_over_budget_creates_warning(self):
        client = FakeGitHubClient(elapsed_seconds=2 * 60 * 60 + 1)
        result = warning.observe_run(client, self.config)
        self.assertEqual(result, "Created slow CI warning on PR #42.")
        method, path, payload = client.writes[0]
        self.assertEqual((method, path), ("POST", "repos/test/repo/issues/42/comments"))
        self.assertIn("@test-author", payload["body"])
        self.assertIn("2h 0m 1s", payload["body"])
        self.assertIn(warning.COMMENT_MARKER, payload["body"])

    def test_over_budget_updates_existing_bot_comment(self):
        comments = [
            {"id": 7, "user": {"login": warning.BOT_LOGIN}, "body": warning.COMMENT_MARKER},
        ]
        client = FakeGitHubClient(elapsed_seconds=2 * 60 * 60, comments=comments)
        result = warning.observe_run(client, self.config)
        self.assertEqual(result, "Updated slow CI warning on PR #42.")
        self.assertEqual(client.writes[0][0:2], ("PATCH", "repos/test/repo/issues/comments/7"))

    def test_recovery_resolves_existing_warning(self):
        comments = [
            {"id": 8, "user": {"login": warning.BOT_LOGIN}, "body": warning.COMMENT_MARKER},
        ]
        client = FakeGitHubClient(elapsed_seconds=60 * 60, comments=comments)
        result = warning.observe_run(client, self.config)
        self.assertEqual(result, "Resolved slow CI warning on PR #42.")
        self.assertIn("Resolved:", client.writes[0][2]["body"])

    def test_resolved_comment_is_not_updated_again(self):
        comments = [
            {
                "id": 8,
                "user": {"login": warning.BOT_LOGIN},
                "body": f"{warning.COMMENT_MARKER}\n{warning.RESOLVED_PREFIX}",
            },
        ]
        client = FakeGitHubClient(elapsed_seconds=60 * 60, comments=comments)
        result = warning.observe_run(client, self.config)
        self.assertIn("remains within", result)
        self.assertEqual(client.writes, [])

    def test_stale_run_skips_before_reading_jobs(self):
        client = FakeGitHubClient(current_sha="b" * 40, elapsed_seconds=2 * 60 * 60)
        result = warning.observe_run(client, self.config)
        self.assertIn("is stale", result)
        self.assertEqual(client.writes, [])

    def test_dry_run_does_not_write(self):
        client = FakeGitHubClient(elapsed_seconds=2 * 60 * 60)
        config = warning.RunConfig(**{**self.config.__dict__, "dry_run": True})
        result = warning.observe_run(client, config)
        self.assertIn("Dry run: would create", result)
        self.assertEqual(client.writes, [])


if __name__ == "__main__":
    unittest.main()
