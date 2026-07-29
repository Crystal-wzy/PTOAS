#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Post a non-blocking PR warning when ci-sim exceeds its runtime budget."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.error import HTTPError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen


COMMENT_MARKER = "<!-- ci-sim-duration-warning -->"
RESOLVED_PREFIX = "Resolved:"
BOT_LOGIN = "github-actions[bot]"
JOB_NAME = "vpto-sim-validation"
SLOW_LABEL = "ci-slow"
SHA_RE = re.compile(r"^[0-9a-fA-F]{40}$")


@dataclass(frozen=True)
class RunConfig:
    repo: str
    run_id: int
    run_head_sha: str
    run_url: str
    soft_timeout_minutes: int
    context_dir: Path
    dry_run: bool = False


class GitHubClient:
    def __init__(self, token: str, api_url: str = "https://api.github.com") -> None:
        self._token = token
        self._api_url = api_url.rstrip("/")

    def request(
        self,
        method: str,
        path: str,
        payload: Optional[Dict[str, Any]] = None,
        query: Optional[Dict[str, Union[str, int]]] = None,
    ) -> Tuple[Any, Dict[str, str]]:
        url = f"{self._api_url}/{path.lstrip('/')}"
        if query:
            url = f"{url}?{urlencode(query)}"
        data = json.dumps(payload).encode() if payload is not None else None
        request = Request(
            url,
            data=data,
            method=method,
            headers={
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {self._token}",
                "X-GitHub-Api-Version": "2022-11-28",
            },
        )
        try:
            with urlopen(request, timeout=30) as response:
                body = response.read()
                return (json.loads(body) if body else None), dict(response.headers)
        except HTTPError as error:
            detail = error.read().decode(errors="replace")
            raise RuntimeError(f"GitHub API {method} {path} failed: {error.code}: {detail}") from error

    def get(self, path: str, query: Optional[Dict[str, Union[str, int]]] = None) -> Any:
        return self.request("GET", path, query=query)[0]

    def post(self, path: str, payload: Dict[str, Any]) -> Any:
        return self.request("POST", path, payload=payload)[0]

    def patch(self, path: str, payload: Dict[str, Any]) -> Any:
        return self.request("PATCH", path, payload=payload)[0]

    def delete(self, path: str) -> Any:
        return self.request("DELETE", path)[0]

    def list_all(self, path: str) -> List[Any]:
        items: List[Any] = []
        page = 1
        while True:
            batch = self.get(path, query={"per_page": 100, "page": page})
            if not isinstance(batch, list):
                raise RuntimeError(f"Expected a list from GitHub API path {path}")
            items.extend(batch)
            if len(batch) < 100:
                return items
            page += 1


def _read_context(context_dir: Path) -> Optional[Tuple[int, str]]:
    number_path = context_dir / "pr-number"
    sha_path = context_dir / "pr-head-sha"
    if not number_path.is_file() or not sha_path.is_file():
        return None
    number_text = number_path.read_text(encoding="utf-8").strip()
    head_sha = sha_path.read_text(encoding="utf-8").strip()
    if not number_text.isdigit() or int(number_text) <= 0 or not SHA_RE.fullmatch(head_sha):
        raise ValueError(f"Invalid PR context in {context_dir}")
    return int(number_text), head_sha


def _parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _format_duration(seconds: int) -> str:
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m {seconds}s"
    return f"{minutes}m {seconds}s"


def _format_budget(minutes: int) -> str:
    hours, minutes = divmod(minutes, 60)
    if hours and minutes:
        return f"{hours}h {minutes}m"
    if hours:
        return f"{hours}h"
    return f"{minutes}m"


def _warning_body(author: str, elapsed: str, budget: str, conclusion: str, run_url: str) -> str:
    return "\n".join(
        [
            COMMENT_MARKER,
            f"Warning: @{author}, ci-sim exceeded its soft runtime budget.",
            "",
            f"- `{JOB_NAME}` runtime: **{elapsed}**",
            f"- Soft budget: **{budget}**",
            f"- Job conclusion: **{conclusion}**",
            f"- [Workflow run]({run_url})",
            "",
            "This warning is advisory only and does not affect required checks. "
            "Please inspect the step timings for an unexpected regression.",
        ]
    )


def _resolved_body(elapsed: str, budget: str, run_url: str) -> str:
    return "\n".join(
        [
            COMMENT_MARKER,
            "Resolved: ci-sim runtime is back within its soft budget.",
            "",
            f"- Latest `{JOB_NAME}` runtime: **{elapsed}**",
            f"- Soft budget: **{budget}**",
            f"- [Workflow run]({run_url})",
            "",
            "The previous duration warning is resolved. This status is advisory only.",
        ]
    )


def observe_run(client: GitHubClient, config: RunConfig) -> str:
    context = _read_context(config.context_dir)
    if context is None:
        return f"No PR context found for workflow run {config.run_id}; skipping duration warning."
    pr_number, context_head_sha = context

    pull = client.get(f"repos/{config.repo}/pulls/{pr_number}")
    if pull["state"] != "open":
        return f"PR #{pr_number} is {pull['state']}; skipping duration warning."
    current_head_sha = pull["head"]["sha"]
    if context_head_sha != config.run_head_sha or current_head_sha != config.run_head_sha:
        return f"Workflow run {config.run_id} is stale for PR #{pr_number}; skipping duration warning."
    has_slow_label = any(label.get("name") == SLOW_LABEL for label in pull.get("labels", []))

    jobs_response = client.get(
        f"repos/{config.repo}/actions/runs/{config.run_id}/jobs",
        query={"filter": "latest", "per_page": 100},
    )
    jobs = [
        job
        for job in jobs_response.get("jobs", [])
        if job.get("name") == JOB_NAME and job.get("started_at") and job.get("completed_at")
    ]
    if not jobs:
        return f"No completed {JOB_NAME} job found in workflow run {config.run_id}."
    job = jobs[0]
    elapsed_seconds = int((_parse_timestamp(job["completed_at"]) - _parse_timestamp(job["started_at"])).total_seconds())
    if elapsed_seconds < 0:
        raise ValueError(f"Invalid job timestamps: {job['started_at']} to {job['completed_at']}")

    elapsed = _format_duration(elapsed_seconds)
    budget = _format_budget(config.soft_timeout_minutes)
    comments = client.list_all(f"repos/{config.repo}/issues/{pr_number}/comments")
    existing = next(
        (
            comment
            for comment in reversed(comments)
            if comment.get("user", {}).get("login") == BOT_LOGIN
            and COMMENT_MARKER in comment.get("body", "")
        ),
        None,
    )

    if elapsed_seconds > config.soft_timeout_minutes * 60:
        body = _warning_body(pull["user"]["login"], elapsed, budget, job.get("conclusion", "unknown"), config.run_url)
        if config.dry_run:
            label_action = "keep" if has_slow_label else "add"
            return (
                f"Dry run: would {label_action} {SLOW_LABEL} and "
                f"{'update' if existing else 'create'} slow CI warning on PR #{pr_number}.\n{body}"
            )
        if not has_slow_label:
            client.post(f"repos/{config.repo}/issues/{pr_number}/labels", {"labels": [SLOW_LABEL]})
        if existing:
            client.patch(f"repos/{config.repo}/issues/comments/{existing['id']}", {"body": body})
            comment_action = "Updated"
        else:
            client.post(f"repos/{config.repo}/issues/{pr_number}/comments", {"body": body})
            comment_action = "Created"
        label_action = "Kept" if has_slow_label else "Applied"
        return f"{label_action} {SLOW_LABEL}; {comment_action.lower()} slow CI warning on PR #{pr_number}."

    comment_needs_resolution = bool(existing and RESOLVED_PREFIX not in existing.get("body", ""))
    if config.dry_run and (has_slow_label or comment_needs_resolution):
        actions = []
        if has_slow_label:
            actions.append(f"remove {SLOW_LABEL}")
        if comment_needs_resolution:
            actions.append("resolve slow CI warning")
        body = _resolved_body(elapsed, budget, config.run_url)
        return f"Dry run: would {' and '.join(actions)} on PR #{pr_number}.\n{body}"

    actions = []
    if has_slow_label:
        encoded_label = quote(SLOW_LABEL, safe="")
        client.delete(f"repos/{config.repo}/issues/{pr_number}/labels/{encoded_label}")
        actions.append(f"Removed {SLOW_LABEL}")
    if comment_needs_resolution:
        body = _resolved_body(elapsed, budget, config.run_url)
        client.patch(f"repos/{config.repo}/issues/comments/{existing['id']}", {"body": body})
        actions.append("resolved slow CI warning")
    if actions:
        return f"{'; '.join(actions)} on PR #{pr_number}."
    if existing:
        return f"CI runtime {elapsed} remains within the {budget} budget."
    return f"CI runtime {elapsed} is within the {budget} budget."


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--run-id", required=True, type=_positive_int)
    parser.add_argument("--run-head-sha", required=True)
    parser.add_argument("--run-url", required=True)
    parser.add_argument("--soft-timeout-minutes", type=_positive_int, default=90)
    parser.add_argument("--pr-context-dir", type=Path, default=Path("pr-context"))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    if not SHA_RE.fullmatch(args.run_head_sha):
        raise ValueError(f"Invalid run head SHA: {args.run_head_sha}")
    token = os.environ.get("GH_TOKEN")
    if not token:
        raise RuntimeError("GH_TOKEN is required")
    config = RunConfig(
        repo=args.repo,
        run_id=args.run_id,
        run_head_sha=args.run_head_sha,
        run_url=args.run_url,
        soft_timeout_minutes=args.soft_timeout_minutes,
        context_dir=args.pr_context_dir,
        dry_run=args.dry_run,
    )
    print(observe_run(GitHubClient(token), config))
    return 0


if __name__ == "__main__":
    sys.exit(main())
