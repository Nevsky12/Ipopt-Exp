#!/usr/bin/env python3
"""Enumerate and classify every branch of every public Ipopt fork."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import subprocess
from pathlib import Path
from typing import Any


def run(*arguments: str, timeout: int = 45) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        arguments,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout,
    )


def classify(sha: str) -> str:
    if run("git", "cat-file", "-e", f"{sha}^{{commit}}", timeout=5).returncode != 0:
        return "custom-or-unfetched"
    if run("git", "merge-base", "--is-ancestor", sha, "origin/stable/3.14", timeout=5).returncode == 0:
        return "upstream-ancestor"
    containing = run("git", "branch", "-r", "--contains", sha, timeout=5)
    if any(line.strip().startswith("origin/") for line in containing.stdout.splitlines()):
        return "upstream-other-branch"
    return "known-other-history"


def resolve(fork: dict[str, Any]) -> dict[str, Any]:
    try:
        result = run("git", "ls-remote", "--heads", f"https://github.com/{fork['full_name']}.git")
    except subprocess.TimeoutExpired:
        return {**fork, "branch_audit": "timeout", "branches": []}
    if result.returncode != 0:
        return {
            **fork,
            "branch_audit": "ls-remote-error",
            "error": result.stderr.strip()[-500:],
            "branches": [],
        }
    branches = []
    for line in result.stdout.splitlines():
        sha, ref = line.split(maxsplit=1)
        name = ref.removeprefix("refs/heads/")
        branches.append(
            {
                "name": name,
                "sha": sha,
                "default": name == fork["default_branch"],
                "screen": classify(sha),
            }
        )
    return {**fork, "branch_audit": "ok", "branches": branches}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inventory", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--jobs", type=int, default=16)
    args = parser.parse_args()

    payload = json.loads(args.inventory.read_text(encoding="utf-8"))
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        results = list(pool.map(resolve, payload["forks"]))
    args.output.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    branches = [branch for fork in results for branch in fork["branches"]]
    counts: dict[str, int] = {}
    for branch in branches:
        counts[branch["screen"]] = counts.get(branch["screen"], 0) + 1
    print(
        json.dumps(
            {
                "forks": len(results),
                "fork_errors": sum(fork["branch_audit"] != "ok" for fork in results),
                "branches": len(branches),
                "unique_heads": len({branch["sha"] for branch in branches}),
                "screens": counts,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
