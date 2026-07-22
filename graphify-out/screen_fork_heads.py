#!/usr/bin/env python3
"""Resolve every public fork default-branch head without spending API quota."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import subprocess
from pathlib import Path
from typing import Any


def run(*arguments: str, timeout: int = 30) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        arguments,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout,
    )


def resolve(fork: dict[str, Any]) -> dict[str, Any]:
    full_name = fork["full_name"]
    branch = fork["default_branch"]
    try:
        result = run(
            "git",
            "ls-remote",
            "--heads",
            f"https://github.com/{full_name}.git",
            f"refs/heads/{branch}",
        )
    except subprocess.TimeoutExpired:
        return {**fork, "head_sha": None, "screen": "timeout"}
    if result.returncode != 0:
        return {
            **fork,
            "head_sha": None,
            "screen": "ls-remote-error",
            "error": result.stderr.strip()[-500:],
        }
    fields = result.stdout.split()
    if not fields:
        return {**fork, "head_sha": None, "screen": "missing-default-head"}
    sha = fields[0]
    known = run("git", "cat-file", "-e", f"{sha}^{{commit}}", timeout=5).returncode == 0
    if not known:
        screen = "custom-or-unfetched"
    elif run("git", "merge-base", "--is-ancestor", sha, "origin/stable/3.14", timeout=5).returncode == 0:
        screen = "upstream-ancestor"
    else:
        screen = "known-other-history"
    return {**fork, "head_sha": sha, "screen": screen}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inventory", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--jobs", type=int, default=16)
    args = parser.parse_args()

    payload = json.loads(args.inventory.read_text(encoding="utf-8"))
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        screened = list(pool.map(resolve, payload["forks"]))
    args.output.write_text(
        json.dumps(screened, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    counts: dict[str, int] = {}
    for fork in screened:
        counts[fork["screen"]] = counts.get(fork["screen"], 0) + 1
    print(json.dumps(counts, sort_keys=True))


if __name__ == "__main__":
    main()
