#!/usr/bin/env python3
"""Fetch the screened fork heads into a temporary bare repo and compare patches."""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any


def git(repository: Path, *arguments: str, timeout: int = 120) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "--git-dir", str(repository), *arguments],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout,
    )


def audit(repository: Path, index: int, fork: dict[str, Any]) -> dict[str, Any]:
    ref = f"refs/fork-audit/{index:03d}"
    remote = f"https://github.com/{fork['full_name']}.git"
    source = f"refs/heads/{fork['default_branch']}"
    fetched = git(repository, "fetch", "--quiet", "--no-tags", "--force", remote, f"+{source}:{ref}")
    if fetched.returncode != 0:
        return {
            **fork,
            "audit": "fetch-error",
            "error": fetched.stderr.strip()[-1000:],
        }

    base = git(repository, "merge-base", "HEAD", ref)
    if base.returncode != 0:
        return {**fork, "audit": "unrelated-history"}
    merge_base = base.stdout.strip()
    counts = git(repository, "rev-list", "--left-right", "--count", f"HEAD...{ref}")
    behind, ahead = (int(value) for value in counts.stdout.split())
    cherry = git(repository, "cherry", "HEAD", ref)
    patches = []
    for line in cherry.stdout.splitlines():
        disposition, sha = line.split(maxsplit=1)
        description = git(repository, "show", "-s", "--format=%ad%x09%an%x09%s", "--date=short", sha)
        date, author, subject = description.stdout.rstrip("\n").split("\t", maxsplit=2)
        files = git(repository, "show", "--format=", "--name-status", sha)
        patches.append(
            {
                "disposition": "novel" if disposition == "+" else "patch-in-upstream",
                "sha": sha,
                "date": date,
                "author": author,
                "subject": subject,
                "files": [line for line in files.stdout.splitlines() if line],
            }
        )
    return {
        **fork,
        "audit": "compared",
        "merge_base": merge_base,
        "behind": behind,
        "ahead": ahead,
        "novel_patches": sum(patch["disposition"] == "novel" for patch in patches),
        "upstreamed_patches": sum(patch["disposition"] == "patch-in-upstream" for patch in patches),
        "patches": patches,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_repository", type=Path)
    parser.add_argument("screened_forks", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    audit_root = Path(tempfile.mkdtemp(prefix="ipopt-fork-audit."))
    repository = audit_root / "repository.git"
    cloned = subprocess.run(
        ["git", "clone", "--bare", "--shared", str(args.source_repository), str(repository)],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=120,
    )
    if cloned.returncode != 0:
        raise RuntimeError(cloned.stderr)

    forks = json.loads(args.screened_forks.read_text(encoding="utf-8"))
    candidates = [fork for fork in forks if fork["screen"] not in {"upstream-ancestor", "ls-remote-error"}]
    results = [audit(repository, index, fork) for index, fork in enumerate(candidates)]
    args.output.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    summary = {
        "audit_repository": str(repository),
        "candidates": len(results),
        "compared": sum(result["audit"] == "compared" for result in results),
        "novel_patches": sum(result.get("novel_patches", 0) for result in results),
        "fetch_errors": sum(result["audit"] == "fetch-error" for result in results),
    }
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
