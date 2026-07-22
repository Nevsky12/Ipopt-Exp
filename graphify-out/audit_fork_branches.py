#!/usr/bin/env python3
"""Patch-equivalence audit for every non-upstream branch head in public forks."""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any


def git(repository: Path, *arguments: str, timeout: int = 180) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "--git-dir", str(repository), *arguments],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout,
    )


def fetch(repository: Path, ref: str, aliases: list[dict[str, Any]]) -> str | None:
    for alias in aliases:
        remote = f"https://github.com/{alias['fork']}.git"
        source = f"refs/heads/{alias['branch']}"
        result = git(repository, "fetch", "--quiet", "--no-tags", "--force", remote, f"+{source}:{ref}")
        if result.returncode == 0:
            return None
    return result.stderr.strip()[-1000:]


def commit_description(repository: Path, sha: str) -> dict[str, str]:
    result = git(repository, "show", "-s", "--format=%H%x00%ad%x00%an%x00%s", "--date=iso-strict", sha)
    commit, date, author, subject = result.stdout.rstrip("\n").split("\0", maxsplit=3)
    return {"sha": commit, "date": date, "author": author, "subject": subject}


def changed_files(repository: Path, sha: str) -> tuple[list[str], int]:
    result = git(repository, "show", "--format=", "--name-status", sha)
    files = [line for line in result.stdout.splitlines() if line]
    return files[:200], len(files)


def audit(repository: Path, index: int, sha: str, aliases: list[dict[str, Any]]) -> dict[str, Any]:
    ref = f"refs/fork-branch-audit/{index:03d}"
    error = fetch(repository, ref, aliases)
    if error:
        return {"sha": sha, "aliases": aliases, "audit": "fetch-error", "error": error}

    head = commit_description(repository, ref)
    base = git(repository, "merge-base", "HEAD", ref)
    if base.returncode != 0:
        history = git(repository, "log", "-50", "--format=%H%x09%ad%x09%an%x09%s", "--date=short", ref)
        return {
            "sha": sha,
            "aliases": aliases,
            "audit": "unrelated-history",
            "head": head,
            "recent_commits": history.stdout.splitlines(),
        }

    counts = git(repository, "rev-list", "--left-right", "--count", f"HEAD...{ref}")
    behind, ahead = (int(value) for value in counts.stdout.split())
    cherry = git(repository, "cherry", "HEAD", ref)
    patches = []
    for line in cherry.stdout.splitlines():
        disposition, patch_sha = line.split(maxsplit=1)
        description = commit_description(repository, patch_sha)
        files, file_count = changed_files(repository, patch_sha)
        patches.append(
            {
                "disposition": "novel" if disposition == "+" else "patch-in-upstream",
                **description,
                "file_count": file_count,
                "files": files,
            }
        )
    return {
        "sha": sha,
        "aliases": aliases,
        "audit": "compared",
        "head": head,
        "merge_base": base.stdout.strip(),
        "behind": behind,
        "ahead": ahead,
        "novel_patches": sum(patch["disposition"] == "novel" for patch in patches),
        "upstreamed_patches": sum(patch["disposition"] == "patch-in-upstream" for patch in patches),
        "patches": patches,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_repository", type=Path)
    parser.add_argument("fork_branches", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    audit_root = Path(tempfile.mkdtemp(prefix="ipopt-fork-branch-audit."))
    repository = audit_root / "repository.git"
    subprocess.run(
        ["git", "clone", "--bare", "--shared", str(args.source_repository), str(repository)],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=120,
    )

    forks = json.loads(args.fork_branches.read_text(encoding="utf-8"))
    aliases_by_sha: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for fork in forks:
        for branch in fork["branches"]:
            if branch["screen"] == "custom-or-unfetched":
                aliases_by_sha[branch["sha"]].append(
                    {
                        "fork": fork["full_name"],
                        "branch": branch["name"],
                        "default": branch["default"],
                        "fork_pushed_at": fork["pushed_at"],
                    }
                )

    results = [
        audit(repository, index, sha, aliases)
        for index, (sha, aliases) in enumerate(sorted(aliases_by_sha.items()))
    ]
    args.output.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "audit_repository": str(repository),
                "unique_heads": len(results),
                "compared": sum(result["audit"] == "compared" for result in results),
                "unrelated": sum(result["audit"] == "unrelated-history" for result in results),
                "fetch_errors": sum(result["audit"] == "fetch-error" for result in results),
                "novel_patches": sum(result.get("novel_patches", 0) for result in results),
                "upstreamed_patches": sum(result.get("upstreamed_patches", 0) for result in results),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
