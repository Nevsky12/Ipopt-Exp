#!/usr/bin/env python3
"""Fetch a reproducible public GitHub inventory for the Ipopt ecosystem."""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


API = "https://api.github.com"
REPOSITORY = "coin-or/Ipopt"


def get_json(url: str) -> tuple[Any, dict[str, str]]:
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "ipopt-modernization-research",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, headers=headers)
    for attempt in range(4):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                result = json.load(response)
                return result, dict(response.headers.items())
        except urllib.error.HTTPError as error:
            if error.code not in {429, 500, 502, 503, 504} or attempt == 3:
                raise
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


def get_all(endpoint: str, **parameters: str) -> tuple[list[Any], list[dict[str, str]]]:
    items: list[Any] = []
    responses: list[dict[str, str]] = []
    page = 1
    while True:
        query = "&".join(
            [*(f"{key}={value}" for key, value in parameters.items()), "per_page=100", f"page={page}"]
        )
        batch, headers = get_json(f"{API}{endpoint}?{query}")
        if not isinstance(batch, list):
            raise TypeError(f"Expected a list from {endpoint}, got {type(batch).__name__}")
        items.extend(batch)
        responses.append(
            {
                "page": str(page),
                "count": str(len(batch)),
                "remaining": headers.get("X-RateLimit-Remaining", ""),
                "reset": headers.get("X-RateLimit-Reset", ""),
            }
        )
        if len(batch) < 100:
            return items, responses
        page += 1


def compact_pull(pull: dict[str, Any]) -> dict[str, Any]:
    return {
        "number": pull["number"],
        "state": pull["state"],
        "draft": pull.get("draft", False),
        "title": pull["title"],
        "body": pull.get("body") or "",
        "author": pull["user"]["login"],
        "created_at": pull["created_at"],
        "updated_at": pull["updated_at"],
        "closed_at": pull.get("closed_at"),
        "merged_at": pull.get("merged_at"),
        "url": pull["html_url"],
        "base": pull["base"]["ref"],
        "head_label": pull["head"]["label"],
        "head_sha": pull["head"]["sha"],
        "labels": [label["name"] for label in pull.get("labels", [])],
    }


def compact_fork(fork: dict[str, Any]) -> dict[str, Any]:
    return {
        "full_name": fork["full_name"],
        "owner": fork["owner"]["login"],
        "url": fork["html_url"],
        "description": fork.get("description"),
        "created_at": fork["created_at"],
        "updated_at": fork["updated_at"],
        "pushed_at": fork["pushed_at"],
        "default_branch": fork["default_branch"],
        "size": fork["size"],
        "stars": fork["stargazers_count"],
        "forks": fork["forks_count"],
        "open_issues": fork["open_issues_count"],
        "language": fork.get("language"),
        "archived": fork["archived"],
        "disabled": fork["disabled"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    repository, repository_headers = get_json(f"{API}/repos/{REPOSITORY}")
    pulls, pull_pages = get_all(f"/repos/{REPOSITORY}/pulls", state="all", sort="created", direction="desc")
    forks, fork_pages = get_all(f"/repos/{REPOSITORY}/forks", sort="newest")

    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": f"{API}/repos/{REPOSITORY}",
        "repository": {
            "full_name": repository["full_name"],
            "default_branch": repository["default_branch"],
            "forks_count": repository["forks_count"],
            "open_issues_count": repository["open_issues_count"],
            "stargazers_count": repository["stargazers_count"],
            "pushed_at": repository["pushed_at"],
            "updated_at": repository["updated_at"],
        },
        "pulls": [compact_pull(pull) for pull in pulls],
        "forks": [compact_fork(fork) for fork in forks],
        "pagination": {
            "pulls": pull_pages,
            "forks": fork_pages,
            "repository_remaining": repository_headers.get("X-RateLimit-Remaining", ""),
        },
    }
    (args.output / "inventory.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "pulls": len(payload["pulls"]),
                "merged": sum(pull["merged_at"] is not None for pull in payload["pulls"]),
                "open": sum(pull["state"] == "open" for pull in payload["pulls"]),
                "forks": len(payload["forks"]),
                "remaining": fork_pages[-1]["remaining"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
