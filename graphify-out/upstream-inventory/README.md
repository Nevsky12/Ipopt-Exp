# Public upstream inventory

Snapshot time: `2026-07-21T16:03:20Z`.

Contents:

- `inventory.json`: official repository metadata, every public pull request
  returned by GitHub API, and every public fork returned by GitHub API;
- `fork-branches.json`: `git ls-remote --heads` result and initial upstream
  classification for every accessible fork;
- `fork-branch-diffs.json`: fetched custom heads and patch-equivalence audit
  against local `HEAD`.

Counts:

- 46 pull requests: 35 merged, 4 open, 7 closed-unmerged;
- 311 API-visible public forks versus repository counter 317;
- 3372 branch refs, 242 unique head SHAs;
- 91 custom refs, 81 unique custom heads;
- 79 heads compared, 2 unrelated histories, 0 custom-head fetch errors;
- 311 unique novel commit SHAs and 43 unique upstream-equivalent commit SHAs.

Reproduction from the Ipopt worktree:

```sh
python3 graphify-out/fetch_upstream_inventory.py graphify-out/upstream-inventory
python3 graphify-out/screen_fork_branches.py \
  graphify-out/upstream-inventory/inventory.json \
  graphify-out/upstream-inventory/fork-branches.json
python3 graphify-out/audit_fork_branches.py . \
  graphify-out/upstream-inventory/fork-branches.json \
  graphify-out/upstream-inventory/fork-branch-diffs.json
```

The first command uses unauthenticated GitHub API by default. Set
`GITHUB_TOKEN` when a larger rate limit is needed. Results are time-dependent:
forks can disappear, become private, or gain branches after the snapshot.

SHA-256:

```text
0706c9cb4cf28efd802bc6187fb67999aeff84388b3d77a7532e026a69a8dbb2  inventory.json
7b248a5c720a7b4c2d95553d7816ac25a6704634da394309b73b9cadbcc8754a  fork-branches.json
8d782601e34064ecc01f6666cf8979a6faaa14f5ead6bc841fa250c8d7855266  fork-branch-diffs.json
```
