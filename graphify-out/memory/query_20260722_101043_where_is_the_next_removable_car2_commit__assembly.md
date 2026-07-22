---
type: "query"
date: "2026-07-22T10:10:43.317951+00:00"
question: "Where is the next removable CAR2 commit, assembly, or factorization overhead after evaluation workspace reuse?"
contributor: "graphify"
outcome: "useful"
source_nodes: [".CommitDirection()", "LegacyAlgorithmCanaryPDSystemSolver", "IteratesVector"]
---

# Q: Where is the next removable CAR2 commit, assembly, or factorization overhead after evaluation workspace reuse?

## Answer

Expanded via graph vocab: [commit, direction, replacement, vector, workspace, factorize, assembly, assembler, schur, refined, residual, candidate]. Traversal and matched profile isolated CommitDirection: its temporary full-direction stable vector plus duplicate equal-precision conversion dominated the function. Direct use of the already validated direction preserves the detached replacement transaction and reduces CAR2 from 21371460949 to 21325152116 instructions; CommitDirection inclusive cost falls by 81.3373 percent.

## Outcome

- Signal: useful

## Source Nodes

- .CommitDirection()
- LegacyAlgorithmCanaryPDSystemSolver
- IteratesVector