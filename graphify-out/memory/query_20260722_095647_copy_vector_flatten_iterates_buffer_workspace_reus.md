---
type: "query"
date: "2026-07-22T09:56:47.496755+00:00"
question: "copy vector flatten iterates buffer workspace reuse span commit restoration initialize solve"
contributor: "graphify"
outcome: "useful"
source_nodes: [".EvaluateSolve()", ".Flatten()", "CopyToStable()", "IteratesVector"]
---

# Q: copy vector flatten iterates buffer workspace reuse span commit restoration initialize solve

## Answer

EvaluateSolve performed eleven owning state copies and candidate-first Flatten performed eight more per Newton request. Per-solver contiguous private workspaces can safely replace them because spans are rebuilt after resize, candidate requests are synchronous, restoration owns a separate solver, and failed partial writes are not published. The accepted CAR2 implementation removes 3249 owning CopyVector calls and reduces the matched profile from 21554845256 to 21371460949 instructions.

## Outcome

- Signal: useful

## Source Nodes

- .EvaluateSolve()
- .Flatten()
- CopyToStable()
- IteratesVector