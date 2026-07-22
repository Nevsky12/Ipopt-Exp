---
type: "path_query"
date: "2026-07-22T01:45:50.604611+00:00"
question: "Path from factorize() to FactorizeStageNumerically()"
contributor: "graphify"
outcome: "useful"
source_nodes: [".factorize()", ".FactorizeStageNumerically()"]
---

# Q: Path from factorize() to FactorizeStageNumerically()

## Answer

factorize() calls FactorizeStageNumerically() directly when certified stage inertia is disabled. The independent full-KKT proof remains outside this numeric inverse, so pruning exact-zero/eliminated Gauss-Jordan work changes neither pivot selection nor the inertia claim boundary.

## Outcome

- Signal: useful

## Source Nodes

- .factorize()
- .FactorizeStageNumerically()