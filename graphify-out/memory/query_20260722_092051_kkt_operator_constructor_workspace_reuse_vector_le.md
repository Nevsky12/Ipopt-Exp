---
type: "code_architecture"
date: "2026-07-22T09:20:51.988086+00:00"
question: "kkt operator constructor workspace reuse vector legacy canary evaluate solve cache factory"
contributor: "graphify"
outcome: "useful"
source_nodes: ["LegacyAlgorithmCanaryPDSystemSolver", "EvaluateSolve", "PrimalDualKktOperator", "NlpKktOperator"]
---

# Q: kkt operator constructor workspace reuse vector legacy canary evaluate solve cache factory

## Answer

Useful: the graph connected LegacyAlgorithmCanaryPDSystemSolver::EvaluateSolve to PrimalDualKktOperator construction and the cached coordinate problem. Reusing the coordinate-path KKT operator until InitializeImpl, while retaining transient matrix-snapshot KKT operators, preserved CAR2 171/171 with zero fallback and reduced Callgrind Ir from 21,825,555,191 to 21,554,845,256 (-270,709,935; -1.240334702%).

## Outcome

- Signal: useful

## Source Nodes

- LegacyAlgorithmCanaryPDSystemSolver
- EvaluateSolve
- PrimalDualKktOperator
- NlpKktOperator