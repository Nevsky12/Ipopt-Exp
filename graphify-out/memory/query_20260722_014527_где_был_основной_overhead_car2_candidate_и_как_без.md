---
type: "query"
date: "2026-07-22T01:45:27.174994+00:00"
question: "Где был основной overhead CAR2 candidate и как безопасно убрать его без изменения stable ABI?"
contributor: "graphify"
outcome: "useful"
source_nodes: ["cached_coordinate_problem_", "LegacyAlgorithmCanaryPDSystemSolver", ".EvaluateSolve()", "BorrowNlpProblem()", "NlpProblemReference", ".factorize()", ".FactorizeStageNumerically()", "TestBorrowedProblemRetainsOwnerCaches()"]
---

# Q: Где был основной overhead CAR2 candidate и как безопасно убрать его без изменения stable ABI?

## Answer

Graph path: cached_coordinate_problem_ is defined by LegacyAlgorithmCanaryPDSystemSolver; its EvaluateSolve calls BorrowNlpProblem, so one owning NlpProblemReference source preserves derivative-structure and fingerprint caches across Newton KKT objects while InitializeImpl and restoration snapshots remain boundaries. The numeric path is factorize -> FactorizeStageNumerically; retaining partial pivoting while skipping eliminated prefixes and exact-zero multiplier updates removes redundant row work. Portable Callgrind fell 32.606B -> 25.404B instructions (-22.09%); CPU-4 Release median fell 2124.192 -> 1722.131 ms and became 11.68% faster than matched MUMPS, with 171/171 accepted and no fallback.

## Outcome

- Signal: useful

## Source Nodes

- cached_coordinate_problem_
- LegacyAlgorithmCanaryPDSystemSolver
- .EvaluateSolve()
- BorrowNlpProblem()
- NlpProblemReference
- .factorize()
- .FactorizeStageNumerically()
- TestBorrowedProblemRetainsOwnerCaches()