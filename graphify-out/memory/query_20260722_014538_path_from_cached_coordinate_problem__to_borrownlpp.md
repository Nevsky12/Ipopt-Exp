---
type: "path_query"
date: "2026-07-22T01:45:38.725404+00:00"
question: "Path from cached_coordinate_problem_ to BorrowNlpProblem()"
contributor: "graphify"
outcome: "useful"
source_nodes: ["cached_coordinate_problem_", "LegacyAlgorithmCanaryPDSystemSolver", ".EvaluateSolve()", "BorrowNlpProblem()"]
---

# Q: Path from cached_coordinate_problem_ to BorrowNlpProblem()

## Answer

cached_coordinate_problem_ <-defines- LegacyAlgorithmCanaryPDSystemSolver -method-> EvaluateSolve() -calls-> BorrowNlpProblem(). This is the ABI-safe ownership boundary: the solver owns the cached model, while each synchronous KKT gets only a forwarding AnyAny value.

## Outcome

- Signal: useful

## Source Nodes

- cached_coordinate_problem_
- LegacyAlgorithmCanaryPDSystemSolver
- .EvaluateSolve()
- BorrowNlpProblem()