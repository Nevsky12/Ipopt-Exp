---
type: "query"
date: "2026-07-22T01:46:52.763563+00:00"
question: "How does the live CAR2 full primal-dual KKT reach fallback-free candidate acceptance, and where are the remaining limits?"
contributor: "graphify"
outcome: "corrected"
correction: "Replace the stale 9.95% slower CAR2 clause with the 2026-07-22 post-optimization CPU-4 result: candidate median 1722.131 ms versus matched MUMPS 1949.923 ms, 11.68% faster; retain the warning that nonlinear trajectories differ."
source_nodes: ["BorderedStageNlpTopology", "SparseBorderedStageDerivativeProvider", "PrimalDualBorderedStageKktAssembler", "BorderedStageStructuredCandidateBackend", "SymmetricBorderedBlockTridiagonalSolver", "PrimalDualKktOperator", "cached_coordinate_problem_", "BorrowNlpProblem()", ".FactorizeStageNumerically()"]
---

# Q: How does the live CAR2 full primal-dual KKT reach fallback-free candidate acceptance, and where are the remaining limits?

## Answer

The validated bordered route and safety gates remain unchanged. The previous 9.95% slower performance clause is superseded: after persistent coordinate/fingerprint ownership and numeric-factor row pruning, seven CPU-4-pinned samples give 1722.131 ms candidate versus 1949.923 ms matched MUMPS, so candidate is 11.68% faster with 171/171 acceptance, zero fallback, and 1.63e-13 violation. Public TNLP stage metadata, bordered restoration, multi-problem and multi-backend evidence, and memory counters remain limits.

## Outcome

- Signal: corrected
- Correction: Replace the stale 9.95% slower CAR2 clause with the 2026-07-22 post-optimization CPU-4 result: candidate median 1722.131 ms versus matched MUMPS 1949.923 ms, 11.68% faster; retain the warning that nonlinear trajectories differ.

## Source Nodes

- BorderedStageNlpTopology
- SparseBorderedStageDerivativeProvider
- PrimalDualBorderedStageKktAssembler
- BorderedStageStructuredCandidateBackend
- SymmetricBorderedBlockTridiagonalSolver
- PrimalDualKktOperator
- cached_coordinate_problem_
- BorrowNlpProblem()
- .FactorizeStageNumerically()