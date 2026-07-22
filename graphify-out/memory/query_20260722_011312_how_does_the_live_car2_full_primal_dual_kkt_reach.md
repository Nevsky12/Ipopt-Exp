---
type: "query"
date: "2026-07-22T01:13:12.314484+00:00"
question: "How does the live CAR2 full primal-dual KKT reach fallback-free candidate acceptance, and where are the remaining limits?"
contributor: "graphify"
outcome: "useful"
source_nodes: ["BorderedStageNlpTopology", "SparseBorderedStageDerivativeProvider", "PrimalDualBorderedStageKktAssembler", ".InvertNextStateBlock()", ".NormalizeDynamics()", ".CertifyReducedInertia()", "BorderedStageStructuredCandidateBackend", "SymmetricBorderedBlockTridiagonalSolver", ".reconstruct_bordered_stage_direction()", "PrimalDualKktOperator"]
---

# Q: How does the live CAR2 full primal-dual KKT reach fallback-free candidate acceptance, and where are the remaining limits?

## Answer

The validated route is explicit local/global stage metadata into BorderedStageNlpTopology, followed by an O(nnz) SparseBorderedStageDerivativeProvider scatter. PrimalDualBorderedStageKktAssembler performs exact complementarity condensation for primal/slack bounds, assembles path equalities and inequalities plus adjacent Hessians and the global arrowhead, and either keeps the dense next-state D natively or applies the exact congruence L=-D^-1. The normalized route gates each inverse by relative pivots and backward error, transforms dynamics/global Jacobians, RHS, and -delta_c L L^T, then reconstructs the original dual as L^T eta. CertifyReducedInertia supplies an independent exact saddle-point theorem before numeric factorization; missing proof grows only the hinted primal or dual perturbation. SymmetricBorderedBlockTridiagonalSolver factors the stage chain once, solves A^-1 B in a fused multi-RHS pass, factors the dense Schur complement, and refines against the original arrowhead. Reconstruction restores bound duals and the outer PrimalDualKktOperator true-residual gate remains mandatory. The CAR2 run accepted 171/171 requests with zero fallback and 1.63e-13 violation, but was 9.95% slower than matched MUMPS; public TNLP stage metadata and bordered restoration remain absent.

## Outcome

- Signal: useful

## Source Nodes

- BorderedStageNlpTopology
- SparseBorderedStageDerivativeProvider
- PrimalDualBorderedStageKktAssembler
- .InvertNextStateBlock()
- .NormalizeDynamics()
- .CertifyReducedInertia()
- BorderedStageStructuredCandidateBackend
- SymmetricBorderedBlockTridiagonalSolver
- .reconstruct_bordered_stage_direction()
- PrimalDualKktOperator
- TestReverseAndRejectedMetadata()