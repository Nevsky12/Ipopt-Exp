---
type: "query"
date: "2026-07-22T02:24:08.031514+00:00"
question: "How does the live CAR2 full primal-dual KKT reach fallback-free candidate acceptance, and where are the remaining limits?"
contributor: "graphify"
outcome: "corrected"
correction: "Replace the superseded 1722.131/1949.923 ms clause with the final 2026-07-22 CPU-4 result: candidate 1672.994 ms versus matched MUMPS 1944.619 ms, 13.97% faster; retain the warning that nonlinear trajectories differ."
source_nodes: ["BorderedStageStructuredCandidateBackend", ".FactorizeStageNumerically()", ".CertifyReducedInertia()", "CheckTrueResidual()"]
---

# Q: How does the live CAR2 full primal-dual KKT reach fallback-free candidate acceptance, and where are the remaining limits?

## Answer

The previous 1722.131 ms versus 1949.923 ms CAR2 result is superseded by the final 2026-07-22 CPU-4 interleaved series. The lazy pivot-column numeric factor preserves partial pivoting, the independent exact full-KKT inertia proof, and structured plus outer residual gates. Candidate median is 1672.994 ms versus matched MUMPS 1944.619 ms, 13.97% faster; 171/171 requests are accepted with zero fallback and maximum violation 1.63e-13. Nonlinear trajectories still differ, so this remains end-to-end evidence rather than an isolated per-iteration kernel comparison.

## Outcome

- Signal: corrected
- Correction: Replace the superseded 1722.131/1949.923 ms clause with the final 2026-07-22 CPU-4 result: candidate 1672.994 ms versus matched MUMPS 1944.619 ms, 13.97% faster; retain the warning that nonlinear trajectories differ.

## Source Nodes

- BorderedStageStructuredCandidateBackend
- .FactorizeStageNumerically()
- .CertifyReducedInertia()
- CheckTrueResidual()