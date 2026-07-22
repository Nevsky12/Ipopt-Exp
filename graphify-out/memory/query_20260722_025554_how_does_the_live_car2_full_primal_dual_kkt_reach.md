---
type: "query"
date: "2026-07-22T02:55:54.503981+00:00"
question: "How does the live CAR2 full primal-dual KKT reach fallback-free candidate acceptance, and where are the remaining limits?"
contributor: "graphify"
outcome: "corrected"
correction: "Use the latest protocol numbers 1692.415/1952.636 ms and the separate order-alternated saved-binary comparison 1667.593/1672.146 ms; do not reuse the superseded 1672.994/1944.619 clause as the final result."
source_nodes: ["PrimalDualBorderedStageKktAssembler", "BorderedStageStructuredCandidateBackend", ".prepare_reusable_bordered_stage_storage()", ".CertifyReducedInertia()", "SymmetricBorderedBlockTridiagonalSolver"]
---

# Q: How does the live CAR2 full primal-dual KKT reach fallback-free candidate acceptance, and where are the remaining limits?

## Answer

Expanded from the original query via vocab: [CAR2, full primal-dual KKT, bordered stage, reusable workspace, structural zeros, exact inertia, structured residual, full residual, fallback]. The live route uses PrimalDualBorderedStageKktAssembler and BorderedStageStructuredCandidateBackend, normalizes implicit dynamics exactly, requires an independent exact full-KKT inertia proof, and retains both residual gates. The 2026-07-22 protocol result is 1692.415 ms versus matched MUMPS 1952.636 ms (13.33% faster), 171/171 accepted with zero fallback. An order-alternated saved-binary comparison measures 1667.593 versus 1672.146 ms end to end and 218.709 versus 226.850 ms in assembly after explicit reusable packed storage preserves structural zeros. Portable Callgrind is 24.513 billion instructions, 1.77% below the preceding candidate. Remaining limits are benchmark-specific CAR2 metadata, make_constraint, no NLP scaling, exact Hessian, no bordered restoration transform, and one CPU/MUMPS baseline.

## Outcome

- Signal: corrected
- Correction: Use the latest protocol numbers 1692.415/1952.636 ms and the separate order-alternated saved-binary comparison 1667.593/1672.146 ms; do not reuse the superseded 1672.994/1944.619 clause as the final result.

## Source Nodes

- PrimalDualBorderedStageKktAssembler
- BorderedStageStructuredCandidateBackend
- .prepare_reusable_bordered_stage_storage()
- .CertifyReducedInertia()
- SymmetricBorderedBlockTridiagonalSolver