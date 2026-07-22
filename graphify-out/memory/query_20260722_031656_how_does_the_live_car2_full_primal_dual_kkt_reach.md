---
type: "query"
date: "2026-07-22T03:16:56.567605+00:00"
question: "How does the live CAR2 full primal-dual KKT reach fallback-free candidate acceptance, and where are the remaining limits?"
contributor: "graphify"
outcome: "corrected"
correction: "Use the latest protocol numbers 1639.743491/1941.044926 ms and the latest matched immediately-previous comparison 1635.114369/1667.824741 ms; the older 1692.415/1952.636 and 1667.593/1672.146 records are historical, not current."
source_nodes: ["BorderedStageStructuredCandidateBackend", "CertifiedInertia", "SymmetricBlockTridiagonalSolver", ".FactorizeStageNumerically()"]
---

# Q: How does the live CAR2 full primal-dual KKT reach fallback-free candidate acceptance, and where are the remaining limits?

## Answer

The previous correction with 1692.415/1952.636 ms and 1667.593/1672.146 ms is superseded by the symmetric numeric Schur milestone. The acceptance contract is unchanged: independent exact full-KKT inertia, structured true residual, outer full-KKT residual, and zero fallback across 171/171 requests. The latest native interleaved protocol is candidate 1639.743491 ms versus MUMPS 1941.044926 ms. The matched x86-64-v3 immediately-previous comparison is 1635.114369 versus 1667.824741 ms end to end and 626.483487 versus 659.897442 ms in the backend. Remaining limits are still CAR2/DTOC3 benchmark-name topology, make_constraint, no NLP scaling, exact Hessian, and no bordered restoration transform.

## Outcome

- Signal: corrected
- Correction: Use the latest protocol numbers 1639.743491/1941.044926 ms and the latest matched immediately-previous comparison 1635.114369/1667.824741 ms; the older 1692.415/1952.636 and 1667.593/1672.146 records are historical, not current.

## Source Nodes

- BorderedStageStructuredCandidateBackend
- CertifiedInertia
- SymmetricBlockTridiagonalSolver
- .FactorizeStageNumerically()