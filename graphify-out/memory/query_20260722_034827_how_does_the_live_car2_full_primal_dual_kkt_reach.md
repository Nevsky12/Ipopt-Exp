---
type: "query"
date: "2026-07-22T03:48:27.246888+00:00"
question: "How does the live CAR2 full primal-dual KKT reach fallback-free candidate acceptance, and where are the remaining limits?"
contributor: "graphify"
outcome: "corrected"
correction: "Use the latest protocol numbers 1632.607869/1933.827904 ms and the latest matched immediately-previous comparison 1632.929334/1636.302818 ms; the 1639.743491/1941.044926 and 1635.114369/1667.824741 records are now historical."
source_nodes: ["BorderedStageStructuredCandidateBackend", "CertifiedInertia", "SymmetricBlockTridiagonalSolver", ".FactorizeStageNumerically()", ".FormMultiplierFixed()", ".ApplySchurComplementFixed()"]
---

# Q: How does the live CAR2 full primal-dual KKT reach fallback-free candidate acceptance, and where are the remaining limits?

## Answer

Expanded through graph vocabulary [factorize, numerically, invert, inverse, pivot, permutation, matrix, schur, residual, inertia, performance, profile]. The acceptance contract is unchanged: independent exact full-KKT inertia, structured and outer full-KKT residual gates, 171/171 accepted requests, and zero fallback. The fixed numeric factor now elides dead pivot stores, fuses permutation restoration with inverse symmetry projection, and dispatches 14x14 multiplier and Schur kernels. Release and sanitizer suites pass 15/15. The latest native interleaved protocol is candidate 1632.607869 ms versus MUMPS 1933.827904 ms, 15.5764% faster. The matched x86-64-v3 immediately-previous comparison is 1632.929334 versus 1636.302818 ms end to end and 620.573769 versus 628.899761 ms in the backend. Callgrind is 24,051,758,865 instructions, 1.0305% below the preceding candidate and 26.2347% below the initial candidate. Remaining limits are CAR2/DTOC3 benchmark-name topology, make_constraint, no NLP scaling, exact Hessian, no bordered restoration transform, and evidence from one CPU/MUMPS baseline.

## Outcome

- Signal: corrected
- Correction: Use the latest protocol numbers 1632.607869/1933.827904 ms and the latest matched immediately-previous comparison 1632.929334/1636.302818 ms; the 1639.743491/1941.044926 and 1635.114369/1667.824741 records are now historical.

## Source Nodes

- BorderedStageStructuredCandidateBackend
- CertifiedInertia
- SymmetricBlockTridiagonalSolver
- .FactorizeStageNumerically()
- .FormMultiplierFixed()
- .ApplySchurComplementFixed()