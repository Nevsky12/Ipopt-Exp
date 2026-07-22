---
type: "performance"
date: "2026-07-22T04:57:12.514852+00:00"
question: "How does the live CAR2 full primal-dual KKT reach fallback-free candidate acceptance, and where are the remaining limits?"
contributor: "graphify"
outcome: "corrected"
correction: "Use the final single-pass-validation numbers 1593.689622/1943.821955 ms, the matched immediately-previous comparison 1586.013662/1636.394557 ms, backend 577.130908/620.433367 ms, and Callgrind 22,958,661,729; the 1625.941141/1940.324801 solve-layout milestone is now historical."
source_nodes: ["BorderedStageStructuredCandidateBackend", "CertifiedInertia", "SymmetricBlockTridiagonalSolver", "SymmetricBorderedBlockTridiagonalSolver", ".factorize()", ".ValidateAndCopyDiagonal()", ".CertifyReducedInertia()"]
---

# Q: How does the live CAR2 full primal-dual KKT reach fallback-free candidate acceptance, and where are the remaining limits?

## Answer

Expanded through graph vocabulary [factorize, stage, numerically, invert, inverse, pivot, permutation, matrix, inertia, residual, performance, profile]. Candidate acceptance still requires independent exact full-KKT inertia, structured true residual, and outer full-KKT residual; all 171 requests are accepted with zero fallback. Single-pass stage validation/copy and bordered delegation remove duplicate scans without changing pivot, inertia, or residual policy, and the symmetric projection remains finite for same-sign coefficients near Number::max(). Release and sanitizer suites pass 15/15. The latest native interleaved protocol is candidate 1593.689622 ms versus coordinate-matched MUMPS 1943.821955 ms, 18.0126% faster. The matched x86-64-v3 immediately-previous comparison is 1586.013662 versus 1636.394557 ms end to end and 577.130908 versus 620.433367 ms in the backend, with the candidate faster in all seven pairs. Callgrind is 22,958,661,729 instructions, 3.0533% below the preceding candidate and 29.5871% below the initial candidate. Remaining limits are CAR2/DTOC3 benchmark-name topology, make_constraint, no NLP scaling, exact Hessian, no bordered restoration transform, and evidence from one CPU and a MUMPS baseline.

## Outcome

- Signal: corrected
- Correction: Use the final single-pass-validation numbers 1593.689622/1943.821955 ms, the matched immediately-previous comparison 1586.013662/1636.394557 ms, backend 577.130908/620.433367 ms, and Callgrind 22,958,661,729; the 1625.941141/1940.324801 solve-layout milestone is now historical.

## Source Nodes

- BorderedStageStructuredCandidateBackend
- CertifiedInertia
- SymmetricBlockTridiagonalSolver
- SymmetricBorderedBlockTridiagonalSolver
- .factorize()
- .ValidateAndCopyDiagonal()
- .CertifyReducedInertia()