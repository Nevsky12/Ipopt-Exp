---
type: "performance"
date: "2026-07-22T04:32:29.394008+00:00"
question: "How does the live CAR2 full primal-dual KKT reach fallback-free candidate acceptance, and where are the remaining limits?"
contributor: "graphify"
outcome: "corrected"
correction: "Use the final solve-layout protocol numbers 1625.941141/1940.324801 ms, the matched immediately-previous comparison 1623.054040/1626.941518 ms, backend 616.362797/621.073565 ms, and Callgrind 23,681,739,194; the 1632.607869/1933.827904 milestone is now historical."
source_nodes: ["BorderedStageStructuredCandidateBackend", "CertifiedInertia", "SymmetricBlockTridiagonalSolver", ".FormMultiplierFixed()", ".ApplySchurComplementFixed()", ".SolveOneImpl()"]
---

# Q: How does the live CAR2 full primal-dual KKT reach fallback-free candidate acceptance, and where are the remaining limits?

## Answer

Expanded through graph vocabulary [solve, rhs, bordered, schur, block, tridiagonal, forward, backward, workspace, memory, performance, profile]. The acceptance contract remains independent exact full-KKT inertia, structured residual, outer full-KKT residual, 171/171 accepted requests, and zero fallback. The multiplier is formed directly in column-major solve layout, fixed one-RHS forward/backward and stage kernels avoid generic scratch overhead, and 12x12 is tested first. Release and sanitizer suites pass 15/15. The latest native interleaved protocol is candidate 1625.941141 ms versus MUMPS 1940.324801 ms, 16.2026% faster. The matched x86-64-v3 immediately-previous comparison is 1623.054040 versus 1626.941518 ms end to end and 616.362797 versus 621.073565 ms in the backend; backend is faster in all seven pairs. Callgrind is 23,681,739,194 instructions, 1.5384% below the preceding candidate and 27.3695% below the initial candidate. Remaining limits are CAR2/DTOC3 benchmark-name topology, make_constraint, no NLP scaling, exact Hessian, no bordered restoration transform, and evidence from one CPU/MUMPS baseline.

## Outcome

- Signal: corrected
- Correction: Use the final solve-layout protocol numbers 1625.941141/1940.324801 ms, the matched immediately-previous comparison 1623.054040/1626.941518 ms, backend 616.362797/621.073565 ms, and Callgrind 23,681,739,194; the 1632.607869/1933.827904 milestone is now historical.

## Source Nodes

- BorderedStageStructuredCandidateBackend
- CertifiedInertia
- SymmetricBlockTridiagonalSolver
- .FormMultiplierFixed()
- .ApplySchurComplementFixed()
- .SolveOneImpl()