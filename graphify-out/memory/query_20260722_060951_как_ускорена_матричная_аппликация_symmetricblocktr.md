---
type: "performance"
date: "2026-07-22T06:09:51.608901+00:00"
question: "Как ускорена матричная аппликация SymmetricBlockTridiagonalSolver и bordered refinement, и какие performance/robustness gates подтвердили изменение?"
contributor: "graphify"
outcome: "useful"
source_nodes: [".apply()", "SymmetricBlockTridiagonalSolver", ".solve_refined_rhs()", "block_tridiagonal_solver_benchmark.cpp", "BorderedBlockTridiagonalWorkspaceProfile"]
---

# Q: Как ускорена матричная аппликация SymmetricBlockTridiagonalSolver и bordered refinement, и какие performance/robustness gates подтвердили изменение?

## Answer

Factorization stores an exactly symmetric projected diagonal, so apply can traverse one triangle and update both endpoints. Fixed 12x12/14x14 dispatch removes runtime-size loop overhead. Bordered internal refinement writes directly to known nonalias workspaces; public apply still stages input/output for alias safety and transactional failure semantics. Tests cover projected asymmetric input, aliasing, application counts, and overflow with unchanged caller output. Seven paired microbenchmarks improve every median by 4.1584–13.1077%, with 139/140 faster pairs and zero checksum mismatch. Callgrind falls from 22,918,435,450 to 22,710,389,439 instructions; solve_refined_rhs falls 14.9492%. Release and ASan/UBSan pass 15/15. The separate CAR2 timer delta is +0.2917%, so no isolated wall-clock speedup is claimed.
Expanded tokens: [assemble, assembler, factorize, stage, numerically, matrix, performance, profile, copy, residual, solve, border]

## Outcome

- Signal: useful

## Source Nodes

- .apply()
- SymmetricBlockTridiagonalSolver
- .solve_refined_rhs()
- block_tridiagonal_solver_benchmark.cpp
- BorderedBlockTridiagonalWorkspaceProfile