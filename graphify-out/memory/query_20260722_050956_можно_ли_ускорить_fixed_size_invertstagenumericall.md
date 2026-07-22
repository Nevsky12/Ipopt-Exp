---
type: "performance"
date: "2026-07-22T05:09:56.781033+00:00"
question: "Можно ли ускорить fixed-size InvertStageNumerically ленивой инициализацией augmented identity и копированием только активного prefix при первом row pivot?"
contributor: "graphify"
outcome: "dead_end"
source_nodes: ["InvertStageNumerically()", ".FactorizeStageNumerically()", "SymmetricBlockTridiagonalSolver", "block_tridiagonal_solver_benchmark.cpp", "TestUncertifiedNumericFactor()"]
---

# Q: Можно ли ускорить fixed-size InvertStageNumerically ленивой инициализацией augmented identity и копированием только активного prefix при первом row pivot?

## Answer

Expanded through graph vocabulary [factorize, numerically, assemble, assembler, matrix, block, stage, performance, profile, copy, residual, inertia]. The attempted fixed-size kernel removed the full N-by-N identity clear, assigned each inverse column on first use, and copied only the live prefix when the first row pivot activated permutation storage. Correctness passed, including a new diagonal-to-pivoted-to-diagonal stale-workspace regression, and all paired checksums matched. However seven order-alternated CPU-4 pairs rejected it: ordinary numeric size-12 cases regressed 0.23%-3.58%, size-14 regressed 3.31%-5.97%, and pivot-heavy size-14 regressed 2.35%-4.34% with only mixed sub-1.6% gains for pivot-heavy size 12. The optimized memset and simpler contiguous loops outperform the extra per-column assignment and prefix-copy control. The solver change was fully reverted; the permanent pivot-heavy benchmark and robustness test remain.

## Outcome

- Signal: dead_end

## Source Nodes

- InvertStageNumerically()
- .FactorizeStageNumerically()
- SymmetricBlockTridiagonalSolver
- block_tridiagonal_solver_benchmark.cpp
- TestUncertifiedNumericFactor()