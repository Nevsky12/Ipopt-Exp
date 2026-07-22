---
type: "performance"
date: "2026-07-22T11:27:44.763406+00:00"
question: "Где следующий устранимый per-solve overhead после caller-owned output, не ослабляя residual, scaling и finite gates?"
contributor: "graphify"
outcome: "useful"
source_nodes: [".EvaluateSolve()", "RelativeInfinityError()", ".FlattenInto()", "Comparison", "PrimalDualKktOperator", "CopyVectorsInto()"]
---

# Q: Где следующий устранимый per-solve overhead после caller-owned output, не ослабляя residual, scaling и finite gates?

## Answer

Expanded-token trace: [evaluate, solve, overhead, removable, candidate, residual, scaling, finite, workspace, vector, copy, commit]. Graphify связал LegacyAlgorithmCanaryPDSystemSolver::EvaluateSolve с RelativeInfinityError, FlattenInto, candidate solve и Comparison. Построчный Callgrind показал после true KKT apply три последовательных полновекторных прохода: residual infinity reduction, scaling и post-scale finite validation. Их можно слить в один index-ordered helper: порядок двух residual max-аккумуляторов остаётся прежним, каждый direction element масштабируется, finite-флаг аккумулируется без ранней публикации. Независимый unscaled finite gate перед KKT apply и reference-direction comparison после scaling сохраняются. CAR2 сохранил 171/171 и zero fallback; профиль снизился с 21234851595 до 21211940826 Ir. 171 отдельных RelativeInfinityError и 29071539 Ir исчезли, EvaluateSolve inclusive снизился на 22912563 Ir; apply_flat, FlattenInto, assembly, provider и CUTEst callbacks не изменились.

## Outcome

- Signal: useful

## Source Nodes

- .EvaluateSolve()
- RelativeInfinityError()
- .FlattenInto()
- Comparison
- PrimalDualKktOperator
- CopyVectorsInto()