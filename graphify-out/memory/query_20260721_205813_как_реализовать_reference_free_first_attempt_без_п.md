---
type: "query"
date: "2026-07-21T20:58:13.169485+00:00"
question: "Как реализовать reference-free first attempt без потери политики устойчивости PDFullSpaceSolver?"
contributor: "graphify"
outcome: "useful"
source_nodes: ["PDFullSpaceSolver", "PDPerturbationHandler", "SolveOnce", "ComputeResiduals", "AugSystemSolver", "IncreaseQuality", "AlgorithmBuilder"]
---

# Q: Как реализовать reference-free first attempt без потери политики устойчивости PDFullSpaceSolver?

## Answer

Reference-free режим не должен переиспользовать LegacyAugSystemDirectBackend, потому что он всё равно факторизует stable AugSystemSolver. Нужен независимый pluggable AnyAny backend, который владеет perturbation retries, inertia, refinement и quality escalation, возвращает accepted regularization и exact inertia certificate. Оболочка проверяет finite full direction и true KKT residual, делает transactional commit, а при любой ошибке вызывает полный PDFullSpaceSolver ровно один раз.

## Outcome

- Signal: useful

## Source Nodes

- PDFullSpaceSolver
- PDPerturbationHandler
- SolveOnce
- ComputeResiduals
- AugSystemSolver
- IncreaseQuality
- AlgorithmBuilder