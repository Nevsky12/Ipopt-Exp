---
type: "query"
date: "2026-07-21T20:24:05.924918+00:00"
question: "Где безопасно реализовать opt-in guarded replacement C++23 direction в legacy Ipopt?"
contributor: "graphify"
outcome: "useful"
source_nodes: ["PDFullSpaceSolver", "PDSystemSolver", "SolveOnce", "ComputeResiduals", "IncreaseQuality", "PDPerturbationHandler", "IteratesVector", "LegacyAlgorithmCanaryPDSystemSolver", "LegacyAlgorithmCanaryMode"]
---

# Q: Где безопасно реализовать opt-in guarded replacement C++23 direction в legacy Ipopt?

## Answer

Decorator должен сначала завершить reference PDSystemSolver::Solve: только после perturbation, inertia retries, refinement и IncreaseQuality доступны принятые tags и direction. Eligible candidate проверяется true residual и equivalence tolerance; в validated replacement все восемь блоков сначала конвертируются в detached IteratesVector, затем выполняется один final Copy. Любая ошибка оставляет reference result. Это проверяет commit boundary, но не ускоряет solve при reference-first.

## Outcome

- Signal: useful

## Source Nodes

- PDFullSpaceSolver
- PDSystemSolver
- SolveOnce
- ComputeResiduals
- IncreaseQuality
- PDPerturbationHandler
- IteratesVector
- LegacyAlgorithmCanaryPDSystemSolver
- LegacyAlgorithmCanaryMode