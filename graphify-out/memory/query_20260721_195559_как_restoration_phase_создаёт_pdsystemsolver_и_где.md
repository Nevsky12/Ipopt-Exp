---
type: "architecture"
date: "2026-07-21T19:55:59.513388+00:00"
question: "Как restoration phase создаёт PDSystemSolver и где безопасно встроить C++23 full-system runtime canary без изменения stable ABI?"
contributor: "graphify"
outcome: "useful"
source_nodes: ["AlgorithmBuilder", "BuildLineSearch", "PDSystemSolverFactory", "PDFullSpaceSolver", "AugRestoSystemSolver", "RestoIpoptNLP", "MinC_1NrmRestorationPhase", "LegacyAlgorithmCanaryBuilder", "LegacyMatrixSnapshot"]
---

# Q: Как restoration phase создаёт PDSystemSolver и где безопасно встроить C++23 full-system runtime canary без изменения stable ABI?

## Answer

Публичный builder seam был неполным: основной PD solver создавался через виртуальный PDSystemSolverFactory, а restoration PDFullSpaceSolver конструировался напрямую внутри AlgorithmBuilder::BuildLineSearch. Теперь BuildLineSearch повторно использует ту же virtual factory с префиксом resto.; default factory распознаёт его и оборачивает общий AugSystemSolver в AugRestoSystemSolver. Невиртуальный PDFullSpaceSolver::AugmentedSystemSolver accessor раскрывает точную owned wrapper/cache без нового vtable slot или изменения layout. Внешний canary поэтому оборачивает main и restoration solver, всегда выполняет reference первым и оставляет его direction авторитетной. Для OrigIpoptNLP/TNLPAdapter используется callback bridge; для RestoIpoptNLP создаётся owning triplet snapshot W, J_c и J_d через публичный TripletHelper, потому что restoration matrices compound и прямой TNLP bridge к ним неприменим.

## Outcome

- Signal: useful

## Source Nodes

- AlgorithmBuilder
- BuildLineSearch
- PDSystemSolverFactory
- PDFullSpaceSolver
- AugRestoSystemSolver
- RestoIpoptNLP
- MinC_1NrmRestorationPhase
- LegacyAlgorithmCanaryBuilder
- LegacyMatrixSnapshot