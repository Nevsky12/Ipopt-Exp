---
type: "call"
date: "2026-07-21T19:19:58.364818+00:00"
question: "Где безопасно встроить C++23 full-system runtime canary, не ломая retry state machine PDFullSpaceSolver и stable ABI?"
contributor: "graphify"
outcome: "useful"
source_nodes: ["AlgorithmBuilder", "IpoptApplication", "PDFullSpaceSolver", "PDSystemSolver", "AugSystemSolver", "OrigIpoptNLP", "TNLPAdapter"]
---

# Q: Где безопасно встроить C++23 full-system runtime canary, не ломая retry state machine PDFullSpaceSolver и stable ABI?

## Answer

Безопасный seam — внешний наследник AlgorithmBuilder, переданный в IpoptApplication::OptimizeNLP(nlp, builder), с override виртуального PDSystemSolverFactory. LegacyAlgorithmCanaryBuilder сначала вызывает базовую factory, затем получает тот же AugSystemSolver через GetAugSystemSolver и оборачивает reference PDSystemSolver. Обёртка всегда сначала полностью выполняет reference Solve, поэтому perturbation/inertia/refinement/IncreaseQuality state machine PDFullSpaceSolver остаётся авторитетной. Только после успешного exact-Hessian OrigIpoptNLP/TNLPAdapter solve canary читает уже принятые PDPert, W, jac_c/jac_d и живые curr_sigma_x/curr_sigma_s с их исходными tags, собирает восьмиблочный C++23 KKT, решает FGMRES через существующий AugSystemSolver и сравнивает direction/residual. Legacy direction не изменяется; ошибки и mismatch по умолчанию лишь учитываются, strict failure включается явно. Это opt-in compatibility gate без изменений stable ABI/default build; он делает дополнительные derivative callbacks/backsolves и потому не является performance mode.

## Outcome

- Signal: useful

## Source Nodes

- AlgorithmBuilder
- IpoptApplication
- PDFullSpaceSolver
- PDSystemSolver
- AugSystemSolver
- OrigIpoptNLP
- TNLPAdapter