---
type: "dfs"
date: "2026-07-21T23:36:38.700859+00:00"
question: "Как live Ipopt full-KKT проходит через bounds, path inequalities, complementarity condensation, stage backend, inertia certification, reconstruction и restoration fallback?"
contributor: "graphify"
outcome: "useful"
source_nodes: ["PrimalDualStageKktAssembler", "PrimalDualStageKktLayoutWorkspace", "StageNlpTopology", "PrimalDualKktOperator", "StageStructuredCandidateBackend", "LiveStageBackendFactory", "RunLiveStageSolve", "PDFullSpaceSolver", "PDPerturbationHandler"]
---

# Q: Как live Ipopt full-KKT проходит через bounds, path inequalities, complementarity condensation, stage backend, inertia certification, reconstruction и restoration fallback?

## Answer

Расширение запроса: kkt, primal, dual, bounds, inequality, slack, complementarity, restoration, stage, structured, assembler, perturbation. PrimalDualStageKktAssembler валидирует явные StageNlpTopology и PrimalDualLayout, собирает симметричную редуцированную систему x/s/y_c/y_d, точно исключает и восстанавливает z_L/z_U/v_L/v_U, а StageStructuredCandidateBackend принимает результат только с точной reduced-inertia сертификацией и residual gates. LiveStageBackendFactory принимает bounded main NLP, но отклоняет compound restoration NLP, после чего candidate-first gate ровно один раз вызывает стабильный PDFullSpaceSolver.

## Outcome

- Signal: useful

## Source Nodes

- PrimalDualStageKktAssembler
- PrimalDualStageKktLayoutWorkspace
- StageNlpTopology
- PrimalDualKktOperator
- StageStructuredCandidateBackend
- LiveStageBackendFactory
- RunLiveStageSolve
- PDFullSpaceSolver
- PDPerturbationHandler