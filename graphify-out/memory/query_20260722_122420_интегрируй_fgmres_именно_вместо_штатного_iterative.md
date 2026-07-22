---
type: "query"
date: "2026-07-22T12:24:20.648538+00:00"
question: "Интегрируй FGMRES именно вместо штатного iterative refinement в PDFullSpaceSolver, а не как отдельный canary-solve с нуля. Используй уже полученный direct-шаг как x_0, текущую факторизацию как right preconditioner без повторной факторизации и full KKT residual как оператор. Сравни с штатным refinement по времени, backsolve/KKT-apply, сходимости и траектории Ipopt. Отдельно протестируй K_delta/M=K_delta и K/M=K_delta. Проведи A/B-бенчмарки на CAR2 и DTOC3."
contributor: "graphify"
outcome: "useful"
source_nodes: ["src_algorithm_ippdfullspacesolver_pdfullspacesolver", "src_algorithm_ippdfullspacesolver_pdfullspacesolver_solveonce", "src_algorithm_ippdfullspacesolver_pdfullspacesolver_solvewithcurrentfactorization", "src_algorithm_ippdfullspacesolver_pdfullspacesolver_solvewithfgmres", "src_algorithm_ippdfullspacesolver_pdfullspacesolver_applykkt", "src_algorithm_ippdfullspacesolver_pdfullspacesolver_computeresiduals", "src_algorithm_ippdperturbationhandler_pdperturbationhandler", "src_algorithm_linearsolvers_iptsymlinearsolver_tsymlinearsolver_multisolve"]
---

# Q: Интегрируй FGMRES именно вместо штатного iterative refinement в PDFullSpaceSolver, а не как отдельный canary-solve с нуля. Используй уже полученный direct-шаг как x_0, текущую факторизацию как right preconditioner без повторной факторизации и full KKT residual как оператор. Сравни с штатным refinement по времени, backsolve/KKT-apply, сходимости и траектории Ipopt. Отдельно протестируй K_delta/M=K_delta и K/M=K_delta. Проведи A/B-бенчмарки на CAR2 и DTOC3.

## Answer

Expanded from original query via vocab: [pdfullspacesolver, refinement, fgmres, factorization, backsolve, preconditioner, residual, kkt, operator, solve, direct, perturbation]. The stock refinement phase begins after SolveOnce has produced the direct step and repeats cached augmented-system solves against the complete residual. The integrated implementation therefore keeps res as x_0, uses SolveWithCurrentFactorization only after the exact 13-tag cache check, applies all eight KKT blocks through ApplyKKT/ComputeResiduals, and selects perturbations in the operator for K_delta/M=K_delta or excludes them for K/M=K_delta. No FGMRES preconditioner call enters SolveOnce or the perturbation/factorization retry path.

## Outcome

- Signal: useful

## Source Nodes

- src_algorithm_ippdfullspacesolver_pdfullspacesolver
- src_algorithm_ippdfullspacesolver_pdfullspacesolver_solveonce
- src_algorithm_ippdfullspacesolver_pdfullspacesolver_solvewithcurrentfactorization
- src_algorithm_ippdfullspacesolver_pdfullspacesolver_solvewithfgmres
- src_algorithm_ippdfullspacesolver_pdfullspacesolver_applykkt
- src_algorithm_ippdfullspacesolver_pdfullspacesolver_computeresiduals
- src_algorithm_ippdperturbationhandler_pdperturbationhandler
- src_algorithm_linearsolvers_iptsymlinearsolver_tsymlinearsolver_multisolve