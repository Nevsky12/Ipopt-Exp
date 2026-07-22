---
type: "query"
date: "2026-07-21T18:24:57.813284+00:00"
question: "Как PDFullSpaceSolver, StdAugSystemSolver и TSymLinearSolver определяют, когда переиспользовать численную факторизацию для нескольких RHS, и какой явный factorize/solve_rhs lifecycle нужен direct preconditioner?"
contributor: "graphify"
outcome: "useful"
source_nodes: ["PDFullSpaceSolver", "SolveOnce", "StdAugSystemSolver", "AugmentedSystemRequiresChange", "TSymLinearSolver", "SparseSymLinearSolverInterface", "MultiSolve"]
---

# Q: Как PDFullSpaceSolver, StdAugSystemSolver и TSymLinearSolver определяют, когда переиспользовать численную факторизацию для нескольких RHS, и какой явный factorize/solve_rhs lifecycle нужен direct preconditioner?

## Answer

PDFullSpaceSolver::SolveOnce сравнивает cache dependencies по тегам W, J_c, J_d, диагоналей и slack-данных. Если система актуальна и не запрошена pretend singular, он повторно вызывает AugSystemSolver::Solve с текущими perturbations. StdAugSystemSolver пересобирает compound matrix только при изменении тегов или scalar deltas; TSymLinearSolver при неизменном теге передает MultiSolve(new_matrix=false), что по SparseSymLinearSolverInterface означает reuse старой факторизации. IncreaseQuality может потребовать refactorization. Поэтому prepared boundary должен проверять полный structural fingerprint и ненулевую caller-owned numeric revision, один раз вызывать factorize и затем только solve_rhs; другая revision обязана отклоняться до backend. GenAugSystemSolver сохраняет эту new_matrix семантику, но его per-call rhssol allocation остается отдельной целью оптимизации.

## Outcome

- Signal: useful

## Source Nodes

- PDFullSpaceSolver
- SolveOnce
- StdAugSystemSolver
- AugmentedSystemRequiresChange
- TSymLinearSolver
- SparseSymLinearSolverInterface
- MultiSolve