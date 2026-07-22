---
type: "query"
date: "2026-07-21T19:06:03.640212+00:00"
question: "Как PDFullSpaceSolver и PDSystemSolver вызывают AugSystemSolver::Solve или MultiSolve, где строятся rhs и IteratesVector, и как retries по perturbation inertia и IncreaseQuality должны быть сохранены при opt-in backend?"
contributor: "graphify"
outcome: "useful"
source_nodes: ["PDFullSpaceSolver", "AugSystemSolver", "MultiSolve", "StdAugSystemSolver", "TSymLinearSolver", "PDPerturbationHandler", "IncreaseQuality", "IteratesVector"]
---

# Q: Как PDFullSpaceSolver и PDSystemSolver вызывают AugSystemSolver::Solve или MultiSolve, где строятся rhs и IteratesVector, и как retries по perturbation inertia и IncreaseQuality должны быть сохранены при opt-in backend?

## Answer

PDFullSpaceSolver::SolveOnce формирует reduced rhs и восстанавливает bound directions; AugSystemSolver решает 4-block x/s/c/d. StdAugSystemSolver сохраняет compound matrix по tags/scalar deltas, а неизменный tag позволяет TSymLinearSolver использовать new_matrix=false. Perturbation, inertia retry, iterative refinement и IncreaseQuality принадлежат PDFullSpaceSolver и не должны переходить в backend. Реализован opt-in LegacyAugSystemDirectBackend: zero-RHS prepare, solve-many, tag/numeric validation и exact 8-to-4 reduction/reconstruction; source-tree, pkg-config, Release и sanitizer gates проходят 6/6.

## Outcome

- Signal: useful

## Source Nodes

- PDFullSpaceSolver
- AugSystemSolver
- MultiSolve
- StdAugSystemSolver
- TSymLinearSolver
- PDPerturbationHandler
- IncreaseQuality
- IteratesVector