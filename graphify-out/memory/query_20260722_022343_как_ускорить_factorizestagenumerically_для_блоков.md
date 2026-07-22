---
type: "query"
date: "2026-07-22T02:23:43.448865+00:00"
question: "Как ускорить FactorizeStageNumerically для блоков 12/14, сохранив partial pivoting, exact inertia boundary и residual gates?"
contributor: "graphify"
outcome: "useful"
source_nodes: [".factorize()", ".FactorizeStageNumerically()", "SymmetricBlockTridiagonalSolver", "SymmetricBorderedBlockTridiagonalSolver", "CertifiedInertia", ".CertifyReducedInertia()", "CheckTrueResidual()"]
---

# Q: Как ускорить FactorizeStageNumerically для блоков 12/14, сохранив partial pivoting, exact inertia boundary и residual gates?

## Answer

Expanded from original query via graph vocab: [factorize, stage, numerically, pivot, inverse, permutation, contiguous, support, bordered, inertia, residual, performance]. The updated graph shows that .factorize() calls .FactorizeStageNumerically(); the method belongs to SymmetricBlockTridiagonalSolver, which is referenced by SymmetricBorderedBlockTridiagonalSolver, while CertifiedInertia/.CertifyReducedInertia() and CheckTrueResidual() remain separate downstream gates. Implemented a lazy pivot-column-order augmented identity for fixed-size Gauss-Jordan: it stays in final storage until the first real row pivot, then preserves exact contiguous processed-prefix support and restores inverse columns once at the end. Partial pivot selection, pivot margin, finite checks, independent exact full-KKT inertia proof, and structured/full residual gates are unchanged. Dense repeated-pivot tests for sizes 12/14, Release 15/15, ASan/UBSan/leak 15/15, and CAR2 171/171 with zero fallback passed. Numeric-factor medians improved 16.19%-18.47%; CAR2 median is 1672.994 ms versus MUMPS 1944.619 ms (13.97% faster); portable Callgrind is 24.955 billion instructions and numeric-factor self cost is 1.773 billion.

## Outcome

- Signal: useful

## Source Nodes

- .factorize()
- .FactorizeStageNumerically()
- SymmetricBlockTridiagonalSolver
- SymmetricBorderedBlockTridiagonalSolver
- CertifiedInertia
- .CertifyReducedInertia()
- CheckTrueResidual()