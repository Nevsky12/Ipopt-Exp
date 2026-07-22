---
type: "query"
date: "2026-07-22T06:57:16.883002+00:00"
question: "Ускоряет ли разбиение Gauss--Jordan elimination row loop вокруг pivot row вместо проверки row == column?"
contributor: "graphify"
outcome: "dead_end"
source_nodes: [".FactorizeStageNumerically()", "InvertStageNumerically()", "block_tridiagonal_solver_benchmark.cpp", "SymmetricBlockTridiagonalSolver"]
---

# Q: Ускоряет ли разбиение Gauss--Jordan elimination row loop вокруг pivot row вместо проверки row == column?

## Answer

Нет. В block_tridiagonal_solver_benchmark на блоках 12/14 и 16--256 stages все 20 медиан factorization ухудшились: ordinary 12 на 2.7534--5.4467%, ordinary 14 на 2.7268--3.4101%, pivot-heavy 12 на 0.8709--3.0696%, pivot-heavy 14 на 0.0832--1.0805%. Быстрее были лишь 25/140 пар; checksum и apply-checksum совпали во всех парах. Изменение полностью откатили, после пересборки SHA-256 снова e707e2dc6761ae7a42dc6a31b0b013eb2cf6cdb5901bf6e3a532bcc86d609e06. Query expansion trace: [factorize, stage, numerically, assembler, memory, copy, fill, workspace, schur, performance, profile, residual].

## Outcome

- Signal: dead_end

## Source Nodes

- .FactorizeStageNumerically()
- InvertStageNumerically()
- block_tridiagonal_solver_benchmark.cpp
- SymmetricBlockTridiagonalSolver