---
type: "query"
date: "2026-07-21T23:11:43.701845+00:00"
question: "Как live DTOC3 candidate-first запрос проходит через lazy factory, equality stage assembler, independent full inertia proof, numeric block factor и stable fallback?"
contributor: "graphify"
outcome: "useful"
source_nodes: ["LazyCandidateFirstBackend", "StageNlpTopology", "EqualityStageKktAssembler", "block_tridiagonal_solver.hpp", "CandidateFirstSolveResult", "TestIndependentInertiaProofGate()", "TestLazyFactoryLifecycle()"]
---

# Q: Как live DTOC3 candidate-first запрос проходит через lazy factory, equality stage assembler, independent full inertia proof, numeric block factor и stable fallback?

## Answer

LazyCandidateFirstBackend создаёт типизированный backend только после появления live KKT и повторно использует его по полному fingerprint и restoration role. DTOC3 provider выдаёт проверенную StageNlpTopology, EqualityStageKktAssembler переставляет полный equality KKT в stage-блоки и независимо доказывает точную full-KKT inertia только для проверенной SPD либо DTOC-like PSD/full-row-rank структуры. Numeric block factor остаётся inexact; StageStructuredCandidateBackend принимает его лишь с независимым exact сертификатом, совпавшим числом отрицательных собственных значений, сошедшимся true-residual refinement и корректной реконструкцией. Внешний canary снова проверяет полный KKT residual и при любой ошибке вызывает стабильный solver ровно один раз.

## Outcome

- Signal: useful

## Source Nodes

- LazyCandidateFirstBackend
- StageNlpTopology
- EqualityStageKktAssembler
- block_tridiagonal_solver.hpp
- CandidateFirstSolveResult
- TestIndependentInertiaProofGate()
- TestLazyFactoryLifecycle()