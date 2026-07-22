---
type: "architecture"
date: "2026-07-22T11:16:36.427735+00:00"
question: "Как убрать owning-копию полного направления в CandidateFirstSolveResult без нарушения lifetime, транзакционности и AnyAny?"
contributor: "graphify"
outcome: "useful"
source_nodes: ["CandidateFirstSolveRequest", "CandidateFirstSolveResult", "StageStructuredCandidateBackend", "BorderedStageStructuredCandidateBackend", "LegacyAlgorithmCanaryPDSystemSolver", "LegacyAlgorithmCanaryPDSystemSolver::EvaluateSolve()", "CopyVectorsInto()"]
---

# Q: Как убрать owning-копию полного направления в CandidateFirstSolveResult без нарушения lifetime, транзакционности и AnyAny?

## Answer

Expanded-token trace: [candidate, direction, result, copy, ownership, borrow, span, workspace, backend, transactional, lifecycle, commit]. CandidateFirstSolveRequest может нести optional std::span<Number> direction_output точного полного KKT-размера. StageStructuredCandidateBackend и BorderedStageStructuredCandidateBackend синхронно реконструируют в него, никогда не удерживают span и при успехе возвращают пустой owning direction плюс однозначный direction_written_to_request_output=true; вызовы без output сохраняют совместимый owning-vector fallback. LegacyAlgorithmCanaryPDSystemSolver::EvaluateSolve передаёт уже существующий flat_reference_workspace_ только в candidate-first ветви без reference_result, валидирует размер, finite, true KKT residual и scaling до публикации, а внутренний Comparison заимствует workspace лишь до немедленного CommitDirection. Ошибка вправе испортить только приватный scratch и не публикует направление. AnyAny остаётся синхронной value/request операцией. CAR2 подтвердил 171/171, zero fallback и прежние work counts; const-vector assignment упал с 173 вызовов и 23,410,741 inclusive Ir до 2 и 96,725, а полный профиль с 21,252,352,975 до 21,234,851,595 Ir.

## Outcome

- Signal: useful

## Source Nodes

- CandidateFirstSolveRequest
- CandidateFirstSolveResult
- StageStructuredCandidateBackend
- BorderedStageStructuredCandidateBackend
- LegacyAlgorithmCanaryPDSystemSolver
- LegacyAlgorithmCanaryPDSystemSolver::EvaluateSolve()
- CopyVectorsInto()