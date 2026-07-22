---
type: "benchmark"
date: "2026-07-22T11:16:51.828296+00:00"
question: "What is the current accepted CAR2 x86-64-v3 Callgrind baseline after caller-owned candidate direction output?"
contributor: "graphify"
outcome: "corrected"
correction: "Replace the certified-full-direction-overwrite current value 21252352975 with 21234851595. Preserve 21252352975 as the immediately preceding accepted profile."
source_nodes: ["CandidateFirstSolveRequest", "CandidateFirstSolveResult", "BorderedStageStructuredCandidateBackend", "LegacyAlgorithmCanaryPDSystemSolver::EvaluateSolve()"]
---

# Q: What is the current accepted CAR2 x86-64-v3 Callgrind baseline after caller-owned candidate direction output?

## Answer

The current accepted release total is 21234851595 instructions; the immediately preceding certified-full-direction-overwrite profile is 21252352975. Release and debug text are identical, debug totals 21234851447, and the accepted step saves 17501380 instructions (0.0824%). The cumulative reduction from 32605780860 is 34.8740%, and the reduction versus the 31077838252 coordinate-matched baseline is 31.6720%.

## Outcome

- Signal: corrected
- Correction: Replace the certified-full-direction-overwrite current value 21252352975 with 21234851595. Preserve 21252352975 as the immediately preceding accepted profile.

## Source Nodes

- CandidateFirstSolveRequest
- CandidateFirstSolveResult
- BorderedStageStructuredCandidateBackend
- LegacyAlgorithmCanaryPDSystemSolver::EvaluateSolve()