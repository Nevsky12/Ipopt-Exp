---
type: "benchmark"
date: "2026-07-22T11:27:44.905517+00:00"
question: "What is the current accepted CAR2 x86-64-v3 Callgrind baseline after fused residual scaling?"
contributor: "graphify"
outcome: "corrected"
correction: "Replace the caller-owned-direction current value 21234851595 with 21211940826. Preserve 21234851595 as the immediately preceding accepted profile."
source_nodes: [".EvaluateSolve()", "RelativeInfinityError()", "Comparison"]
---

# Q: What is the current accepted CAR2 x86-64-v3 Callgrind baseline after fused residual scaling?

## Answer

The current accepted release total is 21211940826 instructions; the immediately preceding caller-owned-direction profile is 21234851595. Release and debug text are identical, debug totals 21211941085, and the accepted step saves 22910769 instructions (0.1079%). The cumulative reduction from 32605780860 is 34.9442%, and the reduction versus the 31077838252 coordinate-matched baseline is 31.7458%.

## Outcome

- Signal: corrected
- Correction: Replace the caller-owned-direction current value 21234851595 with 21211940826. Preserve 21234851595 as the immediately preceding accepted profile.

## Source Nodes

- .EvaluateSolve()
- RelativeInfinityError()
- Comparison