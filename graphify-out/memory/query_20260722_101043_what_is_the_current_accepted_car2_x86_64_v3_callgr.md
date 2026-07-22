---
type: "query"
date: "2026-07-22T10:10:43.373553+00:00"
question: "What is the current accepted CAR2 x86-64-v3 Callgrind baseline after direct equal-precision commit?"
contributor: "graphify"
outcome: "corrected"
correction: "Replace the evaluation-workspace milestone 21371460949 with current 21325152116. Preserve 21371460949 as the immediately preceding accepted profile."
source_nodes: [".CommitDirection()", ".EvaluateSolve()"]
---

# Q: What is the current accepted CAR2 x86-64-v3 Callgrind baseline after direct equal-precision commit?

## Answer

The accepted direct equal-precision detached-commit candidate is current: 21325152116 release instructions and 21325152177 debug instructions with identical text.

## Outcome

- Signal: corrected
- Correction: Replace the evaluation-workspace milestone 21371460949 with current 21325152116. Preserve 21371460949 as the immediately preceding accepted profile.

## Source Nodes

- .CommitDirection()
- .EvaluateSolve()