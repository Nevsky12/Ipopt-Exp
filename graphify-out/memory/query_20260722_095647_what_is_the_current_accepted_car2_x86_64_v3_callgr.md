---
type: "query"
date: "2026-07-22T09:56:47.558395+00:00"
question: "What is the current accepted CAR2 x86-64-v3 Callgrind baseline after the latest optimization?"
contributor: "graphify"
outcome: "corrected"
correction: "Replace the stale current value 21825555191 with 21371460949. The intermediate coordinate-KKT reuse profile is 21554845256."
source_nodes: [".EvaluateSolve()", ".Flatten()"]
---

# Q: What is the current accepted CAR2 x86-64-v3 Callgrind baseline after the latest optimization?

## Answer

The accepted evaluation-vector workspace candidate is now the current profile: 21371460949 release instructions and 21371460928 debug instructions, with identical release/debug text.

## Outcome

- Signal: corrected
- Correction: Replace the stale current value 21825555191 with 21371460949. The intermediate coordinate-KKT reuse profile is 21554845256.

## Source Nodes

- .EvaluateSolve()
- .Flatten()