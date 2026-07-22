---
type: "query"
date: "2026-07-22T10:58:36.113727+00:00"
question: "What is the current accepted CAR2 x86-64-v3 Callgrind baseline after certified full-direction overwrite?"
contributor: "graphify"
outcome: "corrected"
correction: "Replace the fixed-four-state-reconstruction current value 21263975496 with 21252352975. Preserve 21263975496 as the immediately preceding accepted profile."
source_nodes: ["BorderedStageStructuredCandidateBackend", "StageStructuredCandidateBackend"]
---

# Q: What is the current accepted CAR2 x86-64-v3 Callgrind baseline after certified full-direction overwrite?

## Answer

The current accepted release total is 21252352975 instructions; the immediately preceding fixed-four-state-reconstruction profile is 21263975496. Release/debug text is identical, debug totals 21252352951, and the accepted step saves 11622521 instructions (0.0547%).

## Outcome

- Signal: corrected
- Correction: Replace the fixed-four-state-reconstruction current value 21263975496 with 21252352975. Preserve 21263975496 as the immediately preceding accepted profile.

## Source Nodes

- BorderedStageStructuredCandidateBackend
- StageStructuredCandidateBackend