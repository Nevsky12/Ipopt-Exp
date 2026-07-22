---
type: "query"
date: "2026-07-22T10:39:21.755912+00:00"
question: "What is the current accepted CAR2 x86-64-v3 Callgrind baseline after fixed four-state reconstruction?"
contributor: "graphify"
outcome: "corrected"
correction: "Replace the complementarity-validation current value 21310269000 with 21263975496. Preserve 21310269000 as the immediately preceding accepted profile."
source_nodes: ["PrimalDualBorderedStageKktAssembler", ".reconstruct_bordered_stage_direction()", "TestFixedFourStateNormalization()"]
---

# Q: What is the current accepted CAR2 x86-64-v3 Callgrind baseline after fixed four-state reconstruction?

## Answer

The accepted portable x86-64-v3 CAR2 profile is 21263975496 instructions. The immediately preceding reconstruction-complementarity-validation profile is 21310269000. The fixed four-state reconstruction step saves 46293504 instructions (0.2172%). The cumulative reduction from 32605780860 is 34.7846%, and the candidate is 31.5783% below the coordinate-matched ordinary baseline 31077838252. Release/debug .text is identical; debug measures 21263975439.

## Outcome

- Signal: corrected
- Correction: Replace the complementarity-validation current value 21310269000 with 21263975496. Preserve 21310269000 as the immediately preceding accepted profile.

## Source Nodes

- PrimalDualBorderedStageKktAssembler
- .reconstruct_bordered_stage_direction()
- TestFixedFourStateNormalization()