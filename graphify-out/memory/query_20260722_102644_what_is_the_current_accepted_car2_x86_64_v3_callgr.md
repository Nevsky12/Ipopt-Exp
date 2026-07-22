---
type: "query"
date: "2026-07-22T10:26:44.428909+00:00"
question: "What is the current accepted CAR2 x86-64-v3 Callgrind baseline after reconstruction complementarity validation reuse?"
contributor: "graphify"
outcome: "corrected"
correction: "Replace the direct-commit current value 21325152116 with 21310269000. Preserve 21325152116 as the immediately preceding accepted profile."
source_nodes: ["PrimalDualState", ".assemble_bordered_stage_system()", ".reconstruct_bordered_stage_direction()", ".ValidComplementarityState()"]
---

# Q: What is the current accepted CAR2 x86-64-v3 Callgrind baseline after reconstruction complementarity validation reuse?

## Answer

The accepted portable x86-64-v3 CAR2 profile is 21310269000 instructions. The immediately preceding direct equal-precision commit profile is 21325152116. The new step saves 14883116 instructions (0.0698%), and the cumulative reduction from 32605780860 is 34.6427%; versus the coordinate-matched ordinary baseline 31077838252 it is 31.4294% lower. Release/debug .text is identical and debug measures 21310268999.

## Outcome

- Signal: corrected
- Correction: Replace the direct-commit current value 21325152116 with 21310269000. Preserve 21325152116 as the immediately preceding accepted profile.

## Source Nodes

- PrimalDualState
- .assemble_bordered_stage_system()
- .reconstruct_bordered_stage_direction()
- .ValidComplementarityState()