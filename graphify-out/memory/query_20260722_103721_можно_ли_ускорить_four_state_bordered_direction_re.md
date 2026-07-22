---
type: "query"
date: "2026-07-22T10:37:21.690029+00:00"
question: "Можно ли ускорить four-state bordered direction reconstruction, сохранив порядок преобразования normalized dynamics?"
contributor: "graphify"
outcome: "useful"
source_nodes: ["PrimalDualBorderedStageKktAssembler", ".reconstruct_bordered_stage_direction()", "FourStateNormalizationProvider", "TestFixedFourStateNormalization()", "BorderedStageStructuredCandidateBackend"]
---

# Q: Можно ли ускорить four-state bordered direction reconstruction, сохранив порядок преобразования normalized dynamics?

## Answer

Expanded from original query via graph vocab: [reconstruct, bordered, stage, direction, normalization, four, state, fixed, transform, structured, solution, performance]. CAR2 executes the same 4x4 transpose transform for 999 transitions across 171 accepted solves. The generic nested loop repeatedly loaded dynamics ordering and structured/full positions. The accepted C++23 fixed-four-state helper loads four mapped positions and normalized direction values once, then emits four outputs with the original sequential accumulation order 0+term0+term1+term2+term3; every other state count retains the generic fallback. A focused NaN-poisoned 13-entry reconstruction regression requires exact equality to an independent scalar reference. Release and sanitizer suites remain 15/15. CAR2 Callgrind moved from 21310269000 to 21263975496 instructions, saving 46293504 (0.2172%); candidate-solve inclusive saved 46285029 and self saved 41341473 (36.70%). CUTEst and all derivative/constraint callback counts and costs remained identical. CAR2 stayed 171/171 with zero fallback and DTOC3 N5000/N30000 stayed 1/1.

## Outcome

- Signal: useful

## Source Nodes

- PrimalDualBorderedStageKktAssembler
- .reconstruct_bordered_stage_direction()
- FourStateNormalizationProvider
- TestFixedFourStateNormalization()
- BorderedStageStructuredCandidateBackend