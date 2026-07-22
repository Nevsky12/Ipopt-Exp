---
type: "query"
date: "2026-07-22T10:25:08.164597+00:00"
question: "Можно ли безопасно исключить повторную ValidComplementarityState при reconstruction, используя numeric_revision после успешной bordered assembly?"
contributor: "graphify"
outcome: "useful"
source_nodes: ["PrimalDualState", ".assemble_bordered_stage_system()", ".reconstruct_bordered_stage_direction()", ".ValidComplementarityState()", "BorderedStageStructuredCandidateBackend"]
---

# Q: Можно ли безопасно исключить повторную ValidComplementarityState при reconstruction, используя numeric_revision после успешной bordered assembly?

## Answer

Expanded from original query via graph vocab: [numeric, revision, complementarity, state, validation, assembly, reconstruction, cache, successful, slack, robustness]. PrimalDualState defines numeric_revision as the caller-owned generation for all reachable numeric KKT and evaluation-state values. BorderedStageStructuredCandidateBackend synchronously reconstructs only after a successful assembly with the same request. The accepted implementation clears any prior proof at each assembly entry, validates complementarity on every assembly attempt, publishes the revision only after complete successful assembly, and skips the reconstruction scan only for that same nonzero revision; zero and changed revisions retain full validation. CAR2 Callgrind moved from 21325152116 to 21310269000 instructions, saving 14883116 (0.0698%). Reconstruction validation fell from 171 calls and 14535513 inclusive instructions to zero, while all 175 assembly checks and 14875525 inclusive instructions remain. CAR2 stayed 171/171 with zero fallback and identical objective/trajectory; DTOC3 N5000/N30000 stayed 1/1; Release and ASan+UBSan+leak stayed 15/15.

## Outcome

- Signal: useful

## Source Nodes

- PrimalDualState
- .assemble_bordered_stage_system()
- .reconstruct_bordered_stage_direction()
- .ValidComplementarityState()
- BorderedStageStructuredCandidateBackend