---
type: "path_query"
date: "2026-07-22T01:13:20.802606+00:00"
question: "What is the shortest graph path from the full bordered stage assembler to the arrowhead numeric solver?"
contributor: "graphify"
outcome: "useful"
source_nodes: ["PrimalDualBorderedStageKktAssembler", "BorderedStageStructuredLayout", "BorderedStageStructuredCandidateBackend", "SymmetricBorderedBlockTridiagonalSolver"]
---

# Q: What is the shortest graph path from the full bordered stage assembler to the arrowhead numeric solver?

## Answer

PrimalDualBorderedStageKktAssembler references BorderedStageStructuredLayout; BorderedStageStructuredCandidateBackend consumes that same layout and owns SymmetricBorderedBlockTridiagonalSolver. This three-hop path is the static seam between exact NLP/KKT assembly and the numeric arrowhead factorization policy.

## Outcome

- Signal: useful

## Source Nodes

- PrimalDualBorderedStageKktAssembler
- BorderedStageStructuredLayout
- BorderedStageStructuredCandidateBackend
- SymmetricBorderedBlockTridiagonalSolver