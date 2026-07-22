---
type: "query"
date: "2026-07-21T21:57:31.035621+00:00"
question: "How should an equality-only OCP with fixed initial states flow from TNLPAdapter make_constraint through StageNlpTopology and EqualityStageKktAssembler into the candidate-first backend, and which unsupported layouts must fall back to the stable solver?"
contributor: "graphify"
outcome: "useful"
source_nodes: ["TNLPAdapter", "LegacyCoordinateModel", "StageNlpTopology", "PreparedStageDerivativeProvider", "EqualityStageKktAssembler", "StageStructuredCandidateBackend", "PrimalDualKktOperator"]
---

# Q: How should an equality-only OCP with fixed initial states flow from TNLPAdapter make_constraint through StageNlpTopology and EqualityStageKktAssembler into the candidate-first backend, and which unsupported layouts must fall back to the stable solver?

## Answer

Use MAKE_CONSTRAINT so fixed initial states remain primal variables and TNLPAdapter appends their scaled identity rows; map those rows to stage-zero path equalities. A native provider maps canonical controls, states, transition rows, and stage-zero fixed-state rows into StageNlpTopology and fills owned Hessian, current-stage dynamics, and path-Jacobian buffers. EqualityStageKktAssembler permutes the complete equality-only KKT into adjacent stage blocks, and StageStructuredCandidateBackend owns factorization, inertia certification, refinement, retries, and reconstruction. Direct bounds, slacks, path inequalities, restoration auxiliaries, a nonidentity next-state dynamics Jacobian, invalid fingerprints/revisions, or any assembly/certification failure must reject transactionally and invoke the stable fallback exactly once. The remaining implementation gap is the live CUTEst provider and lazy canary binding.

## Outcome

- Signal: useful

## Source Nodes

- TNLPAdapter
- LegacyCoordinateModel
- StageNlpTopology
- PreparedStageDerivativeProvider
- EqualityStageKktAssembler
- StageStructuredCandidateBackend
- PrimalDualKktOperator