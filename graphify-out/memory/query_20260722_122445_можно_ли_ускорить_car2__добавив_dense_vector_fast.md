---
type: "query"
date: "2026-07-22T12:24:45.222183+00:00"
question: "Можно ли ускорить CAR2, добавив dense-vector fast path для копирования и проверки finite в legacy canary?"
contributor: "graphify"
outcome: "dead_end"
source_nodes: ["cxx23_include_ipopt_cxx23_legacy_algorithm_canary_copyvectorinto", "cxx23_include_ipopt_cxx23_legacy_algorithm_canary_copyvectorsinto", "cxx23_include_ipopt_cxx23_legacy_algorithm_canary_scaledirectionandmeasureresidual", "cxx23_include_ipopt_cxx23_legacy_algorithm_canary_legacyalgorithmcanarypdsystemsolver_flatteninto"]
---

# Q: Можно ли ускорить CAR2, добавив dense-vector fast path для копирования и проверки finite в legacy canary?

## Answer

Expanded from original query via vocab: [dense, vector, copy, finite, values, homogeneous, triplet, stable, workspace, fallback, legacy, span]. The dense-only shortcut changed the accepted CAR2 Callgrind total by just -0.01625%, while adding concrete-type coupling and a compound-vector fallback. That is below the project acceptance threshold, so the complete experiment was reverted. Keep the fused ScaleDirectionAndMeasureResidual change; do not recreate this dense-copy specialization without new evidence.

## Outcome

- Signal: dead_end

## Source Nodes

- cxx23_include_ipopt_cxx23_legacy_algorithm_canary_copyvectorinto
- cxx23_include_ipopt_cxx23_legacy_algorithm_canary_copyvectorsinto
- cxx23_include_ipopt_cxx23_legacy_algorithm_canary_scaledirectionandmeasureresidual
- cxx23_include_ipopt_cxx23_legacy_algorithm_canary_legacyalgorithmcanarypdsystemsolver_flatteninto