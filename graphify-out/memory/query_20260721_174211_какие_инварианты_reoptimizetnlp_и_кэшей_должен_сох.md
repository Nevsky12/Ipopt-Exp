---
type: "query"
date: "2026-07-21T17:42:11.006342+00:00"
question: "Какие инварианты ReOptimizeTNLP и кэшей должен сохранять новый SolveSession?"
contributor: "graphify"
outcome: "useful"
source_nodes: ["IpoptApplication::OptimizeTNLP()", "IpoptApplication::ReOptimizeTNLP()", "OrigIpoptNLP::InitializeStructures()", "CachedResults<T>::InvalidateResult()", "IpoptCalculatedQuantities()"]
---

# Q: Какие инварианты ReOptimizeTNLP и кэшей должен сохранять новый SolveSession?

## Answer

Expanded from original query via graph vocab: [adapter, application, cache, cached, calculated, data, initialize, ipopt, optimize, solver, structure, tnlp]. IpoptApplication::OptimizeTNLP constructs the TNLPAdapter and full algorithm objects, while ReOptimizeTNLP requires the same TNLP object and calls the existing algorithm again. call_optimize reinitializes the algorithm each run. OrigIpoptNLP::Initialize clears structural caches unless warm_start_same_structure is set, but always invalidates dummy-dependency entries; InitializeStructures skips GetSpaces only for same-structure warm starts and still refreshes bound data. IpoptCalculatedQuantities retains scratch objects only for same-structure warm starts. Therefore the C++23 SolveSession should gate Krylov workspace reuse on an explicit structural fingerprint, accept fresh numeric state each solve, and reject mismatched layouts before invoking numeric callbacks.

## Outcome

- Signal: useful

## Source Nodes

- IpoptApplication::OptimizeTNLP()
- IpoptApplication::ReOptimizeTNLP()
- OrigIpoptNLP::InitializeStructures()
- CachedResults<T>::InvalidateResult()
- IpoptCalculatedQuantities()