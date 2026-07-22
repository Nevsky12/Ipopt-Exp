---
type: "query"
date: "2026-07-21T20:24:05.893875+00:00"
question: "Как честно сравнивать текущий Ipopt, clean HEAD и FATROP на CUTEst corpus?"
contributor: "graphify"
outcome: "useful"
source_nodes: ["benchmark", "CUTEst", "IpoptApplication", "SolveStatistics", "OptimizeTNLP", "TNLPAdapter"]
---

# Q: Как честно сравнивать текущий Ipopt, clean HEAD и FATROP на CUTEst corpus?

## Answer

Запускать отдельные процессы с ABI-compatible libraries, одинаковыми release flags и dependencies, pin на один CPU, BLAS threads=1, warmup и циклическим interleaving вариантов. Записывать resolved libipopt, raw samples и medians; до timing результата проверять status, deterministic iterations внутри backend, primal dimension, objective и constraint violation. Microbenchmark callback не заменяет solve-level gate.

## Outcome

- Signal: useful

## Source Nodes

- benchmark
- CUTEst
- IpoptApplication
- SolveStatistics
- OptimizeTNLP
- TNLPAdapter