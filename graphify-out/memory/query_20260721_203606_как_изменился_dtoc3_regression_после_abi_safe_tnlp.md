---
type: "query"
date: "2026-07-21T20:36:06.737932+00:00"
question: "Как изменился DTOC3 regression после ABI-safe TNLPAdapter scratch на повторном нормализованном прогоне?"
contributor: "graphify"
outcome: "corrected"
correction: "Использовать последние JSON samples после canonical libipopt path gate; не фиксировать знак субпроцентной разницы как ускорение."
source_nodes: ["TNLPAdapter", "GetSpaces", "jac_g_", "benchmark", "CUTEst"]
---

# Q: Как изменился DTOC3 regression после ABI-safe TNLPAdapter scratch на повторном нормализованном прогоне?

## Answer

Исходный vector-layout вариант давал +1.02% на DTOC3/5000 и +1.19% на DTOC3/30000. После conditional jac_g_ tail финальный 31-sample interleaved run дал +0.60% и +0.08%; семисэмпловый полный corpus меняет знак между задачами. Значит крупный layout regression устранён, но локальный reuse не даёт доказанного solve-level speedup.

## Outcome

- Signal: corrected
- Correction: Использовать последние JSON samples после canonical libipopt path gate; не фиксировать знак субпроцентной разницы как ускорение.

## Source Nodes

- TNLPAdapter
- GetSpaces
- jac_g_
- benchmark
- CUTEst