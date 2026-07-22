---
type: "query"
date: "2026-07-21T20:24:05.865845+00:00"
question: "Почему workspace reuse в TNLPAdapter мог дать solve-level regression на DTOC3?"
contributor: "graphify"
outcome: "corrected"
correction: "Проверять class layout и не добавлять scratch members в stable C++ class; переиспользовать существующую owning allocation с conditional tail capacity."
source_nodes: ["TNLPAdapter", "Eval_grad_f", "Eval_h", "GetSpaces", "jac_g_"]
---

# Q: Почему workspace reuse в TNLPAdapter мог дать solve-level regression на DTOC3?

## Answer

Сам reuse path на DTOC3 без удалённых fixed variables не выполнялся. Причиной были два добавленных std::vector поля: они изменили публичный C++ layout TNLPAdapter и cache locality всех последующих полей. Перенос scratch в хвост существующей jac_g_ allocation восстановил layout и выделяет extra storage только при P_x_full_x_. После исправления нормализованный 31-sample regression изменился с +1.02..1.19% на +0.60..+0.08%; знак субпроцентной разницы нельзя трактовать как ускорение.

## Outcome

- Signal: corrected
- Correction: Проверять class layout и не добавлять scratch members в stable C++ class; переиспользовать существующую owning allocation с conditional tail capacity.

## Source Nodes

- TNLPAdapter
- Eval_grad_f
- Eval_h
- GetSpaces
- jac_g_
