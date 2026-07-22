---
type: "query"
date: "2026-07-22T06:57:16.970115+00:00"
question: "Как сократить работу нормализованного bordered-stage assembler на CAR2, сохранив exact symmetry, arbitrary-state fallback, inertia и residual gates?"
contributor: "graphify"
outcome: "useful"
source_nodes: [".assemble_bordered_stage_system()", "PrimalDualBorderedStageKktAssembler", "primal_dual_bordered_stage_kkt_assembler_test.cpp", "SymmetricBlockTridiagonalSolver"]
---

# Q: Как сократить работу нормализованного bordered-stage assembler на CAR2, сохранив exact symmetry, arbitrary-state fallback, inertia и residual gates?

## Answer

Пакет из трех локальных изменений подтвержден: Gram T*T^T считается по одному треугольнику и зеркалируется; в каноническом next-state coupling обновляются только живые диагональные пары -I поверх явно подготовленных структурных нулей; для четырех состояний C++23 fixed-extent dispatch разворачивает Gram и dynamics products, а generic path остается для остальных размеров. Stable native stepwise A/B: Gram assembly -0.5904% при noise-scale total; sparse -I assembly -0.7836%, total -0.3185%; fixed-4 assembly -1.2526% (7/7), backend -0.3336% (6/7), total -0.2577% (6/7). Callgrind x86-64-v3: total 22,710,389,439 -> 22,549,509,195 (-0.7084%); inclusive assembler 2,480,806,287 -> 2,306,436,243 (-7.0288%, 174,370,044 saved). CAR2 сохранил 171/171 accepted, zero fallback, objective 2.66608466558 и violation 1.62e-13; Release и ASan/UBSan прошли 15/15. Query expansion trace: [factorize, stage, numerically, assembler, memory, copy, fill, workspace, schur, performance, profile, residual].

## Outcome

- Signal: useful

## Source Nodes

- .assemble_bordered_stage_system()
- PrimalDualBorderedStageKktAssembler
- primal_dual_bordered_stage_kkt_assembler_test.cpp
- SymmetricBlockTridiagonalSolver