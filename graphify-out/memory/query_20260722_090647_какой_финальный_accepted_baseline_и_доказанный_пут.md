---
type: "query"
date: "2026-07-22T09:06:47.455906+00:00"
question: "Какой финальный accepted baseline и доказанный путь fused Jacobian products в C++23 CAR2?"
contributor: "graphify"
outcome: "useful"
source_nodes: ["NlpKktOperator", ".nlp_jacobian_products()", ".EvaluateJacobianProducts()", ".MaterializedJacobianProducts()", ".eval_jacobian_products()"]
---

# Q: Какой финальный accepted baseline и доказанный путь fused Jacobian products в C++23 CAR2?

## Answer

Expanded from the graph vocabulary via [kkt, jacobian, products, fused, anyany, nlp, operator, legacy, coordinate, apply]. The accepted path is NlpKktOperator::apply -> .nlp_jacobian_products() -> NlpModelAdapter::.EvaluateJacobianProducts(), with .MaterializedJacobianProducts() for the one-snapshot fallback and LegacyCoordinateModel::.eval_jacobian_products() for scaled coordinates. Release and ASan/UBSan/leak tests pass 15/15; CAR2 passes 171/171 with zero fallback; DTOC3 N=5000 and N=30000 pass 1/1. Matched x86-64-v3 Callgrind is 21,825,555,191 instructions, down 121,532,216 or 0.5538% from the unique-scatter baseline, with unchanged nonlinear trajectory.

## Outcome

- Signal: useful

## Source Nodes

- NlpKktOperator
- .nlp_jacobian_products()
- .EvaluateJacobianProducts()
- .MaterializedJacobianProducts()
- .eval_jacobian_products()