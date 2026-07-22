---
type: "query"
date: "2026-07-22T09:06:12.659800+00:00"
question: "Где в C++23/AnyAny пути KKT безопасно слить J*v и J^T*w без потери транзакционности и fallback-семантики?"
contributor: "graphify"
outcome: "useful"
source_nodes: ["NlpKktOperator", ".nlp_jacobian_products()", ".EvaluateJacobianProducts()", ".MaterializedJacobianProducts()", ".eval_jacobian_products()"]
---

# Q: Где в C++23/AnyAny пути KKT безопасно слить J*v и J^T*w без потери транзакционности и fallback-семантики?

## Answer

Expanded from the graph vocabulary via [kkt, jacobian, products, transpose, fused, materialized, anyany, nlp, operator, legacy, coordinate, apply]. NlpKktOperator::apply in cxx23/include/ipopt/cxx23/kkt_operator.hpp:L127 now calls .nlp_jacobian_products(). NlpModelAdapter in any_nlp_problem.hpp:L463 validates dimensions, rejects overlapping outputs, preserves every aliased input, computes into two scratch buffers, and commits only after success. .EvaluateJacobianProducts() selects a native fused callback, .MaterializedJacobianProducts() evaluates one Jacobian snapshot and visits the sparse entries once, and the mixed path preserves transpose-then-forward dispatch. LegacyCoordinateModel::eval_jacobian_products() in legacy_coordinate_adapter.hpp:L320 expands both directions once and maps both outputs including fixed equalities. Focused AnyAny and legacy tests cover native, materialized, mixed, aliasing, overlap rejection, and transactional failures. Matched CAR2 Callgrind accepted the path at 21,825,555,191 instructions versus 21,947,087,407, with the same 171/171 candidate acceptance and zero fallback.

## Outcome

- Signal: useful

## Source Nodes

- NlpKktOperator
- .nlp_jacobian_products()
- .EvaluateJacobianProducts()
- .MaterializedJacobianProducts()
- .eval_jacobian_products()