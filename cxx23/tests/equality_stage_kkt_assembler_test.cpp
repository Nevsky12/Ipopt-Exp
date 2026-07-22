// Copyright (C) 2026 Ipopt contributors.
// All Rights Reserved.
// This file is distributed under the Eclipse Public License.

#include <ipopt/cxx23/equality_stage_kkt_assembler.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace
{
using namespace Ipopt::Cxx23;

void Check(bool condition, std::string_view message)
{
   if( !condition )
   {
      throw std::runtime_error(std::string(message));
   }
}

void CheckNear(Number actual, Number expected, std::string_view message)
{
   if( std::abs(actual - expected) > 3e-11 )
   {
      throw std::runtime_error(
         std::string(message) + ": actual=" + std::to_string(actual) +
         " expected=" + std::to_string(expected));
   }
}

struct EqualityOcpModel
{
   NlpStructure structure() const
   {
      return {3, 3, 6, 4, 61};
   }

   EvaluationValue<Number> eval_objective(std::span<const Number>)
   {
      return 0.;
   }

   EvaluationResult eval_gradient(std::span<const Number>, std::span<Number> result)
   {
      std::ranges::fill(result, 0.);
      return {};
   }

   EvaluationResult eval_constraints(
      std::span<const Number> x,
      std::span<Number>       result
   )
   {
      result[0] = -x[0] + 2. * x[1];
      result[1] = -.5 * x[0] + 1.5 * x[1] - x[2];
      result[2] = 3. * x[2];
      return {};
   }

   EvaluationResult eval_jacobian_structure(
      std::span<Index> rows,
      std::span<Index> columns
   )
   {
      rows[0] = 0;
      columns[0] = 0;
      rows[1] = 0;
      columns[1] = 1;
      rows[2] = 1;
      columns[2] = 0;
      rows[3] = 1;
      columns[3] = 1;
      rows[4] = 1;
      columns[4] = 2;
      rows[5] = 2;
      columns[5] = 2;
      return {};
   }

   EvaluationResult eval_jacobian_values(
      std::span<const Number>,
      std::span<Number> values
   )
   {
      const std::array<Number, 6> jacobian{{-1., 2., -.5, 1.5, -1., 3.}};
      std::ranges::copy(jacobian, values.begin());
      return {};
   }

   EvaluationResult eval_hessian_structure(
      std::span<Index> rows,
      std::span<Index> columns
   )
   {
      rows[0] = 0;
      columns[0] = 0;
      rows[1] = 1;
      columns[1] = 0;
      rows[2] = 1;
      columns[2] = 1;
      rows[3] = 2;
      columns[3] = 2;
      return {};
   }

   EvaluationResult eval_hessian_values(
      std::span<const Number>,
      Number objective_factor,
      std::span<const Number>,
      std::span<Number> values
   )
   {
      values[0] = 3. * objective_factor;
      values[1] = objective_factor;
      values[2] = 4. * objective_factor;
      values[3] = 2. * objective_factor;
      return {};
   }

   EvaluationResult eval_jacobian_product(
      std::span<const Number>,
      std::span<const Number> direction,
      std::span<Number>       result
   )
   {
      result[0] = -direction[0] + 2. * direction[1];
      result[1] = -.5 * direction[0] + 1.5 * direction[1] - direction[2];
      result[2] = 3. * direction[2];
      return {};
   }

   EvaluationResult eval_jacobian_transpose_product(
      std::span<const Number>,
      std::span<const Number> direction,
      std::span<Number>       result
   )
   {
      result[0] = -direction[0] - .5 * direction[1];
      result[1] = 2. * direction[0] + 1.5 * direction[1];
      result[2] = -direction[1] + 3. * direction[2];
      return {};
   }

   EvaluationResult eval_hessian_product(
      std::span<const Number>,
      Number objective_factor,
      std::span<const Number>,
      std::span<const Number> direction,
      std::span<Number>       result
   )
   {
      result[0] = objective_factor * (3. * direction[0] + direction[1]);
      result[1] = objective_factor * (direction[0] + 4. * direction[1]);
      result[2] = objective_factor * 2. * direction[2];
      return {};
   }
};

struct StageProviderState
{
   Index calls = 0;
   bool fail = false;
};

class EqualityOcpStageProvider
{
public:
   EqualityOcpStageProvider(
      StageNlpTopology topology,
      std::shared_ptr<StageProviderState> state
   )
      : topology_(std::move(topology)),
        state_(std::move(state))
   {
   }

   const StageNlpTopology& stage_nlp_topology() const noexcept
   {
      return topology_;
   }

   EvaluationResult eval_stage_derivatives(
      KktEvaluationPoint,
      StageDerivativeBuffers buffers
   )
   {
      ++state_->calls;
      if( state_->fail )
      {
         return std::unexpected(EvaluationError{
            EvaluationErrorCode::model_failure,
            "injected equality OCP derivative failure"
         });
      }
      const std::array<Number, 5> hessian{{4., 1., 1., 3., 2.}};
      const std::array<Number, 2> dynamics{{1.5, -.5}};
      const std::array<Number, 3> path_equalities{{2., -1., 3.}};
      std::ranges::copy(hessian, buffers.stage_hessians.begin());
      std::ranges::copy(
         dynamics, buffers.dynamics_jacobians_transposed.begin());
      std::ranges::copy(
         path_equalities,
         buffers.path_equality_jacobians_transposed.begin());
      return {};
   }

private:
   StageNlpTopology topology_;
   std::shared_ptr<StageProviderState> state_;
};

struct DtocStyleModel
{
   NlpStructure structure() const
   {
      return {2, 2, 3, 2, 73};
   }

   EvaluationValue<Number> eval_objective(std::span<const Number>)
   {
      return 0.;
   }

   EvaluationResult eval_gradient(
      std::span<const Number>, std::span<Number> result)
   {
      std::ranges::fill(result, 0.);
      return {};
   }

   EvaluationResult eval_constraints(
      std::span<const Number> x, std::span<Number> result)
   {
      result[0] = x[0];
      result[1] = 1.5 * x[0] - x[1];
      return {};
   }

   EvaluationResult eval_jacobian_structure(
      std::span<Index> rows, std::span<Index> columns)
   {
      const std::array<Index, 3> source_rows{{0, 1, 1}};
      const std::array<Index, 3> source_columns{{0, 0, 1}};
      std::ranges::copy(source_rows, rows.begin());
      std::ranges::copy(source_columns, columns.begin());
      return {};
   }

   EvaluationResult eval_jacobian_values(
      std::span<const Number>, std::span<Number> values)
   {
      const std::array<Number, 3> jacobian{{1., 1.5, -1.}};
      std::ranges::copy(jacobian, values.begin());
      return {};
   }

   EvaluationResult eval_hessian_structure(
      std::span<Index> rows, std::span<Index> columns)
   {
      rows[0] = 0;
      columns[0] = 0;
      rows[1] = 1;
      columns[1] = 1;
      return {};
   }

   EvaluationResult eval_hessian_values(
      std::span<const Number>, Number objective_factor,
      std::span<const Number>, std::span<Number> values)
   {
      values[0] = 0.;
      values[1] = 2. * objective_factor;
      return {};
   }

   EvaluationResult eval_jacobian_product(
      std::span<const Number>, std::span<const Number> direction,
      std::span<Number> result)
   {
      result[0] = direction[0];
      result[1] = 1.5 * direction[0] - direction[1];
      return {};
   }

   EvaluationResult eval_jacobian_transpose_product(
      std::span<const Number>, std::span<const Number> direction,
      std::span<Number> result)
   {
      result[0] = direction[0] + 1.5 * direction[1];
      result[1] = -direction[1];
      return {};
   }

   EvaluationResult eval_hessian_product(
      std::span<const Number>, Number objective_factor,
      std::span<const Number>, std::span<const Number> direction,
      std::span<Number> result)
   {
      result[0] = 0.;
      result[1] = 2. * objective_factor * direction[1];
      return {};
   }
};

class DtocStyleStageProvider
{
public:
   DtocStyleStageProvider()
      : topology_(
           {2, 2, 3, 2, 73},
           {{0, 1, 1, 0}, {0, 1, 0, 0}},
           {
              .primal = {0, 1},
              .dynamics = {1},
              .path_equalities = {0},
              .path_inequalities = {}
           },
           81)
   {
   }

   const StageNlpTopology& stage_nlp_topology() const noexcept
   {
      return topology_;
   }

   EvaluationResult eval_stage_derivatives(
      KktEvaluationPoint, StageDerivativeBuffers buffers)
   {
      const std::array<Number, 2> hessian{{0., 2.}};
      std::ranges::copy(hessian, buffers.stage_hessians.begin());
      buffers.dynamics_jacobians_transposed[0] = 1.5;
      buffers.path_equality_jacobians_transposed[0] = 1.;
      return {};
   }

private:
   StageNlpTopology topology_;
};

StageNlpTopology MakeTopology(bool with_inequality = false)
{
   return StageNlpTopology(
      with_inequality
         ? NlpStructure{3, 4, 0, 0, 61}
         : NlpStructure{3, 3, 6, 4, 61},
      {
         {1, 1, 1, with_inequality ? 1U : 0U},
         {0, 1, 1, 0}
      },
      with_inequality
         ? StageNlpOrdering{
              .primal = {1, 0, 2},
              .dynamics = {1},
              .path_equalities = {0, 2},
              .path_inequalities = {3}}
         : StageNlpOrdering{
              .primal = {1, 0, 2},
              .dynamics = {1},
              .path_equalities = {0, 2},
              .path_inequalities = {}},
      71);
}

PrimalDualKktOperator MakeKkt(bool with_bound = false)
{
   return PrimalDualKktOperator(
      MakeNlpProblem(EqualityOcpModel{}),
      {
         .equality_constraints = {2, 0, 1},
         .inequality_constraints = {},
         .primal_lower_bounds = with_bound ? std::vector<Index>{0} : std::vector<Index>{},
         .primal_upper_bounds = {},
         .slack_lower_bounds = {},
         .slack_upper_bounds = {}
      });
}

PrimalDualState State(
   const std::array<Number, 3>& x,
   const std::array<Number, 3>& multipliers,
   std::uint64_t                numeric_revision
)
{
   return {
      .nlp = {x, 1., multipliers},
      .z_lower = {},
      .z_upper = {},
      .v_lower = {},
      .v_upper = {},
      .slack_x_lower = {},
      .slack_x_upper = {},
      .slack_s_lower = {},
      .slack_s_upper = {},
      .regularization = {.2, 0., .1, 0.},
      .numeric_revision = numeric_revision
   };
}

void CheckTrueResidual(
   PrimalDualKktOperator&           kkt,
   PrimalDualState                  state,
   const CandidateFirstSolveResult& result,
   std::span<const Number>          rhs
)
{
   state.regularization = result.accepted_regularization;
   std::array<Number, 6> applied{};
   Check(kkt.apply_flat(state, result.direction, applied).has_value(),
      "equality-stage direction could not be applied");
   for( Index row = 0; row < applied.size(); ++row )
   {
      CheckNear(applied[row], rhs[row], "equality-stage direction fails true KKT");
   }
}

void TestEqualityOcpAssemblyAndCache()
{
   PrimalDualKktOperator kkt = MakeKkt();
   Check(kkt.valid(), "equality OCP KKT layout is invalid");
   const auto provider_state = std::make_shared<StageProviderState>();
   EqualityStageKktAssembler assembler(
      EqualityOcpStageProvider(MakeTopology(), provider_state), kkt);
   const StageStructuredLayout layout = assembler.stage_structured_layout();
   Check(
      layout.block_sizes == std::vector<Index>({3, 3}) &&
         layout.full_direction_dimension == 6 &&
         layout.full_direction_overwrite_certified,
      "equality OCP produced the wrong block topology");
   const std::array<Index, 6> expected_permutation{{1, 0, 4, 2, 3, 5}};
   Check(
      std::ranges::equal(
         assembler.structured_to_full_permutation(), expected_permutation),
      "equality OCP produced the wrong full-KKT permutation");

   StageStructuredCandidateBackend backend(std::move(assembler));
   const std::array<Number, 3> x{{0., 0., 0.}};
   const std::array<Number, 3> multipliers{{0., 0., 0.}};
   const PrimalDualState state = State(x, multipliers, 9);
   const std::array<Number, 6> rhs{{1., -2., 3., 4., -1., .5}};
   const auto first = backend.solve({kkt, state, rhs, 3, false});
   Check(first.has_value(), "equality OCP stage solve failed");
   Check(
      first->work.factorizations == 1 && first->work.backsolves == 1 &&
         first->work.derivative_product_requests == 1 &&
         first->inertia.negative_eigenvalues == 3,
      "equality OCP work/inertia accounting is wrong");
   CheckTrueResidual(kkt, state, *first, rhs);

   const std::array<Number, 6> second_rhs{{-.5, 1., 2., -3., 4., 1.5}};
   const auto second = backend.solve({kkt, state, second_rhs, 3, false});
   Check(second.has_value(), "cached equality OCP stage solve failed");
   Check(
      second->work.derivative_product_requests == 0 && provider_state->calls == 1,
      "same numeric revision reevaluated stage derivatives");
   CheckTrueResidual(kkt, state, *second, second_rhs);

   PrimalDualState revised = state;
   revised.numeric_revision = 10;
   const auto third = backend.solve({kkt, revised, rhs, 3, false});
   Check(third.has_value(), "revised equality OCP stage solve failed");
   Check(
      third->work.derivative_product_requests == 1 && provider_state->calls == 2,
      "changed numeric revision reused stale stage derivatives");
}

void TestReverseStageOrder()
{
   PrimalDualKktOperator kkt = MakeKkt();
   StageNlpTopology topology = MakeTopology();
   EqualityStageKktLayoutWorkspace workspace(
      topology, {2, 0, 1}, {.reverse_stage_order = true});
   EqualityStageKktAssembler assembler(
      EqualityOcpStageProvider(
         std::move(topology), std::make_shared<StageProviderState>()),
      kkt, std::move(workspace));
   Check(assembler.configured(), "reverse equality OCP assembly was rejected");
   const std::array<Index, 6> expected_permutation{{2, 3, 5, 1, 0, 4}};
   Check(
      std::ranges::equal(
         assembler.structured_to_full_permutation(), expected_permutation),
      "reverse equality OCP produced the wrong full-KKT permutation");

   StageStructuredCandidateBackend backend(std::move(assembler));
   const std::array<Number, 3> x{{0., 0., 0.}};
   const std::array<Number, 3> multipliers{{0., 0., 0.}};
   const PrimalDualState state = State(x, multipliers, 11);
   const std::array<Number, 6> rhs{{1., -2., 3., 4., -1., .5}};
   const auto solved = backend.solve({kkt, state, rhs, 3, false});
   Check(solved.has_value(), "reverse equality OCP stage solve failed");
   CheckTrueResidual(kkt, state, *solved, rhs);
}

void TestUnregularizedDtocInertiaProof()
{
   PrimalDualKktOperator kkt(
      MakeNlpProblem(DtocStyleModel{}),
      {
         .equality_constraints = {0, 1},
         .inequality_constraints = {},
         .primal_lower_bounds = {},
         .primal_upper_bounds = {},
         .slack_lower_bounds = {},
         .slack_upper_bounds = {}
      });
   EqualityStageKktAssembler assembler(DtocStyleStageProvider{}, kkt);
   Check(assembler.configured(), "DTOC-style equality assembler was rejected");
   StageStructuredCandidateOptions options;
   options.factorization.require_certified_inertia = false;
   options.maximum_factorization_attempts = 1;
   StageStructuredCandidateBackend backend(std::move(assembler), options);

   const std::array<Number, 2> x{{0., 0.}};
   const std::array<Number, 2> multipliers{{0., 0.}};
   const PrimalDualState state{
      .nlp = {x, 1., multipliers},
      .z_lower = {},
      .z_upper = {},
      .v_lower = {},
      .v_upper = {},
      .slack_x_lower = {},
      .slack_x_upper = {},
      .slack_s_lower = {},
      .slack_s_upper = {},
      .regularization = {},
      .numeric_revision = 1
   };
   const std::array<Number, 4> rhs{{1., -2., .5, 3.}};
   const auto solved = backend.solve({kkt, state, rhs, 2, false});
   Check(
      solved.has_value() &&
         solved->inertia.certainty == CandidateFirstInertiaCertainty::exact &&
         solved->inertia.negative_eigenvalues == 2,
      "DTOC-style structural inertia proof did not promote numeric factors");
   std::array<Number, 4> applied{};
   Check(
      kkt.apply_flat(state, solved->direction, applied).has_value(),
      "DTOC-style accepted direction could not be applied");
   for( Index row = 0; row < rhs.size(); ++row )
   {
      CheckNear(applied[row], rhs[row], "DTOC-style direction fails true KKT");
   }
}

void TestRejectedScopesAndProviderFailure()
{
   PrimalDualKktOperator kkt = MakeKkt();
   const auto failing_state = std::make_shared<StageProviderState>();
   failing_state->fail = true;
   EqualityStageKktAssembler failing_assembler(
      EqualityOcpStageProvider(MakeTopology(), failing_state), kkt);
   StageStructuredCandidateBackend failing_backend(std::move(failing_assembler));
   const std::array<Number, 3> x{};
   const std::array<Number, 3> multipliers{};
   const std::array<Number, 6> rhs{};
   const auto failure = failing_backend.solve({kkt, State(x, multipliers, 1), rhs, 3, false});
   Check(
      !failure.has_value() && failure.error().code == EvaluationErrorCode::model_failure,
      "equality OCP derivative failure was swallowed");

   PrimalDualKktOperator bounded_kkt = MakeKkt(true);
   EqualityStageKktAssembler bounded(
      EqualityOcpStageProvider(
         MakeTopology(), std::make_shared<StageProviderState>()),
      bounded_kkt);
   Check(
      bounded.stage_structured_layout().block_sizes.empty(),
      "bound-constrained KKT was accepted by equality-only assembler");

   EqualityStageKktAssembler inequality(
      EqualityOcpStageProvider(
         MakeTopology(true), std::make_shared<StageProviderState>()),
      kkt);
   Check(
      inequality.stage_structured_layout().block_sizes.empty(),
      "path inequality was accepted by equality-only assembler");

   StageNlpTopology prepared_topology = MakeTopology();
   EqualityStageKktLayoutWorkspace wrong_order(
      prepared_topology, {0, 1, 2});
   EqualityStageKktAssembler prepared_mismatch(
      EqualityOcpStageProvider(
         std::move(prepared_topology),
         std::make_shared<StageProviderState>()),
      kkt, std::move(wrong_order));
   Check(
      !prepared_mismatch.configured(),
      "prepared equality layout with a different live ordering was accepted");
}
} // namespace

int main()
{
   try
   {
      TestEqualityOcpAssemblyAndCache();
      TestReverseStageOrder();
      TestUnregularizedDtocInertiaProof();
      TestRejectedScopesAndProviderFailure();
   }
   catch( const std::exception& exception )
   {
      std::cerr << exception.what() << '\n';
      return 1;
   }
   return 0;
}
