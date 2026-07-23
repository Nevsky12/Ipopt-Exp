// Copyright (C) 2026 Ipopt contributors.
// All Rights Reserved.
// This file is distributed under the Eclipse Public License.

#include <ipopt/cxx23/primal_dual_stage_kkt_assembler.hpp>
#include <ipopt/cxx23/restoration_stage_nlp.hpp>
#include <ipopt/cxx23/sparse_stage_derivative_provider.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

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
   const Number scale = std::max({Number{1.}, std::abs(actual), std::abs(expected)});
   if( std::abs(actual - expected) > 2e-10 * scale )
   {
      throw std::runtime_error(
         std::string(message) + ": actual=" + std::to_string(actual) +
         " expected=" + std::to_string(expected));
   }
}

struct FullStageModel
{
   NlpStructure structure() const
   {
      return {3, 3, 7, 3, 301};
   }

   EvaluationValue<Number> eval_objective(std::span<const Number>)
   {
      return 0.;
   }

   EvaluationResult eval_gradient(
      std::span<const Number>,
      std::span<Number> result
   )
   {
      std::ranges::fill(result, 0.);
      return {};
   }

   EvaluationResult eval_constraints(
      std::span<const Number> x,
      std::span<Number>       result
   )
   {
      result[0] = x[1] + .5 * x[0];
      result[1] = .25 * x[1] - x[0];
      result[2] = 1.2 * x[1] + .7 * x[0] - x[2];
      return {};
   }

   EvaluationResult eval_jacobian_structure(
      std::span<Index> rows,
      std::span<Index> columns
   )
   {
      const std::array<Index, 7> expected_rows{{0, 0, 1, 1, 2, 2, 2}};
      const std::array<Index, 7> expected_columns{{1, 0, 1, 0, 1, 0, 2}};
      std::ranges::copy(expected_rows, rows.begin());
      std::ranges::copy(expected_columns, columns.begin());
      return {};
   }

   EvaluationResult eval_jacobian_values(
      std::span<const Number>,
      std::span<Number> values
   )
   {
      const std::array<Number, 7> expected{{1., .5, .25, -1., 1.2, .7, -1.}};
      std::ranges::copy(expected, values.begin());
      return {};
   }

   EvaluationResult eval_hessian_structure(
      std::span<Index> rows,
      std::span<Index> columns
   )
   {
      const std::array<Index, 3> expected{{0, 1, 2}};
      std::ranges::copy(expected, rows.begin());
      std::ranges::copy(expected, columns.begin());
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
      values[1] = 4. * objective_factor;
      values[2] = 2. * objective_factor;
      return {};
   }

   EvaluationResult eval_jacobian_product(
      std::span<const Number>,
      std::span<const Number> direction,
      std::span<Number>       result
   )
   {
      result[0] = direction[1] + .5 * direction[0];
      result[1] = .25 * direction[1] - direction[0];
      result[2] = 1.2 * direction[1] + .7 * direction[0] - direction[2];
      return {};
   }

   EvaluationResult eval_jacobian_transpose_product(
      std::span<const Number>,
      std::span<const Number> direction,
      std::span<Number>       result
   )
   {
      result[0] = .5 * direction[0] - direction[1] + .7 * direction[2];
      result[1] = direction[0] + .25 * direction[1] + 1.2 * direction[2];
      result[2] = -direction[2];
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
      result[0] = 3. * objective_factor * direction[0];
      result[1] = 4. * objective_factor * direction[1];
      result[2] = 2. * objective_factor * direction[2];
      return {};
   }
};

struct ProviderState
{
   Index calls = 0;
};

StageNlpTopology MakeTopology()
{
   return StageNlpTopology(
      {3, 3, 7, 3, 301},
      {
         {1, 1, 1, 1},
         {0, 1, 0, 0}
      },
      {
         .primal = {1, 0, 2},
         .dynamics = {2},
         .path_equalities = {0},
         .path_inequalities = {1}
      },
      401);
}

class FullStageProvider
{
public:
   explicit FullStageProvider(std::shared_ptr<ProviderState> state)
      : topology_(MakeTopology()), state_(std::move(state))
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
      const std::array<Number, 5> hessian{{4., 0., 0., 3., 2.}};
      const std::array<Number, 2> dynamics{{1.2, .7}};
      const std::array<Number, 2> path_equality{{1., .5}};
      const std::array<Number, 2> path_inequality{{.25, -1.}};
      std::ranges::copy(hessian, buffers.stage_hessians.begin());
      std::ranges::copy(
         dynamics, buffers.dynamics_jacobians_transposed.begin());
      std::ranges::copy(
         path_equality,
         buffers.path_equality_jacobians_transposed.begin());
      std::ranges::copy(
         path_inequality,
         buffers.path_inequality_jacobians_transposed.begin());
      return {};
   }

private:
   StageNlpTopology topology_;
   std::shared_ptr<ProviderState> state_;
};

PrimalDualLayout FullLayout()
{
   return {
      .equality_constraints = {2, 0},
      .inequality_constraints = {1},
      .primal_lower_bounds = {0, 2},
      .primal_upper_bounds = {1, 2},
      .slack_lower_bounds = {0},
      .slack_upper_bounds = {0}
   };
}

PrimalDualKktOperator MakeKkt(PrimalDualLayout layout = FullLayout())
{
   return PrimalDualKktOperator(MakeNlpProblem(FullStageModel{}), std::move(layout));
}

struct StateStorage
{
   std::array<Number, 3> x{{.1, -.2, .3}};
   std::array<Number, 3> multipliers{{.2, -.1, .4}};
   std::array<Number, 2> z_lower{{.8, .4}};
   std::array<Number, 2> z_upper{{.6, .3}};
   std::array<Number, 1> v_lower{{.5}};
   std::array<Number, 1> v_upper{{.2}};
   std::array<Number, 2> slack_x_lower{{2., 1.5}};
   std::array<Number, 2> slack_x_upper{{1.2, 2.5}};
   std::array<Number, 1> slack_s_lower{{.9}};
   std::array<Number, 1> slack_s_upper{{1.1}};

   PrimalDualState view(std::uint64_t revision = 1) const
   {
      return {
         .nlp = {x, 1., multipliers},
         .z_lower = z_lower,
         .z_upper = z_upper,
         .v_lower = v_lower,
         .v_upper = v_upper,
         .slack_x_lower = slack_x_lower,
         .slack_x_upper = slack_x_upper,
         .slack_s_lower = slack_s_lower,
         .slack_s_upper = slack_s_upper,
         .regularization = {.2, .3, .1, .15},
         .numeric_revision = revision
      };
   }
};

constexpr std::array<Number, 13> kRhs{{
   1., -2., .5,
   3.,
   -.75, 1.25,
   -1.5,
   .4, -.6,
   .8, -.3,
   .2,
   -.9
}};

void CheckTrueResidual(
   PrimalDualKktOperator&    kkt,
   PrimalDualState           state,
   std::span<const Number>   direction,
   PrimalDualRegularization  regularization,
   std::span<const Number>   rhs
)
{
   state.regularization = regularization;
   std::array<Number, 13> applied{};
   Check(
      kkt.apply_flat(state, direction, applied).has_value(),
      "full stage direction could not be applied");
   for( Index row = 0; row < rhs.size(); ++row )
   {
      CheckNear(applied[row], rhs[row], "full stage direction fails true KKT");
   }
}

void CheckTrueResidual(
   PrimalDualKktOperator&           kkt,
   PrimalDualState                  state,
   const CandidateFirstSolveResult& result,
   std::span<const Number>          rhs
)
{
   CheckTrueResidual(
      kkt, state, result.direction, result.accepted_regularization, rhs);
}

void TestFusedComplementarityArithmetic()
{
   constexpr Number multiplier = 0x1.2db6b8398e3dcp+43;
   constexpr Number primal_direction = 0x1.c9090f9c5cb32p+99;
   constexpr Number complementarity_direction = -0x1.0d52ee589b2b3p+143;
   const Number expected = std::fma(
      multiplier, primal_direction, complementarity_direction);
   volatile Number rounded_product = multiplier * primal_direction;
   const Number unfused = rounded_product + complementarity_direction;
   Check(expected != unfused, "FMA cancellation fixture is not discriminating");

   StateStorage storage;
   storage.z_lower[0] = multiplier;
   storage.slack_x_lower[0] = 1.;
   const PrimalDualState state = storage.view(17);
   PrimalDualKktOperator kkt = MakeKkt();
   std::array<Number, 13> direction{};
   direction[0] = primal_direction;
   direction[7] = complementarity_direction;
   std::array<Number, 13> applied{};
   Check(
      kkt.apply_flat(state, direction, applied).has_value(),
      "FMA complementarity KKT application failed");
   Check(
      applied[7] == expected,
      "full KKT complementarity row lost a fused cancellation");

   const auto provider_state = std::make_shared<ProviderState>();
   PrimalDualStageKktAssembler assembler(
      FullStageProvider(provider_state), kkt);
   std::array<Number, 7> structured_direction{};
   structured_direction[1] = primal_direction;
   std::array<Number, 13> rhs{};
   rhs[7] = -complementarity_direction;
   std::array<Number, 13> reconstructed{};
   const auto reconstruction = assembler.reconstruct_stage_direction(
      {kkt, state, rhs, 3, false},
      {},
      structured_direction,
      reconstructed);
   Check(
      reconstruction.has_value(),
      "FMA complementarity reconstruction failed");
   Check(
      reconstructed[7] == -expected,
      "complementarity reconstruction lost a fused cancellation");
}

void TestFullEightBlockAssemblyAndCache()
{
   PrimalDualKktOperator kkt = MakeKkt();
   Check(kkt.valid(), "full primal-dual KKT is invalid");
   const auto provider_state = std::make_shared<ProviderState>();
   PrimalDualStageKktAssembler assembler(
      FullStageProvider(provider_state), kkt);
   Check(assembler.configured(), "full stage assembler was rejected");
   const StageStructuredLayout layout = assembler.stage_structured_layout();
   Check(
      layout.block_sizes == std::vector<Index>({5, 2}) &&
         layout.inertia_dimension == 7 &&
         layout.full_direction_dimension == 13 &&
         layout.full_direction_overwrite_certified,
      "full stage assembler produced the wrong dimensions");
   const std::array<Index, 7> expected_permutation{{1, 0, 3, 5, 6, 2, 4}};
   Check(
      std::ranges::equal(
         assembler.structured_to_full_permutation(), expected_permutation),
      "full stage assembler produced the wrong reduced permutation");

   StageStructuredCandidateBackend backend(std::move(assembler));
   const StateStorage storage;
   const PrimalDualState state = storage.view(9);
   const auto first = backend.solve({kkt, state, kRhs, 3, false});
   Check(first.has_value(), "full eight-block stage solve failed");
   Check(
      first->inertia.negative_eigenvalues == 3 &&
         first->work.factorizations == 1 &&
         first->work.backsolves == 1 &&
         first->work.derivative_product_requests == 1,
      "full stage work or inertia accounting is wrong");
   CheckTrueResidual(kkt, state, *first, kRhs);

   PrimalDualState factor_state = state;
   factor_state.regularization = first->accepted_regularization;
   std::array<Number, 13> perturbed;
   std::ranges::copy(first->direction, perturbed.begin());
   perturbed[0] += .2;
   perturbed[3] -= .1;
   perturbed[8] += .05;
   const auto refined = backend.refine({
      kkt, factor_state, kRhs, perturbed, false
   });
   Check(
      refined.has_value() && refined->supported && refined->converged &&
         refined->work.factorizations == 0 &&
         refined->work.backsolves > 0 &&
         refined->work.kkt_applications > 0,
      "full eight-block FGMRES did not reuse the stage factor");
   CheckTrueResidual(
      kkt, factor_state, perturbed, first->accepted_regularization, kRhs);

   std::array<Number, 13> second_rhs = kRhs;
   second_rhs[0] = -1.25;
   second_rhs[8] = .7;
   const auto second = backend.solve({kkt, state, second_rhs, 3, false});
   Check(second.has_value(), "cached full stage solve failed");
   Check(
      second->work.derivative_product_requests == 0 &&
         provider_state->calls == 1,
      "unchanged numeric revision reevaluated full stage derivatives");
   CheckTrueResidual(kkt, state, *second, second_rhs);

   const auto revised = backend.solve({kkt, storage.view(10), kRhs, 3, false});
   Check(revised.has_value(), "revised full stage solve failed");
   Check(
      revised->work.derivative_product_requests == 1 &&
         provider_state->calls == 2,
      "changed numeric revision reused full stage derivatives");
}

void TestPreparedReverseNumericProof()
{
   PrimalDualKktOperator kkt = MakeKkt();
   const auto provider_state = std::make_shared<ProviderState>();
   PrimalDualStageKktLayoutWorkspace layout_workspace(
      MakeTopology(), FullLayout(), {.reverse_stage_order = true});
   Check(layout_workspace.configured(), "prepared full layout was rejected");
   StageStructuredCandidateOptions options;
   options.factorization.require_certified_inertia = false;
   options.maximum_factorization_attempts = 1;
   StageStructuredCandidateWorkspace solver_workspace(
      layout_workspace.stage_structured_layout(), options);
   PrimalDualStageKktAssembler assembler(
      FullStageProvider(provider_state), kkt, std::move(layout_workspace));
   Check(assembler.configured(), "prepared reverse full assembler was rejected");
   const std::array<Index, 7> expected_permutation{{2, 4, 1, 0, 3, 5, 6}};
   Check(
      std::ranges::equal(
         assembler.structured_to_full_permutation(), expected_permutation),
      "reverse full stage permutation is wrong");
   StageStructuredCandidateBackend backend(
      std::move(assembler), std::move(solver_workspace));
   const StateStorage storage;
   const PrimalDualState state = storage.view(12);
   const auto solved = backend.solve({kkt, state, kRhs, 3, false});
   Check(
      solved.has_value() &&
         solved->inertia.certainty == CandidateFirstInertiaCertainty::exact &&
         solved->inertia.negative_eigenvalues == 3,
      "numeric reduced factor was not promoted by its exact proof");
   CheckTrueResidual(kkt, state, *solved, kRhs);
}

SparseStageDerivativeProvider MakeDetachedSparseProvider()
{
   PrimalDualKktOperator exemplar = MakeKkt();
   return SparseStageDerivativeProvider(exemplar, MakeTopology());
}

struct StageDerivativeBufferStorage
{
   explicit StageDerivativeBufferStorage(
      StageDerivativeStorage storage,
      Number                 initial
   )
      : stage_hessians(storage.stage_hessians, initial),
        dynamics(storage.dynamics_jacobians_transposed, initial),
        path_equalities(
           storage.path_equality_jacobians_transposed, initial),
        path_inequalities(
           storage.path_inequality_jacobians_transposed, initial)
   {
   }

   StageDerivativeBuffers buffers()
   {
      return {stage_hessians, dynamics, path_equalities, path_inequalities};
   }

   bool operator==(const StageDerivativeBufferStorage&) const = default;

   std::vector<Number> stage_hessians;
   std::vector<Number> dynamics;
   std::vector<Number> path_equalities;
   std::vector<Number> path_inequalities;
};

void TestSparseStageDerivativeScatter()
{
   SparseStageDerivativeProvider provider = MakeDetachedSparseProvider();
   Check(provider.configured(), "sparse stage derivative scatter was rejected");
   PrimalDualKktOperator kkt = MakeKkt();
   const StateStorage storage;
   const PrimalDualState state = storage.view(13);
   const StageDerivativeStorage derivative_storage =
      provider.stage_nlp_topology().derivative_storage();
   StageDerivativeBufferStorage expected_derivatives(derivative_storage, 0.);
   StageDerivativeBufferStorage dirty_derivatives(
      derivative_storage, std::numeric_limits<Number>::quiet_NaN());
   Check(
      provider.eval_stage_derivatives(
         kkt, state.nlp, expected_derivatives.buffers()).has_value() &&
         provider.eval_stage_derivatives(
            kkt, state.nlp, dirty_derivatives.buffers()).has_value() &&
         dirty_derivatives == expected_derivatives,
      "unique sparse stage scatter retained dirty structural values");
   PrimalDualStageKktAssembler assembler(std::move(provider), kkt);
   Check(
      assembler.configured(),
      "sparse stage provider did not bind to an equivalent live KKT");
   StageStructuredCandidateBackend backend(std::move(assembler));
   const auto solved = backend.solve({kkt, state, kRhs, 3, false});
   Check(
      solved.has_value() && solved->work.derivative_product_requests == 1,
      "sparse stage derivative solve failed");
   CheckTrueResidual(kkt, state, *solved, kRhs);
}

void TestRestorationTopologyTransformation()
{
   EvaluationValue<RestorationStageVariableLayout> variables =
      MakeCanonicalRestoIpoptVariableLayout(3, 2, 1);
   Check(variables.has_value(), "canonical restoration variables were rejected");
   PrimalDualLayout restoration_layout{
      .equality_constraints = {1, 0},
      .inequality_constraints = {2},
      .primal_lower_bounds = {8, 0, 3, 6, 4, 7, 5, 2},
      .primal_upper_bounds = {2, 1},
      .slack_lower_bounds = {0},
      .slack_upper_bounds = {0}
   };
   EvaluationValue<StageNlpTopology> transformed =
      MakeRestorationStageNlpTopology(
         MakeTopology(),
         FullLayout(),
         {9, 3, 13, 3, 601},
         restoration_layout,
         *variables,
         602);
   Check(transformed.has_value(), "valid restoration topology was rejected");
   Check(
      std::ranges::equal(
         transformed->stages(),
         std::vector<OptimalControlStageDimensions>({
            {7, 1, 1, 1},
            {0, 1, 0, 0}
         })) &&
         transformed->ordering().primal ==
            std::vector<Index>({1, 4, 6, 7, 8, 3, 5, 0, 2}) &&
         transformed->ordering().dynamics == std::vector<Index>({1}) &&
         transformed->ordering().path_equalities == std::vector<Index>({0}) &&
         transformed->ordering().path_inequalities == std::vector<Index>({2}),
      "restoration variables or rows were assigned to the wrong stage");

   restoration_layout.primal_lower_bounds.pop_back();
   const auto missing_auxiliary_bound = MakeRestorationStageNlpTopology(
      MakeTopology(),
      FullLayout(),
      {9, 3, 13, 3, 601},
      restoration_layout,
      *variables,
      602);
   Check(
      !missing_auxiliary_bound.has_value(),
      "restoration topology accepted an n/p variable without its lower bound");
}

void TestRejectedMetadataAndRuntimeStates()
{
   PrimalDualLayout wrong_roles = FullLayout();
   wrong_roles.equality_constraints = {2, 1};
   wrong_roles.inequality_constraints = {0};
   PrimalDualStageKktLayoutWorkspace role_mismatch(
      MakeTopology(), std::move(wrong_roles));
   Check(
      !role_mismatch.configured(),
      "path equality/inequality role mismatch was accepted");

   PrimalDualKktOperator kkt = MakeKkt();
   PrimalDualLayout different_layout = FullLayout();
   std::ranges::reverse(different_layout.primal_lower_bounds);
   PrimalDualStageKktLayoutWorkspace prepared(
      MakeTopology(), std::move(different_layout));
   Check(prepared.configured(), "different prepared bound order is itself invalid");
   PrimalDualStageKktAssembler mismatch(
      FullStageProvider(std::make_shared<ProviderState>()),
      kkt,
      std::move(prepared));
   Check(
      !mismatch.configured(),
      "prepared bound layout mismatch was accepted by the live KKT");

   PrimalDualStageKktAssembler assembler(
      FullStageProvider(std::make_shared<ProviderState>()), kkt);
   StageStructuredCandidateBackend backend(std::move(assembler));
   StateStorage invalid_storage;
   invalid_storage.slack_x_lower[0] = 0.;
   const auto invalid_slack = backend.solve({
      kkt, invalid_storage.view(20), kRhs, 3, false});
   Check(
      !invalid_slack.has_value() &&
         invalid_slack.error().code == EvaluationErrorCode::nonfinite_output,
      "nonpositive complementarity slack was accepted");

   StateStorage valid_storage;
   const auto restoration = backend.solve({
      kkt, valid_storage.view(21), kRhs, 3, true});
   Check(
      !restoration.has_value() &&
         restoration.error().code == EvaluationErrorCode::invalid_layout,
      "main-problem stage metadata was reused for restoration");
}
} // namespace

int main()
{
   try
   {
      TestFusedComplementarityArithmetic();
      TestFullEightBlockAssemblyAndCache();
      TestPreparedReverseNumericProof();
      TestSparseStageDerivativeScatter();
      TestRestorationTopologyTransformation();
      TestRejectedMetadataAndRuntimeStates();
   }
   catch( const std::exception& exception )
   {
      std::cerr << exception.what() << '\n';
      return 1;
   }
   return 0;
}
