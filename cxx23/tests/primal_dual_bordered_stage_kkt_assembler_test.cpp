// Copyright (C) 2026 Ipopt contributors.
// All Rights Reserved.
// This file is distributed under the Eclipse Public License.

#include <ipopt/cxx23/primal_dual_bordered_stage_kkt_assembler.hpp>
#include <ipopt/cxx23/sparse_bordered_stage_derivative_provider.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
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
   if( std::abs(actual - expected) > 3e-10 * scale )
   {
      throw std::runtime_error(
         std::string(message) + ": actual=" + std::to_string(actual) +
         " expected=" + std::to_string(expected));
   }
}

struct BorderedModel
{
   bool singular_next_state_jacobian = false;

   NlpStructure structure() const
   {
      return {7, 5, 24, 21, 701};
   }

   static constexpr std::array<Index, 24> jacobian_rows{{
      2, 2, 2, 2,
      0, 0, 0, 0,
      3, 3, 3, 3,
      1, 1, 1, 1, 1, 1,
      4, 4, 4, 4, 4, 4
   }};
   static constexpr std::array<Index, 24> jacobian_columns{{
      2, 0, 5, 4,
      2, 0, 5, 4,
      3, 1, 6, 4,
      2, 0, 5, 4, 1, 6,
      2, 0, 5, 4, 1, 6
   }};
   static constexpr std::array<Number, 24> jacobian_values{{
      .7, 1.1, .05, .2,
      -.4, .6, .08, .1,
      .5, -.3, -.07, .15,
      1.2, .8, -.1, .25, -2., .5,
      .3, -.2, .9, -.15, -.25, -1.5
   }};
   static constexpr std::array<Index, 21> hessian_rows{{
      0, 1, 2, 3, 4, 5, 6,
      2, 5, 5, 3, 6, 6,
      4, 4, 4, 4, 5, 6,
      3, 6
   }};
   static constexpr std::array<Index, 21> hessian_columns{{
      0, 1, 2, 3, 4, 5, 6,
      0, 0, 2, 1, 1, 3,
      0, 1, 2, 3, 4, 4,
      0, 5
   }};
   static constexpr std::array<Number, 21> hessian_values{{
      5., 4.5, 4., 3.5, 6., 4.2, 3.8,
      .2, .1, -.08, -.15, .11, -.09,
      .1, -.2, .3, .25, .07, -.06,
      .12, .05
   }};

   EvaluationValue<Number> eval_objective(std::span<const Number> x)
   {
      std::array<Number, 7> product{};
      HessianProduct(x, product);
      Number value = 0.;
      for( Index i = 0; i < x.size(); ++i )
      {
         value += .5 * x[i] * product[i];
      }
      return value;
   }

   EvaluationResult eval_gradient(
      std::span<const Number> x,
      std::span<Number>       result
   )
   {
      HessianProduct(x, result);
      return {};
   }

   EvaluationResult eval_constraints(
      std::span<const Number> x,
      std::span<Number>       result
   )
   {
      std::ranges::fill(result, 0.);
      for( Index entry = 0; entry < jacobian_values.size(); ++entry )
      {
         result[jacobian_rows[entry]] +=
            JacobianValue(entry) * x[jacobian_columns[entry]];
      }
      return {};
   }

   EvaluationResult eval_jacobian_structure(
      std::span<Index> rows,
      std::span<Index> columns
   )
   {
      std::ranges::copy(jacobian_rows, rows.begin());
      std::ranges::copy(jacobian_columns, columns.begin());
      return {};
   }

   EvaluationResult eval_jacobian_values(
      std::span<const Number>,
      std::span<Number> values
   )
   {
      for( Index entry = 0; entry < jacobian_values.size(); ++entry )
      {
         values[entry] = JacobianValue(entry);
      }
      return {};
   }

   EvaluationResult eval_hessian_structure(
      std::span<Index> rows,
      std::span<Index> columns
   )
   {
      std::ranges::copy(hessian_rows, rows.begin());
      std::ranges::copy(hessian_columns, columns.begin());
      return {};
   }

   EvaluationResult eval_hessian_values(
      std::span<const Number>,
      Number                  objective_factor,
      std::span<const Number>,
      std::span<Number>       values
   )
   {
      for( Index entry = 0; entry < hessian_values.size(); ++entry )
      {
         values[entry] = objective_factor * hessian_values[entry];
      }
      return {};
   }

   EvaluationResult eval_jacobian_product(
      std::span<const Number>,
      std::span<const Number> direction,
      std::span<Number>       result
   )
   {
      std::ranges::fill(result, 0.);
      for( Index entry = 0; entry < jacobian_values.size(); ++entry )
      {
         result[jacobian_rows[entry]] +=
            JacobianValue(entry) * direction[jacobian_columns[entry]];
      }
      return {};
   }

   EvaluationResult eval_jacobian_transpose_product(
      std::span<const Number>,
      std::span<const Number> direction,
      std::span<Number>       result
   )
   {
      std::ranges::fill(result, 0.);
      for( Index entry = 0; entry < jacobian_values.size(); ++entry )
      {
         result[jacobian_columns[entry]] +=
            JacobianValue(entry) * direction[jacobian_rows[entry]];
      }
      return {};
   }

   EvaluationResult eval_hessian_product(
      std::span<const Number>,
      Number                  objective_factor,
      std::span<const Number>,
      std::span<const Number> direction,
      std::span<Number>       result
   )
   {
      HessianProduct(direction, result);
      for( Number& value : result )
      {
         value *= objective_factor;
      }
      return {};
   }

private:
   Number JacobianValue(Index entry) const noexcept
   {
      if( singular_next_state_jacobian && entry == 22 )
      {
         return -4.;
      }
      if( singular_next_state_jacobian && entry == 23 )
      {
         return 1.;
      }
      return jacobian_values[entry];
   }

   static void HessianProduct(
      std::span<const Number> input,
      std::span<Number>       output
   )
   {
      std::ranges::fill(output, 0.);
      for( Index entry = 0; entry < hessian_values.size(); ++entry )
      {
         const Index row = hessian_rows[entry];
         const Index column = hessian_columns[entry];
         const Number value = hessian_values[entry];
         output[row] += value * input[column];
         if( row != column )
         {
            output[column] += value * input[row];
         }
      }
   }
};

BorderedStageNlpTopology MakeTopology()
{
   return BorderedStageNlpTopology(
      {7, 5, 24, 21, 701},
      {
         {1, 2, 1, 1},
         {1, 2, 0, 1}
      },
      {
         .primal = {2, 0, 5, 3, 1, 6},
         .dynamics = {1, 4},
         .path_equalities = {2},
         .path_inequalities = {0, 3}
      },
      {4},
      702);
}

PrimalDualLayout FullLayout()
{
   return {
      .equality_constraints = {1, 2, 4},
      .inequality_constraints = {3, 0},
      .primal_lower_bounds = {4, 0, 3},
      .primal_upper_bounds = {2, 4, 1},
      .slack_lower_bounds = {0, 1},
      .slack_upper_bounds = {1}
   };
}

PrimalDualKktOperator MakeKkt(bool singular_next_state_jacobian = false)
{
   return PrimalDualKktOperator(
      MakeNlpProblem(BorderedModel{singular_next_state_jacobian}), FullLayout());
}

SparseBorderedStageDerivativeProvider MakeDetachedProvider()
{
   PrimalDualKktOperator exemplar = MakeKkt();
   return SparseBorderedStageDerivativeProvider(exemplar, MakeTopology());
}

struct BorderedDerivativeBufferStorage
{
   explicit BorderedDerivativeBufferStorage(
      BorderedStageDerivativeStorage storage,
      Number                         initial
   )
      : stage_hessians(storage.stage_hessians, initial),
        cross_stage_hessians(storage.cross_stage_hessians, initial),
        local_global_hessians(storage.local_global_hessians, initial),
        global_hessian(storage.global_hessian, initial),
        dynamics(storage.dynamics_jacobians_transposed, initial),
        next_state(storage.dynamics_next_state_jacobians, initial),
        path_equalities(
           storage.path_equality_jacobians_transposed, initial),
        path_inequalities(
           storage.path_inequality_jacobians_transposed, initial),
        global_jacobians(storage.global_jacobians_transposed, initial)
   {
   }

   BorderedStageDerivativeBuffers buffers()
   {
      return {
         stage_hessians,
         cross_stage_hessians,
         local_global_hessians,
         global_hessian,
         dynamics,
         next_state,
         path_equalities,
         path_inequalities,
         global_jacobians
      };
   }

   bool operator==(const BorderedDerivativeBufferStorage&) const = default;

   std::vector<Number> stage_hessians;
   std::vector<Number> cross_stage_hessians;
   std::vector<Number> local_global_hessians;
   std::vector<Number> global_hessian;
   std::vector<Number> dynamics;
   std::vector<Number> next_state;
   std::vector<Number> path_equalities;
   std::vector<Number> path_inequalities;
   std::vector<Number> global_jacobians;
};

struct StateStorage
{
   std::array<Number, 7> x{{.1, -.2, .3, -.15, .4, .2, -.35}};
   std::array<Number, 5> multipliers{{.2, -.1, .4, .15, -.25}};
   std::array<Number, 3> z_lower{{.7, .5, .4}};
   std::array<Number, 3> z_upper{{.6, .3, .2}};
   std::array<Number, 2> v_lower{{.45, .35}};
   std::array<Number, 1> v_upper{{.25}};
   std::array<Number, 3> slack_x_lower{{1.4, 1.8, 1.2}};
   std::array<Number, 3> slack_x_upper{{1.1, 1.6, 2.}};
   std::array<Number, 2> slack_s_lower{{.9, 1.3}};
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
         .regularization = {0., 0., 0., 0.},
         .numeric_revision = revision
      };
   }
};

struct FourStateNormalizationModel
{
   NlpStructure structure() const
   {
      return {9, 4, 0, 0, 801};
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
      std::span<const Number>,
      std::span<Number> result
   )
   {
      std::ranges::fill(result, 0.);
      return {};
   }

   EvaluationResult eval_jacobian_structure(
      std::span<Index>,
      std::span<Index>
   )
   {
      return {};
   }

   EvaluationResult eval_jacobian_values(
      std::span<const Number>,
      std::span<Number>
   )
   {
      return {};
   }

   EvaluationResult eval_hessian_structure(
      std::span<Index>,
      std::span<Index>
   )
   {
      return {};
   }

   EvaluationResult eval_hessian_values(
      std::span<const Number>,
      Number,
      std::span<const Number>,
      std::span<Number>
   )
   {
      return {};
   }
};

BorderedStageNlpTopology MakeFourStateNormalizationTopology()
{
   return BorderedStageNlpTopology(
      {9, 4, 0, 0, 801},
      {
         {0, 4, 0, 0},
         {0, 4, 0, 0}
      },
      {
         .primal = {0, 1, 2, 3, 4, 5, 6, 7},
         .dynamics = {0, 1, 2, 3},
         .path_equalities = {},
         .path_inequalities = {}
      },
      {8},
      802);
}

struct FourStateNormalizationProvider
{
   bool singular = false;
   bool asymmetric_hessian = false;
   BorderedStageNlpTopology topology = MakeFourStateNormalizationTopology();

   static constexpr std::array<Number, 16> pivoted_matrix{{
      0., 2., .5, -1.,
      3., -.5, 1., .25,
      .2, 1.5, 4., -.75,
      -1., .3, .6, 2.5
   }};

   static constexpr std::array<Number, 16> singular_matrix{{
      0., 2., .5, -1.,
      3., -.5, 1., .25,
      .2, 1.5, 4., -.75,
      0., 2., .5, -1.
   }};

   const BorderedStageNlpTopology&
   bordered_stage_nlp_topology() const noexcept
   {
      return topology;
   }

   EvaluationResult eval_bordered_stage_derivatives(
      KktEvaluationPoint,
      BorderedStageDerivativeBuffers buffers
   )
   {
      std::ranges::fill(buffers.stage_hessians, 0.);
      std::ranges::fill(buffers.cross_stage_hessians, 0.);
      std::ranges::fill(buffers.local_global_hessians, 0.);
      std::ranges::fill(buffers.global_hessian, 0.);
      std::ranges::fill(buffers.dynamics_jacobians_transposed, 0.);
      std::ranges::fill(buffers.path_equality_jacobians_transposed, 0.);
      std::ranges::fill(buffers.path_inequality_jacobians_transposed, 0.);
      std::ranges::fill(buffers.global_jacobians_transposed, 0.);
      if( asymmetric_hessian )
      {
         for( Index stage_offset : {Index{0}, Index{16}} )
         {
            for( Index diagonal = 0; diagonal < 4; ++diagonal )
            {
               buffers.stage_hessians[
                  stage_offset + diagonal * 4 + diagonal] = 2.;
            }
         }
         buffers.stage_hessians[1] = .1;
         buffers.stage_hessians[4] = .2;
         buffers.global_hessian[0] = 2.;
      }
      for( Index state = 0; state < 4; ++state )
      {
         buffers.dynamics_jacobians_transposed[state * 4 + state] = 1.;
      }
      const auto& matrix = singular ? singular_matrix : pivoted_matrix;
      std::ranges::copy(matrix, buffers.dynamics_next_state_jacobians.begin());
      return {};
   }
};

PrimalDualKktOperator MakeFourStateNormalizationKkt()
{
   return PrimalDualKktOperator(
      MakeNlpProblem(FourStateNormalizationModel{}),
      {
         .equality_constraints = {0, 1, 2, 3},
         .inequality_constraints = {},
         .primal_lower_bounds = {},
         .primal_upper_bounds = {},
         .slack_lower_bounds = {},
         .slack_upper_bounds = {}
      });
}

struct FourStateNormalizationStateStorage
{
   std::array<Number, 9> x{};
   std::array<Number, 4> multipliers{};

   PrimalDualState view(std::uint64_t revision) const
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
         .regularization = {0., 0., 0., 0.},
         .numeric_revision = revision
      };
   }
};

constexpr std::array<Number, 23> kRhs{{
   1., -.5, .75, -1.2, .4, .65, -.8,
   .6, -.3,
   .2, -.7, .45,
   1.1, -.9,
   .35, -.45, .55,
   -.2, .3, -.4,
   .25, -.15,
   .5
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
   std::array<Number, 23> applied{};
   Check(
      kkt.apply_flat(state, direction, applied).has_value(),
      "bordered direction could not be applied to the full KKT");
   for( Index row = 0; row < rhs.size(); ++row )
   {
      CheckNear(applied[row], rhs[row], "bordered direction fails full KKT");
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

void TestSparseDerivativeScatterPlan()
{
   detail::SparseDerivativeScatterPlan unique;
   unique.push_back({0, 1});
   unique.push_back({1, 3});
   Check(
      unique.prepare(5) && unique.unique_targets(),
      "unique sparse derivative scatter plan was not recognized");
   const std::array<Number, 2> unique_source{{1.25, -Number{0.}}};
   std::array<Number, 5> unique_target;
   std::ranges::fill(
      unique_target, std::numeric_limits<Number>::quiet_NaN());
   unique.write(unique_source, unique_target);
   Number reference_zero = 0.;
   reference_zero += -Number{0.};
   Check(
      unique_target[0] == 0. && unique_target[1] == 1.25 &&
         unique_target[2] == 0. && unique_target[3] == 0. &&
         std::signbit(unique_target[3]) == std::signbit(reference_zero) &&
         unique_target[4] == 0.,
      "unique sparse derivative scatter did not overwrite dirty storage");

   detail::SparseDerivativeScatterPlan duplicate;
   duplicate.push_back({0, 1});
   duplicate.push_back({1, 1});
   Check(
      duplicate.prepare(3) && !duplicate.unique_targets(),
      "duplicate sparse derivative target was treated as unique");
   const std::array<Number, 2> duplicate_source{{1.25, -.25}};
   std::array<Number, 3> duplicate_target;
   std::ranges::fill(
      duplicate_target, std::numeric_limits<Number>::quiet_NaN());
   duplicate.write(duplicate_source, duplicate_target);
   Check(
      duplicate_target == std::array<Number, 3>{{0., 1., 0.}},
      "duplicate sparse derivative entries were not accumulated");

   detail::SparseDerivativeScatterPlan invalid;
   invalid.push_back({0, 2});
   Check(
      !invalid.prepare(2),
      "out-of-range sparse derivative target was accepted");
}

void TestSparseBorderedFullDirection()
{
   BorderedStageNlpTopology topology = MakeTopology();
   Check(topology.configured(), "valid bordered topology was rejected");
   const BorderedStageDerivativeStorage storage = topology.derivative_storage();
   Check(
      storage.stage_hessians == 18 &&
         storage.cross_stage_hessians == 9 &&
         storage.local_global_hessians == 6 &&
         storage.global_hessian == 1 &&
         storage.dynamics_jacobians_transposed == 6 &&
         storage.dynamics_next_state_jacobians == 4 &&
         storage.path_equality_jacobians_transposed == 3 &&
         storage.path_inequality_jacobians_transposed == 6 &&
         storage.global_jacobians_transposed == 5,
      "bordered derivative storage profile is wrong");

   SparseBorderedStageDerivativeProvider provider = MakeDetachedProvider();
   Check(provider.configured(), "valid sparse bordered scatter was rejected");
   PrimalDualKktOperator kkt = MakeKkt();
   const StateStorage state_storage;
   const PrimalDualState state = state_storage.view(17);
   const BorderedStageDerivativeStorage derivative_storage =
      provider.bordered_stage_nlp_topology().derivative_storage();
   BorderedDerivativeBufferStorage expected_derivatives(
      derivative_storage, 0.);
   BorderedDerivativeBufferStorage dirty_derivatives(
      derivative_storage, std::numeric_limits<Number>::quiet_NaN());
   Check(
      provider.eval_bordered_stage_derivatives(
         kkt, state.nlp, expected_derivatives.buffers()).has_value() &&
         provider.eval_bordered_stage_derivatives(
            kkt, state.nlp, dirty_derivatives.buffers()).has_value() &&
         dirty_derivatives == expected_derivatives,
      "unique sparse bordered scatter retained dirty structural values");
   PrimalDualBorderedStageKktAssembler assembler(std::move(provider), kkt);
   Check(assembler.configured(), "full bordered assembler was rejected");
   const BorderedStageStructuredLayout layout =
      assembler.bordered_stage_structured_layout();
   Check(
      layout.block_sizes == std::vector<Index>({6, 7}) &&
         layout.border_dimension == 1 &&
         layout.inertia_dimension == 14 &&
         layout.full_direction_dimension == 23 &&
         layout.full_direction_overwrite_certified,
      "full bordered assembler produced wrong dimensions");
   const std::array<Index, 13> expected_permutation{{
      2, 0, 5, 8, 10, 13,
      3, 1, 6, 7, 12, 9, 11
   }};
   Check(
      std::ranges::equal(
         assembler.stage_structured_to_full_permutation(),
         expected_permutation),
      "bordered reduced stage permutation is wrong");

   Index diagonal_values = 0;
   Index lower_values = 0;
   for( Index block = 0; block < layout.block_sizes.size(); ++block )
   {
      diagonal_values += layout.block_sizes[block] * layout.block_sizes[block];
      if( block > 0 )
      {
         lower_values +=
            layout.block_sizes[block] * layout.block_sizes[block - 1];
      }
   }
   const Index stage_dimension =
      layout.inertia_dimension - layout.border_dimension;
   std::vector<Number> diagonal(diagonal_values);
   std::vector<Number> lower(lower_values);
   std::vector<Number> border(
      stage_dimension * layout.border_dimension);
   std::vector<Number> border_diagonal(
      layout.border_dimension * layout.border_dimension);
   std::vector<Number> rhs(layout.inertia_dimension);
   Check(
      assembler.prepare_reusable_bordered_stage_storage(
         diagonal, lower, border, border_diagonal).has_value(),
      "bordered reusable storage preparation failed");
   Check(
      assembler.assemble_bordered_stage_system(
         {kkt, state, kRhs, 5, false},
         {1e-8, 2e-8, 3e-8, 4e-8},
         diagonal,
         lower,
         border,
         border_diagonal,
         rhs).has_value(),
      "initial reusable bordered assembly failed");
   const auto reused = assembler.assemble_bordered_stage_system(
      {kkt, state, kRhs, 5, false},
      {4e-6, 3e-6, 2e-6, 1e-6},
      diagonal,
      lower,
      border,
      border_diagonal,
      rhs);
   Check(reused.has_value(), "reused bordered assembly failed");
   Check(
      reused->independent_full_inertia.has_value() &&
         reused->independent_full_inertia->exact &&
         reused->independent_full_inertia->positive_eigenvalues == 9 &&
         reused->independent_full_inertia->negative_eigenvalues == 5,
      "edge-once bordered inertia certificate has the wrong signature");

   constexpr Index second_block_offset = 6 * 6;
   constexpr Index second_block_size = 7;
   constexpr Index incoming_offset = 5;
   const auto incoming_dual = [&](Index row, Index column) -> Number
   {
      return diagonal[second_block_offset +
         (incoming_offset + row) * second_block_size +
         incoming_offset + column];
   };
   CheckNear(
      incoming_dual(0, 0), -2e-6 * .256,
      "normalized incoming-dual diagonal is wrong");
   CheckNear(
      incoming_dual(1, 0), -2e-6 * .064,
      "normalized incoming-dual coupling is wrong");
   CheckNear(
      incoming_dual(1, 1), -2e-6 * .416,
      "normalized incoming-dual diagonal is wrong");
   Check(
      incoming_dual(0, 1) == incoming_dual(1, 0),
      "normalized incoming-dual block is not exactly symmetric");
   const auto second_block = [&](Index row, Index column) -> Number
   {
      return diagonal[second_block_offset + row * second_block_size + column];
   };
   Check(
      second_block(1, 5) == -1. && second_block(5, 1) == -1. &&
         second_block(2, 6) == -1. && second_block(6, 2) == -1.,
      "normalized next-state coupling does not contain the expected -I");
   Check(
      second_block(1, 6) == 0. && second_block(6, 1) == 0. &&
         second_block(2, 5) == 0. && second_block(5, 2) == 0.,
      "normalized next-state coupling has nonzero off-diagonal entries");

   const Number dirty = std::numeric_limits<Number>::quiet_NaN();
   std::vector<Number> reference_diagonal(diagonal_values, dirty);
   std::vector<Number> reference_lower(lower_values, dirty);
   std::vector<Number> reference_border(
      stage_dimension * layout.border_dimension, dirty);
   std::vector<Number> reference_border_diagonal(
      layout.border_dimension * layout.border_dimension, dirty);
   std::vector<Number> reference_rhs(layout.inertia_dimension, dirty);
   Check(
      assembler.assemble_bordered_stage_system(
         {kkt, state, kRhs, 5, false},
         {4e-6, 3e-6, 2e-6, 1e-6},
         reference_diagonal,
         reference_lower,
         reference_border,
         reference_border_diagonal,
         reference_rhs).has_value(),
      "from-scratch bordered assembly failed on dirty storage");
   const auto check_storage = [](
      std::span<const Number> actual,
      std::span<const Number> expected,
      std::string_view        message)
   {
      Check(actual.size() == expected.size(), message);
      for( Index entry = 0; entry < actual.size(); ++entry )
      {
         CheckNear(actual[entry], expected[entry], message);
      }
   };
   check_storage(diagonal, reference_diagonal, "reused diagonal is stale");
   check_storage(lower, reference_lower, "reused lower block is stale");
   check_storage(border, reference_border, "reused border is stale");
   check_storage(
      border_diagonal,
      reference_border_diagonal,
      "reused border diagonal is stale");
   check_storage(rhs, reference_rhs, "reused reduced RHS is stale");

   std::vector<Number> reconstructed_direction(
      layout.full_direction_dimension,
      std::numeric_limits<Number>::quiet_NaN());
   Check(
      assembler.reconstruct_bordered_stage_direction(
         {kkt, state, kRhs, 5, false},
         {4e-6, 3e-6, 2e-6, 1e-6},
         reference_rhs,
         reconstructed_direction).has_value() &&
         std::ranges::all_of(
            reconstructed_direction,
            [](Number value) { return std::isfinite(value); }),
      "bordered reconstruction did not overwrite the full direction");

   StateStorage invalid_reconstruction_storage;
   invalid_reconstruction_storage.slack_x_lower[0] = 0.;
   std::vector<Number> invalid_direction(layout.full_direction_dimension);
   const auto invalid_reconstruction =
      assembler.reconstruct_bordered_stage_direction(
         {
            kkt,
            invalid_reconstruction_storage.view(18),
            kRhs,
            5,
            false
         },
         {4e-6, 3e-6, 2e-6, 1e-6},
         reference_rhs,
         invalid_direction);
   Check(
      !invalid_reconstruction.has_value() &&
         invalid_reconstruction.error().code ==
            EvaluationErrorCode::nonfinite_output,
      "a new numeric revision reused stale complementarity validation");

   BorderedStageStructuredCandidateOptions options;
   options.factorization.stage_factorization.require_certified_inertia = false;
   options.factorization.schur_factorization.require_certified_inertia = false;
   options.maximum_factorization_attempts = 4;
   options.reuse_accepted_regularization = true;
   BorderedStageStructuredCandidateBackend backend(
      std::move(assembler), options);
   const auto solved = backend.solve({kkt, state, kRhs, 5, false});
   Check(solved.has_value(), "full bordered candidate solve failed");
   Check(
      solved->inertia.certainty == CandidateFirstInertiaCertainty::exact &&
         solved->inertia.negative_eigenvalues == 5 &&
         solved->work.quality_improvements >= 1 &&
         solved->accepted_regularization.c > 0. &&
         solved->accepted_regularization.d > 0.,
      "bordered independent inertia proof or retry accounting is wrong");
   CheckTrueResidual(kkt, state, *solved, kRhs);

   std::array<Number, 23> second_rhs = kRhs;
   second_rhs[4] = -1.1;
   second_rhs[16] = .85;
   std::array<Number, 23> second_direction;
   std::ranges::fill(
      second_direction, std::numeric_limits<Number>::quiet_NaN());
   const auto second = backend.solve({
      kkt, state, second_rhs, 5, false, second_direction
   });
   Check(second.has_value(), "cached bordered candidate solve failed");
   Check(
      second->direction.empty() &&
         second->direction_written_to_request_output &&
         second->work.derivative_product_requests == 0 &&
         second->work.quality_improvements == 0,
      "cached bordered output path repeated work or retained ownership");
   CheckTrueResidual(
      kkt,
      state,
      second_direction,
      second->accepted_regularization,
      second_rhs);
}

void TestReverseAndRejectedMetadata()
{
   PrimalDualKktOperator kkt = MakeKkt();
   SparseBorderedStageDerivativeProvider provider = MakeDetachedProvider();
   PrimalDualBorderedStageKktAssembler reverse(
      std::move(provider),
      kkt,
      {.reverse_stage_order = true});
   Check(reverse.configured(), "reverse bordered assembler was rejected");
   const std::array<Index, 13> expected{{
      3, 1, 6, 7, 12, 9, 11,
      2, 0, 5, 8, 10, 13
   }};
   Check(
      std::ranges::equal(
         reverse.stage_structured_to_full_permutation(), expected),
      "reverse bordered permutation is wrong");
   BorderedStageStructuredCandidateOptions reverse_options;
   reverse_options.factorization.stage_factorization.
      require_certified_inertia = false;
   reverse_options.factorization.schur_factorization.
      require_certified_inertia = false;
   reverse_options.maximum_factorization_attempts = 4;
   BorderedStageStructuredCandidateBackend reverse_backend(
      std::move(reverse), reverse_options);
   const StateStorage reverse_state_storage;
   const PrimalDualState reverse_state = reverse_state_storage.view(18);
   const auto reverse_solved = reverse_backend.solve({
      kkt, reverse_state, kRhs, 5, false});
   Check(
      reverse_solved.has_value(),
      "reverse bordered candidate solve failed");
   CheckTrueResidual(kkt, reverse_state, *reverse_solved, kRhs);

   SparseBorderedStageDerivativeProvider native_provider =
      MakeDetachedProvider();
   PrimalDualBorderedStageKktAssembler native_implicit(
      std::move(native_provider),
      kkt,
      {
         .reverse_stage_order = true,
         .normalize_next_state_jacobians = false
      });
   Check(
      native_implicit.configured(),
      "native implicit bordered assembler was rejected");
   BorderedStageStructuredCandidateBackend native_backend(
      std::move(native_implicit), reverse_options);
   const auto native_solved = native_backend.solve({
      kkt, reverse_state_storage.view(19), kRhs, 5, false});
   Check(
      native_solved.has_value(),
      "native implicit bordered candidate solve failed");
   CheckTrueResidual(
      kkt,
      reverse_state_storage.view(19),
      *native_solved,
      kRhs);

   BorderedStageNlpTopology singular_dynamics = MakeTopology();
   Check(
      singular_dynamics.configured(),
      "singular dynamics test topology was rejected prematurely");
   PrimalDualKktOperator singular_kkt = MakeKkt(true);
   SparseBorderedStageDerivativeProvider singular_provider(
      singular_kkt, std::move(singular_dynamics));
   if( !singular_provider.configured() )
   {
      throw std::runtime_error(
         "singular dynamics sparse provider was rejected prematurely: " +
         singular_provider.configuration_error());
   }
   PrimalDualBorderedStageKktAssembler singular_assembler(
      std::move(singular_provider), singular_kkt);
   Check(
      singular_assembler.configured(),
      "singular dynamics assembler was rejected before numeric evaluation");
   BorderedStageStructuredCandidateBackend singular_backend(
      std::move(singular_assembler), reverse_options);
   const auto singular = singular_backend.solve({
      singular_kkt, reverse_state_storage.view(20), kRhs, 5, false});
   Check(
      !singular.has_value() &&
         singular.error().code == EvaluationErrorCode::model_failure,
      "singular bordered next-state block was not rejected explicitly");

   BorderedStageNlpTopology overlap(
      {7, 5, 24, 21, 701},
      {{1, 2, 1, 1}, {1, 2, 0, 1}},
      {
         .primal = {2, 0, 5, 3, 1, 4},
         .dynamics = {1, 4},
         .path_equalities = {2},
         .path_inequalities = {0, 3}
      },
      {4},
      702);
   Check(
      !overlap.configured(),
      "overlapping local/global primal maps were accepted");

   SparseBorderedStageDerivativeProvider restoration_provider =
      MakeDetachedProvider();
   PrimalDualBorderedStageKktAssembler main_only(
      std::move(restoration_provider), kkt);
   BorderedStageStructuredCandidateOptions options;
   options.factorization.stage_factorization.require_certified_inertia = false;
   options.factorization.schur_factorization.require_certified_inertia = false;
   BorderedStageStructuredCandidateBackend backend(
      std::move(main_only), options);
   const StateStorage state_storage;
   const auto restoration = backend.solve({
      kkt, state_storage.view(20), kRhs, 5, true});
   Check(
      !restoration.has_value() &&
         restoration.error().code == EvaluationErrorCode::invalid_layout,
      "main bordered metadata was reused for restoration");

   StateStorage invalid_state;
   invalid_state.slack_x_lower[0] = 0.;
   const auto invalid_slack = backend.solve({
      kkt, invalid_state.view(21), kRhs, 5, false});
   Check(
      !invalid_slack.has_value() &&
         invalid_slack.error().code == EvaluationErrorCode::nonfinite_output,
      "nonpositive bordered complementarity slack was accepted");
}

void TestFixedFourStateNormalization()
{
   PrimalDualKktOperator kkt = MakeFourStateNormalizationKkt();
   PrimalDualBorderedStageKktAssembler assembler(
      FourStateNormalizationProvider{}, kkt);
   Check(
      assembler.configured(),
      "four-state normalization assembler was rejected");
   const BorderedStageStructuredLayout layout =
      assembler.bordered_stage_structured_layout();
   Check(
      layout.block_sizes == std::vector<Index>({4, 8}) &&
         layout.border_dimension == 1 &&
         layout.inertia_dimension == 13 &&
         layout.full_direction_dimension == 13 &&
         layout.full_direction_overwrite_certified,
      "four-state normalization layout is wrong");

   std::array<Number, 80> diagonal{};
   std::array<Number, 32> lower{};
   std::array<Number, 12> border{};
   std::array<Number, 1> border_diagonal{};
   std::array<Number, 13> reduced_rhs{};
   std::array<Number, 13> full_rhs{};
   const FourStateNormalizationStateStorage state_storage;
   constexpr PrimalDualRegularization regularization{
      1e-6, 0., 2e-6, 0.};
   const auto normalized = assembler.assemble_bordered_stage_system(
      {kkt, state_storage.view(31), full_rhs, 4, false},
      regularization,
      diagonal,
      lower,
      border,
      border_diagonal,
      reduced_rhs);
   Check(
      normalized.has_value() &&
         normalized->independent_full_inertia.has_value() &&
         normalized->independent_full_inertia->exact &&
         normalized->independent_full_inertia->positive_eigenvalues == 9 &&
         normalized->independent_full_inertia->negative_eigenvalues == 4,
      "pivoted four-state normalization failed");

   constexpr Index previous_block_size = 4;
   constexpr Index next_block_offset = 4 * 4;
   constexpr Index next_block_size = 8;
   constexpr Index incoming_offset = 4;
   const auto transform = [&](Index row, Index column) -> Number
   {
      return lower[(incoming_offset + row) * previous_block_size + column];
   };
   for( Index row = 0; row < 4; ++row )
   {
      for( Index column = 0; column < 4; ++column )
      {
         Number product = 0.;
         for( Index inner = 0; inner < 4; ++inner )
         {
            product += transform(row, inner) *
               FourStateNormalizationProvider::pivoted_matrix[
                  inner * 4 + column];
         }
         CheckNear(
            product,
            row == column ? -1. : 0.,
            "pivoted four-state transform is not the negative inverse");

         const Number coupling = diagonal[next_block_offset +
            row * next_block_size + incoming_offset + column];
         Check(
            coupling == (row == column ? -1. : 0.),
            "four-state normalized next-state coupling is not canonical -I");
      }
   }
   for( Index row = 0; row < 4; ++row )
   {
      for( Index column = 0; column < 4; ++column )
      {
         Number gram = 0.;
         for( Index inner = 0; inner < 4; ++inner )
         {
            gram += transform(row, inner) * transform(column, inner);
         }
         const Number actual = diagonal[next_block_offset +
            (incoming_offset + row) * next_block_size +
            incoming_offset + column];
         const Number transpose = diagonal[next_block_offset +
            (incoming_offset + column) * next_block_size +
            incoming_offset + row];
         CheckNear(
            actual,
            -regularization.c * gram,
            "four-state normalized incoming-dual Gram value is wrong");
         Check(
            actual == transpose,
            "four-state normalized incoming-dual Gram is not exactly symmetric");
      }
   }

   const std::array<Number, 13> structured_solution{{
      .25, -.5, .75, -1., 1.25, -1.5, 1.75, -2.,
      .4, -.3, .2, -.1, 2.5
   }};
   std::array<Number, 13> full_direction;
   std::ranges::fill(
      full_direction, std::numeric_limits<Number>::quiet_NaN());
   Check(
      assembler.reconstruct_bordered_stage_direction(
         {kkt, state_storage.view(31), full_rhs, 4, false},
         regularization,
         structured_solution,
         full_direction).has_value(),
      "four-state normalized direction reconstruction failed");
   std::array<Number, 13> expected_direction{};
   const std::span<const Index> permutation =
      assembler.stage_structured_to_full_permutation();
   for( Index structured = 0; structured < permutation.size(); ++structured )
   {
      expected_direction[permutation[structured]] =
         structured_solution[structured];
   }
   expected_direction[8] = structured_solution[12];
   for( Index original = 0; original < 4; ++original )
   {
      Number value = 0.;
      for( Index normalized_index = 0; normalized_index < 4;
           ++normalized_index )
      {
         value += transform(normalized_index, original) *
            structured_solution[8 + normalized_index];
      }
      expected_direction[9 + original] = value;
   }
   for( Index entry = 0; entry < full_direction.size(); ++entry )
   {
      Check(
         full_direction[entry] == expected_direction[entry],
         "four-state reconstructed direction changed operation order");
   }

   PrimalDualBorderedStageKktAssembler singular_assembler(
      FourStateNormalizationProvider{.singular = true}, kkt);
   Check(
      singular_assembler.configured(),
      "singular four-state assembler was rejected before evaluation");
   const auto singular = singular_assembler.assemble_bordered_stage_system(
      {kkt, state_storage.view(32), full_rhs, 4, false},
      regularization,
      diagonal,
      lower,
      border,
      border_diagonal,
      reduced_rhs);
   Check(
      !singular.has_value() &&
         singular.error().code == EvaluationErrorCode::model_failure,
      "singular four-state next-state block was not rejected explicitly");

   PrimalDualBorderedStageKktAssembler asymmetric_assembler(
      FourStateNormalizationProvider{
         .singular = false,
         .asymmetric_hessian = true
      },
      kkt);
   Check(
      asymmetric_assembler.configured(),
      "asymmetric four-state assembler was rejected before evaluation");
   const auto asymmetric =
      asymmetric_assembler.assemble_bordered_stage_system(
         {kkt, state_storage.view(33), full_rhs, 4, false},
         regularization,
         diagonal,
         lower,
         border,
         border_diagonal,
         reduced_rhs);
   Check(
      asymmetric.has_value() &&
         !asymmetric->independent_full_inertia.has_value(),
      "asymmetric stage Hessian received an independent inertia certificate");
   Check(
      diagonal[1] == .1 && diagonal[4] == .2,
      "asymmetric stage Hessian was silently projected during assembly");
}
} // namespace

int main()
{
   try
   {
      TestSparseDerivativeScatterPlan();
      TestSparseBorderedFullDirection();
      TestReverseAndRejectedMetadata();
      TestFixedFourStateNormalization();
   }
   catch( const std::exception& exception )
   {
      std::cerr << exception.what() << '\n';
      return 1;
   }
   return 0;
}
