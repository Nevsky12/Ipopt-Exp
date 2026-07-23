// Copyright (C) 2026 Ipopt contributors.
// All Rights Reserved.
// This file is distributed under the Eclipse Public License.

#ifndef IPOPT_CXX23_PRIMAL_DUAL_BORDERED_STAGE_KKT_ASSEMBLER_HPP
#define IPOPT_CXX23_PRIMAL_DUAL_BORDERED_STAGE_KKT_ASSEMBLER_HPP

#include <ipopt/cxx23/bordered_stage_structured_candidate_backend.hpp>
#include <ipopt/cxx23/bordered_stage_structured_nlp.hpp>
#include <ipopt/cxx23/primal_dual_kkt_operator.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace Ipopt::Cxx23
{
struct PrimalDualBorderedStageKktAssemblerOptions
{
   bool reverse_stage_order = false;
   bool restoration_problem = false;
   bool normalize_next_state_jacobians = true;
   Number next_state_relative_pivot_tolerance =
      1024. * std::numeric_limits<Number>::epsilon();
   Number next_state_backward_error_tolerance =
      4096. * std::numeric_limits<Number>::epsilon();

   friend bool operator==(
      const PrimalDualBorderedStageKktAssemblerOptions&,
      const PrimalDualBorderedStageKktAssemblerOptions&) = default;
};

/** Full eight-block Ipopt assembler for stage chains with global primals.
 *
 * The reduced variables are [local stage blocks; global x]. Bound-dual
 * directions are condensed exactly and reconstructed afterward. The stage
 * blocks retain
 *
 * [u_k,x_k,s_path_k,path_eq_y_k,path_ineq_y_k,incoming_dynamics_y_k],
 *
 * while every explicitly mapped global primal forms the dense arrowhead
 * border. No global role is inferred from sparsity.
 */
template <BorderedStageDerivativeProvider Provider>
class PrimalDualBorderedStageKktAssembler
{
public:
   PrimalDualBorderedStageKktAssembler(
      Provider                                      provider,
      PrimalDualKktOperator&                        kkt,
      PrimalDualBorderedStageKktAssemblerOptions    options = {}
   )
      : derivatives_(
           PrepareBorderedStageDerivativeProvider(std::move(provider))),
        options_(options)
   {
      Configure(kkt);
   }

   BorderedStageStructuredLayout
   bordered_stage_structured_layout() const
   {
      return layout_;
   }

   bool configured() const noexcept
   {
      return configuration_error_.empty();
   }

   const std::string& configuration_error() const noexcept
   {
      return configuration_error_;
   }

   std::span<const Index> stage_structured_to_full_permutation() const noexcept
   {
      return structured_to_full_;
   }

   /** Initialize matrix storage that the caller will preserve between assemblies.
    *
    * The bordered backend only lends these buffers to read-only factorization.
    * Pinning them lets subsequent assemblies retain permanent structural zeros
    * instead of clearing every packed block. Calls with any other storage keep
    * the ordinary from-scratch semantics.
    */
   EvaluationResult prepare_reusable_bordered_stage_storage(
      std::span<Number> diagonal,
      std::span<Number> lower,
      std::span<Number> border,
      std::span<Number> border_diagonal
   )
   {
      if( !configuration_error_.empty() )
      {
         return std::unexpected(EvaluationError{
            EvaluationErrorCode::invalid_layout,
            configuration_error_
         });
      }
      if( diagonal.size() != diagonal_values_ ||
          lower.size() != lower_values_ ||
          border.size() != border_values_ ||
          border_diagonal.size() != border_diagonal_values_ )
      {
         return std::unexpected(EvaluationError{
            EvaluationErrorCode::dimension_mismatch,
            "bordered reusable storage has the wrong size"
         });
      }
      if( reusable_storage_prepared_ && diagonal.data() == reusable_diagonal_ &&
          lower.data() == reusable_lower_ && border.data() == reusable_border_ &&
          border_diagonal.data() == reusable_border_diagonal_ )
      {
         return {};
      }
      std::ranges::fill(diagonal, 0.);
      std::ranges::fill(lower, 0.);
      std::ranges::fill(border, 0.);
      std::ranges::fill(border_diagonal, 0.);
      reusable_diagonal_ = diagonal.data();
      reusable_lower_ = lower.data();
      reusable_border_ = border.data();
      reusable_border_diagonal_ = border_diagonal.data();
      reusable_storage_prepared_ = true;
      return {};
   }

   EvaluationValue<StageStructuredAssemblyReport>
   assemble_bordered_stage_system(
      CandidateFirstSolveRequest request,
      PrimalDualRegularization   regularization,
      std::span<Number>          diagonal,
      std::span<Number>          lower,
      std::span<Number>          border,
      std::span<Number>          border_diagonal,
      std::span<Number>          rhs
   )
   {
      validated_complementarity_revision_ = 0;
      if( !configuration_error_.empty() )
      {
         return AssemblyFailure(
            EvaluationErrorCode::invalid_layout, configuration_error_);
      }
      if( request.restoration_problem != options_.restoration_problem )
      {
         return AssemblyFailure(
            EvaluationErrorCode::invalid_layout,
            "bordered stage request role does not match explicit metadata");
      }
      if( diagonal.size() != diagonal_values_ ||
          lower.size() != lower_values_ ||
          border.size() != border_values_ ||
          border_diagonal.size() != border_diagonal_values_ ||
          rhs.size() != layout_.inertia_dimension )
      {
         return AssemblyFailure(
            EvaluationErrorCode::dimension_mismatch,
            "bordered stage assembler received incorrectly sized storage");
      }
      if( request.state.numeric_revision == 0 )
      {
         return AssemblyFailure(
            EvaluationErrorCode::numeric_mismatch,
            "bordered stage assembly requires a nonzero numeric revision");
      }
      if( request.rhs.size() != layout_.full_direction_dimension )
      {
         return AssemblyFailure(
            EvaluationErrorCode::dimension_mismatch,
            "bordered stage request has the wrong full RHS dimension");
      }
      if( EvaluationResult valid = request.kkt.validate_state(request.state);
          !valid )
      {
         return std::unexpected(valid.error());
      }
      if( !ValidComplementarityState(request.state) )
      {
         return AssemblyFailure(
            EvaluationErrorCode::nonfinite_output,
            "bordered stage state has nonfinite data or nonpositive slacks");
      }

      StageStructuredAssemblyReport report;
      report.eliminated_inertia.exact = true;
      if( !derivatives_cached_ ||
          cached_numeric_revision_ != request.state.numeric_revision )
      {
         EvaluationValue<BorderedStageDerivativeView> evaluated =
            derivatives_.evaluate(request.kkt, request.state.nlp);
         if( !evaluated )
         {
            return std::unexpected(evaluated.error());
         }
         if( options_.normalize_next_state_jacobians )
         {
            if( EvaluationResult normalized = NormalizeDynamics(*evaluated);
                !normalized )
            {
               return std::unexpected(normalized.error());
            }
         }
         derivatives_cached_ = true;
         cached_numeric_revision_ = request.state.numeric_revision;
         report.work.derivative_product_requests = 1;
      }

      const bool reusable_storage = reusable_storage_prepared_ &&
         diagonal.data() == reusable_diagonal_ &&
         lower.data() == reusable_lower_ && border.data() == reusable_border_ &&
         border_diagonal.data() == reusable_border_diagonal_;
      if( !reusable_storage )
      {
         std::ranges::fill(diagonal, 0.);
         std::ranges::fill(lower, 0.);
         std::ranges::fill(border, 0.);
      }
      const BorderedStageNlpTopology& topology = derivatives_.topology();
      const BorderedStageDerivativeView derivative = derivatives_.current_view();
      const Index global_size = layout_.border_dimension;
      // Fold the edge-once primal-Hessian scan into the matrix copies while
      // retaining both asymmetric orientations for an independent rejection.
      const bool prepare_inertia = CanCertifyReducedInertia(regularization);
      bool inertia_symmetric = true;
      Number* const off_diagonal = inertia_off_diagonal_work_.data();
      if( prepare_inertia )
      {
         std::ranges::fill(inertia_off_diagonal_work_, 0.);
      }
      const std::span<const Index> global_primals =
         topology.global_primal_variables();
      for( Index row = 0; row < global_size; ++row )
      {
         const Index diagonal_offset = row * global_size + row;
         border_diagonal[diagonal_offset] =
            derivative.global_hessian[diagonal_offset];
         border_diagonal[diagonal_offset] += regularization.x;
         for( Index column = 0; column < row; ++column )
         {
            const Number forward =
               derivative.global_hessian[row * global_size + column];
            const Number transpose =
               derivative.global_hessian[column * global_size + row];
            border_diagonal[row * global_size + column] = forward;
            border_diagonal[column * global_size + row] = transpose;
            if( prepare_inertia )
            {
               inertia_symmetric =
                  inertia_symmetric && forward == transpose;
               const Number magnitude = std::abs(forward);
               off_diagonal[global_primals[row]] += magnitude;
               off_diagonal[global_primals[column]] += magnitude;
            }
         }
      }

      for( Index position = 0;
           position < topology.stages().size();
           ++position )
      {
         const Index stage = options_.reverse_stage_order
            ? topology.stages().size() - 1 - position
            : position;
         const OptimalControlStageDimensions dimensions = topology.stages()[stage];
         const Index primal_size = dimensions.controls + dimensions.states;
         const Index slack_offset = primal_size;
         const Index path_equality_offset =
            slack_offset + dimensions.path_inequalities;
         const Index path_inequality_offset =
            path_equality_offset + dimensions.path_equalities;
         const Index incoming_offset =
            path_inequality_offset + dimensions.path_inequalities;
         const Index block_size = physical_block_sizes_[stage];
         Number* block = diagonal.data() + diagonal_offsets_[stage];

         const Number* hessian =
            derivative.stage_hessians.data() + topology.hessian_offsets()[stage];
         const Number* local_global =
            derivative.local_global_hessians.data() +
            topology.local_global_hessian_offsets()[stage];
         const Index primal_begin = topology.primal_offsets()[stage];
         for( Index row = 0; row < primal_size; ++row )
         {
            const Index diagonal_offset = row * block_size + row;
            block[diagonal_offset] = hessian[row * primal_size + row];
            block[diagonal_offset] += regularization.x;
            const Index row_generic =
               topology.ordering().primal[primal_begin + row];
            for( Index column = 0; column < row; ++column )
            {
               const Number forward = hessian[row * primal_size + column];
               const Number transpose = hessian[column * primal_size + row];
               block[row * block_size + column] = forward;
               block[column * block_size + row] = transpose;
               if( prepare_inertia )
               {
                  inertia_symmetric =
                     inertia_symmetric && forward == transpose;
                  const Number magnitude = std::abs(forward);
                  off_diagonal[row_generic] += magnitude;
                  off_diagonal[topology.ordering().primal[
                     primal_begin + column]] += magnitude;
               }
            }
            const Index structured = primal_structured_positions_[row_generic];
            for( Index global = 0; global < global_size; ++global )
            {
               const Number value = local_global[row * global_size + global];
               border[structured * global_size + global] = value;
               if( prepare_inertia )
               {
                  const Number magnitude = std::abs(value);
                  off_diagonal[row_generic] += magnitude;
                  off_diagonal[global_primals[global]] += magnitude;
               }
            }
         }
         for( Index constraint = 0;
              constraint < dimensions.path_inequalities;
              ++constraint )
         {
            const Index slack = slack_offset + constraint;
            block[slack * block_size + slack] = regularization.s;
         }

         AddPathJacobian(
            derivative.path_equality_jacobians_transposed.subspan(
               topology.path_equality_jacobian_offsets()[stage],
               primal_size * dimensions.path_equalities),
            primal_size,
            dimensions.path_equalities,
            path_equality_offset,
            block_size,
            block);
         AddPathJacobian(
            derivative.path_inequality_jacobians_transposed.subspan(
               topology.path_inequality_jacobian_offsets()[stage],
               primal_size * dimensions.path_inequalities),
            primal_size,
            dimensions.path_inequalities,
            path_inequality_offset,
            block_size,
            block);
         for( Index constraint = 0;
              constraint < dimensions.path_equalities;
              ++constraint )
         {
            const Index multiplier = path_equality_offset + constraint;
            block[multiplier * block_size + multiplier] = -regularization.c;
         }
         for( Index constraint = 0;
              constraint < dimensions.path_inequalities;
              ++constraint )
         {
            const Index slack = slack_offset + constraint;
            const Index multiplier = path_inequality_offset + constraint;
            block[slack * block_size + multiplier] = -1.;
            block[multiplier * block_size + slack] = -1.;
            block[multiplier * block_size + multiplier] = -regularization.d;
         }

         if( stage > 0 )
         {
            const Index transition = stage - 1;
            const Index next_states = dimensions.states;
            const Number* transform = dynamics_transforms_.data() +
               topology.dynamics_next_state_jacobian_offsets()[transition];
            const Number* next_dynamics =
               derivative.dynamics_next_state_jacobians.data() +
               topology.dynamics_next_state_jacobian_offsets()[transition];
            if( options_.normalize_next_state_jacobians )
            {
               for( Index state = 0; state < dimensions.states; ++state )
               {
                  const Index state_variable = dimensions.controls + state;
                  const Index multiplier = incoming_offset + state;
                  block[state_variable * block_size + multiplier] = -1.;
                  block[multiplier * block_size + state_variable] = -1.;
               }
            }
            else
            {
               for( Index state = 0; state < dimensions.states; ++state )
               {
                  const Index state_variable = dimensions.controls + state;
                  for( Index constraint = 0;
                       constraint < next_states;
                       ++constraint )
                  {
                     const Index multiplier = incoming_offset + constraint;
                     const Number value =
                        next_dynamics[constraint * next_states + state];
                     block[state_variable * block_size + multiplier] = value;
                     block[multiplier * block_size + state_variable] = value;
                  }
               }
            }
            if( options_.normalize_next_state_jacobians )
            {
               if( next_states == 4 )
               {
                  WriteFourStateGram(
                     regularization.c,
                     transform,
                     incoming_offset,
                     block_size,
                     block);
               }
               else
               {
                  for( Index row = 0; row < next_states; ++row )
                  {
                     for( Index column = 0; column <= row; ++column )
                     {
                        Number value = 0.;
                        for( Index inner = 0; inner < next_states; ++inner )
                        {
                           value += transform[row * next_states + inner] *
                              transform[column * next_states + inner];
                        }
                        value *= -regularization.c;
                        block[(incoming_offset + row) * block_size +
                              incoming_offset + column] = value;
                        block[(incoming_offset + column) * block_size +
                              incoming_offset + row] = value;
                     }
                  }
               }
            }
            else
            {
               for( Index row = 0; row < next_states; ++row )
               {
                  const Index multiplier = incoming_offset + row;
                  block[multiplier * block_size + multiplier] =
                     -regularization.c;
               }
            }
         }
      }

      // Preserve transition order so that the independent-inertia sums retain
      // their original floating-point accumulation order.
      for( Index transition = 0;
           transition + 1 < topology.stages().size();
           ++transition )
      {
         const OptimalControlStageDimensions previous =
            topology.stages()[transition];
         const OptimalControlStageDimensions next =
            topology.stages()[transition + 1];
         const Index previous_primal = previous.controls + previous.states;
         const Index next_primal = next.controls + next.states;
         const Index previous_block_size = physical_block_sizes_[transition];
         const Index next_block_size = physical_block_sizes_[transition + 1];
         const Index incoming_offset = next_primal +
            next.path_inequalities + next.path_equalities +
            next.path_inequalities;
         Number* lower_block = lower.data() + lower_offsets_[transition];
         const Number* cross_hessian =
            derivative.cross_stage_hessians.data() +
            topology.cross_stage_hessian_offsets()[transition];
         const Index previous_begin = topology.primal_offsets()[transition];
         const Index next_begin = topology.primal_offsets()[transition + 1];
         for( Index next_variable = 0;
              next_variable < next_primal;
              ++next_variable )
         {
            const Index next_generic =
               topology.ordering().primal[next_begin + next_variable];
            for( Index previous_variable = 0;
                 previous_variable < previous_primal;
                 ++previous_variable )
            {
               const Number value = cross_hessian[
                  next_variable * previous_primal + previous_variable];
               if( options_.reverse_stage_order )
               {
                  lower_block[
                     previous_variable * next_block_size + next_variable] =
                     value;
               }
               else
               {
                  lower_block[
                     next_variable * previous_block_size + previous_variable] =
                     value;
               }
               if( prepare_inertia )
               {
                  const Number magnitude = std::abs(value);
                  off_diagonal[next_generic] += magnitude;
                  off_diagonal[topology.ordering().primal[
                     previous_begin + previous_variable]] += magnitude;
               }
            }
         }

         const Index next_states = next.states;
         const Number* dynamics =
            derivative.dynamics_jacobians_transposed.data() +
            topology.dynamics_jacobian_offsets()[transition];
         const Number* transform = dynamics_transforms_.data() +
            topology.dynamics_next_state_jacobian_offsets()[transition];
         if( options_.normalize_next_state_jacobians && next_states == 4 )
         {
            WriteFourStateNormalizedDynamics(
               dynamics,
               transform,
               previous_primal,
               previous_block_size,
               incoming_offset,
               next_block_size,
               options_.reverse_stage_order,
               lower_block);
         }
         else
         {
            for( Index variable = 0; variable < previous_primal; ++variable )
            {
               for( Index state = 0; state < next_states; ++state )
               {
                  Number value = 0.;
                  if( options_.normalize_next_state_jacobians )
                  {
                     for( Index original = 0;
                          original < next_states;
                          ++original )
                     {
                        value += dynamics[variable * next_states + original] *
                           transform[state * next_states + original];
                     }
                  }
                  else
                  {
                     value = dynamics[variable * next_states + state];
                  }
                  if( options_.reverse_stage_order )
                  {
                     lower_block[variable * next_block_size +
                                 incoming_offset + state] = value;
                  }
                  else
                  {
                     lower_block[(incoming_offset + state) *
                                    previous_block_size + variable] = value;
                  }
               }
            }
         }
      }

      const Index constraint_count = topology.source_structure().constraints;
      for( Index generic = 0; generic < constraint_count; ++generic )
      {
         const Index structured = constraint_structured_positions_[generic];
         for( Index global = 0; global < global_size; ++global )
         {
            border[structured * global_size + global] =
               derivative.global_jacobians_transposed[
                  global * constraint_count + generic];
         }
      }
      if( options_.normalize_next_state_jacobians )
      {
         for( Index transition = 0;
              transition + 1 < topology.stages().size();
              ++transition )
         {
            const Index next_states = topology.stages()[transition + 1].states;
            const Index dynamics_begin = topology.dynamics_offsets()[transition];
            const Number* transform = dynamics_transforms_.data() +
               topology.dynamics_next_state_jacobian_offsets()[transition];
            for( Index normalized = 0;
                 normalized < next_states;
                 ++normalized )
            {
               const Index generic_normalized = topology.ordering().dynamics[
                  dynamics_begin + normalized];
               const Index structured =
                  constraint_structured_positions_[generic_normalized];
               for( Index global = 0; global < global_size; ++global )
               {
                  Number value = 0.;
                  for( Index original = 0; original < next_states; ++original )
                  {
                     const Index generic_original = topology.ordering().dynamics[
                        dynamics_begin + original];
                     value += transform[normalized * next_states + original] *
                        derivative.global_jacobians_transposed[
                           global * constraint_count + generic_original];
                  }
                  border[structured * global_size + global] = value;
               }
            }
         }
      }

      if( !AddPrimalComplementarityDiagonal(
             diagonal,
             border_diagonal,
             request.state.z_lower,
             request.state.slack_x_lower,
             kkt_layout_.primal_lower_bounds) ||
          !AddPrimalComplementarityDiagonal(
             diagonal,
             border_diagonal,
             request.state.z_upper,
             request.state.slack_x_upper,
             kkt_layout_.primal_upper_bounds) ||
          !AddStageComplementarityDiagonal(
             diagonal,
             request.state.v_lower,
             request.state.slack_s_lower,
             kkt_layout_.slack_lower_bounds,
             slack_diagonal_offsets_) ||
          !AddStageComplementarityDiagonal(
             diagonal,
             request.state.v_upper,
             request.state.slack_s_upper,
             kkt_layout_.slack_upper_bounds,
             slack_diagonal_offsets_) )
      {
         return AssemblyFailure(
            EvaluationErrorCode::nonfinite_output,
            "bordered complementarity diagonal is nonfinite");
      }

      if( !WriteStructuredRightHandSide(request, rhs) )
      {
         return AssemblyFailure(
            EvaluationErrorCode::nonfinite_output,
            "bordered reduced right-hand side is nonfinite");
      }

      report.inertia_retry_hint =
         (dimensions_.y_c != 0 && !(regularization.c > 0.)) ||
            (dimensions_.y_d != 0 && !(regularization.d > 0.))
         ? StageStructuredInertiaRetryHint::dual
         : StageStructuredInertiaRetryHint::primal;
      report.independent_full_inertia = CertifyReducedInertia(
         diagonal,
         border_diagonal,
         regularization,
         inertia_symmetric);
      validated_complementarity_revision_ = request.state.numeric_revision;
      return report;
   }

   /** Condense a new full-KKT RHS without rebuilding the bordered matrix. */
   EvaluationResult assemble_bordered_stage_rhs(
      CandidateFirstSolveRequest request,
      std::span<Number>          rhs
   )
   {
      if( !configuration_error_.empty() )
      {
         return std::unexpected(EvaluationError{
            EvaluationErrorCode::invalid_layout,
            configuration_error_
         });
      }
      if( request.restoration_problem != options_.restoration_problem )
      {
         return std::unexpected(EvaluationError{
            EvaluationErrorCode::invalid_layout,
            "bordered stage RHS role does not match explicit metadata"
         });
      }
      if( rhs.size() != layout_.inertia_dimension ||
          request.rhs.size() != layout_.full_direction_dimension )
      {
         return std::unexpected(EvaluationError{
            EvaluationErrorCode::dimension_mismatch,
            "bordered stage RHS assembly has the wrong dimension"
         });
      }
      if( request.state.numeric_revision == 0 ||
          request.state.numeric_revision != cached_numeric_revision_ )
      {
         return std::unexpected(EvaluationError{
            EvaluationErrorCode::numeric_mismatch,
            "bordered stage RHS does not match the current numeric factor"
         });
      }
      if( EvaluationResult valid = request.kkt.validate_state(request.state);
          !valid )
      {
         return valid;
      }
      if( !ValidComplementarityState(request.state) )
      {
         return std::unexpected(EvaluationError{
            EvaluationErrorCode::nonfinite_output,
            "bordered stage RHS has invalid complementarity state"
         });
      }
      if( !WriteStructuredRightHandSide(request, rhs) )
      {
         return std::unexpected(EvaluationError{
            EvaluationErrorCode::nonfinite_output,
            "bordered reduced right-hand side is nonfinite"
         });
      }
      validated_complementarity_revision_ = request.state.numeric_revision;
      return {};
   }

   EvaluationValue<StageStructuredWork>
   reconstruct_bordered_stage_direction(
      CandidateFirstSolveRequest request,
      PrimalDualRegularization,
      std::span<const Number> structured_solution,
      std::span<Number>       full_direction
   )
   {
      if( !configuration_error_.empty() )
      {
         return std::unexpected(EvaluationError{
            EvaluationErrorCode::invalid_layout,
            configuration_error_
         });
      }
      if( request.restoration_problem != options_.restoration_problem )
      {
         return std::unexpected(EvaluationError{
            EvaluationErrorCode::invalid_layout,
            "bordered stage reconstruction role does not match metadata"
         });
      }
      if( structured_solution.size() != layout_.inertia_dimension ||
          full_direction.size() != layout_.full_direction_dimension ||
          request.rhs.size() != layout_.full_direction_dimension )
      {
         return std::unexpected(EvaluationError{
            EvaluationErrorCode::dimension_mismatch,
            "bordered stage reconstruction has invalid dimensions"
         });
      }
      if( EvaluationResult valid = request.kkt.validate_state(request.state);
          !valid )
      {
         return std::unexpected(valid.error());
      }
      const bool complementarity_prevalidated =
         request.state.numeric_revision != 0 &&
         request.state.numeric_revision == validated_complementarity_revision_;
      if( !complementarity_prevalidated &&
          !ValidComplementarityState(request.state) )
      {
         return std::unexpected(EvaluationError{
            EvaluationErrorCode::nonfinite_output,
            "bordered stage reconstruction has invalid complementarity state"
         });
      }
      for( Index structured = 0;
           structured < structured_to_full_.size();
           ++structured )
      {
         full_direction[structured_to_full_[structured]] =
            structured_solution[structured];
      }
      const Index stage_dimension = structured_to_full_.size();
      for( Index global = 0; global < global_to_full_.size(); ++global )
      {
         full_direction[global_to_full_[global]] =
            structured_solution[stage_dimension + global];
      }
      const BorderedStageNlpTopology& topology = derivatives_.topology();
      if( options_.normalize_next_state_jacobians )
      {
         const auto& dynamics_ordering = topology.ordering().dynamics;
         for( Index transition = 0;
              transition + 1 < topology.stages().size();
              ++transition )
         {
            const Index next_states = topology.stages()[transition + 1].states;
            const Index dynamics_begin = topology.dynamics_offsets()[transition];
            const Number* transform = dynamics_transforms_.data() +
               topology.dynamics_next_state_jacobian_offsets()[transition];
            const Index* generic_dynamics =
               dynamics_ordering.data() + dynamics_begin;
            if( next_states == 4 )
            {
               ReconstructFourStateDynamics(
                  transform,
                  generic_dynamics,
                  constraint_structured_positions_.data(),
                  structured_to_full_.data(),
                  structured_solution.data(),
                  full_direction.data());
            }
            else
            {
               for( Index original = 0; original < next_states; ++original )
               {
                  Number value = 0.;
                  for( Index normalized = 0;
                       normalized < next_states;
                       ++normalized )
                  {
                     value += transform[normalized * next_states + original] *
                        structured_solution[constraint_structured_positions_[
                           generic_dynamics[normalized]]];
                  }
                  full_direction[structured_to_full_[
                     constraint_structured_positions_[
                        generic_dynamics[original]]]] = value;
               }
            }
         }
      }

      ReconstructComplementarity(
         request.rhs.subspan(z_lower_offset_, dimensions_.z_lower),
         request.state.z_lower,
         request.state.slack_x_lower,
         kkt_layout_.primal_lower_bounds,
         full_direction.subspan(x_offset_, dimensions_.x),
         full_direction.subspan(z_lower_offset_, dimensions_.z_lower),
         -1.);
      ReconstructComplementarity(
         request.rhs.subspan(z_upper_offset_, dimensions_.z_upper),
         request.state.z_upper,
         request.state.slack_x_upper,
         kkt_layout_.primal_upper_bounds,
         full_direction.subspan(x_offset_, dimensions_.x),
         full_direction.subspan(z_upper_offset_, dimensions_.z_upper),
         1.);
      ReconstructComplementarity(
         request.rhs.subspan(v_lower_offset_, dimensions_.v_lower),
         request.state.v_lower,
         request.state.slack_s_lower,
         kkt_layout_.slack_lower_bounds,
         full_direction.subspan(s_offset_, dimensions_.s),
         full_direction.subspan(v_lower_offset_, dimensions_.v_lower),
         -1.);
      ReconstructComplementarity(
         request.rhs.subspan(v_upper_offset_, dimensions_.v_upper),
         request.state.v_upper,
         request.state.slack_s_upper,
         kkt_layout_.slack_upper_bounds,
         full_direction.subspan(s_offset_, dimensions_.s),
         full_direction.subspan(v_upper_offset_, dimensions_.v_upper),
         1.);
      return StageStructuredWork{};
   }

private:
   bool WriteStructuredRightHandSide(
      CandidateFirstSolveRequest request,
      std::span<Number>          rhs
   ) const noexcept
   {
      for( Index structured = 0;
           structured < structured_to_full_.size();
           ++structured )
      {
         rhs[structured] = request.rhs[structured_to_full_[structured]];
      }
      const Index stage_dimension = structured_to_full_.size();
      for( Index global = 0; global < global_to_full_.size(); ++global )
      {
         rhs[stage_dimension + global] =
            request.rhs[global_to_full_[global]];
      }

      const BorderedStageNlpTopology& topology = derivatives_.topology();
      if( options_.normalize_next_state_jacobians )
      {
         for( Index transition = 0;
              transition + 1 < topology.stages().size();
              ++transition )
         {
            const Index next_states = topology.stages()[transition + 1].states;
            const Index dynamics_begin = topology.dynamics_offsets()[transition];
            const Number* transform = dynamics_transforms_.data() +
               topology.dynamics_next_state_jacobian_offsets()[transition];
            for( Index normalized = 0;
                 normalized < next_states;
                 ++normalized )
            {
               Number value = 0.;
               for( Index original = 0; original < next_states; ++original )
               {
                  const Index generic_original = topology.ordering().dynamics[
                     dynamics_begin + original];
                  const Index full_original = structured_to_full_[
                     constraint_structured_positions_[generic_original]];
                  value += transform[normalized * next_states + original] *
                     request.rhs[full_original];
               }
               const Index generic_normalized = topology.ordering().dynamics[
                  dynamics_begin + normalized];
               rhs[constraint_structured_positions_[generic_normalized]] = value;
            }
         }
      }
      return AddComplementarityRightHandSide(
                rhs,
                request.rhs.subspan(z_lower_offset_, dimensions_.z_lower),
                request.state.slack_x_lower,
                kkt_layout_.primal_lower_bounds,
                primal_structured_positions_,
                1.) &&
         AddComplementarityRightHandSide(
            rhs,
            request.rhs.subspan(z_upper_offset_, dimensions_.z_upper),
            request.state.slack_x_upper,
            kkt_layout_.primal_upper_bounds,
            primal_structured_positions_,
            -1.) &&
         AddComplementarityRightHandSide(
            rhs,
            request.rhs.subspan(v_lower_offset_, dimensions_.v_lower),
            request.state.slack_s_lower,
            kkt_layout_.slack_lower_bounds,
            slack_structured_positions_,
            1.) &&
         AddComplementarityRightHandSide(
            rhs,
            request.rhs.subspan(v_upper_offset_, dimensions_.v_upper),
            request.state.slack_s_upper,
            kkt_layout_.slack_upper_bounds,
            slack_structured_positions_,
            -1.) &&
         AllFinite(rhs);
   }

   template <Index Extent>
   static Number FixedDotProduct(
      const Number* left,
      const Number* right
   ) noexcept
   {
      Number value = 0.;
      for( Index entry = 0; entry < Extent; ++entry )
      {
         value += left[entry] * right[entry];
      }
      return value;
   }

   static void WriteFourStateGram(
      Number        dual_regularization,
      const Number* transform,
      Index         incoming_offset,
      Index         block_size,
      Number*       block
   ) noexcept
   {
      const auto write = [&]<Index Row, Index Column>()
      {
         const Number value = -dual_regularization *
            FixedDotProduct<4>(
               transform + Row * 4,
               transform + Column * 4);
         block[(incoming_offset + Row) * block_size +
               incoming_offset + Column] = value;
         block[(incoming_offset + Column) * block_size +
               incoming_offset + Row] = value;
      };
      write.template operator()<0, 0>();
      write.template operator()<1, 0>();
      write.template operator()<1, 1>();
      write.template operator()<2, 0>();
      write.template operator()<2, 1>();
      write.template operator()<2, 2>();
      write.template operator()<3, 0>();
      write.template operator()<3, 1>();
      write.template operator()<3, 2>();
      write.template operator()<3, 3>();
   }

   static void WriteFourStateNormalizedDynamics(
      const Number* dynamics,
      const Number* transform,
      Index         previous_primal,
      Index         previous_block_size,
      Index         incoming_offset,
      Index         next_block_size,
      bool          reverse_stage_order,
      Number*       lower_block
   ) noexcept
   {
      for( Index variable = 0; variable < previous_primal; ++variable )
      {
         const Number* row = dynamics + variable * 4;
         const Number state_0 = FixedDotProduct<4>(row, transform);
         const Number state_1 = FixedDotProduct<4>(row, transform + 4);
         const Number state_2 = FixedDotProduct<4>(row, transform + 8);
         const Number state_3 = FixedDotProduct<4>(row, transform + 12);
         if( reverse_stage_order )
         {
            Number* target = lower_block +
               variable * next_block_size + incoming_offset;
            target[0] = state_0;
            target[1] = state_1;
            target[2] = state_2;
            target[3] = state_3;
         }
         else
         {
            lower_block[incoming_offset * previous_block_size + variable] =
               state_0;
            lower_block[(incoming_offset + 1) * previous_block_size + variable] =
               state_1;
            lower_block[(incoming_offset + 2) * previous_block_size + variable] =
               state_2;
            lower_block[(incoming_offset + 3) * previous_block_size + variable] =
               state_3;
         }
      }
   }

   static void ReconstructFourStateDynamics(
      const Number* transform,
      const Index*  generic_dynamics,
      const Index*  constraint_structured_positions,
      const Index*  structured_to_full,
      const Number* structured_solution,
      Number*       full_direction
   ) noexcept
   {
      const Index position_0 =
         constraint_structured_positions[generic_dynamics[0]];
      const Index position_1 =
         constraint_structured_positions[generic_dynamics[1]];
      const Index position_2 =
         constraint_structured_positions[generic_dynamics[2]];
      const Index position_3 =
         constraint_structured_positions[generic_dynamics[3]];
      const Number normalized_0 = structured_solution[position_0];
      const Number normalized_1 = structured_solution[position_1];
      const Number normalized_2 = structured_solution[position_2];
      const Number normalized_3 = structured_solution[position_3];
      const auto write = [&]<Index Original>(Index position)
      {
         Number value = 0.;
         value += transform[Original] * normalized_0;
         value += transform[4 + Original] * normalized_1;
         value += transform[8 + Original] * normalized_2;
         value += transform[12 + Original] * normalized_3;
         full_direction[structured_to_full[position]] = value;
      };
      write.template operator()<0>(position_0);
      write.template operator()<1>(position_1);
      write.template operator()<2>(position_2);
      write.template operator()<3>(position_3);
   }

   static bool CheckedAdd(Index left, Index right, Index& result) noexcept
   {
      if( right > std::numeric_limits<Index>::max() - left )
      {
         return false;
      }
      result = left + right;
      return true;
   }

   static bool CheckedMultiply(Index left, Index right, Index& result) noexcept
   {
      if( left != 0 && right > std::numeric_limits<Index>::max() / left )
      {
         return false;
      }
      result = left * right;
      return true;
   }

   static bool AllFinite(std::span<const Number> values) noexcept
   {
      return std::ranges::all_of(
         values, [](Number value) { return std::isfinite(value); });
   }

   static bool PositiveFinite(std::span<const Number> values) noexcept
   {
      return std::ranges::all_of(
         values,
         [](Number value) { return value > 0. && std::isfinite(value); });
   }

   static bool ValidComplementarityState(PrimalDualState state) noexcept
   {
      return AllFinite(state.z_lower) && AllFinite(state.z_upper) &&
         AllFinite(state.v_lower) && AllFinite(state.v_upper) &&
         PositiveFinite(state.slack_x_lower) &&
         PositiveFinite(state.slack_x_upper) &&
         PositiveFinite(state.slack_s_lower) &&
         PositiveFinite(state.slack_s_upper);
   }

   static void AddPathJacobian(
      std::span<const Number> jacobian,
      Index                   primal_size,
      Index                   constraint_size,
      Index                   multiplier_offset,
      Index                   block_size,
      Number*                 block
   ) noexcept
   {
      for( Index variable = 0; variable < primal_size; ++variable )
      {
         for( Index constraint = 0;
              constraint < constraint_size;
              ++constraint )
         {
            const Number value =
               jacobian[variable * constraint_size + constraint];
            const Index multiplier = multiplier_offset + constraint;
            block[variable * block_size + multiplier] = value;
            block[multiplier * block_size + variable] = value;
         }
      }
   }

   bool AddPrimalComplementarityDiagonal(
      std::span<Number>       diagonal,
      std::span<Number>       border_diagonal,
      std::span<const Number> multipliers,
      std::span<const Number> slacks,
      std::span<const Index>  bounds
   ) const noexcept
   {
      for( Index bound = 0; bound < bounds.size(); ++bound )
      {
         const Number value = multipliers[bound] / slacks[bound];
         if( !std::isfinite(value) )
         {
            return false;
         }
         const Index variable = bounds[bound];
         if( primal_diagonal_offsets_[variable] < diagonal_values_ )
         {
            Number& target = diagonal[primal_diagonal_offsets_[variable]];
            target += value;
            if( !std::isfinite(target) )
            {
               return false;
            }
         }
         else
         {
            const Index offset = primal_border_diagonal_offsets_[variable];
            if( offset >= border_diagonal.size() )
            {
               return false;
            }
            Number& target = border_diagonal[offset];
            target += value;
            if( !std::isfinite(target) )
            {
               return false;
            }
         }
      }
      return true;
   }

   static bool AddStageComplementarityDiagonal(
      std::span<Number>       diagonal,
      std::span<const Number> multipliers,
      std::span<const Number> slacks,
      std::span<const Index>  bounds,
      std::span<const Index>  target_offsets
   ) noexcept
   {
      for( Index bound = 0; bound < bounds.size(); ++bound )
      {
         const Number value = multipliers[bound] / slacks[bound];
         if( !std::isfinite(value) )
         {
            return false;
         }
         Number& target = diagonal[target_offsets[bounds[bound]]];
         target += value;
         if( !std::isfinite(target) )
         {
            return false;
         }
      }
      return true;
   }

   static bool AddComplementarityRightHandSide(
      std::span<Number>       reduced_rhs,
      std::span<const Number> complementarity_rhs,
      std::span<const Number> slacks,
      std::span<const Index>  bounds,
      std::span<const Index>  target_positions,
      Number                  sign
   ) noexcept
   {
      for( Index bound = 0; bound < bounds.size(); ++bound )
      {
         const Number value = sign * complementarity_rhs[bound] / slacks[bound];
         if( !std::isfinite(value) )
         {
            return false;
         }
         reduced_rhs[target_positions[bounds[bound]]] += value;
      }
      return true;
   }

   static void ReconstructComplementarity(
      std::span<const Number> rhs,
      std::span<const Number> multipliers,
      std::span<const Number> slacks,
      std::span<const Index>  bounds,
      std::span<const Number> primal_direction,
      std::span<Number>       complementarity_direction,
      Number                  primal_sign
   ) noexcept
   {
      for( Index bound = 0; bound < bounds.size(); ++bound )
      {
         complementarity_direction[bound] =
            std::fma(
               primal_sign * multipliers[bound],
               primal_direction[bounds[bound]],
               rhs[bound]) /
            slacks[bound];
      }
   }

   static EvaluationResult NormalizationFailure(std::string message)
   {
      return std::unexpected(EvaluationError{
         EvaluationErrorCode::model_failure,
         std::move(message)
      });
   }

   template <Index Extent>
   EvaluationResult InvertNextStateBlockImpl(
      std::span<const Number> matrix,
      Index                   dynamic_size,
      std::span<Number>       negative_inverse
   )
   {
      const Index size = Extent == std::dynamic_extent
         ? dynamic_size
         : Extent;
      const Index values = size * size;
      std::ranges::copy(
         matrix,
         normalization_matrix_work_.begin());
      std::ranges::fill_n(normalization_inverse_work_.begin(), values, 0.);
      Number matrix_norm = 0.;
      for( Index row = 0; row < size; ++row )
      {
         Number row_sum = 0.;
         normalization_inverse_work_[row * size + row] = 1.;
         for( Index column = 0; column < size; ++column )
         {
            row_sum += std::abs(matrix[row * size + column]);
         }
         matrix_norm = std::max(matrix_norm, row_sum);
      }
      if( !(matrix_norm > 0.) || !std::isfinite(matrix_norm) )
      {
         return NormalizationFailure(
            "bordered dynamics next-state block is zero or nonfinite");
      }
      const Number pivot_margin =
         options_.next_state_relative_pivot_tolerance * matrix_norm;
      for( Index column = 0; column < size; ++column )
      {
         Index pivot_row = column;
         Number pivot_magnitude = std::abs(
            normalization_matrix_work_[column * size + column]);
         for( Index row = column + 1; row < size; ++row )
         {
            const Number candidate = std::abs(
               normalization_matrix_work_[row * size + column]);
            if( candidate > pivot_magnitude )
            {
               pivot_magnitude = candidate;
               pivot_row = row;
            }
         }
         if( !(pivot_magnitude > pivot_margin) )
         {
            return NormalizationFailure(
               "bordered dynamics next-state block is numerically singular");
         }
         if( pivot_row != column )
         {
            for( Index entry = 0; entry < size; ++entry )
            {
               std::swap(
                  normalization_matrix_work_[column * size + entry],
                  normalization_matrix_work_[pivot_row * size + entry]);
               std::swap(
                  normalization_inverse_work_[column * size + entry],
                  normalization_inverse_work_[pivot_row * size + entry]);
            }
         }
         const Number pivot =
            normalization_matrix_work_[column * size + column];
         for( Index entry = 0; entry < size; ++entry )
         {
            normalization_matrix_work_[column * size + entry] /= pivot;
            normalization_inverse_work_[column * size + entry] /= pivot;
         }
         for( Index row = 0; row < size; ++row )
         {
            if( row == column )
            {
               continue;
            }
            const Number multiplier =
               normalization_matrix_work_[row * size + column];
            for( Index entry = 0; entry < size; ++entry )
            {
               normalization_matrix_work_[row * size + entry] -=
                  multiplier *
                  normalization_matrix_work_[column * size + entry];
               normalization_inverse_work_[row * size + entry] -=
                  multiplier *
                  normalization_inverse_work_[column * size + entry];
            }
         }
      }

      Number inverse_norm = 0.;
      Number residual_norm = 0.;
      for( Index row = 0; row < size; ++row )
      {
         Number inverse_row_sum = 0.;
         Number residual_row_sum = 0.;
         for( Index column = 0; column < size; ++column )
         {
            const Number inverse =
               normalization_inverse_work_[row * size + column];
            negative_inverse[row * size + column] = -inverse;
            inverse_row_sum += std::abs(inverse);
            Number product = 0.;
            for( Index inner = 0; inner < size; ++inner )
            {
               product += normalization_inverse_work_[row * size + inner] *
                  matrix[inner * size + column];
            }
            residual_row_sum += std::abs(
               product - (row == column ? 1. : 0.));
         }
         inverse_norm = std::max(inverse_norm, inverse_row_sum);
         residual_norm = std::max(residual_norm, residual_row_sum);
      }
      const Number backward_error = residual_norm /
         std::max(Number{1.}, inverse_norm * matrix_norm);
      if( !std::isfinite(backward_error) ||
          backward_error > options_.next_state_backward_error_tolerance *
             static_cast<Number>(size) )
      {
         return NormalizationFailure(
            "bordered dynamics next-state inverse failed its backward-error gate");
      }
      return {};
   }

   EvaluationResult InvertNextStateBlock(
      std::span<const Number> matrix,
      Index                   size,
      std::span<Number>       negative_inverse
   )
   {
      if( size == 4 )
      {
         return InvertNextStateBlockImpl<4>(
            matrix, size, negative_inverse);
      }
      return InvertNextStateBlockImpl<std::dynamic_extent>(
         matrix, size, negative_inverse);
   }

   EvaluationResult NormalizeDynamics(
      BorderedStageDerivativeView derivative
   )
   {
      const BorderedStageNlpTopology& topology = derivatives_.topology();
      for( Index transition = 0;
           transition + 1 < topology.stages().size();
           ++transition )
      {
         const Index size = topology.stages()[transition + 1].states;
         const Index offset =
            topology.dynamics_next_state_jacobian_offsets()[transition];
         if( EvaluationResult inverted = InvertNextStateBlock(
                derivative.dynamics_next_state_jacobians.subspan(
                   offset, size * size),
                size,
                std::span<Number>(
                   dynamics_transforms_.data() + offset, size * size));
             !inverted )
         {
            return inverted;
         }
      }
      return {};
   }

   bool CanCertifyReducedInertia(
      PrimalDualRegularization regularization
   ) const noexcept
   {
      return (dimensions_.y_c == 0 || regularization.c > 0.) &&
         (dimensions_.y_d == 0 || regularization.d > 0.);
   }

   std::optional<CertifiedInertia> CertifyReducedInertia(
      std::span<const Number>   diagonal,
      std::span<const Number>   border_diagonal,
      PrimalDualRegularization regularization,
      bool                     inertia_symmetric
   ) noexcept
   {
      if( !CanCertifyReducedInertia(regularization) || !inertia_symmetric )
      {
         return std::nullopt;
      }
      constexpr Number roundoff_multiplier = 128.;
      const BorderedStageNlpTopology& topology = derivatives_.topology();
      const Index global_size = layout_.border_dimension;
      Number* const off_diagonal = inertia_off_diagonal_work_.data();
      const auto stages = topology.stages();
      const auto& primal_order = topology.ordering().primal;
      const std::span<const Index> global_primals =
         topology.global_primal_variables();

      for( Index stage = 0; stage < stages.size(); ++stage )
      {
         const OptimalControlStageDimensions dimensions = stages[stage];
         const Index primal_size = dimensions.controls + dimensions.states;
         const Index block_size = physical_block_sizes_[stage];
         const Number* block = diagonal.data() + diagonal_offsets_[stage];
         const Index primal_begin = topology.primal_offsets()[stage];
         for( Index row = 0; row < primal_size; ++row )
         {
            const Index generic = primal_order[primal_begin + row];
            const Number off_diagonal_sum = off_diagonal[generic];
            const Number value = block[row * block_size + row];
            const Number scale =
               std::max(Number{1.}, std::abs(value) + off_diagonal_sum);
            const Number margin = roundoff_multiplier *
               std::numeric_limits<Number>::epsilon() *
               static_cast<Number>(dimensions_.x) * scale;
            if( !(value - off_diagonal_sum > margin) )
            {
               return std::nullopt;
            }
         }
         for( Index slack = 0;
              slack < dimensions.path_inequalities;
              ++slack )
         {
            const Index local = primal_size + slack;
            const Number value = block[local * block_size + local];
            const Number margin = roundoff_multiplier *
               std::numeric_limits<Number>::epsilon() *
               std::max(Number{1.}, std::abs(value));
            if( !(value > margin) )
            {
               return std::nullopt;
            }
         }
      }
      for( Index row = 0; row < global_size; ++row )
      {
         const Index generic = global_primals[row];
         const Number off_diagonal_sum = off_diagonal[generic];
         const Number value = border_diagonal[row * global_size + row];
         const Number scale =
            std::max(Number{1.}, std::abs(value) + off_diagonal_sum);
         const Number margin = roundoff_multiplier *
            std::numeric_limits<Number>::epsilon() *
            static_cast<Number>(dimensions_.x) * scale;
         if( !(value - off_diagonal_sum > margin) )
         {
            return std::nullopt;
         }
      }
      return CertifiedInertia{
         .positive_eigenvalues = dimensions_.x + dimensions_.s,
         .negative_eigenvalues = dimensions_.y_c + dimensions_.y_d,
         .zero_eigenvalues = 0,
         .certificate_radius = 0.,
         .minimum_separation = 0.,
         .exact = true
      };
   }

   static EvaluationValue<StageStructuredAssemblyReport> AssemblyFailure(
      EvaluationErrorCode code,
      std::string         message
   )
   {
      return std::unexpected(EvaluationError{code, std::move(message)});
   }

   bool ConfigureDimensions(NlpStructure source)
   {
      dimensions_ = {
         source.variables,
         kkt_layout_.inequality_constraints.size(),
         kkt_layout_.equality_constraints.size(),
         kkt_layout_.inequality_constraints.size(),
         kkt_layout_.primal_lower_bounds.size(),
         kkt_layout_.primal_upper_bounds.size(),
         kkt_layout_.slack_lower_bounds.size(),
         kkt_layout_.slack_upper_bounds.size()
      };
      Index reduced = 0;
      Index full = 0;
      if( !CheckedAdd(dimensions_.x, dimensions_.s, reduced) ||
          !CheckedAdd(reduced, dimensions_.y_c, reduced) ||
          !CheckedAdd(reduced, dimensions_.y_d, reduced) )
      {
         configuration_error_ = "bordered reduced dimension overflows Index";
         return false;
      }
      full = reduced;
      if( !CheckedAdd(full, dimensions_.z_lower, full) ||
          !CheckedAdd(full, dimensions_.z_upper, full) ||
          !CheckedAdd(full, dimensions_.v_lower, full) ||
          !CheckedAdd(full, dimensions_.v_upper, full) )
      {
         configuration_error_ = "bordered full dimension overflows Index";
         return false;
      }
      layout_.inertia_dimension = reduced;
      layout_.full_direction_dimension = full;
      x_offset_ = 0;
      s_offset_ = dimensions_.x;
      c_offset_ = s_offset_ + dimensions_.s;
      d_offset_ = c_offset_ + dimensions_.y_c;
      z_lower_offset_ = d_offset_ + dimensions_.y_d;
      z_upper_offset_ = z_lower_offset_ + dimensions_.z_lower;
      v_lower_offset_ = z_upper_offset_ + dimensions_.z_upper;
      v_upper_offset_ = v_lower_offset_ + dimensions_.v_lower;
      return true;
   }

   bool ConfigureConstraintMaps(
      const BorderedStageNlpTopology& topology,
      std::vector<Index>&             equality_multiplier,
      std::vector<Index>&             inequality_multiplier
   )
   {
      const Index constraints = topology.source_structure().constraints;
      const Index missing = constraints;
      equality_multiplier.assign(constraints, missing);
      inequality_multiplier.assign(constraints, missing);
      const auto add = [&](std::span<const Index> rows,
                           std::vector<Index>& map,
                           std::string_view name) -> bool
      {
         for( Index multiplier = 0; multiplier < rows.size(); ++multiplier )
         {
            const Index row = rows[multiplier];
            if( row >= constraints ||
                equality_multiplier[row] != missing ||
                inequality_multiplier[row] != missing )
            {
               configuration_error_ = std::string(name) +
                  " constraint map is out of range or repeated";
               return false;
            }
            map[row] = multiplier;
         }
         return true;
      };
      if( !add(
             kkt_layout_.equality_constraints,
             equality_multiplier,
             "bordered equality") ||
          !add(
             kkt_layout_.inequality_constraints,
             inequality_multiplier,
             "bordered inequality") )
      {
         return false;
      }
      const auto require = [&](std::span<const Index> rows,
                               const std::vector<Index>& map,
                               std::string_view name) -> bool
      {
         for( Index row : rows )
         {
            if( row >= map.size() || map[row] == missing )
            {
               configuration_error_ = std::string(name) +
                  " row has the wrong equality/inequality role";
               return false;
            }
         }
         return true;
      };
      return require(
                topology.ordering().dynamics,
                equality_multiplier,
                "bordered dynamics") &&
         require(
            topology.ordering().path_equalities,
            equality_multiplier,
            "bordered path equality") &&
         require(
            topology.ordering().path_inequalities,
            inequality_multiplier,
            "bordered path inequality");
   }

   bool ValidateBoundMap(
      std::span<const Index> bounds,
      Index                  dimension,
      std::string_view       name
   )
   {
      std::vector<bool> seen(dimension, false);
      for( Index index : bounds )
      {
         if( index >= dimension || seen[index] )
         {
            configuration_error_ = std::string(name) +
               " bound map is out of range or repeated";
            return false;
         }
         seen[index] = true;
      }
      return true;
   }

   bool ConfigurePackedStorage(const BorderedStageNlpTopology& topology)
   {
      layout_.border_dimension = topology.global_primal_variables().size();
      layout_.block_sizes.reserve(topology.stages().size());
      diagonal_offsets_.resize(topology.stages().size());
      lower_offsets_.resize(topology.stages().size() - 1);
      physical_block_sizes_.reserve(topology.stages().size());
      for( Index stage = 0; stage < topology.stages().size(); ++stage )
      {
         const OptimalControlStageDimensions dimensions = topology.stages()[stage];
         const Index primal = dimensions.controls + dimensions.states;
         const Index incoming = stage == 0 ? 0 : dimensions.states;
         Index block = 0;
         if( !CheckedAdd(primal, dimensions.path_inequalities, block) ||
             !CheckedAdd(block, dimensions.path_equalities, block) ||
             !CheckedAdd(block, dimensions.path_inequalities, block) ||
             !CheckedAdd(block, incoming, block) )
         {
            configuration_error_ = "bordered stage block size overflows Index";
            return false;
         }
         physical_block_sizes_.push_back(block);
      }
      Index diagonal_offset = 0;
      Index lower_offset = 0;
      Index stage_dimension = 0;
      for( Index position = 0;
           position < topology.stages().size();
           ++position )
      {
         const Index stage = options_.reverse_stage_order
            ? topology.stages().size() - 1 - position
            : position;
         const Index block = physical_block_sizes_[stage];
         Index square = 0;
         if( !CheckedMultiply(block, block, square) ||
             !CheckedAdd(diagonal_offset, square, diagonal_offset) ||
             !CheckedAdd(stage_dimension, block, stage_dimension) )
         {
            configuration_error_ = "bordered packed storage overflows Index";
            return false;
         }
         layout_.block_sizes.push_back(block);
         diagonal_offsets_[stage] = diagonal_offset - square;
         if( position > 0 )
         {
            Index values = 0;
            if( !CheckedMultiply(
                   block, layout_.block_sizes[position - 1], values) )
            {
               configuration_error_ = "bordered lower storage overflows Index";
               return false;
            }
            const Index transition = options_.reverse_stage_order
               ? stage
               : stage - 1;
            lower_offsets_[transition] = lower_offset;
            if( !CheckedAdd(lower_offset, values, lower_offset) )
            {
               configuration_error_ = "bordered lower storage overflows Index";
               return false;
            }
         }
      }
      Index expected_stage_dimension = 0;
      if( layout_.inertia_dimension < layout_.border_dimension )
      {
         configuration_error_ = "border exceeds reduced KKT dimension";
         return false;
      }
      expected_stage_dimension =
         layout_.inertia_dimension - layout_.border_dimension;
      if( stage_dimension != expected_stage_dimension ||
          !CheckedMultiply(
             stage_dimension, layout_.border_dimension, border_values_) ||
          !CheckedMultiply(
             layout_.border_dimension,
             layout_.border_dimension,
             border_diagonal_values_) )
      {
         configuration_error_ =
            "bordered stage blocks do not cover the reduced KKT";
         return false;
      }
      diagonal_values_ = diagonal_offset;
      lower_values_ = lower_offset;
      return true;
   }

   bool ConfigurePermutation(
      const BorderedStageNlpTopology& topology,
      std::span<const Index>          equality_multiplier,
      std::span<const Index>          inequality_multiplier
   )
   {
      const Index missing_position = layout_.inertia_dimension;
      primal_diagonal_offsets_.assign(dimensions_.x, diagonal_values_);
      primal_border_diagonal_offsets_.assign(
         dimensions_.x, border_diagonal_values_);
      slack_diagonal_offsets_.assign(dimensions_.s, diagonal_values_);
      primal_structured_positions_.assign(dimensions_.x, missing_position);
      slack_structured_positions_.assign(dimensions_.s, missing_position);
      constraint_structured_positions_.assign(
         topology.source_structure().constraints, missing_position);
      structured_to_full_.reserve(
         layout_.inertia_dimension - layout_.border_dimension);

      for( Index position = 0;
           position < topology.stages().size();
           ++position )
      {
         const Index stage = options_.reverse_stage_order
            ? topology.stages().size() - 1 - position
            : position;
         const OptimalControlStageDimensions dimensions = topology.stages()[stage];
         const Index primal_begin = topology.primal_offsets()[stage];
         const Index primal_size = dimensions.controls + dimensions.states;
         const Index block_size = physical_block_sizes_[stage];
         Index local = 0;
         for( Index variable = 0; variable < primal_size; ++variable, ++local )
         {
            const Index generic = topology.ordering().primal[
               primal_begin + variable];
            primal_structured_positions_[generic] = structured_to_full_.size();
            primal_diagonal_offsets_[generic] =
               diagonal_offsets_[stage] + local * block_size + local;
            structured_to_full_.push_back(x_offset_ + generic);
         }
         const Index inequality_begin =
            topology.path_inequality_offsets()[stage];
         for( Index constraint = 0;
              constraint < dimensions.path_inequalities;
              ++constraint, ++local )
         {
            const Index generic = topology.ordering().path_inequalities[
               inequality_begin + constraint];
            const Index multiplier = inequality_multiplier[generic];
            slack_structured_positions_[multiplier] = structured_to_full_.size();
            slack_diagonal_offsets_[multiplier] =
               diagonal_offsets_[stage] + local * block_size + local;
            structured_to_full_.push_back(s_offset_ + multiplier);
         }
         const Index equality_begin = topology.path_equality_offsets()[stage];
         for( Index constraint = 0;
              constraint < dimensions.path_equalities;
              ++constraint, ++local )
         {
            const Index generic = topology.ordering().path_equalities[
               equality_begin + constraint];
            constraint_structured_positions_[generic] = structured_to_full_.size();
            structured_to_full_.push_back(
               c_offset_ + equality_multiplier[generic]);
         }
         for( Index constraint = 0;
              constraint < dimensions.path_inequalities;
              ++constraint, ++local )
         {
            const Index generic = topology.ordering().path_inequalities[
               inequality_begin + constraint];
            constraint_structured_positions_[generic] = structured_to_full_.size();
            structured_to_full_.push_back(
               d_offset_ + inequality_multiplier[generic]);
         }
         if( stage > 0 )
         {
            const Index dynamics_begin = topology.dynamics_offsets()[stage - 1];
            for( Index state = 0; state < dimensions.states; ++state, ++local )
            {
               const Index generic = topology.ordering().dynamics[
                  dynamics_begin + state];
               constraint_structured_positions_[generic] = structured_to_full_.size();
               structured_to_full_.push_back(
                  c_offset_ + equality_multiplier[generic]);
            }
         }
         if( local != block_size )
         {
            configuration_error_ = "bordered permutation does not fill a block";
            return false;
         }
      }

      const Index stage_dimension = structured_to_full_.size();
      global_to_full_.reserve(layout_.border_dimension);
      for( Index global = 0;
           global < topology.global_primal_variables().size();
           ++global )
      {
         const Index generic = topology.global_primal_variables()[global];
         primal_structured_positions_[generic] = stage_dimension + global;
         primal_border_diagonal_offsets_[generic] =
            global * layout_.border_dimension + global;
         global_to_full_.push_back(x_offset_ + generic);
      }

      std::vector<bool> seen(layout_.inertia_dimension, false);
      for( Index full : structured_to_full_ )
      {
         if( full >= layout_.inertia_dimension || seen[full] )
         {
            configuration_error_ = "bordered stage permutation is invalid";
            return false;
         }
         seen[full] = true;
      }
      for( Index full : global_to_full_ )
      {
         if( full >= layout_.inertia_dimension || seen[full] )
         {
            configuration_error_ = "bordered global permutation is invalid";
            return false;
         }
         seen[full] = true;
      }
      if( !std::ranges::all_of(seen, [](bool value) { return value; }) ||
          std::ranges::any_of(
             primal_structured_positions_,
             [&](Index value) { return value >= layout_.inertia_dimension; }) ||
          std::ranges::any_of(
             slack_structured_positions_,
             [&](Index value) { return value >= layout_.inertia_dimension; }) ||
          std::ranges::any_of(
             constraint_structured_positions_,
             [&](Index value) { return value >= stage_dimension; }) )
      {
         configuration_error_ = "bordered permutation does not cover reduced KKT";
         return false;
      }
      return true;
   }

   void Configure(PrimalDualKktOperator& kkt)
   {
      const BorderedStageNlpTopology& topology = derivatives_.topology();
      if( !topology.configured() )
      {
         configuration_error_ = topology.configuration_error();
         return;
      }
      constexpr Number minimum_normalization_tolerance =
         32. * std::numeric_limits<Number>::epsilon();
      if( options_.normalize_next_state_jacobians &&
          (!std::isfinite(options_.next_state_relative_pivot_tolerance) ||
          options_.next_state_relative_pivot_tolerance <
             minimum_normalization_tolerance ||
          !std::isfinite(options_.next_state_backward_error_tolerance) ||
          options_.next_state_backward_error_tolerance <
             minimum_normalization_tolerance) )
      {
         configuration_error_ =
            "bordered dynamics normalization tolerances are below the "
            "roundoff safety floor or nonfinite";
         return;
      }
      if( topology.source_structure() != kkt.nlp_structure() )
      {
         configuration_error_ = "bordered topology does not match live KKT";
         return;
      }
      kkt_layout_ = kkt.layout();
      if( !ConfigureDimensions(topology.source_structure()) )
      {
         return;
      }
      std::vector<Index> equality_multiplier;
      std::vector<Index> inequality_multiplier;
      if( !ConfigureConstraintMaps(
             topology, equality_multiplier, inequality_multiplier) ||
          !ValidateBoundMap(
             kkt_layout_.primal_lower_bounds, dimensions_.x, "primal lower") ||
          !ValidateBoundMap(
             kkt_layout_.primal_upper_bounds, dimensions_.x, "primal upper") ||
          !ValidateBoundMap(
             kkt_layout_.slack_lower_bounds, dimensions_.s, "slack lower") ||
          !ValidateBoundMap(
             kkt_layout_.slack_upper_bounds, dimensions_.s, "slack upper") ||
          !ConfigurePackedStorage(topology) ||
          !ConfigurePermutation(
             topology, equality_multiplier, inequality_multiplier) )
      {
         return;
      }
      if( kkt.dimensions() != dimensions_ ||
          kkt.flat_dimension() != layout_.full_direction_dimension )
      {
         configuration_error_ = "bordered layout does not match KKT dimensions";
         return;
      }
      StructureFingerprintResult fingerprint = kkt.structure_fingerprint();
      if( !fingerprint )
      {
         configuration_error_ = fingerprint.error().message;
         return;
      }
      layout_.kkt_fingerprint = *fingerprint;

      const Index transform_values =
         topology.derivative_storage().dynamics_next_state_jacobians;
      dynamics_transforms_.resize(transform_values);
      Index maximum_states = 0;
      for( Index stage = 1; stage < topology.stages().size(); ++stage )
      {
         maximum_states = std::max(
            maximum_states, topology.stages()[stage].states);
      }
      Index normalization_values = 0;
      if( !CheckedMultiply(
             maximum_states, maximum_states, normalization_values) )
      {
         configuration_error_ =
            "bordered dynamics normalization workspace overflows Index";
         return;
      }
      normalization_matrix_work_.resize(normalization_values);
      normalization_inverse_work_.resize(normalization_values);
      inertia_off_diagonal_work_.resize(dimensions_.x);
      layout_.full_direction_overwrite_certified = true;
   }

   PreparedBorderedStageDerivativeProvider<Provider> derivatives_;
   PrimalDualBorderedStageKktAssemblerOptions options_;
   BorderedStageStructuredLayout layout_;
   PrimalDualLayout kkt_layout_;
   PrimalDualDimensions dimensions_{};
   std::vector<Index> physical_block_sizes_;
   std::vector<Index> diagonal_offsets_;
   std::vector<Index> lower_offsets_;
   std::vector<Index> structured_to_full_;
   std::vector<Index> global_to_full_;
   std::vector<Index> primal_diagonal_offsets_;
   std::vector<Index> primal_border_diagonal_offsets_;
   std::vector<Index> slack_diagonal_offsets_;
   std::vector<Index> primal_structured_positions_;
   std::vector<Index> slack_structured_positions_;
   std::vector<Index> constraint_structured_positions_;
   std::vector<Number> dynamics_transforms_;
   std::vector<Number> normalization_matrix_work_;
   std::vector<Number> normalization_inverse_work_;
   std::vector<Number> inertia_off_diagonal_work_;
   Index diagonal_values_ = 0;
   Index lower_values_ = 0;
   Index border_values_ = 0;
   Index border_diagonal_values_ = 0;
   Index x_offset_ = 0;
   Index s_offset_ = 0;
   Index c_offset_ = 0;
   Index d_offset_ = 0;
   Index z_lower_offset_ = 0;
   Index z_upper_offset_ = 0;
   Index v_lower_offset_ = 0;
   Index v_upper_offset_ = 0;
   std::string configuration_error_;
   bool derivatives_cached_ = false;
   std::uint64_t cached_numeric_revision_ = 0;
   std::uint64_t validated_complementarity_revision_ = 0;
   Number* reusable_diagonal_ = nullptr;
   Number* reusable_lower_ = nullptr;
   Number* reusable_border_ = nullptr;
   Number* reusable_border_diagonal_ = nullptr;
   bool reusable_storage_prepared_ = false;
};
} // namespace Ipopt::Cxx23

#endif
