// Copyright (C) 2026 Ipopt contributors.
// All Rights Reserved.
// This file is distributed under the Eclipse Public License.

#ifndef IPOPT_CXX23_PRIMAL_DUAL_STAGE_KKT_ASSEMBLER_HPP
#define IPOPT_CXX23_PRIMAL_DUAL_STAGE_KKT_ASSEMBLER_HPP

#include <ipopt/cxx23/stage_structured_candidate_backend.hpp>
#include <ipopt/cxx23/stage_structured_nlp.hpp>

#include <algorithm>
#include <array>
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
struct PrimalDualStageKktAssemblerOptions
{
   bool reverse_stage_order = false;
   /** Expected NLP role. Restoration requires its own transformed topology. */
   bool restoration_problem = false;

   friend bool operator==(
      const PrimalDualStageKktAssemblerOptions&,
      const PrimalDualStageKktAssemblerOptions&) = default;
};

/** Precomputed stage layout for Ipopt's complete eight-block direction.
 *
 * The symmetric factorization variables are [x,s,y_c,y_d]. Bound-dual
 * directions [z_L,z_U,v_L,v_U] are eliminated exactly using the current
 * positive complementarity slacks and reconstructed after the stage solve.
 * The expected PrimalDualLayout is explicit production metadata: no variable,
 * constraint, or bound role is inferred from names or sparsity at bind time.
 */
class PrimalDualStageKktLayoutWorkspace
{
public:
   PrimalDualStageKktLayoutWorkspace(
      const StageNlpTopology&             topology,
      PrimalDualLayout                    expected_layout,
      PrimalDualStageKktAssemblerOptions  options = {}
   )
      : source_structure_(topology.source_structure()),
        expected_layout_(std::move(expected_layout)),
        options_(options)
   {
      Configure(topology);
   }

   bool configured() const noexcept
   {
      return configuration_error_.empty();
   }

   const std::string& configuration_error() const noexcept
   {
      return configuration_error_;
   }

   StageStructuredLayout stage_structured_layout() const
   {
      return layout_;
   }

private:
   template <StageDerivativeProvider>
   friend class PrimalDualStageKktAssembler;

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

   bool ConfigureDimensions()
   {
      dimensions_ = {
         source_structure_.variables,
         expected_layout_.inequality_constraints.size(),
         expected_layout_.equality_constraints.size(),
         expected_layout_.inequality_constraints.size(),
         expected_layout_.primal_lower_bounds.size(),
         expected_layout_.primal_upper_bounds.size(),
         expected_layout_.slack_lower_bounds.size(),
         expected_layout_.slack_upper_bounds.size()
      };
      Index reduced = 0;
      Index full = 0;
      if( !CheckedAdd(dimensions_.x, dimensions_.s, reduced) ||
          !CheckedAdd(reduced, dimensions_.y_c, reduced) ||
          !CheckedAdd(reduced, dimensions_.y_d, reduced) )
      {
         configuration_error_ =
            "primal-dual stage reduced dimension overflows Index";
         return false;
      }
      full = reduced;
      if( !CheckedAdd(full, dimensions_.z_lower, full) ||
          !CheckedAdd(full, dimensions_.z_upper, full) ||
          !CheckedAdd(full, dimensions_.v_lower, full) ||
          !CheckedAdd(full, dimensions_.v_upper, full) )
      {
         configuration_error_ =
            "primal-dual stage full dimension overflows Index";
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
      const StageNlpTopology& topology,
      std::vector<Index>&     equality_multiplier,
      std::vector<Index>&     inequality_multiplier
   )
   {
      const Index missing = source_structure_.constraints;
      equality_multiplier.assign(source_structure_.constraints, missing);
      inequality_multiplier.assign(source_structure_.constraints, missing);
      const auto add = [&](std::span<const Index> constraints,
                           std::vector<Index>& map,
                           std::string_view name) -> bool
      {
         for( Index multiplier = 0; multiplier < constraints.size(); ++multiplier )
         {
            const Index constraint = constraints[multiplier];
            if( constraint >= source_structure_.constraints ||
                equality_multiplier[constraint] != missing ||
                inequality_multiplier[constraint] != missing )
            {
               configuration_error_ = std::string(name) +
                  " constraint map is out of range or repeated";
               return false;
            }
            map[constraint] = multiplier;
         }
         return true;
      };
      if( !add(
             expected_layout_.equality_constraints,
             equality_multiplier,
             "equality") ||
          !add(
             expected_layout_.inequality_constraints,
             inequality_multiplier,
             "inequality") )
      {
         return false;
      }
      for( Index constraint = 0;
           constraint < source_structure_.constraints;
           ++constraint )
      {
         if( equality_multiplier[constraint] == missing &&
             inequality_multiplier[constraint] == missing )
         {
            configuration_error_ =
               "primal-dual constraint maps do not cover the NLP";
            return false;
         }
      }

      const StageNlpOrdering& ordering = topology.ordering();
      const auto require_kind = [&](std::span<const Index> constraints,
                                    const std::vector<Index>& map,
                                    std::string_view name) -> bool
      {
         for( Index constraint : constraints )
         {
            if( constraint >= map.size() || map[constraint] == missing )
            {
               configuration_error_ = std::string(name) +
                  " stage row has the wrong equality/inequality role";
               return false;
            }
         }
         return true;
      };
      return require_kind(ordering.dynamics, equality_multiplier, "dynamics") &&
         require_kind(
            ordering.path_equalities,
            equality_multiplier,
            "path equality") &&
         require_kind(
            ordering.path_inequalities,
            inequality_multiplier,
            "path inequality");
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

   bool ConfigurePackedStorage(const StageNlpTopology& topology)
   {
      layout_.block_sizes.reserve(topology.stages().size());
      diagonal_offsets_.resize(topology.stages().size());
      lower_offsets_.resize(topology.stages().size() - 1);
      physical_block_sizes_.reserve(topology.stages().size());
      for( Index stage = 0; stage < topology.stages().size(); ++stage )
      {
         const OptimalControlStageDimensions dimensions = topology.stages()[stage];
         const Index primal = dimensions.controls + dimensions.states;
         const Index incoming = stage == 0 ? 0 : dimensions.states;
         Index block_size = 0;
         if( !CheckedAdd(primal, dimensions.path_inequalities, block_size) ||
             !CheckedAdd(block_size, dimensions.path_equalities, block_size) ||
             !CheckedAdd(block_size, dimensions.path_inequalities, block_size) ||
             !CheckedAdd(block_size, incoming, block_size) )
         {
            configuration_error_ =
               "primal-dual stage block size overflows Index";
            return false;
         }
         physical_block_sizes_.push_back(block_size);
      }

      Index diagonal_offset = 0;
      Index lower_offset = 0;
      Index structured_dimension = 0;
      for( Index position = 0;
           position < topology.stages().size();
           ++position )
      {
         const Index stage = options_.reverse_stage_order
            ? topology.stages().size() - 1 - position
            : position;
         const Index block_size = physical_block_sizes_[stage];
         Index diagonal_values = 0;
         Index next_diagonal_offset = 0;
         Index next_lower_offset = lower_offset;
         if( !CheckedMultiply(block_size, block_size, diagonal_values) ||
             !CheckedAdd(
                diagonal_offset, diagonal_values, next_diagonal_offset) ||
             !CheckedAdd(
                structured_dimension, block_size, structured_dimension) )
         {
            configuration_error_ =
               "primal-dual packed stage storage overflows Index";
            return false;
         }
         if( position > 0 )
         {
            Index lower_values = 0;
            if( !CheckedMultiply(
                   block_size,
                   layout_.block_sizes[position - 1],
                   lower_values) ||
                !CheckedAdd(lower_offset, lower_values, next_lower_offset) )
            {
               configuration_error_ =
                  "primal-dual packed coupling storage overflows Index";
               return false;
            }
         }
         layout_.block_sizes.push_back(block_size);
         diagonal_offsets_[stage] = diagonal_offset;
         diagonal_offset = next_diagonal_offset;
         if( position > 0 )
         {
            const Index transition = options_.reverse_stage_order
               ? stage
               : stage - 1;
            lower_offsets_[transition] = lower_offset;
            lower_offset = next_lower_offset;
         }
      }
      diagonal_values_ = diagonal_offset;
      lower_values_ = lower_offset;
      if( structured_dimension != layout_.inertia_dimension )
      {
         configuration_error_ =
            "primal-dual stage blocks do not cover the reduced KKT";
         return false;
      }
      return true;
   }

   bool ConfigurePermutation(
      const StageNlpTopology&    topology,
      std::span<const Index>     equality_multiplier,
      std::span<const Index>     inequality_multiplier
   )
   {
      const Index missing = layout_.inertia_dimension;
      primal_diagonal_offsets_.assign(dimensions_.x, diagonal_values_);
      slack_diagonal_offsets_.assign(dimensions_.s, diagonal_values_);
      primal_structured_positions_.assign(dimensions_.x, missing);
      slack_structured_positions_.assign(dimensions_.s, missing);
      structured_to_full_.reserve(layout_.inertia_dimension);
      const StageNlpOrdering& ordering = topology.ordering();

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
            const Index generic = ordering.primal[primal_begin + variable];
            primal_structured_positions_[generic] = structured_to_full_.size();
            primal_diagonal_offsets_[generic] =
               diagonal_offsets_[stage] + local * block_size + local;
            structured_to_full_.push_back(x_offset_ + generic);
         }

         const Index inequality_begin = topology.path_inequality_offsets()[stage];
         for( Index constraint = 0;
              constraint < dimensions.path_inequalities;
              ++constraint, ++local )
         {
            const Index generic =
               ordering.path_inequalities[inequality_begin + constraint];
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
            const Index generic =
               ordering.path_equalities[equality_begin + constraint];
            structured_to_full_.push_back(
               c_offset_ + equality_multiplier[generic]);
         }
         for( Index constraint = 0;
              constraint < dimensions.path_inequalities;
              ++constraint, ++local )
         {
            const Index generic =
               ordering.path_inequalities[inequality_begin + constraint];
            structured_to_full_.push_back(
               d_offset_ + inequality_multiplier[generic]);
         }
         if( stage > 0 )
         {
            const Index dynamics_begin = topology.dynamics_offsets()[stage - 1];
            for( Index state = 0;
                 state < dimensions.states;
                 ++state, ++local )
            {
               const Index generic = ordering.dynamics[dynamics_begin + state];
               structured_to_full_.push_back(
                  c_offset_ + equality_multiplier[generic]);
            }
         }
         if( local != block_size )
         {
            configuration_error_ =
               "primal-dual stage permutation does not fill its block";
            return false;
         }
      }

      std::vector<bool> seen(layout_.inertia_dimension, false);
      for( Index index : structured_to_full_ )
      {
         if( index >= seen.size() || seen[index] )
         {
            configuration_error_ =
               "primal-dual structured/reduced permutation is invalid";
            return false;
         }
         seen[index] = true;
      }
      if( !std::ranges::all_of(seen, [](bool value) { return value; }) ||
          std::ranges::any_of(
             primal_diagonal_offsets_,
             [&](Index value) { return value >= diagonal_values_; }) ||
          std::ranges::any_of(
             slack_diagonal_offsets_,
             [&](Index value) { return value >= diagonal_values_; }) )
      {
         configuration_error_ =
            "primal-dual stage permutation does not cover reduced variables";
         return false;
      }
      return true;
   }

   void Configure(const StageNlpTopology& topology)
   {
      if( !topology.configured() )
      {
         configuration_error_ = topology.configuration_error();
         return;
      }
      StructureFingerprintResult topology_fingerprint = topology.fingerprint();
      if( !topology_fingerprint )
      {
         configuration_error_ = topology_fingerprint.error().message;
         return;
      }
      topology_fingerprint_ = *topology_fingerprint;
      if( !ConfigureDimensions() )
      {
         return;
      }
      std::vector<Index> equality_multiplier;
      std::vector<Index> inequality_multiplier;
      if( !ConfigureConstraintMaps(
             topology, equality_multiplier, inequality_multiplier) ||
          !ValidateBoundMap(
             expected_layout_.primal_lower_bounds,
             dimensions_.x,
             "primal lower") ||
          !ValidateBoundMap(
             expected_layout_.primal_upper_bounds,
             dimensions_.x,
             "primal upper") ||
          !ValidateBoundMap(
             expected_layout_.slack_lower_bounds,
             dimensions_.s,
             "slack lower") ||
          !ValidateBoundMap(
             expected_layout_.slack_upper_bounds,
             dimensions_.s,
             "slack upper") ||
          !ConfigurePackedStorage(topology) ||
          !ConfigurePermutation(
             topology, equality_multiplier, inequality_multiplier) )
      {
         return;
      }
   }

   NlpStructure source_structure_;
   PrimalDualLayout expected_layout_;
   PrimalDualStageKktAssemblerOptions options_;
   PrimalDualDimensions dimensions_{};
   StructureFingerprint topology_fingerprint_{0, 0};
   StageStructuredLayout layout_;
   std::vector<Index> physical_block_sizes_;
   std::vector<Index> diagonal_offsets_;
   std::vector<Index> lower_offsets_;
   std::vector<Index> structured_to_full_;
   std::vector<Index> primal_diagonal_offsets_;
   std::vector<Index> slack_diagonal_offsets_;
   std::vector<Index> primal_structured_positions_;
   std::vector<Index> slack_structured_positions_;
   Index diagonal_values_ = 0;
   Index lower_values_ = 0;
   Index x_offset_ = 0;
   Index s_offset_ = 0;
   Index c_offset_ = 0;
   Index d_offset_ = 0;
   Index z_lower_offset_ = 0;
   Index z_upper_offset_ = 0;
   Index v_lower_offset_ = 0;
   Index v_upper_offset_ = 0;
   std::string configuration_error_;
};

/** Stage assembler for Ipopt's complete primal-dual direction.
 *
 * Physical block k is
 *
 * [u_k,x_k,s_path_k,path_eq_y_k,path_ineq_y_k,incoming_dynamics_y_k].
 *
 * Complementarity directions are eliminated exactly into positive diagonal
 * terms z/slack and v/slack. The resulting symmetric block-tridiagonal system
 * has the same inertia target used by AugSystemSolver: y_c + y_d negative
 * eigenvalues. Reconstruction returns all eight blocks and the outer canary
 * independently applies the original nonsymmetric full KKT.
 */
template <StageDerivativeProvider Provider>
class PrimalDualStageKktAssembler
{
public:
   PrimalDualStageKktAssembler(
      Provider                            provider,
      PrimalDualKktOperator&              kkt,
      PrimalDualStageKktAssemblerOptions  options = {}
   )
      : derivatives_(PrepareStageDerivativeProvider(std::move(provider))),
        options_(options)
   {
      Configure(kkt);
   }

   PrimalDualStageKktAssembler(
      Provider                                provider,
      PrimalDualKktOperator&                  kkt,
      PrimalDualStageKktLayoutWorkspace       workspace
   )
      : derivatives_(PrepareStageDerivativeProvider(std::move(provider))),
        options_(workspace.options_)
   {
      Bind(kkt, std::move(workspace));
   }

   StageStructuredLayout stage_structured_layout() const
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

   std::span<const Index> structured_to_full_permutation() const noexcept
   {
      return structured_to_full_;
   }

   EvaluationValue<StageStructuredAssemblyReport> assemble_stage_system(
      CandidateFirstSolveRequest request,
      PrimalDualRegularization   regularization,
      std::span<Number>          diagonal,
      std::span<Number>          lower,
      std::span<Number>          rhs
   )
   {
      if( !configuration_error_.empty() )
      {
         return AssemblyFailure(
            EvaluationErrorCode::invalid_layout, configuration_error_);
      }
      if( request.restoration_problem != options_.restoration_problem )
      {
         return AssemblyFailure(
            EvaluationErrorCode::invalid_layout,
            "primal-dual stage assembler request role does not match its explicit metadata");
      }
      if( diagonal.size() != diagonal_values_ ||
          lower.size() != lower_values_ ||
          rhs.size() != structured_to_full_.size() )
      {
         return AssemblyFailure(
            EvaluationErrorCode::dimension_mismatch,
            "primal-dual stage assembler received incorrectly sized storage");
      }
      if( request.state.numeric_revision == 0 )
      {
         return AssemblyFailure(
            EvaluationErrorCode::numeric_mismatch,
            "primal-dual stage assembly requires a nonzero numeric revision");
      }
      if( request.rhs.size() != layout_.full_direction_dimension )
      {
         return AssemblyFailure(
            EvaluationErrorCode::dimension_mismatch,
            "primal-dual stage request has the wrong full RHS dimension");
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
            "primal-dual stage state has nonfinite data or nonpositive slacks");
      }

      StageStructuredAssemblyReport report;
      report.eliminated_inertia.exact = true;
      if( !derivatives_cached_ ||
          cached_numeric_revision_ != request.state.numeric_revision )
      {
         EvaluationValue<StageDerivativeView> evaluated =
            derivatives_.evaluate(request.kkt, request.state.nlp);
         if( !evaluated )
         {
            return std::unexpected(evaluated.error());
         }
         derivatives_cached_ = true;
         cached_numeric_revision_ = request.state.numeric_revision;
         report.work.derivative_product_requests = 1;
      }

      std::ranges::fill(diagonal, 0.);
      std::ranges::fill(lower, 0.);
      const StageNlpTopology& topology = derivatives_.topology();
      const StageDerivativeView derivative = derivatives_.current_view();
      for( Index stage = 0; stage < topology.stages().size(); ++stage )
      {
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
         for( Index row = 0; row < primal_size; ++row )
         {
            for( Index column = 0; column < primal_size; ++column )
            {
               block[row * block_size + column] =
                  hessian[row * primal_size + column];
            }
            block[row * block_size + row] += regularization.x;
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
            for( Index state = 0; state < dimensions.states; ++state )
            {
               const Index state_variable = dimensions.controls + state;
               const Index multiplier = incoming_offset + state;
               block[state_variable * block_size + multiplier] = -1.;
               block[multiplier * block_size + state_variable] = -1.;
               block[multiplier * block_size + multiplier] = -regularization.c;
            }

            const OptimalControlStageDimensions previous =
               topology.stages()[stage - 1];
            const Index previous_primal = previous.controls + previous.states;
            const Index previous_block_size = physical_block_sizes_[stage - 1];
            Number* lower_block = lower.data() + lower_offsets_[stage - 1];
            const Number* dynamics =
               derivative.dynamics_jacobians_transposed.data() +
               topology.dynamics_jacobian_offsets()[stage - 1];
            for( Index variable = 0; variable < previous_primal; ++variable )
            {
               for( Index state = 0; state < dimensions.states; ++state )
               {
                  if( options_.reverse_stage_order )
                  {
                     lower_block[variable * block_size +
                                 incoming_offset + state] =
                        dynamics[variable * dimensions.states + state];
                  }
                  else
                  {
                     lower_block[(incoming_offset + state) *
                                    previous_block_size +
                                 variable] =
                        dynamics[variable * dimensions.states + state];
                  }
               }
            }
         }
      }

      if( !AddComplementarityDiagonal(
             diagonal,
             request.state.z_lower,
             request.state.slack_x_lower,
             kkt_layout_.primal_lower_bounds,
             primal_diagonal_offsets_) ||
          !AddComplementarityDiagonal(
             diagonal,
             request.state.z_upper,
             request.state.slack_x_upper,
             kkt_layout_.primal_upper_bounds,
             primal_diagonal_offsets_) ||
          !AddComplementarityDiagonal(
             diagonal,
             request.state.v_lower,
             request.state.slack_s_lower,
             kkt_layout_.slack_lower_bounds,
             slack_diagonal_offsets_) ||
          !AddComplementarityDiagonal(
             diagonal,
             request.state.v_upper,
             request.state.slack_s_upper,
             kkt_layout_.slack_upper_bounds,
             slack_diagonal_offsets_) )
      {
         return AssemblyFailure(
            EvaluationErrorCode::nonfinite_output,
            "primal-dual complementarity diagonal is nonfinite");
      }

      for( Index structured = 0;
           structured < structured_to_full_.size();
           ++structured )
      {
         rhs[structured] = request.rhs[structured_to_full_[structured]];
      }
      if( !AddComplementarityRightHandSide(
             rhs,
             request.rhs.subspan(z_lower_offset_, dimensions_.z_lower),
             request.state.slack_x_lower,
             kkt_layout_.primal_lower_bounds,
             primal_structured_positions_,
             1.) ||
          !AddComplementarityRightHandSide(
             rhs,
             request.rhs.subspan(z_upper_offset_, dimensions_.z_upper),
             request.state.slack_x_upper,
             kkt_layout_.primal_upper_bounds,
             primal_structured_positions_,
             -1.) ||
          !AddComplementarityRightHandSide(
             rhs,
             request.rhs.subspan(v_lower_offset_, dimensions_.v_lower),
             request.state.slack_s_lower,
             kkt_layout_.slack_lower_bounds,
             slack_structured_positions_,
             1.) ||
          !AddComplementarityRightHandSide(
             rhs,
             request.rhs.subspan(v_upper_offset_, dimensions_.v_upper),
             request.state.slack_s_upper,
             kkt_layout_.slack_upper_bounds,
             slack_structured_positions_,
             -1.) ||
          !AllFinite(rhs) )
      {
         return AssemblyFailure(
            EvaluationErrorCode::nonfinite_output,
            "primal-dual reduced right-hand side is nonfinite");
      }

      report.independent_full_inertia =
         CertifyReducedInertia(diagonal, regularization);
      return report;
   }

   EvaluationValue<StageStructuredWork> reconstruct_stage_direction(
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
            "primal-dual stage reconstruction role does not match its explicit metadata"
         });
      }
      if( structured_solution.size() != structured_to_full_.size() ||
          full_direction.size() != layout_.full_direction_dimension ||
          request.rhs.size() != layout_.full_direction_dimension )
      {
         return std::unexpected(EvaluationError{
            EvaluationErrorCode::dimension_mismatch,
            "primal-dual stage reconstruction has invalid dimensions or state"
         });
      }
      if( EvaluationResult valid = request.kkt.validate_state(request.state);
          !valid )
      {
         return std::unexpected(valid.error());
      }
      if( !ValidComplementarityState(request.state) )
      {
         return std::unexpected(EvaluationError{
            EvaluationErrorCode::nonfinite_output,
            "primal-dual stage reconstruction has invalid complementarity state"
         });
      }
      for( Index structured = 0;
           structured < structured_to_full_.size();
           ++structured )
      {
         full_direction[structured_to_full_[structured]] =
            structured_solution[structured];
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
      Index         primal_size,
      Index         constraint_size,
      Index         multiplier_offset,
      Index         block_size,
      Number*       block
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

   static bool AddComplementarityDiagonal(
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
            (rhs[bound] + primal_sign * multipliers[bound] *
               primal_direction[bounds[bound]]) /
            slacks[bound];
      }
   }

   std::optional<CertifiedInertia> CertifyReducedInertia(
      std::span<const Number>   diagonal,
      PrimalDualRegularization regularization
   ) const noexcept
   {
      if( (dimensions_.y_c != 0 && !(regularization.c > 0.)) ||
          (dimensions_.y_d != 0 && !(regularization.d > 0.)) )
      {
         return std::nullopt;
      }
      constexpr Number roundoff_multiplier = 128.;
      const StageNlpTopology& topology = derivatives_.topology();
      for( Index stage = 0; stage < topology.stages().size(); ++stage )
      {
         const OptimalControlStageDimensions dimensions = topology.stages()[stage];
         const Index primal_size = dimensions.controls + dimensions.states;
         const Index block_size = physical_block_sizes_[stage];
         const Number* block = diagonal.data() + diagonal_offsets_[stage];
         for( Index row = 0; row < primal_size; ++row )
         {
            Number off_diagonal_sum = 0.;
            for( Index column = 0; column < primal_size; ++column )
            {
               if( block[row * block_size + column] !=
                   block[column * block_size + row] )
               {
                  return std::nullopt;
               }
               if( row != column )
               {
                  off_diagonal_sum +=
                     std::abs(block[row * block_size + column]);
               }
            }
            const Number diagonal_value = block[row * block_size + row];
            const Number scale = std::max(
               1., std::abs(diagonal_value) + off_diagonal_sum);
            const Number margin = roundoff_multiplier *
               std::numeric_limits<Number>::epsilon() *
               static_cast<Number>(primal_size) * scale;
            if( !(diagonal_value - off_diagonal_sum > margin) )
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
               std::max(1., std::abs(value));
            if( !(value > margin) )
            {
               return std::nullopt;
            }
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

   void Configure(PrimalDualKktOperator& kkt)
   {
      Bind(
         kkt,
         PrimalDualStageKktLayoutWorkspace(
            derivatives_.topology(), kkt.layout(), options_));
   }

   void Bind(
      PrimalDualKktOperator&             kkt,
      PrimalDualStageKktLayoutWorkspace  workspace
   )
   {
      if( !workspace.configured() )
      {
         configuration_error_ = workspace.configuration_error();
         return;
      }
      const StageNlpTopology& topology = derivatives_.topology();
      StructureFingerprintResult topology_fingerprint = topology.fingerprint();
      if( !topology_fingerprint ||
          *topology_fingerprint != workspace.topology_fingerprint_ ||
          topology.source_structure() != workspace.source_structure_ ||
          topology.source_structure() != kkt.nlp_structure() ||
          options_ != workspace.options_ )
      {
         configuration_error_ =
            "prepared primal-dual stage layout does not match the topology";
         return;
      }
      if( kkt.layout() != workspace.expected_layout_ ||
          kkt.dimensions() != workspace.dimensions_ ||
          kkt.flat_dimension() != workspace.layout_.full_direction_dimension )
      {
         configuration_error_ =
            "prepared primal-dual stage layout does not match the live KKT";
         return;
      }
      StructureFingerprintResult fingerprint = kkt.structure_fingerprint();
      if( !fingerprint )
      {
         configuration_error_ = fingerprint.error().message;
         return;
      }
      workspace.layout_.kkt_fingerprint = *fingerprint;
      workspace.layout_.full_direction_overwrite_certified = true;
      layout_ = std::move(workspace.layout_);
      kkt_layout_ = std::move(workspace.expected_layout_);
      dimensions_ = workspace.dimensions_;
      physical_block_sizes_ = std::move(workspace.physical_block_sizes_);
      diagonal_offsets_ = std::move(workspace.diagonal_offsets_);
      lower_offsets_ = std::move(workspace.lower_offsets_);
      structured_to_full_ = std::move(workspace.structured_to_full_);
      primal_diagonal_offsets_ =
         std::move(workspace.primal_diagonal_offsets_);
      slack_diagonal_offsets_ =
         std::move(workspace.slack_diagonal_offsets_);
      primal_structured_positions_ =
         std::move(workspace.primal_structured_positions_);
      slack_structured_positions_ =
         std::move(workspace.slack_structured_positions_);
      diagonal_values_ = workspace.diagonal_values_;
      lower_values_ = workspace.lower_values_;
      x_offset_ = workspace.x_offset_;
      s_offset_ = workspace.s_offset_;
      z_lower_offset_ = workspace.z_lower_offset_;
      z_upper_offset_ = workspace.z_upper_offset_;
      v_lower_offset_ = workspace.v_lower_offset_;
      v_upper_offset_ = workspace.v_upper_offset_;
   }

   PreparedStageDerivativeProvider<Provider> derivatives_;
   PrimalDualStageKktAssemblerOptions options_;
   StageStructuredLayout layout_;
   PrimalDualLayout kkt_layout_;
   PrimalDualDimensions dimensions_{};
   std::vector<Index> physical_block_sizes_;
   std::vector<Index> diagonal_offsets_;
   std::vector<Index> lower_offsets_;
   std::vector<Index> structured_to_full_;
   std::vector<Index> primal_diagonal_offsets_;
   std::vector<Index> slack_diagonal_offsets_;
   std::vector<Index> primal_structured_positions_;
   std::vector<Index> slack_structured_positions_;
   Index diagonal_values_ = 0;
   Index lower_values_ = 0;
   Index x_offset_ = 0;
   Index s_offset_ = 0;
   Index z_lower_offset_ = 0;
   Index z_upper_offset_ = 0;
   Index v_lower_offset_ = 0;
   Index v_upper_offset_ = 0;
   std::string configuration_error_;
   bool derivatives_cached_ = false;
   std::uint64_t cached_numeric_revision_ = 0;
};
} // namespace Ipopt::Cxx23

#endif
