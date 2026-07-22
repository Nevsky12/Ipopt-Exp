// Copyright (C) 2026 Ipopt contributors.
// All Rights Reserved.
// This file is distributed under the Eclipse Public License.

#ifndef IPOPT_CXX23_BORDERED_STAGE_STRUCTURED_NLP_HPP
#define IPOPT_CXX23_BORDERED_STAGE_STRUCTURED_NLP_HPP

#include <ipopt/cxx23/stage_structured_nlp.hpp>

#include <algorithm>
#include <concepts>
#include <cstdint>
#include <expected>
#include <limits>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace Ipopt::Cxx23
{
/** Packed derivatives for a stage chain with explicitly global primals. */
struct BorderedStageDerivativeStorage
{
   Index stage_hessians = 0;
   Index cross_stage_hessians = 0;
   Index local_global_hessians = 0;
   Index global_hessian = 0;
   Index dynamics_jacobians_transposed = 0;
   Index dynamics_next_state_jacobians = 0;
   Index path_equality_jacobians_transposed = 0;
   Index path_inequality_jacobians_transposed = 0;
   /** Dense global-by-generic-constraint storage. */
   Index global_jacobians_transposed = 0;
};

/** Immutable OCP topology for an arrowhead caused by a few global variables.
 *
 * ordering.primal contains every stage-local generic NLP variable exactly
 * once; global_primal_variables contains the disjoint remainder. Constraint
 * ordering retains the same canonical dynamics/path contract as
 * StageNlpTopology. A global variable may occur in any stage Hessian or
 * Jacobian row, producing the dense border rather than a rejected cross-stage
 * entry.
 */
class BorderedStageNlpTopology
{
public:
   BorderedStageNlpTopology(
      NlpStructure                               source_structure,
      std::vector<OptimalControlStageDimensions> stages,
      StageNlpOrdering                           ordering,
      std::vector<Index>                         global_primal_variables,
      std::uint64_t                              stage_revision
   )
      : source_structure_(source_structure),
        stages_(std::move(stages)),
        ordering_(std::move(ordering)),
        global_primal_variables_(std::move(global_primal_variables)),
        stage_revision_(stage_revision)
   {
      Configure();
   }

   bool configured() const noexcept
   {
      return configuration_error_.empty();
   }

   const std::string& configuration_error() const noexcept
   {
      return configuration_error_;
   }

   NlpStructure source_structure() const noexcept
   {
      return source_structure_;
   }

   std::uint64_t stage_revision() const noexcept
   {
      return stage_revision_;
   }

   std::span<const OptimalControlStageDimensions> stages() const noexcept
   {
      return stages_;
   }

   const StageNlpOrdering& ordering() const noexcept
   {
      return ordering_;
   }

   std::span<const Index> global_primal_variables() const noexcept
   {
      return global_primal_variables_;
   }

   BorderedStageDerivativeStorage derivative_storage() const noexcept
   {
      return derivative_storage_;
   }

   Index dynamics_constraints() const noexcept
   {
      return dynamics_constraints_;
   }

   Index path_equality_constraints() const noexcept
   {
      return path_equality_constraints_;
   }

   Index path_inequality_constraints() const noexcept
   {
      return path_inequality_constraints_;
   }

   std::span<const Index> primal_offsets() const noexcept
   {
      return primal_offsets_;
   }

   std::span<const Index> dynamics_offsets() const noexcept
   {
      return dynamics_offsets_;
   }

   std::span<const Index> path_equality_offsets() const noexcept
   {
      return path_equality_offsets_;
   }

   std::span<const Index> path_inequality_offsets() const noexcept
   {
      return path_inequality_offsets_;
   }

   std::span<const Index> hessian_offsets() const noexcept
   {
      return hessian_offsets_;
   }

   std::span<const Index> local_global_hessian_offsets() const noexcept
   {
      return local_global_hessian_offsets_;
   }

   std::span<const Index> cross_stage_hessian_offsets() const noexcept
   {
      return cross_stage_hessian_offsets_;
   }

   std::span<const Index> dynamics_jacobian_offsets() const noexcept
   {
      return dynamics_jacobian_offsets_;
   }

   std::span<const Index>
   dynamics_next_state_jacobian_offsets() const noexcept
   {
      return dynamics_next_state_jacobian_offsets_;
   }

   std::span<const Index> path_equality_jacobian_offsets() const noexcept
   {
      return path_equality_jacobian_offsets_;
   }

   std::span<const Index> path_inequality_jacobian_offsets() const noexcept
   {
      return path_inequality_jacobian_offsets_;
   }

   StructureFingerprintResult fingerprint() const
   {
      if( !configured() )
      {
         return std::unexpected(EvaluationError{
            EvaluationErrorCode::invalid_layout,
            configuration_error_
         });
      }
      StructureFingerprint result{
         1469598103934665603ULL,
         0xbb67ae8584caa73bULL
      };
      detail::HashWord(result, 0x424f524445524e4cULL);
      detail::HashWord(result, source_structure_.variables);
      detail::HashWord(result, source_structure_.constraints);
      detail::HashWord(result, source_structure_.jacobian_nonzeros);
      detail::HashWord(result, source_structure_.hessian_nonzeros);
      detail::HashWord(result, source_structure_.revision);
      detail::HashWord(result, stage_revision_);
      detail::HashWord(result, stages_.size());
      for( const OptimalControlStageDimensions& stage : stages_ )
      {
         detail::HashWord(result, stage.controls);
         detail::HashWord(result, stage.states);
         detail::HashWord(result, stage.path_equalities);
         detail::HashWord(result, stage.path_inequalities);
      }
      HashIndices(result, ordering_.primal);
      HashIndices(result, global_primal_variables_);
      HashIndices(result, ordering_.dynamics);
      HashIndices(result, ordering_.path_equalities);
      HashIndices(result, ordering_.path_inequalities);
      return result;
   }

private:
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

   static void HashIndices(
      StructureFingerprint&  fingerprint,
      std::span<const Index> indices
   ) noexcept
   {
      detail::HashWord(fingerprint, indices.size());
      for( Index index : indices )
      {
         detail::HashWord(fingerprint, index);
      }
   }

   static bool MarkUnique(
      std::span<const Index> indices,
      std::vector<bool>&     seen
   )
   {
      for( Index index : indices )
      {
         if( index >= seen.size() || seen[index] )
         {
            return false;
         }
         seen[index] = true;
      }
      return true;
   }

   void Configure()
   {
      if( stages_.empty() )
      {
         configuration_error_ = "bordered stage topology must contain a stage";
         return;
      }
      if( global_primal_variables_.empty() )
      {
         configuration_error_ =
            "bordered stage topology requires a global primal variable";
         return;
      }
      if( stage_revision_ == 0 )
      {
         configuration_error_ =
            "bordered stage topology requires a nonzero revision";
         return;
      }

      primal_offsets_.push_back(0);
      dynamics_offsets_.push_back(0);
      path_equality_offsets_.push_back(0);
      path_inequality_offsets_.push_back(0);
      hessian_offsets_.push_back(0);
      cross_stage_hessian_offsets_.push_back(0);
      local_global_hessian_offsets_.push_back(0);
      dynamics_jacobian_offsets_.push_back(0);
      dynamics_next_state_jacobian_offsets_.push_back(0);
      path_equality_jacobian_offsets_.push_back(0);
      path_inequality_jacobian_offsets_.push_back(0);

      Index primal = 0;
      Index dynamics = 0;
      Index path_equalities = 0;
      Index path_inequalities = 0;
      Index hessians = 0;
      Index cross_stage_hessians = 0;
      Index local_global_hessians = 0;
      Index dynamics_jacobians = 0;
      Index dynamics_next_state_jacobians = 0;
      Index path_equality_jacobians = 0;
      Index path_inequality_jacobians = 0;
      for( Index stage_index = 0; stage_index < stages_.size(); ++stage_index )
      {
         const OptimalControlStageDimensions stage = stages_[stage_index];
         if( stage.states == 0 ||
             stage.controls > std::numeric_limits<Index>::max() - stage.states )
         {
            configuration_error_ =
               "each bordered stage requires local states and a representable size";
            return;
         }
         const Index stage_variables = stage.controls + stage.states;
         Index square = 0;
         Index local_global = 0;
         Index path_equality_values = 0;
         Index path_inequality_values = 0;
         if( !CheckedAdd(primal, stage_variables, primal) ||
             !CheckedMultiply(stage_variables, stage_variables, square) ||
             !CheckedAdd(hessians, square, hessians) ||
             !CheckedMultiply(
                stage_variables,
                global_primal_variables_.size(),
                local_global) ||
             !CheckedAdd(
                local_global_hessians,
                local_global,
                local_global_hessians) ||
             !CheckedAdd(
                path_equalities, stage.path_equalities, path_equalities) ||
             !CheckedAdd(
                path_inequalities, stage.path_inequalities, path_inequalities) ||
             !CheckedMultiply(
                stage_variables,
                stage.path_equalities,
                path_equality_values) ||
             !CheckedAdd(
                path_equality_jacobians,
                path_equality_values,
                path_equality_jacobians) ||
             !CheckedMultiply(
                stage_variables,
                stage.path_inequalities,
                path_inequality_values) ||
             !CheckedAdd(
                path_inequality_jacobians,
                path_inequality_values,
                path_inequality_jacobians) )
         {
            configuration_error_ = "bordered stage derivative storage overflows Index";
            return;
         }
         if( stage_index + 1 < stages_.size() )
         {
            const Index next_states = stages_[stage_index + 1].states;
            const Index next_variables = stages_[stage_index + 1].controls +
               stages_[stage_index + 1].states;
            Index values = 0;
            Index next_state_values = 0;
            Index cross_hessian_values = 0;
            if( next_states == 0 ||
                !CheckedAdd(dynamics, next_states, dynamics) ||
                !CheckedMultiply(stage_variables, next_states, values) ||
                !CheckedAdd(
                   dynamics_jacobians, values, dynamics_jacobians) ||
                !CheckedMultiply(
                   next_states, next_states, next_state_values) ||
                !CheckedAdd(
                   dynamics_next_state_jacobians,
                   next_state_values,
                   dynamics_next_state_jacobians) ||
                !CheckedMultiply(
                   next_variables,
                   stage_variables,
                   cross_hessian_values) ||
                !CheckedAdd(
                   cross_stage_hessians,
                   cross_hessian_values,
                   cross_stage_hessians) )
            {
               configuration_error_ =
                  "bordered stage dynamics storage overflows Index";
               return;
            }
            dynamics_offsets_.push_back(dynamics);
            dynamics_jacobian_offsets_.push_back(dynamics_jacobians);
            dynamics_next_state_jacobian_offsets_.push_back(
               dynamics_next_state_jacobians);
            cross_stage_hessian_offsets_.push_back(cross_stage_hessians);
         }
         primal_offsets_.push_back(primal);
         path_equality_offsets_.push_back(path_equalities);
         path_inequality_offsets_.push_back(path_inequalities);
         hessian_offsets_.push_back(hessians);
         local_global_hessian_offsets_.push_back(local_global_hessians);
         path_equality_jacobian_offsets_.push_back(path_equality_jacobians);
         path_inequality_jacobian_offsets_.push_back(path_inequality_jacobians);
      }

      Index variable_total = 0;
      Index constraint_total = 0;
      Index global_hessian = 0;
      Index global_jacobians = 0;
      if( !CheckedAdd(
             primal, global_primal_variables_.size(), variable_total) ||
          !CheckedAdd(dynamics, path_equalities, constraint_total) ||
          !CheckedAdd(
             constraint_total, path_inequalities, constraint_total) ||
          !CheckedMultiply(
             global_primal_variables_.size(),
             global_primal_variables_.size(),
             global_hessian) ||
          !CheckedMultiply(
             global_primal_variables_.size(),
             source_structure_.constraints,
             global_jacobians) )
      {
         configuration_error_ = "bordered stage totals overflow Index";
         return;
      }
      if( variable_total != source_structure_.variables ||
          constraint_total != source_structure_.constraints )
      {
         configuration_error_ =
            "bordered stage totals do not match the generic NLP dimensions";
         return;
      }
      if( ordering_.primal.size() != primal ||
          ordering_.dynamics.size() != dynamics ||
          ordering_.path_equalities.size() != path_equalities ||
          ordering_.path_inequalities.size() != path_inequalities )
      {
         configuration_error_ =
            "bordered stage ordering has the wrong map dimensions";
         return;
      }
      std::vector<bool> variable_seen(source_structure_.variables, false);
      if( !MarkUnique(ordering_.primal, variable_seen) ||
          !MarkUnique(global_primal_variables_, variable_seen) ||
          !std::ranges::all_of(
             variable_seen, [](bool value) { return value; }) )
      {
         configuration_error_ =
            "local and global primal maps are not a partition of NLP variables";
         return;
      }
      std::vector<bool> constraint_seen(source_structure_.constraints, false);
      if( !MarkUnique(ordering_.dynamics, constraint_seen) ||
          !MarkUnique(ordering_.path_equalities, constraint_seen) ||
          !MarkUnique(ordering_.path_inequalities, constraint_seen) ||
          !std::ranges::all_of(
             constraint_seen, [](bool value) { return value; }) )
      {
         configuration_error_ =
            "bordered stage constraint maps are not a partition of the NLP";
         return;
      }

      dynamics_constraints_ = dynamics;
      path_equality_constraints_ = path_equalities;
      path_inequality_constraints_ = path_inequalities;
      derivative_storage_ = {
         hessians,
         cross_stage_hessians,
         local_global_hessians,
         global_hessian,
         dynamics_jacobians,
         dynamics_next_state_jacobians,
         path_equality_jacobians,
         path_inequality_jacobians,
         global_jacobians
      };
   }

   NlpStructure source_structure_;
   std::vector<OptimalControlStageDimensions> stages_;
   StageNlpOrdering ordering_;
   std::vector<Index> global_primal_variables_;
   std::uint64_t stage_revision_ = 0;
   std::string configuration_error_;
   BorderedStageDerivativeStorage derivative_storage_;
   Index dynamics_constraints_ = 0;
   Index path_equality_constraints_ = 0;
   Index path_inequality_constraints_ = 0;
   std::vector<Index> primal_offsets_;
   std::vector<Index> dynamics_offsets_;
   std::vector<Index> path_equality_offsets_;
   std::vector<Index> path_inequality_offsets_;
   std::vector<Index> hessian_offsets_;
   std::vector<Index> cross_stage_hessian_offsets_;
   std::vector<Index> local_global_hessian_offsets_;
   std::vector<Index> dynamics_jacobian_offsets_;
   std::vector<Index> dynamics_next_state_jacobian_offsets_;
   std::vector<Index> path_equality_jacobian_offsets_;
   std::vector<Index> path_inequality_jacobian_offsets_;
};

struct BorderedStageDerivativeBuffers
{
   std::span<Number> stage_hessians;
   std::span<Number> cross_stage_hessians;
   std::span<Number> local_global_hessians;
   std::span<Number> global_hessian;
   std::span<Number> dynamics_jacobians_transposed;
   std::span<Number> dynamics_next_state_jacobians;
   std::span<Number> path_equality_jacobians_transposed;
   std::span<Number> path_inequality_jacobians_transposed;
   std::span<Number> global_jacobians_transposed;
};

struct BorderedStageDerivativeView
{
   std::span<const Number> stage_hessians;
   std::span<const Number> cross_stage_hessians;
   std::span<const Number> local_global_hessians;
   std::span<const Number> global_hessian;
   std::span<const Number> dynamics_jacobians_transposed;
   std::span<const Number> dynamics_next_state_jacobians;
   std::span<const Number> path_equality_jacobians_transposed;
   std::span<const Number> path_inequality_jacobians_transposed;
   std::span<const Number> global_jacobians_transposed;
};

template <class Provider>
concept HasBorderedStageNlpTopology = requires(const Provider& provider)
{
   { provider.bordered_stage_nlp_topology() } noexcept
      -> std::same_as<const BorderedStageNlpTopology&>;
};

template <class Provider>
concept PointBorderedStageDerivativeProvider = requires(
   Provider&                       provider,
   KktEvaluationPoint             point,
   BorderedStageDerivativeBuffers buffers
)
{
   { provider.eval_bordered_stage_derivatives(point, buffers) }
      -> std::same_as<EvaluationResult>;
};

template <class Provider>
concept KktBorderedStageDerivativeProvider = requires(
   Provider&                       provider,
   PrimalDualKktOperator&          kkt,
   KktEvaluationPoint             point,
   BorderedStageDerivativeBuffers buffers
)
{
   { provider.eval_bordered_stage_derivatives(kkt, point, buffers) }
      -> std::same_as<EvaluationResult>;
};

template <class Provider>
concept BorderedStageDerivativeProvider =
   HasBorderedStageNlpTopology<Provider> &&
   (PointBorderedStageDerivativeProvider<Provider> ||
    KktBorderedStageDerivativeProvider<Provider>);

/** Owns one bordered provider and all topology-sized derivative buffers. */
template <BorderedStageDerivativeProvider Provider>
class PreparedBorderedStageDerivativeProvider
{
public:
   explicit PreparedBorderedStageDerivativeProvider(Provider provider)
      : provider_(std::move(provider)),
        stage_hessians_(Storage().stage_hessians),
        cross_stage_hessians_(Storage().cross_stage_hessians),
        local_global_hessians_(Storage().local_global_hessians),
        global_hessian_(Storage().global_hessian),
        dynamics_jacobians_(Storage().dynamics_jacobians_transposed),
        dynamics_next_state_jacobians_(
           Storage().dynamics_next_state_jacobians),
        path_equality_jacobians_(
           Storage().path_equality_jacobians_transposed),
        path_inequality_jacobians_(
           Storage().path_inequality_jacobians_transposed),
        global_jacobians_(Storage().global_jacobians_transposed)
   {
   }

   const BorderedStageNlpTopology& topology() const noexcept
   {
      return provider_.bordered_stage_nlp_topology();
   }

   EvaluationValue<BorderedStageDerivativeView> evaluate(
      PrimalDualKktOperator& kkt,
      KktEvaluationPoint     point
   )
   {
      BorderedStageDerivativeBuffers buffers{
         stage_hessians_,
         cross_stage_hessians_,
         local_global_hessians_,
         global_hessian_,
         dynamics_jacobians_,
         dynamics_next_state_jacobians_,
         path_equality_jacobians_,
         path_inequality_jacobians_,
         global_jacobians_
      };
      EvaluationResult result;
      if constexpr( KktBorderedStageDerivativeProvider<Provider> )
      {
         result = provider_.eval_bordered_stage_derivatives(kkt, point, buffers);
      }
      else
      {
         result = provider_.eval_bordered_stage_derivatives(point, buffers);
      }
      if( !result )
      {
         return std::unexpected(result.error());
      }
      return current_view();
   }

   BorderedStageDerivativeView current_view() const noexcept
   {
      return {
         stage_hessians_,
         cross_stage_hessians_,
         local_global_hessians_,
         global_hessian_,
         dynamics_jacobians_,
         dynamics_next_state_jacobians_,
         path_equality_jacobians_,
         path_inequality_jacobians_,
         global_jacobians_
      };
   }

private:
   BorderedStageDerivativeStorage Storage() const noexcept
   {
      return provider_.bordered_stage_nlp_topology().derivative_storage();
   }

   Provider provider_;
   std::vector<Number> stage_hessians_;
   std::vector<Number> cross_stage_hessians_;
   std::vector<Number> local_global_hessians_;
   std::vector<Number> global_hessian_;
   std::vector<Number> dynamics_jacobians_;
   std::vector<Number> dynamics_next_state_jacobians_;
   std::vector<Number> path_equality_jacobians_;
   std::vector<Number> path_inequality_jacobians_;
   std::vector<Number> global_jacobians_;
};

template <BorderedStageDerivativeProvider Provider>
PreparedBorderedStageDerivativeProvider<Provider>
PrepareBorderedStageDerivativeProvider(Provider provider)
{
   return PreparedBorderedStageDerivativeProvider<Provider>(std::move(provider));
}
} // namespace Ipopt::Cxx23

#endif
