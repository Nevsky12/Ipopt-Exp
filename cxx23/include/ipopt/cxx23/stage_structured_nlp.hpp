// Copyright (C) 2026 Ipopt contributors.
// All Rights Reserved.
// This file is distributed under the Eclipse Public License.

#ifndef IPOPT_CXX23_STAGE_STRUCTURED_NLP_HPP
#define IPOPT_CXX23_STAGE_STRUCTURED_NLP_HPP

#include <ipopt/cxx23/kkt_operator.hpp>

#include <algorithm>
#include <cmath>
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
class PrimalDualKktOperator;

/** Tangent-space dimensions for one OCP stage. */
struct OptimalControlStageDimensions
{
   Index controls = 0;
   Index states = 0;
   Index path_equalities = 0;
   Index path_inequalities = 0;

   friend bool operator==(
      const OptimalControlStageDimensions&,
      const OptimalControlStageDimensions&) = default;
};

/** Maps canonical stage order to the generic NLP coordinate order.
 *
 * Canonical primal order is [u_0,x_0,...,u_K-1,x_K-1]. Dynamics entries are
 * grouped by transition and have states[k+1] rows. The three constraint maps
 * together must be a permutation of every generic NLP constraint row.
 */
struct StageNlpOrdering
{
   std::vector<Index> primal;
   std::vector<Index> dynamics;
   std::vector<Index> path_equalities;
   std::vector<Index> path_inequalities;

   friend bool operator==(const StageNlpOrdering&, const StageNlpOrdering&) = default;
};

struct StageDerivativeStorage
{
   Index stage_hessians = 0;
   Index dynamics_jacobians_transposed = 0;
   Index path_equality_jacobians_transposed = 0;
   Index path_inequality_jacobians_transposed = 0;
};

/** Immutable, validated OCP topology layered over one generic NLP structure.
 *
 * Hessians are complete row-major (u_k,x_k)-square blocks. Dynamics storage
 * contains row-major transposed current-stage Jacobians with dimensions
 * (nu_k+nx_k) by nx_{k+1}; the next-state Jacobian is canonically -I, matching
 * FATROP's x_{k+1}-f_k convention. Path Jacobians are likewise stored
 * transposed as (nu_k+nx_k) by ng_k blocks.
 */
class StageNlpTopology
{
public:
   StageNlpTopology(
      NlpStructure                               source_structure,
      std::vector<OptimalControlStageDimensions> stages,
      StageNlpOrdering                           ordering,
      std::uint64_t                              stage_revision
   )
      : source_structure_(source_structure),
        stages_(std::move(stages)),
        ordering_(std::move(ordering)),
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

   StageDerivativeStorage derivative_storage() const noexcept
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

   std::span<const Index> dynamics_jacobian_offsets() const noexcept
   {
      return dynamics_jacobian_offsets_;
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
         0x6a09e667f3bcc909ULL
      };
      detail::HashWord(result, 0x53544147454e4c50ULL);
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
      StructureFingerprint&    fingerprint,
      std::span<const Index>   indices
   ) noexcept
   {
      detail::HashWord(fingerprint, indices.size());
      for( Index index : indices )
      {
         detail::HashWord(fingerprint, index);
      }
   }

   void Configure()
   {
      if( stages_.empty() )
      {
         configuration_error_ = "stage NLP topology must contain a stage";
         return;
      }
      if( stage_revision_ == 0 )
      {
         configuration_error_ = "stage NLP topology requires a nonzero revision";
         return;
      }

      primal_offsets_.reserve(stages_.size() + 1);
      dynamics_offsets_.reserve(stages_.size());
      path_equality_offsets_.reserve(stages_.size() + 1);
      path_inequality_offsets_.reserve(stages_.size() + 1);
      hessian_offsets_.reserve(stages_.size() + 1);
      dynamics_jacobian_offsets_.reserve(stages_.size());
      path_equality_jacobian_offsets_.reserve(stages_.size() + 1);
      path_inequality_jacobian_offsets_.reserve(stages_.size() + 1);
      primal_offsets_.push_back(0);
      dynamics_offsets_.push_back(0);
      path_equality_offsets_.push_back(0);
      path_inequality_offsets_.push_back(0);
      hessian_offsets_.push_back(0);
      dynamics_jacobian_offsets_.push_back(0);
      path_equality_jacobian_offsets_.push_back(0);
      path_inequality_jacobian_offsets_.push_back(0);

      Index primal = 0;
      Index dynamics = 0;
      Index path_equalities = 0;
      Index path_inequalities = 0;
      Index hessians = 0;
      Index dynamics_jacobians = 0;
      Index path_equality_jacobians = 0;
      Index path_inequality_jacobians = 0;
      for( Index stage_index = 0; stage_index < stages_.size(); ++stage_index )
      {
         const OptimalControlStageDimensions stage = stages_[stage_index];
         if( stage.states == 0 ||
             stage.controls > std::numeric_limits<Index>::max() - stage.states )
         {
            configuration_error_ =
               "each stage requires state variables and a representable primal size";
            return;
         }
         const Index stage_variables = stage.controls + stage.states;
         Index hessian_values = 0;
         Index path_equality_values = 0;
         Index path_inequality_values = 0;
         if( !CheckedAdd(primal, stage_variables, primal) ||
             !CheckedMultiply(stage_variables, stage_variables, hessian_values) ||
             !CheckedAdd(hessians, hessian_values, hessians) ||
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
            configuration_error_ = "stage NLP topology overflows Index";
            return;
         }
         if( stage_index + 1 < stages_.size() )
         {
            const Index next_states = stages_[stage_index + 1].states;
            Index dynamics_values = 0;
            if( next_states == 0 ||
                !CheckedAdd(dynamics, next_states, dynamics) ||
                !CheckedMultiply(
                   stage_variables, next_states, dynamics_values) ||
                !CheckedAdd(
                   dynamics_jacobians,
                   dynamics_values,
                   dynamics_jacobians) )
            {
               configuration_error_ = "stage dynamics topology overflows Index";
               return;
            }
            dynamics_offsets_.push_back(dynamics);
            dynamics_jacobian_offsets_.push_back(dynamics_jacobians);
         }
         primal_offsets_.push_back(primal);
         path_equality_offsets_.push_back(path_equalities);
         path_inequality_offsets_.push_back(path_inequalities);
         hessian_offsets_.push_back(hessians);
         path_equality_jacobian_offsets_.push_back(path_equality_jacobians);
         path_inequality_jacobian_offsets_.push_back(path_inequality_jacobians);
      }

      Index constraint_total = 0;
      if( !CheckedAdd(dynamics, path_equalities, constraint_total) ||
          !CheckedAdd(constraint_total, path_inequalities, constraint_total) )
      {
         configuration_error_ = "stage constraint topology overflows Index";
         return;
      }
      if( source_structure_.variables != primal ||
          source_structure_.constraints != constraint_total )
      {
         configuration_error_ =
            "stage totals do not match the generic NLP dimensions";
         return;
      }
      if( ordering_.primal.size() != primal ||
          ordering_.dynamics.size() != dynamics ||
          ordering_.path_equalities.size() != path_equalities ||
          ordering_.path_inequalities.size() != path_inequalities )
      {
         configuration_error_ = "stage ordering has the wrong map dimensions";
         return;
      }
      if( !IsPermutation(ordering_.primal, primal) )
      {
         configuration_error_ =
            "stage primal ordering is not a permutation of NLP variables";
         return;
      }
      std::vector<bool> constraint_seen(constraint_total, false);
      if( !MarkUnique(ordering_.dynamics, constraint_seen) ||
          !MarkUnique(ordering_.path_equalities, constraint_seen) ||
          !MarkUnique(ordering_.path_inequalities, constraint_seen) ||
          !std::ranges::all_of(constraint_seen, [](bool value) { return value; }) )
      {
         configuration_error_ =
            "stage constraint ordering is not a partition of NLP constraints";
         return;
      }

      dynamics_constraints_ = dynamics;
      path_equality_constraints_ = path_equalities;
      path_inequality_constraints_ = path_inequalities;
      derivative_storage_ = {
         hessians,
         dynamics_jacobians,
         path_equality_jacobians,
         path_inequality_jacobians
      };
   }

   static bool IsPermutation(std::span<const Index> indices, Index size)
   {
      std::vector<bool> seen(size, false);
      return MarkUnique(indices, seen) &&
         std::ranges::all_of(seen, [](bool value) { return value; });
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

   NlpStructure source_structure_;
   std::vector<OptimalControlStageDimensions> stages_;
   StageNlpOrdering ordering_;
   std::uint64_t stage_revision_;
   std::string configuration_error_;
   StageDerivativeStorage derivative_storage_;
   Index dynamics_constraints_ = 0;
   Index path_equality_constraints_ = 0;
   Index path_inequality_constraints_ = 0;
   std::vector<Index> primal_offsets_;
   std::vector<Index> dynamics_offsets_;
   std::vector<Index> path_equality_offsets_;
   std::vector<Index> path_inequality_offsets_;
   std::vector<Index> hessian_offsets_;
   std::vector<Index> dynamics_jacobian_offsets_;
   std::vector<Index> path_equality_jacobian_offsets_;
   std::vector<Index> path_inequality_jacobian_offsets_;
};

struct StageDerivativeBuffers
{
   std::span<Number> stage_hessians;
   std::span<Number> dynamics_jacobians_transposed;
   std::span<Number> path_equality_jacobians_transposed;
   std::span<Number> path_inequality_jacobians_transposed;
};

struct StageDerivativeView
{
   std::span<const Number> stage_hessians;
   std::span<const Number> dynamics_jacobians_transposed;
   std::span<const Number> path_equality_jacobians_transposed;
   std::span<const Number> path_inequality_jacobians_transposed;
};

template <class Provider>
concept HasStageNlpTopology = requires(
   const Provider& const_provider
)
{
   { const_provider.stage_nlp_topology() } noexcept
      -> std::same_as<const StageNlpTopology&>;
};

template <class Provider>
concept PointStageDerivativeProvider = requires(
   Provider&       provider,
   KktEvaluationPoint point,
   StageDerivativeBuffers buffers
)
{
   { provider.eval_stage_derivatives(point, buffers) }
      -> std::same_as<EvaluationResult>;
};

template <class Provider>
concept KktStageDerivativeProvider = requires(
   Provider&              provider,
   PrimalDualKktOperator& kkt,
   KktEvaluationPoint     point,
   StageDerivativeBuffers buffers
)
{
   { provider.eval_stage_derivatives(kkt, point, buffers) }
      -> std::same_as<EvaluationResult>;
};

template <class Provider>
concept StageDerivativeProvider = HasStageNlpTopology<Provider> &&
   (PointStageDerivativeProvider<Provider> ||
    KktStageDerivativeProvider<Provider>);

/** Owns one stage provider and all packed derivative storage.
 *
 * The provider remains a compile-time capability inside the assembler; the
 * coarse candidate backend is AnyAny-erased once. This avoids type erasure in
 * stage/nonzero loops while retaining the external plugin boundary. Returned
 * views remain valid only until the next evaluation, move, or destruction;
 * providers must not retain the input point or output-buffer spans.
 */
template <StageDerivativeProvider Provider>
class PreparedStageDerivativeProvider
{
public:
   explicit PreparedStageDerivativeProvider(Provider provider)
      : provider_(std::move(provider)),
        stage_hessians_(
           provider_.stage_nlp_topology().derivative_storage().stage_hessians),
        dynamics_jacobians_(
           provider_.stage_nlp_topology()
              .derivative_storage()
              .dynamics_jacobians_transposed),
        path_equality_jacobians_(
           provider_.stage_nlp_topology()
              .derivative_storage()
              .path_equality_jacobians_transposed),
        path_inequality_jacobians_(
           provider_.stage_nlp_topology()
              .derivative_storage()
              .path_inequality_jacobians_transposed)
   {
   }

   const StageNlpTopology& topology() const noexcept
   {
      return provider_.stage_nlp_topology();
   }

   StageDerivativeView current_view() const noexcept
   {
      return {
         stage_hessians_,
         dynamics_jacobians_,
         path_equality_jacobians_,
         path_inequality_jacobians_
      };
   }

   EvaluationValue<StageDerivativeView> evaluate(KktEvaluationPoint point)
      requires PointStageDerivativeProvider<Provider>
   {
      return Evaluate(
         point,
         [&](StageDerivativeBuffers buffers)
         {
            return provider_.eval_stage_derivatives(point, buffers);
         });
   }

   EvaluationValue<StageDerivativeView> evaluate(
      PrimalDualKktOperator& kkt,
      KktEvaluationPoint    point
   )
   {
      return Evaluate(
         point,
         [&](StageDerivativeBuffers buffers)
         {
            if constexpr( KktStageDerivativeProvider<Provider> )
            {
               return provider_.eval_stage_derivatives(kkt, point, buffers);
            }
            else
            {
               return provider_.eval_stage_derivatives(point, buffers);
            }
         });
   }

private:
   template <class EvaluateProvider>
   EvaluationValue<StageDerivativeView> Evaluate(
      KktEvaluationPoint point,
      EvaluateProvider&& evaluate_provider
   )
   {
      const StageNlpTopology& topology = provider_.stage_nlp_topology();
      if( !topology.configured() )
      {
         return std::unexpected(EvaluationError{
            EvaluationErrorCode::invalid_layout,
            topology.configuration_error()
         });
      }
      const NlpStructure source = topology.source_structure();
      if( point.x.size() != source.variables )
      {
         return std::unexpected(EvaluationError{
            EvaluationErrorCode::dimension_mismatch,
            "stage derivative point has the wrong primal dimension"
         });
      }
      if( point.constraint_multipliers.size() != source.constraints )
      {
         return std::unexpected(EvaluationError{
            EvaluationErrorCode::dimension_mismatch,
            "stage derivative point has the wrong multiplier dimension"
         });
      }
      EvaluationResult evaluated = std::forward<EvaluateProvider>(
         evaluate_provider)({
            stage_hessians_,
            dynamics_jacobians_,
            path_equality_jacobians_,
            path_inequality_jacobians_
         });
      if( !evaluated )
      {
         return std::unexpected(evaluated.error());
      }
      if( !AllFinite(stage_hessians_) ||
          !AllFinite(dynamics_jacobians_) ||
          !AllFinite(path_equality_jacobians_) ||
          !AllFinite(path_inequality_jacobians_) )
      {
         return std::unexpected(EvaluationError{
            EvaluationErrorCode::nonfinite_output,
            "stage derivative provider returned a nonfinite coefficient"
         });
      }
      return current_view();
   }

   static bool AllFinite(std::span<const Number> values) noexcept
   {
      return std::ranges::all_of(
         values, [](Number value) { return std::isfinite(value); });
   }

   Provider provider_;
   std::vector<Number> stage_hessians_;
   std::vector<Number> dynamics_jacobians_;
   std::vector<Number> path_equality_jacobians_;
   std::vector<Number> path_inequality_jacobians_;
};

template <StageDerivativeProvider Provider>
PreparedStageDerivativeProvider<Provider> PrepareStageDerivativeProvider(
   Provider provider
)
{
   return PreparedStageDerivativeProvider<Provider>(std::move(provider));
}
} // namespace Ipopt::Cxx23

#endif
