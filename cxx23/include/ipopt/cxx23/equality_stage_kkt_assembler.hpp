// Copyright (C) 2026 Ipopt contributors.
// All Rights Reserved.
// This file is distributed under the Eclipse Public License.

#ifndef IPOPT_CXX23_EQUALITY_STAGE_KKT_ASSEMBLER_HPP
#define IPOPT_CXX23_EQUALITY_STAGE_KKT_ASSEMBLER_HPP

#include <ipopt/cxx23/stage_structured_candidate_backend.hpp>
#include <ipopt/cxx23/stage_structured_nlp.hpp>

#include <algorithm>
#include <cstdint>
#include <expected>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace Ipopt::Cxx23
{
struct EqualityStageKktAssemblerOptions
{
   bool reverse_stage_order = false;

   friend bool operator==(
      const EqualityStageKktAssemblerOptions&,
      const EqualityStageKktAssemblerOptions&) = default;
};

/** Precomputed equality-stage permutation and packed block topology.
 *
 * This allocation bundle can be built while a stage topology and expected
 * equality ordering are known, before a live KKT matrix is available. Binding
 * still checks the complete topology fingerprint, KKT dimensions, equality
 * ordering, and live KKT fingerprint before consuming the bundle.
 */
class EqualityStageKktLayoutWorkspace
{
public:
   EqualityStageKktLayoutWorkspace(
      const StageNlpTopology&          topology,
      std::vector<Index>               equality_constraints,
      EqualityStageKktAssemblerOptions options = {}
   )
      : source_structure_(topology.source_structure()),
        equality_constraints_(std::move(equality_constraints)),
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

private:
   template <StageDerivativeProvider>
   friend class EqualityStageKktAssembler;

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

   void Configure(const StageNlpTopology& topology)
   {
      if( !topology.configured() )
      {
         configuration_error_ = topology.configuration_error();
         return;
      }
      if( topology.path_inequality_constraints() != 0 )
      {
         configuration_error_ =
            "equality-stage assembler does not support path inequalities";
         return;
      }
      StructureFingerprintResult topology_fingerprint = topology.fingerprint();
      if( !topology_fingerprint )
      {
         configuration_error_ = topology_fingerprint.error().message;
         return;
      }
      topology_fingerprint_ = *topology_fingerprint;
      if( equality_constraints_.size() != source_structure_.constraints )
      {
         configuration_error_ =
            "equality-stage KKT equality map has the wrong dimension";
         return;
      }

      std::vector<Index> constraint_to_multiplier(
         source_structure_.constraints, source_structure_.constraints);
      for( Index multiplier = 0;
           multiplier < equality_constraints_.size();
           ++multiplier )
      {
         const Index constraint = equality_constraints_[multiplier];
         if( constraint >= source_structure_.constraints ||
             constraint_to_multiplier[constraint] !=
                source_structure_.constraints )
         {
            configuration_error_ =
               "equality-stage KKT equality map is not a permutation";
            return;
         }
         constraint_to_multiplier[constraint] = multiplier;
      }

      if( !CheckedAdd(
             source_structure_.variables,
             source_structure_.constraints,
             layout_.full_direction_dimension) )
      {
         configuration_error_ =
            "equality-stage full KKT dimension overflows Index";
         return;
      }
      layout_.block_sizes.reserve(topology.stages().size());
      diagonal_offsets_.resize(topology.stages().size());
      lower_offsets_.resize(topology.stages().size() - 1);
      physical_block_sizes_.reserve(topology.stages().size());
      for( Index stage = 0; stage < topology.stages().size(); ++stage )
      {
         const OptimalControlStageDimensions dimensions = topology.stages()[stage];
         const Index primal_size = dimensions.controls + dimensions.states;
         const Index incoming = stage == 0 ? 0 : dimensions.states;
         Index block_size = 0;
         if( !CheckedAdd(primal_size, dimensions.path_equalities, block_size) ||
             !CheckedAdd(block_size, incoming, block_size) )
         {
            configuration_error_ =
               "equality-stage packed block layout overflows Index";
            return;
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
         Index lower_values = 0;
         Index next_diagonal_offset = 0;
         Index next_lower_offset = lower_offset;
         if( !CheckedMultiply(block_size, block_size, diagonal_values) ||
             !CheckedAdd(
                diagonal_offset, diagonal_values, next_diagonal_offset) ||
             !CheckedAdd(
                structured_dimension, block_size, structured_dimension) )
         {
            configuration_error_ =
               "equality-stage packed block layout overflows Index";
            return;
         }
         if( position > 0 &&
             (!CheckedMultiply(
                 block_size,
                 layout_.block_sizes[position - 1],
                 lower_values) ||
              !CheckedAdd(lower_offset, lower_values, next_lower_offset)) )
         {
            configuration_error_ =
               "equality-stage packed coupling layout overflows Index";
            return;
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
      if( structured_dimension != layout_.full_direction_dimension )
      {
         configuration_error_ =
            "equality-stage blocks do not cover the full KKT dimension";
         return;
      }

      structured_to_full_.reserve(layout_.full_direction_dimension);
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
         for( Index variable = 0; variable < primal_size; ++variable )
         {
            structured_to_full_.push_back(
               ordering.primal[primal_begin + variable]);
         }
         const Index path_begin = topology.path_equality_offsets()[stage];
         for( Index constraint = 0;
              constraint < dimensions.path_equalities;
              ++constraint )
         {
            const Index generic_constraint =
               ordering.path_equalities[path_begin + constraint];
            structured_to_full_.push_back(
               source_structure_.variables +
               constraint_to_multiplier[generic_constraint]);
         }
         if( stage > 0 )
         {
            const Index dynamics_begin = topology.dynamics_offsets()[stage - 1];
            for( Index state = 0; state < dimensions.states; ++state )
            {
               const Index generic_constraint =
                  ordering.dynamics[dynamics_begin + state];
               structured_to_full_.push_back(
                  source_structure_.variables +
                  constraint_to_multiplier[generic_constraint]);
            }
         }
      }
      std::vector<bool> seen(layout_.full_direction_dimension, false);
      for( Index index : structured_to_full_ )
      {
         if( index >= seen.size() || seen[index] )
         {
            configuration_error_ =
               "equality-stage structured/full permutation is invalid";
            return;
         }
         seen[index] = true;
      }
   }

   NlpStructure source_structure_;
   std::vector<Index> equality_constraints_;
   EqualityStageKktAssemblerOptions options_;
   StructureFingerprint topology_fingerprint_{0, 0};
   StageStructuredLayout layout_;
   std::vector<Index> physical_block_sizes_;
   std::vector<Index> diagonal_offsets_;
   std::vector<Index> lower_offsets_;
   std::vector<Index> structured_to_full_;
   Index diagonal_values_ = 0;
   Index lower_values_ = 0;
   std::string configuration_error_;
};

/** Full-KKT stage assembler for equality-only optimal-control NLPs.
 *
 * Structured block k is
 *
 *   [u_k, x_k, path_multiplier_k, incoming_dynamics_multiplier_k].
 *
 * The incoming multiplier is absent at k=0. With the canonical dynamics
 * Jacobian [B_k A_k -I], only adjacent stage blocks couple. No variable is
 * condensed, so the structured inertia is already the complete KKT inertia.
 * Bounds, path inequalities/slacks, restoration auxiliaries, and a nonidentity
 * next-state dynamics Jacobian are intentionally rejected by this first
 * assembler rather than silently destroying the block contract.
 */
template <StageDerivativeProvider Provider>
class EqualityStageKktAssembler
{
public:
   EqualityStageKktAssembler(
      Provider               provider,
      PrimalDualKktOperator& kkt,
      EqualityStageKktAssemblerOptions options = {}
   )
      : derivatives_(PrepareStageDerivativeProvider(std::move(provider))),
        options_(options)
   {
      Configure(kkt);
   }

   EqualityStageKktAssembler(
      Provider                        provider,
      PrimalDualKktOperator&          kkt,
      EqualityStageKktLayoutWorkspace workspace
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
      if( diagonal.size() != diagonal_values_ ||
          lower.size() != lower_values_ ||
          rhs.size() != structured_to_full_.size() )
      {
         return AssemblyFailure(
            EvaluationErrorCode::dimension_mismatch,
            "equality-stage assembler received incorrectly sized packed storage");
      }
      if( request.state.numeric_revision == 0 )
      {
         return AssemblyFailure(
            EvaluationErrorCode::numeric_mismatch,
            "equality-stage assembly requires a nonzero numeric revision");
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

         const Number* path_jacobian =
            derivative.path_equality_jacobians_transposed.data() +
            topology.path_equality_jacobian_offsets()[stage];
         for( Index variable = 0; variable < primal_size; ++variable )
         {
            for( Index constraint = 0;
                 constraint < dimensions.path_equalities;
                 ++constraint )
            {
               const Number value =
                  path_jacobian[variable * dimensions.path_equalities + constraint];
               const Index multiplier = primal_size + constraint;
               block[variable * block_size + multiplier] = value;
               block[multiplier * block_size + variable] = value;
            }
         }
         for( Index constraint = 0;
              constraint < dimensions.path_equalities;
              ++constraint )
         {
            const Index multiplier = primal_size + constraint;
            block[multiplier * block_size + multiplier] = -regularization.c;
         }

         if( stage > 0 )
         {
            const Index incoming_offset =
               primal_size + dimensions.path_equalities;
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

      for( Index structured = 0; structured < structured_to_full_.size(); ++structured )
      {
         rhs[structured] = request.rhs[structured_to_full_[structured]];
      }
      report.independent_full_inertia =
         CertifyFullInertia(diagonal, regularization);
      return report;
   }

   EvaluationValue<StageStructuredWork> reconstruct_stage_direction(
      CandidateFirstSolveRequest,
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
      if( structured_solution.size() != structured_to_full_.size() ||
          full_direction.size() != structured_to_full_.size() )
      {
         return std::unexpected(EvaluationError{
            EvaluationErrorCode::dimension_mismatch,
            "equality-stage reconstruction has the wrong direction dimension"
         });
      }
      for( Index structured = 0; structured < structured_to_full_.size(); ++structured )
      {
         full_direction[structured_to_full_[structured]] =
            structured_solution[structured];
      }
      return StageStructuredWork{};
   }

private:
   std::optional<CertifiedInertia> CertifyFullInertia(
      std::span<const Number>      diagonal,
      PrimalDualRegularization    regularization
   ) const noexcept
   {
      constexpr Number roundoff_multiplier = 128.;
      const StageNlpTopology& topology = derivatives_.topology();
      bool strictly_positive_definite = true;
      bool diagonal_positive_semidefinite = true;
      std::vector<std::pair<Index, Index>> zero_primal_entries;
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
               strictly_positive_definite = false;
            }
            if( off_diagonal_sum != 0. )
            {
               diagonal_positive_semidefinite = false;
            }
            else if( diagonal_value > margin )
            {
               // This represented diagonal is certifiably positive.
            }
            else if( diagonal_value == 0. )
            {
               zero_primal_entries.emplace_back(stage, row);
            }
            else
            {
               diagonal_positive_semidefinite = false;
            }
         }
      }
      const NlpStructure source = topology.source_structure();
      const auto certificate = [&]()
      {
         return CertifiedInertia{
            .positive_eigenvalues = source.variables,
            .negative_eigenvalues = source.constraints,
            .zero_eigenvalues = 0,
            .certificate_radius = 0.,
            .minimum_separation = 0.,
            .exact = true
         };
      };
      if( strictly_positive_definite && regularization.c > 0. )
      {
         // H is SPD and -delta_c I - J H^-1 J^T is negative definite.
         return certificate();
      }
      if( regularization.x != 0. || regularization.c != 0. ||
          !diagonal_positive_semidefinite )
      {
         return std::nullopt;
      }

      const OptimalControlStageDimensions initial = topology.stages().front();
      for( const auto& [stage, row] : zero_primal_entries )
      {
         static_cast<void>(row);
         if( stage != 0 )
         {
            return std::nullopt;
         }
      }
      for( Index stage = 1; stage < topology.stages().size(); ++stage )
      {
         if( topology.stages()[stage].path_equalities != 0 )
         {
            return std::nullopt;
         }
      }
      if( zero_primal_entries.size() != initial.path_equalities )
      {
         return std::nullopt;
      }
      const Index initial_primal = initial.controls + initial.states;
      const Index initial_block_size = physical_block_sizes_.front();
      const Number* initial_block =
         diagonal.data() + diagonal_offsets_.front();
      std::vector<bool> zero_used(zero_primal_entries.size(), false);
      for( Index constraint = 0;
           constraint < initial.path_equalities;
           ++constraint )
      {
         Index selected = zero_primal_entries.size();
         for( Index zero = 0; zero < zero_primal_entries.size(); ++zero )
         {
            const Index variable = zero_primal_entries[zero].second;
            const Number value = initial_block[
               variable * initial_block_size + initial_primal + constraint];
            if( value != 0. )
            {
               if( selected != zero_primal_entries.size() )
               {
                  return std::nullopt;
               }
               selected = zero;
            }
         }
         if( selected == zero_primal_entries.size() || zero_used[selected] )
         {
            return std::nullopt;
         }
         zero_used[selected] = true;
      }
      if( !std::ranges::all_of(zero_used, [](bool value) { return value; }) )
      {
         return std::nullopt;
      }

      // The path rows pivot on every null(H) initial variable. Dynamics rows
      // then have a backward sequence of unique -I next-state pivots, proving
      // full row rank and ker(H) intersect ker(J) = {0}.
      return certificate();
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

   static EvaluationValue<StageStructuredAssemblyReport> AssemblyFailure(
      EvaluationErrorCode code,
      std::string         message
   )
   {
      return std::unexpected(EvaluationError{code, std::move(message)});
   }

   void Configure(PrimalDualKktOperator& kkt)
   {
      const StageNlpTopology& topology = derivatives_.topology();
      const PrimalDualLayout& kkt_layout = kkt.layout();
      Bind(
         kkt,
         EqualityStageKktLayoutWorkspace(
            topology,
            std::vector<Index>(
               kkt_layout.equality_constraints.begin(),
               kkt_layout.equality_constraints.end()),
            options_));
   }

   void Bind(
      PrimalDualKktOperator&          kkt,
      EqualityStageKktLayoutWorkspace workspace
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
            "prepared equality-stage layout does not match the live topology";
         return;
      }
      const PrimalDualDimensions dimensions = kkt.dimensions();
      const NlpStructure source = topology.source_structure();
      if( dimensions.x != source.variables ||
          dimensions.y_c != source.constraints ||
          dimensions.s != 0 || dimensions.y_d != 0 ||
          dimensions.z_lower != 0 || dimensions.z_upper != 0 ||
          dimensions.v_lower != 0 || dimensions.v_upper != 0 )
      {
         configuration_error_ =
            "equality-stage assembler requires a bound-free equality-only full KKT";
         return;
      }
      if( !std::ranges::equal(
             kkt.layout().equality_constraints,
             workspace.equality_constraints_) ||
          kkt.flat_dimension() != workspace.layout_.full_direction_dimension )
      {
         configuration_error_ =
            "prepared equality-stage layout does not match the live KKT layout";
         return;
      }
      StructureFingerprintResult fingerprint = kkt.structure_fingerprint();
      if( !fingerprint )
      {
         configuration_error_ = fingerprint.error().message;
         return;
      }
      workspace.layout_.kkt_fingerprint = *fingerprint;
      layout_ = std::move(workspace.layout_);
      physical_block_sizes_ = std::move(workspace.physical_block_sizes_);
      diagonal_offsets_ = std::move(workspace.diagonal_offsets_);
      lower_offsets_ = std::move(workspace.lower_offsets_);
      structured_to_full_ = std::move(workspace.structured_to_full_);
      diagonal_values_ = workspace.diagonal_values_;
      lower_values_ = workspace.lower_values_;
   }

   PreparedStageDerivativeProvider<Provider> derivatives_;
   EqualityStageKktAssemblerOptions options_;
   StageStructuredLayout layout_;
   std::vector<Index> physical_block_sizes_;
   std::vector<Index> diagonal_offsets_;
   std::vector<Index> lower_offsets_;
   std::vector<Index> structured_to_full_;
   Index diagonal_values_ = 0;
   Index lower_values_ = 0;
   std::string configuration_error_;
   bool derivatives_cached_ = false;
   std::uint64_t cached_numeric_revision_ = 0;
};
} // namespace Ipopt::Cxx23

#endif
