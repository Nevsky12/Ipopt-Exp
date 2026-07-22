// Copyright (C) 2026 Ipopt contributors.
// All Rights Reserved.
// This file is distributed under the Eclipse Public License.

#ifndef IPOPT_CXX23_SPARSE_STAGE_DERIVATIVE_PROVIDER_HPP
#define IPOPT_CXX23_SPARSE_STAGE_DERIVATIVE_PROVIDER_HPP

#include <ipopt/cxx23/primal_dual_kkt_operator.hpp>
#include <ipopt/cxx23/sparse_derivative_scatter.hpp>
#include <ipopt/cxx23/stage_structured_nlp.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace Ipopt::Cxx23
{
struct SparseStageDerivativeProviderOptions
{
   Number canonical_next_state_tolerance =
      256. * std::numeric_limits<Number>::epsilon();

   friend bool operator==(
      const SparseStageDerivativeProviderOptions&,
      const SparseStageDerivativeProviderOptions&) = default;
};

/** Scatters one live sparse NLP Hessian/Jacobian into validated stage blocks.
 *
 * The exemplar is used only while constructing immutable scatter maps; it is
 * never retained. Numeric evaluation uses the PrimalDualKktOperator supplied
 * with the current solve request, which keeps a lazy backend safe when the
 * canary recreates its matrix snapshot on every iteration.
 *
 * The accepted structure is deliberately strict: Hessian entries must remain
 * inside one stage, path rows may touch only their own stage, and a dynamics
 * row may touch the current stage plus exactly the matching next-state entry
 * with numeric value -1. These checks make the packed stage representation an
 * explicit capability rather than a sparsity heuristic.
 */
class SparseStageDerivativeProvider
{
public:
   SparseStageDerivativeProvider(
      PrimalDualKktOperator&               exemplar,
      StageNlpTopology                     topology,
      SparseStageDerivativeProviderOptions options = {}
   )
      : topology_(std::move(topology)),
        options_(options)
   {
      Configure(exemplar);
   }

   const StageNlpTopology& stage_nlp_topology() const noexcept
   {
      return topology_;
   }

   bool configured() const noexcept
   {
      return configuration_error_.empty();
   }

   const std::string& configuration_error() const noexcept
   {
      return configuration_error_;
   }

   EvaluationResult eval_stage_derivatives(
      PrimalDualKktOperator& kkt,
      KktEvaluationPoint     point,
      StageDerivativeBuffers buffers
   )
   {
      if( !configured() )
      {
         return Failure(
            EvaluationErrorCode::invalid_layout, configuration_error_);
      }
      const StageDerivativeStorage storage = topology_.derivative_storage();
      if( buffers.stage_hessians.size() != storage.stage_hessians ||
          buffers.dynamics_jacobians_transposed.size() !=
             storage.dynamics_jacobians_transposed ||
          buffers.path_equality_jacobians_transposed.size() !=
             storage.path_equality_jacobians_transposed ||
          buffers.path_inequality_jacobians_transposed.size() !=
             storage.path_inequality_jacobians_transposed )
      {
         return Failure(
            EvaluationErrorCode::dimension_mismatch,
            "sparse stage provider received incorrectly sized derivative buffers");
      }
      if( kkt.nlp_structure() != topology_.source_structure() )
      {
         return Failure(
            EvaluationErrorCode::structure_mismatch,
            "sparse stage provider source structure changed");
      }
      StructureFingerprintResult fingerprint = kkt.structure_fingerprint();
      if( !fingerprint )
      {
         return std::unexpected(fingerprint.error());
      }
      if( *fingerprint != kkt_fingerprint_ )
      {
         return Failure(
            EvaluationErrorCode::structure_mismatch,
            "sparse stage provider KKT fingerprint changed");
      }
      if( EvaluationResult evaluated =
             kkt.nlp_hessian_values(point, hessian_values_);
          !evaluated )
      {
         return evaluated;
      }
      if( EvaluationResult evaluated =
             kkt.nlp_jacobian_values(point, jacobian_values_);
          !evaluated )
      {
         return evaluated;
      }
      if( !AllFinite(hessian_values_) || !AllFinite(jacobian_values_) )
      {
         return Failure(
            EvaluationErrorCode::nonfinite_output,
            "sparse stage provider received nonfinite derivative values");
      }

      ScatterValues(
         hessian_scatter_, hessian_values_, buffers.stage_hessians);
      ScatterValues(
         dynamics_scatter_,
         jacobian_values_,
         buffers.dynamics_jacobians_transposed);
      ScatterValues(
         path_equality_scatter_,
         jacobian_values_,
         buffers.path_equality_jacobians_transposed);
      ScatterValues(
         path_inequality_scatter_,
         jacobian_values_,
         buffers.path_inequality_jacobians_transposed);
      ScatterValues(
         canonical_next_state_scatter_,
         jacobian_values_,
         canonical_next_state_values_);
      if( !all_output_scatter_targets_unique_ &&
          (!AllFinite(buffers.stage_hessians) ||
           !AllFinite(buffers.dynamics_jacobians_transposed) ||
           !AllFinite(buffers.path_equality_jacobians_transposed) ||
           !AllFinite(buffers.path_inequality_jacobians_transposed)) )
      {
         return Failure(
            EvaluationErrorCode::nonfinite_output,
            "sparse stage scatter overflowed a derivative coefficient");
      }
      for( Number value : canonical_next_state_values_ )
      {
         const Number scale = std::max(Number{1.}, std::abs(value));
         if( !std::isfinite(value) ||
             std::abs(value + 1.) >
                options_.canonical_next_state_tolerance * scale )
         {
            return Failure(
               EvaluationErrorCode::invalid_layout,
               "sparse stage dynamics does not have canonical next-state -I");
         }
      }
      return {};
   }

private:
   enum class ConstraintKind : std::uint8_t
   {
      missing,
      dynamics,
      path_equality,
      path_inequality
   };

   struct Coordinate
   {
      Index stage = 0;
      Index local = 0;
   };

   struct ConstraintCoordinate
   {
      ConstraintKind kind = ConstraintKind::missing;
      Coordinate coordinate;
   };

   using ScatterPlan = detail::SparseDerivativeScatterPlan;

   static EvaluationResult Failure(
      EvaluationErrorCode code,
      std::string         message
   )
   {
      return std::unexpected(EvaluationError{code, std::move(message)});
   }

   void Fail(std::string message)
   {
      if( configuration_error_.empty() )
      {
         configuration_error_ = std::move(message);
      }
   }

   static bool AllFinite(std::span<const Number> values) noexcept
   {
      return std::ranges::all_of(
         values, [](Number value) { return std::isfinite(value); });
   }

   static void ScatterValues(
      const ScatterPlan&      scatter,
      std::span<const Number> source,
      std::span<Number>       target
   ) noexcept
   {
      scatter.write(source, target);
   }

   bool PrepareScatterPlan(ScatterPlan& scatter, Index target_size)
   {
      if( !scatter.prepare(target_size) )
      {
         Fail("sparse stage scatter target is out of range");
         return false;
      }
      return true;
   }

   bool ConfigureCoordinates(
      std::vector<Coordinate>&           variables,
      std::vector<ConstraintCoordinate>& constraints
   )
   {
      const NlpStructure source = topology_.source_structure();
      const auto stages = topology_.stages();
      const StageNlpOrdering& ordering = topology_.ordering();
      variables.resize(source.variables);
      std::vector<bool> variable_seen(source.variables, false);
      for( Index stage = 0; stage < stages.size(); ++stage )
      {
         const Index begin = topology_.primal_offsets()[stage];
         const Index size = stages[stage].controls + stages[stage].states;
         for( Index local = 0; local < size; ++local )
         {
            const Index generic = ordering.primal[begin + local];
            if( generic >= variables.size() || variable_seen[generic] )
            {
               Fail("sparse stage primal ordering is not a permutation");
               return false;
            }
            variable_seen[generic] = true;
            variables[generic] = {stage, local};
         }
      }

      constraints.resize(source.constraints);
      const auto assign = [&](std::span<const Index> generic_rows,
                              ConstraintKind kind,
                              std::span<const Index> offsets,
                              auto count) -> bool
      {
         for( Index stage = 0; stage + 1 < offsets.size(); ++stage )
         {
            for( Index local = 0; local < count(stage); ++local )
            {
               const Index generic = generic_rows[offsets[stage] + local];
               if( generic >= constraints.size() ||
                   constraints[generic].kind != ConstraintKind::missing )
               {
                  Fail("sparse stage constraint ordering is not a permutation");
                  return false;
               }
               constraints[generic] = {kind, {stage, local}};
            }
         }
         return true;
      };
      if( !assign(
             ordering.path_equalities,
             ConstraintKind::path_equality,
             topology_.path_equality_offsets(),
             [&](Index stage) { return stages[stage].path_equalities; }) ||
          !assign(
             ordering.path_inequalities,
             ConstraintKind::path_inequality,
             topology_.path_inequality_offsets(),
             [&](Index stage) { return stages[stage].path_inequalities; }) )
      {
         return false;
      }
      for( Index transition = 0; transition + 1 < stages.size(); ++transition )
      {
         const Index begin = topology_.dynamics_offsets()[transition];
         for( Index state = 0; state < stages[transition + 1].states; ++state )
         {
            const Index generic = ordering.dynamics[begin + state];
            if( generic >= constraints.size() ||
                constraints[generic].kind != ConstraintKind::missing )
            {
               Fail("sparse stage dynamics ordering is not a permutation");
               return false;
            }
            constraints[generic] = {
               ConstraintKind::dynamics, {transition, state}};
         }
      }
      if( !std::ranges::all_of(
             constraints,
             [](const ConstraintCoordinate& coordinate)
             {
                return coordinate.kind != ConstraintKind::missing;
             }) )
      {
         Fail("sparse stage constraint ordering does not cover the NLP");
         return false;
      }
      return true;
   }

   bool ConfigureHessian(
      std::span<const Index>      rows,
      std::span<const Index>      columns,
      std::span<const Coordinate> variables
   )
   {
      const auto stages = topology_.stages();
      for( Index entry = 0; entry < rows.size(); ++entry )
      {
         if( rows[entry] >= variables.size() ||
             columns[entry] >= variables.size() )
         {
            Fail("sparse stage Hessian index is out of range");
            return false;
         }
         const Coordinate row = variables[rows[entry]];
         const Coordinate column = variables[columns[entry]];
         if( row.stage != column.stage )
         {
            Fail("sparse stage Hessian has a cross-stage entry");
            return false;
         }
         const Index size =
            stages[row.stage].controls + stages[row.stage].states;
         const Index offset = topology_.hessian_offsets()[row.stage];
         hessian_scatter_.push_back({
            entry, offset + row.local * size + column.local});
         if( row.local != column.local )
         {
            hessian_scatter_.push_back({
               entry, offset + column.local * size + row.local});
         }
      }
      return true;
   }

   bool ConfigureJacobian(
      std::span<const Index>                rows,
      std::span<const Index>                columns,
      std::span<const Coordinate>           variables,
      std::span<const ConstraintCoordinate> constraints
   )
   {
      const auto stages = topology_.stages();
      std::vector<bool> canonical_seen(topology_.dynamics_constraints(), false);
      for( Index entry = 0; entry < rows.size(); ++entry )
      {
         if( rows[entry] >= constraints.size() ||
             columns[entry] >= variables.size() )
         {
            Fail("sparse stage Jacobian index is out of range");
            return false;
         }
         const ConstraintCoordinate constraint = constraints[rows[entry]];
         const Coordinate variable = variables[columns[entry]];
         const Index stage = constraint.coordinate.stage;
         const Index local = constraint.coordinate.local;
         if( constraint.kind == ConstraintKind::path_equality ||
             constraint.kind == ConstraintKind::path_inequality )
         {
            if( variable.stage != stage )
            {
               Fail("sparse path Jacobian has a cross-stage entry");
               return false;
            }
            const Index count = constraint.kind == ConstraintKind::path_equality
               ? stages[stage].path_equalities
               : stages[stage].path_inequalities;
            const Index offset = constraint.kind == ConstraintKind::path_equality
               ? topology_.path_equality_jacobian_offsets()[stage]
               : topology_.path_inequality_jacobian_offsets()[stage];
            ScatterPlan& scatter =
               constraint.kind == ConstraintKind::path_equality
               ? path_equality_scatter_
               : path_inequality_scatter_;
            scatter.push_back({entry, offset + variable.local * count + local});
            continue;
         }

         const Index current_stage = stage;
         const Index next_stage = stage + 1;
         const Index next_states = stages[next_stage].states;
         if( variable.stage == current_stage )
         {
            const Index target =
               topology_.dynamics_jacobian_offsets()[current_stage] +
               variable.local * next_states + local;
            dynamics_scatter_.push_back({entry, target});
         }
         else if( variable.stage == next_stage &&
                  variable.local == stages[next_stage].controls + local )
         {
            const Index target =
               topology_.dynamics_offsets()[current_stage] + local;
            canonical_next_state_scatter_.push_back({entry, target});
            canonical_seen[target] = true;
         }
         else
         {
            Fail("sparse dynamics Jacobian is not current-stage plus next-state -I");
            return false;
         }
      }
      if( !std::ranges::all_of(canonical_seen, [](bool value) { return value; }) )
      {
         Fail("sparse dynamics Jacobian omits a canonical next-state entry");
         return false;
      }
      return true;
   }

   void Configure(PrimalDualKktOperator& exemplar)
   {
      if( !topology_.configured() )
      {
         Fail(topology_.configuration_error());
         return;
      }
      if( !std::isfinite(options_.canonical_next_state_tolerance) ||
          options_.canonical_next_state_tolerance < 0. )
      {
         Fail("sparse stage canonical tolerance is invalid");
         return;
      }
      if( !exemplar.valid() ||
          exemplar.nlp_structure() != topology_.source_structure() )
      {
         Fail("sparse stage topology does not match the exemplar KKT");
         return;
      }
      StructureFingerprintResult fingerprint = exemplar.structure_fingerprint();
      if( !fingerprint )
      {
         Fail(fingerprint.error().message);
         return;
      }
      kkt_fingerprint_ = *fingerprint;

      const NlpStructure source = topology_.source_structure();
      std::vector<Index> hessian_rows(source.hessian_nonzeros);
      std::vector<Index> hessian_columns(source.hessian_nonzeros);
      std::vector<Index> jacobian_rows(source.jacobian_nonzeros);
      std::vector<Index> jacobian_columns(source.jacobian_nonzeros);
      if( EvaluationResult structure = exemplar.nlp_hessian_structure(
             hessian_rows, hessian_columns);
          !structure )
      {
         Fail(structure.error().message);
         return;
      }
      if( EvaluationResult structure = exemplar.nlp_jacobian_structure(
             jacobian_rows, jacobian_columns);
          !structure )
      {
         Fail(structure.error().message);
         return;
      }

      std::vector<Coordinate> variables;
      std::vector<ConstraintCoordinate> constraints;
      if( !ConfigureCoordinates(variables, constraints) ||
          !ConfigureHessian(hessian_rows, hessian_columns, variables) ||
          !ConfigureJacobian(
             jacobian_rows, jacobian_columns, variables, constraints) )
      {
         return;
      }
      const StageDerivativeStorage storage = topology_.derivative_storage();
      if( !PrepareScatterPlan(hessian_scatter_, storage.stage_hessians) ||
          !PrepareScatterPlan(
             dynamics_scatter_, storage.dynamics_jacobians_transposed) ||
          !PrepareScatterPlan(
             path_equality_scatter_,
             storage.path_equality_jacobians_transposed) ||
          !PrepareScatterPlan(
             path_inequality_scatter_,
             storage.path_inequality_jacobians_transposed) ||
          !PrepareScatterPlan(
             canonical_next_state_scatter_,
             topology_.dynamics_constraints()) )
      {
         return;
      }
      all_output_scatter_targets_unique_ =
         hessian_scatter_.unique_targets() &&
         dynamics_scatter_.unique_targets() &&
         path_equality_scatter_.unique_targets() &&
         path_inequality_scatter_.unique_targets();
      hessian_values_.resize(source.hessian_nonzeros);
      jacobian_values_.resize(source.jacobian_nonzeros);
      canonical_next_state_values_.resize(topology_.dynamics_constraints());
   }

   StageNlpTopology topology_;
   SparseStageDerivativeProviderOptions options_;
   StructureFingerprint kkt_fingerprint_{0, 0};
   ScatterPlan hessian_scatter_;
   ScatterPlan dynamics_scatter_;
   ScatterPlan path_equality_scatter_;
   ScatterPlan path_inequality_scatter_;
   ScatterPlan canonical_next_state_scatter_;
   bool all_output_scatter_targets_unique_ = false;
   std::vector<Number> hessian_values_;
   std::vector<Number> jacobian_values_;
   std::vector<Number> canonical_next_state_values_;
   std::string configuration_error_;
};
} // namespace Ipopt::Cxx23

#endif
