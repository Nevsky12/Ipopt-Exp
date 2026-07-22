// Copyright (C) 2026 Ipopt contributors.
// All Rights Reserved.
// This file is distributed under the Eclipse Public License.

#ifndef IPOPT_CXX23_SPARSE_BORDERED_STAGE_DERIVATIVE_PROVIDER_HPP
#define IPOPT_CXX23_SPARSE_BORDERED_STAGE_DERIVATIVE_PROVIDER_HPP

#include <ipopt/cxx23/bordered_stage_structured_nlp.hpp>
#include <ipopt/cxx23/primal_dual_kkt_operator.hpp>

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
struct SparseBorderedStageDerivativeProviderOptions
{
   friend bool operator==(
      const SparseBorderedStageDerivativeProviderOptions&,
      const SparseBorderedStageDerivativeProviderOptions&) = default;
};

/** O(nnz) scatter from a live sparse KKT into local-stage and border blocks.
 *
 * The constructor retains no reference to the exemplar. Local-local Hessian
 * entries may remain within one stage or couple adjacent stages. Local-global
 * and global-global entries form the primal border. Path rows may touch their
 * own local stage plus any global variable; dynamics may additionally touch a
 * general square next-state block. The assembler normalizes that block exactly.
 */
class SparseBorderedStageDerivativeProvider
{
public:
   SparseBorderedStageDerivativeProvider(
      PrimalDualKktOperator&                       exemplar,
      BorderedStageNlpTopology                     topology,
      SparseBorderedStageDerivativeProviderOptions = {}
   )
      : topology_(std::move(topology))
   {
      Configure(exemplar);
   }

   const BorderedStageNlpTopology&
   bordered_stage_nlp_topology() const noexcept
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

   EvaluationResult eval_bordered_stage_derivatives(
      PrimalDualKktOperator&          kkt,
      KktEvaluationPoint             point,
      BorderedStageDerivativeBuffers buffers
   )
   {
      if( !configured() )
      {
         return Failure(
            EvaluationErrorCode::invalid_layout, configuration_error_);
      }
      const BorderedStageDerivativeStorage storage =
         topology_.derivative_storage();
      if( buffers.stage_hessians.size() != storage.stage_hessians ||
          buffers.cross_stage_hessians.size() !=
             storage.cross_stage_hessians ||
          buffers.local_global_hessians.size() !=
             storage.local_global_hessians ||
          buffers.global_hessian.size() != storage.global_hessian ||
          buffers.dynamics_jacobians_transposed.size() !=
             storage.dynamics_jacobians_transposed ||
          buffers.dynamics_next_state_jacobians.size() !=
             storage.dynamics_next_state_jacobians ||
          buffers.path_equality_jacobians_transposed.size() !=
             storage.path_equality_jacobians_transposed ||
          buffers.path_inequality_jacobians_transposed.size() !=
             storage.path_inequality_jacobians_transposed ||
          buffers.global_jacobians_transposed.size() !=
             storage.global_jacobians_transposed )
      {
         return Failure(
            EvaluationErrorCode::dimension_mismatch,
            "sparse bordered provider received incorrectly sized buffers");
      }
      if( kkt.nlp_structure() != topology_.source_structure() )
      {
         return Failure(
            EvaluationErrorCode::structure_mismatch,
            "sparse bordered provider source structure changed");
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
            "sparse bordered provider KKT fingerprint changed");
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
            "sparse bordered provider received nonfinite derivatives");
      }

      std::ranges::fill(buffers.stage_hessians, 0.);
      std::ranges::fill(buffers.cross_stage_hessians, 0.);
      std::ranges::fill(buffers.local_global_hessians, 0.);
      std::ranges::fill(buffers.global_hessian, 0.);
      std::ranges::fill(buffers.dynamics_jacobians_transposed, 0.);
      std::ranges::fill(buffers.dynamics_next_state_jacobians, 0.);
      std::ranges::fill(buffers.path_equality_jacobians_transposed, 0.);
      std::ranges::fill(buffers.path_inequality_jacobians_transposed, 0.);
      std::ranges::fill(buffers.global_jacobians_transposed, 0.);
      ScatterValues(
         stage_hessian_scatter_, hessian_values_, buffers.stage_hessians);
      ScatterValues(
         cross_stage_hessian_scatter_,
         hessian_values_,
         buffers.cross_stage_hessians);
      ScatterValues(
         local_global_hessian_scatter_,
         hessian_values_,
         buffers.local_global_hessians);
      ScatterValues(
         global_hessian_scatter_, hessian_values_, buffers.global_hessian);
      ScatterValues(
         dynamics_scatter_,
         jacobian_values_,
         buffers.dynamics_jacobians_transposed);
      ScatterValues(
         dynamics_next_state_scatter_,
         jacobian_values_,
         buffers.dynamics_next_state_jacobians);
      ScatterValues(
         path_equality_scatter_,
         jacobian_values_,
         buffers.path_equality_jacobians_transposed);
      ScatterValues(
         path_inequality_scatter_,
         jacobian_values_,
         buffers.path_inequality_jacobians_transposed);
      ScatterValues(
         global_jacobian_scatter_,
         jacobian_values_,
         buffers.global_jacobians_transposed);
      if( !AllFinite(buffers.stage_hessians) ||
          !AllFinite(buffers.cross_stage_hessians) ||
          !AllFinite(buffers.local_global_hessians) ||
          !AllFinite(buffers.global_hessian) ||
          !AllFinite(buffers.dynamics_jacobians_transposed) ||
          !AllFinite(buffers.dynamics_next_state_jacobians) ||
          !AllFinite(buffers.path_equality_jacobians_transposed) ||
          !AllFinite(buffers.path_inequality_jacobians_transposed) ||
          !AllFinite(buffers.global_jacobians_transposed) )
      {
         return Failure(
            EvaluationErrorCode::nonfinite_output,
            "sparse bordered scatter overflowed a derivative coefficient");
      }
      return {};
   }

private:
   enum class VariableKind : std::uint8_t
   {
      missing,
      local,
      global
   };

   enum class ConstraintKind : std::uint8_t
   {
      missing,
      dynamics,
      path_equality,
      path_inequality
   };

   struct VariableCoordinate
   {
      VariableKind kind = VariableKind::missing;
      Index stage = 0;
      Index local = 0;
      Index global = 0;
   };

   struct ConstraintCoordinate
   {
      ConstraintKind kind = ConstraintKind::missing;
      Index stage = 0;
      Index local = 0;
   };

   struct Scatter
   {
      Index source = 0;
      Index target = 0;
   };

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
      std::span<const Scatter> scatter,
      std::span<const Number>  source,
      std::span<Number>        target
   ) noexcept
   {
      for( const Scatter entry : scatter )
      {
         target[entry.target] += source[entry.source];
      }
   }

   bool ConfigureCoordinates(
      std::vector<VariableCoordinate>&   variables,
      std::vector<ConstraintCoordinate>& constraints
   )
   {
      const NlpStructure source = topology_.source_structure();
      const auto stages = topology_.stages();
      const StageNlpOrdering& ordering = topology_.ordering();
      variables.resize(source.variables);
      for( Index stage = 0; stage < stages.size(); ++stage )
      {
         const Index begin = topology_.primal_offsets()[stage];
         const Index size = stages[stage].controls + stages[stage].states;
         for( Index local = 0; local < size; ++local )
         {
            const Index generic = ordering.primal[begin + local];
            if( generic >= variables.size() ||
                variables[generic].kind != VariableKind::missing )
            {
               Fail("sparse bordered local primal ordering is invalid");
               return false;
            }
            variables[generic] = {VariableKind::local, stage, local, 0};
         }
      }
      for( Index global = 0;
           global < topology_.global_primal_variables().size();
           ++global )
      {
         const Index generic = topology_.global_primal_variables()[global];
         if( generic >= variables.size() ||
             variables[generic].kind != VariableKind::missing )
         {
            Fail("sparse bordered global primal ordering is invalid");
            return false;
         }
         variables[generic] = {VariableKind::global, 0, 0, global};
      }

      constraints.resize(source.constraints);
      const auto assign = [&](std::span<const Index> rows,
                              ConstraintKind kind,
                              std::span<const Index> offsets,
                              auto count) -> bool
      {
         for( Index stage = 0; stage + 1 < offsets.size(); ++stage )
         {
            for( Index local = 0; local < count(stage); ++local )
            {
               const Index generic = rows[offsets[stage] + local];
               if( generic >= constraints.size() ||
                   constraints[generic].kind != ConstraintKind::missing )
               {
                  Fail("sparse bordered constraint ordering is invalid");
                  return false;
               }
               constraints[generic] = {kind, stage, local};
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
      for( Index stage = 0; stage + 1 < stages.size(); ++stage )
      {
         const Index begin = topology_.dynamics_offsets()[stage];
         for( Index state = 0; state < stages[stage + 1].states; ++state )
         {
            const Index generic = ordering.dynamics[begin + state];
            if( generic >= constraints.size() ||
                constraints[generic].kind != ConstraintKind::missing )
            {
               Fail("sparse bordered dynamics ordering is invalid");
               return false;
            }
            constraints[generic] = {
               ConstraintKind::dynamics, stage, state};
         }
      }
      return std::ranges::all_of(
         constraints,
         [](const ConstraintCoordinate& value)
         {
            return value.kind != ConstraintKind::missing;
         });
   }

   bool ConfigureHessian(
      std::span<const Index>                 rows,
      std::span<const Index>                 columns,
      std::span<const VariableCoordinate>    variables
   )
   {
      const auto stages = topology_.stages();
      const Index global_size = topology_.global_primal_variables().size();
      for( Index entry = 0; entry < rows.size(); ++entry )
      {
         if( rows[entry] >= variables.size() ||
             columns[entry] >= variables.size() )
         {
            Fail("sparse bordered Hessian index is out of range");
            return false;
         }
         const VariableCoordinate row = variables[rows[entry]];
         const VariableCoordinate column = variables[columns[entry]];
         if( row.kind == VariableKind::local &&
             column.kind == VariableKind::local )
         {
            if( row.stage == column.stage )
            {
               const Index size =
                  stages[row.stage].controls + stages[row.stage].states;
               const Index offset = topology_.hessian_offsets()[row.stage];
               stage_hessian_scatter_.push_back({
                  entry, offset + row.local * size + column.local});
               if( row.local != column.local )
               {
                  stage_hessian_scatter_.push_back({
                     entry, offset + column.local * size + row.local});
               }
               continue;
            }
            const VariableCoordinate previous = row.stage < column.stage
               ? row
               : column;
            const VariableCoordinate next = row.stage < column.stage
               ? column
               : row;
            if( next.stage != previous.stage + 1 )
            {
               Fail("sparse bordered Hessian couples nonadjacent local stages");
               return false;
            }
            const Index previous_size = stages[previous.stage].controls +
               stages[previous.stage].states;
            cross_stage_hessian_scatter_.push_back({
               entry,
               topology_.cross_stage_hessian_offsets()[previous.stage] +
                  next.local * previous_size + previous.local
            });
         }
         else if( row.kind == VariableKind::global &&
                  column.kind == VariableKind::global )
         {
            global_hessian_scatter_.push_back({
               entry, row.global * global_size + column.global});
            if( row.global != column.global )
            {
               global_hessian_scatter_.push_back({
                  entry, column.global * global_size + row.global});
            }
         }
         else
         {
            const VariableCoordinate local = row.kind == VariableKind::local
               ? row
               : column;
            const VariableCoordinate global = row.kind == VariableKind::global
               ? row
               : column;
            if( local.kind != VariableKind::local ||
                global.kind != VariableKind::global )
            {
               Fail("sparse bordered Hessian variable map is incomplete");
               return false;
            }
            const Index target =
               topology_.local_global_hessian_offsets()[local.stage] +
               local.local * global_size + global.global;
            local_global_hessian_scatter_.push_back({entry, target});
         }
      }
      return true;
   }

   bool ConfigureJacobian(
      std::span<const Index>                    rows,
      std::span<const Index>                    columns,
      std::span<const VariableCoordinate>       variables,
      std::span<const ConstraintCoordinate>     constraints
   )
   {
      const auto stages = topology_.stages();
      const Index constraint_count = topology_.source_structure().constraints;
      for( Index entry = 0; entry < rows.size(); ++entry )
      {
         if( rows[entry] >= constraints.size() ||
             columns[entry] >= variables.size() )
         {
            Fail("sparse bordered Jacobian index is out of range");
            return false;
         }
         const ConstraintCoordinate constraint = constraints[rows[entry]];
         const VariableCoordinate variable = variables[columns[entry]];
         if( variable.kind == VariableKind::global )
         {
            global_jacobian_scatter_.push_back({
               entry, variable.global * constraint_count + rows[entry]});
            continue;
         }
         if( variable.kind != VariableKind::local )
         {
            Fail("sparse bordered Jacobian variable map is incomplete");
            return false;
         }
         const Index stage = constraint.stage;
         const Index local = constraint.local;
         if( constraint.kind == ConstraintKind::path_equality ||
             constraint.kind == ConstraintKind::path_inequality )
         {
            if( variable.stage != stage )
            {
               Fail("sparse bordered path row has a local cross-stage entry");
               return false;
            }
            const Index count = constraint.kind == ConstraintKind::path_equality
               ? stages[stage].path_equalities
               : stages[stage].path_inequalities;
            const Index offset = constraint.kind == ConstraintKind::path_equality
               ? topology_.path_equality_jacobian_offsets()[stage]
               : topology_.path_inequality_jacobian_offsets()[stage];
            std::vector<Scatter>& scatter =
               constraint.kind == ConstraintKind::path_equality
               ? path_equality_scatter_
               : path_inequality_scatter_;
            scatter.push_back({
               entry, offset + variable.local * count + local});
            continue;
         }

         const Index next_stage = stage + 1;
         const Index next_states = stages[next_stage].states;
         if( variable.stage == stage )
         {
            dynamics_scatter_.push_back({
               entry,
               topology_.dynamics_jacobian_offsets()[stage] +
                  variable.local * next_states + local
            });
         }
         else if( variable.stage == next_stage &&
                  variable.local >= stages[next_stage].controls )
         {
            const Index state =
               variable.local - stages[next_stage].controls;
            dynamics_next_state_scatter_.push_back({
               entry,
               topology_.dynamics_next_state_jacobian_offsets()[stage] +
                  local * next_states + state
            });
         }
         else
         {
            Fail("sparse bordered dynamics has an unsupported local coupling");
            return false;
         }
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
      if( !exemplar.valid() ||
          exemplar.nlp_structure() != topology_.source_structure() )
      {
         Fail("sparse bordered topology does not match the exemplar KKT");
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

      std::vector<VariableCoordinate> variables;
      std::vector<ConstraintCoordinate> constraints;
      if( !ConfigureCoordinates(variables, constraints) ||
          !ConfigureHessian(hessian_rows, hessian_columns, variables) ||
          !ConfigureJacobian(
             jacobian_rows, jacobian_columns, variables, constraints) )
      {
         return;
      }
      hessian_values_.resize(source.hessian_nonzeros);
      jacobian_values_.resize(source.jacobian_nonzeros);
   }

   BorderedStageNlpTopology topology_;
   StructureFingerprint kkt_fingerprint_{0, 0};
   std::vector<Scatter> stage_hessian_scatter_;
   std::vector<Scatter> cross_stage_hessian_scatter_;
   std::vector<Scatter> local_global_hessian_scatter_;
   std::vector<Scatter> global_hessian_scatter_;
   std::vector<Scatter> dynamics_scatter_;
   std::vector<Scatter> dynamics_next_state_scatter_;
   std::vector<Scatter> path_equality_scatter_;
   std::vector<Scatter> path_inequality_scatter_;
   std::vector<Scatter> global_jacobian_scatter_;
   std::vector<Number> hessian_values_;
   std::vector<Number> jacobian_values_;
   std::string configuration_error_;
};
} // namespace Ipopt::Cxx23

#endif
