// Copyright (C) 2026 Ipopt contributors.
// All Rights Reserved.
// This file is distributed under the Eclipse Public License.

#ifndef IPOPT_CXX23_RESTORATION_STAGE_NLP_HPP
#define IPOPT_CXX23_RESTORATION_STAGE_NLP_HPP

#include <ipopt/cxx23/primal_dual_kkt_operator.hpp>
#include <ipopt/cxx23/stage_structured_nlp.hpp>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace Ipopt::Cxx23
{
/** Explicit flat-variable maps for RestoIpoptNLP's compound primal vector.
 *
 * equality_n/equality_p and inequality_n/inequality_p are indexed in the
 * corresponding multiplier order of the main PrimalDualLayout. Their signs
 * in the restoration constraints are +n-p.
 */
struct RestorationStageVariableLayout
{
   std::vector<Index> original_primal;
   std::vector<Index> equality_n;
   std::vector<Index> equality_p;
   std::vector<Index> inequality_n;
   std::vector<Index> inequality_p;

   friend bool operator==(
      const RestorationStageVariableLayout&,
      const RestorationStageVariableLayout&) = default;
};

namespace restoration_stage_detail
{
inline EvaluationError Error(std::string message)
{
   return {
      EvaluationErrorCode::invalid_layout,
      std::move(message)
   };
}

inline bool CheckedAdd(Index left, Index right, Index& result) noexcept
{
   if( right > std::numeric_limits<Index>::max() - left )
   {
      return false;
   }
   result = left + right;
   return true;
}

inline bool ValidateUniqueMap(
   std::span<const Index> map,
   Index                  dimension
)
{
   std::vector<bool> seen(dimension, false);
   for( Index index : map )
   {
      if( index >= dimension || seen[index] )
      {
         return false;
      }
      seen[index] = true;
   }
   return true;
}

inline bool SameIndexSet(
   std::span<const Index> left,
   std::span<const Index> right
)
{
   if( left.size() != right.size() )
   {
      return false;
   }
   std::vector<Index> sorted_left(left.begin(), left.end());
   std::vector<Index> sorted_right(right.begin(), right.end());
   std::ranges::sort(sorted_left);
   std::ranges::sort(sorted_right);
   return sorted_left == sorted_right;
}

inline bool ConfigureConstraintPositions(
   const PrimalDualLayout& layout,
   Index                   constraints,
   std::vector<Index>&     equality_position,
   std::vector<Index>&     inequality_position
)
{
   const Index missing = constraints;
   equality_position.assign(constraints, missing);
   inequality_position.assign(constraints, missing);
   for( Index position = 0;
        position < layout.equality_constraints.size();
        ++position )
   {
      const Index row = layout.equality_constraints[position];
      if( row >= constraints || equality_position[row] != missing ||
          inequality_position[row] != missing )
      {
         return false;
      }
      equality_position[row] = position;
   }
   for( Index position = 0;
        position < layout.inequality_constraints.size();
        ++position )
   {
      const Index row = layout.inequality_constraints[position];
      if( row >= constraints || equality_position[row] != missing ||
          inequality_position[row] != missing )
      {
         return false;
      }
      inequality_position[row] = position;
   }
   for( Index row = 0; row < constraints; ++row )
   {
      if( equality_position[row] == missing &&
          inequality_position[row] == missing )
      {
         return false;
      }
   }
   return true;
}
} // namespace restoration_stage_detail

/** Canonical flat order used by RestoIpoptNLP: [x,n_c,p_c,n_d,p_d]. */
inline EvaluationValue<RestorationStageVariableLayout>
MakeCanonicalRestoIpoptVariableLayout(
   Index original_variables,
   Index equality_constraints,
   Index inequality_constraints
)
{
   Index equality_p_offset = 0;
   Index inequality_n_offset = 0;
   Index inequality_p_offset = 0;
   Index total = 0;
   if( !restoration_stage_detail::CheckedAdd(
          original_variables, equality_constraints, equality_p_offset) ||
       !restoration_stage_detail::CheckedAdd(
          equality_p_offset, equality_constraints, inequality_n_offset) ||
       !restoration_stage_detail::CheckedAdd(
          inequality_n_offset, inequality_constraints, inequality_p_offset) ||
       !restoration_stage_detail::CheckedAdd(
          inequality_p_offset, inequality_constraints, total) )
   {
      return std::unexpected(restoration_stage_detail::Error(
         "canonical restoration variable dimension overflows Index"));
   }
   RestorationStageVariableLayout result;
   result.original_primal.resize(original_variables);
   result.equality_n.resize(equality_constraints);
   result.equality_p.resize(equality_constraints);
   result.inequality_n.resize(inequality_constraints);
   result.inequality_p.resize(inequality_constraints);
   for( Index i = 0; i < original_variables; ++i )
   {
      result.original_primal[i] = i;
   }
   for( Index i = 0; i < equality_constraints; ++i )
   {
      result.equality_n[i] = original_variables + i;
      result.equality_p[i] = equality_p_offset + i;
   }
   for( Index i = 0; i < inequality_constraints; ++i )
   {
      result.inequality_n[i] = inequality_n_offset + i;
      result.inequality_p[i] = inequality_p_offset + i;
   }
   static_cast<void>(total);
   return result;
}

/** Build separate stage metadata for RestoIpoptNLP's compound formulation.
 *
 * Every n/p pair is placed in the stage of its constraint. For a dynamics
 * row this is the previous stage, so its +I/-I columns remain in the ordinary
 * outgoing dynamics Jacobian and the only next-stage derivative stays -I.
 * The function validates all role, variable, and bound maps explicitly.
 */
inline EvaluationValue<StageNlpTopology> MakeRestorationStageNlpTopology(
   const StageNlpTopology&             main_topology,
   const PrimalDualLayout&             main_layout,
   NlpStructure                        restoration_structure,
   const PrimalDualLayout&             restoration_layout,
   const RestorationStageVariableLayout& variables,
   std::uint64_t                       stage_revision
)
{
   using restoration_stage_detail::Error;
   if( !main_topology.configured() )
   {
      return std::unexpected(Error(
         "main stage topology is invalid: " +
         main_topology.configuration_error()));
   }
   if( stage_revision == 0 )
   {
      return std::unexpected(Error(
         "restoration stage topology requires a nonzero revision"));
   }
   const NlpStructure main_structure = main_topology.source_structure();
   const Index equality_count = main_layout.equality_constraints.size();
   const Index inequality_count = main_layout.inequality_constraints.size();
   Index expected_constraints = 0;
   Index expected_variables = 0;
   Index twice_equalities = 0;
   Index twice_inequalities = 0;
   if( !restoration_stage_detail::CheckedAdd(
          equality_count, inequality_count, expected_constraints) ||
       !restoration_stage_detail::CheckedAdd(
          equality_count, equality_count, twice_equalities) ||
       !restoration_stage_detail::CheckedAdd(
          inequality_count, inequality_count, twice_inequalities) ||
       !restoration_stage_detail::CheckedAdd(
          main_structure.variables, twice_equalities, expected_variables) ||
       !restoration_stage_detail::CheckedAdd(
          expected_variables, twice_inequalities, expected_variables) )
   {
      return std::unexpected(Error(
         "restoration stage dimensions overflow Index"));
   }
   if( main_structure.constraints != expected_constraints ||
       restoration_structure.constraints != expected_constraints ||
       restoration_structure.variables != expected_variables ||
       restoration_layout.equality_constraints.size() != equality_count ||
       restoration_layout.inequality_constraints.size() != inequality_count )
   {
      return std::unexpected(Error(
         "restoration dimensions do not match the main primal-dual layout"));
   }

   std::vector<Index> main_equality_position;
   std::vector<Index> main_inequality_position;
   std::vector<Index> restoration_equality_position;
   std::vector<Index> restoration_inequality_position;
   if( !restoration_stage_detail::ConfigureConstraintPositions(
          main_layout,
          main_structure.constraints,
          main_equality_position,
          main_inequality_position) ||
       !restoration_stage_detail::ConfigureConstraintPositions(
          restoration_layout,
          restoration_structure.constraints,
          restoration_equality_position,
          restoration_inequality_position) )
   {
      return std::unexpected(Error(
         "restoration constraint role maps are invalid"));
   }

   if( variables.original_primal.size() != main_structure.variables ||
       variables.equality_n.size() != equality_count ||
       variables.equality_p.size() != equality_count ||
       variables.inequality_n.size() != inequality_count ||
       variables.inequality_p.size() != inequality_count )
   {
      return std::unexpected(Error(
         "restoration compound variable maps have the wrong dimensions"));
   }
   std::vector<Index> all_variables;
   all_variables.reserve(expected_variables);
   all_variables.insert(
      all_variables.end(),
      variables.original_primal.begin(),
      variables.original_primal.end());
   all_variables.insert(
      all_variables.end(), variables.equality_n.begin(), variables.equality_n.end());
   all_variables.insert(
      all_variables.end(), variables.equality_p.begin(), variables.equality_p.end());
   all_variables.insert(
      all_variables.end(), variables.inequality_n.begin(), variables.inequality_n.end());
   all_variables.insert(
      all_variables.end(), variables.inequality_p.begin(), variables.inequality_p.end());
   if( !restoration_stage_detail::ValidateUniqueMap(
          all_variables, restoration_structure.variables) ||
       all_variables.size() != restoration_structure.variables )
   {
      return std::unexpected(Error(
         "restoration compound variable maps are not a full permutation"));
   }

   const auto map_original_bounds = [&](std::span<const Index> bounds)
   {
      std::vector<Index> mapped;
      mapped.reserve(bounds.size());
      for( Index bound : bounds )
      {
         if( bound >= variables.original_primal.size() )
         {
            mapped.clear();
            mapped.push_back(restoration_structure.variables);
            return mapped;
         }
         mapped.push_back(variables.original_primal[bound]);
      }
      return mapped;
   };
   std::vector<Index> expected_lower =
      map_original_bounds(main_layout.primal_lower_bounds);
   expected_lower.insert(
      expected_lower.end(), variables.equality_n.begin(), variables.equality_n.end());
   expected_lower.insert(
      expected_lower.end(), variables.equality_p.begin(), variables.equality_p.end());
   expected_lower.insert(
      expected_lower.end(), variables.inequality_n.begin(), variables.inequality_n.end());
   expected_lower.insert(
      expected_lower.end(), variables.inequality_p.begin(), variables.inequality_p.end());
   const std::vector<Index> expected_upper =
      map_original_bounds(main_layout.primal_upper_bounds);
   if( !restoration_stage_detail::ValidateUniqueMap(
          main_layout.primal_lower_bounds, main_structure.variables) ||
       !restoration_stage_detail::ValidateUniqueMap(
          main_layout.primal_upper_bounds, main_structure.variables) ||
       !restoration_stage_detail::ValidateUniqueMap(
          main_layout.slack_lower_bounds, inequality_count) ||
       !restoration_stage_detail::ValidateUniqueMap(
          main_layout.slack_upper_bounds, inequality_count) ||
       !restoration_stage_detail::ValidateUniqueMap(
          restoration_layout.primal_lower_bounds,
          restoration_structure.variables) ||
       !restoration_stage_detail::ValidateUniqueMap(
          restoration_layout.primal_upper_bounds,
          restoration_structure.variables) ||
       !restoration_stage_detail::ValidateUniqueMap(
          restoration_layout.slack_lower_bounds, inequality_count) ||
       !restoration_stage_detail::ValidateUniqueMap(
          restoration_layout.slack_upper_bounds, inequality_count) ||
       !restoration_stage_detail::SameIndexSet(
          expected_lower, restoration_layout.primal_lower_bounds) ||
       !restoration_stage_detail::SameIndexSet(
          expected_upper, restoration_layout.primal_upper_bounds) ||
       !restoration_stage_detail::SameIndexSet(
          main_layout.slack_lower_bounds,
          restoration_layout.slack_lower_bounds) ||
       !restoration_stage_detail::SameIndexSet(
          main_layout.slack_upper_bounds,
          restoration_layout.slack_upper_bounds) )
   {
      return std::unexpected(Error(
         "restoration bounds do not match original bounds plus n/p lower bounds"));
   }

   const Index missing = main_structure.constraints;
   const StageNlpOrdering& main_ordering = main_topology.ordering();
   const auto map_equality_row = [&](Index main_row) -> Index
   {
      const Index position = main_equality_position[main_row];
      return position == missing
         ? restoration_structure.constraints
         : restoration_layout.equality_constraints[position];
   };
   const auto map_inequality_row = [&](Index main_row) -> Index
   {
      const Index position = main_inequality_position[main_row];
      return position == missing
         ? restoration_structure.constraints
         : restoration_layout.inequality_constraints[position];
   };

   std::vector<OptimalControlStageDimensions> stages;
   stages.reserve(main_topology.stages().size());
   StageNlpOrdering ordering;
   ordering.primal.reserve(restoration_structure.variables);
   ordering.dynamics.reserve(main_ordering.dynamics.size());
   ordering.path_equalities.reserve(main_ordering.path_equalities.size());
   ordering.path_inequalities.reserve(main_ordering.path_inequalities.size());
   for( Index stage = 0; stage < main_topology.stages().size(); ++stage )
   {
      const OptimalControlStageDimensions main = main_topology.stages()[stage];
      const Index outgoing = stage + 1 < main_topology.stages().size()
         ? main_topology.stages()[stage + 1].states
         : 0;
      Index local_auxiliary_pairs = 0;
      Index local_auxiliaries = 0;
      Index controls = 0;
      if( !restoration_stage_detail::CheckedAdd(
             main.path_equalities,
             main.path_inequalities,
             local_auxiliary_pairs) ||
          !restoration_stage_detail::CheckedAdd(
             local_auxiliary_pairs, outgoing, local_auxiliary_pairs) ||
          !restoration_stage_detail::CheckedAdd(
             local_auxiliary_pairs,
             local_auxiliary_pairs,
             local_auxiliaries) ||
          !restoration_stage_detail::CheckedAdd(
             main.controls, local_auxiliaries, controls) )
      {
         return std::unexpected(Error(
            "restoration per-stage control dimension overflows Index"));
      }
      stages.push_back({
         controls,
         main.states,
         main.path_equalities,
         main.path_inequalities
      });

      const Index primal_begin = main_topology.primal_offsets()[stage];
      for( Index control = 0; control < main.controls; ++control )
      {
         ordering.primal.push_back(
            variables.original_primal[
               main_ordering.primal[primal_begin + control]]);
      }
      const Index equality_begin = main_topology.path_equality_offsets()[stage];
      for( Index local = 0; local < main.path_equalities; ++local )
      {
         const Index main_row = main_ordering.path_equalities[
            equality_begin + local];
         const Index position = main_equality_position[main_row];
         if( position == missing )
         {
            return std::unexpected(Error(
               "main path equality has the wrong constraint role"));
         }
         ordering.primal.push_back(variables.equality_n[position]);
         ordering.primal.push_back(variables.equality_p[position]);
         ordering.path_equalities.push_back(map_equality_row(main_row));
      }
      const Index inequality_begin =
         main_topology.path_inequality_offsets()[stage];
      for( Index local = 0; local < main.path_inequalities; ++local )
      {
         const Index main_row = main_ordering.path_inequalities[
            inequality_begin + local];
         const Index position = main_inequality_position[main_row];
         if( position == missing )
         {
            return std::unexpected(Error(
               "main path inequality has the wrong constraint role"));
         }
         ordering.primal.push_back(variables.inequality_n[position]);
         ordering.primal.push_back(variables.inequality_p[position]);
         ordering.path_inequalities.push_back(map_inequality_row(main_row));
      }
      if( stage + 1 < main_topology.stages().size() )
      {
         const Index dynamics_begin = main_topology.dynamics_offsets()[stage];
         for( Index state = 0; state < outgoing; ++state )
         {
            const Index main_row = main_ordering.dynamics[
               dynamics_begin + state];
            const Index position = main_equality_position[main_row];
            if( position == missing )
            {
               return std::unexpected(Error(
                  "main dynamics row has the wrong constraint role"));
            }
            ordering.primal.push_back(variables.equality_n[position]);
            ordering.primal.push_back(variables.equality_p[position]);
            ordering.dynamics.push_back(map_equality_row(main_row));
         }
      }
      for( Index state = 0; state < main.states; ++state )
      {
         ordering.primal.push_back(
            variables.original_primal[
               main_ordering.primal[
                  primal_begin + main.controls + state]]);
      }
   }

   StageNlpTopology result(
      restoration_structure,
      std::move(stages),
      std::move(ordering),
      stage_revision);
   if( !result.configured() )
   {
      return std::unexpected(Error(
         "constructed restoration topology is invalid: " +
         result.configuration_error()));
   }
   return result;
}
} // namespace Ipopt::Cxx23

#endif
