// Copyright (C) 2026 Ipopt contributors.
// All Rights Reserved.
// This file is distributed under the Eclipse Public License.

#ifndef IPOPT_CXX23_LEGACY_COORDINATE_ADAPTER_HPP
#define IPOPT_CXX23_LEGACY_COORDINATE_ADAPTER_HPP

#include <ipopt/cxx23/any_nlp_problem.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <expected>
#include <limits>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace Ipopt::Cxx23
{
struct FixedVariableEquality
{
   Index full_variable = 0;
   Number rhs = 0.;

   friend bool operator==(const FixedVariableEquality&, const FixedVariableEquality&) = default;
};

/** Maps a full user model to TNLPAdapter's reduced, ordered coordinates.
 *
 * internal_to_full_variables is the expanded-position map P_x_full_x. Values
 * not named by that map remain equal to full_variable_template, which is how
 * MAKE_PARAMETER represents fixed variables. Equality rows precede optional
 * MAKE_CONSTRAINT rows and inequality rows in the adapted problem.
 */
struct LegacyCoordinateMap
{
   std::vector<Index> internal_to_full_variables;
   std::vector<Number> full_variable_template;
   std::vector<Index> equality_from_full_constraints;
   std::vector<Number> equality_rhs;
   std::vector<FixedVariableEquality> fixed_variable_equalities;
   std::vector<Index> inequality_from_full_constraints;
};

/** Diagonal scaling used by OrigIpoptNLP.
 *
 * Empty vectors mean identity scaling. The equality vector includes appended
 * fixed-variable equalities. Internally x_scaled = variable * x_unscaled.
 */
struct LegacyNlpScaling
{
   Number objective = 1.;
   std::vector<Number> variables;
   std::vector<Number> equalities;
   std::vector<Number> inequalities;
};

namespace detail
{
class LegacyCoordinateModel
{
public:
   LegacyCoordinateModel(
      AnyNlpProblem       source,
      LegacyCoordinateMap coordinate_map,
      LegacyNlpScaling    scaling
   )
      : source_(std::move(source)),
        coordinate_map_(std::move(coordinate_map)),
        scaling_(std::move(scaling)),
        source_structure_(source_.nlp_structure())
   {
      initialization_result_ = Initialize();
   }

   LegacyCoordinateModel(const LegacyCoordinateModel&) = delete;
   LegacyCoordinateModel& operator=(const LegacyCoordinateModel&) = delete;
   LegacyCoordinateModel(LegacyCoordinateModel&&) = default;
   LegacyCoordinateModel& operator=(LegacyCoordinateModel&&) = delete;

   const EvaluationResult& initialization_result() const noexcept
   {
      return initialization_result_;
   }

   NlpStructure structure() const
   {
      return structure_;
   }

   JacobianProductCapabilities jacobian_product_capabilities() const noexcept
   {
      return jacobian_capabilities_;
   }

   HessianProductCapabilities hessian_product_capabilities() const noexcept
   {
      return hessian_capabilities_;
   }

   EvaluationValue<Number> eval_objective(std::span<const Number> x)
   {
      ExpandX(x);
      EvaluationValue<Number> objective = source_.nlp_objective(full_x_);
      if( !objective )
      {
         return std::unexpected(objective.error());
      }
      return scaling_.objective * *objective;
   }

   EvaluationResult eval_gradient(
      std::span<const Number> x,
      std::span<Number>       result
   )
   {
      ExpandX(x);
      if( EvaluationResult evaluated = source_.nlp_gradient(full_x_, full_gradient_); !evaluated )
      {
         return evaluated;
      }
      for( Index i = 0; i < structure_.variables; ++i )
      {
         output_scratch_[i] = scaling_.objective *
            full_gradient_[coordinate_map_.internal_to_full_variables[i]] /
            scaling_.variables[i];
      }
      std::copy_n(output_scratch_.begin(), structure_.variables, result.begin());
      return {};
   }

   EvaluationResult eval_constraints(
      std::span<const Number> x,
      std::span<Number>       result
   )
   {
      ExpandX(x);
      if( EvaluationResult evaluated = source_.nlp_constraints(full_x_, full_constraints_); !evaluated )
      {
         return evaluated;
      }

      const Index base_equalities = coordinate_map_.equality_from_full_constraints.size();
      for( Index i = 0; i < base_equalities; ++i )
      {
         output_scratch_[i] = scaling_.equalities[i] *
            (full_constraints_[coordinate_map_.equality_from_full_constraints[i]] -
             coordinate_map_.equality_rhs[i]);
      }
      for( Index i = 0; i < coordinate_map_.fixed_variable_equalities.size(); ++i )
      {
         const Index row = base_equalities + i;
         const FixedVariableEquality& equality = coordinate_map_.fixed_variable_equalities[i];
         output_scratch_[row] = scaling_.equalities[row] *
            (full_x_[equality.full_variable] - equality.rhs);
      }
      for( Index i = 0; i < coordinate_map_.inequality_from_full_constraints.size(); ++i )
      {
         output_scratch_[equality_count_ + i] = scaling_.inequalities[i] *
            full_constraints_[coordinate_map_.inequality_from_full_constraints[i]];
      }
      std::copy_n(output_scratch_.begin(), structure_.constraints, result.begin());
      return {};
   }

   EvaluationResult eval_jacobian_structure(
      std::span<Index> rows,
      std::span<Index> columns
   )
   {
      std::ranges::copy(jacobian_rows_, rows.begin());
      std::ranges::copy(jacobian_columns_, columns.begin());
      return {};
   }

   EvaluationResult eval_jacobian_values(
      std::span<const Number> x,
      std::span<Number>       values
   )
   {
      ExpandX(x);
      if( EvaluationResult evaluated = source_.nlp_jacobian_values(
             full_x_, full_jacobian_values_);
          !evaluated )
      {
         return evaluated;
      }

      for( Index i = 0; i < structure_.jacobian_nonzeros; ++i )
      {
         const Number unscaled_value = jacobian_sources_[i] < source_structure_.jacobian_nonzeros
            ? full_jacobian_values_[jacobian_sources_[i]]
            : 1.;
         output_scratch_[i] = ConstraintScaling(jacobian_rows_[i]) * unscaled_value /
            scaling_.variables[jacobian_columns_[i]];
      }
      std::copy_n(output_scratch_.begin(), structure_.jacobian_nonzeros, values.begin());
      return {};
   }

   EvaluationResult eval_hessian_structure(
      std::span<Index> rows,
      std::span<Index> columns
   )
   {
      std::ranges::copy(hessian_rows_, rows.begin());
      std::ranges::copy(hessian_columns_, columns.begin());
      return {};
   }

   EvaluationResult eval_hessian_values(
      std::span<const Number> x,
      Number                  objective_factor,
      std::span<const Number> multipliers,
      std::span<Number>       values
   )
   {
      ExpandX(x);
      ExpandMultipliers(multipliers);
      if( EvaluationResult evaluated = source_.nlp_hessian_values(
             full_x_, scaling_.objective * objective_factor,
             full_multipliers_, full_hessian_values_);
          !evaluated )
      {
         return evaluated;
      }

      for( Index i = 0; i < structure_.hessian_nonzeros; ++i )
      {
         output_scratch_[i] = full_hessian_values_[hessian_sources_[i]] /
            (scaling_.variables[hessian_rows_[i]] *
             scaling_.variables[hessian_columns_[i]]);
      }
      std::copy_n(output_scratch_.begin(), structure_.hessian_nonzeros, values.begin());
      return {};
   }

   EvaluationResult eval_jacobian_product(
      std::span<const Number> x,
      std::span<const Number> direction,
      std::span<Number>       result
   )
   {
      ExpandX(x);
      ExpandDirection(direction);
      if( EvaluationResult evaluated = source_.nlp_jacobian_product(
             full_x_, full_direction_, full_constraint_product_);
          !evaluated )
      {
         return evaluated;
      }

      const Index base_equalities = coordinate_map_.equality_from_full_constraints.size();
      for( Index i = 0; i < base_equalities; ++i )
      {
         output_scratch_[i] = scaling_.equalities[i] *
            full_constraint_product_[coordinate_map_.equality_from_full_constraints[i]];
      }
      for( Index i = 0; i < coordinate_map_.fixed_variable_equalities.size(); ++i )
      {
         const Index row = base_equalities + i;
         output_scratch_[row] = scaling_.equalities[row] *
            full_direction_[coordinate_map_.fixed_variable_equalities[i].full_variable];
      }
      for( Index i = 0; i < coordinate_map_.inequality_from_full_constraints.size(); ++i )
      {
         output_scratch_[equality_count_ + i] = scaling_.inequalities[i] *
            full_constraint_product_[coordinate_map_.inequality_from_full_constraints[i]];
      }
      std::copy_n(output_scratch_.begin(), structure_.constraints, result.begin());
      return {};
   }

   EvaluationResult eval_jacobian_transpose_product(
      std::span<const Number> x,
      std::span<const Number> direction,
      std::span<Number>       result
   )
   {
      ExpandX(x);
      std::ranges::fill(full_constraint_direction_, 0.);
      for( Index i = 0; i < coordinate_map_.equality_from_full_constraints.size(); ++i )
      {
         full_constraint_direction_[coordinate_map_.equality_from_full_constraints[i]] =
            scaling_.equalities[i] * direction[i];
      }
      for( Index i = 0; i < coordinate_map_.inequality_from_full_constraints.size(); ++i )
      {
         full_constraint_direction_[coordinate_map_.inequality_from_full_constraints[i]] =
            scaling_.inequalities[i] * direction[equality_count_ + i];
      }
      if( EvaluationResult evaluated = source_.nlp_jacobian_transpose_product(
             full_x_, full_constraint_direction_, full_variable_product_);
          !evaluated )
      {
         return evaluated;
      }

      for( Index i = 0; i < structure_.variables; ++i )
      {
         output_scratch_[i] =
            full_variable_product_[coordinate_map_.internal_to_full_variables[i]] /
            scaling_.variables[i];
      }
      const Index base_equalities = coordinate_map_.equality_from_full_constraints.size();
      for( Index i = 0; i < coordinate_map_.fixed_variable_equalities.size(); ++i )
      {
         const Index row = base_equalities + i;
         const Index internal = full_to_internal_[
            coordinate_map_.fixed_variable_equalities[i].full_variable];
         output_scratch_[internal] +=
            scaling_.equalities[row] * direction[row] / scaling_.variables[internal];
      }
      std::copy_n(output_scratch_.begin(), structure_.variables, result.begin());
      return {};
   }

   EvaluationResult eval_jacobian_products(
      std::span<const Number> x,
      std::span<const Number> forward_direction,
      std::span<const Number> transpose_direction,
      std::span<Number>       forward_result,
      std::span<Number>       transpose_result
   )
   {
      ExpandX(x);
      ExpandDirection(forward_direction);
      std::ranges::fill(full_constraint_direction_, 0.);
      for( Index i = 0; i < coordinate_map_.equality_from_full_constraints.size(); ++i )
      {
         full_constraint_direction_[coordinate_map_.equality_from_full_constraints[i]] =
            scaling_.equalities[i] * transpose_direction[i];
      }
      for( Index i = 0; i < coordinate_map_.inequality_from_full_constraints.size(); ++i )
      {
         full_constraint_direction_[coordinate_map_.inequality_from_full_constraints[i]] =
            scaling_.inequalities[i] * transpose_direction[equality_count_ + i];
      }
      if( EvaluationResult evaluated = source_.nlp_jacobian_products(
             full_x_, full_direction_, full_constraint_direction_,
             full_constraint_product_, full_variable_product_);
          !evaluated )
      {
         return evaluated;
      }

      const Index base_equalities = coordinate_map_.equality_from_full_constraints.size();
      for( Index i = 0; i < base_equalities; ++i )
      {
         forward_result[i] = scaling_.equalities[i] *
            full_constraint_product_[coordinate_map_.equality_from_full_constraints[i]];
      }
      for( Index i = 0; i < coordinate_map_.fixed_variable_equalities.size(); ++i )
      {
         const Index row = base_equalities + i;
         forward_result[row] = scaling_.equalities[row] *
            full_direction_[coordinate_map_.fixed_variable_equalities[i].full_variable];
      }
      for( Index i = 0; i < coordinate_map_.inequality_from_full_constraints.size(); ++i )
      {
         forward_result[equality_count_ + i] = scaling_.inequalities[i] *
            full_constraint_product_[coordinate_map_.inequality_from_full_constraints[i]];
      }

      for( Index i = 0; i < structure_.variables; ++i )
      {
         transpose_result[i] =
            full_variable_product_[coordinate_map_.internal_to_full_variables[i]] /
            scaling_.variables[i];
      }
      for( Index i = 0; i < coordinate_map_.fixed_variable_equalities.size(); ++i )
      {
         const Index row = base_equalities + i;
         const Index internal = full_to_internal_[
            coordinate_map_.fixed_variable_equalities[i].full_variable];
         transpose_result[internal] +=
            scaling_.equalities[row] * transpose_direction[row] /
            scaling_.variables[internal];
      }
      return {};
   }

   EvaluationResult eval_hessian_product(
      std::span<const Number> x,
      Number                  objective_factor,
      std::span<const Number> multipliers,
      std::span<const Number> direction,
      std::span<Number>       result
   )
   {
      ExpandX(x);
      ExpandMultipliers(multipliers);
      ExpandDirection(direction);
      if( EvaluationResult evaluated = source_.nlp_hessian_product(
             full_x_, scaling_.objective * objective_factor, full_multipliers_,
             full_direction_, full_variable_product_);
          !evaluated )
      {
         return evaluated;
      }

      for( Index i = 0; i < structure_.variables; ++i )
      {
         output_scratch_[i] =
            full_variable_product_[coordinate_map_.internal_to_full_variables[i]] /
            scaling_.variables[i];
      }
      std::copy_n(output_scratch_.begin(), structure_.variables, result.begin());
      return {};
   }

private:
   static constexpr Index missing_index = std::numeric_limits<Index>::max();

   static EvaluationResult InvalidLayout(std::string message)
   {
      return std::unexpected(EvaluationError{
         EvaluationErrorCode::invalid_layout,
         std::move(message)
      });
   }

   static EvaluationResult NormalizeScaling(
      std::vector<Number>& values,
      Index                expected,
      const char*          name
   )
   {
      if( values.empty() )
      {
         values.assign(expected, 1.);
      }
      else if( values.size() != expected )
      {
         return InvalidLayout(
            std::string(name) + " scaling has size " + std::to_string(values.size()) +
            ", expected " + std::to_string(expected));
      }
      for( Index i = 0; i < values.size(); ++i )
      {
         if( !std::isfinite(values[i]) || values[i] == 0. )
         {
            return InvalidLayout(
               std::string(name) + " scaling at index " + std::to_string(i) +
               " must be finite and nonzero");
         }
      }
      return {};
   }

   EvaluationResult Initialize()
   {
      if( coordinate_map_.full_variable_template.size() != source_structure_.variables )
      {
         return InvalidLayout(
            "full variable template has size " +
            std::to_string(coordinate_map_.full_variable_template.size()) +
            ", expected " + std::to_string(source_structure_.variables));
      }
      for( Index i = 0; i < coordinate_map_.full_variable_template.size(); ++i )
      {
         if( !std::isfinite(coordinate_map_.full_variable_template[i]) )
         {
            return InvalidLayout(
               "full variable template at index " + std::to_string(i) +
               " must be finite");
         }
      }
      if( coordinate_map_.equality_rhs.size() !=
          coordinate_map_.equality_from_full_constraints.size() )
      {
         return InvalidLayout("equality RHS size does not match the equality map");
      }
      for( Index i = 0; i < coordinate_map_.equality_rhs.size(); ++i )
      {
         if( !std::isfinite(coordinate_map_.equality_rhs[i]) )
         {
            return InvalidLayout(
               "equality RHS at index " + std::to_string(i) + " must be finite");
         }
      }

      full_to_internal_.assign(source_structure_.variables, missing_index);
      for( Index i = 0; i < coordinate_map_.internal_to_full_variables.size(); ++i )
      {
         const Index full = coordinate_map_.internal_to_full_variables[i];
         if( full >= source_structure_.variables )
         {
            return InvalidLayout(
               "variable map index " + std::to_string(full) + " is out of range");
         }
         if( full_to_internal_[full] != missing_index )
         {
            return InvalidLayout(
               "variable map contains duplicate full index " + std::to_string(full));
         }
         full_to_internal_[full] = i;
      }

      equality_from_full_.assign(source_structure_.constraints, missing_index);
      inequality_from_full_.assign(source_structure_.constraints, missing_index);
      for( Index i = 0; i < coordinate_map_.equality_from_full_constraints.size(); ++i )
      {
         const Index full = coordinate_map_.equality_from_full_constraints[i];
         if( full >= source_structure_.constraints )
         {
            return InvalidLayout(
               "equality map index " + std::to_string(full) + " is out of range");
         }
         if( equality_from_full_[full] != missing_index )
         {
            return InvalidLayout(
               "equality map contains duplicate full index " + std::to_string(full));
         }
         equality_from_full_[full] = i;
      }
      for( Index i = 0; i < coordinate_map_.inequality_from_full_constraints.size(); ++i )
      {
         const Index full = coordinate_map_.inequality_from_full_constraints[i];
         if( full >= source_structure_.constraints )
         {
            return InvalidLayout(
               "inequality map index " + std::to_string(full) + " is out of range");
         }
         if( inequality_from_full_[full] != missing_index ||
             equality_from_full_[full] != missing_index )
         {
            return InvalidLayout(
               "constraint maps contain duplicate full index " + std::to_string(full));
         }
         inequality_from_full_[full] = i;
      }

      std::vector<bool> fixed_equalities(source_structure_.variables, false);
      for( Index i = 0; i < coordinate_map_.fixed_variable_equalities.size(); ++i )
      {
         const FixedVariableEquality& equality = coordinate_map_.fixed_variable_equalities[i];
         if( equality.full_variable >= source_structure_.variables )
         {
            return InvalidLayout(
               "fixed equality variable " + std::to_string(equality.full_variable) +
               " is out of range");
         }
         if( full_to_internal_[equality.full_variable] == missing_index )
         {
            return InvalidLayout(
               "fixed equality variable " + std::to_string(equality.full_variable) +
               " was removed from the internal variables");
         }
         if( fixed_equalities[equality.full_variable] )
         {
            return InvalidLayout(
               "fixed equalities contain duplicate variable " +
               std::to_string(equality.full_variable));
         }
         if( !std::isfinite(equality.rhs) )
         {
            return InvalidLayout(
               "fixed equality RHS at index " + std::to_string(i) + " must be finite");
         }
         fixed_equalities[equality.full_variable] = true;
      }

      if( coordinate_map_.equality_from_full_constraints.size() >
          std::numeric_limits<Index>::max() - coordinate_map_.fixed_variable_equalities.size() )
      {
         return InvalidLayout("equality dimension overflows Index");
      }
      equality_count_ = coordinate_map_.equality_from_full_constraints.size() +
         coordinate_map_.fixed_variable_equalities.size();
      if( equality_count_ > std::numeric_limits<Index>::max() -
          coordinate_map_.inequality_from_full_constraints.size() )
      {
         return InvalidLayout("constraint dimension overflows Index");
      }
      const Index constraint_count = equality_count_ +
         coordinate_map_.inequality_from_full_constraints.size();

      if( !std::isfinite(scaling_.objective) || scaling_.objective == 0. )
      {
         return InvalidLayout("objective scaling must be finite and nonzero");
      }
      if( EvaluationResult valid = NormalizeScaling(
             scaling_.variables, coordinate_map_.internal_to_full_variables.size(), "variable");
          !valid )
      {
         return valid;
      }
      if( EvaluationResult valid = NormalizeScaling(
             scaling_.equalities, equality_count_, "equality");
          !valid )
      {
         return valid;
      }
      if( EvaluationResult valid = NormalizeScaling(
             scaling_.inequalities,
             coordinate_map_.inequality_from_full_constraints.size(), "inequality");
          !valid )
      {
         return valid;
      }

      jacobian_capabilities_ = source_.nlp_jacobian_product_capabilities();
      hessian_capabilities_ = source_.nlp_hessian_product_capabilities();
      const StructureFingerprintResult source_fingerprint = source_.nlp_structure_fingerprint();
      if( !source_fingerprint )
      {
         return std::unexpected(source_fingerprint.error());
      }

      std::vector<Index> full_jacobian_rows(source_structure_.jacobian_nonzeros);
      std::vector<Index> full_jacobian_columns(source_structure_.jacobian_nonzeros);
      if( EvaluationResult evaluated = source_.nlp_jacobian_structure(
             full_jacobian_rows, full_jacobian_columns);
          !evaluated )
      {
         return evaluated;
      }
      for( Index i = 0; i < source_structure_.jacobian_nonzeros; ++i )
      {
         const Index internal_column = full_to_internal_[full_jacobian_columns[i]];
         const Index internal_row = equality_from_full_[full_jacobian_rows[i]];
         if( internal_column != missing_index && internal_row != missing_index )
         {
            jacobian_rows_.push_back(internal_row);
            jacobian_columns_.push_back(internal_column);
            jacobian_sources_.push_back(i);
         }
      }
      const Index base_equalities = coordinate_map_.equality_from_full_constraints.size();
      for( Index i = 0; i < coordinate_map_.fixed_variable_equalities.size(); ++i )
      {
         jacobian_rows_.push_back(base_equalities + i);
         jacobian_columns_.push_back(
            full_to_internal_[coordinate_map_.fixed_variable_equalities[i].full_variable]);
         jacobian_sources_.push_back(source_structure_.jacobian_nonzeros);
      }
      for( Index i = 0; i < source_structure_.jacobian_nonzeros; ++i )
      {
         const Index internal_column = full_to_internal_[full_jacobian_columns[i]];
         const Index inequality = inequality_from_full_[full_jacobian_rows[i]];
         if( internal_column != missing_index && inequality != missing_index )
         {
            jacobian_rows_.push_back(equality_count_ + inequality);
            jacobian_columns_.push_back(internal_column);
            jacobian_sources_.push_back(i);
         }
      }

      std::vector<Index> full_hessian_rows(source_structure_.hessian_nonzeros);
      std::vector<Index> full_hessian_columns(source_structure_.hessian_nonzeros);
      if( EvaluationResult evaluated = source_.nlp_hessian_structure(
             full_hessian_rows, full_hessian_columns);
          !evaluated )
      {
         return evaluated;
      }
      for( Index i = 0; i < source_structure_.hessian_nonzeros; ++i )
      {
         Index row = full_to_internal_[full_hessian_rows[i]];
         Index column = full_to_internal_[full_hessian_columns[i]];
         if( row == missing_index || column == missing_index )
         {
            continue;
         }
         if( row < column )
         {
            std::swap(row, column);
         }
         hessian_rows_.push_back(row);
         hessian_columns_.push_back(column);
         hessian_sources_.push_back(i);
      }

      StructureFingerprint derived_revision = *source_fingerprint;
      HashWord(derived_revision, 0x6c656761637978ULL);
      HashIndices(derived_revision, coordinate_map_.internal_to_full_variables, 1);
      HashIndices(derived_revision, coordinate_map_.equality_from_full_constraints, 2);
      HashWord(derived_revision, coordinate_map_.fixed_variable_equalities.size());
      for( const FixedVariableEquality& equality : coordinate_map_.fixed_variable_equalities )
      {
         HashWord(derived_revision, equality.full_variable);
      }
      HashIndices(derived_revision, coordinate_map_.inequality_from_full_constraints, 3);
      const std::uint64_t revision = derived_revision.low ^
         (derived_revision.high << 17U) ^ (derived_revision.high >> 47U);

      structure_ = {
         coordinate_map_.internal_to_full_variables.size(),
         constraint_count,
         jacobian_rows_.size(),
         hessian_rows_.size(),
         revision
      };

      full_x_ = coordinate_map_.full_variable_template;
      full_gradient_.resize(source_structure_.variables);
      full_constraints_.resize(source_structure_.constraints);
      full_jacobian_values_.resize(source_structure_.jacobian_nonzeros);
      full_hessian_values_.resize(source_structure_.hessian_nonzeros);
      full_direction_.resize(source_structure_.variables);
      full_constraint_direction_.resize(source_structure_.constraints);
      full_constraint_product_.resize(source_structure_.constraints);
      full_variable_product_.resize(source_structure_.variables);
      full_multipliers_.resize(source_structure_.constraints);
      output_scratch_.resize(std::max({
         structure_.variables,
         structure_.constraints,
         structure_.jacobian_nonzeros,
         structure_.hessian_nonzeros
      }));
      return {};
   }

   static void HashIndices(
      StructureFingerprint&       fingerprint,
      std::span<const Index>      indices,
      std::uint64_t               discriminator
   ) noexcept
   {
      HashWord(fingerprint, discriminator);
      HashWord(fingerprint, indices.size());
      for( const Index index : indices )
      {
         HashWord(fingerprint, index);
      }
   }

   void ExpandX(std::span<const Number> x)
   {
      for( Index i = 0; i < structure_.variables; ++i )
      {
         full_x_[coordinate_map_.internal_to_full_variables[i]] =
            x[i] / scaling_.variables[i];
      }
   }

   void ExpandDirection(std::span<const Number> direction)
   {
      std::ranges::fill(full_direction_, 0.);
      for( Index i = 0; i < structure_.variables; ++i )
      {
         full_direction_[coordinate_map_.internal_to_full_variables[i]] =
            direction[i] / scaling_.variables[i];
      }
   }

   void ExpandMultipliers(std::span<const Number> multipliers)
   {
      std::ranges::fill(full_multipliers_, 0.);
      for( Index i = 0; i < coordinate_map_.equality_from_full_constraints.size(); ++i )
      {
         full_multipliers_[coordinate_map_.equality_from_full_constraints[i]] =
            scaling_.equalities[i] * multipliers[i];
      }
      for( Index i = 0; i < coordinate_map_.inequality_from_full_constraints.size(); ++i )
      {
         full_multipliers_[coordinate_map_.inequality_from_full_constraints[i]] =
            scaling_.inequalities[i] * multipliers[equality_count_ + i];
      }
   }

   Number ConstraintScaling(Index row) const
   {
      if( row < equality_count_ )
      {
         return scaling_.equalities[row];
      }
      return scaling_.inequalities[row - equality_count_];
   }

   AnyNlpProblem source_;
   LegacyCoordinateMap coordinate_map_;
   LegacyNlpScaling scaling_;
   const NlpStructure source_structure_;
   NlpStructure structure_;
   EvaluationResult initialization_result_;
   Index equality_count_ = 0;
   JacobianProductCapabilities jacobian_capabilities_;
   HessianProductCapabilities hessian_capabilities_;
   std::vector<Index> full_to_internal_;
   std::vector<Index> equality_from_full_;
   std::vector<Index> inequality_from_full_;
   std::vector<Index> jacobian_rows_;
   std::vector<Index> jacobian_columns_;
   std::vector<Index> jacobian_sources_;
   std::vector<Index> hessian_rows_;
   std::vector<Index> hessian_columns_;
   std::vector<Index> hessian_sources_;
   std::vector<Number> full_x_;
   std::vector<Number> full_gradient_;
   std::vector<Number> full_constraints_;
   std::vector<Number> full_jacobian_values_;
   std::vector<Number> full_hessian_values_;
   std::vector<Number> full_direction_;
   std::vector<Number> full_constraint_direction_;
   std::vector<Number> full_constraint_product_;
   std::vector<Number> full_variable_product_;
   std::vector<Number> full_multipliers_;
   std::vector<Number> output_scratch_;
};
} // namespace detail

inline EvaluationValue<AnyNlpProblem> MakeLegacyCoordinateProblem(
   AnyNlpProblem       source,
   LegacyCoordinateMap coordinate_map,
   LegacyNlpScaling    scaling = {}
)
{
   detail::LegacyCoordinateModel model(
      std::move(source), std::move(coordinate_map), std::move(scaling));
   if( !model.initialization_result() )
   {
      return std::unexpected(model.initialization_result().error());
   }
   AnyNlpProblem result = MakeNlpProblem(std::move(model));
   return result;
}
} // namespace Ipopt::Cxx23

#endif
