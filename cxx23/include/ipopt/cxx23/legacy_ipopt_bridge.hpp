// Copyright (C) 2026 Ipopt contributors.
// All Rights Reserved.
// This file is distributed under the Eclipse Public License.

#ifndef IPOPT_CXX23_LEGACY_IPOPT_BRIDGE_HPP
#define IPOPT_CXX23_LEGACY_IPOPT_BRIDGE_HPP

#include <ipopt/cxx23/legacy_coordinate_adapter.hpp>

#include <IpException.hpp>
#include <IpExpansionMatrix.hpp>
#include <IpOrigIpoptNLP.hpp>
#include <IpTNLPAdapter.hpp>
#include <IpTripletHelper.hpp>

#include <algorithm>
#include <cstdint>
#include <exception>
#include <expected>
#include <limits>
#include <new>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace Ipopt::Cxx23
{
/** Owning snapshot of the stable Ipopt coordinate conversion.
 *
 * Export is valid only after OrigIpoptNLP::InitializeStructures has completed.
 * No SmartPtr, Matrix, Vector, or pointer into the stable solver escapes this
 * object, so it can safely outlive the legacy solve objects.
 */
struct LegacyIpoptBridgeData
{
   LegacyCoordinateMap coordinate_map;
   LegacyNlpScaling scaling;
};

namespace legacy_ipopt_detail
{
inline EvaluationError BridgeError(
   EvaluationErrorCode code,
   std::string         message
)
{
   return {code, std::move(message)};
}

inline EvaluationValue<Index> CheckedIndex(
   ::Ipopt::Index   value,
   std::string_view name
)
{
   if( !std::in_range<Index>(value) )
   {
      return std::unexpected(BridgeError(
         EvaluationErrorCode::invalid_layout,
         std::string(name) + " is outside the C++23 Index range"));
   }
   return static_cast<Index>(value);
}

inline EvaluationValue<std::vector<Index>> ExpansionPositions(
   const ::Ipopt::ExpansionMatrix& matrix,
   Index                           expected_rows,
   Index                           expected_columns,
   std::string_view                name
)
{
   EvaluationValue<Index> rows = CheckedIndex(matrix.NRows(), name);
   if( !rows )
   {
      return std::unexpected(rows.error());
   }
   EvaluationValue<Index> columns = CheckedIndex(matrix.NCols(), name);
   if( !columns )
   {
      return std::unexpected(columns.error());
   }
   if( *rows != expected_rows || *columns != expected_columns )
   {
      return std::unexpected(BridgeError(
         EvaluationErrorCode::invalid_layout,
         std::string(name) + " has dimensions " + std::to_string(*rows) +
            "x" + std::to_string(*columns) + ", expected " +
            std::to_string(expected_rows) + "x" +
            std::to_string(expected_columns)));
   }

   std::vector<Index> positions(expected_columns);
   const ::Ipopt::Index* stable_positions = matrix.ExpandedPosIndices();
   if( expected_columns != 0 && stable_positions == nullptr )
   {
      return std::unexpected(BridgeError(
         EvaluationErrorCode::invalid_layout,
         std::string(name) + " has no expanded-position array"));
   }
   for( Index i = 0; i < expected_columns; ++i )
   {
      EvaluationValue<Index> position = CheckedIndex(stable_positions[i], name);
      if( !position )
      {
         return std::unexpected(position.error());
      }
      if( *position >= expected_rows )
      {
         return std::unexpected(BridgeError(
            EvaluationErrorCode::invalid_layout,
            std::string(name) + " contains out-of-range position " +
               std::to_string(*position)));
      }
      positions[i] = *position;
   }
   return positions;
}

enum class ScalingKind
{
   variables,
   equalities,
   inequalities
};

inline EvaluationValue<std::vector<Number>> ExtractScaling(
   ::Ipopt::NLPScalingObject&                 scaling,
   const ::Ipopt::SmartPtr<const ::Ipopt::VectorSpace>& space,
   ScalingKind                                kind
)
{
   if( !::Ipopt::IsValid(space) )
   {
      return std::unexpected(BridgeError(
         EvaluationErrorCode::invalid_layout,
         "legacy scaling vector space is not initialized"));
   }
   EvaluationValue<Index> dimension = CheckedIndex(space->Dim(), "scaling dimension");
   if( !dimension )
   {
      return std::unexpected(dimension.error());
   }

   ::Ipopt::SmartPtr<::Ipopt::Vector> ones = space->MakeNew();
   ones->Set(1.);
   ::Ipopt::SmartPtr<const ::Ipopt::Vector> scaled;
   switch( kind )
   {
      case ScalingKind::variables:
         scaled = scaling.apply_vector_scaling_x(::Ipopt::ConstPtr(ones));
         break;
      case ScalingKind::equalities:
         scaled = scaling.apply_vector_scaling_c(::Ipopt::ConstPtr(ones));
         break;
      case ScalingKind::inequalities:
         scaled = scaling.apply_vector_scaling_d(::Ipopt::ConstPtr(ones));
         break;
   }
   if( !::Ipopt::IsValid(scaled) || scaled->Dim() != space->Dim() )
   {
      return std::unexpected(BridgeError(
         EvaluationErrorCode::invalid_layout,
         "legacy scaling operation returned an invalid vector"));
   }

   std::vector<::Ipopt::Number> stable_values(*dimension);
   ::Ipopt::TripletHelper::FillValuesFromVector(
      space->Dim(), *scaled, stable_values.data());
   std::vector<Number> result(*dimension);
   for( Index i = 0; i < *dimension; ++i )
   {
      result[i] = static_cast<Number>(stable_values[i]);
   }
   return result;
}

/** Safe owning adapter around the full, unscaled TNLP callbacks.
 *
 * TNLP's new_x/new_lambda flags are always true. This may forgo a legacy
 * callback cache, but cannot expose the stale-x bug from the experimental
 * jac_vp branch. AnyNlpProblem supplies dimension, alias, sparsity, and
 * materialized-product validation around this adapter.
 */
class TnlpModel
{
public:
   explicit TnlpModel(
      ::Ipopt::SmartPtr<::Ipopt::TNLP> tnlp,
      std::uint64_t                    structural_revision
   )
      : tnlp_(std::move(tnlp))
   {
      initialization_result_ = Initialize(structural_revision);
   }

   TnlpModel(const TnlpModel&) = delete;
   TnlpModel& operator=(const TnlpModel&) = delete;
   TnlpModel(TnlpModel&&) = default;
   TnlpModel& operator=(TnlpModel&&) = delete;

   const EvaluationResult& initialization_result() const noexcept
   {
      return initialization_result_;
   }

   NlpStructure structure() const
   {
      return structure_;
   }

   EvaluationValue<Number> eval_objective(std::span<const Number> x)
   {
      CopyToStable(x, stable_x_);
      ::Ipopt::Number objective = 0.;
      EvaluationResult evaluated = Invoke(
         "TNLP::eval_f",
         [&]
         {
            return tnlp_->eval_f(
               stable_variables_, stable_x_.data(), true, objective);
         });
      if( !evaluated )
      {
         return std::unexpected(evaluated.error());
      }
      return static_cast<Number>(objective);
   }

   EvaluationResult eval_gradient(
      std::span<const Number> x,
      std::span<Number>       result
   )
   {
      CopyToStable(x, stable_x_);
      EvaluationResult evaluated = Invoke(
         "TNLP::eval_grad_f",
         [&]
         {
            return tnlp_->eval_grad_f(
               stable_variables_, stable_x_.data(), true,
               stable_gradient_.data());
         });
      if( !evaluated )
      {
         return evaluated;
      }
      CopyFromStable(stable_gradient_, result);
      return {};
   }

   EvaluationResult eval_constraints(
      std::span<const Number> x,
      std::span<Number>       result
   )
   {
      CopyToStable(x, stable_x_);
      EvaluationResult evaluated = Invoke(
         "TNLP::eval_g",
         [&]
         {
            return tnlp_->eval_g(
               stable_variables_, stable_x_.data(), true,
               stable_constraints_, stable_constraint_values_.data());
         });
      if( !evaluated )
      {
         return evaluated;
      }
      CopyFromStable(stable_constraint_values_, result);
      return {};
   }

   EvaluationResult eval_jacobian_structure(
      std::span<Index> rows,
      std::span<Index> columns
   )
   {
      EvaluationResult evaluated = Invoke(
         "TNLP::eval_jac_g(structure)",
         [&]
         {
            return tnlp_->eval_jac_g(
               stable_variables_, nullptr, false,
               stable_constraints_, stable_jacobian_nonzeros_,
               stable_jacobian_rows_.data(), stable_jacobian_columns_.data(),
               nullptr);
         });
      if( !evaluated )
      {
         return evaluated;
      }
      return CopySparsity(
         stable_jacobian_rows_, stable_jacobian_columns_, rows, columns,
         "TNLP Jacobian");
   }

   EvaluationResult eval_jacobian_values(
      std::span<const Number> x,
      std::span<Number>       values
   )
   {
      CopyToStable(x, stable_x_);
      EvaluationResult evaluated = Invoke(
         "TNLP::eval_jac_g(values)",
         [&]
         {
            return tnlp_->eval_jac_g(
               stable_variables_, stable_x_.data(), true,
               stable_constraints_, stable_jacobian_nonzeros_,
               nullptr, nullptr, stable_jacobian_values_.data());
         });
      if( !evaluated )
      {
         return evaluated;
      }
      CopyFromStable(stable_jacobian_values_, values);
      return {};
   }

   EvaluationResult eval_hessian_structure(
      std::span<Index> rows,
      std::span<Index> columns
   )
   {
      EvaluationResult evaluated = Invoke(
         "TNLP::eval_h(structure)",
         [&]
         {
            return tnlp_->eval_h(
               stable_variables_, nullptr, false, 0.,
               stable_constraints_, nullptr, false,
               stable_hessian_nonzeros_, stable_hessian_rows_.data(),
               stable_hessian_columns_.data(), nullptr);
         });
      if( !evaluated )
      {
         return evaluated;
      }
      return CopySparsity(
         stable_hessian_rows_, stable_hessian_columns_, rows, columns,
         "TNLP Hessian");
   }

   EvaluationResult eval_hessian_values(
      std::span<const Number> x,
      Number                  objective_factor,
      std::span<const Number> multipliers,
      std::span<Number>       values
   )
   {
      CopyToStable(x, stable_x_);
      CopyToStable(multipliers, stable_multipliers_);
      EvaluationResult evaluated = Invoke(
         "TNLP::eval_h(values)",
         [&]
         {
            return tnlp_->eval_h(
               stable_variables_, stable_x_.data(), true,
               static_cast<::Ipopt::Number>(objective_factor),
               stable_constraints_, stable_multipliers_.data(), true,
               stable_hessian_nonzeros_, nullptr, nullptr,
               stable_hessian_values_.data());
         });
      if( !evaluated )
      {
         return evaluated;
      }
      CopyFromStable(stable_hessian_values_, values);
      return {};
   }

private:
   template <class Callback>
   static EvaluationResult Invoke(
      std::string_view operation,
      Callback&&       callback
   )
   {
      try
      {
         if( !std::forward<Callback>(callback)() )
         {
            return std::unexpected(BridgeError(
               EvaluationErrorCode::model_failure,
               std::string(operation) + " returned false"));
         }
         return {};
      }
      catch( const ::Ipopt::IpoptException& error )
      {
         return std::unexpected(BridgeError(
            EvaluationErrorCode::model_failure,
            std::string(operation) + " threw: " + error.Message()));
      }
      catch( const std::bad_alloc& )
      {
         throw;
      }
      catch( const std::exception& error )
      {
         return std::unexpected(BridgeError(
            EvaluationErrorCode::model_failure,
            std::string(operation) + " threw: " + error.what()));
      }
   }

   EvaluationResult Initialize(std::uint64_t structural_revision)
   {
      if( !::Ipopt::IsValid(tnlp_) )
      {
         return std::unexpected(BridgeError(
            EvaluationErrorCode::invalid_layout,
            "cannot adapt a null TNLP"));
      }
      ::Ipopt::TNLP::IndexStyleEnum index_style = ::Ipopt::TNLP::C_STYLE;
      EvaluationResult described = Invoke(
         "TNLP::get_nlp_info",
         [&]
         {
            return tnlp_->get_nlp_info(
               stable_variables_, stable_constraints_,
               stable_jacobian_nonzeros_, stable_hessian_nonzeros_,
               index_style);
         });
      if( !described )
      {
         return described;
      }
      EvaluationValue<Index> variables =
         CheckedIndex(stable_variables_, "TNLP variable dimension");
      EvaluationValue<Index> constraints =
         CheckedIndex(stable_constraints_, "TNLP constraint dimension");
      EvaluationValue<Index> jacobian_nonzeros =
         CheckedIndex(stable_jacobian_nonzeros_, "TNLP Jacobian nonzero count");
      EvaluationValue<Index> hessian_nonzeros =
         CheckedIndex(stable_hessian_nonzeros_, "TNLP Hessian nonzero count");
      if( !variables )
      {
         return std::unexpected(variables.error());
      }
      if( !constraints )
      {
         return std::unexpected(constraints.error());
      }
      if( !jacobian_nonzeros )
      {
         return std::unexpected(jacobian_nonzeros.error());
      }
      if( !hessian_nonzeros )
      {
         return std::unexpected(hessian_nonzeros.error());
      }
      if( index_style != ::Ipopt::TNLP::C_STYLE &&
          index_style != ::Ipopt::TNLP::FORTRAN_STYLE )
      {
         return std::unexpected(BridgeError(
            EvaluationErrorCode::invalid_sparsity,
            "TNLP returned an unknown sparsity index style"));
      }
      fortran_style_ = index_style == ::Ipopt::TNLP::FORTRAN_STYLE;
      structure_ = {
         *variables,
         *constraints,
         *jacobian_nonzeros,
         *hessian_nonzeros,
         structural_revision
      };
      stable_x_.resize(*variables);
      stable_gradient_.resize(*variables);
      stable_constraint_values_.resize(*constraints);
      stable_multipliers_.resize(*constraints);
      stable_jacobian_rows_.resize(*jacobian_nonzeros);
      stable_jacobian_columns_.resize(*jacobian_nonzeros);
      stable_jacobian_values_.resize(*jacobian_nonzeros);
      stable_hessian_rows_.resize(*hessian_nonzeros);
      stable_hessian_columns_.resize(*hessian_nonzeros);
      stable_hessian_values_.resize(*hessian_nonzeros);
      return {};
   }

   template <class StableContainer>
   static void CopyToStable(
      std::span<const Number> source,
      StableContainer&        destination
   )
   {
      for( Index i = 0; i < source.size(); ++i )
      {
         destination[i] = static_cast<::Ipopt::Number>(source[i]);
      }
   }

   static void CopyFromStable(
      const std::vector<::Ipopt::Number>& source,
      std::span<Number>                   destination
   )
   {
      for( Index i = 0; i < source.size(); ++i )
      {
         destination[i] = static_cast<Number>(source[i]);
      }
   }

   EvaluationResult CopySparsity(
      const std::vector<::Ipopt::Index>& source_rows,
      const std::vector<::Ipopt::Index>& source_columns,
      std::span<Index>                   rows,
      std::span<Index>                   columns,
      std::string_view                   name
   ) const
   {
      for( Index i = 0; i < source_rows.size(); ++i )
      {
         ::Ipopt::Index stable_row = source_rows[i];
         ::Ipopt::Index stable_column = source_columns[i];
         if( fortran_style_ )
         {
            if( stable_row <= 0 || stable_column <= 0 )
            {
               return std::unexpected(BridgeError(
                  EvaluationErrorCode::invalid_sparsity,
                  std::string(name) +
                     " contains a nonpositive FORTRAN_STYLE index"));
            }
            --stable_row;
            --stable_column;
         }
         EvaluationValue<Index> row = CheckedIndex(stable_row, name);
         EvaluationValue<Index> column = CheckedIndex(stable_column, name);
         if( !row )
         {
            return std::unexpected(row.error());
         }
         if( !column )
         {
            return std::unexpected(column.error());
         }
         rows[i] = *row;
         columns[i] = *column;
      }
      return {};
   }

   ::Ipopt::SmartPtr<::Ipopt::TNLP> tnlp_;
   NlpStructure structure_;
   EvaluationResult initialization_result_;
   ::Ipopt::Index stable_variables_ = 0;
   ::Ipopt::Index stable_constraints_ = 0;
   ::Ipopt::Index stable_jacobian_nonzeros_ = 0;
   ::Ipopt::Index stable_hessian_nonzeros_ = 0;
   bool fortran_style_ = false;
   std::vector<::Ipopt::Number> stable_x_;
   std::vector<::Ipopt::Number> stable_gradient_;
   std::vector<::Ipopt::Number> stable_constraint_values_;
   std::vector<::Ipopt::Number> stable_multipliers_;
   std::vector<::Ipopt::Index> stable_jacobian_rows_;
   std::vector<::Ipopt::Index> stable_jacobian_columns_;
   std::vector<::Ipopt::Number> stable_jacobian_values_;
   std::vector<::Ipopt::Index> stable_hessian_rows_;
   std::vector<::Ipopt::Index> stable_hessian_columns_;
   std::vector<::Ipopt::Number> stable_hessian_values_;
};

inline EvaluationValue<LegacyIpoptBridgeData> ExportImpl(
   ::Ipopt::TNLPAdapter& adapter,
   ::Ipopt::OrigIpoptNLP& orig_nlp
)
{
   ::Ipopt::SmartPtr<::Ipopt::NLP> underlying = orig_nlp.nlp();
   if( !::Ipopt::IsValid(underlying) ||
       ::Ipopt::GetRawPtr(underlying) != static_cast<::Ipopt::NLP*>(&adapter) )
   {
      return std::unexpected(BridgeError(
         EvaluationErrorCode::structure_mismatch,
         "OrigIpoptNLP is not backed by the supplied TNLPAdapter"));
   }

   ::Ipopt::SmartPtr<const ::Ipopt::VectorSpace> x_space;
   ::Ipopt::SmartPtr<const ::Ipopt::VectorSpace> c_space;
   ::Ipopt::SmartPtr<const ::Ipopt::VectorSpace> d_space;
   ::Ipopt::SmartPtr<const ::Ipopt::VectorSpace> x_lower_space;
   ::Ipopt::SmartPtr<const ::Ipopt::MatrixSpace> p_x_lower_space;
   ::Ipopt::SmartPtr<const ::Ipopt::VectorSpace> x_upper_space;
   ::Ipopt::SmartPtr<const ::Ipopt::MatrixSpace> p_x_upper_space;
   ::Ipopt::SmartPtr<const ::Ipopt::VectorSpace> d_lower_space;
   ::Ipopt::SmartPtr<const ::Ipopt::MatrixSpace> p_d_lower_space;
   ::Ipopt::SmartPtr<const ::Ipopt::VectorSpace> d_upper_space;
   ::Ipopt::SmartPtr<const ::Ipopt::MatrixSpace> p_d_upper_space;
   ::Ipopt::SmartPtr<const ::Ipopt::MatrixSpace> jacobian_c_space;
   ::Ipopt::SmartPtr<const ::Ipopt::MatrixSpace> jacobian_d_space;
   ::Ipopt::SmartPtr<const ::Ipopt::SymMatrixSpace> hessian_space;
   orig_nlp.GetSpaces(
      x_space, c_space, d_space,
      x_lower_space, p_x_lower_space, x_upper_space, p_x_upper_space,
      d_lower_space, p_d_lower_space, d_upper_space, p_d_upper_space,
      jacobian_c_space, jacobian_d_space, hessian_space);
   if( !::Ipopt::IsValid(x_space) || !::Ipopt::IsValid(c_space) ||
       !::Ipopt::IsValid(d_space) )
   {
      return std::unexpected(BridgeError(
         EvaluationErrorCode::invalid_layout,
         "OrigIpoptNLP structures have not been initialized"));
   }

   ::Ipopt::Index stable_full_variables = 0;
   ::Ipopt::Index stable_full_constraints = 0;
   adapter.GetFullDimensions(stable_full_variables, stable_full_constraints);
   EvaluationValue<Index> full_variables =
      CheckedIndex(stable_full_variables, "full variable dimension");
   if( !full_variables )
   {
      return std::unexpected(full_variables.error());
   }
   EvaluationValue<Index> full_constraints =
      CheckedIndex(stable_full_constraints, "full constraint dimension");
   if( !full_constraints )
   {
      return std::unexpected(full_constraints.error());
   }
   EvaluationValue<Index> internal_variables =
      CheckedIndex(x_space->Dim(), "internal variable dimension");
   EvaluationValue<Index> internal_equalities =
      CheckedIndex(c_space->Dim(), "internal equality dimension");
   EvaluationValue<Index> internal_inequalities =
      CheckedIndex(d_space->Dim(), "internal inequality dimension");
   if( !internal_variables )
   {
      return std::unexpected(internal_variables.error());
   }
   if( !internal_equalities )
   {
      return std::unexpected(internal_equalities.error());
   }
   if( !internal_inequalities )
   {
      return std::unexpected(internal_inequalities.error());
   }

   ::Ipopt::SmartPtr<const ::Ipopt::ExpansionMatrix> p_x_full_x;
   ::Ipopt::SmartPtr<const ::Ipopt::ExpansionMatrix> p_x_x_lower;
   ::Ipopt::SmartPtr<const ::Ipopt::ExpansionMatrix> p_x_x_upper;
   ::Ipopt::SmartPtr<const ::Ipopt::ExpansionMatrix> p_c_g;
   ::Ipopt::SmartPtr<const ::Ipopt::ExpansionMatrix> p_d_g;
   adapter.GetPermutationMatrices(
      p_x_full_x, p_x_x_lower, p_x_x_upper, p_c_g, p_d_g);
   if( !::Ipopt::IsValid(p_c_g) || !::Ipopt::IsValid(p_d_g) )
   {
      return std::unexpected(BridgeError(
         EvaluationErrorCode::invalid_layout,
         "TNLPAdapter constraint maps have not been initialized"));
   }

   ::Ipopt::Index stable_fixed_count = 0;
   ::Ipopt::Index* stable_fixed_indices = nullptr;
   ::Ipopt::TNLPAdapter::FixedVariableTreatmentEnum fixed_treatment;
   adapter.GetFixedVariables(
      stable_fixed_count, stable_fixed_indices, fixed_treatment);
   EvaluationValue<Index> fixed_count =
      CheckedIndex(stable_fixed_count, "fixed variable count");
   if( !fixed_count )
   {
      return std::unexpected(fixed_count.error());
   }
   if( *fixed_count != 0 && stable_fixed_indices == nullptr )
   {
      return std::unexpected(BridgeError(
         EvaluationErrorCode::invalid_layout,
         "TNLPAdapter fixed-variable map is missing"));
   }

   const bool removes_fixed_variables =
      fixed_treatment == ::Ipopt::TNLPAdapter::MAKE_PARAMETER ||
      fixed_treatment == ::Ipopt::TNLPAdapter::MAKE_PARAMETER_NODUAL;
   const bool adds_fixed_equalities =
      fixed_treatment == ::Ipopt::TNLPAdapter::MAKE_CONSTRAINT;
   if( removes_fixed_variables != ::Ipopt::IsValid(p_x_full_x) && *fixed_count != 0 )
   {
      return std::unexpected(BridgeError(
         EvaluationErrorCode::invalid_layout,
         "TNLPAdapter variable map disagrees with fixed-variable treatment"));
   }

   LegacyIpoptBridgeData result;
   if( ::Ipopt::IsValid(p_x_full_x) )
   {
      EvaluationValue<std::vector<Index>> positions = ExpansionPositions(
         *p_x_full_x, *full_variables, *internal_variables, "P_x_full_x");
      if( !positions )
      {
         return std::unexpected(positions.error());
      }
      result.coordinate_map.internal_to_full_variables = std::move(*positions);
   }
   else
   {
      if( *internal_variables != *full_variables )
      {
         return std::unexpected(BridgeError(
            EvaluationErrorCode::invalid_layout,
            "identity variable map has inconsistent dimensions"));
      }
      result.coordinate_map.internal_to_full_variables.resize(*full_variables);
      for( Index i = 0; i < *full_variables; ++i )
      {
         result.coordinate_map.internal_to_full_variables[i] = i;
      }
   }

   std::vector<::Ipopt::Number> stable_template(*full_variables);
   ::Ipopt::SmartPtr<::Ipopt::Vector> zero_x = x_space->MakeNew();
   zero_x->Set(0.);
   adapter.ResortX(*zero_x, stable_template.data(), true);
   result.coordinate_map.full_variable_template.resize(*full_variables);
   for( Index i = 0; i < *full_variables; ++i )
   {
      result.coordinate_map.full_variable_template[i] =
         static_cast<Number>(stable_template[i]);
   }

   EvaluationValue<Index> base_equalities =
      CheckedIndex(p_c_g->NCols(), "base equality dimension");
   if( !base_equalities )
   {
      return std::unexpected(base_equalities.error());
   }
   const Index appended_equalities = adds_fixed_equalities ? *fixed_count : 0;
   if( *base_equalities >
       std::numeric_limits<Index>::max() - appended_equalities )
   {
      return std::unexpected(BridgeError(
         EvaluationErrorCode::invalid_layout,
         "TNLPAdapter equality dimension overflows C++23 Index"));
   }
   const Index expected_equalities = *base_equalities + appended_equalities;
   if( expected_equalities != *internal_equalities )
   {
      return std::unexpected(BridgeError(
         EvaluationErrorCode::invalid_layout,
         "TNLPAdapter equality map does not cover the internal equality space"));
   }
   EvaluationValue<std::vector<Index>> equality_positions = ExpansionPositions(
      *p_c_g, *full_constraints, *base_equalities, "P_c_g");
   if( !equality_positions )
   {
      return std::unexpected(equality_positions.error());
   }
   result.coordinate_map.equality_from_full_constraints =
      std::move(*equality_positions);

   EvaluationValue<std::vector<Index>> inequality_positions = ExpansionPositions(
      *p_d_g, *full_constraints, *internal_inequalities, "P_d_g");
   if( !inequality_positions )
   {
      return std::unexpected(inequality_positions.error());
   }
   result.coordinate_map.inequality_from_full_constraints =
      std::move(*inequality_positions);

   const ::Ipopt::Number* equality_rhs = adapter.GetC_Rhs();
   if( *internal_equalities != 0 && equality_rhs == nullptr )
   {
      return std::unexpected(BridgeError(
         EvaluationErrorCode::invalid_layout,
         "TNLPAdapter equality RHS has not been initialized"));
   }
   result.coordinate_map.equality_rhs.resize(*base_equalities);
   for( Index i = 0; i < *base_equalities; ++i )
   {
      result.coordinate_map.equality_rhs[i] =
         static_cast<Number>(equality_rhs[i]);
   }
   if( adds_fixed_equalities )
   {
      result.coordinate_map.fixed_variable_equalities.reserve(*fixed_count);
      for( Index i = 0; i < *fixed_count; ++i )
      {
         EvaluationValue<Index> fixed =
            CheckedIndex(stable_fixed_indices[i], "fixed variable index");
         if( !fixed )
         {
            return std::unexpected(fixed.error());
         }
         if( *fixed >= *full_variables )
         {
            return std::unexpected(BridgeError(
               EvaluationErrorCode::invalid_layout,
               "TNLPAdapter fixed-variable index is out of range"));
         }
         result.coordinate_map.fixed_variable_equalities.push_back({
            *fixed,
            static_cast<Number>(equality_rhs[*base_equalities + i])
         });
      }
   }

   ::Ipopt::SmartPtr<::Ipopt::NLPScalingObject> scaling = orig_nlp.NLP_scaling();
   if( !::Ipopt::IsValid(scaling) )
   {
      return std::unexpected(BridgeError(
         EvaluationErrorCode::invalid_layout,
         "OrigIpoptNLP scaling object is not initialized"));
   }
   result.scaling.objective =
      static_cast<Number>(scaling->apply_obj_scaling(1.));
   EvaluationValue<std::vector<Number>> variable_scaling =
      ExtractScaling(*scaling, x_space, ScalingKind::variables);
   if( !variable_scaling )
   {
      return std::unexpected(variable_scaling.error());
   }
   EvaluationValue<std::vector<Number>> equality_scaling =
      ExtractScaling(*scaling, c_space, ScalingKind::equalities);
   if( !equality_scaling )
   {
      return std::unexpected(equality_scaling.error());
   }
   EvaluationValue<std::vector<Number>> inequality_scaling =
      ExtractScaling(*scaling, d_space, ScalingKind::inequalities);
   if( !inequality_scaling )
   {
      return std::unexpected(inequality_scaling.error());
   }
   result.scaling.variables = std::move(*variable_scaling);
   result.scaling.equalities = std::move(*equality_scaling);
   result.scaling.inequalities = std::move(*inequality_scaling);
   return result;
}
} // namespace legacy_ipopt_detail

inline EvaluationValue<AnyNlpProblem> MakeLegacyTnlpProblem(
   ::Ipopt::SmartPtr<::Ipopt::TNLP> tnlp,
   std::uint64_t                    structural_revision = 0
)
{
   legacy_ipopt_detail::TnlpModel model(
      std::move(tnlp), structural_revision);
   if( !model.initialization_result() )
   {
      return std::unexpected(model.initialization_result().error());
   }
   return MakeNlpProblem(std::move(model));
}

inline EvaluationValue<LegacyIpoptBridgeData> ExportLegacyIpoptBridgeData(
   ::Ipopt::TNLPAdapter& adapter,
   ::Ipopt::OrigIpoptNLP& orig_nlp
)
{
   try
   {
      return legacy_ipopt_detail::ExportImpl(adapter, orig_nlp);
   }
   catch( const ::Ipopt::IpoptException& error )
   {
      return std::unexpected(legacy_ipopt_detail::BridgeError(
         EvaluationErrorCode::model_failure,
         "legacy Ipopt bridge failed: " + error.Message()));
   }
   catch( const std::bad_alloc& )
   {
      throw;
   }
   catch( const std::exception& error )
   {
      return std::unexpected(legacy_ipopt_detail::BridgeError(
         EvaluationErrorCode::model_failure,
         "legacy Ipopt bridge failed: " + std::string(error.what())));
   }
}

inline EvaluationValue<AnyNlpProblem> MakeLegacyIpoptCoordinateProblem(
   AnyNlpProblem          full_unscaled_problem,
   ::Ipopt::TNLPAdapter& adapter,
   ::Ipopt::OrigIpoptNLP& orig_nlp
)
{
   EvaluationValue<LegacyIpoptBridgeData> bridge =
      ExportLegacyIpoptBridgeData(adapter, orig_nlp);
   if( !bridge )
   {
      return std::unexpected(bridge.error());
   }
   return MakeLegacyCoordinateProblem(
      std::move(full_unscaled_problem),
      std::move(bridge->coordinate_map),
      std::move(bridge->scaling));
}

inline EvaluationValue<AnyNlpProblem> MakeLegacyIpoptCoordinateProblem(
   ::Ipopt::TNLPAdapter& adapter,
   ::Ipopt::OrigIpoptNLP& orig_nlp,
   std::uint64_t          structural_revision = 0
)
{
   EvaluationValue<AnyNlpProblem> source =
      MakeLegacyTnlpProblem(adapter.tnlp(), structural_revision);
   if( !source )
   {
      return std::unexpected(source.error());
   }
   return MakeLegacyIpoptCoordinateProblem(
      std::move(*source), adapter, orig_nlp);
}
} // namespace Ipopt::Cxx23

#endif
