// Copyright (C) 2026 Ipopt contributors.
// All Rights Reserved.
// This file is distributed under the Eclipse Public License.

#ifndef IPOPT_CXX23_LEGACY_MATRIX_SNAPSHOT_HPP
#define IPOPT_CXX23_LEGACY_MATRIX_SNAPSHOT_HPP

#include <ipopt/cxx23/any_nlp_problem.hpp>

#include <IpException.hpp>
#include <IpMatrix.hpp>
#include <IpSymMatrix.hpp>
#include <IpTripletHelper.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <exception>
#include <expected>
#include <limits>
#include <new>
#include <ranges>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace Ipopt::Cxx23
{
struct LegacyMatrixSnapshotViews
{
   const ::Ipopt::SymMatrix* hessian = nullptr;
   const ::Ipopt::Matrix* jacobian_equalities = nullptr;
   const ::Ipopt::Matrix* jacobian_inequalities = nullptr;
   std::uint64_t structural_revision = 1;
};

namespace legacy_matrix_snapshot_detail
{
struct Triplets
{
   std::vector<Index> rows;
   std::vector<Index> columns;
   std::vector<Number> values;
};

inline EvaluationValue<Index> CheckedIndex(
   ::Ipopt::Index value,
   std::string_view name
)
{
   if( value < 0 || !std::in_range<Index>(value) )
   {
      return std::unexpected(EvaluationError{
         EvaluationErrorCode::dimension_mismatch,
         std::string(name) + " is not representable by C++23 Index"
      });
   }
   return static_cast<Index>(value);
}

inline EvaluationValue<Triplets> Snapshot(
   const ::Ipopt::Matrix& matrix,
   Index                  expected_rows,
   Index                  expected_columns,
   Index                  row_offset,
   bool                   symmetric,
   std::string_view       name
)
{
   EvaluationValue<Index> rows = CheckedIndex(matrix.NRows(), name);
   EvaluationValue<Index> columns = CheckedIndex(matrix.NCols(), name);
   if( !rows )
   {
      return std::unexpected(rows.error());
   }
   if( !columns )
   {
      return std::unexpected(columns.error());
   }
   if( *rows != expected_rows || *columns != expected_columns )
   {
      return std::unexpected(EvaluationError{
         EvaluationErrorCode::dimension_mismatch,
         std::string(name) + " dimensions do not match the KKT layout"
      });
   }
   if( !matrix.HasValidNumbers() )
   {
      return std::unexpected(EvaluationError{
         EvaluationErrorCode::nonfinite_output,
         std::string(name) + " contains a nonfinite value"
      });
   }

   const ::Ipopt::Index stable_nonzeros =
      ::Ipopt::TripletHelper::GetNumberEntries(matrix);
   EvaluationValue<Index> nonzeros = CheckedIndex(stable_nonzeros, name);
   if( !nonzeros )
   {
      return std::unexpected(nonzeros.error());
   }
   std::vector<::Ipopt::Index> stable_rows(*nonzeros);
   std::vector<::Ipopt::Index> stable_columns(*nonzeros);
   std::vector<::Ipopt::Number> stable_values(*nonzeros);
   ::Ipopt::Index empty_index = 0;
   ::Ipopt::Number empty_value = 0.;
   ::Ipopt::TripletHelper::FillRowCol(
      stable_nonzeros, matrix,
      stable_rows.empty() ? &empty_index : stable_rows.data(),
      stable_columns.empty() ? &empty_index : stable_columns.data());
   ::Ipopt::TripletHelper::FillValues(
      stable_nonzeros, matrix,
      stable_values.empty() ? &empty_value : stable_values.data());

   Triplets result{
      .rows = std::vector<Index>(*nonzeros),
      .columns = std::vector<Index>(*nonzeros),
      .values = std::vector<Number>(*nonzeros)
   };
   for( Index i = 0; i < *nonzeros; ++i )
   {
      if( stable_rows[i] <= 0 || stable_columns[i] <= 0 )
      {
         return std::unexpected(EvaluationError{
            EvaluationErrorCode::invalid_sparsity,
            std::string(name) + " contains a nonpositive triplet index"
         });
      }
      EvaluationValue<Index> row = CheckedIndex(stable_rows[i] - 1, name);
      EvaluationValue<Index> column = CheckedIndex(stable_columns[i] - 1, name);
      if( !row )
      {
         return std::unexpected(row.error());
      }
      if( !column )
      {
         return std::unexpected(column.error());
      }
      if( *row >= expected_rows || *column >= expected_columns )
      {
         return std::unexpected(EvaluationError{
            EvaluationErrorCode::invalid_sparsity,
            std::string(name) + " triplet lies outside its matrix dimensions"
         });
      }
      if( row_offset > std::numeric_limits<Index>::max() - *row )
      {
         return std::unexpected(EvaluationError{
            EvaluationErrorCode::dimension_mismatch,
            std::string(name) + " row offset overflows C++23 Index"
         });
      }
      result.rows[i] = *row + row_offset;
      result.columns[i] = *column;
      if( symmetric && result.rows[i] < result.columns[i] )
      {
         std::swap(result.rows[i], result.columns[i]);
      }
      result.values[i] = static_cast<Number>(stable_values[i]);
      if( !std::isfinite(result.values[i]) )
      {
         return std::unexpected(EvaluationError{
            EvaluationErrorCode::nonfinite_output,
            std::string(name) + " contains a value not representable by C++23 Number"
         });
      }
   }
   return result;
}

class Model
{
public:
   Model(
      NlpStructure structure,
      Triplets     jacobian,
      Triplets     hessian
   )
      : structure_(structure),
        jacobian_(std::move(jacobian)),
        hessian_(std::move(hessian)),
        jacobian_product_scratch_(structure.constraints),
        variable_product_scratch_(structure.variables)
   {
   }

   NlpStructure structure() const
   {
      return structure_;
   }

   EvaluationValue<Number> eval_objective(std::span<const Number>)
   {
      return 0.;
   }

   EvaluationResult eval_gradient(
      std::span<const Number>,
      std::span<Number> result
   )
   {
      std::ranges::fill(result, 0.);
      return {};
   }

   EvaluationResult eval_constraints(
      std::span<const Number>,
      std::span<Number> result
   )
   {
      std::ranges::fill(result, 0.);
      return {};
   }

   EvaluationResult eval_jacobian_structure(
      std::span<Index> rows,
      std::span<Index> columns
   )
   {
      std::ranges::copy(jacobian_.rows, rows.begin());
      std::ranges::copy(jacobian_.columns, columns.begin());
      return {};
   }

   EvaluationResult eval_jacobian_values(
      std::span<const Number>,
      std::span<Number> values
   )
   {
      std::ranges::copy(jacobian_.values, values.begin());
      return {};
   }

   EvaluationResult eval_hessian_structure(
      std::span<Index> rows,
      std::span<Index> columns
   )
   {
      std::ranges::copy(hessian_.rows, rows.begin());
      std::ranges::copy(hessian_.columns, columns.begin());
      return {};
   }

   EvaluationResult eval_hessian_values(
      std::span<const Number>,
      Number,
      std::span<const Number>,
      std::span<Number> values
   )
   {
      std::ranges::copy(hessian_.values, values.begin());
      return {};
   }

   JacobianProductCapabilities jacobian_product_capabilities() const noexcept
   {
      return {true, true};
   }

   HessianProductCapabilities hessian_product_capabilities() const noexcept
   {
      return {true};
   }

   EvaluationResult eval_jacobian_product(
      std::span<const Number>,
      std::span<const Number> direction,
      std::span<Number>       result
   )
   {
      std::ranges::fill(jacobian_product_scratch_, 0.);
      for( Index i = 0; i < jacobian_.values.size(); ++i )
      {
         jacobian_product_scratch_[jacobian_.rows[i]] +=
            jacobian_.values[i] * direction[jacobian_.columns[i]];
      }
      return CommitFinite(jacobian_product_scratch_, result, "Jacobian product");
   }

   EvaluationResult eval_jacobian_transpose_product(
      std::span<const Number>,
      std::span<const Number> direction,
      std::span<Number>       result
   )
   {
      std::ranges::fill(variable_product_scratch_, 0.);
      for( Index i = 0; i < jacobian_.values.size(); ++i )
      {
         variable_product_scratch_[jacobian_.columns[i]] +=
            jacobian_.values[i] * direction[jacobian_.rows[i]];
      }
      return CommitFinite(variable_product_scratch_, result, "Jacobian transpose product");
   }

   EvaluationResult eval_hessian_product(
      std::span<const Number>,
      Number,
      std::span<const Number>,
      std::span<const Number> direction,
      std::span<Number>       result
   )
   {
      std::ranges::fill(variable_product_scratch_, 0.);
      for( Index i = 0; i < hessian_.values.size(); ++i )
      {
         const Index row = hessian_.rows[i];
         const Index column = hessian_.columns[i];
         variable_product_scratch_[row] +=
            hessian_.values[i] * direction[column];
         if( row != column )
         {
            variable_product_scratch_[column] +=
               hessian_.values[i] * direction[row];
         }
      }
      return CommitFinite(variable_product_scratch_, result, "Hessian product");
   }

private:
   static EvaluationResult CommitFinite(
      std::span<const Number> source,
      std::span<Number>       destination,
      std::string_view        name
   )
   {
      if( !std::ranges::all_of(
             source, [](Number value) { return std::isfinite(value); }) )
      {
         return std::unexpected(EvaluationError{
            EvaluationErrorCode::nonfinite_output,
            std::string(name) + " produced a nonfinite value"
         });
      }
      std::ranges::copy(source, destination.begin());
      return {};
   }

   const NlpStructure structure_;
   Triplets jacobian_;
   Triplets hessian_;
   std::vector<Number> jacobian_product_scratch_;
   std::vector<Number> variable_product_scratch_;
};
} // namespace legacy_matrix_snapshot_detail

/** Snapshots the already materialized stable W/J_c/J_d matrices.
 *
 * This is intended for equivalence checks of wrapped IpoptNLP formulations
 * such as restoration. Numeric values and triplet structure are copied; no
 * stable Matrix or Vector view escapes into the returned AnyNlpProblem.
 */
inline EvaluationValue<AnyNlpProblem> MakeLegacyMatrixSnapshotProblem(
   LegacyMatrixSnapshotViews views
)
{
   if( views.hessian == nullptr || views.jacobian_equalities == nullptr ||
       views.jacobian_inequalities == nullptr )
   {
      return std::unexpected(EvaluationError{
         EvaluationErrorCode::invalid_layout,
         "legacy matrix snapshot requires W, J_c, and J_d"
      });
   }
   if( views.structural_revision == 0 )
   {
      return std::unexpected(EvaluationError{
         EvaluationErrorCode::structure_mismatch,
         "legacy matrix snapshot requires a nonzero structural revision"
      });
   }

   try
   {
      EvaluationValue<Index> variables =
         legacy_matrix_snapshot_detail::CheckedIndex(
            views.hessian->Dim(), "legacy Hessian dimension");
      EvaluationValue<Index> equalities =
         legacy_matrix_snapshot_detail::CheckedIndex(
            views.jacobian_equalities->NRows(), "legacy equality count");
      EvaluationValue<Index> inequalities =
         legacy_matrix_snapshot_detail::CheckedIndex(
            views.jacobian_inequalities->NRows(), "legacy inequality count");
      if( !variables )
      {
         return std::unexpected(variables.error());
      }
      if( !equalities )
      {
         return std::unexpected(equalities.error());
      }
      if( !inequalities )
      {
         return std::unexpected(inequalities.error());
      }
      if( *inequalities > std::numeric_limits<Index>::max() - *equalities )
      {
         return std::unexpected(EvaluationError{
            EvaluationErrorCode::dimension_mismatch,
            "legacy constraint count overflows C++23 Index"
         });
      }
      const Index constraints = *equalities + *inequalities;

      EvaluationValue<legacy_matrix_snapshot_detail::Triplets> equality =
         legacy_matrix_snapshot_detail::Snapshot(
            *views.jacobian_equalities, *equalities, *variables, 0, false,
            "legacy equality Jacobian");
      EvaluationValue<legacy_matrix_snapshot_detail::Triplets> inequality =
         legacy_matrix_snapshot_detail::Snapshot(
            *views.jacobian_inequalities, *inequalities, *variables,
            *equalities, false, "legacy inequality Jacobian");
      EvaluationValue<legacy_matrix_snapshot_detail::Triplets> hessian =
         legacy_matrix_snapshot_detail::Snapshot(
            *views.hessian, *variables, *variables, 0, true,
            "legacy Hessian");
      if( !equality )
      {
         return std::unexpected(equality.error());
      }
      if( !inequality )
      {
         return std::unexpected(inequality.error());
      }
      if( !hessian )
      {
         return std::unexpected(hessian.error());
      }

      legacy_matrix_snapshot_detail::Triplets jacobian =
         std::move(*equality);
      jacobian.rows.insert(
         jacobian.rows.end(), inequality->rows.begin(), inequality->rows.end());
      jacobian.columns.insert(
         jacobian.columns.end(), inequality->columns.begin(), inequality->columns.end());
      jacobian.values.insert(
         jacobian.values.end(), inequality->values.begin(), inequality->values.end());
      const NlpStructure structure{
         .variables = *variables,
         .constraints = constraints,
         .jacobian_nonzeros = jacobian.values.size(),
         .hessian_nonzeros = hessian->values.size(),
         .revision = views.structural_revision
      };
      return MakeNlpProblem(legacy_matrix_snapshot_detail::Model(
         structure, std::move(jacobian), std::move(*hessian)));
   }
   catch( const std::bad_alloc& )
   {
      throw;
   }
   catch( const ::Ipopt::IpoptException& exception )
   {
      return std::unexpected(EvaluationError{
         EvaluationErrorCode::model_failure,
         "legacy matrix snapshot caught Ipopt exception: " + exception.Message()
      });
   }
   catch( const std::exception& exception )
   {
      return std::unexpected(EvaluationError{
         EvaluationErrorCode::model_failure,
         "legacy matrix snapshot caught exception: " + std::string(exception.what())
      });
   }
   catch( ... )
   {
      return std::unexpected(EvaluationError{
         EvaluationErrorCode::model_failure,
         "legacy matrix snapshot caught an unknown exception"
      });
   }
}
} // namespace Ipopt::Cxx23

#endif
