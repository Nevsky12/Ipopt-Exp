// Copyright (C) 2026 Ipopt contributors.
// All Rights Reserved.
// This file is distributed under the Eclipse Public License.

#ifndef IPOPT_CXX23_BORDERED_BLOCK_TRIDIAGONAL_SOLVER_HPP
#define IPOPT_CXX23_BORDERED_BLOCK_TRIDIAGONAL_SOLVER_HPP

#include <ipopt/cxx23/block_tridiagonal_solver.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace Ipopt::Cxx23
{
/** Packed storage for [A B; B^T C], where A is block tridiagonal. */
struct BorderedBlockTridiagonalStorage
{
   Index stage_dimension = 0;
   Index border_dimension = 0;
   Index dimension = 0;
   Index diagonal_values = 0;
   Index lower_values = 0;
   Index border_values = 0;
   Index border_diagonal_values = 0;
};

struct BorderedBlockTridiagonalOptions
{
   BlockTridiagonalFactorizationOptions stage_factorization;
   BlockTridiagonalFactorizationOptions schur_factorization;
   Number relative_schur_symmetry_tolerance =
      128. * std::sqrt(std::numeric_limits<Number>::epsilon());
};

struct BorderedBlockTridiagonalFactorizationReport
{
   BlockTridiagonalFactorizationReport stage;
   BlockTridiagonalFactorizationReport schur;
   /** Numeric sign count for the computed Schur complement.
    *
    * It is deliberately inexact: forming C-B^T A^-1 B in floating point
    * needs an additional propagated error certificate before Sylvester's law
    * may certify the inertia of the original arrowhead matrix. Candidate-first
    * users must provide an independent exact full-system proof.
    */
   CertifiedInertia numeric_inertia;
   Number relative_schur_asymmetry = std::numeric_limits<Number>::infinity();
};

struct BorderedBlockTridiagonalStatistics
{
   Index factorization_attempts = 0;
   Index successful_factorizations = 0;
   Index solve_calls = 0;
   Index solved_right_hand_sides = 0;
   Index fused_border_right_hand_sides = 0;
   Index refinement_steps = 0;
   Index matrix_applications = 0;
};

struct BorderedBlockTridiagonalWorkspaceProfile
{
   BlockTridiagonalWorkspaceProfile stage;
   BlockTridiagonalWorkspaceProfile schur;
   Index border = 0;
   Index border_diagonal = 0;
   Index inverse_border = 0;
   Index schur_values = 0;
   Index stage_solution = 0;
   Index border_rhs = 0;
   Index border_solution = 0;
   Index solution = 0;
   Index residual = 0;
   Index correction = 0;
   Index matrix_application_input = 0;
   Index matrix_application_output = 0;

   friend bool operator==(
      const BorderedBlockTridiagonalWorkspaceProfile&,
      const BorderedBlockTridiagonalWorkspaceProfile&) = default;
};

/** Direct solver for a symmetric block-tridiagonal matrix with a small border.
 *
 * For K=[A B;B^T C], factorization computes X=A^-1 B in one fused multi-RHS
 * traversal and factors S=C-B^T X. A solve then uses
 *
 *   y=A^-1 f, q=S^-1(g-B^T y), x=y-Xq.
 *
 * The constructor fixes every workspace size. factorize(), solve_rhs(),
 * apply(), and solve_refined_rhs() allocate no topology-sized storage.
 */
class SymmetricBorderedBlockTridiagonalSolver
{
public:
   explicit SymmetricBorderedBlockTridiagonalSolver(
      std::vector<Index>                    block_sizes,
      Index                                 border_dimension,
      BorderedBlockTridiagonalOptions       options = {}
   )
      : border_dimension_(border_dimension),
        options_(options),
        stage_solver_(
           std::move(block_sizes),
           std::max<Index>(border_dimension, 1),
           options_.stage_factorization),
        schur_solver_(
           border_dimension == 0
              ? std::vector<Index>{}
              : std::vector<Index>{border_dimension},
           1,
           options_.schur_factorization)
   {
      Configure();
   }

   SymmetricBorderedBlockTridiagonalSolver(
      const SymmetricBorderedBlockTridiagonalSolver&) = delete;
   SymmetricBorderedBlockTridiagonalSolver& operator=(
      const SymmetricBorderedBlockTridiagonalSolver&) = delete;
   SymmetricBorderedBlockTridiagonalSolver(
      SymmetricBorderedBlockTridiagonalSolver&&) = default;
   SymmetricBorderedBlockTridiagonalSolver& operator=(
      SymmetricBorderedBlockTridiagonalSolver&&) = default;

   bool configured() const noexcept
   {
      return configuration_error_.empty();
   }

   bool factorized() const noexcept
   {
      return factorized_;
   }

   const std::string& configuration_error() const noexcept
   {
      return configuration_error_;
   }

   const std::vector<Index>& block_sizes() const noexcept
   {
      return stage_solver_.block_sizes();
   }

   BorderedBlockTridiagonalStorage storage() const noexcept
   {
      return storage_;
   }

   const BorderedBlockTridiagonalFactorizationReport&
   factorization_report() const noexcept
   {
      return factorization_report_;
   }

   const BorderedBlockTridiagonalStatistics& statistics() const noexcept
   {
      return statistics_;
   }

   BorderedBlockTridiagonalWorkspaceProfile workspace_profile() const noexcept
   {
      return {
         stage_solver_.workspace_profile(),
         schur_solver_.workspace_profile(),
         border_.capacity(),
         border_diagonal_.capacity(),
         inverse_border_.capacity(),
         schur_values_.capacity(),
         stage_solution_.capacity(),
         border_rhs_.capacity(),
         border_solution_.capacity(),
         solution_work_.capacity(),
         residual_work_.capacity(),
         correction_work_.capacity(),
         matrix_application_input_.capacity(),
         matrix_application_output_.capacity()
      };
   }

   EvaluationValue<BorderedBlockTridiagonalFactorizationReport> factorize(
      std::span<const Number> diagonal_blocks,
      std::span<const Number> lower_blocks,
      std::span<const Number> border,
      std::span<const Number> border_diagonal
   )
   {
      ++statistics_.factorization_attempts;
      factorized_ = false;
      factorization_report_ = {};
      if( !configured() )
      {
         return Failure<BorderedBlockTridiagonalFactorizationReport>(
            EvaluationErrorCode::invalid_layout, configuration_error_);
      }
      if( diagonal_blocks.size() != storage_.diagonal_values ||
          lower_blocks.size() != storage_.lower_values ||
          border.size() != storage_.border_values ||
          border_diagonal.size() != storage_.border_diagonal_values )
      {
         return Failure<BorderedBlockTridiagonalFactorizationReport>(
            EvaluationErrorCode::dimension_mismatch,
            "bordered block-tridiagonal packed storage has the wrong size");
      }
      // The stage solver validates its diagonal and lower inputs while copying
      // them into owning storage. Avoid scanning those packed blocks twice.
      if( !AllFinite(border) || !AllFinite(border_diagonal) )
      {
         return Failure<BorderedBlockTridiagonalFactorizationReport>(
            EvaluationErrorCode::nonfinite_output,
            "bordered block-tridiagonal matrix contains a nonfinite coefficient");
      }
      if( EvaluationResult symmetric = ValidateBorderDiagonal(border_diagonal);
          !symmetric )
      {
         return std::unexpected(symmetric.error());
      }

      EvaluationValue<BlockTridiagonalFactorizationReport> stage =
         stage_solver_.factorize(diagonal_blocks, lower_blocks);
      if( !stage )
      {
         return std::unexpected(stage.error());
      }
      std::ranges::copy(border, border_.begin());
      std::ranges::copy(border_diagonal, border_diagonal_.begin());
      if( EvaluationResult solved = stage_solver_.solve_many(
             border_, border_dimension_, inverse_border_);
          !solved )
      {
         return std::unexpected(solved.error());
      }
      SaturatingAdd(
         statistics_.fused_border_right_hand_sides, border_dimension_);

      Number squared_asymmetry = 0.;
      Number squared_scale = 0.;
      for( Index row = 0; row < border_dimension_; ++row )
      {
         for( Index column = 0; column < border_dimension_; ++column )
         {
            Number value = border_diagonal_[row * border_dimension_ + column];
            for( Index inner = 0; inner < storage_.stage_dimension; ++inner )
            {
               value -= border_[inner * border_dimension_ + row] *
                  inverse_border_[inner * border_dimension_ + column];
            }
            schur_values_[row * border_dimension_ + column] = value;
            squared_scale += value * value;
         }
      }
      for( Index row = 0; row < border_dimension_; ++row )
      {
         for( Index column = 0; column < row; ++column )
         {
            const Number lower = schur_values_[row * border_dimension_ + column];
            const Number upper = schur_values_[column * border_dimension_ + row];
            const Number difference = lower - upper;
            squared_asymmetry += difference * difference;
            const Number average = .5 * (lower + upper);
            schur_values_[row * border_dimension_ + column] = average;
            schur_values_[column * border_dimension_ + row] = average;
         }
      }
      const Number scale = std::max(Number{1.}, std::sqrt(squared_scale));
      const Number relative_asymmetry = std::sqrt(squared_asymmetry) / scale;
      if( !std::isfinite(relative_asymmetry) ||
          relative_asymmetry > options_.relative_schur_symmetry_tolerance )
      {
         return Failure<BorderedBlockTridiagonalFactorizationReport>(
            EvaluationErrorCode::model_failure,
            "bordered Schur complement lost numerical symmetry");
      }

      EvaluationValue<BlockTridiagonalFactorizationReport> schur =
         schur_solver_.factorize(schur_values_, {});
      if( !schur )
      {
         return std::unexpected(schur.error());
      }
      factorization_report_.stage = *stage;
      factorization_report_.schur = *schur;
      factorization_report_.relative_schur_asymmetry = relative_asymmetry;
      factorization_report_.numeric_inertia = ComposeNumericInertia(
         stage->inertia, schur->inertia);
      factorized_ = true;
      ++statistics_.successful_factorizations;
      return factorization_report_;
   }

   /** Solve one full arrowhead RHS. Caller output is unchanged on failure. */
   EvaluationResult solve_rhs(
      std::span<const Number> input,
      std::span<Number>       output
   )
   {
      if( EvaluationResult valid = ValidateSolve(input, output); !valid )
      {
         return valid;
      }
      if( EvaluationResult solved = SolveImpl(input, solution_work_); !solved )
      {
         return solved;
      }
      std::ranges::copy(solution_work_, output.begin());
      ++statistics_.solve_calls;
      ++statistics_.solved_right_hand_sides;
      return {};
   }

   /** Apply the original supplied arrowhead matrix. Alias-safe. */
   EvaluationResult apply(
      std::span<const Number> input,
      std::span<Number>       output
   )
   {
      if( !factorized_ )
      {
         return Error(
            EvaluationErrorCode::model_failure,
            "bordered matrix apply requires a successful factorization");
      }
      if( input.size() != storage_.dimension ||
          output.size() != storage_.dimension )
      {
         return Error(
            EvaluationErrorCode::dimension_mismatch,
            "bordered matrix apply has the wrong vector dimension");
      }
      if( !AllFinite(input) )
      {
         return Error(
            EvaluationErrorCode::nonfinite_output,
            "bordered matrix apply input is nonfinite");
      }
      std::ranges::copy(input, matrix_application_input_.begin());
      if( EvaluationResult applied = ApplyOriginalNoAlias(
             matrix_application_input_, matrix_application_output_);
          !applied )
      {
         return applied;
      }
      std::ranges::copy(matrix_application_output_, output.begin());
      return {};
   }

   /** Solve and refine against the original arrowhead matrix. */
   EvaluationValue<BlockRefinementReport> solve_refined_rhs(
      std::span<const Number> input,
      std::span<Number>       output,
      BlockRefinementOptions  options = {}
   )
   {
      if( !std::isfinite(options.relative_tolerance) ||
          options.relative_tolerance < 0. )
      {
         return Failure<BlockRefinementReport>(
            EvaluationErrorCode::invalid_layout,
            "bordered refinement tolerance must be finite and nonnegative");
      }
      if( EvaluationResult valid = ValidateSolve(input, output); !valid )
      {
         return std::unexpected(valid.error());
      }
      if( EvaluationResult solved = SolveImpl(input, solution_work_); !solved )
      {
         return std::unexpected(solved.error());
      }

      const Number rhs_norm = EuclideanNorm(input);
      const Number denominator = std::max(Number{1.}, rhs_norm);
      Number previous_relative = std::numeric_limits<Number>::infinity();
      BlockRefinementReport report;
      for( Index step = 0;; ++step )
      {
         if( EvaluationResult applied = ApplyOriginalNoAlias(
                solution_work_, residual_work_);
             !applied )
         {
            return std::unexpected(applied.error());
         }
         for( Index row = 0; row < storage_.dimension; ++row )
         {
            residual_work_[row] = input[row] - residual_work_[row];
         }
         const Number relative = EuclideanNorm(residual_work_) / denominator;
         if( !std::isfinite(relative) )
         {
            return Failure<BlockRefinementReport>(
               EvaluationErrorCode::nonfinite_output,
               "bordered refinement residual is nonfinite");
         }
         if( step == 0 )
         {
            report.initial_relative_residual = relative;
         }
         report.final_relative_residual = relative;
         report.steps = step;
         if( relative <= options.relative_tolerance )
         {
            report.status = BlockRefinementStatus::converged;
            std::ranges::copy(solution_work_, output.begin());
            ++statistics_.solve_calls;
            ++statistics_.solved_right_hand_sides;
            return report;
         }
         if( step >= options.maximum_steps )
         {
            report.status = BlockRefinementStatus::iteration_limit;
            return report;
         }
         if( relative >= previous_relative )
         {
            report.status = BlockRefinementStatus::residual_increase;
            return report;
         }
         previous_relative = relative;
         if( EvaluationResult corrected = SolveImpl(
                residual_work_, correction_work_);
             !corrected )
         {
            return std::unexpected(corrected.error());
         }
         for( Index row = 0; row < storage_.dimension; ++row )
         {
            solution_work_[row] += correction_work_[row];
         }
         ++statistics_.refinement_steps;
      }
   }

private:
   template <class Value>
   static EvaluationValue<Value> Failure(
      EvaluationErrorCode code,
      std::string         message
   )
   {
      return std::unexpected(EvaluationError{code, std::move(message)});
   }

   static EvaluationResult Error(
      EvaluationErrorCode code,
      std::string         message
   )
   {
      return std::unexpected(EvaluationError{code, std::move(message)});
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

   static bool AllFinite(std::span<const Number> values) noexcept
   {
      return std::ranges::all_of(
         values, [](Number value) { return std::isfinite(value); });
   }

   static Number EuclideanNorm(std::span<const Number> values) noexcept
   {
      Number scale = 0.;
      Number sum = 1.;
      for( Number value : values )
      {
         const Number magnitude = std::abs(value);
         if( magnitude == 0. )
         {
            continue;
         }
         if( scale < magnitude )
         {
            const Number ratio = scale / magnitude;
            sum = 1. + sum * ratio * ratio;
            scale = magnitude;
         }
         else
         {
            const Number ratio = magnitude / scale;
            sum += ratio * ratio;
         }
      }
      return scale == 0. ? 0. : scale * std::sqrt(sum);
   }

   static void SaturatingAdd(Index& target, Index increment) noexcept
   {
      target = increment > std::numeric_limits<Index>::max() - target
         ? std::numeric_limits<Index>::max()
         : target + increment;
   }

   EvaluationResult ApplyOriginalNoAlias(
      std::span<const Number> input,
      std::span<Number>       output
   )
   {
      const std::span<const Number> stage_input(
         input.data(), storage_.stage_dimension);
      std::span<Number> stage_output(
         output.data(), storage_.stage_dimension);
      if( EvaluationResult applied = stage_solver_.ApplyOriginalNoAlias(
             stage_input, stage_output);
          !applied )
      {
         return applied;
      }
      const Number* border_input = input.data() + storage_.stage_dimension;
      Number* border_output = output.data() + storage_.stage_dimension;
      std::ranges::fill_n(border_output, border_dimension_, 0.);
      for( Index stage = 0; stage < storage_.stage_dimension; ++stage )
      {
         for( Index border = 0; border < border_dimension_; ++border )
         {
            const Number value = border_[stage * border_dimension_ + border];
            stage_output[stage] += value * border_input[border];
            border_output[border] += value * stage_input[stage];
         }
      }
      for( Index row = 0; row < border_dimension_; ++row )
      {
         for( Index column = 0; column < border_dimension_; ++column )
         {
            border_output[row] +=
               border_diagonal_[row * border_dimension_ + column] *
               border_input[column];
         }
      }
      if( !AllFinite(output) )
      {
         return Error(
            EvaluationErrorCode::nonfinite_output,
            "bordered matrix apply produced a nonfinite value");
      }
      ++statistics_.matrix_applications;
      return {};
   }

   void Configure()
   {
      if( border_dimension_ == 0 )
      {
         configuration_error_ =
            "bordered block-tridiagonal solver requires a nonzero border";
         return;
      }
      if( !stage_solver_.configured() || !schur_solver_.configured() )
      {
         configuration_error_ =
            "bordered block-tridiagonal topology is invalid";
         return;
      }
      if( !std::isfinite(options_.relative_schur_symmetry_tolerance) ||
          options_.relative_schur_symmetry_tolerance <
             32. * std::numeric_limits<Number>::epsilon() )
      {
         configuration_error_ =
            "bordered Schur symmetry tolerance is invalid";
         return;
      }
      const BlockTridiagonalStorage stage = stage_solver_.storage();
      Index dimension = 0;
      Index border_values = 0;
      Index border_diagonal_values = 0;
      if( !CheckedAdd(stage.dimension, border_dimension_, dimension) ||
          !CheckedMultiply(
             stage.dimension, border_dimension_, border_values) ||
          !CheckedMultiply(
             border_dimension_, border_dimension_, border_diagonal_values) )
      {
         configuration_error_ =
            "bordered block-tridiagonal storage overflows Index";
         return;
      }
      storage_ = {
         stage.dimension,
         border_dimension_,
         dimension,
         stage.diagonal_values,
         stage.lower_values,
         border_values,
         border_diagonal_values
      };
      border_.resize(border_values);
      border_diagonal_.resize(border_diagonal_values);
      inverse_border_.resize(border_values);
      schur_values_.resize(border_diagonal_values);
      stage_solution_.resize(stage.dimension);
      border_rhs_.resize(border_dimension_);
      border_solution_.resize(border_dimension_);
      solution_work_.resize(dimension);
      residual_work_.resize(dimension);
      correction_work_.resize(dimension);
      matrix_application_input_.resize(dimension);
      matrix_application_output_.resize(dimension);
   }

   EvaluationResult ValidateBorderDiagonal(
      std::span<const Number> values
   ) const
   {
      for( Index row = 0; row < border_dimension_; ++row )
      {
         for( Index column = 0; column < row; ++column )
         {
            const Number lower = values[row * border_dimension_ + column];
            const Number upper = values[column * border_dimension_ + row];
            const Number scale = std::max(
               {Number{1.}, std::abs(lower), std::abs(upper)});
            if( std::abs(lower - upper) >
                options_.stage_factorization.symmetry_tolerance * scale )
            {
               return Error(
                  EvaluationErrorCode::invalid_layout,
                  "border diagonal block is not symmetric");
            }
         }
      }
      return {};
   }

   EvaluationResult ValidateSolve(
      std::span<const Number> input,
      std::span<Number>       output
   ) const
   {
      if( !factorized_ )
      {
         return Error(
            EvaluationErrorCode::model_failure,
            "bordered solve requires a successful factorization");
      }
      if( input.size() != storage_.dimension ||
          output.size() != storage_.dimension )
      {
         return Error(
            EvaluationErrorCode::dimension_mismatch,
            "bordered solve has the wrong vector dimension");
      }
      if( !AllFinite(input) )
      {
         return Error(
            EvaluationErrorCode::nonfinite_output,
            "bordered solve input is nonfinite");
      }
      return {};
   }

   EvaluationResult SolveImpl(
      std::span<const Number> input,
      std::span<Number>       output
   )
   {
      const std::span<const Number> input_stage(
         input.data(), storage_.stage_dimension);
      const std::span<const Number> input_border(
         input.data() + storage_.stage_dimension, border_dimension_);
      if( EvaluationResult solved = stage_solver_.solve_rhs(
             input_stage, stage_solution_);
          !solved )
      {
         return solved;
      }
      for( Index border = 0; border < border_dimension_; ++border )
      {
         Number value = input_border[border];
         for( Index stage = 0; stage < storage_.stage_dimension; ++stage )
         {
            value -= border_[stage * border_dimension_ + border] *
               stage_solution_[stage];
         }
         border_rhs_[border] = value;
      }
      if( EvaluationResult solved = schur_solver_.solve_rhs(
             border_rhs_, border_solution_);
          !solved )
      {
         return solved;
      }
      for( Index stage = 0; stage < storage_.stage_dimension; ++stage )
      {
         Number value = stage_solution_[stage];
         for( Index border = 0; border < border_dimension_; ++border )
         {
            value -= inverse_border_[stage * border_dimension_ + border] *
               border_solution_[border];
         }
         output[stage] = value;
      }
      std::ranges::copy(
         border_solution_, output.begin() + storage_.stage_dimension);
      if( !AllFinite(output) )
      {
         return Error(
            EvaluationErrorCode::nonfinite_output,
            "bordered solve produced a nonfinite value");
      }
      return {};
   }

   static CertifiedInertia ComposeNumericInertia(
      const CertifiedInertia& stage,
      const CertifiedInertia& schur
   ) noexcept
   {
      CertifiedInertia result;
      result.positive_eigenvalues = stage.positive_eigenvalues +
         schur.positive_eigenvalues;
      result.negative_eigenvalues = stage.negative_eigenvalues +
         schur.negative_eigenvalues;
      result.zero_eigenvalues = stage.zero_eigenvalues +
         schur.zero_eigenvalues;
      result.certificate_radius = std::numeric_limits<Number>::infinity();
      result.minimum_separation = 0.;
      result.exact = false;
      return result;
   }

   Index border_dimension_ = 0;
   BorderedBlockTridiagonalOptions options_;
   SymmetricBlockTridiagonalSolver stage_solver_;
   SymmetricBlockTridiagonalSolver schur_solver_;
   BorderedBlockTridiagonalStorage storage_;
   BorderedBlockTridiagonalFactorizationReport factorization_report_;
   BorderedBlockTridiagonalStatistics statistics_;
   std::vector<Number> border_;
   std::vector<Number> border_diagonal_;
   std::vector<Number> inverse_border_;
   std::vector<Number> schur_values_;
   std::vector<Number> stage_solution_;
   std::vector<Number> border_rhs_;
   std::vector<Number> border_solution_;
   std::vector<Number> solution_work_;
   std::vector<Number> residual_work_;
   std::vector<Number> correction_work_;
   std::vector<Number> matrix_application_input_;
   std::vector<Number> matrix_application_output_;
   std::string configuration_error_;
   bool factorized_ = false;
};
} // namespace Ipopt::Cxx23

#endif
