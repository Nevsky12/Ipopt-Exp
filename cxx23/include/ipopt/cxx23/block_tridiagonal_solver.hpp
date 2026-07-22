// Copyright (C) 2026 Ipopt contributors.
// All Rights Reserved.
// This file is distributed under the Eclipse Public License.

#ifndef IPOPT_CXX23_BLOCK_TRIDIAGONAL_SOLVER_HPP
#define IPOPT_CXX23_BLOCK_TRIDIAGONAL_SOLVER_HPP

#include <ipopt/cxx23/any_nlp_problem.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace Ipopt::Cxx23
{
class SymmetricBorderedBlockTridiagonalSolver;

/** Packed storage for a symmetric block-tridiagonal matrix.
 *
 * Every diagonal block is stored as a complete row-major square matrix. For
 * stage k > 0, the corresponding lower block is the row-major matrix
 * A(k,k-1), with block_sizes[k] rows and block_sizes[k-1] columns.
 */
struct BlockTridiagonalStorage
{
   Index dimension = 0;
   Index diagonal_values = 0;
   Index lower_values = 0;
};

struct BlockTridiagonalFactorizationOptions
{
   Number symmetry_tolerance = 256. * std::numeric_limits<Number>::epsilon();
   Number jacobi_tolerance = 32. * std::numeric_limits<Number>::epsilon();
   Number relative_inertia_tolerance =
      1024. * std::numeric_limits<Number>::epsilon();
   Number absolute_inertia_tolerance = 0.;
   Number relative_numeric_pivot_tolerance =
      1024. * std::numeric_limits<Number>::epsilon();
   Index maximum_jacobi_sweeps = 64;
   /** Permit a numerical factor only when the assembler supplies an
    * independent exact full-KKT inertia proof. Local inertia remains inexact. */
   bool require_certified_inertia = true;
};

struct CertifiedInertia
{
   Index positive_eigenvalues = 0;
   Index negative_eigenvalues = 0;
   Index zero_eigenvalues = 0;
   Number certificate_radius = 0.;
   Number minimum_separation = 0.;
   bool exact = false;
};

struct BlockTridiagonalFactorizationReport
{
   CertifiedInertia inertia;
   Index jacobi_sweeps = 0;
};

struct BlockTridiagonalStatistics
{
   Index factorization_attempts = 0;
   Index successful_factorizations = 0;
   Index solve_calls = 0;
   Index solved_right_hand_sides = 0;
   Index refinement_steps = 0;
   Index matrix_applications = 0;
};

struct BlockTridiagonalWorkspaceProfile
{
   Index original_diagonal = 0;
   Index original_lower = 0;
   Index orthogonal_blocks = 0;
   Index eigenvalues = 0;
   Index multipliers = 0;
   Index inverse_blocks = 0;
   Index matrix_work = 0;
   Index stage_matrix_work = 0;
   Index stage_rhs_work = 0;
   Index solution_work = 0;
   Index residual_work = 0;
   Index correction_work = 0;
   Index matrix_application_work = 0;

   friend bool operator==(
      const BlockTridiagonalWorkspaceProfile&,
      const BlockTridiagonalWorkspaceProfile&) = default;
};

enum class BlockRefinementStatus
{
   converged,
   iteration_limit,
   residual_increase
};

struct BlockRefinementOptions
{
   Number relative_tolerance = 1e-10;
   Index maximum_steps = 5;
};

struct BlockRefinementReport
{
   BlockRefinementStatus status = BlockRefinementStatus::iteration_limit;
   Index steps = 0;
   Number initial_relative_residual = std::numeric_limits<Number>::infinity();
   Number final_relative_residual = std::numeric_limits<Number>::infinity();
};

/** Variable-size symmetric block-LDL^T solver.
 *
 * The constructor fixes both the stage topology and maximum simultaneous RHS
 * count, so factorize(), solve_rhs(), solve_many(), and refinement perform no
 * allocation. By default, Schur blocks are diagonalized by a symmetric
 * cyclic-Jacobi factorization. A factorization succeeds only when every
 * computed eigenvalue interval excludes zero. Sylvester's law of inertia then
 * makes the sum of the Schur-block signs an exact inertia certificate for the
 * supplied matrix. The explicitly uncertified mode instead uses a faster
 * partial-pivot inverse and reports inexact local inertia; an independent
 * full-system proof and the true-residual gate must promote such a candidate.
 *
 * This is a generic block-elimination kernel, not FATROP's OCP null-space
 * Riccati recursion. An OCP-specific assembler may use it when its ordering
 * produces a block-tridiagonal full or condensed KKT system.
 */
class SymmetricBlockTridiagonalSolver
{
public:
   explicit SymmetricBlockTridiagonalSolver(
      std::vector<Index> block_sizes,
      Index              maximum_right_hand_sides = 1,
      BlockTridiagonalFactorizationOptions options = {}
   )
      : block_sizes_(std::move(block_sizes)),
        maximum_right_hand_sides_(maximum_right_hand_sides),
        options_(options)
   {
      ConfigureStorage();
   }

   SymmetricBlockTridiagonalSolver(
      const SymmetricBlockTridiagonalSolver&
   ) = delete;
   SymmetricBlockTridiagonalSolver& operator=(
      const SymmetricBlockTridiagonalSolver&
   ) = delete;
   SymmetricBlockTridiagonalSolver(
      SymmetricBlockTridiagonalSolver&&
   ) = default;
   SymmetricBlockTridiagonalSolver& operator=(
      SymmetricBlockTridiagonalSolver&&
   ) = default;

   const std::vector<Index>& block_sizes() const noexcept
   {
      return block_sizes_;
   }

   BlockTridiagonalStorage storage() const noexcept
   {
      return storage_;
   }

   Index maximum_right_hand_sides() const noexcept
   {
      return maximum_right_hand_sides_;
   }

   bool configured() const noexcept
   {
      return configuration_error_.empty();
   }

   bool factorized() const noexcept
   {
      return factorized_;
   }

   const BlockTridiagonalFactorizationReport& factorization_report() const noexcept
   {
      return factorization_report_;
   }

   const BlockTridiagonalStatistics& statistics() const noexcept
   {
      return statistics_;
   }

   BlockTridiagonalWorkspaceProfile workspace_profile() const noexcept
   {
      return {
         original_diagonal_.capacity(),
         original_lower_.capacity(),
         orthogonal_blocks_.capacity(),
         eigenvalues_.capacity(),
         multipliers_.capacity(),
         inverse_blocks_.capacity(),
         matrix_work_.capacity(),
         stage_matrix_original_work_.capacity(),
         stage_rhs_work_.capacity(),
         solution_work_.capacity(),
         residual_work_.capacity(),
         correction_work_.capacity(),
         matrix_application_work_.capacity()
      };
   }

   /** Factorize packed diagonal and lower blocks.
    *
    * The input is copied. No caller-owned view is retained. A failed attempt
    * invalidates any previous factors.
    */
   EvaluationValue<BlockTridiagonalFactorizationReport> factorize(
      std::span<const Number> diagonal_blocks,
      std::span<const Number> lower_blocks
   )
   {
      ++statistics_.factorization_attempts;
      factorized_ = false;
      factorization_report_ = {};
      if( !configured() )
      {
         return Failure(EvaluationErrorCode::invalid_layout, configuration_error_);
      }
      if( diagonal_blocks.size() != storage_.diagonal_values )
      {
         return Failure(
            EvaluationErrorCode::dimension_mismatch,
            "packed diagonal block storage has size " +
               std::to_string(diagonal_blocks.size()) + ", expected " +
               std::to_string(storage_.diagonal_values));
      }
      if( lower_blocks.size() != storage_.lower_values )
      {
         return Failure(
            EvaluationErrorCode::dimension_mismatch,
            "packed lower block storage has size " +
               std::to_string(lower_blocks.size()) + ", expected " +
               std::to_string(storage_.lower_values));
      }
      if( EvaluationResult symmetry = ValidateAndCopyDiagonal(diagonal_blocks); !symmetry )
      {
         return std::unexpected(symmetry.error());
      }
      if( EvaluationResult finite = ValidateAndCopyLower(lower_blocks); !finite )
      {
         return std::unexpected(finite.error());
      }

      CertifiedInertia inertia;
      inertia.minimum_separation = std::numeric_limits<Number>::infinity();
      Index total_sweeps = 0;
      Number inherited_schur_error = 0.;
      for( Index stage = 0; stage < block_sizes_.size(); ++stage )
      {
         const Index current_size = block_sizes_[stage];
         const Index diagonal_offset = diagonal_offsets_[stage];
         std::ranges::copy(
            std::span<const Number>(
               original_diagonal_.data() + diagonal_offset,
               current_size * current_size),
            matrix_work_.begin());

         if( stage > 0 )
         {
            const Index previous_size = block_sizes_[stage - 1];
            const Index lower_offset = lower_offsets_[stage - 1];
            FormMultiplier(
               lower_offset, current_size, previous_size,
               multiplier_offsets_[stage - 1],
               diagonal_offsets_[stage - 1]);
            ApplySchurComplement(
               lower_offset, multiplier_offsets_[stage - 1],
               current_size, previous_size);
            if( options_.require_certified_inertia )
            {
               const Number lower_norm = FrobeniusNorm(
                  std::span<const Number>(
                     original_lower_.data() + lower_offset,
                     current_size * previous_size));
               const Number inverse_norm = FrobeniusNorm(
                  std::span<const Number>(
                     inverse_blocks_.data() + diagonal_offsets_[stage - 1],
                     previous_size * previous_size));
               const Number diagonal_norm = FrobeniusNorm(
                  std::span<const Number>(
                     original_diagonal_.data() + diagonal_offset,
                     current_size * current_size));
               inherited_schur_error = lower_norm * lower_norm *
                  inverse_error_bounds_[stage - 1] +
                  options_.relative_inertia_tolerance *
                     (diagonal_norm + lower_norm * lower_norm * inverse_norm);
            }
            else
            {
               inherited_schur_error = 0.;
            }
         }

         if( options_.require_certified_inertia )
         {
            // The certified inverse carries an explicit error bound, so keep
            // its two-sided update and round both orientations together.
            SymmetrizeWork(current_size);
         }
         EvaluationValue<StageFactorizationReport> stage_report =
            options_.require_certified_inertia
            ? FactorizeStage(stage, current_size, inherited_schur_error)
            : FactorizeStageNumerically(stage, current_size);
         if( !stage_report )
         {
            return std::unexpected(stage_report.error());
         }
         inertia.positive_eigenvalues += stage_report->positive_eigenvalues;
         inertia.negative_eigenvalues += stage_report->negative_eigenvalues;
         inertia.certificate_radius =
            std::max(inertia.certificate_radius, stage_report->certificate_radius);
         inertia.minimum_separation =
            std::min(inertia.minimum_separation, stage_report->minimum_separation);
         total_sweeps += stage_report->sweeps;

         if( stage + 1 < block_sizes_.size() &&
             options_.require_certified_inertia )
         {
            FormInverse(stage, current_size);
            EvaluationValue<Number> inverse_error = CertifyInverse(
               stage, current_size, inherited_schur_error);
            if( !inverse_error )
            {
               return std::unexpected(inverse_error.error());
            }
            inverse_error_bounds_[stage] = *inverse_error;
         }
      }

      inertia.zero_eigenvalues = 0;
      inertia.exact = options_.require_certified_inertia;
      if( !inertia.exact )
      {
         inertia.certificate_radius =
            std::numeric_limits<Number>::infinity();
         inertia.minimum_separation = 0.;
      }
      factorization_report_.inertia = inertia;
      factorization_report_.jacobi_sweeps = total_sweeps;
      factorized_ = true;
      ++statistics_.successful_factorizations;
      return factorization_report_;
   }

   /** Solve one RHS. Caller output is unchanged on failure. */
   EvaluationResult solve_rhs(
      std::span<const Number> input,
      std::span<Number>       output
   )
   {
      return SolveMany(input, 1, output);
   }

   /** Solve a row-major multivector with layout value[row * rhs_count + rhs].
    *
    * All RHS share one forward/backward stage traversal. This API avoids the
    * sequential horizon sweeps that made naive multi-RHS border handling
    * expensive in the FATROP experiments.
    */
   EvaluationResult solve_many(
      std::span<const Number> input,
      Index                   rhs_count,
      std::span<Number>       output
   )
   {
      return SolveMany(input, rhs_count, output);
   }

   /** Apply the last successfully factorized original matrix. */
   EvaluationResult apply(
      std::span<const Number> input,
      std::span<Number>       output
   )
   {
      if( !factorized_ )
      {
         return Error(
            EvaluationErrorCode::model_failure,
            "block-tridiagonal apply requested before factorization");
      }
      if( input.size() != storage_.dimension )
      {
         return detail::DimensionMismatch(
            "block-tridiagonal apply input", input.size(), storage_.dimension);
      }
      if( output.size() != storage_.dimension )
      {
         return detail::DimensionMismatch(
            "block-tridiagonal apply output", output.size(), storage_.dimension);
      }
      if( EvaluationResult applied = ApplyOriginalNoAlias(
             input, matrix_application_work_);
          !applied )
      {
         return applied;
      }
      std::ranges::copy(matrix_application_work_, output.begin());
      return {};
   }

   /** Direct solve followed by true-residual iterative refinement.
    *
    * A nonconverged refinement report is returned as a value, but output is
    * committed only for status=converged. This makes the report usable as a
    * candidate-first acceptance gate without leaking a rejected direction.
    */
   EvaluationValue<BlockRefinementReport> solve_refined_rhs(
      std::span<const Number> input,
      std::span<Number>       output,
      BlockRefinementOptions  refinement = {}
   )
   {
      if( !std::isfinite(refinement.relative_tolerance) ||
          refinement.relative_tolerance < 0. )
      {
         return Failure<BlockRefinementReport>(
            EvaluationErrorCode::invalid_layout,
            "refinement tolerance must be finite and nonnegative");
      }
      if( EvaluationResult dimensions = ValidateSolve(input, 1, output); !dimensions )
      {
         return std::unexpected(dimensions.error());
      }

      SolveManyImpl(input, 1, solution_work_);
      if( !AllFinite(std::span<const Number>(solution_work_.data(), storage_.dimension)) )
      {
         return Failure<BlockRefinementReport>(
            EvaluationErrorCode::nonfinite_output,
            "block-tridiagonal direct solve produced a nonfinite value");
      }
      ++statistics_.solve_calls;
      ++statistics_.solved_right_hand_sides;

      BlockRefinementReport report;
      const Number denominator = std::max(1., InfinityNorm(input));
      Number residual = ComputeResidual(input, solution_work_) / denominator;
      report.initial_relative_residual = residual;
      report.final_relative_residual = residual;
      if( !std::isfinite(residual) )
      {
         return Failure<BlockRefinementReport>(
            EvaluationErrorCode::nonfinite_output,
            "block-tridiagonal refinement produced a nonfinite residual");
      }
      if( residual <= refinement.relative_tolerance )
      {
         report.status = BlockRefinementStatus::converged;
         std::ranges::copy_n(solution_work_.begin(), storage_.dimension, output.begin());
         return report;
      }

      Number previous_residual = residual;
      for( Index step = 0; step < refinement.maximum_steps; ++step )
      {
         SolveManyImpl(residual_work_, 1, correction_work_);
         for( Index row = 0; row < storage_.dimension; ++row )
         {
            solution_work_[row] += correction_work_[row];
         }
         if( !AllFinite(std::span<const Number>(solution_work_.data(), storage_.dimension)) )
         {
            return Failure<BlockRefinementReport>(
               EvaluationErrorCode::nonfinite_output,
               "block-tridiagonal refinement produced a nonfinite solution");
         }
         residual = ComputeResidual(input, solution_work_) / denominator;
         ++statistics_.solve_calls;
         ++statistics_.solved_right_hand_sides;
         ++statistics_.refinement_steps;
         report.steps = step + 1;
         report.final_relative_residual = residual;
         if( !std::isfinite(residual) )
         {
            return Failure<BlockRefinementReport>(
               EvaluationErrorCode::nonfinite_output,
               "block-tridiagonal refinement produced a nonfinite residual");
         }
         if( residual <= refinement.relative_tolerance )
         {
            report.status = BlockRefinementStatus::converged;
            std::ranges::copy_n(solution_work_.begin(), storage_.dimension, output.begin());
            return report;
         }
         if( residual >= previous_residual )
         {
            report.status = BlockRefinementStatus::residual_increase;
            return report;
         }
         previous_residual = residual;
      }
      report.status = BlockRefinementStatus::iteration_limit;
      return report;
   }

private:
   friend class SymmetricBorderedBlockTridiagonalSolver;

   struct StageFactorizationReport
   {
      Index positive_eigenvalues = 0;
      Index negative_eigenvalues = 0;
      Number certificate_radius = 0.;
      Number minimum_separation = 0.;
      Index sweeps = 0;
   };

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

   void ConfigureStorage()
   {
      if( block_sizes_.empty() )
      {
         configuration_error_ = "block-tridiagonal topology must contain a stage";
         return;
      }
      if( maximum_right_hand_sides_ == 0 )
      {
         configuration_error_ = "maximum RHS count must be positive";
         return;
      }
      if( !ValidOptions() )
      {
         configuration_error_ = "block-tridiagonal factorization options are invalid";
         return;
      }

      block_offsets_.reserve(block_sizes_.size() + 1);
      diagonal_offsets_.reserve(block_sizes_.size());
      lower_offsets_.reserve(block_sizes_.size() - 1);
      multiplier_offsets_.reserve(block_sizes_.size() - 1);
      block_offsets_.push_back(0);
      Index dimension = 0;
      Index diagonal_count = 0;
      Index lower_count = 0;
      Index maximum_block_size = 0;
      for( Index stage = 0; stage < block_sizes_.size(); ++stage )
      {
         const Index size = block_sizes_[stage];
         if( size == 0 )
         {
            configuration_error_ = "block-tridiagonal stage sizes must be positive";
            return;
         }
         maximum_block_size = std::max(maximum_block_size, size);
         diagonal_offsets_.push_back(diagonal_count);
         Index square = 0;
         if( !CheckedMultiply(size, size, square) ||
             !CheckedAdd(diagonal_count, square, diagonal_count) ||
             !CheckedAdd(dimension, size, dimension) )
         {
            configuration_error_ = "block-tridiagonal topology overflows Index";
            return;
         }
         block_offsets_.push_back(dimension);
         if( stage > 0 )
         {
            lower_offsets_.push_back(lower_count);
            multiplier_offsets_.push_back(lower_count);
            Index rectangle = 0;
            if( !CheckedMultiply(size, block_sizes_[stage - 1], rectangle) ||
                !CheckedAdd(lower_count, rectangle, lower_count) )
            {
               configuration_error_ = "block-tridiagonal topology overflows Index";
               return;
            }
         }
      }

      Index solve_storage = 0;
      Index stage_rhs_storage = 0;
      Index square_workspace = 0;
      if( !CheckedMultiply(dimension, maximum_right_hand_sides_, solve_storage) ||
          !CheckedMultiply(
             maximum_block_size, maximum_right_hand_sides_, stage_rhs_storage) ||
          !CheckedMultiply(maximum_block_size, maximum_block_size, square_workspace) )
      {
         configuration_error_ = "block-tridiagonal workspace overflows Index";
         return;
      }

      storage_ = {dimension, diagonal_count, lower_count};
      original_diagonal_.resize(diagonal_count);
      original_lower_.resize(lower_count);
      if( options_.require_certified_inertia )
      {
         orthogonal_blocks_.resize(diagonal_count);
         eigenvalues_.resize(dimension);
      }
      multipliers_.resize(lower_count);
      matrix_work_.resize(square_workspace);
      stage_matrix_original_work_.resize(square_workspace);
      inverse_blocks_.resize(diagonal_count);
      inverse_error_bounds_.resize(block_sizes_.size());
      stage_rhs_work_.resize(stage_rhs_storage);
      solution_work_.resize(solve_storage);
      residual_work_.resize(dimension);
      correction_work_.resize(dimension);
      matrix_application_work_.resize(dimension);
   }

   bool ValidOptions() const noexcept
   {
      constexpr Number minimum_roundoff_margin =
         64. * std::numeric_limits<Number>::epsilon();
      return std::isfinite(options_.symmetry_tolerance) &&
         options_.symmetry_tolerance >= 0. &&
         std::isfinite(options_.jacobi_tolerance) &&
         options_.jacobi_tolerance >= 0. &&
         std::isfinite(options_.relative_inertia_tolerance) &&
         options_.relative_inertia_tolerance >= minimum_roundoff_margin &&
         std::isfinite(options_.absolute_inertia_tolerance) &&
         options_.absolute_inertia_tolerance >= 0. &&
         std::isfinite(options_.relative_numeric_pivot_tolerance) &&
         options_.relative_numeric_pivot_tolerance >=
            minimum_roundoff_margin &&
         options_.maximum_jacobi_sweeps > 0;
   }

   static bool AllFinite(std::span<const Number> values) noexcept
   {
      return std::ranges::all_of(
         values, [](Number value) { return std::isfinite(value); });
   }

   static EvaluationResult Error(EvaluationErrorCode code, std::string message)
   {
      return std::unexpected(EvaluationError{code, std::move(message)});
   }

   template <class Value = BlockTridiagonalFactorizationReport>
   static EvaluationValue<Value> Failure(EvaluationErrorCode code, std::string message)
   {
      return std::unexpected(EvaluationError{code, std::move(message)});
   }

   EvaluationResult ValidateAndCopyDiagonal(std::span<const Number> diagonal_blocks)
   {
      for( Index stage = 0; stage < block_sizes_.size(); ++stage )
      {
         const Index size = block_sizes_[stage];
         const Index offset = diagonal_offsets_[stage];
         for( Index row = 0; row < size; ++row )
         {
            const Index diagonal = offset + row * size + row;
            const Number diagonal_value = diagonal_blocks[diagonal];
            if( !std::isfinite(diagonal_value) )
            {
               return Error(
                  EvaluationErrorCode::nonfinite_output,
                  "block-tridiagonal matrix contains a nonfinite coefficient");
            }
            original_diagonal_[diagonal] = diagonal_value;
            for( Index column = 0; column < row; ++column )
            {
               const Number forward = diagonal_blocks[offset + row * size + column];
               const Number transpose = diagonal_blocks[offset + column * size + row];
               if( !std::isfinite(forward) || !std::isfinite(transpose) )
               {
                  return Error(
                     EvaluationErrorCode::nonfinite_output,
                     "block-tridiagonal matrix contains a nonfinite coefficient");
               }
               const Number scale = std::max({1., std::abs(forward), std::abs(transpose)});
               if( std::abs(forward - transpose) > options_.symmetry_tolerance * scale )
               {
                  return Error(
                     EvaluationErrorCode::invalid_layout,
                     "diagonal stage block " + std::to_string(stage) +
                        " is not symmetric within tolerance at (" +
                        std::to_string(row) + "," +
                        std::to_string(column) + "): a_ij=" +
                        std::to_string(forward) + " a_ji=" +
                        std::to_string(transpose));
               }
               // Halving before the addition keeps the symmetric projection
               // finite for same-sign inputs near Number::max().
               const Number average = 0.5 * forward + 0.5 * transpose;
               original_diagonal_[offset + row * size + column] = average;
               original_diagonal_[offset + column * size + row] = average;
            }
         }
      }
      return {};
   }

   EvaluationResult ValidateAndCopyLower(std::span<const Number> lower_blocks)
   {
      for( Index entry = 0; entry < lower_blocks.size(); ++entry )
      {
         const Number value = lower_blocks[entry];
         if( !std::isfinite(value) )
         {
            return Error(
               EvaluationErrorCode::nonfinite_output,
               "block-tridiagonal matrix contains a nonfinite coefficient");
         }
         original_lower_[entry] = value;
      }
      return {};
   }

   void FormInverse(Index stage, Index size)
   {
      const Index diagonal_offset = diagonal_offsets_[stage];
      const Index eigen_offset = block_offsets_[stage];
      Number* inverse = inverse_blocks_.data() + diagonal_offset;
      for( Index row = 0; row < size; ++row )
      {
         for( Index column = 0; column < size; ++column )
         {
            Number value = 0.;
            for( Index eigen = 0; eigen < size; ++eigen )
            {
               value += orthogonal_blocks_[diagonal_offset + row * size + eigen] *
                  orthogonal_blocks_[diagonal_offset + column * size + eigen] /
                  eigenvalues_[eigen_offset + eigen];
            }
            inverse[row * size + column] = value;
         }
      }
   }

   void FormMultiplier(
      Index lower_offset,
      Index rows,
      Index columns,
      Index multiplier_offset,
      Index inverse_offset
   )
   {
      if( rows == 12 && columns == 12 )
      {
         FormMultiplierFixed<12, 12>(
            lower_offset, rows, columns, multiplier_offset, inverse_offset);
         return;
      }
      if( rows == 14 && columns == 14 )
      {
         FormMultiplierFixed<14, 14>(
            lower_offset, rows, columns, multiplier_offset, inverse_offset);
         return;
      }
      if( rows == 14 && columns == 12 )
      {
         FormMultiplierFixed<14, 12>(
            lower_offset, rows, columns, multiplier_offset, inverse_offset);
         return;
      }
      if( rows == 12 && columns == 14 )
      {
         FormMultiplierFixed<12, 14>(
            lower_offset, rows, columns, multiplier_offset, inverse_offset);
         return;
      }
      if( rows == 5 && columns == 5 )
      {
         FormMultiplierFixed<5, 5>(
            lower_offset, rows, columns, multiplier_offset, inverse_offset);
         return;
      }
      if( rows == 5 && columns == 4 )
      {
         FormMultiplierFixed<5, 4>(
            lower_offset, rows, columns, multiplier_offset, inverse_offset);
         return;
      }
      if( rows == 4 && columns == 5 )
      {
         FormMultiplierFixed<4, 5>(
            lower_offset, rows, columns, multiplier_offset, inverse_offset);
         return;
      }
      if( rows == 8 && columns == 8 )
      {
         FormMultiplierFixed<8, 8>(
            lower_offset, rows, columns, multiplier_offset, inverse_offset);
         return;
      }
      FormMultiplierFixed<0, 0>(
         lower_offset, rows, columns, multiplier_offset, inverse_offset);
   }

   template <Index StaticRows, Index StaticColumns>
   void FormMultiplierFixed(
      Index lower_offset,
      Index runtime_rows,
      Index runtime_columns,
      Index multiplier_offset,
      Index inverse_offset
   )
   {
      const Index rows = StaticRows == 0 ? runtime_rows : StaticRows;
      const Index columns =
         StaticColumns == 0 ? runtime_columns : StaticColumns;
      const Number* inverse = inverse_blocks_.data() + inverse_offset;
      for( Index row = 0; row < rows; ++row )
      {
         for( Index column = 0; column < columns; ++column )
         {
            Number value = 0.;
            for( Index inner = 0; inner < columns; ++inner )
            {
               value += original_lower_[lower_offset + row * columns + inner] *
                  inverse[inner * columns + column];
            }
            // Retain multiplier columns contiguously: both forward and
            // backward solves traverse this layout without strided loads.
            multipliers_[multiplier_offset + column * rows + row] = value;
         }
      }
   }

   void ApplySchurComplement(
      Index lower_offset,
      Index multiplier_offset,
      Index rows,
      Index columns
   )
   {
      if( options_.require_certified_inertia )
      {
         ApplySchurComplementDispatched<false>(
            lower_offset, multiplier_offset, rows, columns);
      }
      else
      {
         // Numeric inverses are projected onto symmetry once.  The Schur
         // product is therefore a symmetric rank-k update: evaluate one
         // triangle and mirror it instead of repeating every dot product.
         ApplySchurComplementDispatched<true>(
            lower_offset, multiplier_offset, rows, columns);
      }
   }

   template <bool SymmetricTriangle>
   void ApplySchurComplementDispatched(
      Index lower_offset,
      Index multiplier_offset,
      Index rows,
      Index columns
   )
   {
      if( rows == 12 && columns == 12 )
      {
         ApplySchurComplementFixed<SymmetricTriangle, 12, 12>(
            lower_offset, multiplier_offset, rows, columns);
         return;
      }
      if( rows == 14 && columns == 14 )
      {
         ApplySchurComplementFixed<SymmetricTriangle, 14, 14>(
            lower_offset, multiplier_offset, rows, columns);
         return;
      }
      if( rows == 14 && columns == 12 )
      {
         ApplySchurComplementFixed<SymmetricTriangle, 14, 12>(
            lower_offset, multiplier_offset, rows, columns);
         return;
      }
      if( rows == 12 && columns == 14 )
      {
         ApplySchurComplementFixed<SymmetricTriangle, 12, 14>(
            lower_offset, multiplier_offset, rows, columns);
         return;
      }
      if( rows == 5 && columns == 5 )
      {
         ApplySchurComplementFixed<SymmetricTriangle, 5, 5>(
            lower_offset, multiplier_offset, rows, columns);
         return;
      }
      if( rows == 5 && columns == 4 )
      {
         ApplySchurComplementFixed<SymmetricTriangle, 5, 4>(
            lower_offset, multiplier_offset, rows, columns);
         return;
      }
      if( rows == 4 && columns == 5 )
      {
         ApplySchurComplementFixed<SymmetricTriangle, 4, 5>(
            lower_offset, multiplier_offset, rows, columns);
         return;
      }
      if( rows == 8 && columns == 8 )
      {
         ApplySchurComplementFixed<SymmetricTriangle, 8, 8>(
            lower_offset, multiplier_offset, rows, columns);
         return;
      }
      ApplySchurComplementFixed<SymmetricTriangle, 0, 0>(
         lower_offset, multiplier_offset, rows, columns);
   }

   template <bool SymmetricTriangle, Index StaticRows, Index StaticColumns>
   void ApplySchurComplementFixed(
      Index lower_offset,
      Index multiplier_offset,
      Index runtime_rows,
      Index runtime_columns
   )
   {
      const Index rows = StaticRows == 0 ? runtime_rows : StaticRows;
      const Index columns =
         StaticColumns == 0 ? runtime_columns : StaticColumns;
      for( Index row = 0; row < rows; ++row )
      {
         const Index column_count = SymmetricTriangle ? row + 1 : rows;
         for( Index column = 0; column < column_count; ++column )
         {
            Number update = 0.;
            for( Index inner = 0; inner < columns; ++inner )
            {
               update += multipliers_[multiplier_offset + inner * rows + row] *
                  original_lower_[lower_offset + column * columns + inner];
            }
            const Number value = matrix_work_[row * rows + column] - update;
            matrix_work_[row * rows + column] = value;
            if constexpr( SymmetricTriangle )
            {
               matrix_work_[column * rows + row] = value;
            }
         }
      }
   }

   void SymmetrizeWork(Index size)
   {
      for( Index row = 0; row < size; ++row )
      {
         for( Index column = 0; column < row; ++column )
         {
            const Number average = 0.5 *
               (matrix_work_[row * size + column] +
                matrix_work_[column * size + row]);
            matrix_work_[row * size + column] = average;
            matrix_work_[column * size + row] = average;
         }
      }
   }

   static Number MatrixInfinityNorm(std::span<const Number> matrix, Index size)
   {
      Number result = 0.;
      for( Index row = 0; row < size; ++row )
      {
         Number row_sum = 0.;
         for( Index column = 0; column < size; ++column )
         {
            row_sum += std::abs(matrix[row * size + column]);
         }
         result = std::max(result, row_sum);
      }
      return result;
   }

   static Number OffDiagonalFrobeniusNorm(
      std::span<const Number> matrix,
      Index                   size
   )
   {
      Number squared_norm = 0.;
      for( Index row = 0; row < size; ++row )
      {
         for( Index column = 0; column < row; ++column )
         {
            const Number value = matrix[row * size + column];
            squared_norm += 2. * value * value;
         }
      }
      return std::sqrt(squared_norm);
   }

   EvaluationValue<StageFactorizationReport> FactorizeStage(
      Index stage,
      Index size,
      Number inherited_schur_error
   )
   {
      std::ranges::copy_n(
         matrix_work_.begin(), size * size, stage_matrix_original_work_.begin());
      const Index diagonal_offset = diagonal_offsets_[stage];
      Number* orthogonal = orthogonal_blocks_.data() + diagonal_offset;
      std::ranges::fill_n(orthogonal, size * size, 0.);
      for( Index diagonal = 0; diagonal < size; ++diagonal )
      {
         orthogonal[diagonal * size + diagonal] = 1.;
      }

      const Number matrix_scale = MatrixInfinityNorm(
         std::span<const Number>(matrix_work_.data(), size * size), size);
      if( !std::isfinite(matrix_scale) )
      {
         return Failure<StageFactorizationReport>(
            EvaluationErrorCode::nonfinite_output,
            "Schur block formation produced a nonfinite value at stage " +
               std::to_string(stage));
      }
      const Number convergence_threshold = options_.jacobi_tolerance * matrix_scale;
      Number off_diagonal = OffDiagonalFrobeniusNorm(
         std::span<const Number>(matrix_work_.data(), size * size), size);
      Index sweeps = 0;
      while( off_diagonal > convergence_threshold &&
             sweeps < options_.maximum_jacobi_sweeps )
      {
         for( Index first = 0; first < size; ++first )
         {
            for( Index second = first + 1; second < size; ++second )
            {
               RotateJacobi(first, second, size, orthogonal);
            }
         }
         ++sweeps;
         off_diagonal = OffDiagonalFrobeniusNorm(
            std::span<const Number>(matrix_work_.data(), size * size), size);
         if( !std::isfinite(off_diagonal) )
         {
            return Failure<StageFactorizationReport>(
               EvaluationErrorCode::nonfinite_output,
               "Jacobi factorization produced a nonfinite value at stage " +
                  std::to_string(stage));
         }
      }
      if( off_diagonal > convergence_threshold )
      {
         return Failure<StageFactorizationReport>(
            EvaluationErrorCode::model_failure,
            "Jacobi factorization did not converge at stage " +
               std::to_string(stage));
      }

      StageFactorizationReport report;
      report.minimum_separation = std::numeric_limits<Number>::infinity();
      report.sweeps = sweeps;
      const Index eigen_offset = block_offsets_[stage];
      for( Index eigen = 0; eigen < size; ++eigen )
      {
         const Number value = matrix_work_[eigen * size + eigen];
         eigenvalues_[eigen_offset + eigen] = value;
         if( !std::isfinite(value) )
         {
            return Failure<StageFactorizationReport>(
               EvaluationErrorCode::nonfinite_output,
               "Jacobi factorization produced a nonfinite eigenvalue at stage " +
                  std::to_string(stage));
         }
         const Number numeric_pivot_margin =
            options_.relative_numeric_pivot_tolerance * matrix_scale +
            options_.absolute_inertia_tolerance;
         if( !(std::abs(value) > numeric_pivot_margin) )
         {
            return Failure<StageFactorizationReport>(
               EvaluationErrorCode::model_failure,
               "Schur block is numerically singular at stage " +
                  std::to_string(stage));
         }
         if( value > 0. )
         {
            ++report.positive_eigenvalues;
         }
         else
         {
            ++report.negative_eigenvalues;
         }
      }

      const Number orthogonality_error = OrthogonalityError(orthogonal, size);
      const Number reconstruction_error = ReconstructionError(
         orthogonal,
         std::span<const Number>(eigenvalues_.data() + eigen_offset, size),
         size);
      if( !std::isfinite(orthogonality_error) ||
          !std::isfinite(reconstruction_error) ||
          !(orthogonality_error < 1.) )
      {
         return Failure<StageFactorizationReport>(
            EvaluationErrorCode::model_failure,
            "Schur block eigensystem is not certifiably nonsingular at stage " +
               std::to_string(stage));
      }
      const Number certificate_radius = reconstruction_error +
         inherited_schur_error +
         options_.relative_inertia_tolerance * matrix_scale +
         options_.absolute_inertia_tolerance;
      report.certificate_radius = certificate_radius;
      for( Index eigen = 0; eigen < size; ++eigen )
      {
         const Number separation =
            (1. - orthogonality_error) *
               std::abs(eigenvalues_[eigen_offset + eigen]) -
            certificate_radius;
         if( !(separation > 0.) && options_.require_certified_inertia )
         {
            return Failure<StageFactorizationReport>(
               EvaluationErrorCode::model_failure,
               "Schur block inertia is singular or uncertified at stage " +
                  std::to_string(stage) + ": |lambda|=" +
                  std::to_string(
                     std::abs(eigenvalues_[eigen_offset + eigen])) +
                  " certificate_radius=" +
                  std::to_string(certificate_radius) +
                  " inherited_error=" +
                  std::to_string(inherited_schur_error));
         }
         if( options_.require_certified_inertia )
         {
            report.minimum_separation =
               std::min(report.minimum_separation, separation);
         }
         else
         {
            report.minimum_separation = 0.;
         }
      }
      return report;
   }

   EvaluationValue<StageFactorizationReport> FactorizeStageNumerically(
      Index stage,
      Index size
   )
   {
      const Number matrix_scale = MatrixInfinityNorm(
         std::span<const Number>(matrix_work_.data(), size * size), size);
      if( !std::isfinite(matrix_scale) )
      {
         return Failure<StageFactorizationReport>(
            EvaluationErrorCode::nonfinite_output,
            "Schur block formation produced a nonfinite value at stage " +
               std::to_string(stage));
      }

      const Number pivot_margin =
         options_.relative_numeric_pivot_tolerance * matrix_scale +
         options_.absolute_inertia_tolerance;
      EvaluationResult inverted = [&]() -> EvaluationResult
      {
         switch( size )
         {
            case 1: return InvertStageNumerically<1>(stage, size, pivot_margin);
            case 2: return InvertStageNumerically<2>(stage, size, pivot_margin);
            case 3: return InvertStageNumerically<3>(stage, size, pivot_margin);
            case 4: return InvertStageNumerically<4>(stage, size, pivot_margin);
            case 5: return InvertStageNumerically<5>(stage, size, pivot_margin);
            case 6: return InvertStageNumerically<6>(stage, size, pivot_margin);
            case 7: return InvertStageNumerically<7>(stage, size, pivot_margin);
            case 8: return InvertStageNumerically<8>(stage, size, pivot_margin);
            case 12: return InvertStageNumerically<12>(stage, size, pivot_margin);
            case 14: return InvertStageNumerically<14>(stage, size, pivot_margin);
            default: return InvertStageNumerically<0>(stage, size, pivot_margin);
         }
      }();
      if( !inverted )
      {
         return std::unexpected(inverted.error());
      }
      Number* inverse = inverse_blocks_.data() + diagonal_offsets_[stage];
      if( !AllFinite(std::span<const Number>(inverse, size * size)) )
      {
         return Failure<StageFactorizationReport>(
            EvaluationErrorCode::nonfinite_output,
            "Schur block inverse produced a nonfinite value at stage " +
               std::to_string(stage));
      }
      return StageFactorizationReport{
         .certificate_radius = std::numeric_limits<Number>::infinity(),
         .minimum_separation = 0.
      };
   }

   void SymmetrizeInverse(Index stage, Index size)
   {
      Number* inverse = inverse_blocks_.data() + diagonal_offsets_[stage];
      for( Index row = 0; row < size; ++row )
      {
         for( Index column = 0; column < row; ++column )
         {
            const Number average = 0.5 *
               (inverse[row * size + column] +
                inverse[column * size + row]);
            inverse[row * size + column] = average;
            inverse[column * size + row] = average;
         }
      }
   }

   template <Index StaticSize>
   EvaluationResult InvertStageNumerically(
      Index stage,
      Index runtime_size,
      Number pivot_margin
   )
   {
      const Index size = StaticSize == 0 ? runtime_size : StaticSize;
      Number* const stored_inverse =
         inverse_blocks_.data() + diagonal_offsets_[stage];
      Number* inverse = stored_inverse;
      std::array<Index, StaticSize> inverse_column_permutation{};
      bool inverse_columns_permuted = false;
      std::ranges::fill_n(inverse, size * size, 0.);
      for( Index diagonal = 0; diagonal < size; ++diagonal )
      {
         inverse[diagonal * size + diagonal] = 1.;
         if constexpr( StaticSize != 0 )
         {
            inverse_column_permutation[diagonal] = diagonal;
         }
      }
      for( Index column = 0; column < size; ++column )
      {
         Index pivot_row = column;
         Number pivot_magnitude = std::abs(
            matrix_work_[column * size + column]);
         for( Index row = column + 1; row < size; ++row )
         {
            const Number candidate = std::abs(
               matrix_work_[row * size + column]);
            if( candidate > pivot_magnitude )
            {
               pivot_magnitude = candidate;
               pivot_row = row;
            }
         }
         if( !(pivot_magnitude > pivot_margin) )
         {
            return Error(
               EvaluationErrorCode::model_failure,
               "Schur block is numerically singular at stage " +
                  std::to_string(stage));
         }
         if( pivot_row != column )
         {
            // Earlier matrix columns are dead after their pivot.  Swapping
            // them again cannot affect a later search or elimination and
            // dominates these small dense stage factors.
            for( Index entry = column; entry < size; ++entry )
            {
               std::swap(
                  matrix_work_[column * size + entry],
                  matrix_work_[pivot_row * size + entry]);
            }
            if constexpr( StaticSize != 0 )
            {
               if( !inverse_columns_permuted )
               {
                  std::ranges::copy_n(
                     inverse,
                     size * size,
                     stage_matrix_original_work_.begin());
                  inverse = stage_matrix_original_work_.data();
                  inverse_columns_permuted = true;
               }
               // The augmented identity is kept in pivot-column order.  At
               // step k, an unprocessed row has exact support only in [0,k)
               // and its own identity column.  Combining the RHS row swap
               // with the matching column relabel therefore moves only the
               // already-active prefix and keeps all future updates
               // contiguous.
               for( Index entry = 0; entry < column; ++entry )
               {
                  std::swap(
                     inverse[column * size + entry],
                     inverse[pivot_row * size + entry]);
               }
               std::swap(
                  inverse_column_permutation[column],
                  inverse_column_permutation[pivot_row]);
            }
            else
            {
               for( Index entry = 0; entry < size; ++entry )
               {
                  std::swap(
                     inverse[column * size + entry],
                     inverse[pivot_row * size + entry]);
               }
            }
         }

         const Number pivot = matrix_work_[column * size + column];
         for( Index entry = column + 1; entry < size; ++entry )
         {
            matrix_work_[column * size + entry] /= pivot;
         }
         if constexpr( StaticSize != 0 )
         {
            for( Index entry = 0; entry <= column; ++entry )
            {
               inverse[column * size + entry] /= pivot;
            }
         }
         else
         {
            for( Index entry = 0; entry < size; ++entry )
            {
               inverse[column * size + entry] /= pivot;
            }
         }
         for( Index row = 0; row < size; ++row )
         {
            if( row == column )
            {
               continue;
            }
            const Number multiplier = matrix_work_[row * size + column];
            if( multiplier == 0. )
            {
               continue;
            }
            for( Index entry = column + 1; entry < size; ++entry )
            {
               matrix_work_[row * size + entry] -=
                  multiplier * matrix_work_[column * size + entry];
            }
            if constexpr( StaticSize != 0 )
            {
               for( Index entry = 0; entry <= column; ++entry )
               {
                  inverse[row * size + entry] -=
                     multiplier * inverse[column * size + entry];
               }
            }
            else
            {
               for( Index entry = 0; entry < size; ++entry )
               {
                  inverse[row * size + entry] -=
                     multiplier * inverse[column * size + entry];
               }
            }
         }
      }
      if constexpr( StaticSize != 0 )
      {
         if( inverse_columns_permuted )
         {
            // Restore the pivoted identity-column labels while projecting the
            // inverse onto symmetry, so both operations share one write pass.
            std::array<Index, StaticSize> stored_inverse_column;
            for( Index column = 0; column < size; ++column )
            {
               stored_inverse_column[inverse_column_permutation[column]] = column;
            }
            for( Index row = 0; row < size; ++row )
            {
               stored_inverse[row * size + row] =
                  inverse[row * size + stored_inverse_column[row]];
               for( Index column = 0; column < row; ++column )
               {
                  const Number average = 0.5 *
                     (inverse[row * size + stored_inverse_column[column]] +
                      inverse[column * size + stored_inverse_column[row]]);
                  stored_inverse[row * size + column] = average;
                  stored_inverse[column * size + row] = average;
               }
            }
            return {};
         }
      }
      // The source block is symmetric.  Project roundoff back onto that
      // invariant so multiplier formation, the Schur update, and solves all
      // consume the same symmetric approximate inverse.
      SymmetrizeInverse(stage, size);
      return {};
   }

   EvaluationValue<Number> CertifyInverse(
      Index stage,
      Index size,
      Number inherited_schur_error
   ) const
   {
      const Number* inverse =
         inverse_blocks_.data() + diagonal_offsets_[stage];
      Number squared_residual = 0.;
      for( Index row = 0; row < size; ++row )
      {
         for( Index column = 0; column < size; ++column )
         {
            Number product = 0.;
            for( Index inner = 0; inner < size; ++inner )
            {
               product += stage_matrix_original_work_[row * size + inner] *
                  inverse[inner * size + column];
            }
            const Number residual = (row == column ? 1. : 0.) - product;
            squared_residual += residual * residual;
         }
      }
      const Number computed_residual = std::sqrt(squared_residual);
      const Number schur_norm = FrobeniusNorm(
         std::span<const Number>(stage_matrix_original_work_.data(), size * size));
      const Number inverse_norm = FrobeniusNorm(
         std::span<const Number>(inverse, size * size));
      const Number true_residual_bound = computed_residual +
         inherited_schur_error * inverse_norm +
         options_.relative_inertia_tolerance *
            (1. + schur_norm * inverse_norm);
      if( !std::isfinite(true_residual_bound) ||
          !std::isfinite(inverse_norm) ||
          !(true_residual_bound < 1.) )
      {
         return std::unexpected(EvaluationError{
            EvaluationErrorCode::model_failure,
            "Schur inverse is not certifiable at stage " +
               std::to_string(stage)
         });
      }
      const Number inverse_error = inverse_norm * true_residual_bound /
         (1. - true_residual_bound);
      if( !std::isfinite(inverse_error) )
      {
         return std::unexpected(EvaluationError{
            EvaluationErrorCode::nonfinite_output,
            "Schur inverse error bound is nonfinite at stage " +
               std::to_string(stage)
         });
      }
      return inverse_error;
   }

   static Number OrthogonalityError(const Number* orthogonal, Index size)
   {
      Number squared_error = 0.;
      for( Index row = 0; row < size; ++row )
      {
         for( Index column = 0; column < size; ++column )
         {
            Number product = 0.;
            for( Index inner = 0; inner < size; ++inner )
            {
               product += orthogonal[inner * size + row] *
                  orthogonal[inner * size + column];
            }
            const Number error = product - (row == column ? 1. : 0.);
            squared_error += error * error;
         }
      }
      return std::sqrt(squared_error);
   }

   Number ReconstructionError(
      const Number*           orthogonal,
      std::span<const Number> eigenvalues,
      Index                   size
   ) const
   {
      Number squared_error = 0.;
      for( Index row = 0; row < size; ++row )
      {
         for( Index column = 0; column < size; ++column )
         {
            Number reconstructed = 0.;
            for( Index eigen = 0; eigen < size; ++eigen )
            {
               reconstructed += orthogonal[row * size + eigen] *
                  eigenvalues[eigen] * orthogonal[column * size + eigen];
            }
            const Number error =
               stage_matrix_original_work_[row * size + column] - reconstructed;
            squared_error += error * error;
         }
      }
      return std::sqrt(squared_error);
   }

   void RotateJacobi(Index first, Index second, Index size, Number* orthogonal)
   {
      const Number cross = matrix_work_[first * size + second];
      if( cross == 0. )
      {
         return;
      }
      const Number first_diagonal = matrix_work_[first * size + first];
      const Number second_diagonal = matrix_work_[second * size + second];
      const Number tau = (second_diagonal - first_diagonal) / (2. * cross);
      const Number tangent = tau == 0.
         ? std::copysign(1., cross)
         : std::copysign(1., tau) / (std::abs(tau) + std::hypot(1., tau));
      const Number cosine = 1. / std::hypot(1., tangent);
      const Number sine = tangent * cosine;

      for( Index entry = 0; entry < size; ++entry )
      {
         if( entry == first || entry == second )
         {
            continue;
         }
         const Number entry_first = matrix_work_[entry * size + first];
         const Number entry_second = matrix_work_[entry * size + second];
         const Number rotated_first = cosine * entry_first - sine * entry_second;
         const Number rotated_second = sine * entry_first + cosine * entry_second;
         matrix_work_[entry * size + first] = rotated_first;
         matrix_work_[first * size + entry] = rotated_first;
         matrix_work_[entry * size + second] = rotated_second;
         matrix_work_[second * size + entry] = rotated_second;
      }
      matrix_work_[first * size + first] = first_diagonal - tangent * cross;
      matrix_work_[second * size + second] = second_diagonal + tangent * cross;
      matrix_work_[first * size + second] = 0.;
      matrix_work_[second * size + first] = 0.;

      for( Index row = 0; row < size; ++row )
      {
         const Number first_vector = orthogonal[row * size + first];
         const Number second_vector = orthogonal[row * size + second];
         orthogonal[row * size + first] =
            cosine * first_vector - sine * second_vector;
         orthogonal[row * size + second] =
            sine * first_vector + cosine * second_vector;
      }
   }

   EvaluationResult ValidateSolve(
      std::span<const Number> input,
      Index                   rhs_count,
      std::span<Number>       output
   ) const
   {
      if( !factorized_ )
      {
         return Error(
            EvaluationErrorCode::model_failure,
            "block-tridiagonal solve requested before factorization");
      }
      if( rhs_count == 0 || rhs_count > maximum_right_hand_sides_ )
      {
         return Error(
            EvaluationErrorCode::dimension_mismatch,
            "block-tridiagonal RHS count is outside the prepared capacity");
      }
      Index expected = 0;
      if( !CheckedMultiply(storage_.dimension, rhs_count, expected) )
      {
         return Error(
            EvaluationErrorCode::dimension_mismatch,
            "block-tridiagonal RHS storage overflows Index");
      }
      if( input.size() != expected )
      {
         return detail::DimensionMismatch(
            "block-tridiagonal solve input", input.size(), expected);
      }
      if( output.size() != expected )
      {
         return detail::DimensionMismatch(
            "block-tridiagonal solve output", output.size(), expected);
      }
      if( !AllFinite(input) )
      {
         return Error(
            EvaluationErrorCode::nonfinite_output,
            "block-tridiagonal RHS contains a nonfinite value");
      }
      return {};
   }

   EvaluationResult SolveMany(
      std::span<const Number> input,
      Index                   rhs_count,
      std::span<Number>       output
   )
   {
      if( EvaluationResult valid = ValidateSolve(input, rhs_count, output); !valid )
      {
         return valid;
      }
      const Index value_count = storage_.dimension * rhs_count;
      SolveManyImpl(input, rhs_count, solution_work_);
      if( !AllFinite(std::span<const Number>(solution_work_.data(), value_count)) )
      {
         return Error(
            EvaluationErrorCode::nonfinite_output,
            "block-tridiagonal solve produced a nonfinite value");
      }
      std::ranges::copy_n(solution_work_.begin(), value_count, output.begin());
      ++statistics_.solve_calls;
      statistics_.solved_right_hand_sides += rhs_count;
      return {};
   }

   void SolveManyImpl(
      std::span<const Number> input,
      Index                   rhs_count,
      std::span<Number>       result
   )
   {
      if( rhs_count == 1 )
      {
         SolveOneImpl(input, result);
         return;
      }
      const Index value_count = storage_.dimension * rhs_count;
      std::ranges::copy_n(input.begin(), value_count, result.begin());

      for( Index stage = 1; stage < block_sizes_.size(); ++stage )
      {
         const Index rows = block_sizes_[stage];
         const Index columns = block_sizes_[stage - 1];
         const Index current_offset = block_offsets_[stage];
         const Index previous_offset = block_offsets_[stage - 1];
         const Index multiplier_offset = multiplier_offsets_[stage - 1];
         for( Index column = 0; column < columns; ++column )
         {
            for( Index row = 0; row < rows; ++row )
            {
               const Number coefficient =
                  multipliers_[multiplier_offset + column * rows + row];
               AddScaledContiguous(
                  -coefficient,
                  result.data() + (previous_offset + column) * rhs_count,
                  result.data() + (current_offset + row) * rhs_count,
                  rhs_count);
            }
         }
      }

      for( Index stage = 0; stage < block_sizes_.size(); ++stage )
      {
         SolveStage(stage, rhs_count, result);
      }

      for( Index reverse = block_sizes_.size() - 1; reverse > 0; --reverse )
      {
         const Index stage = reverse;
         const Index rows = block_sizes_[stage];
         const Index columns = block_sizes_[stage - 1];
         const Index current_offset = block_offsets_[stage];
         const Index previous_offset = block_offsets_[stage - 1];
         const Index multiplier_offset = multiplier_offsets_[stage - 1];
         for( Index column = 0; column < columns; ++column )
         {
            for( Index row = 0; row < rows; ++row )
            {
               const Number coefficient =
                  multipliers_[multiplier_offset + column * rows + row];
               AddScaledContiguous(
                  -coefficient,
                  result.data() + (current_offset + row) * rhs_count,
                  result.data() + (previous_offset + column) * rhs_count,
                  rhs_count);
            }
         }
      }
   }

   void SolveOneImpl(
      std::span<const Number> input,
      std::span<Number>       result
   )
   {
      std::ranges::copy_n(input.begin(), storage_.dimension, result.begin());
      for( Index stage = 1; stage < block_sizes_.size(); ++stage )
      {
         const Index rows = block_sizes_[stage];
         const Index columns = block_sizes_[stage - 1];
         const Index current_offset = block_offsets_[stage];
         const Index previous_offset = block_offsets_[stage - 1];
         const Index multiplier_offset = multiplier_offsets_[stage - 1];
         ForwardEliminateOne(
            multipliers_.data() + multiplier_offset,
            result.data() + previous_offset,
            result.data() + current_offset,
            rows,
            columns);
      }
      for( Index stage = 0; stage < block_sizes_.size(); ++stage )
      {
         SolveStage(stage, 1, result);
      }
      for( Index reverse = block_sizes_.size() - 1; reverse > 0; --reverse )
      {
         const Index stage = reverse;
         const Index rows = block_sizes_[stage];
         const Index columns = block_sizes_[stage - 1];
         const Index current_offset = block_offsets_[stage];
         const Index previous_offset = block_offsets_[stage - 1];
         const Index multiplier_offset = multiplier_offsets_[stage - 1];
         BackwardEliminateOne(
            multipliers_.data() + multiplier_offset,
            result.data() + current_offset,
            result.data() + previous_offset,
            rows,
            columns);
      }
   }

   void ForwardEliminateOne(
      const Number* multiplier,
      const Number* previous,
      Number*       current,
      Index         rows,
      Index         columns
   ) const noexcept
   {
      if( rows == 12 && columns == 12 )
      {
         ForwardEliminateOneFixed<12, 12>(
            multiplier, previous, current, rows, columns);
         return;
      }
      if( rows == 14 && columns == 14 )
      {
         ForwardEliminateOneFixed<14, 14>(
            multiplier, previous, current, rows, columns);
         return;
      }
      if( rows == 14 && columns == 12 )
      {
         ForwardEliminateOneFixed<14, 12>(
            multiplier, previous, current, rows, columns);
         return;
      }
      if( rows == 12 && columns == 14 )
      {
         ForwardEliminateOneFixed<12, 14>(
            multiplier, previous, current, rows, columns);
         return;
      }
      if( rows == 5 && columns == 5 )
      {
         ForwardEliminateOneFixed<5, 5>(
            multiplier, previous, current, rows, columns);
         return;
      }
      if( rows == 5 && columns == 4 )
      {
         ForwardEliminateOneFixed<5, 4>(
            multiplier, previous, current, rows, columns);
         return;
      }
      if( rows == 4 && columns == 5 )
      {
         ForwardEliminateOneFixed<4, 5>(
            multiplier, previous, current, rows, columns);
         return;
      }
      if( rows == 8 && columns == 8 )
      {
         ForwardEliminateOneFixed<8, 8>(
            multiplier, previous, current, rows, columns);
         return;
      }
      ForwardEliminateOneFixed<0, 0>(
         multiplier, previous, current, rows, columns);
   }

   template <Index StaticRows, Index StaticColumns>
   static void ForwardEliminateOneFixed(
      const Number* multiplier,
      const Number* previous,
      Number*       current,
      Index         runtime_rows,
      Index         runtime_columns
   ) noexcept
   {
      const Index rows = StaticRows == 0 ? runtime_rows : StaticRows;
      const Index columns =
         StaticColumns == 0 ? runtime_columns : StaticColumns;
      for( Index column = 0; column < columns; ++column )
      {
         const Number value = previous[column];
         const Number* multiplier_column = multiplier + column * rows;
         for( Index row = 0; row < rows; ++row )
         {
            current[row] -= multiplier_column[row] * value;
         }
      }
   }

   void BackwardEliminateOne(
      const Number* multiplier,
      const Number* current,
      Number*       previous,
      Index         rows,
      Index         columns
   ) const noexcept
   {
      if( rows == 12 && columns == 12 )
      {
         BackwardEliminateOneFixed<12, 12>(
            multiplier, current, previous, rows, columns);
         return;
      }
      if( rows == 14 && columns == 14 )
      {
         BackwardEliminateOneFixed<14, 14>(
            multiplier, current, previous, rows, columns);
         return;
      }
      if( rows == 14 && columns == 12 )
      {
         BackwardEliminateOneFixed<14, 12>(
            multiplier, current, previous, rows, columns);
         return;
      }
      if( rows == 12 && columns == 14 )
      {
         BackwardEliminateOneFixed<12, 14>(
            multiplier, current, previous, rows, columns);
         return;
      }
      if( rows == 5 && columns == 5 )
      {
         BackwardEliminateOneFixed<5, 5>(
            multiplier, current, previous, rows, columns);
         return;
      }
      if( rows == 5 && columns == 4 )
      {
         BackwardEliminateOneFixed<5, 4>(
            multiplier, current, previous, rows, columns);
         return;
      }
      if( rows == 4 && columns == 5 )
      {
         BackwardEliminateOneFixed<4, 5>(
            multiplier, current, previous, rows, columns);
         return;
      }
      if( rows == 8 && columns == 8 )
      {
         BackwardEliminateOneFixed<8, 8>(
            multiplier, current, previous, rows, columns);
         return;
      }
      BackwardEliminateOneFixed<0, 0>(
         multiplier, current, previous, rows, columns);
   }

   template <Index StaticRows, Index StaticColumns>
   static void BackwardEliminateOneFixed(
      const Number* multiplier,
      const Number* current,
      Number*       previous,
      Index         runtime_rows,
      Index         runtime_columns
   ) noexcept
   {
      const Index rows = StaticRows == 0 ? runtime_rows : StaticRows;
      const Index columns =
         StaticColumns == 0 ? runtime_columns : StaticColumns;
      for( Index column = 0; column < columns; ++column )
      {
         Number value = previous[column];
         const Number* multiplier_column = multiplier + column * rows;
         for( Index row = 0; row < rows; ++row )
         {
            value -= multiplier_column[row] * current[row];
         }
         previous[column] = value;
      }
   }

   void SolveStage(Index stage, Index rhs_count, std::span<Number> values)
   {
      const Index size = block_sizes_[stage];
      const Index block_offset = block_offsets_[stage];
      const Index diagonal_offset = diagonal_offsets_[stage];
      if( !options_.require_certified_inertia )
      {
         switch( size )
         {
            case 1: SolveStageWithInverse<1>(stage, size, rhs_count, values); break;
            case 2: SolveStageWithInverse<2>(stage, size, rhs_count, values); break;
            case 3: SolveStageWithInverse<3>(stage, size, rhs_count, values); break;
            case 4: SolveStageWithInverse<4>(stage, size, rhs_count, values); break;
            case 5: SolveStageWithInverse<5>(stage, size, rhs_count, values); break;
            case 6: SolveStageWithInverse<6>(stage, size, rhs_count, values); break;
            case 7: SolveStageWithInverse<7>(stage, size, rhs_count, values); break;
            case 8: SolveStageWithInverse<8>(stage, size, rhs_count, values); break;
            case 12: SolveStageWithInverse<12>(stage, size, rhs_count, values); break;
            case 14: SolveStageWithInverse<14>(stage, size, rhs_count, values); break;
            default: SolveStageWithInverse<0>(stage, size, rhs_count, values); break;
         }
         return;
      }
      const Number* orthogonal = orthogonal_blocks_.data() + diagonal_offset;
      const Number* eigenvalues = eigenvalues_.data() + block_offset;

      std::ranges::fill_n(stage_rhs_work_.begin(), size * rhs_count, 0.);
      for( Index eigen = 0; eigen < size; ++eigen )
      {
         for( Index row = 0; row < size; ++row )
         {
            const Number coefficient = orthogonal[row * size + eigen];
            AddScaledContiguous(
               coefficient,
               values.data() + (block_offset + row) * rhs_count,
               stage_rhs_work_.data() + eigen * rhs_count,
               rhs_count);
         }
         for( Index rhs = 0; rhs < rhs_count; ++rhs )
         {
            stage_rhs_work_[eigen * rhs_count + rhs] /= eigenvalues[eigen];
         }
      }
      for( Index row = 0; row < size; ++row )
      {
         std::ranges::fill_n(
            values.begin() + block_offset * rhs_count + row * rhs_count,
            rhs_count,
            0.);
         for( Index eigen = 0; eigen < size; ++eigen )
         {
            const Number coefficient = orthogonal[row * size + eigen];
            AddScaledContiguous(
               coefficient,
               stage_rhs_work_.data() + eigen * rhs_count,
               values.data() + (block_offset + row) * rhs_count,
               rhs_count);
         }
      }
   }

   template <Index StaticSize>
   void SolveStageWithInverse(
      Index stage,
      Index runtime_size,
      Index rhs_count,
      std::span<Number> values
   )
   {
      const Index size = StaticSize == 0 ? runtime_size : StaticSize;
      const Index block_offset = block_offsets_[stage];
      const Number* inverse =
         inverse_blocks_.data() + diagonal_offsets_[stage];
      if( rhs_count == 1 )
      {
         if constexpr( StaticSize != 0 )
         {
            std::array<Number, StaticSize> fixed_stage_rhs;
            for( Index row = 0; row < size; ++row )
            {
               Number result = 0.;
               for( Index column = 0; column < size; ++column )
               {
                  result += inverse[row * size + column] *
                     values[block_offset + column];
               }
               fixed_stage_rhs[row] = result;
            }
            std::ranges::copy(
               fixed_stage_rhs, values.begin() + block_offset);
            return;
         }
         for( Index row = 0; row < size; ++row )
         {
            Number result = 0.;
            for( Index column = 0; column < size; ++column )
            {
               result += inverse[row * size + column] *
                  values[block_offset + column];
            }
            stage_rhs_work_[row] = result;
         }
         std::ranges::copy_n(
            stage_rhs_work_.begin(), size, values.begin() + block_offset);
         return;
      }

      std::ranges::fill_n(stage_rhs_work_.begin(), size * rhs_count, 0.);
      for( Index row = 0; row < size; ++row )
      {
         for( Index column = 0; column < size; ++column )
         {
            AddScaledContiguous(
               inverse[row * size + column],
               values.data() + (block_offset + column) * rhs_count,
               stage_rhs_work_.data() + row * rhs_count,
               rhs_count);
         }
      }
      std::ranges::copy_n(
         stage_rhs_work_.begin(), size * rhs_count,
         values.begin() + block_offset * rhs_count);
   }

   static void AddScaledContiguous(
      Number        scale,
      const Number* input,
      Number*       output,
      Index         size
   ) noexcept
   {
      Index entry = 0;
      for( ; entry + 8 <= size; entry += 8 )
      {
         output[entry] += scale * input[entry];
         output[entry + 1] += scale * input[entry + 1];
         output[entry + 2] += scale * input[entry + 2];
         output[entry + 3] += scale * input[entry + 3];
         output[entry + 4] += scale * input[entry + 4];
         output[entry + 5] += scale * input[entry + 5];
         output[entry + 6] += scale * input[entry + 6];
         output[entry + 7] += scale * input[entry + 7];
      }
      for( ; entry < size; ++entry )
      {
         output[entry] += scale * input[entry];
      }
   }

   template <Index StaticSize>
   static void ApplySymmetricDiagonalBlock(
      const Number* diagonal,
      const Number* input,
      Number*       output
   ) noexcept
   {
      for( Index row = 0; row < StaticSize; ++row )
      {
         output[row] += diagonal[row * StaticSize + row] * input[row];
         for( Index column = 0; column < row; ++column )
         {
            const Number coefficient = diagonal[row * StaticSize + column];
            output[row] += coefficient * input[column];
            output[column] += coefficient * input[row];
         }
      }
   }

   static void ApplySymmetricDiagonalBlock(
      Index         size,
      const Number* diagonal,
      const Number* input,
      Number*       output
   ) noexcept
   {
      if( size == 12 )
      {
         ApplySymmetricDiagonalBlock<12>(diagonal, input, output);
         return;
      }
      if( size == 14 )
      {
         ApplySymmetricDiagonalBlock<14>(diagonal, input, output);
         return;
      }
      for( Index row = 0; row < size; ++row )
      {
         output[row] += diagonal[row * size + row] * input[row];
         for( Index column = 0; column < row; ++column )
         {
            const Number coefficient = diagonal[row * size + column];
            output[row] += coefficient * input[column];
            output[column] += coefficient * input[row];
         }
      }
   }

   void ApplyOriginal(
      std::span<const Number> input,
      std::span<Number>       output
   ) const
   {
      std::ranges::fill(output, 0.);
      for( Index stage = 0; stage < block_sizes_.size(); ++stage )
      {
         const Index size = block_sizes_[stage];
         const Index block_offset = block_offsets_[stage];
         const Index diagonal_offset = diagonal_offsets_[stage];
         ApplySymmetricDiagonalBlock(
            size,
            original_diagonal_.data() + diagonal_offset,
            input.data() + block_offset,
            output.data() + block_offset);
         if( stage == 0 )
         {
            continue;
         }
         const Index previous_size = block_sizes_[stage - 1];
         const Index previous_offset = block_offsets_[stage - 1];
         const Index lower_offset = lower_offsets_[stage - 1];
         for( Index row = 0; row < size; ++row )
         {
            for( Index column = 0; column < previous_size; ++column )
            {
               const Number coefficient =
                  original_lower_[lower_offset + row * previous_size + column];
               output[block_offset + row] += coefficient * input[previous_offset + column];
               output[previous_offset + column] += coefficient * input[block_offset + row];
            }
         }
      }
   }

   EvaluationResult ApplyOriginalNoAlias(
      std::span<const Number> input,
      std::span<Number>       output
   )
   {
      ApplyOriginal(input, output);
      if( !AllFinite(output) )
      {
         return Error(
            EvaluationErrorCode::nonfinite_output,
            "block-tridiagonal matrix application produced a nonfinite value");
      }
      ++statistics_.matrix_applications;
      return {};
   }

   static Number InfinityNorm(std::span<const Number> values)
   {
      Number result = 0.;
      for( Number value : values )
      {
         result = std::max(result, std::abs(value));
      }
      return result;
   }

   static Number FrobeniusNorm(std::span<const Number> values)
   {
      Number squared_norm = 0.;
      for( Number value : values )
      {
         squared_norm += value * value;
      }
      return std::sqrt(squared_norm);
   }

   Number ComputeResidual(
      std::span<const Number> rhs,
      std::span<const Number> solution
   )
   {
      ApplyOriginal(solution.first(storage_.dimension), matrix_application_work_);
      Number norm = 0.;
      for( Index row = 0; row < storage_.dimension; ++row )
      {
         residual_work_[row] = rhs[row] - matrix_application_work_[row];
         norm = std::max(norm, std::abs(residual_work_[row]));
      }
      ++statistics_.matrix_applications;
      return norm;
   }

   std::vector<Index> block_sizes_;
   Index maximum_right_hand_sides_;
   BlockTridiagonalFactorizationOptions options_;
   std::string configuration_error_;
   BlockTridiagonalStorage storage_;
   std::vector<Index> block_offsets_;
   std::vector<Index> diagonal_offsets_;
   std::vector<Index> lower_offsets_;
   std::vector<Index> multiplier_offsets_;
   std::vector<Number> original_diagonal_;
   std::vector<Number> original_lower_;
   std::vector<Number> orthogonal_blocks_;
   std::vector<Number> eigenvalues_;
   std::vector<Number> multipliers_;
   std::vector<Number> matrix_work_;
   std::vector<Number> stage_matrix_original_work_;
   std::vector<Number> inverse_blocks_;
   std::vector<Number> inverse_error_bounds_;
   std::vector<Number> stage_rhs_work_;
   std::vector<Number> solution_work_;
   std::vector<Number> residual_work_;
   std::vector<Number> correction_work_;
   std::vector<Number> matrix_application_work_;
   BlockTridiagonalFactorizationReport factorization_report_;
   BlockTridiagonalStatistics statistics_;
   bool factorized_ = false;
};
} // namespace Ipopt::Cxx23

#endif
