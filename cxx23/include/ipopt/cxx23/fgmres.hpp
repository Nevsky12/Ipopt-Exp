// Copyright (C) 2026 Ipopt contributors.
// All Rights Reserved.
// This file is distributed under the Eclipse Public License.

#ifndef IPOPT_CXX23_FGMRES_HPP
#define IPOPT_CXX23_FGMRES_HPP

#include <ipopt/cxx23/any_nlp_problem.hpp>

#include <algorithm>
#include <cmath>
#include <concepts>
#include <limits>
#include <span>
#include <utility>
#include <vector>

namespace Ipopt::Cxx23
{
enum class FgmresStatus
{
   converged,
   maximum_iterations,
   numerical_breakdown,
   nonfinite_value,
   invalid_configuration
};

struct FgmresOptions
{
   Index restart = 30;
   Index maximum_iterations = 100;
   Number relative_tolerance = 1e-8;
   Number absolute_tolerance = 0.;
   Number breakdown_tolerance = 64. * std::numeric_limits<Number>::epsilon();
   bool reorthogonalize = true;
   bool check_finite = true;
};

struct FgmresResult
{
   FgmresStatus status = FgmresStatus::invalid_configuration;
   Index iterations = 0;
   Index cycles = 0;
   Index operator_evaluations = 0;
   Index preconditioner_evaluations = 0;
   Number right_hand_side_norm = 0.;
   Number initial_residual_norm = 0.;
   Number final_residual_norm = 0.;
   Number convergence_threshold = 0.;

   bool converged() const noexcept
   {
      return status == FgmresStatus::converged;
   }
};

template <class Operator>
concept FgmresLinearOperator = requires(
   Operator&               apply,
   std::span<const Number> input,
   std::span<Number>       output
)
{
   { apply(input, output) } -> std::same_as<EvaluationResult>;
};

template <class Preconditioner>
concept FgmresPreconditioner = requires(
   Preconditioner&          apply,
   Index                    iteration,
   std::span<const Number> input,
   std::span<Number>       output
)
{
   { apply(iteration, input, output) } -> std::same_as<EvaluationResult>;
};

struct IdentityPreconditioner
{
   EvaluationResult operator()(
      Index,
      std::span<const Number> input,
      std::span<Number>       output
   ) const
   {
      std::ranges::copy(input, output.begin());
      return {};
   }
};

/** Restarted flexible GMRES with right preconditioning.
 *
 * All storage is allocated by the constructor. A successful solver return
 * writes the best verified candidate to solution. A callback error is
 * propagated through std::expected and leaves the caller's solution unchanged.
 * One solver instance is reusable but not safe for concurrent calls.
 */
class FgmresSolver
{
public:
   explicit FgmresSolver(Index dimension, FgmresOptions options = {})
      : dimension_(dimension),
        options_(options),
        solution_(dimension),
        base_solution_(dimension),
        residual_(dimension),
        operator_output_(dimension),
        arnoldi_work_(dimension),
        basis_((options.restart + 1) * dimension),
        preconditioned_basis_(options.restart * dimension),
        hessenberg_((options.restart + 1) * options.restart),
        givens_cosines_(options.restart),
        givens_sines_(options.restart),
        projected_rhs_(options.restart + 1),
        coefficients_(options.restart)
   {
   }

   Index dimension() const noexcept
   {
      return dimension_;
   }

   const FgmresOptions& options() const noexcept
   {
      return options_;
   }

   template <FgmresLinearOperator Operator, FgmresPreconditioner Preconditioner>
   EvaluationValue<FgmresResult> solve(
      Operator&&               apply_operator,
      Preconditioner&&         apply_preconditioner,
      std::span<const Number> right_hand_side,
      std::span<Number>       solution
   )
   {
      FgmresResult result;
      if( !ConfigurationIsValid() )
      {
         result.status = FgmresStatus::invalid_configuration;
         return result;
      }
      if( right_hand_side.size() != dimension_ )
      {
         return std::unexpected(
            detail::DimensionMismatch(
               "right hand side", right_hand_side.size(), dimension_).error());
      }
      if( solution.size() != dimension_ )
      {
         return std::unexpected(
            detail::DimensionMismatch("solution", solution.size(), dimension_).error());
      }

      std::ranges::copy(solution, solution_.begin());
      result.right_hand_side_norm = StableNorm(right_hand_side);
      result.convergence_threshold = std::max(
         options_.absolute_tolerance,
         options_.relative_tolerance * result.right_hand_side_norm);

      const auto finish = [&](FgmresStatus status, Number residual_norm)
         -> EvaluationValue<FgmresResult>
      {
         result.status = status;
         result.final_residual_norm = residual_norm;
         std::ranges::copy(solution_, solution.begin());
         return result;
      };

      if( dimension_ == 0 )
      {
         result.initial_residual_norm = 0.;
         return finish(FgmresStatus::converged, 0.);
      }

      if( EvaluationResult evaluated = apply_operator(solution_, operator_output_); !evaluated )
      {
         return std::unexpected(evaluated.error());
      }
      ++result.operator_evaluations;
      if( options_.check_finite && !AllFinite(operator_output_) )
      {
         result.initial_residual_norm = std::numeric_limits<Number>::quiet_NaN();
         return finish(FgmresStatus::nonfinite_value, result.initial_residual_norm);
      }
      Subtract(right_hand_side, operator_output_, residual_);
      Number residual_norm = StableNorm(residual_);
      result.initial_residual_norm = residual_norm;
      if( !std::isfinite(residual_norm) )
      {
         return finish(FgmresStatus::nonfinite_value, residual_norm);
      }
      if( residual_norm <= result.convergence_threshold )
      {
         return finish(FgmresStatus::converged, residual_norm);
      }
      if( options_.maximum_iterations == 0 )
      {
         return finish(FgmresStatus::maximum_iterations, residual_norm);
      }

      while( result.iterations < options_.maximum_iterations )
      {
         ++result.cycles;
         std::ranges::copy(solution_, base_solution_.begin());
         ResetCycleStorage();

         std::span<Number> first_basis = Basis(0);
         ScaleCopy(1. / residual_norm, residual_, first_basis);
         projected_rhs_[0] = residual_norm;

         const Index remaining_iterations = options_.maximum_iterations - result.iterations;
         const Index cycle_steps = std::min(options_.restart, remaining_iterations);
         bool restart_cycle = false;
         for( Index step = 0; step < cycle_steps; ++step )
         {
            std::span<Number> preconditioned = PreconditionedBasis(step);
            if( EvaluationResult evaluated = apply_preconditioner(
                   result.iterations, std::span<const Number>(Basis(step)), preconditioned);
                !evaluated )
            {
               return std::unexpected(evaluated.error());
            }
            ++result.preconditioner_evaluations;
            if( options_.check_finite && !AllFinite(preconditioned) )
            {
               return finish(FgmresStatus::nonfinite_value, residual_norm);
            }

            if( EvaluationResult evaluated = apply_operator(preconditioned, arnoldi_work_); !evaluated )
            {
               return std::unexpected(evaluated.error());
            }
            ++result.operator_evaluations;
            ++result.iterations;
            if( options_.check_finite && !AllFinite(arnoldi_work_) )
            {
               return finish(FgmresStatus::nonfinite_value, residual_norm);
            }

            const Number unorthogonalized_norm = StableNorm(arnoldi_work_);
            const Index orthogonalization_passes = options_.reorthogonalize ? 2 : 1;
            for( Index pass = 0; pass < orthogonalization_passes; ++pass )
            {
               for( Index basis_index = 0; basis_index <= step; ++basis_index )
               {
                  const std::span<const Number> basis_vector = Basis(basis_index);
                  const Number projection = Dot(arnoldi_work_, basis_vector);
                  Hessenberg(basis_index, step) += projection;
                  Axpy(-projection, basis_vector, arnoldi_work_);
               }
            }

            const Number next_basis_norm = StableNorm(arnoldi_work_);
            if( !std::isfinite(unorthogonalized_norm) || !std::isfinite(next_basis_norm) )
            {
               return finish(FgmresStatus::nonfinite_value, residual_norm);
            }
            Hessenberg(step + 1, step) = next_basis_norm;
            const bool happy_breakdown = next_basis_norm <=
               options_.breakdown_tolerance * unorthogonalized_norm;
            if( !happy_breakdown )
            {
               ScaleCopy(1. / next_basis_norm, arnoldi_work_, Basis(step + 1));
            }

            ApplyPreviousRotations(step);
            const Number diagonal = Hessenberg(step, step);
            const Number subdiagonal = Hessenberg(step + 1, step);
            const Number rotation_norm = std::hypot(diagonal, subdiagonal);
            const Number rotation_scale = std::max(
               unorthogonalized_norm, std::numeric_limits<Number>::min());
            if( !std::isfinite(rotation_norm) ||
                rotation_norm <= options_.breakdown_tolerance * rotation_scale )
            {
               return finish(FgmresStatus::numerical_breakdown, residual_norm);
            }

            givens_cosines_[step] = diagonal / rotation_norm;
            givens_sines_[step] = subdiagonal / rotation_norm;
            Hessenberg(step, step) = rotation_norm;
            Hessenberg(step + 1, step) = 0.;
            projected_rhs_[step + 1] =
               -givens_sines_[step] * projected_rhs_[step];
            projected_rhs_[step] *= givens_cosines_[step];

            const Number estimated_residual = std::abs(projected_rhs_[step + 1]);
            const bool estimated_converged =
               estimated_residual <= result.convergence_threshold;
            const bool cycle_finished = step + 1 == cycle_steps;
            if( estimated_converged || happy_breakdown || cycle_finished )
            {
               if( !BuildCandidate(step + 1) )
               {
                  return finish(FgmresStatus::numerical_breakdown, residual_norm);
               }
               if( EvaluationResult evaluated = apply_operator(solution_, operator_output_); !evaluated )
               {
                  return std::unexpected(evaluated.error());
               }
               ++result.operator_evaluations;
               if( options_.check_finite && !AllFinite(operator_output_) )
               {
                  return finish(FgmresStatus::nonfinite_value, residual_norm);
               }
               Subtract(right_hand_side, operator_output_, residual_);
               residual_norm = StableNorm(residual_);
               if( !std::isfinite(residual_norm) )
               {
                  return finish(FgmresStatus::nonfinite_value, residual_norm);
               }
               if( residual_norm <= result.convergence_threshold )
               {
                  return finish(FgmresStatus::converged, residual_norm);
               }
               if( result.iterations >= options_.maximum_iterations )
               {
                  return finish(FgmresStatus::maximum_iterations, residual_norm);
               }
               restart_cycle = true;
               break;
            }
         }
         if( !restart_cycle )
         {
            return finish(FgmresStatus::numerical_breakdown, residual_norm);
         }
      }

      return finish(FgmresStatus::maximum_iterations, residual_norm);
   }

private:
   bool ConfigurationIsValid() const noexcept
   {
      return options_.restart > 0 &&
         std::isfinite(options_.relative_tolerance) &&
         options_.relative_tolerance >= 0. &&
         std::isfinite(options_.absolute_tolerance) &&
         options_.absolute_tolerance >= 0. &&
         std::isfinite(options_.breakdown_tolerance) &&
         options_.breakdown_tolerance > 0.;
   }

   static bool AllFinite(std::span<const Number> values) noexcept
   {
      return std::ranges::all_of(values, [](Number value) { return std::isfinite(value); });
   }

   static Number StableNorm(std::span<const Number> values) noexcept
   {
      Number scale = 0.;
      Number sum_of_squares = 1.;
      for( Number value : values )
      {
         const Number magnitude = std::abs(value);
         if( !std::isfinite(magnitude) )
         {
            return magnitude;
         }
         if( magnitude == 0. )
         {
            continue;
         }
         if( scale < magnitude )
         {
            const Number ratio = scale / magnitude;
            sum_of_squares = 1. + sum_of_squares * ratio * ratio;
            scale = magnitude;
         }
         else
         {
            const Number ratio = magnitude / scale;
            sum_of_squares += ratio * ratio;
         }
      }
      return scale == 0. ? 0. : scale * std::sqrt(sum_of_squares);
   }

   static Number Dot(
      std::span<const Number> left,
      std::span<const Number> right
   ) noexcept
   {
      Number result = 0.;
      for( Index i = 0; i < left.size(); ++i )
      {
         result = std::fma(left[i], right[i], result);
      }
      return result;
   }

   static void Axpy(
      Number                  alpha,
      std::span<const Number> input,
      std::span<Number>       output
   ) noexcept
   {
      for( Index i = 0; i < input.size(); ++i )
      {
         output[i] = std::fma(alpha, input[i], output[i]);
      }
   }

   static void ScaleCopy(
      Number                  alpha,
      std::span<const Number> input,
      std::span<Number>       output
   ) noexcept
   {
      for( Index i = 0; i < input.size(); ++i )
      {
         output[i] = alpha * input[i];
      }
   }

   static void Subtract(
      std::span<const Number> left,
      std::span<const Number> right,
      std::span<Number>       result
   ) noexcept
   {
      for( Index i = 0; i < left.size(); ++i )
      {
         result[i] = left[i] - right[i];
      }
   }

   std::span<Number> Basis(Index index) noexcept
   {
      return std::span<Number>(basis_).subspan(index * dimension_, dimension_);
   }

   std::span<const Number> Basis(Index index) const noexcept
   {
      return std::span<const Number>(basis_).subspan(index * dimension_, dimension_);
   }

   std::span<Number> PreconditionedBasis(Index index) noexcept
   {
      return std::span<Number>(preconditioned_basis_).subspan(
         index * dimension_, dimension_);
   }

   std::span<const Number> PreconditionedBasis(Index index) const noexcept
   {
      return std::span<const Number>(preconditioned_basis_).subspan(
         index * dimension_, dimension_);
   }

   Number& Hessenberg(Index row, Index column) noexcept
   {
      return hessenberg_[column * (options_.restart + 1) + row];
   }

   Number Hessenberg(Index row, Index column) const noexcept
   {
      return hessenberg_[column * (options_.restart + 1) + row];
   }

   void ResetCycleStorage()
   {
      std::ranges::fill(hessenberg_, 0.);
      std::ranges::fill(givens_cosines_, 0.);
      std::ranges::fill(givens_sines_, 0.);
      std::ranges::fill(projected_rhs_, 0.);
      std::ranges::fill(coefficients_, 0.);
   }

   void ApplyPreviousRotations(Index column) noexcept
   {
      for( Index row = 0; row < column; ++row )
      {
         const Number upper = Hessenberg(row, column);
         const Number lower = Hessenberg(row + 1, column);
         Hessenberg(row, column) =
            givens_cosines_[row] * upper + givens_sines_[row] * lower;
         Hessenberg(row + 1, column) =
            -givens_sines_[row] * upper + givens_cosines_[row] * lower;
      }
   }

   bool BuildCandidate(Index steps)
   {
      for( Index reverse = steps; reverse > 0; --reverse )
      {
         const Index row = reverse - 1;
         Number value = projected_rhs_[row];
         Number row_scale = 0.;
         for( Index column = row + 1; column < steps; ++column )
         {
            value = std::fma(-Hessenberg(row, column), coefficients_[column], value);
            row_scale = std::max(row_scale, std::abs(Hessenberg(row, column)));
         }
         const Number diagonal = Hessenberg(row, row);
         row_scale = std::max(row_scale, std::abs(diagonal));
         const Number minimum_diagonal = options_.breakdown_tolerance *
            std::max(row_scale, std::numeric_limits<Number>::min());
         if( !std::isfinite(diagonal) || std::abs(diagonal) <= minimum_diagonal )
         {
            return false;
         }
         coefficients_[row] = value / diagonal;
         if( !std::isfinite(coefficients_[row]) )
         {
            return false;
         }
      }

      std::ranges::copy(base_solution_, solution_.begin());
      for( Index index = 0; index < steps; ++index )
      {
         Axpy(coefficients_[index], PreconditionedBasis(index), solution_);
      }
      return !options_.check_finite || AllFinite(solution_);
   }

   const Index dimension_;
   const FgmresOptions options_;
   std::vector<Number> solution_;
   std::vector<Number> base_solution_;
   std::vector<Number> residual_;
   std::vector<Number> operator_output_;
   std::vector<Number> arnoldi_work_;
   std::vector<Number> basis_;
   std::vector<Number> preconditioned_basis_;
   std::vector<Number> hessenberg_;
   std::vector<Number> givens_cosines_;
   std::vector<Number> givens_sines_;
   std::vector<Number> projected_rhs_;
   std::vector<Number> coefficients_;
};
} // namespace Ipopt::Cxx23

#endif
