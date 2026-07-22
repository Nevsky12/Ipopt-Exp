// Copyright (C) 2026 Ipopt contributors.
// All Rights Reserved.
// This file is distributed under the Eclipse Public License.

#include <ipopt/cxx23/solve_session.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace
{
using namespace Ipopt::Cxx23;

void Check(bool condition, std::string_view message)
{
   if( !condition )
   {
      throw std::runtime_error(std::string(message));
   }
}

void CheckNear(Number actual, Number expected, Number tolerance, std::string_view message)
{
   if( std::abs(actual - expected) > tolerance )
   {
      throw std::runtime_error(
         std::string(message) + ": actual=" + std::to_string(actual) +
         " expected=" + std::to_string(expected));
   }
}

struct DenseThreeByThree
{
   std::array<Number, 9> values{4., 1., 0.,
                                1., 3., 1.,
                                0., 1., 2.};
   Index calls = 0;

   EvaluationResult operator()(std::span<const Number> input, std::span<Number> output)
   {
      ++calls;
      for( Index row = 0; row < 3; ++row )
      {
         output[row] = 0.;
         for( Index column = 0; column < 3; ++column )
         {
            output[row] += values[row * 3 + column] * input[column];
         }
      }
      return {};
   }
};

struct ChangingDiagonalPreconditioner
{
   Index calls = 0;

   EvaluationResult operator()(
      Index                   iteration,
      std::span<const Number> input,
      std::span<Number>       output
   )
   {
      ++calls;
      constexpr std::array<Number, 3> diagonal{4., 3., 2.};
      for( Index i = 0; i < input.size(); ++i )
      {
         const Number variation = 1. + 0.05 * static_cast<Number>((iteration + i) % 3);
         output[i] = input[i] / (diagonal[i] * variation);
      }
      return {};
   }
};

struct FailingIdentity
{
   Index calls = 0;
   Index fail_on_call = 0;
   bool nonfinite = false;

   EvaluationResult operator()(std::span<const Number> input, std::span<Number> output)
   {
      ++calls;
      if( calls == fail_on_call )
      {
         return std::unexpected(EvaluationError{
            EvaluationErrorCode::model_failure,
            "injected linear-operator failure"
         });
      }
      std::ranges::copy(input, output.begin());
      if( nonfinite && !output.empty() )
      {
         output[0] = std::numeric_limits<Number>::quiet_NaN();
      }
      return {};
   }
};

struct ZeroOperator
{
   EvaluationResult operator()(std::span<const Number>, std::span<Number> output) const
   {
      std::ranges::fill(output, 0.);
      return {};
   }
};

struct DenseMatrix
{
   Index dimension;
   std::vector<Number> values;

   EvaluationResult operator()(std::span<const Number> input, std::span<Number> output) const
   {
      for( Index row = 0; row < dimension; ++row )
      {
         output[row] = 0.;
         for( Index column = 0; column < dimension; ++column )
         {
            output[row] = std::fma(
               values[row * dimension + column], input[column], output[row]);
         }
      }
      return {};
   }
};

struct QuadraticKktModel
{
   std::uint64_t revision = 1;
   Number objective_scale = 1.;
   bool fail_hessian = false;

   NlpStructure structure() const
   {
      return {2, 1, 2, 2, revision};
   }

   EvaluationValue<Number> eval_objective(std::span<const Number> x)
   {
      return objective_scale * (x[0] * x[0] + 2. * x[1] * x[1]);
   }

   EvaluationResult eval_gradient(std::span<const Number> x, std::span<Number> result)
   {
      result[0] = 2. * objective_scale * x[0];
      result[1] = 4. * objective_scale * x[1];
      return {};
   }

   EvaluationResult eval_constraints(std::span<const Number> x, std::span<Number> result)
   {
      result[0] = x[0] + x[1];
      return {};
   }

   EvaluationResult eval_jacobian_structure(std::span<Index> rows, std::span<Index> columns)
   {
      rows[0] = 0;
      columns[0] = 0;
      rows[1] = 0;
      columns[1] = 1;
      return {};
   }

   EvaluationResult eval_jacobian_values(std::span<const Number>, std::span<Number> values)
   {
      values[0] = 1.;
      values[1] = 1.;
      return {};
   }

   EvaluationResult eval_hessian_structure(std::span<Index> rows, std::span<Index> columns)
   {
      rows[0] = 0;
      columns[0] = 0;
      rows[1] = 1;
      columns[1] = 1;
      return {};
   }

   EvaluationResult eval_hessian_values(
      std::span<const Number>,
      Number                  objective_factor,
      std::span<const Number>,
      std::span<Number>       values
   )
   {
      if( fail_hessian )
      {
         return std::unexpected(EvaluationError{
            EvaluationErrorCode::model_failure,
            "injected session Hessian failure"
         });
      }
      values[0] = 2. * objective_scale * objective_factor;
      values[1] = 4. * objective_scale * objective_factor;
      return {};
   }
};

void TestRestartedFlexibleSolveAndReuse()
{
   FgmresOptions options;
   options.restart = 2;
   options.maximum_iterations = 40;
   options.relative_tolerance = 1e-12;
   options.absolute_tolerance = 1e-14;
   FgmresSolver solver(3, options);

   DenseThreeByThree matrix;
   ChangingDiagonalPreconditioner preconditioner;
   const std::array<Number, 3> right_hand_side{6., 10., 8.};
   std::array<Number, 3> solution{};
   const EvaluationValue<FgmresResult> first = solver.solve(
      matrix, preconditioner, right_hand_side, solution);
   Check(first.has_value(), "flexible solve returned a callback error");
   Check(first->converged(), "restarted flexible solve did not converge");
   Check(first->iterations > options.restart, "restart path was not exercised");
   Check(first->preconditioner_evaluations == first->iterations, "preconditioner count is wrong");
   Check(first->operator_evaluations == matrix.calls, "operator count is wrong");
   CheckNear(solution[0], 1., 1e-10, "flexible solution component 0");
   CheckNear(solution[1], 2., 1e-10, "flexible solution component 1");
   CheckNear(solution[2], 3., 1e-10, "flexible solution component 2");

   const std::array<Number, 3> second_rhs{3., 1., 4.};
   std::array<Number, 3> second_solution{};
   const EvaluationValue<FgmresResult> second = solver.solve(
      matrix, preconditioner, second_rhs, second_solution);
   Check(second.has_value() && second->converged(), "reused solver did not converge");

   std::array<Number, 3> residual{};
   Check(matrix(second_solution, residual).has_value(), "reference residual evaluation failed");
   for( Index i = 0; i < residual.size(); ++i )
   {
      CheckNear(residual[i], second_rhs[i], 1e-10, "reused solver residual");
   }
}

void TestDenseReferenceFamily()
{
   for( Index dimension = 2; dimension <= 8; ++dimension )
   {
      DenseMatrix matrix{dimension, std::vector<Number>(dimension * dimension)};
      for( Index row = 0; row < dimension; ++row )
      {
         Number off_diagonal_sum = 0.;
         for( Index column = 0; column < dimension; ++column )
         {
            if( row == column )
            {
               continue;
            }
            const Number value = 0.05 * std::sin(
               static_cast<Number>((row + 1) * (column + 2)));
            matrix.values[row * dimension + column] = value;
            off_diagonal_sum += std::abs(value);
         }
         matrix.values[row * dimension + row] = 1.5 + off_diagonal_sum +
            0.1 * static_cast<Number>(row);
      }

      std::vector<Number> exact_solution(dimension);
      for( Index i = 0; i < dimension; ++i )
      {
         exact_solution[i] = std::cos(static_cast<Number>(i + 1));
      }
      std::vector<Number> right_hand_side(dimension);
      Check(
         matrix(exact_solution, right_hand_side).has_value(),
         "dense reference product failed");

      const auto changing_preconditioner = [&](
         Index iteration, std::span<const Number> input, std::span<Number> output)
         -> EvaluationResult
      {
         for( Index i = 0; i < dimension; ++i )
         {
            const Number variation = 1. +
               0.02 * static_cast<Number>((iteration + 2 * i) % 5);
            output[i] = input[i] /
               (matrix.values[i * dimension + i] * variation);
         }
         return {};
      };

      FgmresOptions options;
      options.restart = std::min<Index>(3, dimension);
      options.maximum_iterations = 100;
      options.relative_tolerance = 5e-12;
      options.absolute_tolerance = 1e-13;
      FgmresSolver solver(dimension, options);
      std::vector<Number> solution(dimension);
      const EvaluationValue<FgmresResult> result = solver.solve(
         matrix, changing_preconditioner, right_hand_side, solution);
      Check(result.has_value(), "dense reference family returned a callback error");
      Check(result->converged(), "dense reference family did not converge");
      for( Index i = 0; i < dimension; ++i )
      {
         CheckNear(solution[i], exact_solution[i], 1e-9, "dense reference solution");
      }
   }
}

void TestIterationLimitsAndBreakdown()
{
   const std::array<Number, 2> right_hand_side{1., 2.};
   IdentityPreconditioner identity_preconditioner;

   FgmresOptions zero_options;
   zero_options.restart = 2;
   zero_options.maximum_iterations = 0;
   zero_options.relative_tolerance = 1e-14;
   FgmresSolver zero_iteration_solver(2, zero_options);
   FailingIdentity identity;
   std::array<Number, 2> zero_solution{};
   const EvaluationValue<FgmresResult> limited = zero_iteration_solver.solve(
      identity, identity_preconditioner, right_hand_side, zero_solution);
   Check(limited.has_value(), "zero-iteration solve returned an error");
   Check(limited->status == FgmresStatus::maximum_iterations, "zero limit returned wrong status");
   Check(limited->iterations == 0, "zero limit performed an iteration");

   std::array<Number, 2> exact_solution{1., 2.};
   const EvaluationValue<FgmresResult> already_solved = zero_iteration_solver.solve(
      identity, identity_preconditioner, right_hand_side, exact_solution);
   Check(already_solved.has_value() && already_solved->converged(),
      "exact initial solution did not converge with zero limit");

   FgmresOptions exact_limit_options;
   exact_limit_options.restart = 1;
   exact_limit_options.maximum_iterations = 1;
   exact_limit_options.relative_tolerance = 1e-14;
   FgmresSolver exact_limit_solver(2, exact_limit_options);
   FailingIdentity exact_limit_identity;
   std::array<Number, 2> exact_limit_solution{};
   const EvaluationValue<FgmresResult> exact_limit = exact_limit_solver.solve(
      exact_limit_identity,
      identity_preconditioner,
      right_hand_side,
      exact_limit_solution);
   Check(exact_limit.has_value() && exact_limit->converged(),
      "convergence at the exact iteration limit was reported as failure");
   Check(exact_limit->iterations == 1, "exact-limit iteration count is wrong");

   FgmresOptions breakdown_options;
   breakdown_options.restart = 2;
   breakdown_options.maximum_iterations = 4;
   FgmresSolver breakdown_solver(2, breakdown_options);
   ZeroOperator zero_operator;
   std::array<Number, 2> breakdown_solution{};
   const EvaluationValue<FgmresResult> breakdown = breakdown_solver.solve(
      zero_operator, identity_preconditioner, right_hand_side, breakdown_solution);
   Check(breakdown.has_value(), "breakdown returned a callback error");
   Check(
      breakdown->status == FgmresStatus::numerical_breakdown,
      "zero operator did not report numerical breakdown");
   Check(breakdown->iterations == 1, "breakdown iteration count is wrong");

   FgmresOptions invalid_options;
   invalid_options.restart = 0;
   FgmresSolver invalid_solver(2, invalid_options);
   FailingIdentity unused_operator;
   std::array<Number, 2> invalid_solution{};
   const EvaluationValue<FgmresResult> invalid = invalid_solver.solve(
      unused_operator, identity_preconditioner, right_hand_side, invalid_solution);
   Check(invalid.has_value(), "invalid configuration returned a callback error");
   Check(
      invalid->status == FgmresStatus::invalid_configuration,
      "zero restart was accepted");
   Check(unused_operator.calls == 0, "invalid configuration called the operator");
}

void TestFailureAndNonfiniteGuarantees()
{
   FgmresOptions options;
   options.restart = 2;
   options.maximum_iterations = 4;
   options.relative_tolerance = 1e-14;
   FgmresSolver solver(2, options);
   IdentityPreconditioner identity_preconditioner;
   const std::array<Number, 2> right_hand_side{1., 2.};

   FailingIdentity failing;
   failing.fail_on_call = 3;
   std::array<Number, 2> solution{7., 8.};
   const EvaluationValue<FgmresResult> failure = solver.solve(
      failing, identity_preconditioner, right_hand_side, solution);
   Check(!failure.has_value(), "injected operator failure was ignored");
   Check(
      failure.error().code == EvaluationErrorCode::model_failure,
      "injected operator failure returned the wrong error");
   Check(
      solution == std::array<Number, 2>{7., 8.},
      "callback failure modified the caller's solution");

   FailingIdentity nonfinite;
   nonfinite.nonfinite = true;
   std::array<Number, 2> nonfinite_solution{3., 4.};
   const EvaluationValue<FgmresResult> nonfinite_result = solver.solve(
      nonfinite, identity_preconditioner, right_hand_side, nonfinite_solution);
   Check(nonfinite_result.has_value(), "nonfinite operator returned callback failure");
   Check(
      nonfinite_result->status == FgmresStatus::nonfinite_value,
      "nonfinite operator output was accepted");
}

void TestKktIntegration()
{
   NlpKktOperator kkt(MakeNlpProblem(QuadraticKktModel{}));
   const std::array<Number, 2> point_x{};
   const std::array<Number, 1> multipliers{};
   const std::array<Number, 2> primal_diagonal{0.5, 0.25};
   const std::array<Number, 1> dual_diagonal{0.1};
   const KktEvaluationPoint point{point_x, 1., multipliers};
   const KktDiagonal diagonal{primal_diagonal, dual_diagonal};

   const std::array<Number, 3> exact_solution{1., -2., 0.5};
   std::array<Number, 3> right_hand_side{};
   Check(
      kkt.apply(
         point,
         {
            std::span<const Number>(exact_solution).first<2>(),
            std::span<const Number>(exact_solution).last<1>()
         },
         diagonal,
         {
            std::span<Number>(right_hand_side).first<2>(),
            std::span<Number>(right_hand_side).last<1>()
         }).has_value(),
      "KKT reference product failed");

   const auto apply_kkt = [&](std::span<const Number> input, std::span<Number> output)
      -> EvaluationResult
   {
      return kkt.apply(
         point,
         {input.first<2>(), input.last<1>()},
         diagonal,
         {output.first<2>(), output.last<1>()});
   };

   FgmresOptions options;
   options.restart = 3;
   options.maximum_iterations = 6;
   options.relative_tolerance = 1e-13;
   options.absolute_tolerance = 1e-14;
   FgmresSolver solver(3, options);
   IdentityPreconditioner identity_preconditioner;
   std::array<Number, 3> solution{};
   const EvaluationValue<FgmresResult> result = solver.solve(
      apply_kkt, identity_preconditioner, right_hand_side, solution);
   Check(result.has_value(), "KKT FGMRES returned a callback error");
   Check(result->converged(), "KKT FGMRES did not converge");
   for( Index i = 0; i < solution.size(); ++i )
   {
      CheckNear(solution[i], exact_solution[i], 1e-11, "KKT FGMRES solution");
   }
}

void TestPrimalDualKktIntegration()
{
   PrimalDualKktOperator kkt(
      MakeNlpProblem(QuadraticKktModel{}),
      {
         .equality_constraints = {},
         .inequality_constraints = {0},
         .primal_lower_bounds = {0, 1},
         .primal_upper_bounds = {},
         .slack_lower_bounds = {0},
         .slack_upper_bounds = {}
      });
   Check(kkt.valid(), "FGMRES primal-dual layout is invalid");
   Check(kkt.flat_dimension() == 7, "FGMRES primal-dual dimension is wrong");

   const std::array<Number, 2> point_x{};
   const std::array<Number, 1> multipliers{};
   const std::array<Number, 2> z_lower{2., 3.};
   const std::array<Number, 0> z_upper{};
   const std::array<Number, 1> v_lower{4.};
   const std::array<Number, 0> v_upper{};
   const std::array<Number, 2> slack_x_lower{5., 6.};
   const std::array<Number, 0> slack_x_upper{};
   const std::array<Number, 1> slack_s_lower{7.};
   const std::array<Number, 0> slack_s_upper{};
   const PrimalDualState state{
      .nlp = {point_x, 1., multipliers},
      .z_lower = z_lower,
      .z_upper = z_upper,
      .v_lower = v_lower,
      .v_upper = v_upper,
      .slack_x_lower = slack_x_lower,
      .slack_x_upper = slack_x_upper,
      .slack_s_lower = slack_s_lower,
      .slack_s_upper = slack_s_upper,
      .regularization = {0.5, 0.25, 0.1, 0.1}
   };

   const std::array<Number, 7> exact_solution{
      1., -2., 0.5, 0.25, -0.75, 1.25, -1.5
   };
   std::array<Number, 7> right_hand_side{};
   Check(
      kkt.apply_flat(state, exact_solution, right_hand_side).has_value(),
      "primal-dual reference product failed");
   const auto apply_kkt = [&](std::span<const Number> input, std::span<Number> output)
      -> EvaluationResult
   {
      return kkt.apply_flat(state, input, output);
   };

   FgmresOptions options;
   options.restart = 7;
   options.maximum_iterations = 14;
   options.relative_tolerance = 1e-13;
   options.absolute_tolerance = 1e-14;
   FgmresSolver solver(7, options);
   IdentityPreconditioner identity_preconditioner;
   std::array<Number, 7> solution{};
   const EvaluationValue<FgmresResult> result = solver.solve(
      apply_kkt, identity_preconditioner, right_hand_side, solution);
   Check(result.has_value(), "primal-dual FGMRES returned a callback error");
   Check(result->converged(), "primal-dual FGMRES did not converge");
   for( Index i = 0; i < solution.size(); ++i )
   {
      CheckNear(solution[i], exact_solution[i], 1e-10, "primal-dual FGMRES solution");
   }
}

void TestSolveSessionReuseAndFingerprint()
{
   const PrimalDualLayout layout{
      .equality_constraints = {},
      .inequality_constraints = {0},
      .primal_lower_bounds = {0, 1},
      .primal_upper_bounds = {},
      .slack_lower_bounds = {0},
      .slack_upper_bounds = {}
   };
   PrimalDualKktOperator first(
      MakeNlpProblem(QuadraticKktModel{}), layout);

   const std::array<Number, 2> point_x{};
   const std::array<Number, 1> multipliers{};
   const std::array<Number, 2> z_lower{2., 3.};
   const std::array<Number, 0> z_upper{};
   const std::array<Number, 1> v_lower{4.};
   const std::array<Number, 0> v_upper{};
   const std::array<Number, 2> slack_x_lower{5., 6.};
   const std::array<Number, 0> slack_x_upper{};
   const std::array<Number, 1> slack_s_lower{7.};
   const std::array<Number, 0> slack_s_upper{};
   const PrimalDualState state{
      .nlp = {point_x, 1., multipliers},
      .z_lower = z_lower,
      .z_upper = z_upper,
      .v_lower = v_lower,
      .v_upper = v_upper,
      .slack_x_lower = slack_x_lower,
      .slack_x_upper = slack_x_upper,
      .slack_s_lower = slack_s_lower,
      .slack_s_upper = slack_s_upper,
      .regularization = {0.5, 0.25, 0.1, 0.1}
   };

   FgmresOptions options;
   options.restart = 7;
   options.maximum_iterations = 14;
   options.relative_tolerance = 1e-13;
   options.absolute_tolerance = 1e-14;
   EvaluationValue<PrimalDualSolveSession> created =
      MakePrimalDualSolveSession(first, options);
   Check(created.has_value(), "solve-session construction failed");
   PrimalDualSolveSession session = std::move(*created);
   Check(session.dimension() == 7, "solve-session dimension is wrong");
   Check(session.options().restart == 7, "solve-session options were not retained");

   const std::array<Number, 7> first_exact{
      1., -2., 0.5, 0.25, -0.75, 1.25, -1.5
   };
   std::array<Number, 7> first_rhs{};
   Check(
      first.apply_flat(state, first_exact, first_rhs).has_value(),
      "first solve-session reference product failed");
   std::array<Number, 7> first_solution{};
   const EvaluationValue<FgmresResult> first_result = session.solve(
      first, state, first_rhs, first_solution);
   Check(
      first_result.has_value() && first_result->converged(),
      "first solve-session solve did not converge");
   for( Index i = 0; i < first_solution.size(); ++i )
   {
      CheckNear(first_solution[i], first_exact[i], 1e-10, "first session solution");
   }

   PrimalDualKktOperator changed_values(
      MakeNlpProblem(QuadraticKktModel{.revision = 1, .objective_scale = 2.}),
      layout);
   const EvaluationValue<bool> compatible = session.compatible(changed_values);
   Check(
      compatible.has_value() && *compatible,
      "numeric-only model change was treated as structural");
   const std::array<Number, 7> second_exact{
      -0.5, 1.25, -1., 0.75, 1.5, -0.25, 2.
   };
   std::array<Number, 7> second_rhs{};
   Check(
      changed_values.apply_flat(state, second_exact, second_rhs).has_value(),
      "second solve-session reference product failed");
   std::array<Number, 7> second_solution{};
   const EvaluationValue<FgmresResult> second_result = session.solve(
      changed_values, state, second_rhs, second_solution);
   Check(
      second_result.has_value() && second_result->converged(),
      "reused solve session did not converge");
   for( Index i = 0; i < second_solution.size(); ++i )
   {
      CheckNear(second_solution[i], second_exact[i], 1e-10, "reused session solution");
   }

   PrimalDualKktOperator changed_structure(
      MakeNlpProblem(QuadraticKktModel{.revision = 2}), layout);
   std::array<Number, 7> rejected_solution{
      11., 12., 13., 14., 15., 16., 17.
   };
   const std::array<Number, 7> rejected_before = rejected_solution;
   const EvaluationValue<FgmresResult> rejected = session.solve(
      changed_structure, state, second_rhs, rejected_solution);
   Check(!rejected.has_value(), "structure mismatch was accepted by solve session");
   Check(
      rejected.error().code == EvaluationErrorCode::structure_mismatch,
      "structure mismatch returned the wrong error code");
   Check(
      rejected_solution == rejected_before,
      "structure mismatch modified the caller's solution");

   PrimalDualKktOperator failing(
      MakeNlpProblem(QuadraticKktModel{
         .revision = 1,
         .objective_scale = 1.,
         .fail_hessian = true
      }),
      layout);
   std::array<Number, 7> failed_solution{
      21., 22., 23., 24., 25., 26., 27.
   };
   const std::array<Number, 7> failed_before = failed_solution;
   const EvaluationValue<FgmresResult> failed = session.solve(
      failing, state, first_rhs, failed_solution);
   Check(!failed.has_value(), "solve-session callback failure was ignored");
   Check(
      failed.error().code == EvaluationErrorCode::model_failure,
      "solve-session callback returned the wrong error code");
   Check(
      failed_solution == failed_before,
      "solve-session callback failure modified the solution");

   const SolveSessionStatistics& statistics = session.statistics();
   Check(statistics.solve_requests == 4, "solve-session request count is wrong");
   Check(statistics.converged_solves == 2, "solve-session convergence count is wrong");
   Check(
      statistics.structure_rejections == 1,
      "solve-session structure rejection count is wrong");
   Check(
      statistics.evaluation_failures == 1,
      "solve-session evaluation failure count is wrong");

   PrimalDualKktOperator invalid(
      MakeNlpProblem(QuadraticKktModel{}),
      {
         .equality_constraints = {0},
         .inequality_constraints = {0},
         .primal_lower_bounds = {},
         .primal_upper_bounds = {},
         .slack_lower_bounds = {},
         .slack_upper_bounds = {}
      });
   const EvaluationValue<PrimalDualSolveSession> invalid_session =
      MakePrimalDualSolveSession(invalid, options);
   Check(!invalid_session.has_value(), "invalid layout produced a solve session");
   Check(
      invalid_session.error().code == EvaluationErrorCode::invalid_layout,
      "invalid solve-session layout returned the wrong error code");
}
} // namespace

int main()
{
   try
   {
      TestRestartedFlexibleSolveAndReuse();
      TestDenseReferenceFamily();
      TestIterationLimitsAndBreakdown();
      TestFailureAndNonfiniteGuarantees();
      TestKktIntegration();
      TestPrimalDualKktIntegration();
      TestSolveSessionReuseAndFingerprint();
   }
   catch( const std::exception& error )
   {
      std::cerr << "ipopt_cxx23_fgmres_test: " << error.what() << '\n';
      return EXIT_FAILURE;
   }
   std::cout << "ipopt_cxx23_fgmres_test: passed\n";
   return EXIT_SUCCESS;
}
