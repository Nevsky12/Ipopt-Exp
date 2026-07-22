// Copyright (C) 2026 Ipopt contributors.
// All Rights Reserved.
// This file is distributed under the Eclipse Public License.

#include <ipopt/cxx23/direct_preconditioner.hpp>
#include <ipopt/cxx23/solve_session.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace
{
using namespace Ipopt::Cxx23;

struct TinyKktModel
{
   NlpStructure structure() const
   {
      return {2, 1, 2, 3, 31};
   }

   EvaluationValue<Number> eval_objective(std::span<const Number>)
   {
      return 0.;
   }

   EvaluationResult eval_gradient(std::span<const Number>, std::span<Number> result)
   {
      std::ranges::fill(result, 0.);
      return {};
   }

   EvaluationResult eval_constraints(
      std::span<const Number> x,
      std::span<Number>       result
   )
   {
      result[0] = x[0] + 2. * x[1];
      return {};
   }

   EvaluationResult eval_jacobian_structure(
      std::span<Index> rows,
      std::span<Index> columns
   )
   {
      rows[0] = 0;
      columns[0] = 0;
      rows[1] = 0;
      columns[1] = 1;
      return {};
   }

   EvaluationResult eval_jacobian_values(
      std::span<const Number>,
      std::span<Number> values
   )
   {
      values[0] = 1.;
      values[1] = 2.;
      return {};
   }

   EvaluationResult eval_hessian_structure(
      std::span<Index> rows,
      std::span<Index> columns
   )
   {
      rows[0] = 0;
      columns[0] = 0;
      rows[1] = 1;
      columns[1] = 0;
      rows[2] = 1;
      columns[2] = 1;
      return {};
   }

   EvaluationResult eval_hessian_values(
      std::span<const Number>,
      Number objective_factor,
      std::span<const Number>,
      std::span<Number> values
   )
   {
      values[0] = 4. * objective_factor;
      values[1] = objective_factor;
      values[2] = 3. * objective_factor;
      return {};
   }

   EvaluationResult eval_jacobian_product(
      std::span<const Number>,
      std::span<const Number> direction,
      std::span<Number>       result
   )
   {
      result[0] = direction[0] + 2. * direction[1];
      return {};
   }

   EvaluationResult eval_jacobian_transpose_product(
      std::span<const Number>,
      std::span<const Number> direction,
      std::span<Number>       result
   )
   {
      result[0] = direction[0];
      result[1] = 2. * direction[0];
      return {};
   }

   EvaluationResult eval_hessian_product(
      std::span<const Number>,
      Number objective_factor,
      std::span<const Number>,
      std::span<const Number> direction,
      std::span<Number>       result
   )
   {
      result[0] = objective_factor * (4. * direction[0] + direction[1]);
      result[1] = objective_factor * (direction[0] + 3. * direction[1]);
      return {};
   }
};

struct BackendCounters
{
   Index factorization_calls = 0;
   Index solve_calls = 0;
   bool fail_factorization = false;
   bool fail_solve = false;
   bool return_nonfinite = false;
};

class DenseDirectBackend
{
public:
   DenseDirectBackend(
      std::array<Number, 9>     matrix,
      StructureFingerprint     fingerprint,
      std::uint64_t            numeric_revision,
      std::shared_ptr<BackendCounters> counters,
      Index                    reported_dimension = 3
   )
      : matrix_(matrix),
        fingerprint_(fingerprint),
        numeric_revision_(numeric_revision),
        counters_(std::move(counters)),
        reported_dimension_(reported_dimension)
   {
   }

   Index dimension() const noexcept
   {
      return reported_dimension_;
   }

   StructureFingerprint structure_fingerprint() const noexcept
   {
      return fingerprint_;
   }

   std::uint64_t numeric_revision() const noexcept
   {
      return numeric_revision_;
   }

   EvaluationResult factorize()
   {
      ++counters_->factorization_calls;
      if( counters_->fail_factorization )
      {
         return Failure("injected factorization failure");
      }
      lu_ = matrix_;
      for( Index column = 0; column < 3; ++column )
      {
         Index pivot = column;
         for( Index row = column + 1; row < 3; ++row )
         {
            if( std::abs(lu_[row * 3 + column]) > std::abs(lu_[pivot * 3 + column]) )
            {
               pivot = row;
            }
         }
         if( std::abs(lu_[pivot * 3 + column]) < 1e-14 )
         {
            return Failure("dense reference factorization is singular");
         }
         pivots_[column] = pivot;
         if( pivot != column )
         {
            for( Index entry = 0; entry < 3; ++entry )
            {
               std::swap(lu_[column * 3 + entry], lu_[pivot * 3 + entry]);
            }
         }
         for( Index row = column + 1; row < 3; ++row )
         {
            lu_[row * 3 + column] /= lu_[column * 3 + column];
            for( Index entry = column + 1; entry < 3; ++entry )
            {
               lu_[row * 3 + entry] -=
                  lu_[row * 3 + column] * lu_[column * 3 + entry];
            }
         }
      }
      factorized_ = true;
      return {};
   }

   EvaluationResult solve_rhs(
      std::span<const Number> input,
      std::span<Number>       output
   )
   {
      ++counters_->solve_calls;
      output[0] = 12345.;
      if( counters_->fail_solve )
      {
         return Failure("injected solve failure");
      }
      if( !factorized_ )
      {
         return Failure("solve_rhs called before factorize");
      }
      std::ranges::copy(input, output.begin());
      for( Index column = 0; column < 3; ++column )
      {
         if( pivots_[column] != column )
         {
            std::swap(output[column], output[pivots_[column]]);
         }
      }
      for( Index row = 1; row < 3; ++row )
      {
         for( Index column = 0; column < row; ++column )
         {
            output[row] -= lu_[row * 3 + column] * output[column];
         }
      }
      for( Index reverse = 0; reverse < 3; ++reverse )
      {
         const Index row = 2 - reverse;
         for( Index column = row + 1; column < 3; ++column )
         {
            output[row] -= lu_[row * 3 + column] * output[column];
         }
         output[row] /= lu_[row * 3 + row];
      }
      if( counters_->return_nonfinite )
      {
         output[0] = std::numeric_limits<Number>::quiet_NaN();
      }
      return {};
   }

private:
   static EvaluationResult Failure(std::string message)
   {
      return std::unexpected(EvaluationError{
         EvaluationErrorCode::model_failure,
         std::move(message)
      });
   }

   std::array<Number, 9> matrix_;
   std::array<Number, 9> lu_{};
   std::array<Index, 3> pivots_{};
   StructureFingerprint fingerprint_;
   std::uint64_t numeric_revision_;
   std::shared_ptr<BackendCounters> counters_;
   Index reported_dimension_;
   bool factorized_ = false;
};

void Check(bool condition, std::string_view message)
{
   if( !condition )
   {
      throw std::runtime_error(std::string(message));
   }
}

void CheckNear(Number actual, Number expected, std::string_view message)
{
   if( std::abs(actual - expected) > 1e-11 )
   {
      throw std::runtime_error(
         std::string(message) + ": actual=" + std::to_string(actual) +
         " expected=" + std::to_string(expected));
   }
}

void CheckNear(
   const std::array<Number, 3>& actual,
   const std::array<Number, 3>& expected,
   std::string_view             message
)
{
   for( Index i = 0; i < actual.size(); ++i )
   {
      CheckNear(actual[i], expected[i], message);
   }
}

PrimalDualState State(
   const std::array<Number, 2>& x,
   const std::array<Number, 1>& multipliers,
   std::uint64_t                numeric_revision
)
{
   return {
      .nlp = {x, 1., multipliers},
      .z_lower = {},
      .z_upper = {},
      .v_lower = {},
      .v_upper = {},
      .slack_x_lower = {},
      .slack_x_upper = {},
      .slack_s_lower = {},
      .slack_s_upper = {},
      .regularization = {.5, 0., .25, 0.},
      .numeric_revision = numeric_revision
   };
}

constexpr std::array<Number, 9> kkt_matrix{{
   4.5, 1., 1.,
   1., 3.5, 2.,
   1., 2., -.25
}};

void TestPreparedSolveManyLifecycle()
{
   PrimalDualKktOperator kkt(
      MakeNlpProblem(TinyKktModel{}),
      {
         .equality_constraints = {0},
         .inequality_constraints = {},
         .primal_lower_bounds = {},
         .primal_upper_bounds = {},
         .slack_lower_bounds = {},
         .slack_upper_bounds = {}
      });
   Check(kkt.valid(), "tiny KKT layout was rejected");
   const StructureFingerprintResult fingerprint = kkt.structure_fingerprint();
   Check(fingerprint.has_value(), "tiny KKT fingerprint failed");

   const std::array<Number, 2> x{{0., 0.}};
   const std::array<Number, 1> multipliers{{0.}};
   const PrimalDualState state = State(x, multipliers, 42);
   const auto backend_counters = std::make_shared<BackendCounters>();
   EvaluationValue<PreparedDirectPreconditioner> prepared_result =
      PrepareDirectPreconditioner(
         kkt,
         state,
         MakeDirectSolverBackend(DenseDirectBackend{
            kkt_matrix, *fingerprint, state.numeric_revision, backend_counters
         }));
   Check(prepared_result.has_value(), "valid direct preconditioner was rejected");
   PreparedDirectPreconditioner preconditioner = std::move(*prepared_result);
   Check(backend_counters->factorization_calls == 1, "backend was not factorized once");

   FgmresOptions options;
   options.restart = 3;
   options.maximum_iterations = 3;
   options.relative_tolerance = 1e-12;
   options.absolute_tolerance = 1e-14;
   EvaluationValue<PrimalDualSolveSession> session_result =
      MakePrimalDualSolveSession(kkt, options);
   Check(session_result.has_value(), "direct-preconditioned solve session failed");
   PrimalDualSolveSession session = std::move(*session_result);

   const std::array<Number, 3> exact{{1., -2., .5}};
   std::array<Number, 3> right_hand_side{};
   Check(kkt.apply_flat(state, exact, right_hand_side).has_value(), "KKT RHS failed");
   std::array<Number, 3> solution{};
   EvaluationValue<FgmresResult> first = session.solve(
      kkt, state, preconditioner, right_hand_side, solution);
   Check(first.has_value() && first->converged(), "direct-preconditioned solve failed");
   Check(first->iterations == 1, "exact direct preconditioner did not converge in one step");
   CheckNear(solution, exact, "first direct-preconditioned solution");

   const std::array<Number, 3> second_exact{{-.25, 3., 2.}};
   Check(
      kkt.apply_flat(state, second_exact, right_hand_side).has_value(),
      "second KKT RHS failed");
   std::ranges::fill(solution, 0.);
   EvaluationValue<FgmresResult> second = session.solve(
      kkt, state, preconditioner, right_hand_side, solution);
   Check(second.has_value() && second->converged(), "second direct-preconditioned solve failed");
   CheckNear(solution, second_exact, "second direct-preconditioned solution");
   Check(backend_counters->factorization_calls == 1, "solve_rhs refactorized the backend");
   Check(backend_counters->solve_calls == 2, "backend solve_rhs count is wrong");
   Check(
      preconditioner.statistics().factorizations == 1 &&
         preconditioner.statistics().successful_solves == 2,
      "direct preconditioner statistics are wrong");

   std::array<Number, 3> aliased = right_hand_side;
   Check(
      preconditioner(7, std::span<const Number>(aliased), aliased).has_value(),
      "aliased direct preconditioner solve failed");
   CheckNear(aliased, second_exact, "aliased direct preconditioner solution");

   PrimalDualState stale_state = state;
   stale_state.numeric_revision = 43;
   std::array<Number, 3> untouched{{101., 102., 103.}};
   const Index solves_before_rejection = backend_counters->solve_calls;
   EvaluationValue<FgmresResult> stale = session.solve(
      kkt, stale_state, preconditioner, right_hand_side, untouched);
   Check(!stale.has_value(), "stale numeric factorization was accepted");
   Check(
      stale.error().code == EvaluationErrorCode::numeric_mismatch,
      "stale factorization returned the wrong error");
   Check(
      untouched == std::array<Number, 3>{{101., 102., 103.}},
      "numeric mismatch modified the solution");
   Check(
      backend_counters->solve_calls == solves_before_rejection,
      "numeric mismatch entered the direct backend");
   Check(
      session.statistics().preconditioner_rejections == 1,
      "session did not count the preconditioner rejection");

   backend_counters->fail_solve = true;
   std::array<Number, 3> failed_output{{104., 105., 106.}};
   EvaluationResult failure = preconditioner(8, right_hand_side, failed_output);
   Check(!failure.has_value(), "direct backend solve failure was ignored");
   Check(
      failed_output == std::array<Number, 3>{{104., 105., 106.}},
      "failed direct solve modified its output");
   backend_counters->fail_solve = false;
   backend_counters->return_nonfinite = true;
   EvaluationResult nonfinite = preconditioner(9, right_hand_side, failed_output);
   Check(!nonfinite.has_value(), "nonfinite direct solution was accepted");
   Check(
      nonfinite.error().code == EvaluationErrorCode::nonfinite_output,
      "nonfinite direct solution returned the wrong error");
   Check(
      failed_output == std::array<Number, 3>{{104., 105., 106.}},
      "nonfinite direct solve modified its output");
}

void TestPreparationValidation()
{
   PrimalDualKktOperator kkt(
      MakeNlpProblem(TinyKktModel{}),
      {
         .equality_constraints = {0},
         .inequality_constraints = {},
         .primal_lower_bounds = {},
         .primal_upper_bounds = {},
         .slack_lower_bounds = {},
         .slack_upper_bounds = {}
      });
   const StructureFingerprintResult fingerprint = kkt.structure_fingerprint();
   Check(fingerprint.has_value(), "validation fixture fingerprint failed");
   const std::array<Number, 2> x{{0., 0.}};
   const std::array<Number, 1> multipliers{{0.}};
   const PrimalDualState state = State(x, multipliers, 42);

   StructureFingerprint wrong_fingerprint = *fingerprint;
   ++wrong_fingerprint.low;
   const auto structure_counters = std::make_shared<BackendCounters>();
   EvaluationValue<PreparedDirectPreconditioner> wrong_structure =
      PrepareDirectPreconditioner(
         kkt,
         state,
         MakeDirectSolverBackend(DenseDirectBackend{
            kkt_matrix, wrong_fingerprint, 42, structure_counters
         }));
   Check(!wrong_structure.has_value(), "wrong direct structure was accepted");
   Check(
      wrong_structure.error().code == EvaluationErrorCode::structure_mismatch,
      "wrong direct structure returned the wrong error");
   Check(structure_counters->factorization_calls == 0, "wrong structure was factorized");

   const auto dimension_counters = std::make_shared<BackendCounters>();
   EvaluationValue<PreparedDirectPreconditioner> wrong_dimension =
      PrepareDirectPreconditioner(
         kkt,
         state,
         MakeDirectSolverBackend(DenseDirectBackend{
            kkt_matrix, *fingerprint, 42, dimension_counters, 2
         }));
   Check(!wrong_dimension.has_value(), "wrong direct dimension was accepted");
   Check(
      wrong_dimension.error().code == EvaluationErrorCode::dimension_mismatch,
      "wrong direct dimension returned the wrong error");
   Check(dimension_counters->factorization_calls == 0, "wrong dimension was factorized");

   const auto numeric_counters = std::make_shared<BackendCounters>();
   EvaluationValue<PreparedDirectPreconditioner> wrong_numeric =
      PrepareDirectPreconditioner(
         kkt,
         state,
         MakeDirectSolverBackend(DenseDirectBackend{
            kkt_matrix, *fingerprint, 41, numeric_counters
         }));
   Check(!wrong_numeric.has_value(), "wrong numeric revision was accepted");
   Check(
      wrong_numeric.error().code == EvaluationErrorCode::numeric_mismatch,
      "wrong numeric revision returned the wrong error");
   Check(numeric_counters->factorization_calls == 0, "wrong numeric revision was factorized");

   const auto zero_counters = std::make_shared<BackendCounters>();
   const PrimalDualState unversioned_state = State(x, multipliers, 0);
   EvaluationValue<PreparedDirectPreconditioner> unversioned =
      PrepareDirectPreconditioner(
         kkt,
         unversioned_state,
         MakeDirectSolverBackend(DenseDirectBackend{
            kkt_matrix, *fingerprint, 0, zero_counters
         }));
   Check(!unversioned.has_value(), "unversioned numeric state was accepted");
   Check(
      unversioned.error().code == EvaluationErrorCode::numeric_mismatch,
      "unversioned state returned the wrong error");
   Check(zero_counters->factorization_calls == 0, "unversioned state was factorized");

   const auto invalid_state_counters = std::make_shared<BackendCounters>();
   PrimalDualState invalid_state = state;
   invalid_state.nlp.x = {};
   EvaluationValue<PreparedDirectPreconditioner> invalid_state_result =
      PrepareDirectPreconditioner(
         kkt,
         invalid_state,
         MakeDirectSolverBackend(DenseDirectBackend{
            kkt_matrix, *fingerprint, 42, invalid_state_counters
         }));
   Check(!invalid_state_result.has_value(), "invalid KKT state was accepted");
   Check(
      invalid_state_result.error().code == EvaluationErrorCode::dimension_mismatch,
      "invalid KKT state returned the wrong error");
   Check(
      invalid_state_counters->factorization_calls == 0,
      "invalid KKT state was factorized");

   const auto failing_counters = std::make_shared<BackendCounters>();
   failing_counters->fail_factorization = true;
   EvaluationValue<PreparedDirectPreconditioner> failing =
      PrepareDirectPreconditioner(
         kkt,
         state,
         MakeDirectSolverBackend(DenseDirectBackend{
            kkt_matrix, *fingerprint, 42, failing_counters
         }));
   Check(!failing.has_value(), "factorization failure was ignored");
   Check(
      failing.error().code == EvaluationErrorCode::model_failure,
      "factorization failure returned the wrong error");
   Check(failing_counters->factorization_calls == 1, "factorization was not attempted once");
}
} // namespace

int main()
{
   try
   {
      TestPreparedSolveManyLifecycle();
      TestPreparationValidation();
      std::cout << "direct preconditioner tests passed\n";
      return 0;
   }
   catch( const std::exception& error )
   {
      std::cerr << error.what() << '\n';
      return 1;
   }
}
