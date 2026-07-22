// Copyright (C) 2026 Ipopt contributors.
// All Rights Reserved.
// This file is distributed under the Eclipse Public License.

#include <ipopt/cxx23/legacy_coordinate_adapter.hpp>
#include <ipopt/cxx23/kkt_operator.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace
{
using namespace Ipopt::Cxx23;

constexpr Index full_variables = 4;
constexpr Index full_constraints = 3;

struct Counters
{
   int gradient = 0;
   int jacobian_values = 0;
   int hessian_values = 0;
   int jacobian_products = 0;
   int jacobian_transpose_products = 0;
   int hessian_products = 0;
   bool fail_gradient = false;
   bool fail_jacobian_product = false;
   bool fail_hessian_product = false;
};

constexpr std::array<std::array<Number, full_variables>, full_variables> objective_hessian{{
   {{2., 1., 0., 2.}},
   {{1., 4., 3., 0.}},
   {{0., 3., 6., 1.}},
   {{2., 0., 1., 8.}}
}};
constexpr std::array<Number, full_variables> objective_linear{{1., -1., 2., -2.}};

std::array<Number, full_variables * full_variables> DenseHessian(
   Number                  objective_factor,
   std::span<const Number> multipliers
)
{
   std::array<Number, full_variables * full_variables> result{};
   for( Index row = 0; row < full_variables; ++row )
   {
      for( Index column = 0; column < full_variables; ++column )
      {
         result[row * full_variables + column] =
            objective_factor * objective_hessian[row][column];
      }
   }
   result[0 * full_variables + 1] += multipliers[0];
   result[1 * full_variables + 0] += multipliers[0];
   result[1 * full_variables + 2] += multipliers[1];
   result[2 * full_variables + 1] += multipliers[1];
   result[3 * full_variables + 3] += 2. * multipliers[1];
   result[0 * full_variables + 0] += 2. * multipliers[2];
   result[2 * full_variables + 3] += multipliers[2];
   result[3 * full_variables + 2] += multipliers[2];
   return result;
}

std::array<Number, full_constraints * full_variables> DenseJacobian(
   std::span<const Number> x
)
{
   return {
      x[1], x[0], 1., 2.,
      0., x[2], x[1], 2. * x[3],
      2. * x[0], 0., x[3], x[2]
   };
}

struct FullModel
{
   explicit FullModel(std::shared_ptr<Counters> counters, std::uint64_t revision = 19)
      : counters(std::move(counters)),
        revision(revision)
   {
   }

   NlpStructure structure() const
   {
      return {full_variables, full_constraints, 10, 8, revision};
   }

   EvaluationValue<Number> eval_objective(std::span<const Number> x)
   {
      Number result = 0.;
      for( Index row = 0; row < full_variables; ++row )
      {
         result += objective_linear[row] * x[row];
         for( Index column = 0; column < full_variables; ++column )
         {
            result += .5 * x[row] * objective_hessian[row][column] * x[column];
         }
      }
      return result;
   }

   EvaluationResult eval_gradient(
      std::span<const Number> x,
      std::span<Number>       result
   )
   {
      ++counters->gradient;
      result[0] = 12345.;
      if( counters->fail_gradient )
      {
         return Failure("gradient failure");
      }
      for( Index row = 0; row < full_variables; ++row )
      {
         result[row] = objective_linear[row];
         for( Index column = 0; column < full_variables; ++column )
         {
            result[row] += objective_hessian[row][column] * x[column];
         }
      }
      return {};
   }

   EvaluationResult eval_constraints(
      std::span<const Number> x,
      std::span<Number>       result
   )
   {
      result[0] = x[0] * x[1] + x[2] + 2. * x[3];
      result[1] = x[1] * x[2] + x[3] * x[3];
      result[2] = x[0] * x[0] + x[2] * x[3];
      return {};
   }

   EvaluationResult eval_jacobian_structure(
      std::span<Index> rows,
      std::span<Index> columns
   )
   {
      constexpr std::array<Index, 10> expected_rows{{0, 0, 0, 0, 1, 1, 1, 2, 2, 2}};
      constexpr std::array<Index, 10> expected_columns{{0, 1, 2, 3, 1, 2, 3, 0, 2, 3}};
      std::ranges::copy(expected_rows, rows.begin());
      std::ranges::copy(expected_columns, columns.begin());
      return {};
   }

   EvaluationResult eval_jacobian_values(
      std::span<const Number> x,
      std::span<Number>       values
   )
   {
      ++counters->jacobian_values;
      const std::array<Number, full_constraints * full_variables> dense = DenseJacobian(x);
      constexpr std::array<Index, 10> rows{{0, 0, 0, 0, 1, 1, 1, 2, 2, 2}};
      constexpr std::array<Index, 10> columns{{0, 1, 2, 3, 1, 2, 3, 0, 2, 3}};
      for( Index i = 0; i < values.size(); ++i )
      {
         values[i] = dense[rows[i] * full_variables + columns[i]];
      }
      return {};
   }

   EvaluationResult eval_hessian_structure(
      std::span<Index> rows,
      std::span<Index> columns
   )
   {
      constexpr std::array<Index, 8> expected_rows{{0, 1, 1, 2, 2, 3, 3, 3}};
      constexpr std::array<Index, 8> expected_columns{{0, 0, 1, 1, 2, 0, 2, 3}};
      std::ranges::copy(expected_rows, rows.begin());
      std::ranges::copy(expected_columns, columns.begin());
      return {};
   }

   EvaluationResult eval_hessian_values(
      std::span<const Number>,
      Number                  objective_factor,
      std::span<const Number> multipliers,
      std::span<Number>       values
   )
   {
      ++counters->hessian_values;
      const std::array<Number, full_variables * full_variables> dense =
         DenseHessian(objective_factor, multipliers);
      constexpr std::array<Index, 8> rows{{0, 1, 1, 2, 2, 3, 3, 3}};
      constexpr std::array<Index, 8> columns{{0, 0, 1, 1, 2, 0, 2, 3}};
      for( Index i = 0; i < values.size(); ++i )
      {
         values[i] = dense[rows[i] * full_variables + columns[i]];
      }
      return {};
   }

   static EvaluationResult Failure(std::string message)
   {
      return std::unexpected(EvaluationError{
         EvaluationErrorCode::model_failure,
         std::move(message)
      });
   }

   std::shared_ptr<Counters> counters;
   std::uint64_t revision;
};

struct FullNativeModel : FullModel
{
   using FullModel::FullModel;

   EvaluationResult eval_jacobian_product(
      std::span<const Number> x,
      std::span<const Number> direction,
      std::span<Number>       result
   )
   {
      ++counters->jacobian_products;
      result[0] = 12345.;
      if( counters->fail_jacobian_product )
      {
         return Failure("Jacobian product failure");
      }
      const std::array<Number, full_constraints * full_variables> dense = DenseJacobian(x);
      for( Index row = 0; row < full_constraints; ++row )
      {
         result[row] = 0.;
         for( Index column = 0; column < full_variables; ++column )
         {
            result[row] += dense[row * full_variables + column] * direction[column];
         }
      }
      return {};
   }

   EvaluationResult eval_jacobian_transpose_product(
      std::span<const Number> x,
      std::span<const Number> direction,
      std::span<Number>       result
   )
   {
      ++counters->jacobian_transpose_products;
      const std::array<Number, full_constraints * full_variables> dense = DenseJacobian(x);
      for( Index column = 0; column < full_variables; ++column )
      {
         result[column] = 0.;
         for( Index row = 0; row < full_constraints; ++row )
         {
            result[column] += dense[row * full_variables + column] * direction[row];
         }
      }
      return {};
   }

   EvaluationResult eval_hessian_product(
      std::span<const Number>,
      Number                  objective_factor,
      std::span<const Number> multipliers,
      std::span<const Number> direction,
      std::span<Number>       result
   )
   {
      ++counters->hessian_products;
      result[0] = 12345.;
      if( counters->fail_hessian_product )
      {
         return Failure("Hessian product failure");
      }
      const std::array<Number, full_variables * full_variables> dense =
         DenseHessian(objective_factor, multipliers);
      for( Index row = 0; row < full_variables; ++row )
      {
         result[row] = 0.;
         for( Index column = 0; column < full_variables; ++column )
         {
            result[row] += dense[row * full_variables + column] * direction[column];
         }
      }
      return {};
   }
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
   if( std::abs(actual - expected) > 1e-12 )
   {
      throw std::runtime_error(
         std::string(message) + ": actual=" + std::to_string(actual) +
         " expected=" + std::to_string(expected));
   }
}

template <std::size_t Size>
void CheckNear(
   const std::array<Number, Size>& actual,
   const std::array<Number, Size>& expected,
   std::string_view                message
)
{
   for( Index i = 0; i < Size; ++i )
   {
      CheckNear(actual[i], expected[i], message);
   }
}

LegacyCoordinateMap CoordinateMap(Number fixed_value = 5., Number first_rhs = 7.)
{
   return {
      .internal_to_full_variables = {3, 0, 2},
      .full_variable_template = {0., fixed_value, 0., 0.},
      .equality_from_full_constraints = {2, 0},
      .equality_rhs = {first_rhs, 11.},
      .fixed_variable_equalities = {{0, 4.}},
      .inequality_from_full_constraints = {1}
   };
}

LegacyNlpScaling Scaling()
{
   return {
      .objective = -2.,
      .variables = {2., 4., 5.},
      .equalities = {3., 7., 11.},
      .inequalities = {13.}
   };
}

AnyNlpProblem AdaptNative(const std::shared_ptr<Counters>& counters)
{
   EvaluationValue<AnyNlpProblem> adapted = MakeLegacyCoordinateProblem(
      MakeNlpProblem(FullNativeModel{counters}), CoordinateMap(), Scaling());
   Check(adapted.has_value(), "valid coordinate adapter was rejected");
   return std::move(*adapted);
}

void TestAllTransformations()
{
   const auto counters = std::make_shared<Counters>();
   AnyNlpProblem problem = AdaptNative(counters);
   Check(problem.nlp_structure().variables == 3, "adapted variable dimension is wrong");
   Check(problem.nlp_structure().constraints == 4, "adapted constraint dimension is wrong");
   Check(problem.nlp_structure().jacobian_nonzeros == 9, "adapted Jacobian nnz is wrong");
   Check(problem.nlp_structure().hessian_nonzeros == 5, "adapted Hessian nnz is wrong");
   Check(
      problem.nlp_jacobian_product_capabilities() == JacobianProductCapabilities{true, true},
      "native source Jacobian capabilities were not preserved");
   Check(
      problem.nlp_hessian_product_capabilities() == HessianProductCapabilities{true},
      "native source Hessian capability was not preserved");

   const std::array<Number, 3> x{{4., 4., 15.}};
   const EvaluationValue<Number> objective = problem.nlp_objective(x);
   Check(objective.has_value(), "scaled objective failed");
   CheckNear(*objective, -304., "scaled objective");

   std::array<Number, 3> gradient{};
   Check(problem.nlp_gradient(x, gradient).has_value(), "scaled gradient failed");
   CheckNear(gradient, std::array<Number, 3>{{-19., -6., -14.8}}, "scaled gradient");

   std::array<Number, 4> constraints{};
   Check(problem.nlp_constraints(x, constraints).has_value(), "scaled constraints failed");
   CheckNear(constraints, std::array<Number, 4>{{0., 7., -33., 247.}}, "scaled constraints");

   std::array<Index, 9> jacobian_rows{};
   std::array<Index, 9> jacobian_columns{};
   Check(
      problem.nlp_jacobian_structure(jacobian_rows, jacobian_columns).has_value(),
      "adapted Jacobian structure failed");
   Check(
      jacobian_rows == std::array<Index, 9>{{1, 1, 1, 0, 0, 0, 2, 3, 3}},
      "adapted Jacobian row order differs from TNLPAdapter");
   Check(
      jacobian_columns == std::array<Index, 9>{{1, 2, 0, 1, 2, 0, 1, 2, 0}},
      "adapted Jacobian columns are wrong");

   std::array<Number, 9> jacobian_values{};
   Check(
      problem.nlp_jacobian_values(x, jacobian_values).has_value(),
      "adapted Jacobian values failed");
   CheckNear(
      jacobian_values,
      std::array<Number, 9>{{8.75, 1.4, 7., 1.5, 1.2, 4.5, 2.75, 13., 26.}},
      "adapted Jacobian values");

   const std::array<Number, 3> primal_direction{{2., -3., 4.}};
   std::array<Number, 4> jacobian_product{};
   const int jacobian_values_before = counters->jacobian_values;
   Check(
      problem.nlp_jacobian_product(x, primal_direction, jacobian_product).has_value(),
      "adapted native Jacobian product failed");
   CheckNear(
      jacobian_product,
      std::array<Number, 4>{{9.3, -6.65, -8.25, 104.}},
      "adapted Jacobian product");
   Check(
      counters->jacobian_values == jacobian_values_before,
      "adapted native Jacobian product materialized values");

   const std::array<Number, 4> dual_direction{{.5, -1., 2., 3.}};
   std::array<Number, 3> transpose_product{};
   Check(
      problem.nlp_jacobian_transpose_product(x, dual_direction, transpose_product).has_value(),
      "adapted native transposed Jacobian product failed");
   CheckNear(
      transpose_product,
      std::array<Number, 3>{{73.25, -2.5, 38.2}},
      "adapted transposed Jacobian product");

   std::array<Index, 5> hessian_rows{};
   std::array<Index, 5> hessian_columns{};
   Check(
      problem.nlp_hessian_structure(hessian_rows, hessian_columns).has_value(),
      "adapted Hessian structure failed");
   Check(
      hessian_rows == std::array<Index, 5>{{1, 2, 1, 2, 0}} &&
         hessian_columns == std::array<Index, 5>{{1, 2, 0, 0, 0}},
      "adapted Hessian structure is wrong");

   std::array<Number, 5> hessian_values{};
   Check(
      problem.nlp_hessian_values(x, 1.5, dual_direction, hessian_values).has_value(),
      "adapted Hessian values failed");
   CheckNear(
      hessian_values,
      std::array<Number, 5>{{-.1875, -.72, -.75, -.15, 13.5}},
      "adapted Hessian values");

   std::array<Number, 3> hessian_product{};
   const int hessian_values_before = counters->hessian_values;
   Check(
      problem.nlp_hessian_product(
         x, 1.5, dual_direction, primal_direction, hessian_product).has_value(),
      "adapted native Hessian product failed");
   CheckNear(
      hessian_product,
      std::array<Number, 3>{{28.65, -.9375, -3.18}},
      "adapted Hessian product");
   Check(
      counters->hessian_values == hessian_values_before,
      "adapted native Hessian product materialized values");

   std::array<Number, 4> aliased{{2., -3., 4., 999.}};
   Check(
      problem.nlp_jacobian_product(
         x, std::span<const Number>(aliased).first<3>(), aliased).has_value(),
      "aliased adapted Jacobian product failed");
   CheckNear(
      aliased,
      std::array<Number, 4>{{9.3, -6.65, -8.25, 104.}},
      "aliased adapted Jacobian product");
}

void TestFallbackCapabilitiesAndStrongFailures()
{
   const auto fallback_counters = std::make_shared<Counters>();
   EvaluationValue<AnyNlpProblem> fallback_result = MakeLegacyCoordinateProblem(
      MakeNlpProblem(FullModel{fallback_counters}), CoordinateMap(), Scaling());
   Check(fallback_result.has_value(), "fallback coordinate adapter construction failed");
   AnyNlpProblem fallback = std::move(*fallback_result);
   Check(
      fallback.nlp_jacobian_product_capabilities() == JacobianProductCapabilities{false, false},
      "fallback Jacobian capabilities were hidden by the adapter");
   Check(
      fallback.nlp_hessian_product_capabilities() == HessianProductCapabilities{false},
      "fallback Hessian capability was hidden by the adapter");

   const std::array<Number, 3> x{{4., 4., 15.}};
   const std::array<Number, 3> primal_direction{{2., -3., 4.}};
   const std::array<Number, 4> dual_direction{{.5, -1., 2., 3.}};
   std::array<Number, 4> jacobian_product{};
   Check(
      fallback.nlp_jacobian_product(x, primal_direction, jacobian_product).has_value(),
      "adapted fallback Jacobian product failed");
   Check(fallback_counters->jacobian_values == 1, "fallback Jacobian was not materialized");
   std::array<Number, 3> hessian_product{};
   Check(
      fallback.nlp_hessian_product(
         x, 1.5, dual_direction, primal_direction, hessian_product).has_value(),
      "adapted fallback Hessian product failed");
   Check(fallback_counters->hessian_values == 1, "fallback Hessian was not materialized");

   const auto failing_counters = std::make_shared<Counters>();
   AnyNlpProblem failing = AdaptNative(failing_counters);
   failing_counters->fail_gradient = true;
   std::array<Number, 3> gradient{{101., 102., 103.}};
   const EvaluationResult gradient_failure = failing.nlp_gradient(x, gradient);
   Check(!gradient_failure.has_value(), "gradient failure was ignored");
   Check(
      gradient == std::array<Number, 3>{{101., 102., 103.}},
      "failed adapted gradient modified its output");

   failing_counters->fail_jacobian_product = true;
   std::array<Number, 4> untouched_jacobian{{104., 105., 106., 107.}};
   const EvaluationResult jacobian_failure = failing.nlp_jacobian_product(
      x, primal_direction, untouched_jacobian);
   Check(!jacobian_failure.has_value(), "Jacobian product failure was ignored");
   Check(
      untouched_jacobian == std::array<Number, 4>{{104., 105., 106., 107.}},
      "failed adapted Jacobian product modified its output");

   failing_counters->fail_hessian_product = true;
   std::array<Number, 3> untouched_hessian{{108., 109., 110.}};
   const EvaluationResult hessian_failure = failing.nlp_hessian_product(
      x, 1.5, dual_direction, primal_direction, untouched_hessian);
   Check(!hessian_failure.has_value(), "Hessian product failure was ignored");
   Check(
      untouched_hessian == std::array<Number, 3>{{108., 109., 110.}},
      "failed adapted Hessian product modified its output");
}

void TestKktComposition()
{
   const auto counters = std::make_shared<Counters>();
   NlpKktOperator kkt(AdaptNative(counters));
   const std::array<Number, 3> x{{4., 4., 15.}};
   const std::array<Number, 4> multipliers{{.5, -1., 2., 3.}};
   const std::array<Number, 3> primal_direction{{2., -3., 4.}};
   const std::array<Number, 4> dual_direction{{.5, -1., 2., 3.}};
   const std::array<Number, 3> primal_diagonal{{.1, .2, .3}};
   const std::array<Number, 4> dual_diagonal{{.4, .5, .6, .7}};
   std::array<Number, 3> primal_result{};
   std::array<Number, 4> dual_result{};
   const EvaluationResult applied = kkt.apply(
      {x, 1.5, multipliers},
      {primal_direction, dual_direction},
      {primal_diagonal, dual_diagonal},
      {primal_result, dual_result});
   Check(applied.has_value(), "coordinate-adapted KKT apply failed");
   CheckNear(
      primal_result,
      std::array<Number, 3>{{102.1, -4.0375, 36.22}},
      "coordinate-adapted KKT primal block");
   CheckNear(
      dual_result,
      std::array<Number, 4>{{9.1, -6.15, -9.45, 101.9}},
      "coordinate-adapted KKT dual block");
}

void TestFingerprintAndValidation()
{
   const auto first_counters = std::make_shared<Counters>();
   EvaluationValue<AnyNlpProblem> first_result = MakeLegacyCoordinateProblem(
      MakeNlpProblem(FullNativeModel{first_counters}), CoordinateMap(5., 7.), Scaling());
   const auto numeric_counters = std::make_shared<Counters>();
   EvaluationValue<AnyNlpProblem> numeric_result = MakeLegacyCoordinateProblem(
      MakeNlpProblem(FullNativeModel{numeric_counters}), CoordinateMap(6., 8.), Scaling());
   Check(first_result.has_value() && numeric_result.has_value(), "fingerprint fixtures failed");
   AnyNlpProblem first = std::move(*first_result);
   AnyNlpProblem numeric = std::move(*numeric_result);
   const StructureFingerprintResult first_fingerprint = first.nlp_structure_fingerprint();
   const StructureFingerprintResult numeric_fingerprint = numeric.nlp_structure_fingerprint();
   Check(
      first_fingerprint.has_value() && numeric_fingerprint.has_value(),
      "adapted fingerprint evaluation failed");
   Check(
      *first_fingerprint == *numeric_fingerprint,
      "numeric fixed values or RHS incorrectly changed the structural fingerprint");

   LegacyCoordinateMap changed_map = CoordinateMap();
   changed_map.internal_to_full_variables = {3, 2, 0};
   const auto changed_counters = std::make_shared<Counters>();
   EvaluationValue<AnyNlpProblem> changed_result = MakeLegacyCoordinateProblem(
      MakeNlpProblem(FullNativeModel{changed_counters}), std::move(changed_map), Scaling());
   Check(changed_result.has_value(), "changed map fixture failed");
   AnyNlpProblem changed = std::move(*changed_result);
   const StructureFingerprintResult changed_fingerprint = changed.nlp_structure_fingerprint();
   Check(changed_fingerprint.has_value(), "changed map fingerprint failed");
   Check(*first_fingerprint != *changed_fingerprint, "coordinate map was omitted from fingerprint");

   LegacyCoordinateMap duplicate = CoordinateMap();
   duplicate.internal_to_full_variables = {3, 0, 3};
   const auto duplicate_counters = std::make_shared<Counters>();
   EvaluationValue<AnyNlpProblem> duplicate_result = MakeLegacyCoordinateProblem(
      MakeNlpProblem(FullNativeModel{duplicate_counters}), std::move(duplicate), Scaling());
   Check(!duplicate_result.has_value(), "duplicate variable map was accepted");
   Check(
      duplicate_result.error().code == EvaluationErrorCode::invalid_layout,
      "duplicate variable map returned the wrong error");

   LegacyNlpScaling zero_scaling = Scaling();
   zero_scaling.variables[1] = 0.;
   const auto scaling_counters = std::make_shared<Counters>();
   EvaluationValue<AnyNlpProblem> scaling_result = MakeLegacyCoordinateProblem(
      MakeNlpProblem(FullNativeModel{scaling_counters}), CoordinateMap(),
      std::move(zero_scaling));
   Check(!scaling_result.has_value(), "zero variable scaling was accepted");
   Check(
      scaling_result.error().code == EvaluationErrorCode::invalid_layout,
      "zero variable scaling returned the wrong error");

   LegacyCoordinateMap removed_fixed = CoordinateMap();
   removed_fixed.fixed_variable_equalities = {{1, 5.}};
   const auto removed_counters = std::make_shared<Counters>();
   EvaluationValue<AnyNlpProblem> removed_result = MakeLegacyCoordinateProblem(
      MakeNlpProblem(FullNativeModel{removed_counters}), std::move(removed_fixed), Scaling());
   Check(!removed_result.has_value(), "fixed equality on a removed variable was accepted");
   Check(
      removed_result.error().code == EvaluationErrorCode::invalid_layout,
      "removed fixed equality returned the wrong error");
}
} // namespace

int main()
{
   try
   {
      TestAllTransformations();
      TestFallbackCapabilitiesAndStrongFailures();
      TestKktComposition();
      TestFingerprintAndValidation();
      std::cout << "legacy coordinate adapter tests passed\n";
      return 0;
   }
   catch( const std::exception& error )
   {
      std::cerr << error.what() << '\n';
      return 1;
   }
}
