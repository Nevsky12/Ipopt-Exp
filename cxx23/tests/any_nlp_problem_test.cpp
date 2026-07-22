// Copyright (C) 2026 Ipopt contributors.
// All Rights Reserved.
// This file is distributed under the Eclipse Public License.

#include <ipopt/cxx23/primal_dual_kkt_operator.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string_view>

namespace
{
using namespace Ipopt::Cxx23;

struct Counters
{
   int jacobian_structures = 0;
   int hessian_structures = 0;
   int jacobian_values = 0;
   int hessian_values = 0;
   int jacobian_products = 0;
   int jacobian_transpose_products = 0;
   int jacobian_product_pairs = 0;
   int hessian_products = 0;
   bool fail_jacobian_values = false;
   bool fail_jacobian_product_pair = false;
   bool fail_hessian_values = false;
};

struct FallbackModel
{
   std::shared_ptr<Counters> counters;
   std::uint64_t revision = 7;

   NlpStructure structure() const
   {
      return {3, 2, 4, 4, revision};
   }

   EvaluationValue<Number> eval_objective(std::span<const Number> x)
   {
      return x[0] * x[0] * x[0] / 3. + x[1] * x[1] + x[2] * x[2] + x[0] * x[2];
   }

   EvaluationResult eval_gradient(std::span<const Number> x, std::span<Number> result)
   {
      result[0] = x[0] * x[0] + x[2];
      result[1] = 2. * x[1];
      result[2] = 2. * x[2] + x[0];
      return {};
   }

   EvaluationResult eval_constraints(std::span<const Number> x, std::span<Number> result)
   {
      result[0] = (2. + x[0]) * x[0] - x[2];
      result[1] = 3. * x[1] + 4. * x[2];
      return {};
   }

   EvaluationResult eval_jacobian_structure(std::span<Index> rows, std::span<Index> columns)
   {
      ++counters->jacobian_structures;
      rows[0] = 0;
      columns[0] = 0;
      rows[1] = 0;
      columns[1] = 2;
      rows[2] = 1;
      columns[2] = 1;
      rows[3] = 1;
      columns[3] = 2;
      return {};
   }

   EvaluationResult eval_jacobian_values(std::span<const Number> x, std::span<Number> values)
   {
      ++counters->jacobian_values;
      if( counters->fail_jacobian_values )
      {
         return std::unexpected(EvaluationError{
            EvaluationErrorCode::model_failure,
            "model rejected Jacobian values"
         });
      }
      values[0] = 2. + x[0];
      values[1] = -1.;
      values[2] = 3.;
      values[3] = 4.;
      return {};
   }

   EvaluationResult eval_hessian_structure(std::span<Index> rows, std::span<Index> columns)
   {
      ++counters->hessian_structures;
      rows[0] = 0;
      columns[0] = 0;
      rows[1] = 1;
      columns[1] = 1;
      rows[2] = 2;
      columns[2] = 0;
      rows[3] = 2;
      columns[3] = 2;
      return {};
   }

   EvaluationResult eval_hessian_values(
      std::span<const Number> x,
      Number                  objective_factor,
      std::span<const Number> multipliers,
      std::span<Number>       values
   )
   {
      ++counters->hessian_values;
      if( counters->fail_hessian_values )
      {
         return std::unexpected(EvaluationError{
            EvaluationErrorCode::model_failure,
            "model rejected Hessian values"
         });
      }
      values[0] = 2. * objective_factor * x[0] + 2. * multipliers[0];
      values[1] = 2. * objective_factor;
      values[2] = objective_factor;
      values[3] = 2. * objective_factor;
      return {};
   }
};

struct NativeModel : FallbackModel
{
   EvaluationResult eval_jacobian_product(
      std::span<const Number> x,
      std::span<const Number> direction,
      std::span<Number>       result
   )
   {
      ++counters->jacobian_products;
      result[0] = (2. + x[0]) * direction[0] - direction[2];
      result[1] = 3. * direction[1] + 4. * direction[2];
      return {};
   }

   EvaluationResult eval_jacobian_transpose_product(
      std::span<const Number> x,
      std::span<const Number> direction,
      std::span<Number>       result
   )
   {
      ++counters->jacobian_transpose_products;
      result[0] = (2. + x[0]) * direction[0];
      result[1] = 3. * direction[1];
      result[2] = -direction[0] + 4. * direction[1];
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
      ++counters->hessian_products;
      result[0] = (2. * objective_factor * x[0] + 2. * multipliers[0]) * direction[0] +
         objective_factor * direction[2];
      result[1] = 2. * objective_factor * direction[1];
      result[2] = objective_factor * direction[0] + 2. * objective_factor * direction[2];
      return {};
   }
};

struct ForwardOnlyModel : FallbackModel
{
   EvaluationResult eval_jacobian_product(
      std::span<const Number> x,
      std::span<const Number> direction,
      std::span<Number>       result
   )
   {
      ++counters->jacobian_products;
      result[0] = (2. + x[0]) * direction[0] - direction[2];
      result[1] = 3. * direction[1] + 4. * direction[2];
      return {};
   }
};

struct FusedNativeModel : NativeModel
{
   explicit FusedNativeModel(std::shared_ptr<Counters> counters)
      : NativeModel{{std::move(counters)}}
   {
   }

   EvaluationResult eval_jacobian_products(
      std::span<const Number> x,
      std::span<const Number> forward_direction,
      std::span<const Number> transpose_direction,
      std::span<Number>       forward_result,
      std::span<Number>       transpose_result
   )
   {
      ++counters->jacobian_product_pairs;
      forward_result[0] = 12345.;
      transpose_result[0] = 23456.;
      if( counters->fail_jacobian_product_pair )
      {
         return std::unexpected(EvaluationError{
            EvaluationErrorCode::model_failure,
            "model rejected fused Jacobian products"
         });
      }
      forward_result[0] =
         (2. + x[0]) * forward_direction[0] - forward_direction[2];
      forward_result[1] =
         3. * forward_direction[1] + 4. * forward_direction[2];
      transpose_result[0] = (2. + x[0]) * transpose_direction[0];
      transpose_result[1] = 3. * transpose_direction[1];
      transpose_result[2] =
         -transpose_direction[0] + 4. * transpose_direction[1];
      return {};
   }
};

struct InvalidSparsityModel : FallbackModel
{
   explicit InvalidSparsityModel(std::shared_ptr<Counters> counters)
      : FallbackModel{std::move(counters)}
   {
   }

   EvaluationResult eval_jacobian_structure(std::span<Index> rows, std::span<Index> columns)
   {
      ++counters->jacobian_structures;
      std::ranges::fill(rows, 0);
      std::ranges::fill(columns, 0);
      rows.back() = structure().constraints;
      return {};
   }
};

struct InvalidHessianSparsityModel : FallbackModel
{
   explicit InvalidHessianSparsityModel(std::shared_ptr<Counters> counters)
      : FallbackModel{std::move(counters)}
   {
   }

   EvaluationResult eval_hessian_structure(std::span<Index> rows, std::span<Index> columns)
   {
      ++counters->hessian_structures;
      std::ranges::fill(rows, 0);
      std::ranges::fill(columns, 0);
      columns.front() = 1;
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
   if( std::abs(actual - expected) > 1e-13 )
   {
      throw std::runtime_error(
         std::string(message) + ": actual=" + std::to_string(actual) +
         " expected=" + std::to_string(expected));
   }
}

void TestMaterializedFallback()
{
   const auto counters = std::make_shared<Counters>();
   AnyNlpProblem problem = MakeNlpProblem(FallbackModel{counters});
   Check(
      problem.nlp_jacobian_product_capabilities() ==
         JacobianProductCapabilities{false, false},
      "fallback dispatch was not reported");
   Check(
      problem.nlp_hessian_product_capabilities() ==
         HessianProductCapabilities{false},
      "Hessian fallback dispatch was not reported");

   const std::array<Number, 3> x{1., 2., 3.};
   const std::array<Number, 3> direction{5., 7., 11.};
   std::array<Number, 2> product{};
   EvaluationResult forward = problem.nlp_jacobian_product(x, direction, product);
   Check(forward.has_value(), "fallback Jacobian product failed");
   CheckNear(product[0], 4., "fallback J*v row 0");
   CheckNear(product[1], 65., "fallback J*v row 1");

   const std::array<Number, 2> transpose_direction{13., 17.};
   std::array<Number, 3> transpose_product{};
   EvaluationResult transpose = problem.nlp_jacobian_transpose_product(
      x, transpose_direction, transpose_product);
   Check(transpose.has_value(), "fallback transposed Jacobian product failed");
   CheckNear(transpose_product[0], 39., "fallback J^T*v column 0");
   CheckNear(transpose_product[1], 51., "fallback J^T*v column 1");
   CheckNear(transpose_product[2], 55., "fallback J^T*v column 2");

   std::array<Number, 2> paired_forward{};
   std::array<Number, 3> paired_transpose{};
   const int values_before_pair = counters->jacobian_values;
   EvaluationResult paired = problem.nlp_jacobian_products(
      x, direction, transpose_direction, paired_forward, paired_transpose);
   Check(paired.has_value(), "fused fallback Jacobian products failed");
   CheckNear(paired_forward[0], 4., "fused fallback J*v row 0");
   CheckNear(paired_forward[1], 65., "fused fallback J*v row 1");
   CheckNear(paired_transpose[0], 39., "fused fallback J^T*v column 0");
   CheckNear(paired_transpose[1], 51., "fused fallback J^T*v column 1");
   CheckNear(paired_transpose[2], 55., "fused fallback J^T*v column 2");
   Check(
      counters->jacobian_values == values_before_pair + 1,
      "fused fallback evaluated Jacobian values more than once");

   const std::array<Number, 2> multipliers{2., 3.};
   std::array<Number, 3> hessian_product{};
   EvaluationResult hessian = problem.nlp_hessian_product(
      x, 1.5, multipliers, direction, hessian_product);
   Check(hessian.has_value(), "fallback Hessian product failed");
   CheckNear(hessian_product[0], 51.5, "fallback H*v component 0");
   CheckNear(hessian_product[1], 21., "fallback H*v component 1");
   CheckNear(hessian_product[2], 40.5, "fallback H*v component 2");

   const StructureFingerprintResult fingerprint = problem.nlp_structure_fingerprint();
   const StructureFingerprintResult fingerprint_again = problem.nlp_structure_fingerprint();
   Check(fingerprint.has_value(), "fingerprint construction failed");
   Check(fingerprint == fingerprint_again, "fingerprint is not stable");
   Check(counters->jacobian_structures == 1, "Jacobian structure was not cached");
   Check(counters->hessian_structures == 1, "Hessian structure was not cached");
   Check(counters->jacobian_values == 3, "Jacobian values callback count is wrong");
   Check(counters->hessian_values == 1, "Hessian values callback count is wrong");
}

void TestNativeAndAliasedProducts()
{
   const auto counters = std::make_shared<Counters>();
   AnyNlpProblem problem = MakeNlpProblem(NativeModel{{counters}});
   Check(
      problem.nlp_jacobian_product_capabilities() ==
         JacobianProductCapabilities{true, true},
      "native dispatch was not reported");
   Check(
      problem.nlp_hessian_product_capabilities() ==
         HessianProductCapabilities{true},
      "native Hessian dispatch was not reported");

   const std::array<Number, 3> x{1., 2., 3.};
   std::array<Number, 3> aliased_forward{5., 7., 11.};
   EvaluationResult forward = problem.nlp_jacobian_product(
      x,
      std::span<const Number>(aliased_forward),
      std::span<Number>(aliased_forward).first<2>());
   Check(forward.has_value(), "aliased native Jacobian product failed");
   CheckNear(aliased_forward[0], 4., "aliased native J*v row 0");
   CheckNear(aliased_forward[1], 65., "aliased native J*v row 1");

   std::array<Number, 3> aliased_x{1., 2., 3.};
   const std::array<Number, 3> x_alias_direction{5., 7., 11.};
   EvaluationResult x_alias = problem.nlp_jacobian_product(
      std::span<const Number>(aliased_x),
      x_alias_direction,
      std::span<Number>(aliased_x).first<2>());
   Check(x_alias.has_value(), "x-aliased native Jacobian product failed");
   CheckNear(aliased_x[0], 4., "x-aliased native J*v row 0");
   CheckNear(aliased_x[1], 65., "x-aliased native J*v row 1");

   std::array<Number, 3> aliased_transpose{13., 17., -1.};
   EvaluationResult transpose = problem.nlp_jacobian_transpose_product(
      x,
      std::span<const Number>(aliased_transpose).first<2>(),
      aliased_transpose);
   Check(transpose.has_value(), "aliased native transposed Jacobian product failed");
   CheckNear(aliased_transpose[0], 39., "aliased native J^T*v column 0");
   CheckNear(aliased_transpose[1], 51., "aliased native J^T*v column 1");
   CheckNear(aliased_transpose[2], 55., "aliased native J^T*v column 2");

   std::array<Number, 3> paired_forward{5., 7., 11.};
   std::array<Number, 3> paired_transpose{13., 17., -1.};
   EvaluationResult paired = problem.nlp_jacobian_products(
      x,
      std::span<const Number>(paired_forward),
      std::span<const Number>(paired_transpose).first<2>(),
      std::span<Number>(paired_forward).first<2>(),
      paired_transpose);
   Check(paired.has_value(), "aliased native Jacobian products failed");
   CheckNear(paired_forward[0], 4., "aliased fused native J*v row 0");
   CheckNear(paired_forward[1], 65., "aliased fused native J*v row 1");
   CheckNear(paired_transpose[0], 39., "aliased fused native J^T*v column 0");
   CheckNear(paired_transpose[1], 51., "aliased fused native J^T*v column 1");
   CheckNear(paired_transpose[2], 55., "aliased fused native J^T*v column 2");

   std::array<Number, 3> fully_aliased_hessian{1., 2., 3.};
   EvaluationResult hessian = problem.nlp_hessian_product(
      std::span<const Number>(fully_aliased_hessian),
      1.5,
      std::span<const Number>(fully_aliased_hessian).first<2>(),
      std::span<const Number>(fully_aliased_hessian),
      fully_aliased_hessian);
   Check(hessian.has_value(), "fully aliased native Hessian product failed");
   CheckNear(fully_aliased_hessian[0], 9.5, "fully aliased native H*v component 0");
   CheckNear(fully_aliased_hessian[1], 6., "fully aliased native H*v component 1");
   CheckNear(fully_aliased_hessian[2], 10.5, "fully aliased native H*v component 2");

   Check(counters->jacobian_products == 3, "native Jacobian product callback count is wrong");
   Check(
      counters->jacobian_transpose_products == 2,
      "native transpose product callback count is wrong");
   Check(counters->hessian_products == 1, "native Hessian product callback count is wrong");
   Check(
      counters->jacobian_values == 0 && counters->hessian_values == 0,
      "native products unexpectedly materialized derivatives");
}

void TestNativeFusedProducts()
{
   const auto counters = std::make_shared<Counters>();
   AnyNlpProblem problem = MakeNlpProblem(FusedNativeModel{counters});
   const std::array<Number, 3> x{1., 2., 3.};
   const std::array<Number, 3> forward_direction{5., 7., 11.};
   const std::array<Number, 2> transpose_direction{13., 17.};
   std::array<Number, 2> forward_result{};
   std::array<Number, 3> transpose_result{};
   Check(
      problem.nlp_jacobian_products(
         x, forward_direction, transpose_direction,
         forward_result, transpose_result).has_value(),
      "native fused Jacobian products failed");
   CheckNear(forward_result[0], 4., "native fused J*v row 0");
   CheckNear(forward_result[1], 65., "native fused J*v row 1");
   CheckNear(transpose_result[0], 39., "native fused J^T*v column 0");
   CheckNear(transpose_result[1], 51., "native fused J^T*v column 1");
   CheckNear(transpose_result[2], 55., "native fused J^T*v column 2");
   Check(
      counters->jacobian_product_pairs == 1 &&
         counters->jacobian_products == 0 &&
         counters->jacobian_transpose_products == 0 &&
         counters->jacobian_values == 0,
      "native fused dispatch used an individual or materialized path");

   counters->fail_jacobian_product_pair = true;
   std::array<Number, 2> untouched_forward{101., 102.};
   std::array<Number, 3> untouched_transpose{103., 104., 105.};
   const EvaluationResult failure = problem.nlp_jacobian_products(
      x, forward_direction, transpose_direction,
      untouched_forward, untouched_transpose);
   Check(!failure.has_value(), "native fused Jacobian product failure was ignored");
   Check(
      failure.error().code == EvaluationErrorCode::model_failure,
      "native fused failure returned the wrong error code");
   Check(
      untouched_forward == std::array<Number, 2>{101., 102.} &&
         untouched_transpose == std::array<Number, 3>{103., 104., 105.},
      "failed native fused Jacobian products modified an output");
   Check(
      counters->jacobian_product_pairs == 2 &&
         counters->jacobian_products == 0 &&
         counters->jacobian_transpose_products == 0 &&
         counters->jacobian_values == 0,
      "failed native fused dispatch used an individual or materialized path");
}

void TestErrorsAndFingerprintRevision()
{
   const auto counters = std::make_shared<Counters>();
   AnyNlpProblem problem = MakeNlpProblem(FallbackModel{counters, 7});
   const std::array<Number, 3> x{1., 2., 3.};
   const std::array<Number, 2> short_direction{1., 2.};
   std::array<Number, 2> result{};
   const EvaluationResult dimensions = problem.nlp_jacobian_product(
      x, short_direction, result);
   Check(!dimensions.has_value(), "dimension mismatch was accepted");
   Check(
      dimensions.error().code == EvaluationErrorCode::dimension_mismatch,
      "dimension mismatch returned the wrong error code");
   Check(counters->jacobian_values == 0, "dimension mismatch called the model");

   counters->fail_jacobian_values = true;
   const std::array<Number, 3> direction{1., 2., 3.};
   const EvaluationResult failure = problem.nlp_jacobian_product(x, direction, result);
   Check(!failure.has_value(), "model failure was ignored");
   Check(
      failure.error().code == EvaluationErrorCode::model_failure,
      "model failure returned the wrong error code");

   const std::array<Number, 2> transpose_direction{4., 5.};
   std::array<Number, 2> untouched_forward{201., 202.};
   std::array<Number, 3> untouched_transpose{203., 204., 205.};
   const EvaluationResult paired_failure = problem.nlp_jacobian_products(
      x, direction, transpose_direction,
      untouched_forward, untouched_transpose);
   Check(!paired_failure.has_value(), "fused model failure was ignored");
   Check(
      untouched_forward == std::array<Number, 2>{201., 202.} &&
         untouched_transpose == std::array<Number, 3>{203., 204., 205.},
      "failed fused Jacobian products modified an output");

   counters->fail_jacobian_values = false;
   std::array<Number, 4> overlapping_results{301., 302., 303., 304.};
   const std::array<Number, 4> overlapping_before = overlapping_results;
   const int values_before_overlap = counters->jacobian_values;
   const EvaluationResult overlap = problem.nlp_jacobian_products(
      x, direction, transpose_direction,
      std::span<Number>(overlapping_results).first<2>(),
      std::span<Number>(overlapping_results).subspan<1, 3>());
   Check(!overlap.has_value(), "overlapping fused product outputs were accepted");
   Check(
      overlap.error().code == EvaluationErrorCode::overlapping_outputs,
      "overlapping fused outputs returned the wrong error code");
   Check(
      overlapping_results == overlapping_before &&
         counters->jacobian_values == values_before_overlap,
      "fused output-overlap validation modified state or called the model");

   counters->fail_hessian_values = true;
   const std::array<Number, 2> multipliers{2., 3.};
   std::array<Number, 3> hessian_result{101., 102., 103.};
   const EvaluationResult hessian_failure = problem.nlp_hessian_product(
      x, 1.5, multipliers, direction, hessian_result);
   Check(!hessian_failure.has_value(), "Hessian model failure was ignored");
   Check(
      hessian_failure.error().code == EvaluationErrorCode::model_failure,
      "Hessian model failure returned the wrong error code");
   Check(
      hessian_result == std::array<Number, 3>{101., 102., 103.},
      "failed Hessian product modified its output");

   const auto counters_other = std::make_shared<Counters>();
   AnyNlpProblem other = MakeNlpProblem(FallbackModel{counters_other, 8});
   const StructureFingerprintResult first = problem.nlp_structure_fingerprint();
   const StructureFingerprintResult second = other.nlp_structure_fingerprint();
   Check(first.has_value() && second.has_value(), "fingerprint evaluation failed");
   Check(*first != *second, "structure revision was omitted from fingerprint");

   const auto native_counters = std::make_shared<Counters>();
   AnyNlpProblem native = MakeNlpProblem(NativeModel{{native_counters, 7}});
   const StructureFingerprintResult native_fingerprint = native.nlp_structure_fingerprint();
   Check(native_fingerprint.has_value(), "native fingerprint evaluation failed");
   Check(*first != *native_fingerprint, "derivative capabilities were omitted from fingerprint");

   const auto invalid_counters = std::make_shared<Counters>();
   AnyNlpProblem invalid = MakeNlpProblem(InvalidSparsityModel{invalid_counters});
   const StructureFingerprintResult invalid_fingerprint = invalid.nlp_structure_fingerprint();
   Check(!invalid_fingerprint.has_value(), "invalid sparsity was accepted");
   Check(
      invalid_fingerprint.error().code == EvaluationErrorCode::invalid_sparsity,
      "invalid sparsity returned the wrong error code");
   Check(
      invalid_counters->jacobian_values == 0,
      "invalid sparsity evaluated Jacobian values");

   const auto invalid_hessian_counters = std::make_shared<Counters>();
   AnyNlpProblem invalid_hessian = MakeNlpProblem(
      InvalidHessianSparsityModel{invalid_hessian_counters});
   const StructureFingerprintResult invalid_hessian_fingerprint =
      invalid_hessian.nlp_structure_fingerprint();
   Check(!invalid_hessian_fingerprint.has_value(), "upper-triangle Hessian was accepted");
   Check(
      invalid_hessian_fingerprint.error().code == EvaluationErrorCode::invalid_sparsity,
      "invalid Hessian sparsity returned the wrong error code");
   Check(
      invalid_hessian_counters->hessian_values == 0,
      "invalid Hessian sparsity evaluated Hessian values");
}

void TestBorrowedProblemRetainsOwnerCaches()
{
   const auto counters = std::make_shared<Counters>();
   AnyNlpProblem owner = MakeNlpProblem(FallbackModel{counters});
   AnyNlpProblem first = BorrowNlpProblem(owner);
   AnyNlpProblem second = BorrowNlpProblem(owner);

   const StructureFingerprintResult owner_fingerprint =
      owner.nlp_structure_fingerprint();
   const StructureFingerprintResult first_fingerprint =
      first.nlp_structure_fingerprint();
   const StructureFingerprintResult second_fingerprint =
      second.nlp_structure_fingerprint();
   Check(
      owner_fingerprint.has_value() && first_fingerprint.has_value() &&
         second_fingerprint.has_value(),
      "borrowed problem fingerprint failed");
   Check(
      owner_fingerprint == first_fingerprint &&
         owner_fingerprint == second_fingerprint,
      "borrowed problem changed the owner's fingerprint");
   Check(
      counters->jacobian_structures == 1 && counters->hessian_structures == 1,
      "borrowed problems did not retain the owner's structure caches");

   const std::array<Number, 3> x{1., 2., 3.};
   const std::array<Number, 3> direction{5., 7., 11.};
   std::array<Number, 2> product{};
   Check(
      second.nlp_jacobian_product(x, direction, product).has_value(),
      "borrowed problem Jacobian product failed");
   CheckNear(product[0], 4., "borrowed problem J*v row 0");
   CheckNear(product[1], 65., "borrowed problem J*v row 1");
}

void TestPartialNativeCapability()
{
   const auto counters = std::make_shared<Counters>();
   AnyNlpProblem problem = MakeNlpProblem(ForwardOnlyModel{{counters}});
   Check(
      problem.nlp_jacobian_product_capabilities() ==
         JacobianProductCapabilities{true, false},
      "partial native capability was not reported");
   Check(
      problem.nlp_hessian_product_capabilities() ==
         HessianProductCapabilities{false},
      "partial model Hessian fallback was not reported");

   const std::array<Number, 3> x{1., 2., 3.};
   const std::array<Number, 3> direction{5., 7., 11.};
   std::array<Number, 2> product{};
   Check(
      problem.nlp_jacobian_product(x, direction, product).has_value(),
      "partial native forward product failed");

   const std::array<Number, 2> transpose_direction{13., 17.};
   std::array<Number, 3> transpose_product{};
   Check(
      problem.nlp_jacobian_transpose_product(
         x, transpose_direction, transpose_product).has_value(),
      "partial native fallback transpose product failed");
   std::array<Number, 2> paired_product{};
   std::array<Number, 3> paired_transpose{};
   Check(
      problem.nlp_jacobian_products(
         x, direction, transpose_direction,
         paired_product, paired_transpose).has_value(),
      "partial native fused products failed");
   CheckNear(paired_product[0], 4., "partial fused J*v row 0");
   CheckNear(paired_product[1], 65., "partial fused J*v row 1");
   CheckNear(paired_transpose[0], 39., "partial fused J^T*v column 0");
   CheckNear(paired_transpose[1], 51., "partial fused J^T*v column 1");
   CheckNear(paired_transpose[2], 55., "partial fused J^T*v column 2");
   Check(
      counters->jacobian_products == 2,
      "partial capability missed the native forward path");
   Check(
      counters->jacobian_values == 2,
      "partial capability missed the transpose fallback path");
}

void TestKktOperator()
{
   const auto counters = std::make_shared<Counters>();
   NlpKktOperator kkt(MakeNlpProblem(NativeModel{{counters}}));
   Check(
      kkt.capabilities() == KktOperatorCapabilities{
         JacobianProductCapabilities{true, true}, HessianProductCapabilities{true}},
      "KKT operator capabilities are wrong");

   const std::array<Number, 3> x{1., 2., 3.};
   const std::array<Number, 2> multipliers{2., 3.};
   std::array<Number, 3> primal{5., 7., 11.};
   std::array<Number, 2> dual{13., 17.};
   const std::array<Number, 3> primal_diagonal{0.1, 0.2, 0.3};
   const std::array<Number, 2> dual_diagonal{0.4, 0.5};
   const EvaluationResult applied = kkt.apply(
      {x, 1.5, multipliers},
      {std::span<const Number>(primal), std::span<const Number>(dual)},
      {primal_diagonal, dual_diagonal},
      {primal, dual});
   Check(applied.has_value(), "native KKT operator failed");
   CheckNear(primal[0], 91., "KKT primal component 0");
   CheckNear(primal[1], 73.4, "KKT primal component 1");
   CheckNear(primal[2], 98.8, "KKT primal component 2");
   CheckNear(dual[0], -1.2, "KKT dual component 0");
   CheckNear(dual[1], 56.5, "KKT dual component 1");
   Check(counters->hessian_products == 1, "KKT operator missed native H*v");
   Check(counters->jacobian_products == 1, "KKT operator missed native J*v");
   Check(
      counters->jacobian_transpose_products == 1,
      "KKT operator missed native J^T*v");

   std::array<Number, 4> overlapping_outputs{};
   const EvaluationResult overlap = kkt.apply(
      {x, 1.5, multipliers},
      {x, multipliers},
      {primal_diagonal, dual_diagonal},
      {
         std::span<Number>(overlapping_outputs).first<3>(),
         std::span<Number>(overlapping_outputs).subspan<2, 2>()
      });
   Check(!overlap.has_value(), "overlapping KKT outputs were accepted");
   Check(
      overlap.error().code == EvaluationErrorCode::overlapping_outputs,
      "overlapping KKT outputs returned the wrong error code");
   Check(counters->hessian_products == 1, "invalid KKT output called the model");

   const auto failing_counters = std::make_shared<Counters>();
   failing_counters->fail_jacobian_values = true;
   NlpKktOperator failing(MakeNlpProblem(FallbackModel{failing_counters}));
   std::array<Number, 3> untouched_primal{101., 102., 103.};
   std::array<Number, 2> untouched_dual{104., 105.};
   const EvaluationResult failure = failing.apply(
      {x, 1.5, multipliers},
      {x, multipliers},
      {primal_diagonal, dual_diagonal},
      {untouched_primal, untouched_dual});
   Check(!failure.has_value(), "KKT callback failure was ignored");
   Check(
      untouched_primal == std::array<Number, 3>{101., 102., 103.} &&
         untouched_dual == std::array<Number, 2>{104., 105.},
      "failed KKT apply modified an output");
}

void TestPrimalDualKktOperator()
{
   const auto counters = std::make_shared<Counters>();
   PrimalDualKktOperator kkt(
      MakeNlpProblem(FallbackModel{counters}),
      {
         .equality_constraints = {1},
         .inequality_constraints = {0},
         .primal_lower_bounds = {0, 2},
         .primal_upper_bounds = {1},
         .slack_lower_bounds = {0},
         .slack_upper_bounds = {0}
      });
   Check(kkt.valid(), "valid primal-dual layout was rejected");
   Check(kkt.flat_dimension() == 11, "primal-dual flat dimension is wrong");

   const std::array<Number, 3> x{1., 2., 3.};
   const std::array<Number, 2> multipliers{2., 3.};
   const std::array<Number, 2> z_lower{5., 7.};
   const std::array<Number, 1> z_upper{11.};
   const std::array<Number, 1> v_lower{13.};
   const std::array<Number, 1> v_upper{17.};
   const std::array<Number, 2> slack_x_lower{19., 23.};
   const std::array<Number, 1> slack_x_upper{29.};
   const std::array<Number, 1> slack_s_lower{31.};
   const std::array<Number, 1> slack_s_upper{37.};
   const PrimalDualState state{
      .nlp = {x, 1.5, multipliers},
      .z_lower = z_lower,
      .z_upper = z_upper,
      .v_lower = v_lower,
      .v_upper = v_upper,
      .slack_x_lower = slack_x_lower,
      .slack_x_upper = slack_x_upper,
      .slack_s_lower = slack_s_lower,
      .slack_s_upper = slack_s_upper,
      .regularization = {0.1, 0.2, 0.3, 0.4}
   };

   std::array<Number, 3> dx{0.5, 1.5, 2.5};
   std::array<Number, 1> ds{3.5};
   std::array<Number, 1> dy_c{4.5};
   std::array<Number, 1> dy_d{5.5};
   std::array<Number, 2> dz_lower{6.5, 7.5};
   std::array<Number, 1> dz_upper{8.5};
   std::array<Number, 1> dv_lower{9.5};
   std::array<Number, 1> dv_upper{10.5};
   const EvaluationResult applied = kkt.apply(
      state,
      {dx, ds, dy_c, dy_d, dz_lower, dz_upper, dv_lower, dv_upper},
      {dx, ds, dy_c, dy_d, dz_lower, dz_upper, dv_lower, dv_upper});
   Check(applied.has_value(), "aliased primal-dual KKT apply failed");
   const std::array<Number, 11> expected{
      17.3, 26.65, 13.5, -3.8, 13.15, -6.7,
      126., 190., 230., 340., 329.
   };
   const std::array<Number, 11> block_result{
      dx[0], dx[1], dx[2], ds[0], dy_c[0], dy_d[0],
      dz_lower[0], dz_lower[1], dz_upper[0], dv_lower[0], dv_upper[0]
   };
   for( Index i = 0; i < expected.size(); ++i )
   {
      CheckNear(block_result[i], expected[i], "primal-dual block result");
   }

   const std::array<Number, 11> flat_direction{
      0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5
   };
   std::array<Number, 11> flat_aliased = flat_direction;
   const EvaluationResult flat_applied = kkt.apply_flat(
      state, std::span<const Number>(flat_aliased), flat_aliased);
   Check(flat_applied.has_value(), "aliased flat primal-dual KKT apply failed");
   for( Index i = 0; i < expected.size(); ++i )
   {
      CheckNear(flat_aliased[i], expected[i], "primal-dual flat result");
   }

   const StructureFingerprintResult fingerprint = kkt.structure_fingerprint();
   Check(fingerprint.has_value(), "primal-dual fingerprint failed");
   const auto other_counters = std::make_shared<Counters>();
   PrimalDualKktOperator other_layout(
      MakeNlpProblem(FallbackModel{other_counters}),
      {
         .equality_constraints = {0},
         .inequality_constraints = {1},
         .primal_lower_bounds = {0, 2},
         .primal_upper_bounds = {1},
         .slack_lower_bounds = {0},
         .slack_upper_bounds = {0}
      });
   const StructureFingerprintResult other_fingerprint =
      other_layout.structure_fingerprint();
   Check(
      other_fingerprint.has_value() && *fingerprint != *other_fingerprint,
      "primal-dual layout was omitted from fingerprint");

   const int jacobian_calls_before_overlap = counters->jacobian_values;
   const int hessian_calls_before_overlap = counters->hessian_values;
   std::array<Number, 4> overlapping_output{201., 202., 203., 204.};
   const std::array<Number, 4> overlapping_output_before = overlapping_output;
   std::array<Number, 1> overlap_y_c{};
   std::array<Number, 1> overlap_y_d{};
   std::array<Number, 2> overlap_z_lower{};
   std::array<Number, 1> overlap_z_upper{};
   std::array<Number, 1> overlap_v_lower{};
   std::array<Number, 1> overlap_v_upper{};
   const std::span<const Number> flat_direction_span(flat_direction);
   const EvaluationResult overlap = kkt.apply(
      state,
      {
         flat_direction_span.first<3>(), flat_direction_span.subspan<3, 1>(),
         flat_direction_span.subspan<4, 1>(), flat_direction_span.subspan<5, 1>(),
         flat_direction_span.subspan<6, 2>(), flat_direction_span.subspan<8, 1>(),
         flat_direction_span.subspan<9, 1>(), flat_direction_span.subspan<10, 1>()
      },
      {
         std::span<Number>(overlapping_output).first<3>(),
         std::span<Number>(overlapping_output).subspan<2, 1>(),
         overlap_y_c, overlap_y_d, overlap_z_lower, overlap_z_upper,
         overlap_v_lower, overlap_v_upper
      });
   Check(!overlap.has_value(), "overlapping primal-dual outputs were accepted");
   Check(
      overlap.error().code == EvaluationErrorCode::overlapping_outputs,
      "overlapping primal-dual outputs returned the wrong error code");
   Check(
      overlapping_output == overlapping_output_before,
      "overlap validation modified a primal-dual output");
   Check(
      counters->jacobian_values == jacobian_calls_before_overlap &&
         counters->hessian_values == hessian_calls_before_overlap,
      "overlap validation called the model");

   counters->fail_jacobian_values = true;
   std::array<Number, 11> untouched{
      101., 102., 103., 104., 105., 106., 107., 108., 109., 110., 111.
   };
   const std::array<Number, 11> untouched_before = untouched;
   const EvaluationResult failure = kkt.apply_flat(state, flat_direction, untouched);
   Check(!failure.has_value(), "primal-dual callback failure was ignored");
   Check(untouched == untouched_before, "failed primal-dual apply modified its output");

   const auto invalid_counters = std::make_shared<Counters>();
   PrimalDualKktOperator invalid(
      MakeNlpProblem(FallbackModel{invalid_counters}),
      {
         .equality_constraints = {1},
         .inequality_constraints = {1},
         .primal_lower_bounds = {0, 2},
         .primal_upper_bounds = {1},
         .slack_lower_bounds = {0},
         .slack_upper_bounds = {0}
      });
   Check(!invalid.valid(), "repeated primal-dual constraint was accepted");
   std::array<Number, 11> invalid_result{};
   const EvaluationResult invalid_apply = invalid.apply_flat(
      state, flat_direction, invalid_result);
   Check(!invalid_apply.has_value(), "invalid primal-dual layout was applied");
   Check(
      invalid_apply.error().code == EvaluationErrorCode::invalid_layout,
      "invalid primal-dual layout returned the wrong error code");
   Check(
      invalid_counters->jacobian_values == 0 && invalid_counters->hessian_values == 0,
      "invalid primal-dual layout called the model");
}
} // namespace

int main()
{
   try
   {
      TestMaterializedFallback();
      TestNativeAndAliasedProducts();
      TestNativeFusedProducts();
      TestErrorsAndFingerprintRevision();
      TestBorrowedProblemRetainsOwnerCaches();
      TestPartialNativeCapability();
      TestKktOperator();
      TestPrimalDualKktOperator();
   }
   catch( const std::exception& error )
   {
      std::cerr << "ipopt_cxx23_any_nlp_problem_test: " << error.what() << '\n';
      return EXIT_FAILURE;
   }
   std::cout << "ipopt_cxx23_any_nlp_problem_test: passed\n";
   return EXIT_SUCCESS;
}
