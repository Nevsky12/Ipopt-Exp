// Copyright (C) 2026 Ipopt contributors.
// All Rights Reserved.
// This file is distributed under the Eclipse Public License.

#ifndef IPOPT_CXX23_ANY_NLP_PROBLEM_HPP
#define IPOPT_CXX23_ANY_NLP_PROBLEM_HPP

#include <anyany/anyany.hpp>
#include <anyany/anyany_macro.hpp>

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace Ipopt::Cxx23
{
using Number = double;
using Index = std::size_t;

enum class EvaluationErrorCode
{
   dimension_mismatch,
   invalid_sparsity,
   invalid_layout,
   structure_mismatch,
   numeric_mismatch,
   nonfinite_output,
   overlapping_outputs,
   model_failure
};

struct EvaluationError
{
   EvaluationErrorCode code;
   std::string message;

   friend bool operator==(const EvaluationError&, const EvaluationError&) = default;
};

using EvaluationResult = std::expected<void, EvaluationError>;

template <class Value>
using EvaluationValue = std::expected<Value, EvaluationError>;

struct NlpStructure
{
   Index variables = 0;
   Index constraints = 0;
   Index jacobian_nonzeros = 0;
   Index hessian_nonzeros = 0;
   std::uint64_t revision = 0;

   friend bool operator==(const NlpStructure&, const NlpStructure&) = default;
};

struct StructureFingerprint
{
   std::uint64_t low;
   std::uint64_t high;

   friend bool operator==(const StructureFingerprint&, const StructureFingerprint&) = default;
};

using StructureFingerprintResult = EvaluationValue<StructureFingerprint>;

struct JacobianProductCapabilities
{
   bool forward_native = false;
   bool transpose_native = false;

   friend bool operator==(
      const JacobianProductCapabilities&,
      const JacobianProductCapabilities&) = default;
};

struct HessianProductCapabilities
{
   bool native = false;

   friend bool operator==(
      const HessianProductCapabilities&,
      const HessianProductCapabilities&) = default;
};

namespace detail
{
inline EvaluationResult DimensionMismatch(
   std::string_view argument,
   Index            actual,
   Index            expected
)
{
   return std::unexpected(EvaluationError{
      EvaluationErrorCode::dimension_mismatch,
      std::string(argument) + " has size " + std::to_string(actual) +
         ", expected " + std::to_string(expected)
   });
}

inline bool Overlaps(std::span<const Number> input, std::span<Number> output) noexcept
{
   if( input.empty() || output.empty() )
   {
      return false;
   }
   const auto input_begin = reinterpret_cast<std::uintptr_t>(input.data());
   const auto input_end = input_begin + input.size_bytes();
   const auto output_begin = reinterpret_cast<std::uintptr_t>(output.data());
   const auto output_end = output_begin + output.size_bytes();
   return input_begin < output_end && output_begin < input_end;
}

inline std::span<const Number> PreserveAliasedInput(
   std::span<const Number> input,
   std::span<Number>       output,
   std::vector<Number>&    scratch
)
{
   if( !Overlaps(input, output) )
   {
      return input;
   }
   scratch.assign(input.begin(), input.end());
   return scratch;
}

inline void HashWord(StructureFingerprint& fingerprint, std::uint64_t value) noexcept
{
   constexpr std::uint64_t fnv_prime = 1099511628211ULL;
   fingerprint.low ^= value;
   fingerprint.low *= fnv_prime;

   fingerprint.high += value + 0x9e3779b97f4a7c15ULL +
      (fingerprint.high << 6U) + (fingerprint.high >> 2U);
   fingerprint.high *= 0xd6e8feb86659fd93ULL;
}

template <class Model>
concept HasNativeJacobianProduct = requires(
   Model&                  model,
   std::span<const Number> x,
   std::span<const Number> direction,
   std::span<Number>       result
)
{
   { model.eval_jacobian_product(x, direction, result) } -> std::same_as<EvaluationResult>;
};

template <class Model>
concept HasNativeJacobianTransposeProduct = requires(
   Model&                  model,
   std::span<const Number> x,
   std::span<const Number> direction,
   std::span<Number>       result
)
{
   { model.eval_jacobian_transpose_product(x, direction, result) } -> std::same_as<EvaluationResult>;
};

template <class Model>
concept HasNativeHessianProduct = requires(
   Model&                  model,
   std::span<const Number> x,
   Number                  objective_factor,
   std::span<const Number> multipliers,
   std::span<const Number> direction,
   std::span<Number>       result
)
{
   { model.eval_hessian_product(x, objective_factor, multipliers, direction, result) }
      -> std::same_as<EvaluationResult>;
};

template <class Model>
concept HasJacobianProductCapabilities = requires(const Model& model)
{
   { model.jacobian_product_capabilities() } noexcept
      -> std::same_as<JacobianProductCapabilities>;
};

template <class Model>
concept HasHessianProductCapabilities = requires(const Model& model)
{
   { model.hessian_product_capabilities() } noexcept
      -> std::same_as<HessianProductCapabilities>;
};
} // namespace detail

template <class Model>
concept NlpModel = requires(
   Model&                  model,
   const Model&            const_model,
   std::span<const Number> input,
   std::span<Number>       output,
   std::span<Index>        indices
)
{
   { const_model.structure() } -> std::same_as<NlpStructure>;
   { model.eval_objective(input) } -> std::same_as<EvaluationValue<Number>>;
   { model.eval_gradient(input, output) } -> std::same_as<EvaluationResult>;
   { model.eval_constraints(input, output) } -> std::same_as<EvaluationResult>;
   { model.eval_jacobian_structure(indices, indices) } -> std::same_as<EvaluationResult>;
   { model.eval_jacobian_values(input, output) } -> std::same_as<EvaluationResult>;
   { model.eval_hessian_structure(indices, indices) } -> std::same_as<EvaluationResult>;
   { model.eval_hessian_values(input, Number{}, input, output) } -> std::same_as<EvaluationResult>;
};

template <NlpModel Model>
class NlpModelAdapter
{
public:
   explicit NlpModelAdapter(Model model)
      : model_(std::move(model)),
        structure_(model_.structure())
   {
   }

   NlpStructure nlp_structure() const
   {
      return structure_;
   }

   EvaluationValue<Number> nlp_objective(std::span<const Number> x)
   {
      const NlpStructure structure = nlp_structure();
      if( x.size() != structure.variables )
      {
         return std::unexpected(detail::DimensionMismatch("x", x.size(), structure.variables).error());
      }
      return model_.eval_objective(x);
   }

   EvaluationResult nlp_gradient(std::span<const Number> x, std::span<Number> result)
   {
      const NlpStructure structure = nlp_structure();
      if( x.size() != structure.variables )
      {
         return detail::DimensionMismatch("x", x.size(), structure.variables);
      }
      if( result.size() != structure.variables )
      {
         return detail::DimensionMismatch("gradient", result.size(), structure.variables);
      }
      return model_.eval_gradient(x, result);
   }

   EvaluationResult nlp_constraints(std::span<const Number> x, std::span<Number> result)
   {
      const NlpStructure structure = nlp_structure();
      if( x.size() != structure.variables )
      {
         return detail::DimensionMismatch("x", x.size(), structure.variables);
      }
      if( result.size() != structure.constraints )
      {
         return detail::DimensionMismatch("constraints", result.size(), structure.constraints);
      }
      return model_.eval_constraints(x, result);
   }

   EvaluationResult nlp_jacobian_structure(std::span<Index> rows, std::span<Index> columns)
   {
      const NlpStructure structure = nlp_structure();
      if( rows.size() != structure.jacobian_nonzeros )
      {
         return detail::DimensionMismatch("Jacobian rows", rows.size(), structure.jacobian_nonzeros);
      }
      if( columns.size() != structure.jacobian_nonzeros )
      {
         return detail::DimensionMismatch("Jacobian columns", columns.size(), structure.jacobian_nonzeros);
      }
      if( EvaluationResult cached = EnsureJacobianStructure(); !cached )
      {
         return cached;
      }
      std::ranges::copy(jacobian_rows_, rows.begin());
      std::ranges::copy(jacobian_columns_, columns.begin());
      return {};
   }

   EvaluationResult nlp_jacobian_values(std::span<const Number> x, std::span<Number> values)
   {
      const NlpStructure structure = nlp_structure();
      if( x.size() != structure.variables )
      {
         return detail::DimensionMismatch("x", x.size(), structure.variables);
      }
      if( values.size() != structure.jacobian_nonzeros )
      {
         return detail::DimensionMismatch("Jacobian values", values.size(), structure.jacobian_nonzeros);
      }
      return model_.eval_jacobian_values(x, values);
   }

   EvaluationResult nlp_hessian_structure(std::span<Index> rows, std::span<Index> columns)
   {
      const NlpStructure structure = nlp_structure();
      if( rows.size() != structure.hessian_nonzeros )
      {
         return detail::DimensionMismatch("Hessian rows", rows.size(), structure.hessian_nonzeros);
      }
      if( columns.size() != structure.hessian_nonzeros )
      {
         return detail::DimensionMismatch("Hessian columns", columns.size(), structure.hessian_nonzeros);
      }
      if( EvaluationResult cached = EnsureHessianStructure(); !cached )
      {
         return cached;
      }
      std::ranges::copy(hessian_rows_, rows.begin());
      std::ranges::copy(hessian_columns_, columns.begin());
      return {};
   }

   EvaluationResult nlp_hessian_values(
      std::span<const Number> x,
      Number                  objective_factor,
      std::span<const Number> multipliers,
      std::span<Number>       values
   )
   {
      const NlpStructure structure = nlp_structure();
      if( x.size() != structure.variables )
      {
         return detail::DimensionMismatch("x", x.size(), structure.variables);
      }
      if( multipliers.size() != structure.constraints )
      {
         return detail::DimensionMismatch(
            "constraint multipliers", multipliers.size(), structure.constraints);
      }
      if( values.size() != structure.hessian_nonzeros )
      {
         return detail::DimensionMismatch("Hessian values", values.size(), structure.hessian_nonzeros);
      }
      return model_.eval_hessian_values(x, objective_factor, multipliers, values);
   }

   JacobianProductCapabilities nlp_jacobian_product_capabilities() const noexcept
   {
      if constexpr( detail::HasJacobianProductCapabilities<Model> )
      {
         const JacobianProductCapabilities capabilities =
            model_.jacobian_product_capabilities();
         return {
            capabilities.forward_native && detail::HasNativeJacobianProduct<Model>,
            capabilities.transpose_native &&
               detail::HasNativeJacobianTransposeProduct<Model>
         };
      }
      return {
         detail::HasNativeJacobianProduct<Model>,
         detail::HasNativeJacobianTransposeProduct<Model>
      };
   }

   HessianProductCapabilities nlp_hessian_product_capabilities() const noexcept
   {
      if constexpr( detail::HasHessianProductCapabilities<Model> )
      {
         const HessianProductCapabilities capabilities =
            model_.hessian_product_capabilities();
         return {capabilities.native && detail::HasNativeHessianProduct<Model>};
      }
      return {detail::HasNativeHessianProduct<Model>};
   }

   EvaluationResult nlp_jacobian_product(
      std::span<const Number> x,
      std::span<const Number> direction,
      std::span<Number>       result
   )
   {
      const NlpStructure structure = nlp_structure();
      if( EvaluationResult dimensions = ValidateProductDimensions(
             x, direction, result, structure.variables, structure.constraints);
          !dimensions )
      {
         return dimensions;
      }

      const std::span<const Number> safe_x = detail::PreserveAliasedInput(x, result, x_scratch_);
      const std::span<const Number> safe_direction =
         detail::PreserveAliasedInput(direction, result, direction_scratch_);
      if constexpr( detail::HasNativeJacobianProduct<Model> )
      {
         return model_.eval_jacobian_product(safe_x, safe_direction, result);
      }
      return MaterializedJacobianProduct(safe_x, safe_direction, result, false);
   }

   EvaluationResult nlp_jacobian_transpose_product(
      std::span<const Number> x,
      std::span<const Number> direction,
      std::span<Number>       result
   )
   {
      const NlpStructure structure = nlp_structure();
      if( EvaluationResult dimensions = ValidateProductDimensions(
             x, direction, result, structure.constraints, structure.variables);
          !dimensions )
      {
         return dimensions;
      }

      const std::span<const Number> safe_x = detail::PreserveAliasedInput(x, result, x_scratch_);
      const std::span<const Number> safe_direction =
         detail::PreserveAliasedInput(direction, result, direction_scratch_);
      if constexpr( detail::HasNativeJacobianTransposeProduct<Model> )
      {
         return model_.eval_jacobian_transpose_product(safe_x, safe_direction, result);
      }
      return MaterializedJacobianProduct(safe_x, safe_direction, result, true);
   }

   EvaluationResult nlp_hessian_product(
      std::span<const Number> x,
      Number                  objective_factor,
      std::span<const Number> multipliers,
      std::span<const Number> direction,
      std::span<Number>       result
   )
   {
      const NlpStructure structure = nlp_structure();
      if( x.size() != structure.variables )
      {
         return detail::DimensionMismatch("x", x.size(), structure.variables);
      }
      if( multipliers.size() != structure.constraints )
      {
         return detail::DimensionMismatch(
            "constraint multipliers", multipliers.size(), structure.constraints);
      }
      if( direction.size() != structure.variables )
      {
         return detail::DimensionMismatch("direction", direction.size(), structure.variables);
      }
      if( result.size() != structure.variables )
      {
         return detail::DimensionMismatch("product result", result.size(), structure.variables);
      }

      const std::span<const Number> safe_x = detail::PreserveAliasedInput(x, result, x_scratch_);
      const std::span<const Number> safe_multipliers =
         detail::PreserveAliasedInput(multipliers, result, multiplier_scratch_);
      const std::span<const Number> safe_direction =
         detail::PreserveAliasedInput(direction, result, direction_scratch_);
      if constexpr( detail::HasNativeHessianProduct<Model> )
      {
         return model_.eval_hessian_product(
            safe_x, objective_factor, safe_multipliers, safe_direction, result);
      }
      return MaterializedHessianProduct(
         safe_x, objective_factor, safe_multipliers, safe_direction, result);
   }

   StructureFingerprintResult nlp_structure_fingerprint()
   {
      if( structure_fingerprint_.has_value() )
      {
         return *structure_fingerprint_;
      }
      if( EvaluationResult cached = EnsureJacobianStructure(); !cached )
      {
         return std::unexpected(cached.error());
      }
      if( EvaluationResult cached = EnsureHessianStructure(); !cached )
      {
         return std::unexpected(cached.error());
      }

      constexpr std::uint64_t fnv_offset = 14695981039346656037ULL;
      StructureFingerprint result{fnv_offset, 0x243f6a8885a308d3ULL};
      const NlpStructure structure = nlp_structure();
      detail::HashWord(result, structure.variables);
      detail::HashWord(result, structure.constraints);
      detail::HashWord(result, structure.jacobian_nonzeros);
      detail::HashWord(result, structure.hessian_nonzeros);
      detail::HashWord(result, structure.revision);
      const JacobianProductCapabilities jacobian_capabilities =
         nlp_jacobian_product_capabilities();
      const HessianProductCapabilities hessian_capabilities =
         nlp_hessian_product_capabilities();
      detail::HashWord(result, jacobian_capabilities.forward_native);
      detail::HashWord(result, jacobian_capabilities.transpose_native);
      detail::HashWord(result, hessian_capabilities.native);
      for( Index i = 0; i < structure.jacobian_nonzeros; ++i )
      {
         detail::HashWord(result, jacobian_rows_[i]);
         detail::HashWord(result, jacobian_columns_[i]);
      }
      for( Index i = 0; i < structure.hessian_nonzeros; ++i )
      {
         detail::HashWord(result, hessian_rows_[i]);
         detail::HashWord(result, hessian_columns_[i]);
      }
      structure_fingerprint_ = result;
      return *structure_fingerprint_;
   }

private:
   EvaluationResult ValidateProductDimensions(
      std::span<const Number> x,
      std::span<const Number> direction,
      std::span<Number>       result,
      Index                   expected_direction,
      Index                   expected_result
   ) const
   {
      const NlpStructure structure = nlp_structure();
      if( x.size() != structure.variables )
      {
         return detail::DimensionMismatch("x", x.size(), structure.variables);
      }
      if( direction.size() != expected_direction )
      {
         return detail::DimensionMismatch("direction", direction.size(), expected_direction);
      }
      if( result.size() != expected_result )
      {
         return detail::DimensionMismatch("product result", result.size(), expected_result);
      }
      return {};
   }

   EvaluationResult EnsureJacobianStructure()
   {
      if( jacobian_structure_cached_ )
      {
         return {};
      }

      const NlpStructure structure = nlp_structure();
      jacobian_rows_.resize(structure.jacobian_nonzeros);
      jacobian_columns_.resize(structure.jacobian_nonzeros);
      jacobian_values_.resize(structure.jacobian_nonzeros);
      if( EvaluationResult evaluated = model_.eval_jacobian_structure(
             jacobian_rows_, jacobian_columns_);
          !evaluated )
      {
         return evaluated;
      }
      for( Index i = 0; i < structure.jacobian_nonzeros; ++i )
      {
         if( jacobian_rows_[i] >= structure.constraints ||
             jacobian_columns_[i] >= structure.variables )
         {
            return std::unexpected(EvaluationError{
               EvaluationErrorCode::invalid_sparsity,
               "Jacobian nonzero " + std::to_string(i) + " is outside matrix dimensions"
            });
         }
      }
      jacobian_structure_cached_ = true;
      return {};
   }

   EvaluationResult EnsureHessianStructure()
   {
      if( hessian_structure_cached_ )
      {
         return {};
      }

      const NlpStructure structure = nlp_structure();
      hessian_rows_.resize(structure.hessian_nonzeros);
      hessian_columns_.resize(structure.hessian_nonzeros);
      hessian_values_.resize(structure.hessian_nonzeros);
      if( EvaluationResult evaluated = model_.eval_hessian_structure(
             hessian_rows_, hessian_columns_);
          !evaluated )
      {
         return evaluated;
      }
      for( Index i = 0; i < structure.hessian_nonzeros; ++i )
      {
         if( hessian_rows_[i] >= structure.variables ||
             hessian_columns_[i] >= structure.variables )
         {
            return std::unexpected(EvaluationError{
               EvaluationErrorCode::invalid_sparsity,
               "Hessian nonzero " + std::to_string(i) + " is outside matrix dimensions"
            });
         }
         if( hessian_rows_[i] < hessian_columns_[i] )
         {
            return std::unexpected(EvaluationError{
               EvaluationErrorCode::invalid_sparsity,
               "Hessian nonzero " + std::to_string(i) +
                  " is outside the declared lower triangle"
            });
         }
      }
      hessian_structure_cached_ = true;
      return {};
   }

   EvaluationResult MaterializedJacobianProduct(
      std::span<const Number> x,
      std::span<const Number> direction,
      std::span<Number>       result,
      bool                    transpose
   )
   {
      if( EvaluationResult cached = EnsureJacobianStructure(); !cached )
      {
         return cached;
      }
      if( EvaluationResult evaluated = model_.eval_jacobian_values(x, jacobian_values_); !evaluated )
      {
         return evaluated;
      }

      std::ranges::fill(result, 0.);
      for( Index i = 0; i < jacobian_values_.size(); ++i )
      {
         if( transpose )
         {
            result[jacobian_columns_[i]] += jacobian_values_[i] * direction[jacobian_rows_[i]];
         }
         else
         {
            result[jacobian_rows_[i]] += jacobian_values_[i] * direction[jacobian_columns_[i]];
         }
      }
      return {};
   }

   EvaluationResult MaterializedHessianProduct(
      std::span<const Number> x,
      Number                  objective_factor,
      std::span<const Number> multipliers,
      std::span<const Number> direction,
      std::span<Number>       result
   )
   {
      if( EvaluationResult cached = EnsureHessianStructure(); !cached )
      {
         return cached;
      }
      if( EvaluationResult evaluated = model_.eval_hessian_values(
             x, objective_factor, multipliers, hessian_values_);
          !evaluated )
      {
         return evaluated;
      }

      std::ranges::fill(result, 0.);
      for( Index i = 0; i < hessian_values_.size(); ++i )
      {
         const Index row = hessian_rows_[i];
         const Index column = hessian_columns_[i];
         result[row] += hessian_values_[i] * direction[column];
         if( row != column )
         {
            result[column] += hessian_values_[i] * direction[row];
         }
      }
      return {};
   }

   Model model_;
   const NlpStructure structure_;
   bool jacobian_structure_cached_ = false;
   bool hessian_structure_cached_ = false;
   std::vector<Index> jacobian_rows_;
   std::vector<Index> jacobian_columns_;
   std::vector<Number> jacobian_values_;
   std::vector<Index> hessian_rows_;
   std::vector<Index> hessian_columns_;
   std::vector<Number> hessian_values_;
   std::vector<Number> x_scratch_;
   std::vector<Number> multiplier_scratch_;
   std::vector<Number> direction_scratch_;
   std::optional<StructureFingerprint> structure_fingerprint_;
};

anyany_method(nlp_structure,
   (const& self) requires(self.nlp_structure())->NlpStructure);
anyany_method(nlp_objective,
   (&self, std::span<const Number> x) requires(self.nlp_objective(x))->EvaluationValue<Number>);
anyany_method(nlp_gradient,
   (&self, std::span<const Number> x, std::span<Number> result)
      requires(self.nlp_gradient(x, result))->EvaluationResult);
anyany_method(nlp_constraints,
   (&self, std::span<const Number> x, std::span<Number> result)
      requires(self.nlp_constraints(x, result))->EvaluationResult);
anyany_method(nlp_jacobian_structure,
   (&self, std::span<Index> rows, std::span<Index> columns)
      requires(self.nlp_jacobian_structure(rows, columns))->EvaluationResult);
anyany_method(nlp_jacobian_values,
   (&self, std::span<const Number> x, std::span<Number> values)
      requires(self.nlp_jacobian_values(x, values))->EvaluationResult);
anyany_method(nlp_hessian_structure,
   (&self, std::span<Index> rows, std::span<Index> columns)
      requires(self.nlp_hessian_structure(rows, columns))->EvaluationResult);
anyany_method(nlp_hessian_values,
   (&self, std::span<const Number> x, Number objective_factor,
      std::span<const Number> multipliers, std::span<Number> values)
      requires(self.nlp_hessian_values(x, objective_factor, multipliers, values))->EvaluationResult);
anyany_method(nlp_jacobian_product_capabilities,
   (const& self) requires(self.nlp_jacobian_product_capabilities())->JacobianProductCapabilities);
anyany_method(nlp_hessian_product_capabilities,
   (const& self) requires(self.nlp_hessian_product_capabilities())->HessianProductCapabilities);
anyany_method(nlp_jacobian_product,
   (&self, std::span<const Number> x, std::span<const Number> direction, std::span<Number> result)
      requires(self.nlp_jacobian_product(x, direction, result))->EvaluationResult);
anyany_method(nlp_jacobian_transpose_product,
   (&self, std::span<const Number> x, std::span<const Number> direction, std::span<Number> result)
      requires(self.nlp_jacobian_transpose_product(x, direction, result))->EvaluationResult);
anyany_method(nlp_hessian_product,
   (&self, std::span<const Number> x, Number objective_factor,
      std::span<const Number> multipliers, std::span<const Number> direction,
      std::span<Number> result)
      requires(self.nlp_hessian_product(x, objective_factor, multipliers, direction, result))
         ->EvaluationResult);
anyany_method(nlp_structure_fingerprint,
   (&self) requires(self.nlp_structure_fingerprint())->StructureFingerprintResult);

using AnyNlpProblem = aa::any_with<
   aa::move,
   aa::type_info,
   nlp_structure,
   nlp_objective,
   nlp_gradient,
   nlp_constraints,
   nlp_jacobian_structure,
   nlp_jacobian_values,
   nlp_hessian_structure,
   nlp_hessian_values,
   nlp_jacobian_product_capabilities,
   nlp_hessian_product_capabilities,
   nlp_jacobian_product,
   nlp_jacobian_transpose_product,
   nlp_hessian_product,
   nlp_structure_fingerprint>;

/** Non-owning AnyAny forwarding object for a problem with a longer lifetime.
 *
 * This keeps the owning problem's derivative-structure and fingerprint caches
 * alive while a short-lived consumer retains the usual AnyNlpProblem value
 * interface.  The returned object must not outlive problem.
 */
class NlpProblemReference
{
public:
   explicit NlpProblemReference(AnyNlpProblem& problem) noexcept
      : problem_(*problem.operator&())
   {
   }

   NlpStructure nlp_structure() const
   {
      return problem_.nlp_structure();
   }

   EvaluationValue<Number> nlp_objective(std::span<const Number> x)
   {
      return problem_.nlp_objective(x);
   }

   EvaluationResult nlp_gradient(
      std::span<const Number> x,
      std::span<Number>       result
   )
   {
      return problem_.nlp_gradient(x, result);
   }

   EvaluationResult nlp_constraints(
      std::span<const Number> x,
      std::span<Number>       result
   )
   {
      return problem_.nlp_constraints(x, result);
   }

   EvaluationResult nlp_jacobian_structure(
      std::span<Index> rows,
      std::span<Index> columns
   )
   {
      return problem_.nlp_jacobian_structure(rows, columns);
   }

   EvaluationResult nlp_jacobian_values(
      std::span<const Number> x,
      std::span<Number>       values
   )
   {
      return problem_.nlp_jacobian_values(x, values);
   }

   EvaluationResult nlp_hessian_structure(
      std::span<Index> rows,
      std::span<Index> columns
   )
   {
      return problem_.nlp_hessian_structure(rows, columns);
   }

   EvaluationResult nlp_hessian_values(
      std::span<const Number> x,
      Number                  objective_factor,
      std::span<const Number> multipliers,
      std::span<Number>       values
   )
   {
      return problem_.nlp_hessian_values(
         x, objective_factor, multipliers, values);
   }

   JacobianProductCapabilities nlp_jacobian_product_capabilities() const noexcept
   {
      return problem_.nlp_jacobian_product_capabilities();
   }

   HessianProductCapabilities nlp_hessian_product_capabilities() const noexcept
   {
      return problem_.nlp_hessian_product_capabilities();
   }

   EvaluationResult nlp_jacobian_product(
      std::span<const Number> x,
      std::span<const Number> direction,
      std::span<Number>       result
   )
   {
      return problem_.nlp_jacobian_product(x, direction, result);
   }

   EvaluationResult nlp_jacobian_transpose_product(
      std::span<const Number> x,
      std::span<const Number> direction,
      std::span<Number>       result
   )
   {
      return problem_.nlp_jacobian_transpose_product(x, direction, result);
   }

   EvaluationResult nlp_hessian_product(
      std::span<const Number> x,
      Number                  objective_factor,
      std::span<const Number> multipliers,
      std::span<const Number> direction,
      std::span<Number>       result
   )
   {
      return problem_.nlp_hessian_product(
         x, objective_factor, multipliers, direction, result);
   }

   StructureFingerprintResult nlp_structure_fingerprint()
   {
      return problem_.nlp_structure_fingerprint();
   }

private:
   AnyNlpProblem::ref problem_;
};

inline AnyNlpProblem BorrowNlpProblem(AnyNlpProblem& problem)
{
   return NlpProblemReference(problem);
}

template <NlpModel Model>
AnyNlpProblem MakeNlpProblem(Model model)
{
   return NlpModelAdapter<Model>(std::move(model));
}
} // namespace Ipopt::Cxx23

#endif
