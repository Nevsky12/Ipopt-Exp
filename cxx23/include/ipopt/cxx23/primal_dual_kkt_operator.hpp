// Copyright (C) 2026 Ipopt contributors.
// All Rights Reserved.
// This file is distributed under the Eclipse Public License.

#ifndef IPOPT_CXX23_PRIMAL_DUAL_KKT_OPERATOR_HPP
#define IPOPT_CXX23_PRIMAL_DUAL_KKT_OPERATOR_HPP

#include <ipopt/cxx23/kkt_operator.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace Ipopt::Cxx23
{
struct PrimalDualLayout
{
   std::vector<Index> equality_constraints;
   std::vector<Index> inequality_constraints;
   std::vector<Index> primal_lower_bounds;
   std::vector<Index> primal_upper_bounds;
   std::vector<Index> slack_lower_bounds;
   std::vector<Index> slack_upper_bounds;

   friend bool operator==(
      const PrimalDualLayout&,
      const PrimalDualLayout&) = default;
};

struct PrimalDualDimensions
{
   Index x;
   Index s;
   Index y_c;
   Index y_d;
   Index z_lower;
   Index z_upper;
   Index v_lower;
   Index v_upper;

   friend bool operator==(
      const PrimalDualDimensions&,
      const PrimalDualDimensions&) = default;

   Index total() const noexcept
   {
      return x + s + y_c + y_d + z_lower + z_upper + v_lower + v_upper;
   }
};

struct PrimalDualRegularization
{
   Number x;
   Number s;
   Number c;
   Number d;
};

struct PrimalDualState
{
   KktEvaluationPoint nlp;
   std::span<const Number> z_lower;
   std::span<const Number> z_upper;
   std::span<const Number> v_lower;
   std::span<const Number> v_upper;
   std::span<const Number> slack_x_lower;
   std::span<const Number> slack_x_upper;
   std::span<const Number> slack_s_lower;
   std::span<const Number> slack_s_upper;
   PrimalDualRegularization regularization;
   /** Caller-owned generation for numeric KKT data and evaluation state.
    *
    * Matrix-free application does not require this token. Prepared
    * factorizations require a nonzero value and reject a different generation.
    */
   std::uint64_t numeric_revision = 0;
};

struct PrimalDualDirection
{
   std::span<const Number> x;
   std::span<const Number> s;
   std::span<const Number> y_c;
   std::span<const Number> y_d;
   std::span<const Number> z_lower;
   std::span<const Number> z_upper;
   std::span<const Number> v_lower;
   std::span<const Number> v_upper;
};

struct PrimalDualResult
{
   std::span<Number> x;
   std::span<Number> s;
   std::span<Number> y_c;
   std::span<Number> y_d;
   std::span<Number> z_lower;
   std::span<Number> z_upper;
   std::span<Number> v_lower;
   std::span<Number> v_upper;
};

/** Matrix-free form of the eight-block primal-dual system used by
 * PDFullSpaceSolver::ComputeResiduals.
 *
 * Constraint and bound index maps express the ExpansionMatrix blocks without
 * materializing them. Scaling and fixed-variable conversion are supplied by
 * LegacyCoordinateAdapter when bridging the stable TNLP coordinates.
 */
class PrimalDualKktOperator
{
public:
   PrimalDualKktOperator(AnyNlpProblem problem, PrimalDualLayout layout)
      : nlp_kkt_(std::move(problem)),
        layout_(std::move(layout)),
        nlp_structure_(nlp_kkt_.structure()),
        dimensions_{
           nlp_structure_.variables,
           layout_.inequality_constraints.size(),
           layout_.equality_constraints.size(),
           layout_.inequality_constraints.size(),
           layout_.primal_lower_bounds.size(),
           layout_.primal_upper_bounds.size(),
           layout_.slack_lower_bounds.size(),
           layout_.slack_upper_bounds.size()
        },
        constraint_direction_(nlp_structure_.constraints),
        primal_diagonal_(nlp_structure_.variables),
        constraint_diagonal_(nlp_structure_.constraints),
        x_result_(dimensions_.x),
        constraint_result_(nlp_structure_.constraints),
        s_result_(dimensions_.s),
        y_c_result_(dimensions_.y_c),
        y_d_result_(dimensions_.y_d),
        z_lower_result_(dimensions_.z_lower),
        z_upper_result_(dimensions_.z_upper),
        v_lower_result_(dimensions_.v_lower),
        v_upper_result_(dimensions_.v_upper)
   {
      configuration_error_ = ValidateLayout();
   }

   PrimalDualDimensions dimensions() const noexcept
   {
      return dimensions_;
   }

   NlpStructure nlp_structure() const noexcept
   {
      return nlp_structure_;
   }

   const PrimalDualLayout& layout() const noexcept
   {
      return layout_;
   }

   Index flat_dimension() const noexcept
   {
      return dimensions_.total();
   }

   bool valid() const noexcept
   {
      return !configuration_error_.has_value();
   }

   KktOperatorCapabilities capabilities() const noexcept
   {
      return nlp_kkt_.capabilities();
   }

   EvaluationResult validate_state(PrimalDualState state) const
   {
      if( configuration_error_.has_value() )
      {
         return std::unexpected(*configuration_error_);
      }
      return ValidateStateDimensions(state);
   }

   StructureFingerprintResult structure_fingerprint()
   {
      if( configuration_error_.has_value() )
      {
         return std::unexpected(*configuration_error_);
      }
      if( structure_fingerprint_.has_value() )
      {
         return *structure_fingerprint_;
      }
      StructureFingerprintResult base = nlp_kkt_.structure_fingerprint();
      if( !base )
      {
         return base;
      }
      StructureFingerprint result = *base;
      HashIndexList(result, 1, layout_.equality_constraints);
      HashIndexList(result, 2, layout_.inequality_constraints);
      HashIndexList(result, 3, layout_.primal_lower_bounds);
      HashIndexList(result, 4, layout_.primal_upper_bounds);
      HashIndexList(result, 5, layout_.slack_lower_bounds);
      HashIndexList(result, 6, layout_.slack_upper_bounds);
      structure_fingerprint_ = result;
      return *structure_fingerprint_;
   }

   EvaluationResult nlp_jacobian_structure(
      std::span<Index> rows,
      std::span<Index> columns
   )
   {
      if( configuration_error_.has_value() )
      {
         return std::unexpected(*configuration_error_);
      }
      return nlp_kkt_.jacobian_structure(rows, columns);
   }

   EvaluationResult nlp_jacobian_values(
      KktEvaluationPoint point,
      std::span<Number>  values
   )
   {
      if( configuration_error_.has_value() )
      {
         return std::unexpected(*configuration_error_);
      }
      return nlp_kkt_.jacobian_values(point, values);
   }

   EvaluationResult nlp_hessian_structure(
      std::span<Index> rows,
      std::span<Index> columns
   )
   {
      if( configuration_error_.has_value() )
      {
         return std::unexpected(*configuration_error_);
      }
      return nlp_kkt_.hessian_structure(rows, columns);
   }

   EvaluationResult nlp_hessian_values(
      KktEvaluationPoint point,
      std::span<Number>  values
   )
   {
      if( configuration_error_.has_value() )
      {
         return std::unexpected(*configuration_error_);
      }
      return nlp_kkt_.hessian_values(point, values);
   }

   EvaluationResult apply(
      PrimalDualState     state,
      PrimalDualDirection direction,
      PrimalDualResult    result
   )
   {
      if( configuration_error_.has_value() )
      {
         return std::unexpected(*configuration_error_);
      }
      if( EvaluationResult dimensions = ValidateDimensions(state, direction, result); !dimensions )
      {
         return dimensions;
      }
      if( OutputsOverlap(result) )
      {
         return std::unexpected(EvaluationError{
            EvaluationErrorCode::overlapping_outputs,
            "primal-dual KKT output blocks overlap"
         });
      }

      std::ranges::fill(constraint_direction_, 0.);
      for( Index i = 0; i < dimensions_.y_c; ++i )
      {
         constraint_direction_[layout_.equality_constraints[i]] = direction.y_c[i];
      }
      for( Index i = 0; i < dimensions_.y_d; ++i )
      {
         constraint_direction_[layout_.inequality_constraints[i]] = direction.y_d[i];
      }
      std::ranges::fill(primal_diagonal_, state.regularization.x);
      for( Index index : layout_.equality_constraints )
      {
         constraint_diagonal_[index] = state.regularization.c;
      }
      for( Index index : layout_.inequality_constraints )
      {
         constraint_diagonal_[index] = state.regularization.d;
      }

      if( EvaluationResult evaluated = nlp_kkt_.apply(
             state.nlp,
             {direction.x, constraint_direction_},
             {primal_diagonal_, constraint_diagonal_},
             {x_result_, constraint_result_});
          !evaluated )
      {
         return evaluated;
      }

      for( Index i = 0; i < dimensions_.z_lower; ++i )
      {
         x_result_[layout_.primal_lower_bounds[i]] -= direction.z_lower[i];
      }
      for( Index i = 0; i < dimensions_.z_upper; ++i )
      {
         x_result_[layout_.primal_upper_bounds[i]] += direction.z_upper[i];
      }

      for( Index i = 0; i < dimensions_.s; ++i )
      {
         s_result_[i] = -direction.y_d[i] +
            state.regularization.s * direction.s[i];
      }
      for( Index i = 0; i < dimensions_.v_lower; ++i )
      {
         s_result_[layout_.slack_lower_bounds[i]] -= direction.v_lower[i];
      }
      for( Index i = 0; i < dimensions_.v_upper; ++i )
      {
         s_result_[layout_.slack_upper_bounds[i]] += direction.v_upper[i];
      }

      for( Index i = 0; i < dimensions_.y_c; ++i )
      {
         y_c_result_[i] = constraint_result_[layout_.equality_constraints[i]];
      }
      for( Index i = 0; i < dimensions_.y_d; ++i )
      {
         y_d_result_[i] = constraint_result_[layout_.inequality_constraints[i]] -
            direction.s[i];
      }

      for( Index i = 0; i < dimensions_.z_lower; ++i )
      {
         z_lower_result_[i] = state.slack_x_lower[i] * direction.z_lower[i] +
            state.z_lower[i] * direction.x[layout_.primal_lower_bounds[i]];
      }
      for( Index i = 0; i < dimensions_.z_upper; ++i )
      {
         z_upper_result_[i] = state.slack_x_upper[i] * direction.z_upper[i] -
            state.z_upper[i] * direction.x[layout_.primal_upper_bounds[i]];
      }
      for( Index i = 0; i < dimensions_.v_lower; ++i )
      {
         v_lower_result_[i] = state.slack_s_lower[i] * direction.v_lower[i] +
            state.v_lower[i] * direction.s[layout_.slack_lower_bounds[i]];
      }
      for( Index i = 0; i < dimensions_.v_upper; ++i )
      {
         v_upper_result_[i] = state.slack_s_upper[i] * direction.v_upper[i] -
            state.v_upper[i] * direction.s[layout_.slack_upper_bounds[i]];
      }

      CopyResults(result);
      return {};
   }

   EvaluationResult apply_flat(
      PrimalDualState        state,
      std::span<const Number> direction,
      std::span<Number>       result
   )
   {
      if( direction.size() != flat_dimension() )
      {
         return detail::DimensionMismatch(
            "flat primal-dual direction", direction.size(), flat_dimension());
      }
      if( result.size() != flat_dimension() )
      {
         return detail::DimensionMismatch(
            "flat primal-dual result", result.size(), flat_dimension());
      }
      return apply(state, SplitDirection(direction), SplitResult(result));
   }

private:
   std::optional<EvaluationError> ValidateLayout() const
   {
      std::vector<bool> seen_constraints(nlp_structure_.constraints, false);
      const auto add_constraints = [&](const std::vector<Index>& indices)
         -> std::optional<EvaluationError>
      {
         for( Index index : indices )
         {
            if( index >= nlp_structure_.constraints )
            {
               return EvaluationError{
                  EvaluationErrorCode::invalid_layout,
                  "constraint layout index " + std::to_string(index) + " is out of range"
               };
            }
            if( seen_constraints[index] )
            {
               return EvaluationError{
                  EvaluationErrorCode::invalid_layout,
                  "constraint layout index " + std::to_string(index) + " is repeated"
               };
            }
            seen_constraints[index] = true;
         }
         return std::nullopt;
      };
      if( std::optional<EvaluationError> error = add_constraints(layout_.equality_constraints) )
      {
         return error;
      }
      if( std::optional<EvaluationError> error = add_constraints(layout_.inequality_constraints) )
      {
         return error;
      }
      if( std::ranges::any_of(seen_constraints, [](bool seen) { return !seen; }) )
      {
         return EvaluationError{
            EvaluationErrorCode::invalid_layout,
            "constraint layout does not cover every model constraint"
         };
      }

      if( std::optional<EvaluationError> error = ValidateBoundIndices(
             "primal lower", layout_.primal_lower_bounds, dimensions_.x) )
      {
         return error;
      }
      if( std::optional<EvaluationError> error = ValidateBoundIndices(
             "primal upper", layout_.primal_upper_bounds, dimensions_.x) )
      {
         return error;
      }
      if( std::optional<EvaluationError> error = ValidateBoundIndices(
             "slack lower", layout_.slack_lower_bounds, dimensions_.s) )
      {
         return error;
      }
      return ValidateBoundIndices(
         "slack upper", layout_.slack_upper_bounds, dimensions_.s);
   }

   static std::optional<EvaluationError> ValidateBoundIndices(
      std::string_view          name,
      const std::vector<Index>& indices,
      Index                     dimension
   )
   {
      std::vector<bool> seen(dimension, false);
      for( Index index : indices )
      {
         if( index >= dimension )
         {
            return EvaluationError{
               EvaluationErrorCode::invalid_layout,
               std::string(name) + " bound index " + std::to_string(index) +
                  " is out of range"
            };
         }
         if( seen[index] )
         {
            return EvaluationError{
               EvaluationErrorCode::invalid_layout,
               std::string(name) + " bound index " + std::to_string(index) +
                  " is repeated"
            };
         }
         seen[index] = true;
      }
      return std::nullopt;
   }

   EvaluationResult ValidateDimensions(
      PrimalDualState     state,
      PrimalDualDirection direction,
      PrimalDualResult    result
   ) const
   {
      if( EvaluationResult checked = ValidateStateDimensions(state); !checked )
      {
         return checked;
      }
      const std::array<std::span<const Number>, 8> direction_blocks{
         direction.x, direction.s, direction.y_c, direction.y_d,
         direction.z_lower, direction.z_upper, direction.v_lower, direction.v_upper
      };
      const std::array<std::span<Number>, 8> result_blocks{
         result.x, result.s, result.y_c, result.y_d,
         result.z_lower, result.z_upper, result.v_lower, result.v_upper
      };
      const std::array<Index, 8> expected{
         dimensions_.x, dimensions_.s, dimensions_.y_c, dimensions_.y_d,
         dimensions_.z_lower, dimensions_.z_upper,
         dimensions_.v_lower, dimensions_.v_upper
      };
      constexpr std::array<std::string_view, 8> direction_names{
         "x direction", "s direction", "y_c direction", "y_d direction",
         "z lower direction", "z upper direction",
         "v lower direction", "v upper direction"
      };
      constexpr std::array<std::string_view, 8> result_names{
         "x result", "s result", "y_c result", "y_d result",
         "z lower result", "z upper result", "v lower result", "v upper result"
      };
      for( Index block = 0; block < expected.size(); ++block )
      {
         if( direction_blocks[block].size() != expected[block] )
         {
            return detail::DimensionMismatch(
               direction_names[block], direction_blocks[block].size(), expected[block]);
         }
         if( result_blocks[block].size() != expected[block] )
         {
            return detail::DimensionMismatch(
               result_names[block], result_blocks[block].size(), expected[block]);
         }
      }
      return {};
   }

   EvaluationResult ValidateStateDimensions(PrimalDualState state) const
   {
      const std::array<std::span<const Number>, 10> state_blocks{
         state.nlp.x, state.nlp.constraint_multipliers,
         state.z_lower, state.z_upper, state.v_lower, state.v_upper,
         state.slack_x_lower, state.slack_x_upper,
         state.slack_s_lower, state.slack_s_upper
      };
      const std::array<Index, 10> expected{
         dimensions_.x, dimensions_.y_c + dimensions_.y_d,
         dimensions_.z_lower, dimensions_.z_upper,
         dimensions_.v_lower, dimensions_.v_upper,
         dimensions_.z_lower, dimensions_.z_upper,
         dimensions_.v_lower, dimensions_.v_upper
      };
      constexpr std::array<std::string_view, 10> names{
         "x state", "constraint multiplier state",
         "z lower state", "z upper state", "v lower state", "v upper state",
         "x lower slack", "x upper slack", "s lower slack", "s upper slack"
      };
      for( Index block = 0; block < expected.size(); ++block )
      {
         if( state_blocks[block].size() != expected[block] )
         {
            return detail::DimensionMismatch(
               names[block], state_blocks[block].size(), expected[block]);
         }
      }
      return {};
   }

   static bool OutputsOverlap(PrimalDualResult result) noexcept
   {
      const std::array<std::span<Number>, 8> blocks{
         result.x, result.s, result.y_c, result.y_d,
         result.z_lower, result.z_upper, result.v_lower, result.v_upper
      };
      for( Index first = 0; first < blocks.size(); ++first )
      {
         for( Index second = first + 1; second < blocks.size(); ++second )
         {
            if( detail::Overlaps(std::span<const Number>(blocks[first]), blocks[second]) )
            {
               return true;
            }
         }
      }
      return false;
   }

   void CopyResults(PrimalDualResult result) const
   {
      std::ranges::copy(x_result_, result.x.begin());
      std::ranges::copy(s_result_, result.s.begin());
      std::ranges::copy(y_c_result_, result.y_c.begin());
      std::ranges::copy(y_d_result_, result.y_d.begin());
      std::ranges::copy(z_lower_result_, result.z_lower.begin());
      std::ranges::copy(z_upper_result_, result.z_upper.begin());
      std::ranges::copy(v_lower_result_, result.v_lower.begin());
      std::ranges::copy(v_upper_result_, result.v_upper.begin());
   }

   PrimalDualDirection SplitDirection(std::span<const Number> flat) const noexcept
   {
      Index offset = 0;
      auto take = [&](Index size)
      {
         const std::span<const Number> block = flat.subspan(offset, size);
         offset += size;
         return block;
      };
      return {
         take(dimensions_.x), take(dimensions_.s),
         take(dimensions_.y_c), take(dimensions_.y_d),
         take(dimensions_.z_lower), take(dimensions_.z_upper),
         take(dimensions_.v_lower), take(dimensions_.v_upper)
      };
   }

   PrimalDualResult SplitResult(std::span<Number> flat) const noexcept
   {
      Index offset = 0;
      auto take = [&](Index size)
      {
         const std::span<Number> block = flat.subspan(offset, size);
         offset += size;
         return block;
      };
      return {
         take(dimensions_.x), take(dimensions_.s),
         take(dimensions_.y_c), take(dimensions_.y_d),
         take(dimensions_.z_lower), take(dimensions_.z_upper),
         take(dimensions_.v_lower), take(dimensions_.v_upper)
      };
   }

   static void HashIndexList(
      StructureFingerprint&     fingerprint,
      std::uint64_t             tag,
      const std::vector<Index>& indices
   ) noexcept
   {
      detail::HashWord(fingerprint, tag);
      detail::HashWord(fingerprint, indices.size());
      for( Index index : indices )
      {
         detail::HashWord(fingerprint, index);
      }
   }

   NlpKktOperator nlp_kkt_;
   const PrimalDualLayout layout_;
   const NlpStructure nlp_structure_;
   const PrimalDualDimensions dimensions_;
   std::optional<EvaluationError> configuration_error_;
   std::optional<StructureFingerprint> structure_fingerprint_;
   std::vector<Number> constraint_direction_;
   std::vector<Number> primal_diagonal_;
   std::vector<Number> constraint_diagonal_;
   std::vector<Number> x_result_;
   std::vector<Number> constraint_result_;
   std::vector<Number> s_result_;
   std::vector<Number> y_c_result_;
   std::vector<Number> y_d_result_;
   std::vector<Number> z_lower_result_;
   std::vector<Number> z_upper_result_;
   std::vector<Number> v_lower_result_;
   std::vector<Number> v_upper_result_;
};
} // namespace Ipopt::Cxx23

#endif
