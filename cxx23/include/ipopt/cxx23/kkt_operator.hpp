// Copyright (C) 2026 Ipopt contributors.
// All Rights Reserved.
// This file is distributed under the Eclipse Public License.

#ifndef IPOPT_CXX23_KKT_OPERATOR_HPP
#define IPOPT_CXX23_KKT_OPERATOR_HPP

#include <ipopt/cxx23/any_nlp_problem.hpp>

#include <algorithm>
#include <span>
#include <utility>
#include <vector>

namespace Ipopt::Cxx23
{
struct KktEvaluationPoint
{
   std::span<const Number> x;
   Number objective_factor;
   std::span<const Number> constraint_multipliers;
};

struct KktDirection
{
   std::span<const Number> primal;
   std::span<const Number> dual;
};

struct KktDiagonal
{
   std::span<const Number> primal;
   std::span<const Number> dual;
};

struct KktResult
{
   std::span<Number> primal;
   std::span<Number> dual;
};

struct KktOperatorCapabilities
{
   JacobianProductCapabilities jacobian;
   HessianProductCapabilities hessian;

   friend bool operator==(
      const KktOperatorCapabilities&,
      const KktOperatorCapabilities&) = default;
};

/** Applies the regularized NLP saddle-point block
 *
 *     [ H(x, lambda) + D_x    J(x)^T ] [ p ]
 *     [ J(x)                 -D_y   ] [ q ]
 *
 * where H is the Hessian of the Lagrangian. This is deliberately smaller
 * than Ipopt's full primal-dual system: slack and bound-complementarity blocks
 * belong in a later adapter around this reusable core.
 */
class NlpKktOperator
{
public:
   explicit NlpKktOperator(AnyNlpProblem problem)
      : problem_(std::move(problem)),
        structure_(problem_.nlp_structure()),
        hessian_product_(structure_.variables),
        jacobian_transpose_product_(structure_.variables),
        jacobian_product_(structure_.constraints)
   {
   }

   NlpStructure structure() const noexcept
   {
      return structure_;
   }

   KktOperatorCapabilities capabilities() const noexcept
   {
      return {
         problem_.nlp_jacobian_product_capabilities(),
         problem_.nlp_hessian_product_capabilities()
      };
   }

   StructureFingerprintResult structure_fingerprint()
   {
      return problem_.nlp_structure_fingerprint();
   }

   EvaluationResult jacobian_structure(
      std::span<Index> rows,
      std::span<Index> columns
   )
   {
      return problem_.nlp_jacobian_structure(rows, columns);
   }

   EvaluationResult jacobian_values(
      KktEvaluationPoint point,
      std::span<Number>  values
   )
   {
      return problem_.nlp_jacobian_values(point.x, values);
   }

   EvaluationResult hessian_structure(
      std::span<Index> rows,
      std::span<Index> columns
   )
   {
      return problem_.nlp_hessian_structure(rows, columns);
   }

   EvaluationResult hessian_values(
      KktEvaluationPoint point,
      std::span<Number>  values
   )
   {
      return problem_.nlp_hessian_values(
         point.x,
         point.objective_factor,
         point.constraint_multipliers,
         values);
   }

   EvaluationResult apply(
      KktEvaluationPoint point,
      KktDirection       direction,
      KktDiagonal        diagonal,
      KktResult          result
   )
   {
      if( EvaluationResult dimensions = ValidateDimensions(
             point, direction, diagonal, result);
          !dimensions )
      {
         return dimensions;
      }
      if( detail::Overlaps(
             std::span<const Number>(result.primal), result.dual) )
      {
         return std::unexpected(EvaluationError{
            EvaluationErrorCode::overlapping_outputs,
            "primal and dual KKT outputs overlap"
         });
      }

      if( EvaluationResult evaluated = problem_.nlp_hessian_product(
             point.x,
             point.objective_factor,
             point.constraint_multipliers,
             direction.primal,
             hessian_product_);
          !evaluated )
      {
         return evaluated;
      }
      if( EvaluationResult evaluated = problem_.nlp_jacobian_products(
             point.x, direction.primal, direction.dual,
             jacobian_product_, jacobian_transpose_product_);
          !evaluated )
      {
         return evaluated;
      }

      for( Index i = 0; i < structure_.variables; ++i )
      {
         hessian_product_[i] += jacobian_transpose_product_[i] +
            diagonal.primal[i] * direction.primal[i];
      }
      for( Index i = 0; i < structure_.constraints; ++i )
      {
         jacobian_product_[i] -= diagonal.dual[i] * direction.dual[i];
      }

      std::ranges::copy(hessian_product_, result.primal.begin());
      std::ranges::copy(jacobian_product_, result.dual.begin());
      return {};
   }

private:
   EvaluationResult ValidateDimensions(
      KktEvaluationPoint point,
      KktDirection       direction,
      KktDiagonal        diagonal,
      KktResult          result
   ) const
   {
      if( point.x.size() != structure_.variables )
      {
         return detail::DimensionMismatch("x", point.x.size(), structure_.variables);
      }
      if( point.constraint_multipliers.size() != structure_.constraints )
      {
         return detail::DimensionMismatch(
            "constraint multipliers",
            point.constraint_multipliers.size(),
            structure_.constraints);
      }
      if( direction.primal.size() != structure_.variables )
      {
         return detail::DimensionMismatch(
            "primal direction", direction.primal.size(), structure_.variables);
      }
      if( direction.dual.size() != structure_.constraints )
      {
         return detail::DimensionMismatch(
            "dual direction", direction.dual.size(), structure_.constraints);
      }
      if( diagonal.primal.size() != structure_.variables )
      {
         return detail::DimensionMismatch(
            "primal diagonal", diagonal.primal.size(), structure_.variables);
      }
      if( diagonal.dual.size() != structure_.constraints )
      {
         return detail::DimensionMismatch(
            "dual diagonal", diagonal.dual.size(), structure_.constraints);
      }
      if( result.primal.size() != structure_.variables )
      {
         return detail::DimensionMismatch(
            "primal result", result.primal.size(), structure_.variables);
      }
      if( result.dual.size() != structure_.constraints )
      {
         return detail::DimensionMismatch(
            "dual result", result.dual.size(), structure_.constraints);
      }
      return {};
   }

   AnyNlpProblem problem_;
   const NlpStructure structure_;
   std::vector<Number> hessian_product_;
   std::vector<Number> jacobian_transpose_product_;
   std::vector<Number> jacobian_product_;
};
} // namespace Ipopt::Cxx23

#endif
