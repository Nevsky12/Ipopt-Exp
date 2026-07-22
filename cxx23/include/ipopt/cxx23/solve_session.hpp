// Copyright (C) 2026 Ipopt contributors.
// All Rights Reserved.
// This file is distributed under the Eclipse Public License.

#ifndef IPOPT_CXX23_SOLVE_SESSION_HPP
#define IPOPT_CXX23_SOLVE_SESSION_HPP

#include <ipopt/cxx23/fgmres.hpp>
#include <ipopt/cxx23/primal_dual_kkt_operator.hpp>

#include <concepts>
#include <span>
#include <utility>

namespace Ipopt::Cxx23
{
struct SolveSessionStatistics
{
   Index solve_requests = 0;
   Index converged_solves = 0;
   Index structure_rejections = 0;
   Index preconditioner_rejections = 0;
   Index evaluation_failures = 0;
};

template <class Preconditioner>
concept KktStateValidatedPreconditioner = requires(
   Preconditioner&        preconditioner,
   PrimalDualKktOperator& kkt,
   PrimalDualState        state
)
{
   { preconditioner.validate(kkt, state) } -> std::same_as<EvaluationResult>;
};

/** Reuses Krylov storage only across structurally compatible KKT operators.
 *
 * Numeric model data and the primal-dual state are deliberately supplied on
 * every solve. A matching fingerprint permits workspace reuse; it does not
 * cache derivative values. The session is movable, not copyable, and is not
 * safe for concurrent calls.
 */
class PrimalDualSolveSession
{
public:
   PrimalDualSolveSession(
      Index                dimension,
      StructureFingerprint fingerprint,
      FgmresOptions        options = {}
   )
      : fingerprint_(fingerprint),
        solver_(dimension, options)
   {
   }

   PrimalDualSolveSession(const PrimalDualSolveSession&) = delete;
   PrimalDualSolveSession& operator=(const PrimalDualSolveSession&) = delete;
   PrimalDualSolveSession(PrimalDualSolveSession&&) = default;
   PrimalDualSolveSession& operator=(PrimalDualSolveSession&&) = delete;

   Index dimension() const noexcept
   {
      return solver_.dimension();
   }

   StructureFingerprint fingerprint() const noexcept
   {
      return fingerprint_;
   }

   const FgmresOptions& options() const noexcept
   {
      return solver_.options();
   }

   const SolveSessionStatistics& statistics() const noexcept
   {
      return statistics_;
   }

   EvaluationValue<bool> compatible(PrimalDualKktOperator& kkt)
   {
      StructureFingerprintResult current = kkt.structure_fingerprint();
      if( !current )
      {
         return std::unexpected(current.error());
      }
      return kkt.flat_dimension() == dimension() && *current == fingerprint_;
   }

   template <FgmresPreconditioner Preconditioner>
   EvaluationValue<FgmresResult> solve(
      PrimalDualKktOperator& kkt,
      PrimalDualState        state,
      Preconditioner&&       apply_preconditioner,
      std::span<const Number> right_hand_side,
      std::span<Number>       solution
   )
   {
      ++statistics_.solve_requests;
      EvaluationValue<bool> is_compatible = compatible(kkt);
      if( !is_compatible )
      {
         ++statistics_.evaluation_failures;
         return std::unexpected(is_compatible.error());
      }
      if( !*is_compatible )
      {
         ++statistics_.structure_rejections;
         return std::unexpected(EvaluationError{
            EvaluationErrorCode::structure_mismatch,
            "KKT structure does not match the solve session"
         });
      }
      if constexpr( KktStateValidatedPreconditioner<Preconditioner> )
      {
         if( EvaluationResult valid = apply_preconditioner.validate(kkt, state); !valid )
         {
            ++statistics_.preconditioner_rejections;
            return std::unexpected(valid.error());
         }
      }

      const auto apply_operator = [&](std::span<const Number> input, std::span<Number> output)
         -> EvaluationResult
      {
         return kkt.apply_flat(state, input, output);
      };
      EvaluationValue<FgmresResult> result = solver_.solve(
         apply_operator,
         std::forward<Preconditioner>(apply_preconditioner),
         right_hand_side,
         solution);
      if( !result )
      {
         ++statistics_.evaluation_failures;
      }
      else if( result->converged() )
      {
         ++statistics_.converged_solves;
      }
      return result;
   }

   EvaluationValue<FgmresResult> solve(
      PrimalDualKktOperator& kkt,
      PrimalDualState        state,
      std::span<const Number> right_hand_side,
      std::span<Number>       solution
   )
   {
      IdentityPreconditioner identity;
      return solve(kkt, state, identity, right_hand_side, solution);
   }

private:
   const StructureFingerprint fingerprint_;
   FgmresSolver solver_;
   SolveSessionStatistics statistics_;
};

inline EvaluationValue<PrimalDualSolveSession> MakePrimalDualSolveSession(
   PrimalDualKktOperator& kkt,
   FgmresOptions           options = {}
)
{
   StructureFingerprintResult fingerprint = kkt.structure_fingerprint();
   if( !fingerprint )
   {
      return std::unexpected(fingerprint.error());
   }
   return PrimalDualSolveSession(kkt.flat_dimension(), *fingerprint, options);
}
} // namespace Ipopt::Cxx23

#endif
