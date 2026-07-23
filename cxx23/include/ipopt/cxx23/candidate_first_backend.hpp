// Copyright (C) 2026 Ipopt contributors.
// All Rights Reserved.
// This file is distributed under the Eclipse Public License.

#ifndef IPOPT_CXX23_CANDIDATE_FIRST_BACKEND_HPP
#define IPOPT_CXX23_CANDIDATE_FIRST_BACKEND_HPP

#include <ipopt/cxx23/primal_dual_kkt_operator.hpp>

#include <anyany/anyany.hpp>
#include <anyany/anyany_macro.hpp>

#include <concepts>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

namespace Ipopt::Cxx23
{
/** Views for one reference-free primal-dual solve attempt.
 *
 * The backend owns perturbation selection, inertia retries, refinement, and
 * any quality escalation. These views are valid only for the duration of
 * candidate_first_solve(); retaining them is an error. direction_output is an
 * optional caller-owned, non-publishing scratch span. A backend may overwrite
 * it even when the solve later fails; the caller must expose it only after a
 * successful result explicitly selects it and all outer validation passes.
 */
struct CandidateFirstSolveRequest
{
   CandidateFirstSolveRequest(
      PrimalDualKktOperator&  kkt_value,
      PrimalDualState         state_value,
      std::span<const Number> rhs_value,
      Index                   required_negative_eigenvalues_value = 0,
      bool                    restoration_problem_value = false,
      std::span<Number>       direction_output_value = {}
   )
      : kkt(kkt_value),
        state(state_value),
        rhs(rhs_value),
        required_negative_eigenvalues(
           required_negative_eigenvalues_value),
        restoration_problem(restoration_problem_value),
        direction_output(direction_output_value)
   {
   }

   PrimalDualKktOperator& kkt;
   PrimalDualState state;
   std::span<const Number> rhs;
   Index required_negative_eigenvalues = 0;
   bool restoration_problem = false;
   std::span<Number> direction_output;
};

enum class CandidateFirstInertiaCertainty
{
   unavailable,
   exact
};

struct CandidateFirstInertiaCertificate
{
   CandidateFirstInertiaCertainty certainty =
      CandidateFirstInertiaCertainty::unavailable;
   Index negative_eigenvalues = 0;
};

struct CandidateFirstWorkStatistics
{
   Index factorizations = 0;
   Index backsolves = 0;
   Index kkt_applications = 0;
   Index derivative_product_requests = 0;
   Index refinement_steps = 0;
   Index quality_improvements = 0;
};

/** Views for refining the last successful candidate with its current factor.
 *
 * The backend must treat direction as the initial guess and overwrite it only
 * with a finite refined candidate.  This call is synchronous: no request view
 * may be retained.  A supported implementation reuses the factor produced by
 * the immediately preceding solve() and must report zero new factorizations.
 */
struct CandidateFirstRefinementRequest
{
   CandidateFirstRefinementRequest(
      PrimalDualKktOperator&  kkt_value,
      PrimalDualState         state_value,
      std::span<const Number> rhs_value,
      std::span<Number>       direction_value,
      bool                    restoration_problem_value = false
   )
      : kkt(kkt_value),
        state(state_value),
        rhs(rhs_value),
        direction(direction_value),
        restoration_problem(restoration_problem_value)
   {
   }

   PrimalDualKktOperator& kkt;
   PrimalDualState state;
   std::span<const Number> rhs;
   std::span<Number> direction;
   bool restoration_problem = false;
};

struct CandidateFirstRefinementResult
{
   CandidateFirstWorkStatistics work;
   bool supported = false;
   bool converged = false;
};

/** A backend's final proposed solution of KKT * direction = rhs.
 *
 * direction is deliberately unscaled by PDSystemSolver's alpha. Normally it
 * owns the proposal. A backend may instead leave it empty and set
 * direction_written_to_request_output after completely writing the request's
 * exact-size output span. The wrapper independently checks dimensions,
 * finiteness, the exact inertia certificate, and the true residual with
 * accepted_regularization before committing either representation.
 */
struct CandidateFirstSolveResult
{
   std::vector<Number> direction;
   PrimalDualRegularization accepted_regularization{0., 0., 0., 0.};
   CandidateFirstInertiaCertificate inertia;
   CandidateFirstWorkStatistics work;
   bool converged = false;
   bool direction_written_to_request_output = false;
};

template <class Backend>
concept CandidateFirstBackend = requires(
   Backend& backend,
   CandidateFirstSolveRequest request
)
{
   { backend.solve(request) }
      -> std::same_as<EvaluationValue<CandidateFirstSolveResult>>;
};

template <class Backend>
concept CandidateFirstRefinementBackend = requires(
   Backend& backend,
   CandidateFirstRefinementRequest request
)
{
   { backend.refine(request) }
      -> std::same_as<EvaluationValue<CandidateFirstRefinementResult>>;
};

template <CandidateFirstBackend Backend>
class CandidateFirstBackendAdapter
{
public:
   explicit CandidateFirstBackendAdapter(Backend backend)
      : backend_(std::move(backend))
   {
   }

   EvaluationValue<CandidateFirstSolveResult> candidate_first_solve(
      CandidateFirstSolveRequest request
   )
   {
      return backend_.solve(request);
   }

   EvaluationValue<CandidateFirstRefinementResult> candidate_first_refine(
      CandidateFirstRefinementRequest request
   )
   {
      if constexpr( CandidateFirstRefinementBackend<Backend> )
      {
         return backend_.refine(request);
      }
      else
      {
         return CandidateFirstRefinementResult{};
      }
   }

private:
   Backend backend_;
};

anyany_method(candidate_first_solve,
   (&self, CandidateFirstSolveRequest request)
      requires(self.candidate_first_solve(request))
      ->EvaluationValue<CandidateFirstSolveResult>);

anyany_method(candidate_first_refine,
   (&self, CandidateFirstRefinementRequest request)
      requires(self.candidate_first_refine(request))
      ->EvaluationValue<CandidateFirstRefinementResult>);

using AnyCandidateFirstBackend = aa::any_with<
   aa::move,
   aa::type_info,
   candidate_first_solve,
   candidate_first_refine>;

using SharedCandidateFirstBackend = std::shared_ptr<AnyCandidateFirstBackend>;

template <CandidateFirstBackend Backend>
SharedCandidateFirstBackend MakeCandidateFirstBackend(Backend backend)
{
   return std::make_shared<AnyCandidateFirstBackend>(
      CandidateFirstBackendAdapter<Backend>(std::move(backend)));
}

template <class Factory>
concept CandidateFirstBackendFactory = requires(
   Factory&                 factory,
   PrimalDualKktOperator&   kkt,
   bool                     restoration_problem
)
{
   typename std::invoke_result_t<
      Factory&, PrimalDualKktOperator&, bool>::value_type;
   typename std::invoke_result_t<
      Factory&, PrimalDualKktOperator&, bool>::error_type;
   requires std::same_as<
      typename std::invoke_result_t<
         Factory&, PrimalDualKktOperator&, bool>::error_type,
      EvaluationError>;
   requires CandidateFirstBackend<
      typename std::invoke_result_t<
         Factory&, PrimalDualKktOperator&, bool>::value_type>;
   { factory(kkt, restoration_problem) } -> std::same_as<
      std::invoke_result_t<Factory&, PrimalDualKktOperator&, bool>>;
};

/** Lazily constructs a typed backend for the current full-KKT structure.
 *
 * A successful backend is reused while the KKT fingerprint and restoration
 * role remain unchanged. Reconfiguration is transactional: factory failure
 * leaves the prior backend intact, so a later request for the prior structure
 * can still proceed. The factory and backend are invoked synchronously and
 * must not retain the supplied KKT reference or request views. This object is
 * intentionally not safe for concurrent solves.
 */
template <CandidateFirstBackendFactory Factory>
class LazyCandidateFirstBackend
{
   using FactoryResult =
      std::invoke_result_t<Factory&, PrimalDualKktOperator&, bool>;
   using Backend = typename FactoryResult::value_type;

public:
   explicit LazyCandidateFirstBackend(Factory factory)
      : factory_(std::move(factory))
   {
   }

   EvaluationValue<CandidateFirstSolveResult> solve(
      CandidateFirstSolveRequest request
   )
   {
      StructureFingerprintResult fingerprint =
         request.kkt.structure_fingerprint();
      if( !fingerprint )
      {
         return std::unexpected(fingerprint.error());
      }
      if( !backend_ || !fingerprint_ || *fingerprint_ != *fingerprint ||
          restoration_problem_ != request.restoration_problem )
      {
         FactoryResult created =
            factory_(request.kkt, request.restoration_problem);
         if( !created )
         {
            return std::unexpected(created.error());
         }
         backend_.emplace(std::move(*created));
         fingerprint_ = *fingerprint;
         restoration_problem_ = request.restoration_problem;
      }
      return backend_->solve(request);
   }

   EvaluationValue<CandidateFirstRefinementResult> refine(
      CandidateFirstRefinementRequest request
   ) requires CandidateFirstRefinementBackend<Backend>
   {
      if( !backend_ )
      {
         return std::unexpected(EvaluationError{
            EvaluationErrorCode::numeric_mismatch,
            "candidate refinement requested before a successful backend solve"
         });
      }
      return backend_->refine(request);
   }

private:
   Factory factory_;
   std::optional<Backend> backend_;
   std::optional<StructureFingerprint> fingerprint_;
   bool restoration_problem_ = false;
};

template <CandidateFirstBackendFactory Factory>
SharedCandidateFirstBackend MakeLazyCandidateFirstBackend(Factory factory)
{
   return MakeCandidateFirstBackend(
      LazyCandidateFirstBackend<Factory>(std::move(factory)));
}
} // namespace Ipopt::Cxx23

#endif
