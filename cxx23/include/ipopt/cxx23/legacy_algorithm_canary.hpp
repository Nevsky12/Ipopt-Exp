// Copyright (C) 2026 Ipopt contributors.
// All Rights Reserved.
// This file is distributed under the Eclipse Public License.

#ifndef IPOPT_CXX23_LEGACY_ALGORITHM_CANARY_HPP
#define IPOPT_CXX23_LEGACY_ALGORITHM_CANARY_HPP

#include <ipopt/cxx23/candidate_first_backend.hpp>
#include <ipopt/cxx23/legacy_aug_system_backend.hpp>
#include <ipopt/cxx23/legacy_ipopt_bridge.hpp>
#include <ipopt/cxx23/legacy_matrix_snapshot.hpp>
#include <ipopt/cxx23/solve_session.hpp>

#include <IpAlgBuilder.hpp>
#include <IpOrigIpoptNLP.hpp>
#include <IpTNLPAdapter.hpp>
#include <IpTripletHelper.hpp>

#if __has_include(<IpPDFullSpaceSolver.hpp>)
#include <IpPDFullSpaceSolver.hpp>
#define IPOPT_CXX23_HAS_PDFULLSPACE_HEADER 1
#else
#define IPOPT_CXX23_HAS_PDFULLSPACE_HEADER 0
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <concepts>
#include <cstdint>
#include <exception>
#include <expected>
#include <limits>
#include <new>
#include <numeric>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace Ipopt::Cxx23
{
inline constexpr Number LegacyStablePrecisionTolerance(
   Number double_precision_tolerance,
   Number stable_epsilon_multiplier
) noexcept
{
   return std::max(
      double_precision_tolerance,
      stable_epsilon_multiplier *
         static_cast<Number>(std::numeric_limits<::Ipopt::Number>::epsilon()));
}

enum class LegacyAlgorithmCanaryMode
{
   shadow,
   validated_replacement,
   candidate_first
};

struct LegacyAlgorithmCanaryOptions
{
   FgmresOptions fgmres{
      .restart = 20,
      .maximum_iterations = 20,
      .relative_tolerance = LegacyStablePrecisionTolerance(1e-11, 128.),
      .absolute_tolerance = LegacyStablePrecisionTolerance(1e-13, 128.),
      .breakdown_tolerance = 1e-14,
      .reorthogonalize = true,
      .check_finite = true
   };
   Number direction_relative_tolerance =
      LegacyStablePrecisionTolerance(1e-8, 512.);
   Number residual_relative_tolerance =
      LegacyStablePrecisionTolerance(1e-9, 512.);
   LegacyAlgorithmCanaryMode mode = LegacyAlgorithmCanaryMode::shadow;
   /** Required by candidate_first. The shared AnyAny backend is invoked
    * synchronously by main and restoration solvers; it must not retain request
    * views and must provide its own perturbation/inertia/quality policy. */
   SharedCandidateFirstBackend candidate_first_backend;
   bool fail_on_error = false;
   bool fail_on_mismatch = false;
};

struct LegacyAlgorithmCanaryStatistics
{
   Index reference_solve_requests = 0;
   Index checked_solves = 0;
   Index matching_solves = 0;
   Index mismatching_solves = 0;
   Index skipped_solves = 0;
   Index evaluation_failures = 0;
   Index restoration_reference_solve_requests = 0;
   Index restoration_checked_solves = 0;
   Index restoration_matching_solves = 0;
   Index restoration_mismatching_solves = 0;
   Index restoration_skipped_solves = 0;
   Index restoration_evaluation_failures = 0;
   Index replacement_commit_requests = 0;
   Index committed_replacements = 0;
   Index replacement_commit_failures = 0;
   Index restoration_committed_replacements = 0;
   Index candidate_first_requests = 0;
   Index candidate_first_accepted = 0;
   Index candidate_first_fallbacks = 0;
   Index candidate_first_failures = 0;
   Index candidate_first_inertia_rejections = 0;
   Index candidate_first_residual_rejections = 0;
   Index candidate_first_commit_failures = 0;
   Index candidate_first_factorizations = 0;
   Index candidate_first_backsolves = 0;
   Index candidate_first_kkt_applications = 0;
   Index candidate_first_derivative_product_requests = 0;
   Index candidate_first_validation_kkt_applications = 0;
   Index candidate_first_validation_derivative_product_requests = 0;
   Index candidate_first_refinement_steps = 0;
   Index candidate_first_quality_improvements = 0;
   Index restoration_candidate_first_requests = 0;
   Index restoration_candidate_first_accepted = 0;
   Index restoration_candidate_first_fallbacks = 0;
   std::string last_candidate_fallback_reason;
   Number maximum_direction_relative_error = 0.;
   Number maximum_residual_relative_error = 0.;
};

namespace legacy_algorithm_canary_detail
{
inline EvaluationError CanaryError(std::string message)
{
   return EvaluationError{
      EvaluationErrorCode::model_failure,
      std::move(message)
   };
}

inline EvaluationValue<std::vector<Number>> CopyVector(
   const ::Ipopt::Vector& vector,
   std::string_view       name
)
{
   EvaluationValue<Index> dimension =
      legacy_ipopt_detail::CheckedIndex(vector.Dim(), name);
   if( !dimension )
   {
      return std::unexpected(dimension.error());
   }
   std::vector<::Ipopt::Number> stable(*dimension);
   ::Ipopt::TripletHelper::FillValuesFromVector(
      vector.Dim(), vector, stable.data());
   std::vector<Number> result(*dimension);
   for( Index i = 0; i < *dimension; ++i )
   {
      result[i] = static_cast<Number>(stable[i]);
      if( !std::isfinite(result[i]) )
      {
         return std::unexpected(EvaluationError{
            EvaluationErrorCode::nonfinite_output,
            std::string(name) + " contains a value not representable by C++23 Number"
         });
      }
   }
   return result;
}

inline void Append(
   const std::vector<Number>& source,
   std::vector<Number>&       destination
)
{
   destination.insert(destination.end(), source.begin(), source.end());
}

inline Number RelativeInfinityError(
   std::span<const Number> actual,
   std::span<const Number> expected
) noexcept
{
   Number difference = 0.;
   Number scale = 1.;
   for( Index i = 0; i < actual.size(); ++i )
   {
      difference = std::max(difference, std::abs(actual[i] - expected[i]));
      scale = std::max(scale, std::abs(expected[i]));
   }
   return difference / scale;
}

struct Comparison
{
   Number direction_relative_error = 0.;
   Number residual_relative_error = 0.;
   bool converged = false;
   std::vector<Number> direction;
   PrimalDualRegularization regularization{0., 0., 0., 0.};
   CandidateFirstWorkStatistics work;
};

#if IPOPT_CXX23_HAS_PDFULLSPACE_HEADER
template <class FullSpaceSolver>
::Ipopt::AugSystemSolver* AugmentedSolverIfAvailable(
   ::Ipopt::PDSystemSolver& reference
)
{
   auto* full_space = dynamic_cast<FullSpaceSolver*>(&reference);
   if( full_space == nullptr )
   {
      return nullptr;
   }
   if constexpr( requires(FullSpaceSolver& solver) {
      { solver.AugmentedSystemSolver() }
         -> std::same_as<::Ipopt::AugSystemSolver&>;
   } )
   {
      return &full_space->AugmentedSystemSolver();
   }
   return nullptr;
}
#endif
} // namespace legacy_algorithm_canary_detail

/** Runs a guarded C++23 full-system path around the stable solver.
 *
 * shadow and validated_replacement run the reference policy first. The opt-in
 * candidate_first mode instead asks an independent AnyAny backend to own
 * perturbation/inertia/refinement/quality policy, verifies its exact inertia
 * certificate and true residual, and calls the complete reference solver only
 * on rejection. Direct OrigIpoptNLP/TNLPAdapter solves use the callback bridge;
 * wrapped formulations such as restoration use an owning W/J snapshot.
 * Inexact, refinement-only, and limited-memory calls are always delegated.
 */
class LegacyAlgorithmCanaryPDSystemSolver final : public ::Ipopt::PDSystemSolver
{
public:
   LegacyAlgorithmCanaryPDSystemSolver(
      ::Ipopt::PDSystemSolver&                    reference_solver,
      ::Ipopt::AugSystemSolver&                   augmented_solver,
      LegacyAlgorithmCanaryOptions                options,
      LegacyAlgorithmCanaryStatistics&            statistics
   )
      : reference_solver_(&reference_solver),
        augmented_solver_(&augmented_solver),
        options_(std::move(options)),
        statistics_(statistics)
   {
   }

   bool InitializeImpl(
      const ::Ipopt::OptionsList& options,
      const std::string&          prefix
   ) override
   {
      cached_coordinate_problem_.reset();
      cached_coordinate_adapter_ = nullptr;
      cached_orig_nlp_ = nullptr;
      std::string hessian_approximation;
      options.GetStringValue(
         "hessian_approximation", hessian_approximation, prefix);
      exact_hessian_ = hessian_approximation == "exact";
      restoration_problem_ = IpNLP().objective_depends_on_mu();
      return reference_solver_->Initialize(
         Jnlst(), IpNLP(), IpData(), IpCq(), options, prefix);
   }

   bool Solve(
      ::Ipopt::Number                alpha,
      ::Ipopt::Number                beta,
      const ::Ipopt::IteratesVector& rhs,
      ::Ipopt::IteratesVector&       result,
      bool                           allow_inexact = false,
      bool                           improve_solution = false
   ) override
   {
      const bool eligible = exact_hessian_ && !allow_inexact &&
         !improve_solution && beta == 0. && alpha != 0. &&
         std::isfinite(static_cast<Number>(alpha));

      if( options_.mode == LegacyAlgorithmCanaryMode::candidate_first )
      {
         if( !eligible )
         {
            CountSkippedSolve();
            return SolveReference(
               alpha, beta, rhs, result, allow_inexact, improve_solution);
         }

         ++statistics_.candidate_first_requests;
         if( restoration_problem_ )
         {
            ++statistics_.restoration_candidate_first_requests;
         }
         try
         {
            EvaluationValue<legacy_algorithm_canary_detail::Comparison>
               comparison = [&]()
               {
                  auto& timing =
                     IpData().TimingStats().PDSystemSolverTotal();
                  timing.Start();
                  try
                  {
                     EvaluationValue<legacy_algorithm_canary_detail::Comparison>
                        evaluated = EvaluateSolve(alpha, rhs, nullptr);
                     timing.End();
                     return evaluated;
                  }
                  catch( ... )
                  {
                     timing.EndIfStarted();
                     throw;
                  }
               }();
            if( !comparison )
            {
               RecordCandidateFallback(
                  CandidateFallbackCause::failure,
                  comparison.error().message);
               return SolveReference(
                  alpha, beta, rhs, result, allow_inexact, improve_solution);
            }

            statistics_.maximum_residual_relative_error = std::max(
               statistics_.maximum_residual_relative_error,
               comparison->residual_relative_error);
            if( !comparison->converged ||
                comparison->residual_relative_error >
                   options_.residual_relative_tolerance )
            {
               RecordCandidateFallback(
                  CandidateFallbackCause::residual,
                  "candidate did not satisfy the true-residual gate");
               Jnlst().Printf(
                  ::Ipopt::J_WARNING, ::Ipopt::J_LINEAR_ALGEBRA,
                  "C++23 candidate-first fallback: converged=%d residual=%e\n",
                  comparison->converged ? 1 : 0,
                  comparison->residual_relative_error);
               return SolveReference(
                  alpha, beta, rhs, result, allow_inexact, improve_solution);
            }

            ++statistics_.replacement_commit_requests;
            EvaluationResult committed = CommitDirection(
               comparison->direction, result);
            if( !committed )
            {
               ++statistics_.replacement_commit_failures;
               RecordCandidateFallback(
                  CandidateFallbackCause::commit,
                  committed.error().message);
               return SolveReference(
                  alpha, beta, rhs, result, allow_inexact, improve_solution);
            }
            CommitRegularization(comparison->regularization);
            ++statistics_.candidate_first_accepted;
            if( restoration_problem_ )
            {
               ++statistics_.restoration_candidate_first_accepted;
            }
            Jnlst().Printf(
               ::Ipopt::J_DETAILED, ::Ipopt::J_LINEAR_ALGEBRA,
               "C++23 candidate-first accepted: residual=%e "
               "factorizations=%zu backsolves=%zu\n",
               comparison->residual_relative_error,
               comparison->work.factorizations,
               comparison->work.backsolves);
            return true;
         }
         catch( const std::bad_alloc& )
         {
            throw;
         }
         catch( const ::Ipopt::IpoptException& exception )
         {
            RecordCandidateFallback(
               CandidateFallbackCause::failure, exception.Message());
         }
         catch( const std::exception& exception )
         {
            RecordCandidateFallback(
               CandidateFallbackCause::failure, exception.what());
         }
         catch( ... )
         {
            RecordCandidateFallback(
               CandidateFallbackCause::failure, "unknown exception");
         }
         return SolveReference(
            alpha, beta, rhs, result, allow_inexact, improve_solution);
      }

      const bool solved = SolveReference(
         alpha, beta, rhs, result, allow_inexact, improve_solution);
      if( !solved )
      {
         return false;
      }

      if( !eligible )
      {
         CountSkippedSolve();
         return true;
      }

      try
      {
         EvaluationValue<legacy_algorithm_canary_detail::Comparison> comparison =
            EvaluateSolve(alpha, rhs, &result);
         if( !comparison )
         {
            ++statistics_.evaluation_failures;
            if( restoration_problem_ )
            {
               ++statistics_.restoration_evaluation_failures;
            }
            Jnlst().Printf(
               ::Ipopt::J_WARNING, ::Ipopt::J_LINEAR_ALGEBRA,
               "C++23 canary skipped after evaluation failure: %s\n",
               comparison.error().message.c_str());
            return !options_.fail_on_error;
         }

         ++statistics_.checked_solves;
         if( restoration_problem_ )
         {
            ++statistics_.restoration_checked_solves;
         }
         statistics_.maximum_direction_relative_error = std::max(
            statistics_.maximum_direction_relative_error,
            comparison->direction_relative_error);
         statistics_.maximum_residual_relative_error = std::max(
            statistics_.maximum_residual_relative_error,
            comparison->residual_relative_error);
         const bool matches = comparison->converged &&
            comparison->direction_relative_error <=
               options_.direction_relative_tolerance &&
            comparison->residual_relative_error <=
               options_.residual_relative_tolerance;
         if( matches )
         {
            ++statistics_.matching_solves;
            if( restoration_problem_ )
            {
               ++statistics_.restoration_matching_solves;
            }
            Jnlst().Printf(
               ::Ipopt::J_DETAILED, ::Ipopt::J_LINEAR_ALGEBRA,
               "C++23 canary match: direction=%e residual=%e\n",
               comparison->direction_relative_error,
               comparison->residual_relative_error);
            if( options_.mode ==
                LegacyAlgorithmCanaryMode::validated_replacement )
            {
               ++statistics_.replacement_commit_requests;
               EvaluationResult committed = CommitDirection(
                  comparison->direction, result);
               if( !committed )
               {
                  ++statistics_.replacement_commit_failures;
                  Jnlst().Printf(
                     ::Ipopt::J_WARNING, ::Ipopt::J_LINEAR_ALGEBRA,
                     "C++23 replacement kept the reference direction after "
                     "commit failure: %s\n",
                     committed.error().message.c_str());
                  return !options_.fail_on_error;
               }
               ++statistics_.committed_replacements;
               if( restoration_problem_ )
               {
                  ++statistics_.restoration_committed_replacements;
               }
            }
            return true;
         }

         ++statistics_.mismatching_solves;
         if( restoration_problem_ )
         {
            ++statistics_.restoration_mismatching_solves;
         }
         Jnlst().Printf(
            ::Ipopt::J_WARNING, ::Ipopt::J_LINEAR_ALGEBRA,
            "C++23 canary mismatch: converged=%d direction=%e residual=%e\n",
            comparison->converged ? 1 : 0,
            comparison->direction_relative_error,
            comparison->residual_relative_error);
         return !options_.fail_on_mismatch;
      }
      catch( const std::bad_alloc& )
      {
         throw;
      }
      catch( const ::Ipopt::IpoptException& exception )
      {
         ++statistics_.evaluation_failures;
         if( restoration_problem_ )
         {
            ++statistics_.restoration_evaluation_failures;
         }
         Jnlst().Printf(
            ::Ipopt::J_WARNING, ::Ipopt::J_LINEAR_ALGEBRA,
            "C++23 canary caught Ipopt exception: %s\n",
            exception.Message().c_str());
         return !options_.fail_on_error;
      }
      catch( const std::exception& exception )
      {
         ++statistics_.evaluation_failures;
         if( restoration_problem_ )
         {
            ++statistics_.restoration_evaluation_failures;
         }
         Jnlst().Printf(
            ::Ipopt::J_WARNING, ::Ipopt::J_LINEAR_ALGEBRA,
            "C++23 canary caught exception: %s\n", exception.what());
         return !options_.fail_on_error;
      }
      catch( ... )
      {
         ++statistics_.evaluation_failures;
         if( restoration_problem_ )
         {
            ++statistics_.restoration_evaluation_failures;
         }
         Jnlst().Printf(
            ::Ipopt::J_WARNING, ::Ipopt::J_LINEAR_ALGEBRA,
            "C++23 canary caught an unknown exception\n");
         return !options_.fail_on_error;
      }
   }

private:
   enum class CandidateFallbackCause
   {
      failure,
      residual,
      commit
   };

   bool SolveReference(
      ::Ipopt::Number                alpha,
      ::Ipopt::Number                beta,
      const ::Ipopt::IteratesVector& rhs,
      ::Ipopt::IteratesVector&       result,
      bool                           allow_inexact,
      bool                           improve_solution
   )
   {
      ++statistics_.reference_solve_requests;
      if( restoration_problem_ )
      {
         ++statistics_.restoration_reference_solve_requests;
      }
      return reference_solver_->Solve(
         alpha, beta, rhs, result, allow_inexact, improve_solution);
   }

   void CountSkippedSolve()
   {
      ++statistics_.skipped_solves;
      if( restoration_problem_ )
      {
         ++statistics_.restoration_skipped_solves;
      }
   }

   void AddCandidateWork(const CandidateFirstWorkStatistics& work)
   {
      statistics_.candidate_first_factorizations += work.factorizations;
      statistics_.candidate_first_backsolves += work.backsolves;
      statistics_.candidate_first_kkt_applications += work.kkt_applications;
      statistics_.candidate_first_derivative_product_requests +=
         work.derivative_product_requests;
      statistics_.candidate_first_refinement_steps += work.refinement_steps;
      statistics_.candidate_first_quality_improvements +=
         work.quality_improvements;
   }

   void RecordCandidateFallback(
      CandidateFallbackCause cause,
      std::string_view        reason
   )
   {
      statistics_.last_candidate_fallback_reason = reason;
      ++statistics_.candidate_first_fallbacks;
      if( restoration_problem_ )
      {
         ++statistics_.restoration_candidate_first_fallbacks;
      }
      if( cause == CandidateFallbackCause::residual )
      {
         ++statistics_.candidate_first_residual_rejections;
      }
      else
      {
         ++statistics_.candidate_first_failures;
         ++statistics_.evaluation_failures;
         if( restoration_problem_ )
         {
            ++statistics_.restoration_evaluation_failures;
         }
      }
      if( cause == CandidateFallbackCause::commit )
      {
         ++statistics_.candidate_first_commit_failures;
      }
      Jnlst().Printf(
         ::Ipopt::J_WARNING, ::Ipopt::J_LINEAR_ALGEBRA,
         "C++23 candidate-first using stable fallback: %s\n",
         std::string(reason).c_str());
   }

   void CommitRegularization(PrimalDualRegularization regularization)
   {
      const ::Ipopt::Number delta_x =
         static_cast<::Ipopt::Number>(regularization.x);
      const ::Ipopt::Number delta_s =
         static_cast<::Ipopt::Number>(regularization.s);
      const ::Ipopt::Number delta_c =
         static_cast<::Ipopt::Number>(regularization.c);
      const ::Ipopt::Number delta_d =
         static_cast<::Ipopt::Number>(regularization.d);
      IpData().setPDPert(delta_x, delta_s, delta_c, delta_d);
      IpData().Set_info_regu_x(delta_x);
   }

   EvaluationValue<legacy_algorithm_canary_detail::Comparison> EvaluateSolve(
      ::Ipopt::Number                alpha,
      const ::Ipopt::IteratesVector& rhs,
      const ::Ipopt::IteratesVector* reference_result
   )
   {
      auto* orig_nlp = dynamic_cast<::Ipopt::OrigIpoptNLP*>(&IpNLP());
      ::Ipopt::TNLPAdapter* adapter = nullptr;
      if( orig_nlp != nullptr )
      {
         adapter = dynamic_cast<::Ipopt::TNLPAdapter*>(
            ::Ipopt::GetRawPtr(orig_nlp->nlp()));
      }

      EvaluationValue<std::vector<Number>> x =
         legacy_algorithm_canary_detail::CopyVector(
            *IpData().curr()->x(), "current x");
      EvaluationValue<std::vector<Number>> y_c =
         legacy_algorithm_canary_detail::CopyVector(
            *IpData().curr()->y_c(), "current equality multipliers");
      EvaluationValue<std::vector<Number>> y_d =
         legacy_algorithm_canary_detail::CopyVector(
            *IpData().curr()->y_d(), "current inequality multipliers");
      EvaluationValue<std::vector<Number>> z_lower =
         legacy_algorithm_canary_detail::CopyVector(
            *IpData().curr()->z_L(), "current lower-bound multipliers");
      EvaluationValue<std::vector<Number>> z_upper =
         legacy_algorithm_canary_detail::CopyVector(
            *IpData().curr()->z_U(), "current upper-bound multipliers");
      EvaluationValue<std::vector<Number>> v_lower =
         legacy_algorithm_canary_detail::CopyVector(
            *IpData().curr()->v_L(), "current slack-lower multipliers");
      EvaluationValue<std::vector<Number>> v_upper =
         legacy_algorithm_canary_detail::CopyVector(
            *IpData().curr()->v_U(), "current slack-upper multipliers");
      EvaluationValue<std::vector<Number>> slack_x_lower =
         legacy_algorithm_canary_detail::CopyVector(
            *IpCq().curr_slack_x_L(), "current x lower slack");
      EvaluationValue<std::vector<Number>> slack_x_upper =
         legacy_algorithm_canary_detail::CopyVector(
            *IpCq().curr_slack_x_U(), "current x upper slack");
      EvaluationValue<std::vector<Number>> slack_s_lower =
         legacy_algorithm_canary_detail::CopyVector(
            *IpCq().curr_slack_s_L(), "current s lower slack");
      EvaluationValue<std::vector<Number>> slack_s_upper =
         legacy_algorithm_canary_detail::CopyVector(
            *IpCq().curr_slack_s_U(), "current s upper slack");
      if( !x || !y_c || !y_d || !z_lower || !z_upper || !v_lower || !v_upper ||
          !slack_x_lower || !slack_x_upper || !slack_s_lower || !slack_s_upper )
      {
         return std::unexpected(legacy_algorithm_canary_detail::CanaryError(
            "C++23 canary could not copy the current primal-dual state"));
      }

      ::Ipopt::SmartPtr<const ::Ipopt::SymMatrix> hessian = IpData().W();
      ::Ipopt::SmartPtr<const ::Ipopt::Matrix> jacobian_equalities =
         IpCq().curr_jac_c();
      ::Ipopt::SmartPtr<const ::Ipopt::Matrix> jacobian_inequalities =
         IpCq().curr_jac_d();
      ::Ipopt::SmartPtr<const ::Ipopt::Vector> complementarity_x_diagonal =
         IpCq().curr_sigma_x();
      ::Ipopt::SmartPtr<const ::Ipopt::Vector> complementarity_s_diagonal =
         IpCq().curr_sigma_s();
      if( !::Ipopt::IsValid(hessian) ||
          !::Ipopt::IsValid(jacobian_equalities) ||
          !::Ipopt::IsValid(jacobian_inequalities) ||
          !::Ipopt::IsValid(complementarity_x_diagonal) ||
          !::Ipopt::IsValid(complementarity_s_diagonal) )
      {
         return std::unexpected(legacy_algorithm_canary_detail::CanaryError(
            "C++23 canary stable KKT matrices are unavailable"));
      }

      EvaluationValue<AnyNlpProblem> problem = [&]()
      {
         if( adapter == nullptr || orig_nlp == nullptr )
         {
            return MakeLegacyMatrixSnapshotProblem({
               .hessian = ::Ipopt::GetRawPtr(hessian),
               .jacobian_equalities = ::Ipopt::GetRawPtr(jacobian_equalities),
               .jacobian_inequalities = ::Ipopt::GetRawPtr(jacobian_inequalities),
               .structural_revision = 1
            });
         }
         if( !cached_coordinate_problem_.has_value() ||
             cached_coordinate_adapter_ != adapter ||
             cached_orig_nlp_ != orig_nlp )
         {
            EvaluationValue<AnyNlpProblem> prepared =
               MakeLegacyIpoptCoordinateProblem(*adapter, *orig_nlp, 1);
            if( !prepared )
            {
               return prepared;
            }
            cached_coordinate_problem_.emplace(std::move(*prepared));
            cached_coordinate_adapter_ = adapter;
            cached_orig_nlp_ = orig_nlp;
         }
         return EvaluationValue<AnyNlpProblem>{
            BorrowNlpProblem(*cached_coordinate_problem_)};
      }();
      if( !problem )
      {
         return std::unexpected(problem.error());
      }
      const NlpStructure structure = problem->nlp_structure();
      if( x->size() != structure.variables ||
          y_c->size() + y_d->size() != structure.constraints )
      {
         return std::unexpected(EvaluationError{
            EvaluationErrorCode::dimension_mismatch,
            "C++23 canary model and current iterate dimensions differ"
         });
      }

      PrimalDualLayout layout;
      layout.equality_constraints.resize(y_c->size());
      std::iota(layout.equality_constraints.begin(), layout.equality_constraints.end(), 0);
      layout.inequality_constraints.resize(y_d->size());
      std::iota(
         layout.inequality_constraints.begin(),
         layout.inequality_constraints.end(), y_c->size());
      EvaluationValue<std::vector<Index>> primal_lower = BoundPositions(
         IpNLP().Px_L(), x->size(), z_lower->size(), "P_x_lower");
      EvaluationValue<std::vector<Index>> primal_upper = BoundPositions(
         IpNLP().Px_U(), x->size(), z_upper->size(), "P_x_upper");
      EvaluationValue<std::vector<Index>> slack_lower = BoundPositions(
         IpNLP().Pd_L(), y_d->size(), v_lower->size(), "P_s_lower");
      EvaluationValue<std::vector<Index>> slack_upper = BoundPositions(
         IpNLP().Pd_U(), y_d->size(), v_upper->size(), "P_s_upper");
      if( !primal_lower )
      {
         return std::unexpected(primal_lower.error());
      }
      if( !primal_upper )
      {
         return std::unexpected(primal_upper.error());
      }
      if( !slack_lower )
      {
         return std::unexpected(slack_lower.error());
      }
      if( !slack_upper )
      {
         return std::unexpected(slack_upper.error());
      }
      layout.primal_lower_bounds = std::move(*primal_lower);
      layout.primal_upper_bounds = std::move(*primal_upper);
      layout.slack_lower_bounds = std::move(*slack_lower);
      layout.slack_upper_bounds = std::move(*slack_upper);

      PrimalDualKktOperator kkt(std::move(*problem), std::move(layout));
      if( !kkt.valid() )
      {
         return std::unexpected(legacy_algorithm_canary_detail::CanaryError(
            "C++23 canary constructed an invalid KKT layout"));
      }

      std::vector<Number> multipliers;
      multipliers.reserve(y_c->size() + y_d->size());
      legacy_algorithm_canary_detail::Append(*y_c, multipliers);
      legacy_algorithm_canary_detail::Append(*y_d, multipliers);
      PrimalDualRegularization regularization{0., 0., 0., 0.};
      if( reference_result != nullptr )
      {
         ::Ipopt::Number stable_delta_x;
         ::Ipopt::Number stable_delta_s;
         ::Ipopt::Number stable_delta_c;
         ::Ipopt::Number stable_delta_d;
         IpData().getPDPert(
            stable_delta_x, stable_delta_s, stable_delta_c, stable_delta_d);
         regularization = {
            static_cast<Number>(stable_delta_x),
            static_cast<Number>(stable_delta_s),
            static_cast<Number>(stable_delta_c),
            static_cast<Number>(stable_delta_d)
         };
      }
      if( !std::isfinite(regularization.x) ||
          !std::isfinite(regularization.s) ||
          !std::isfinite(regularization.c) ||
          !std::isfinite(regularization.d) )
      {
         return std::unexpected(EvaluationError{
            EvaluationErrorCode::nonfinite_output,
            "C++23 canary perturbation is not finite"
         });
      }
      ++numeric_revision_;
      if( numeric_revision_ == 0 )
      {
         ++numeric_revision_;
      }
      PrimalDualState state{
         .nlp = {*x, 1., multipliers},
         .z_lower = *z_lower,
         .z_upper = *z_upper,
         .v_lower = *v_lower,
         .v_upper = *v_upper,
         .slack_x_lower = *slack_x_lower,
         .slack_x_upper = *slack_x_upper,
         .slack_s_lower = *slack_s_lower,
         .slack_s_upper = *slack_s_upper,
         .regularization = regularization,
         .numeric_revision = numeric_revision_
      };

      EvaluationValue<std::vector<Number>> flat_rhs = Flatten(rhs);
      if( !flat_rhs )
      {
         return std::unexpected(flat_rhs.error());
      }
      if( flat_rhs->size() != kkt.flat_dimension() )
      {
         return std::unexpected(EvaluationError{
            EvaluationErrorCode::dimension_mismatch,
            "C++23 canary KKT and IteratesVector dimensions differ"
         });
      }

      std::vector<Number> flat_reference;
      if( reference_result != nullptr )
      {
         EvaluationValue<std::vector<Number>> copied_reference =
            Flatten(*reference_result);
         if( !copied_reference )
         {
            return std::unexpected(copied_reference.error());
         }
         if( copied_reference->size() != kkt.flat_dimension() )
         {
            return std::unexpected(EvaluationError{
               EvaluationErrorCode::dimension_mismatch,
               "C++23 canary reference direction has the wrong dimension"
            });
         }
         flat_reference = std::move(*copied_reference);
      }

      std::vector<Number> unscaled_solution;
      CandidateFirstWorkStatistics work;
      bool converged = false;
      if( reference_result == nullptr )
      {
         if( !options_.candidate_first_backend )
         {
            return std::unexpected(legacy_algorithm_canary_detail::CanaryError(
               "candidate_first mode requires a C++23 backend"));
         }
         const Index required_negative_eigenvalues =
            kkt.dimensions().y_c + kkt.dimensions().y_d;
         EvaluationValue<CandidateFirstSolveResult> candidate =
            options_.candidate_first_backend->candidate_first_solve({
               .kkt = kkt,
               .state = state,
               .rhs = *flat_rhs,
               .required_negative_eigenvalues = required_negative_eigenvalues,
               .restoration_problem = restoration_problem_
            });
         if( !candidate )
         {
            return std::unexpected(candidate.error());
         }
         AddCandidateWork(candidate->work);
         if( candidate->inertia.certainty !=
                CandidateFirstInertiaCertainty::exact ||
             candidate->inertia.negative_eigenvalues !=
                required_negative_eigenvalues )
         {
            ++statistics_.candidate_first_inertia_rejections;
            return std::unexpected(legacy_algorithm_canary_detail::CanaryError(
               "candidate backend did not certify the required exact inertia"));
         }
         if( EvaluationResult valid =
                ValidateCandidateRegularization(
                   candidate->accepted_regularization);
             !valid )
         {
            return std::unexpected(valid.error());
         }
         if( candidate->direction.size() != kkt.flat_dimension() )
         {
            return std::unexpected(EvaluationError{
               EvaluationErrorCode::dimension_mismatch,
               "candidate backend returned the wrong direction dimension"
            });
         }
         regularization = candidate->accepted_regularization;
         state.regularization = regularization;
         unscaled_solution = std::move(candidate->direction);
         work = candidate->work;
         converged = candidate->converged;
      }
      else
      {
         const LegacyAugSystemViews views{
            .solver = ::Ipopt::GetRawPtr(augmented_solver_),
            .hessian = ::Ipopt::GetRawPtr(hessian),
            .jacobian_equalities = ::Ipopt::GetRawPtr(jacobian_equalities),
            .jacobian_inequalities = ::Ipopt::GetRawPtr(jacobian_inequalities),
            .x_prototype = ::Ipopt::GetRawPtr(rhs.x()),
            .s_prototype = ::Ipopt::GetRawPtr(rhs.s()),
            .equality_prototype = ::Ipopt::GetRawPtr(rhs.y_c()),
            .inequality_prototype = ::Ipopt::GetRawPtr(rhs.y_d()),
            .complementarity_x_diagonal =
               ::Ipopt::GetRawPtr(complementarity_x_diagonal),
            .complementarity_s_diagonal =
               ::Ipopt::GetRawPtr(complementarity_s_diagonal)
         };
         EvaluationValue<PreparedDirectPreconditioner> preconditioner_result =
            PrepareLegacyAugSystemPreconditioner(
               views, kkt, state, {.check_inertia = false});
         if( !preconditioner_result )
         {
            return std::unexpected(preconditioner_result.error());
         }
         PreparedDirectPreconditioner preconditioner =
            std::move(*preconditioner_result);
         EvaluationValue<PrimalDualSolveSession> session_result =
            MakePrimalDualSolveSession(kkt, options_.fgmres);
         if( !session_result )
         {
            return std::unexpected(session_result.error());
         }
         PrimalDualSolveSession session = std::move(*session_result);
         unscaled_solution.assign(kkt.flat_dimension(), 0.);
         EvaluationValue<FgmresResult> solved = session.solve(
            kkt, state, preconditioner, *flat_rhs, unscaled_solution);
         if( !solved )
         {
            return std::unexpected(solved.error());
         }
         converged = solved->converged();
      }
      if( !std::ranges::all_of(
             unscaled_solution,
             [](Number value) { return std::isfinite(value); }) )
      {
         return std::unexpected(EvaluationError{
            EvaluationErrorCode::nonfinite_output,
            "C++23 solver returned a nonfinite direction"
         });
      }

      std::vector<Number> reconstructed_rhs(kkt.flat_dimension());
      if( reference_result == nullptr )
      {
         ++statistics_.candidate_first_validation_kkt_applications;
         statistics_.candidate_first_validation_derivative_product_requests += 3;
      }
      EvaluationResult applied = kkt.apply_flat(
         state, unscaled_solution, reconstructed_rhs);
      if( !applied )
      {
         return std::unexpected(applied.error());
      }

      std::vector<Number> solution = unscaled_solution;
      for( Number& value : solution )
      {
         value *= static_cast<Number>(alpha);
      }
      if( !std::ranges::all_of(
             solution, [](Number value) { return std::isfinite(value); }) )
      {
         return std::unexpected(EvaluationError{
            EvaluationErrorCode::nonfinite_output,
            "C++23 scaled direction is nonfinite"
         });
      }

      return legacy_algorithm_canary_detail::Comparison{
         .direction_relative_error = reference_result == nullptr
            ? 0.
            : legacy_algorithm_canary_detail::RelativeInfinityError(
                 solution, flat_reference),
         .residual_relative_error =
            legacy_algorithm_canary_detail::RelativeInfinityError(
               reconstructed_rhs, *flat_rhs),
         .converged = converged,
         .direction = std::move(solution),
         .regularization = regularization,
         .work = work
      };
   }

   static EvaluationResult ValidateCandidateRegularization(
      PrimalDualRegularization regularization
   )
   {
      const std::array<Number, 4> values{{
         regularization.x,
         regularization.s,
         regularization.c,
         regularization.d
      }};
      for( Number value : values )
      {
         if( value < 0. || !std::isfinite(value) ||
             !std::isfinite(
                static_cast<Number>(static_cast<::Ipopt::Number>(value))) )
         {
            return std::unexpected(EvaluationError{
               EvaluationErrorCode::nonfinite_output,
               "candidate regularization must be finite, nonnegative, and "
               "representable by Ipopt::Number"
            });
         }
      }
      return {};
   }

   static EvaluationResult CommitDirection(
      std::span<const Number>       direction,
      ::Ipopt::IteratesVector& result
   )
   {
      ::Ipopt::SmartPtr<::Ipopt::IteratesVector> replacement =
         result.MakeNewIteratesVector(true);
      std::array<::Ipopt::SmartPtr<::Ipopt::Vector>, 8> blocks{{
         replacement->x_NonConst(),
         replacement->s_NonConst(),
         replacement->y_c_NonConst(),
         replacement->y_d_NonConst(),
         replacement->z_L_NonConst(),
         replacement->z_U_NonConst(),
         replacement->v_L_NonConst(),
         replacement->v_U_NonConst()
      }};

      Index dimension = 0;
      for( const ::Ipopt::SmartPtr<::Ipopt::Vector>& block : blocks )
      {
         EvaluationValue<Index> block_dimension =
            legacy_ipopt_detail::CheckedIndex(
               block->Dim(), "replacement direction block");
         if( !block_dimension )
         {
            return std::unexpected(block_dimension.error());
         }
         if( *block_dimension > std::numeric_limits<Index>::max() - dimension )
         {
            return std::unexpected(EvaluationError{
               EvaluationErrorCode::dimension_mismatch,
               "replacement direction dimension overflows C++23 Index"
            });
         }
         dimension += *block_dimension;
      }
      if( direction.size() != dimension )
      {
         return std::unexpected(EvaluationError{
            EvaluationErrorCode::dimension_mismatch,
            "replacement direction has the wrong flat dimension"
         });
      }

      std::vector<::Ipopt::Number> stable(direction.size());
      for( Index i = 0; i < direction.size(); ++i )
      {
         stable[i] = static_cast<::Ipopt::Number>(direction[i]);
         if( !std::isfinite(static_cast<Number>(stable[i])) )
         {
            return std::unexpected(EvaluationError{
               EvaluationErrorCode::nonfinite_output,
               "replacement direction is not representable by Ipopt::Number"
            });
         }
      }

      Index offset = 0;
      ::Ipopt::Number empty = 0.;
      for( const ::Ipopt::SmartPtr<::Ipopt::Vector>& block : blocks )
      {
         const Index block_dimension = static_cast<Index>(block->Dim());
         const ::Ipopt::Number* values = block_dimension == 0
            ? &empty
            : stable.data() + offset;
         ::Ipopt::TripletHelper::PutValuesInVector(
            block->Dim(), values, *block);
         offset += block_dimension;
      }

      // All allocation, conversion, and block writes happened in a detached
      // vector.  The accepted legacy direction remains the fallback until
      // this final commit.
      result.Copy(*replacement);
      return {};
   }

   static EvaluationValue<std::vector<Index>> BoundPositions(
      const ::Ipopt::SmartPtr<const ::Ipopt::Matrix>& matrix,
      Index                                            rows,
      Index                                            columns,
      std::string_view                                 name
   )
   {
      if( !::Ipopt::IsValid(matrix) )
      {
         return std::unexpected(EvaluationError{
            EvaluationErrorCode::invalid_layout,
            std::string(name) + " is unavailable"
         });
      }
      EvaluationValue<Index> matrix_rows =
         legacy_ipopt_detail::CheckedIndex(matrix->NRows(), name);
      EvaluationValue<Index> matrix_columns =
         legacy_ipopt_detail::CheckedIndex(matrix->NCols(), name);
      if( !matrix_rows )
      {
         return std::unexpected(matrix_rows.error());
      }
      if( !matrix_columns )
      {
         return std::unexpected(matrix_columns.error());
      }
      if( *matrix_rows != rows || *matrix_columns != columns )
      {
         return std::unexpected(EvaluationError{
            EvaluationErrorCode::dimension_mismatch,
            std::string(name) + " dimensions do not match the KKT layout"
         });
      }

      const ::Ipopt::Index stable_nonzeros =
         ::Ipopt::TripletHelper::GetNumberEntries(*matrix);
      EvaluationValue<Index> nonzeros =
         legacy_ipopt_detail::CheckedIndex(stable_nonzeros, name);
      if( !nonzeros )
      {
         return std::unexpected(nonzeros.error());
      }
      if( *nonzeros != columns )
      {
         return std::unexpected(EvaluationError{
            EvaluationErrorCode::invalid_layout,
            std::string(name) + " is not a one-entry-per-column expansion"
         });
      }

      std::vector<::Ipopt::Index> stable_rows(*nonzeros);
      std::vector<::Ipopt::Index> stable_columns(*nonzeros);
      std::vector<::Ipopt::Number> stable_values(*nonzeros);
      ::Ipopt::Index empty_row = 0;
      ::Ipopt::Index empty_column = 0;
      ::Ipopt::Number empty_value = 0.;
      ::Ipopt::TripletHelper::FillRowCol(
         stable_nonzeros, *matrix,
         stable_rows.empty() ? &empty_row : stable_rows.data(),
         stable_columns.empty() ? &empty_column : stable_columns.data());
      ::Ipopt::TripletHelper::FillValues(
         stable_nonzeros, *matrix,
         stable_values.empty() ? &empty_value : stable_values.data());

      std::vector<Index> positions(columns, 0);
      std::vector<bool> seen(columns, false);
      const Number arithmetic_epsilon = std::max(
         std::numeric_limits<Number>::epsilon(),
         static_cast<Number>(
            std::numeric_limits<::Ipopt::Number>::epsilon()));
      for( Index i = 0; i < *nonzeros; ++i )
      {
         if( stable_rows[i] <= 0 || stable_columns[i] <= 0 )
         {
            return std::unexpected(EvaluationError{
               EvaluationErrorCode::invalid_layout,
               std::string(name) + " contains a nonpositive triplet index"
            });
         }
         EvaluationValue<Index> row = legacy_ipopt_detail::CheckedIndex(
            stable_rows[i] - 1, name);
         EvaluationValue<Index> column = legacy_ipopt_detail::CheckedIndex(
            stable_columns[i] - 1, name);
         if( !row )
         {
            return std::unexpected(row.error());
         }
         if( !column )
         {
            return std::unexpected(column.error());
         }
         const Number value = static_cast<Number>(stable_values[i]);
         if( *row >= rows || *column >= columns || seen[*column] ||
             !std::isfinite(value) ||
             std::abs(value - 1.) > 64. * arithmetic_epsilon )
         {
            return std::unexpected(EvaluationError{
               EvaluationErrorCode::invalid_layout,
               std::string(name) + " is not a unit expansion matrix"
            });
         }
         positions[*column] = *row;
         seen[*column] = true;
      }
      return positions;
   }

   static EvaluationValue<std::vector<Number>> Flatten(
      const ::Ipopt::IteratesVector& iterates
   )
   {
      EvaluationValue<std::vector<Number>> x =
         legacy_algorithm_canary_detail::CopyVector(*iterates.x(), "flat x");
      EvaluationValue<std::vector<Number>> s =
         legacy_algorithm_canary_detail::CopyVector(*iterates.s(), "flat s");
      EvaluationValue<std::vector<Number>> y_c =
         legacy_algorithm_canary_detail::CopyVector(*iterates.y_c(), "flat y_c");
      EvaluationValue<std::vector<Number>> y_d =
         legacy_algorithm_canary_detail::CopyVector(*iterates.y_d(), "flat y_d");
      EvaluationValue<std::vector<Number>> z_lower =
         legacy_algorithm_canary_detail::CopyVector(*iterates.z_L(), "flat z_L");
      EvaluationValue<std::vector<Number>> z_upper =
         legacy_algorithm_canary_detail::CopyVector(*iterates.z_U(), "flat z_U");
      EvaluationValue<std::vector<Number>> v_lower =
         legacy_algorithm_canary_detail::CopyVector(*iterates.v_L(), "flat v_L");
      EvaluationValue<std::vector<Number>> v_upper =
         legacy_algorithm_canary_detail::CopyVector(*iterates.v_U(), "flat v_U");
      if( !x || !s || !y_c || !y_d || !z_lower || !z_upper || !v_lower || !v_upper )
      {
         return std::unexpected(EvaluationError{
            EvaluationErrorCode::model_failure,
            "C++23 canary could not flatten an IteratesVector"
         });
      }
      std::vector<Number> result;
      result.reserve(
         x->size() + s->size() + y_c->size() + y_d->size() +
         z_lower->size() + z_upper->size() + v_lower->size() + v_upper->size());
      legacy_algorithm_canary_detail::Append(*x, result);
      legacy_algorithm_canary_detail::Append(*s, result);
      legacy_algorithm_canary_detail::Append(*y_c, result);
      legacy_algorithm_canary_detail::Append(*y_d, result);
      legacy_algorithm_canary_detail::Append(*z_lower, result);
      legacy_algorithm_canary_detail::Append(*z_upper, result);
      legacy_algorithm_canary_detail::Append(*v_lower, result);
      legacy_algorithm_canary_detail::Append(*v_upper, result);
      return result;
   }

   ::Ipopt::SmartPtr<::Ipopt::PDSystemSolver> reference_solver_;
   ::Ipopt::SmartPtr<::Ipopt::AugSystemSolver> augmented_solver_;
   const LegacyAlgorithmCanaryOptions options_;
   LegacyAlgorithmCanaryStatistics& statistics_;
   std::optional<AnyNlpProblem> cached_coordinate_problem_;
   const ::Ipopt::TNLPAdapter* cached_coordinate_adapter_ = nullptr;
   const ::Ipopt::OrigIpoptNLP* cached_orig_nlp_ = nullptr;
   std::uint64_t numeric_revision_ = 0;
   bool exact_hessian_ = false;
   bool restoration_problem_ = false;
};

/** Explicit runtime opt-in that reuses the stable AlgorithmBuilder graph. */
class LegacyAlgorithmCanaryBuilder final : public ::Ipopt::AlgorithmBuilder
{
public:
   explicit LegacyAlgorithmCanaryBuilder(
      LegacyAlgorithmCanaryOptions options = {}
   )
      : options_(std::move(options))
   {
   }

   ::Ipopt::SmartPtr<::Ipopt::PDSystemSolver> PDSystemSolverFactory(
      const ::Ipopt::Journalist&  journalist,
      const ::Ipopt::OptionsList& options,
      const std::string&           prefix
   ) override
   {
      ::Ipopt::SmartPtr<::Ipopt::PDSystemSolver> reference =
         ::Ipopt::AlgorithmBuilder::PDSystemSolverFactory(
            journalist, options, prefix);
      ::Ipopt::AugSystemSolver* augmented = nullptr;
#if IPOPT_CXX23_HAS_PDFULLSPACE_HEADER
      augmented = legacy_algorithm_canary_detail::
         AugmentedSolverIfAvailable<::Ipopt::PDFullSpaceSolver>(*reference);
#endif
      ::Ipopt::SmartPtr<::Ipopt::AugSystemSolver> fallback_augmented;
      if( augmented == nullptr )
      {
         fallback_augmented = GetAugSystemSolver(journalist, options, prefix);
         augmented = ::Ipopt::GetRawPtr(fallback_augmented);
      }
      return new LegacyAlgorithmCanaryPDSystemSolver(
         *reference, *augmented, options_, statistics_);
   }

   const LegacyAlgorithmCanaryStatistics& statistics() const noexcept
   {
      return statistics_;
   }

   static constexpr bool restoration_factory_supported() noexcept
   {
#if IPOPT_CXX23_HAS_PDFULLSPACE_HEADER
      return requires(::Ipopt::PDFullSpaceSolver& solver) {
         { solver.AugmentedSystemSolver() }
            -> std::same_as<::Ipopt::AugSystemSolver&>;
      };
#else
      return false;
#endif
   }

private:
   const LegacyAlgorithmCanaryOptions options_;
   LegacyAlgorithmCanaryStatistics statistics_;
};
} // namespace Ipopt::Cxx23

#undef IPOPT_CXX23_HAS_PDFULLSPACE_HEADER

#endif
