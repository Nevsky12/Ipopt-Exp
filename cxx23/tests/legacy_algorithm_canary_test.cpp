// Copyright (C) 2026 Ipopt contributors.
// All Rights Reserved.
// This file is distributed under the Eclipse Public License.

#include <ipopt/cxx23/legacy_algorithm_canary.hpp>

#include <IpIpoptApplication.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace
{
namespace Stable = ::Ipopt;
namespace Cxx = ::Ipopt::Cxx23;

constexpr Cxx::Number StableTolerance(
   Cxx::Number double_precision_tolerance,
   Cxx::Number stable_epsilon_multiplier
) noexcept
{
   return std::max(
      double_precision_tolerance,
      stable_epsilon_multiplier *
         static_cast<Cxx::Number>(
            std::numeric_limits<Stable::Number>::epsilon()));
}

void Check(bool condition, std::string_view message)
{
   if( !condition )
   {
      throw std::runtime_error(std::string(message));
   }
}

enum class CandidateScenario
{
   disabled,
   accepted,
   backend_failure_once,
   inertia_rejection_once,
   residual_rejection_once
};

constexpr std::string_view CandidateScenarioName(
   CandidateScenario scenario
) noexcept
{
   switch( scenario )
   {
      case CandidateScenario::disabled:
         return "disabled";
      case CandidateScenario::accepted:
         return "accepted";
      case CandidateScenario::backend_failure_once:
         return "backend_failure_once";
      case CandidateScenario::inertia_rejection_once:
         return "inertia_rejection_once";
      case CandidateScenario::residual_rejection_once:
         return "residual_rejection_once";
   }
   return "unknown";
}

struct CandidateBackendCounters
{
   Cxx::Index requests = 0;
   Cxx::Index restoration_requests = 0;
   Cxx::Index completed_solves = 0;
   Cxx::Index kkt_applications = 0;
   Cxx::Index derivative_product_requests = 0;
   Cxx::Index failures_remaining = 0;
   Cxx::Index inertia_rejections_remaining = 0;
   Cxx::Index residual_rejections_remaining = 0;
};

class DenseCandidateBackend
{
public:
   explicit DenseCandidateBackend(
      std::shared_ptr<CandidateBackendCounters> counters
   )
      : counters_(std::move(counters))
   {
   }

   Cxx::EvaluationValue<Cxx::CandidateFirstSolveResult> solve(
      Cxx::CandidateFirstSolveRequest request
   )
   {
      ++counters_->requests;
      if( request.restoration_problem )
      {
         ++counters_->restoration_requests;
      }
      if( counters_->failures_remaining > 0 )
      {
         --counters_->failures_remaining;
         return std::unexpected(Cxx::EvaluationError{
            Cxx::EvaluationErrorCode::model_failure,
            "injected candidate backend failure"
         });
      }

      const Cxx::Index dimension = request.kkt.flat_dimension();
      if( request.rhs.size() != dimension )
      {
         return std::unexpected(Cxx::EvaluationError{
            Cxx::EvaluationErrorCode::dimension_mismatch,
            "dense candidate received the wrong RHS dimension"
         });
      }
      ++counters_->completed_solves;
      if( counters_->inertia_rejections_remaining > 0 )
      {
         --counters_->inertia_rejections_remaining;
         Cxx::CandidateFirstSolveResult result = Result(
            std::vector<Cxx::Number>(dimension, 0.), request, true);
         ++result.inertia.negative_eigenvalues;
         return result;
      }
      if( counters_->residual_rejections_remaining > 0 )
      {
         --counters_->residual_rejections_remaining;
         return Result(
            std::vector<Cxx::Number>(dimension, 0.), request, true);
      }

      std::vector<Cxx::Number> matrix(dimension * dimension, 0.);
      std::vector<Cxx::Number> basis(dimension, 0.);
      std::vector<Cxx::Number> column(dimension, 0.);
      for( Cxx::Index j = 0; j < dimension; ++j )
      {
         ++counters_->kkt_applications;
         counters_->derivative_product_requests += 3;
         basis[j] = 1.;
         Cxx::EvaluationResult applied = request.kkt.apply_flat(
            request.state, basis, column);
         if( !applied )
         {
            return std::unexpected(applied.error());
         }
         for( Cxx::Index i = 0; i < dimension; ++i )
         {
            matrix[i * dimension + j] = column[i];
         }
         basis[j] = 0.;
      }

      std::vector<Cxx::Number> direction(
         request.rhs.begin(), request.rhs.end());
      const Cxx::Number pivot_floor =
         1024. * std::numeric_limits<Cxx::Number>::epsilon();
      for( Cxx::Index k = 0; k < dimension; ++k )
      {
         Cxx::Index pivot = k;
         Cxx::Number pivot_magnitude =
            std::abs(matrix[k * dimension + k]);
         for( Cxx::Index i = k + 1; i < dimension; ++i )
         {
            const Cxx::Number magnitude =
               std::abs(matrix[i * dimension + k]);
            if( magnitude > pivot_magnitude )
            {
               pivot = i;
               pivot_magnitude = magnitude;
            }
         }
         if( !std::isfinite(pivot_magnitude) ||
             pivot_magnitude <= pivot_floor )
         {
            return std::unexpected(Cxx::EvaluationError{
               Cxx::EvaluationErrorCode::model_failure,
               "dense candidate matrix is numerically singular"
            });
         }
         if( pivot != k )
         {
            for( Cxx::Index j = k; j < dimension; ++j )
            {
               std::swap(
                  matrix[k * dimension + j],
                  matrix[pivot * dimension + j]);
            }
            std::swap(direction[k], direction[pivot]);
         }
         for( Cxx::Index i = k + 1; i < dimension; ++i )
         {
            const Cxx::Number factor =
               matrix[i * dimension + k] / matrix[k * dimension + k];
            matrix[i * dimension + k] = 0.;
            for( Cxx::Index j = k + 1; j < dimension; ++j )
            {
               matrix[i * dimension + j] -=
                  factor * matrix[k * dimension + j];
            }
            direction[i] -= factor * direction[k];
         }
      }
      for( Cxx::Index i = dimension; i-- > 0; )
      {
         for( Cxx::Index j = i + 1; j < dimension; ++j )
         {
            direction[i] -= matrix[i * dimension + j] * direction[j];
         }
         direction[i] /= matrix[i * dimension + i];
      }
      return Result(std::move(direction), request, true, dimension);
   }

private:
   static Cxx::CandidateFirstSolveResult Result(
      std::vector<Cxx::Number>              direction,
      const Cxx::CandidateFirstSolveRequest& request,
      bool                                  converged,
      Cxx::Index                            kkt_applications = 0
   )
   {
      return {
         .direction = std::move(direction),
         .accepted_regularization = request.state.regularization,
         .inertia = {
            .certainty = Cxx::CandidateFirstInertiaCertainty::exact,
            .negative_eigenvalues = request.required_negative_eigenvalues
         },
         .work = {
            .factorizations = 1,
            .backsolves = 1,
            .kkt_applications = kkt_applications,
            .derivative_product_requests = 3 * kkt_applications,
            .refinement_steps = 0,
            .quality_improvements = 0
         },
         .converged = converged
      };
   }

   std::shared_ptr<CandidateBackendCounters> counters_;
};

class CanaryTnlp final : public Stable::TNLP
{
public:
   explicit CanaryTnlp(bool start_infeasible = false)
      : start_infeasible_(start_infeasible)
   {
   }

   bool get_nlp_info(
      Stable::Index&  variables,
      Stable::Index&  constraints,
      Stable::Index&  jacobian_nonzeros,
      Stable::Index&  hessian_nonzeros,
      IndexStyleEnum& index_style
   ) override
   {
      variables = 3;
      constraints = 2;
      jacobian_nonzeros = 4;
      hessian_nonzeros = 3;
      index_style = C_STYLE;
      return true;
   }

   bool get_bounds_info(
      Stable::Index,
      Stable::Number* x_lower,
      Stable::Number* x_upper,
      Stable::Index,
      Stable::Number* g_lower,
      Stable::Number* g_upper
   ) override
   {
      constexpr std::array<Stable::Number, 3> expected_x_lower{{0., 0., 0.}};
      constexpr std::array<Stable::Number, 3> expected_x_upper{{5., 5., 0.}};
      constexpr std::array<Stable::Number, 2> expected_g_lower{{3., -1.5}};
      constexpr std::array<Stable::Number, 2> expected_g_upper{{3., -.5}};
      std::ranges::copy(expected_x_lower, x_lower);
      std::ranges::copy(expected_x_upper, x_upper);
      std::ranges::copy(expected_g_lower, g_lower);
      std::ranges::copy(expected_g_upper, g_upper);
      return true;
   }

   bool get_starting_point(
      Stable::Index,
      bool            init_x,
      Stable::Number* x,
      bool,
      Stable::Number*,
      Stable::Number*,
      Stable::Index,
      bool,
      Stable::Number*
   ) override
   {
      if( init_x )
      {
         const std::array<Stable::Number, 3> initial = start_infeasible_
            ? std::array<Stable::Number, 3>{{4., .25, 0.}}
            : std::array<Stable::Number, 3>{{.9, 2.1, 0.}};
         std::ranges::copy(initial, x);
      }
      return true;
   }

   bool eval_f(
      Stable::Index,
      const Stable::Number* x,
      bool,
      Stable::Number& objective
   ) override
   {
      objective = (x[0] - 1.) * (x[0] - 1.) +
         (x[1] - 2.) * (x[1] - 2.) + x[2] * x[2];
      return true;
   }

   bool eval_grad_f(
      Stable::Index,
      const Stable::Number* x,
      bool,
      Stable::Number* gradient
   ) override
   {
      gradient[0] = 2. * (x[0] - 1.);
      gradient[1] = 2. * (x[1] - 2.);
      gradient[2] = 2. * x[2];
      return true;
   }

   bool eval_g(
      Stable::Index,
      const Stable::Number* x,
      bool,
      Stable::Index,
      Stable::Number* constraints
   ) override
   {
      constraints[0] = x[0] + x[1];
      constraints[1] = x[0] - x[1];
      return true;
   }

   bool eval_jac_g(
      Stable::Index,
      const Stable::Number*,
      bool,
      Stable::Index,
      Stable::Index,
      Stable::Index* rows,
      Stable::Index* columns,
      Stable::Number* values
   ) override
   {
      if( values == nullptr )
      {
         constexpr std::array<Stable::Index, 4> expected_rows{{0, 0, 1, 1}};
         constexpr std::array<Stable::Index, 4> expected_columns{{0, 1, 0, 1}};
         std::ranges::copy(expected_rows, rows);
         std::ranges::copy(expected_columns, columns);
      }
      else
      {
         constexpr std::array<Stable::Number, 4> expected{{1., 1., 1., -1.}};
         std::ranges::copy(expected, values);
      }
      return true;
   }

   bool eval_h(
      Stable::Index,
      const Stable::Number*,
      bool,
      Stable::Number objective_factor,
      Stable::Index,
      const Stable::Number*,
      bool,
      Stable::Index,
      Stable::Index* rows,
      Stable::Index* columns,
      Stable::Number* values
   ) override
   {
      if( values == nullptr )
      {
         constexpr std::array<Stable::Index, 3> diagonal{{0, 1, 2}};
         std::ranges::copy(diagonal, rows);
         std::ranges::copy(diagonal, columns);
      }
      else
      {
         std::fill_n(values, 3, 2. * objective_factor);
      }
      return true;
   }

   bool get_scaling_parameters(
      Stable::Number& objective_scaling,
      bool&           use_x_scaling,
      Stable::Index,
      Stable::Number* x_scaling,
      bool&           use_g_scaling,
      Stable::Index,
      Stable::Number* g_scaling
   ) override
   {
      objective_scaling = 2.;
      use_x_scaling = true;
      use_g_scaling = true;
      constexpr std::array<Stable::Number, 3> expected_x{{2., 3., 4.}};
      constexpr std::array<Stable::Number, 2> expected_g{{5., 7.}};
      std::ranges::copy(expected_x, x_scaling);
      std::ranges::copy(expected_g, g_scaling);
      return true;
   }

   bool intermediate_callback(
      Stable::AlgorithmMode mode,
      Stable::Index,
      Stable::Number,
      Stable::Number,
      Stable::Number,
      Stable::Number,
      Stable::Number,
      Stable::Number,
      Stable::Number,
      Stable::Number,
      Stable::Index,
      const Stable::IpoptData*,
      Stable::IpoptCalculatedQuantities*
   ) override
   {
      if( mode == Stable::RestorationPhaseMode )
      {
         ++restoration_callbacks_;
      }
      return true;
   }

   Stable::Index restoration_callbacks() const noexcept
   {
      return restoration_callbacks_;
   }

   void finalize_solution(
      Stable::SolverReturn,
      Stable::Index,
      const Stable::Number*,
      const Stable::Number*,
      const Stable::Number*,
      Stable::Index,
      const Stable::Number*,
      const Stable::Number*,
      Stable::Number,
      const Stable::IpoptData*,
      Stable::IpoptCalculatedQuantities*
   ) override
   {
   }

private:
   const bool start_infeasible_;
   Stable::Index restoration_callbacks_ = 0;
};

void TestTreatment(
   const std::string& fixed_treatment,
   bool               start_with_restoration = false,
   bool               reoptimize = false,
   bool               force_nonfatal_mismatch = false,
   bool               validated_replacement = false,
   CandidateScenario  candidate_scenario = CandidateScenario::disabled
)
{
   Check(
      !validated_replacement || candidate_scenario == CandidateScenario::disabled,
      "replacement and candidate-first scenarios are mutually exclusive");
   Stable::SmartPtr<Stable::IpoptApplication> application =
      new Stable::IpoptApplication(false);
   Check(
      application->Options()->SetIntegerValue("print_level", 0),
      "failed to set print_level");
   Check(
      application->Options()->SetIntegerValue("max_iter", 40),
      "failed to set max_iter");
   Check(
      application->Options()->SetNumericValue(
         "tol", StableTolerance(1e-9, 128.)),
      "failed to set tolerance");
   Check(
      application->Options()->SetStringValue(
         "fixed_variable_treatment", fixed_treatment),
      "failed to set fixed-variable treatment");
   Check(
      application->Options()->SetStringValue(
         "nlp_scaling_method", "user-scaling"),
      "failed to enable user scaling");
   if( start_with_restoration )
   {
      Check(
         application->Options()->SetStringValue("start_with_resto", "yes"),
         "failed to force restoration");
   }
   Check(
      application->Initialize("") == Stable::Solve_Succeeded,
      "Ipopt application initialization failed");

   Stable::SmartPtr<CanaryTnlp> tnlp =
      new CanaryTnlp(start_with_restoration);
   Stable::SmartPtr<Stable::NLP> nlp =
      new Stable::TNLPAdapter(tnlp, application->Jnlst());
   Cxx::LegacyAlgorithmCanaryOptions canary_options;
   canary_options.direction_relative_tolerance =
      StableTolerance(2e-8, 512.);
   canary_options.residual_relative_tolerance =
      StableTolerance(2e-10, 512.);
   canary_options.fail_on_error = true;
   canary_options.fail_on_mismatch = !force_nonfatal_mismatch;
   std::shared_ptr<CandidateBackendCounters> candidate_counters =
      std::make_shared<CandidateBackendCounters>();
   if( candidate_scenario != CandidateScenario::disabled )
   {
      canary_options.mode = Cxx::LegacyAlgorithmCanaryMode::candidate_first;
      if( candidate_scenario == CandidateScenario::backend_failure_once )
      {
         candidate_counters->failures_remaining = 1;
      }
      if( candidate_scenario == CandidateScenario::inertia_rejection_once )
      {
         candidate_counters->inertia_rejections_remaining = 1;
      }
      if( candidate_scenario == CandidateScenario::residual_rejection_once )
      {
         candidate_counters->residual_rejections_remaining = 1;
      }
      canary_options.candidate_first_backend = Cxx::MakeCandidateFirstBackend(
         DenseCandidateBackend(candidate_counters));
   }
   else
   {
      canary_options.mode = validated_replacement
         ? Cxx::LegacyAlgorithmCanaryMode::validated_replacement
         : Cxx::LegacyAlgorithmCanaryMode::shadow;
   }
   if( force_nonfatal_mismatch )
   {
      canary_options.fgmres.maximum_iterations = 0;
   }
   Stable::SmartPtr<Cxx::LegacyAlgorithmCanaryBuilder> canary_builder =
      new Cxx::LegacyAlgorithmCanaryBuilder(canary_options);
   Stable::SmartPtr<Stable::AlgorithmBuilder> builder =
      Stable::GetRawPtr(canary_builder);

   const Stable::ApplicationReturnStatus status =
      application->OptimizeNLP(nlp, builder);
   Check(status == Stable::Solve_Succeeded, "canary optimization did not converge");
   const Cxx::LegacyAlgorithmCanaryStatistics& statistics =
      canary_builder->statistics();
   const Cxx::Index progress_after_first_solve =
      candidate_scenario == CandidateScenario::disabled
      ? statistics.checked_solves
      : statistics.candidate_first_accepted;
   if( reoptimize )
   {
      const Stable::ApplicationReturnStatus reoptimized =
         application->ReOptimizeNLP(nlp);
      Check(
         reoptimized == Stable::Solve_Succeeded,
         "canary reoptimization did not converge");
      Check(
         (candidate_scenario == CandidateScenario::disabled
             ? statistics.checked_solves
             : statistics.candidate_first_accepted) > progress_after_first_solve,
         "C++23 canary did not check a reoptimized solve");
   }
   if( candidate_scenario != CandidateScenario::disabled )
   {
      Check(
         statistics.candidate_first_requests > 0,
         "candidate-first backend was not requested");
      Check(
         statistics.candidate_first_accepted > 0,
         "candidate-first backend never committed a direction");
      Check(
         statistics.candidate_first_requests ==
            statistics.candidate_first_accepted +
               statistics.candidate_first_fallbacks,
         "candidate-first request accounting is inconsistent");
      Check(
         statistics.candidate_first_factorizations ==
            candidate_counters->completed_solves,
         "candidate-first factorization accounting is inconsistent");
      Check(
         statistics.candidate_first_backsolves ==
            candidate_counters->completed_solves,
         "candidate-first backsolve accounting is inconsistent");
      Check(
         statistics.candidate_first_kkt_applications ==
            candidate_counters->kkt_applications,
         "candidate-first KKT application accounting is inconsistent");
      Check(
         statistics.candidate_first_derivative_product_requests ==
            candidate_counters->derivative_product_requests,
         "candidate-first derivative request accounting is inconsistent");
      Check(
         statistics.candidate_first_validation_kkt_applications > 0 &&
            statistics.candidate_first_validation_derivative_product_requests ==
               3 * statistics.candidate_first_validation_kkt_applications,
         "candidate-first residual validation accounting is inconsistent");
      Check(
         candidate_counters->requests == statistics.candidate_first_requests,
         "AnyAny backend request accounting is inconsistent");
      if( candidate_scenario == CandidateScenario::accepted )
      {
         Check(
            statistics.candidate_first_fallbacks == 0,
            "accepted candidate unexpectedly used the stable fallback");
         Check(
            statistics.reference_solve_requests == statistics.skipped_solves,
            "accepted candidate performed an avoidable reference solve");
      }
      else
      {
         Check(
            statistics.candidate_first_fallbacks == 1,
            "injected candidate rejection did not fall back exactly once");
         Check(
            statistics.reference_solve_requests ==
               statistics.skipped_solves + statistics.candidate_first_fallbacks,
            "candidate rejection did not call the reference solver exactly once");
      }
      if( candidate_scenario == CandidateScenario::backend_failure_once )
      {
         Check(
            statistics.candidate_first_failures == 1,
            "injected backend failure was not counted");
         Check(
            statistics.candidate_first_residual_rejections == 0,
            "backend failure was misclassified as a residual rejection");
      }
      if( candidate_scenario == CandidateScenario::inertia_rejection_once )
      {
         Check(
            statistics.candidate_first_inertia_rejections == 1,
            "wrong candidate inertia was not rejected");
         Check(
            statistics.candidate_first_failures == 1,
            "inertia rejection was not counted as a candidate failure");
         Check(
            statistics.candidate_first_residual_rejections == 0,
            "inertia rejection was misclassified as a residual rejection");
      }
      if( candidate_scenario == CandidateScenario::residual_rejection_once )
      {
         Check(
            statistics.candidate_first_residual_rejections == 1,
            "bad candidate residual was not rejected");
         Check(
            statistics.candidate_first_failures == 0,
            "residual rejection was misclassified as a backend failure");
      }
   }
   else
   {
      Check(statistics.reference_solve_requests > 0, "reference solver was not called");
      Check(statistics.checked_solves > 0, "C++23 canary did not check a solve");
      if( force_nonfatal_mismatch )
      {
         Check(
            statistics.mismatching_solves > 0,
            "C++23 canary did not report the injected mismatch");
      }
      else
      {
         Check(
            statistics.matching_solves == statistics.checked_solves,
            "C++23 canary reported a direction mismatch");
         Check(
            statistics.mismatching_solves == 0,
            "C++23 canary mismatch count is nonzero");
      }
      Check(
         statistics.evaluation_failures == 0,
         "C++23 canary evaluation failed");
      if( validated_replacement )
      {
         Check(
            statistics.replacement_commit_requests == statistics.matching_solves,
            "C++23 replacement selected a nonmatching direction");
         Check(
            statistics.committed_replacements ==
               statistics.replacement_commit_requests,
            "C++23 replacement did not commit every selected direction");
         Check(
            statistics.replacement_commit_failures == 0,
            "C++23 replacement commit failed");
         if( force_nonfatal_mismatch )
         {
            Check(
               statistics.committed_replacements < statistics.checked_solves,
               "C++23 replacement committed an injected mismatch");
         }
      }
   }
   if( start_with_restoration )
   {
      Check(tnlp->restoration_callbacks() > 0, "forced restoration did not run");
      if( Cxx::LegacyAlgorithmCanaryBuilder::restoration_factory_supported() )
      {
         if( candidate_scenario != CandidateScenario::disabled )
         {
            Check(
               statistics.restoration_candidate_first_requests > 0,
               "restoration candidate-first backend was not requested");
            Check(
               statistics.restoration_candidate_first_accepted > 0,
               "restoration candidate-first direction was not committed");
            Check(
               candidate_counters->restoration_requests ==
                  statistics.restoration_candidate_first_requests,
               "restoration AnyAny request accounting is inconsistent");
         }
         else
         {
            Check(
               statistics.restoration_reference_solve_requests > 0,
               "restoration reference solver was not wrapped");
            Check(
               statistics.restoration_checked_solves > 0,
               "C++23 canary did not check a restoration solve");
            Check(
               statistics.restoration_matching_solves ==
                  statistics.restoration_checked_solves,
               "C++23 canary reported a restoration mismatch");
            Check(
               statistics.restoration_mismatching_solves == 0,
               "C++23 canary restoration mismatch count is nonzero");
            Check(
               statistics.restoration_evaluation_failures == 0,
               "C++23 canary restoration evaluation failed");
            if( validated_replacement )
            {
               Check(
                  statistics.restoration_committed_replacements ==
                     statistics.restoration_matching_solves,
                  "C++23 replacement did not commit every restoration match");
            }
         }
      }
   }
   if( candidate_scenario != CandidateScenario::disabled &&
       std::getenv("IPOPT_CXX23_PRINT_CANDIDATE_STATS") != nullptr )
   {
      std::cout << fixed_treatment << ','
                << CandidateScenarioName(candidate_scenario) << ','
                << (start_with_restoration ? 1 : 0) << ','
                << (reoptimize ? 1 : 0) << ','
                << statistics.candidate_first_requests << ','
                << statistics.candidate_first_accepted << ','
                << statistics.candidate_first_fallbacks << ','
                << statistics.reference_solve_requests << ','
                << statistics.candidate_first_factorizations << ','
                << statistics.candidate_first_backsolves << ','
                << statistics.candidate_first_kkt_applications << ','
                << statistics.candidate_first_derivative_product_requests << ','
                << statistics.candidate_first_validation_kkt_applications
                << '\n';
   }
}
} // namespace

int main()
{
   try
   {
      if( std::getenv("IPOPT_CXX23_PRINT_CANDIDATE_STATS") != nullptr )
      {
         std::cout << "fixed_treatment,scenario,restoration,reoptimize,requests,"
                      "accepted,fallbacks,reference_solves,factorizations,"
                      "backsolves,backend_kkt_applications,"
                      "backend_derivative_product_requests,"
                      "validation_kkt_applications\n";
      }
      TestTreatment("make_parameter");
      TestTreatment("make_parameter_nodual");
      TestTreatment("make_constraint");
      TestTreatment("relax_bounds");
      TestTreatment("make_parameter", true);
      TestTreatment("make_parameter_nodual", true);
      TestTreatment("make_constraint", true);
      TestTreatment("relax_bounds", true);
      TestTreatment("make_parameter", false, true, false, true);
      TestTreatment("make_parameter", true, false, false, true);
      TestTreatment("make_parameter", false, false, true, true);
      TestTreatment(
         "make_parameter", false, false, false, false,
         CandidateScenario::accepted);
      TestTreatment(
         "make_parameter_nodual", false, false, false, false,
         CandidateScenario::accepted);
      TestTreatment(
         "make_constraint", false, false, false, false,
         CandidateScenario::accepted);
      TestTreatment(
         "relax_bounds", false, false, false, false,
         CandidateScenario::accepted);
      TestTreatment(
         "make_parameter", false, true, false, false,
         CandidateScenario::accepted);
      TestTreatment(
         "make_parameter", true, false, false, false,
         CandidateScenario::accepted);
      TestTreatment(
         "make_parameter", false, false, false, false,
         CandidateScenario::backend_failure_once);
      TestTreatment(
         "make_parameter", false, false, false, false,
         CandidateScenario::inertia_rejection_once);
      TestTreatment(
         "make_parameter", false, false, false, false,
         CandidateScenario::residual_rejection_once);
   }
   catch( const std::exception& exception )
   {
      std::cerr << exception.what() << '\n';
      return 1;
   }
   return 0;
}
