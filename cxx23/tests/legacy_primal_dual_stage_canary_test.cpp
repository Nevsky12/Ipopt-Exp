// Copyright (C) 2026 Ipopt contributors.
// All Rights Reserved.
// This file is distributed under the Eclipse Public License.

#include <ipopt/cxx23/legacy_algorithm_canary.hpp>
#include <ipopt/cxx23/primal_dual_stage_kkt_assembler.hpp>
#include <ipopt/cxx23/restoration_stage_nlp.hpp>
#include <ipopt/cxx23/sparse_stage_derivative_provider.hpp>

#include <IpIpoptApplication.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace
{
namespace Stable = ::Ipopt;
namespace Cxx = ::Ipopt::Cxx23;

void Check(bool condition, std::string_view message)
{
   if( !condition )
   {
      throw std::runtime_error(std::string(message));
   }
}

struct LiveStageCounters
{
   Cxx::Index factory_calls = 0;
   Cxx::Index main_creations = 0;
   Cxx::Index restoration_creations = 0;
};

class LiveStageTnlp final : public Stable::TNLP
{
public:
   explicit LiveStageTnlp(bool infeasible_start)
      : infeasible_start_(infeasible_start)
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
      constraints = 3;
      jacobian_nonzeros = 7;
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
      // Generic primal order is [x_0,u_0,x_1].
      constexpr std::array<Stable::Number, 3> lower_x{{-2., -1., -3.}};
      constexpr std::array<Stable::Number, 3> upper_x{{2., 1., 3.}};
      // Original TNLP rows are [path equality, path inequality, dynamics].
      constexpr std::array<Stable::Number, 3> lower_g{{0., -.5, 0.}};
      constexpr std::array<Stable::Number, 3> upper_g{{0., .5, 0.}};
      std::ranges::copy(lower_x, x_lower);
      std::ranges::copy(upper_x, x_upper);
      std::ranges::copy(lower_g, g_lower);
      std::ranges::copy(upper_g, g_upper);
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
         const std::array<Stable::Number, 3> initial = infeasible_start_
            ? std::array<Stable::Number, 3>{{1.5, .8, -1.}}
            : std::array<Stable::Number, 3>{{-.4, .2, -.04}};
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
      objective = .5 * (3. * x[0] * x[0] +
         4. * x[1] * x[1] + 2. * x[2] * x[2]);
      return true;
   }

   bool eval_grad_f(
      Stable::Index,
      const Stable::Number* x,
      bool,
      Stable::Number* gradient
   ) override
   {
      gradient[0] = 3. * x[0];
      gradient[1] = 4. * x[1];
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
      constraints[0] = x[1] + .5 * x[0];
      constraints[1] = .25 * x[1] - x[0];
      constraints[2] = 1.2 * x[1] + .7 * x[0] - x[2];
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
         constexpr std::array<Stable::Index, 7> expected_rows{{
            0, 0, 1, 1, 2, 2, 2
         }};
         constexpr std::array<Stable::Index, 7> expected_columns{{
            1, 0, 1, 0, 1, 0, 2
         }};
         std::ranges::copy(expected_rows, rows);
         std::ranges::copy(expected_columns, columns);
      }
      else
      {
         constexpr std::array<Stable::Number, 7> expected{{
            1., .5, .25, -1., 1.2, .7, -1.
         }};
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
         values[0] = 3. * objective_factor;
         values[1] = 4. * objective_factor;
         values[2] = 2. * objective_factor;
      }
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

   void finalize_solution(
      Stable::SolverReturn,
      Stable::Index,
      const Stable::Number* x,
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
      std::copy_n(x, final_x_.size(), final_x_.begin());
   }

   std::span<const Stable::Number> final_x() const noexcept
   {
      return final_x_;
   }

   Stable::Index restoration_callbacks() const noexcept
   {
      return restoration_callbacks_;
   }

private:
   bool infeasible_start_;
   std::array<Stable::Number, 3> final_x_{};
   Stable::Index restoration_callbacks_ = 0;
};

Cxx::StageNlpTopology MakeLiveTopology(std::uint64_t source_revision)
{
   return Cxx::StageNlpTopology(
      {3, 3, 7, 3, source_revision},
      {
         {1, 1, 1, 1},
         {0, 1, 0, 0}
      },
      {
         .primal = {1, 0, 2},
         // TNLPAdapter moves equalities before inequalities, preserving
         // original order within each category: [path_eq,dynamics,path_ineq].
         .dynamics = {1},
         .path_equalities = {0},
         .path_inequalities = {2}
      },
      501);
}

Cxx::PrimalDualLayout MainLiveLayout()
{
   return {
      .equality_constraints = {0, 1},
      .inequality_constraints = {2},
      .primal_lower_bounds = {0, 1, 2},
      .primal_upper_bounds = {0, 1, 2},
      .slack_lower_bounds = {0},
      .slack_upper_bounds = {0}
   };
}

using LiveStageProvider = Cxx::SparseStageDerivativeProvider;
using LiveAssembler = Cxx::PrimalDualStageKktAssembler<LiveStageProvider>;
using LiveBackend = Cxx::StageStructuredCandidateBackend<LiveAssembler>;

class LiveStageBackendFactory
{
public:
   explicit LiveStageBackendFactory(
      std::shared_ptr<LiveStageCounters> counters
   )
      : counters_(std::move(counters))
   {
   }

   Cxx::EvaluationValue<LiveBackend> operator()(
      Cxx::PrimalDualKktOperator& kkt,
      bool                        restoration_problem
   )
   {
      ++counters_->factory_calls;
      const Cxx::NlpStructure live_structure = kkt.nlp_structure();
      Cxx::EvaluationValue<Cxx::StageNlpTopology> topology = [&]()
         -> Cxx::EvaluationValue<Cxx::StageNlpTopology>
      {
         if( !restoration_problem )
         {
            if( live_structure.variables != 3 ||
                live_structure.constraints != 3 ||
                live_structure.jacobian_nonzeros != 7 ||
                live_structure.hessian_nonzeros != 3 ||
                kkt.layout() != MainLiveLayout() )
            {
               return std::unexpected(Cxx::EvaluationError{
                  Cxx::EvaluationErrorCode::structure_mismatch,
                  "live OCP topology does not match the initialized main NLP"
               });
            }
            return MakeLiveTopology(live_structure.revision);
         }
         Cxx::EvaluationValue<Cxx::RestorationStageVariableLayout> variables =
            Cxx::MakeCanonicalRestoIpoptVariableLayout(3, 2, 1);
         if( !variables )
         {
            return std::unexpected(variables.error());
         }
         return Cxx::MakeRestorationStageNlpTopology(
            MakeLiveTopology(1),
            MainLiveLayout(),
            live_structure,
            kkt.layout(),
            *variables,
            502);
      }();
      if( !topology )
      {
         return std::unexpected(topology.error());
      }
      LiveStageProvider provider(kkt, std::move(*topology));
      if( !provider.configured() )
      {
         return std::unexpected(Cxx::EvaluationError{
            Cxx::EvaluationErrorCode::invalid_layout,
            provider.configuration_error()
         });
      }
      LiveAssembler assembler(
         std::move(provider),
         kkt,
         {
            .reverse_stage_order = true,
            .restoration_problem = restoration_problem
         });
      if( !assembler.configured() )
      {
         return std::unexpected(Cxx::EvaluationError{
            Cxx::EvaluationErrorCode::invalid_layout,
            assembler.configuration_error()
         });
      }
      if( restoration_problem )
      {
         ++counters_->restoration_creations;
      }
      else
      {
         ++counters_->main_creations;
      }
      Cxx::StageStructuredCandidateOptions options;
      options.refinement.relative_tolerance = 1e-11;
      options.refinement.maximum_steps = 4;
      return LiveBackend(std::move(assembler), options);
   }

private:
   std::shared_ptr<LiveStageCounters> counters_;
};

void RunLiveStageSolve(bool start_with_restoration)
{
   Stable::SmartPtr<Stable::IpoptApplication> application =
      new Stable::IpoptApplication(false);
   Check(
      application->Options()->SetIntegerValue("print_level", 0),
      "failed to set live stage print level");
   Check(
      application->Options()->SetIntegerValue("max_iter", 40),
      "failed to set live stage iteration limit");
   Check(
      application->Options()->SetNumericValue("tol", 1e-9),
      "failed to set live stage tolerance");
   Check(
      application->Options()->SetStringValue("nlp_scaling_method", "none"),
      "failed to disable live stage scaling");
   Check(
      application->Options()->SetStringValue(
         "fixed_variable_treatment", "make_parameter"),
      "failed to set live stage fixed-variable treatment");
   if( start_with_restoration )
   {
      Check(
         application->Options()->SetStringValue("start_with_resto", "yes"),
         "failed to force live stage restoration");
   }
   Check(
      application->Initialize("") == Stable::Solve_Succeeded,
      "live stage Ipopt initialization failed");

   Stable::SmartPtr<LiveStageTnlp> tnlp =
      new LiveStageTnlp(start_with_restoration);
   Stable::SmartPtr<Stable::NLP> nlp =
      new Stable::TNLPAdapter(tnlp, application->Jnlst());
   const auto counters = std::make_shared<LiveStageCounters>();
   Cxx::LegacyAlgorithmCanaryOptions canary_options;
   canary_options.mode = Cxx::LegacyAlgorithmCanaryMode::candidate_first;
   canary_options.residual_relative_tolerance = 2e-9;
   canary_options.candidate_first_backend = Cxx::MakeLazyCandidateFirstBackend(
      LiveStageBackendFactory(counters));
   Stable::SmartPtr<Cxx::LegacyAlgorithmCanaryBuilder> canary_builder =
      new Cxx::LegacyAlgorithmCanaryBuilder(canary_options);
   Stable::SmartPtr<Stable::AlgorithmBuilder> builder =
      Stable::GetRawPtr(canary_builder);

   const Stable::ApplicationReturnStatus status =
      application->OptimizeNLP(nlp, builder);
   Check(status == Stable::Solve_Succeeded, "live stage solve did not converge");
   const Cxx::LegacyAlgorithmCanaryStatistics& statistics =
      canary_builder->statistics();
   if( statistics.candidate_first_requests == 0 ||
       statistics.candidate_first_accepted == 0 )
   {
      throw std::runtime_error(
         "live full-stage backend was not accepted: requests=" +
         std::to_string(statistics.candidate_first_requests) +
         " fallbacks=" +
         std::to_string(statistics.candidate_first_fallbacks) +
         " reason=" + statistics.last_candidate_fallback_reason);
   }
   Check(
      statistics.candidate_first_requests ==
         statistics.candidate_first_accepted +
            statistics.candidate_first_fallbacks,
      "live full-stage request accounting is inconsistent");
   Check(
      statistics.reference_solve_requests ==
         statistics.skipped_solves + statistics.candidate_first_fallbacks,
      "live full-stage fallback did not call stable code exactly once");
   Check(
      statistics.candidate_first_inertia_rejections == 0 &&
         statistics.candidate_first_residual_rejections == 0 &&
         statistics.candidate_first_commit_failures == 0,
      "live full-stage direction failed a candidate gate");
   Check(
      counters->main_creations > 0 &&
         counters->factory_calls ==
            counters->main_creations + counters->restoration_creations &&
         statistics.candidate_first_derivative_product_requests ==
            statistics.candidate_first_accepted,
      "live full-stage lazy lifecycle or derivative accounting is wrong");
   Check(
      std::ranges::all_of(
         tnlp->final_x(),
         [](Stable::Number value) { return std::abs(value) < 2e-7; }),
      "live full-stage solve converged to the wrong primal point");

   if( start_with_restoration &&
       Cxx::LegacyAlgorithmCanaryBuilder::restoration_factory_supported() )
   {
      Check(
         tnlp->restoration_callbacks() > 0 &&
            statistics.restoration_candidate_first_requests > 0,
         "forced live-stage restoration was not routed through the canary");
      Check(
         statistics.restoration_candidate_first_accepted ==
            statistics.restoration_candidate_first_requests &&
            statistics.restoration_candidate_first_fallbacks == 0 &&
            statistics.candidate_first_fallbacks == 0 &&
            counters->restoration_creations > 0,
         "explicit restoration stage metadata was not accepted");
   }
   else
   {
      Check(
         statistics.candidate_first_fallbacks == 0,
         "ordinary live full-stage solve used stable fallback");
   }
}
} // namespace

int main()
{
   try
   {
      RunLiveStageSolve(false);
      RunLiveStageSolve(true);
   }
   catch( const std::exception& exception )
   {
      std::cerr << exception.what() << '\n';
      return 1;
   }
   return 0;
}
