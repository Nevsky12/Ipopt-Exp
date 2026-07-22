// Copyright (C) 2026 Ipopt contributors.
// All Rights Reserved.
// This file is distributed under the Eclipse Public License.

#include <IpAlgBuilder.hpp>
#include <IpIpoptApplication.hpp>
#include <IpIpoptData.hpp>
#include <IpPDFullSpaceSolver.hpp>
#include <IpTNLPAdapter.hpp>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace
{
namespace Stable = ::Ipopt;

void Check(bool condition, std::string_view message)
{
   if( !condition )
   {
      throw std::runtime_error(std::string(message));
   }
}

class EqualityQuadratic final : public Stable::TNLP
{
public:
   bool get_nlp_info(
      Stable::Index& n,
      Stable::Index& m,
      Stable::Index& nnz_jac_g,
      Stable::Index& nnz_h_lag,
      IndexStyleEnum& index_style
   ) override
   {
      n = 2;
      m = 1;
      nnz_jac_g = 2;
      nnz_h_lag = 2;
      index_style = C_STYLE;
      return true;
   }

   bool get_bounds_info(
      Stable::Index,
      Stable::Number* x_l,
      Stable::Number* x_u,
      Stable::Index,
      Stable::Number* g_l,
      Stable::Number* g_u
   ) override
   {
      std::fill_n(x_l, 2, -10.);
      std::fill_n(x_u, 2, 10.);
      g_l[0] = 3.;
      g_u[0] = 3.;
      return true;
   }

   bool get_starting_point(
      Stable::Index,
      bool init_x,
      Stable::Number* x,
      bool init_z,
      Stable::Number*,
      Stable::Number*,
      Stable::Index,
      bool init_lambda,
      Stable::Number*
   ) override
   {
      if( !init_x || init_z || init_lambda )
      {
         return false;
      }
      x[0] = -1.;
      x[1] = 1.;
      return true;
   }

   bool eval_f(
      Stable::Index,
      const Stable::Number* x,
      bool,
      Stable::Number& objective
   ) override
   {
      const Stable::Number first = x[0] - 1.;
      const Stable::Number second = x[1] - 2.;
      objective = .5 * (first * first + second * second);
      return true;
   }

   bool eval_grad_f(
      Stable::Index,
      const Stable::Number* x,
      bool,
      Stable::Number* gradient
   ) override
   {
      gradient[0] = x[0] - 1.;
      gradient[1] = x[1] - 2.;
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
         rows[0] = 0;
         rows[1] = 0;
         columns[0] = 0;
         columns[1] = 1;
      }
      else
      {
         values[0] = 1.;
         values[1] = 1.;
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
         rows[0] = 0;
         columns[0] = 0;
         rows[1] = 1;
         columns[1] = 1;
      }
      else
      {
         values[0] = objective_factor;
         values[1] = objective_factor;
      }
      return true;
   }

   bool intermediate_callback(
      Stable::AlgorithmMode,
      Stable::Index iteration,
      Stable::Number objective,
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
      trajectory_.push_back(std::make_pair(iteration, objective));
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
      Stable::Number objective,
      const Stable::IpoptData*,
      Stable::IpoptCalculatedQuantities*
   ) override
   {
      final_x_[0] = x[0];
      final_x_[1] = x[1];
      final_objective_ = objective;
   }

   const std::vector<std::pair<Stable::Index, Stable::Number>>&
   trajectory() const noexcept
   {
      return trajectory_;
   }

   Stable::Number final_x(Stable::Index index) const noexcept
   {
      return final_x_[index];
   }

   Stable::Number final_objective() const noexcept
   {
      return final_objective_;
   }

private:
   Stable::Number final_x_[2]{0., 0.};
   Stable::Number final_objective_ = 0.;
   std::vector<std::pair<Stable::Index, Stable::Number>> trajectory_;
};

class CapturingBuilder final : public Stable::AlgorithmBuilder
{
public:
   Stable::SmartPtr<Stable::PDSystemSolver> PDSystemSolverFactory(
      const Stable::Journalist& journalist,
      const Stable::OptionsList& options,
      const std::string& prefix
   ) override
   {
      Stable::SmartPtr<Stable::PDSystemSolver> solver =
         Stable::AlgorithmBuilder::PDSystemSolverFactory(
            journalist, options, prefix);
      auto* full = dynamic_cast<Stable::PDFullSpaceSolver*>(
         Stable::GetRawPtr(solver));
      if( full != nullptr )
      {
         solvers_.push_back(full);
      }
      return solver;
   }

   Stable::PDFullSpaceSolverRefinementStatistics statistics() const
   {
      Stable::PDFullSpaceSolverRefinementStatistics total;
      for( const auto& solver : solvers_ )
      {
         const auto& source = solver->RefinementStatistics();
         total.solve_calls += source.solve_calls;
         total.direct_solve_calls += source.direct_solve_calls;
         total.factorization_attempts += source.factorization_attempts;
         total.refinement_phases += source.refinement_phases;
         total.refinement_iterations += source.refinement_iterations;
         total.refinement_backsolves += source.refinement_backsolves;
         total.kkt_applications += source.kkt_applications;
         total.converged_refinements += source.converged_refinements;
         total.failed_refinements += source.failed_refinements;
         total.preconditioner_cache_misses +=
            source.preconditioner_cache_misses;
         total.maximum_initial_residual_ratio = std::max(
            total.maximum_initial_residual_ratio,
            source.maximum_initial_residual_ratio);
         total.maximum_final_residual_ratio = std::max(
            total.maximum_final_residual_ratio,
            source.maximum_final_residual_ratio);
         total.maximum_delta_c = std::max(
            total.maximum_delta_c, source.maximum_delta_c);
         total.maximum_delta_d = std::max(
            total.maximum_delta_d, source.maximum_delta_d);
      }
      return total;
   }

private:
   std::vector<Stable::SmartPtr<Stable::PDFullSpaceSolver>> solvers_;
};

struct RunResult
{
   Stable::PDFullSpaceSolverRefinementStatistics statistics;
   Stable::Number objective;
   Stable::Number first;
   Stable::Number second;
   std::vector<std::pair<Stable::Index, Stable::Number>> trajectory;
};

RunResult Run(std::string_view method)
{
   Stable::SmartPtr<Stable::IpoptApplication> application =
      IpoptApplicationFactory();
   application->Options()->SetIntegerValue("print_level", 0);
   application->Options()->SetStringValue("sb", "yes");
   application->Options()->SetStringValue("linear_solver", "mumps");
   application->Options()->SetStringValue("nlp_scaling_method", "none");
   application->Options()->SetStringValue("perturb_always_cd", "yes");
   application->Options()->SetStringValue(
      "linear_system_refinement", std::string(method));
   application->Options()->SetNumericValue("tol", 1e-10);
   application->Options()->SetIntegerValue("max_iter", 50);
   Check(
      application->Initialize() == Stable::Solve_Succeeded,
      "Ipopt initialization failed");

   Stable::SmartPtr<EqualityQuadratic> problem = new EqualityQuadratic();
   Stable::SmartPtr<Stable::TNLP> tnlp = Stable::GetRawPtr(problem);
   Stable::SmartPtr<Stable::NLP> nlp =
      new Stable::TNLPAdapter(tnlp, application->Jnlst());
   Stable::SmartPtr<CapturingBuilder> capturing_builder =
      new CapturingBuilder();
   Stable::SmartPtr<Stable::AlgorithmBuilder> builder =
      Stable::GetRawPtr(capturing_builder);

   const Stable::ApplicationReturnStatus status =
      application->OptimizeNLP(nlp, builder);
   Check(
      status == Stable::Solve_Succeeded
         || status == Stable::Solved_To_Acceptable_Level,
      "refinement integration solve did not converge");

   return RunResult{
      capturing_builder->statistics(),
      problem->final_objective(),
      problem->final_x(0),
      problem->final_x(1),
      problem->trajectory()
   };
}

void TestIntegratedRefinementModes()
{
   const RunResult stock = Run("iterative-refinement");
   const RunResult perturbed = Run("fgmres-kdelta");
   const RunResult unperturbed = Run("fgmres-k");

   for( const RunResult* result : {&stock, &perturbed, &unperturbed} )
   {
      Check(result->statistics.refinement_phases != 0,
            "refinement phase was not exercised");
      Check(result->statistics.maximum_delta_c > 0.,
            "forced constraint perturbation was not active");
      Check(result->statistics.maximum_delta_d > 0.,
            "forced inequality perturbation was not active");
      Check(result->statistics.preconditioner_cache_misses == 0,
            "refinement requested a new factorization");
      Check(std::abs(result->first - 1.) < 1e-7,
            "first primal solution is inaccurate");
      Check(std::abs(result->second - 2.) < 1e-7,
            "second primal solution is inaccurate");
      Check(result->objective < 1e-12,
            "quadratic objective is inaccurate");
      Check(!result->trajectory.empty(),
            "Ipopt trajectory was not captured");
   }

   Check(
      stock.statistics.kkt_applications
         == stock.statistics.refinement_phases
            + stock.statistics.refinement_iterations,
      "stock refinement KKT-apply accounting is inconsistent");
   for( const RunResult* result : {&perturbed, &unperturbed} )
   {
      Check(
         result->statistics.refinement_backsolves
            == result->statistics.refinement_iterations,
         "FGMRES did not use exactly one cached backsolve per iteration");
      Check(
         result->statistics.kkt_applications
            == result->statistics.refinement_phases
               + 2 * result->statistics.refinement_iterations,
         "FGMRES full-KKT apply accounting is inconsistent");
   }

   Check(
      unperturbed.statistics.maximum_initial_residual_ratio
         > 100. * perturbed.statistics.maximum_initial_residual_ratio,
      "K/M=K_delta did not expose the perturbation residual");
   Check(
      unperturbed.statistics.maximum_final_residual_ratio <= 1e-10,
      "K/M=K_delta did not converge against the unperturbed KKT");
}
} // namespace

int main()
{
   try
   {
      TestIntegratedRefinementModes();
   }
   catch( const std::exception& error )
   {
      std::cerr << "ipopt_cxx23_pdfullspace_fgmres_refinement_test: "
                << error.what() << '\n';
      return 1;
   }
   std::cout << "ipopt_cxx23_pdfullspace_fgmres_refinement_test: passed\n";
   return 0;
}
