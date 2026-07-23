// Copyright (C) 2026 Ipopt contributors.
// All Rights Reserved.
// This file is distributed under the Eclipse Public License.

#include <IpIpoptApplication.hpp>
#include <IpSolveStatistics.hpp>
#include <IpTNLP.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
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

class TrackedTnlp : public Stable::TNLP
{
public:
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
      ++trajectory_points_;
      if( mode == Stable::RestorationPhaseMode )
      {
         ++restoration_points_;
      }
      return true;
   }

   void finalize_solution(
      Stable::SolverReturn,
      Stable::Index n,
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
      final_x_.assign(x, x + n);
      final_objective_ = objective;
   }

   virtual Stable::Number solution_error() const = 0;
   virtual Stable::Number constraint_error() const = 0;

   Stable::Number final_objective() const
   {
      return final_objective_;
   }

   Stable::Index trajectory_points() const
   {
      return trajectory_points_;
   }

   Stable::Index restoration_points() const
   {
      return restoration_points_;
   }

protected:
   std::span<const Stable::Number> final_x() const
   {
      return final_x_;
   }

private:
   std::vector<Stable::Number> final_x_;
   Stable::Number final_objective_ =
      std::numeric_limits<Stable::Number>::quiet_NaN();
   Stable::Index trajectory_points_ = 0;
   Stable::Index restoration_points_ = 0;
};

class IllScaledNlp final : public TrackedTnlp
{
public:
   IllScaledNlp(
      Stable::Number small_scale = 1e-12,
      Stable::Number large_scale = 1e12
   )
      : small_scale_(small_scale),
        large_scale_(large_scale)
   {
   }

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
      x_l[0] = .01 * small_scale_;
      x_u[0] = 100. * small_scale_;
      x_l[1] = .01 * large_scale_;
      x_u[1] = 100. * large_scale_;
      g_l[0] = 2.;
      g_u[0] = 2.;
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
      x[0] = .2 * small_scale_;
      x[1] = 2.5 * large_scale_;
      return true;
   }

   bool eval_f(
      Stable::Index,
      const Stable::Number* x,
      bool,
      Stable::Number& objective
   ) override
   {
      const Stable::Number first = x[0] / small_scale_ - 1.;
      const Stable::Number second = x[1] / large_scale_ - 1.;
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
      gradient[0] = (x[0] / small_scale_ - 1.) / small_scale_;
      gradient[1] = (x[1] / large_scale_ - 1.) / large_scale_;
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
      constraints[0] = x[0] / small_scale_ + x[1] / large_scale_;
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
         values[0] = 1. / small_scale_;
         values[1] = 1. / large_scale_;
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
         values[0] = objective_factor / (small_scale_ * small_scale_);
         values[1] = objective_factor / (large_scale_ * large_scale_);
      }
      return true;
   }

   Stable::Number solution_error() const override
   {
      if( final_x().size() != 2 )
      {
         return std::numeric_limits<Stable::Number>::infinity();
      }
      return std::max(
         std::abs(final_x()[0] / small_scale_ - 1.),
         std::abs(final_x()[1] / large_scale_ - 1.));
   }

   Stable::Number constraint_error() const override
   {
      if( final_x().size() != 2 )
      {
         return std::numeric_limits<Stable::Number>::infinity();
      }
      return std::abs(
         final_x()[0] / small_scale_ +
         final_x()[1] / large_scale_ - 2.);
   }

private:
   Stable::Number small_scale_;
   Stable::Number large_scale_;
};

class DegenerateNlp final : public TrackedTnlp
{
public:
   explicit DegenerateNlp(Stable::Number epsilon)
      : epsilon_(epsilon)
   {
   }

   bool get_nlp_info(
      Stable::Index& n,
      Stable::Index& m,
      Stable::Index& nnz_jac_g,
      Stable::Index& nnz_h_lag,
      IndexStyleEnum& index_style
   ) override
   {
      n = 2;
      m = 2;
      nnz_jac_g = 4;
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
      g_l[0] = g_u[0] = 3.;
      g_l[1] = g_u[1] = 3. + 2. * epsilon_;
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
      constraints[1] = x[0] + (1. + epsilon_) * x[1];
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
         columns[0] = 0;
         rows[1] = 0;
         columns[1] = 1;
         rows[2] = 1;
         columns[2] = 0;
         rows[3] = 1;
         columns[3] = 1;
      }
      else
      {
         values[0] = 1.;
         values[1] = 1.;
         values[2] = 1.;
         values[3] = 1. + epsilon_;
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

   Stable::Number solution_error() const override
   {
      if( final_x().size() != 2 )
      {
         return std::numeric_limits<Stable::Number>::infinity();
      }
      return std::max(
         std::abs(final_x()[0] - 1.),
         std::abs(final_x()[1] - 2.));
   }

   Stable::Number constraint_error() const override
   {
      if( final_x().size() != 2 )
      {
         return std::numeric_limits<Stable::Number>::infinity();
      }
      return std::max(
         std::abs(final_x()[0] + final_x()[1] - 3.),
         std::abs(
            final_x()[0] + (1. + epsilon_) * final_x()[1] -
            (3. + 2. * epsilon_)));
   }

private:
   Stable::Number epsilon_;
};

struct RunResult
{
   Stable::ApplicationReturnStatus status;
   Stable::Index iterations;
   Stable::Index trajectory_points;
   Stable::Index restoration_points;
   Stable::Number objective;
   Stable::Number solution_error;
   Stable::Number constraint_error;
   double milliseconds;
};

RunResult RunCase(std::string_view name)
{
   Stable::SmartPtr<TrackedTnlp> problem;
   if( name == "ill-scaled-exact" ||
       name == "ill-scaled-restoration" )
   {
      problem = new IllScaledNlp();
   }
   else if( name == "ill-scaled-lbfgs" )
   {
      problem = new IllScaledNlp(1e-4, 1e4);
   }
   else if( name == "near-degenerate" )
   {
      problem = new DegenerateNlp(1e-10);
   }
   else if( name == "degenerate" )
   {
      problem = new DegenerateNlp(0.);
   }
   else
   {
      throw std::runtime_error("unknown adversarial case: " + std::string(name));
   }

   Stable::SmartPtr<Stable::IpoptApplication> application =
      IpoptApplicationFactory();
   const char* print_level = std::getenv("IPOPT_ADVERSARIAL_PRINT_LEVEL");
   application->Options()->SetIntegerValue(
      "print_level", print_level == nullptr ? 0 : std::atoi(print_level));
   application->Options()->SetStringValue("sb", "yes");
   application->Options()->SetStringValue("linear_solver", "mumps");
   application->Options()->SetStringValue("nlp_scaling_method", "none");
   application->Options()->SetStringValue(
      "linear_system_refinement", "iterative-refinement");
   application->Options()->SetNumericValue("tol", 1e-10);
   application->Options()->SetNumericValue("acceptable_tol", 1e-4);
   application->Options()->SetIntegerValue("acceptable_iter", 5);
   application->Options()->SetIntegerValue("max_iter", 200);
   if( name == "ill-scaled-lbfgs" )
   {
      application->Options()->SetStringValue(
         "hessian_approximation", "limited-memory");
      application->Options()->SetStringValue(
         "limited_memory_initialization", "scalar4");
   }
   if( name == "ill-scaled-restoration" )
   {
      application->Options()->SetStringValue("start_with_resto", "yes");
   }
   if( name == "degenerate" )
   {
      application->Options()->SetStringValue("dependency_detector", "mumps");
   }
   Check(
      application->Initialize() == Stable::Solve_Succeeded,
      "Ipopt initialization failed");

   Stable::SmartPtr<Stable::TNLP> tnlp = Stable::GetRawPtr(problem);
   const auto start = std::chrono::steady_clock::now();
   const Stable::ApplicationReturnStatus status =
      application->OptimizeTNLP(tnlp);
   const auto stop = std::chrono::steady_clock::now();
   const double milliseconds =
      std::chrono::duration<double, std::milli>(stop - start).count();
   const Stable::Index iterations =
      Stable::IsValid(application->Statistics())
      ? application->Statistics()->IterationCount()
      : -1;
   return {
      status,
      iterations,
      problem->trajectory_points(),
      problem->restoration_points(),
      problem->final_objective(),
      problem->solution_error(),
      problem->constraint_error(),
      milliseconds
   };
}

void Validate(std::string_view name, const RunResult& result)
{
   Check(
      result.status == Stable::Solve_Succeeded ||
         result.status == Stable::Solved_To_Acceptable_Level,
      std::string(name) + " did not converge (status=" +
         std::to_string(static_cast<int>(result.status)) +
         ", iterations=" + std::to_string(result.iterations) +
         ", objective=" + std::to_string(result.objective) +
         ", solution_error=" + std::to_string(result.solution_error) +
         ", constraint_error=" + std::to_string(result.constraint_error) +
         ")");
   const Stable::Number solution_tolerance =
      name == "ill-scaled-lbfgs" ? 2e-5 :
      (name.starts_with("ill-scaled") ? 2e-6 : 2e-7);
   Check(
      result.solution_error <= solution_tolerance,
      std::string(name) + " has an inaccurate solution");
   Check(
      result.constraint_error <= 2e-8,
      std::string(name) + " has an inaccurate constraint");
   Check(
      std::isfinite(result.objective),
      std::string(name) + " has a nonfinite objective");
   if( name == "ill-scaled-restoration" )
   {
      Check(
         result.restoration_points > 0,
         "forced restoration case did not enter restoration");
   }
}
} // namespace

int main(int argc, char** argv)
{
   try
   {
      std::vector<std::string> cases{
         "ill-scaled-exact",
         "ill-scaled-lbfgs",
         "ill-scaled-restoration",
         "near-degenerate",
         "degenerate"
      };
      int repeats = 1;
      for( int argument = 1; argument < argc; ++argument )
      {
         const std::string_view option(argv[argument]);
         if( option == "--case" && argument + 1 < argc )
         {
            cases = {argv[++argument]};
         }
         else if( option == "--repeats" && argument + 1 < argc )
         {
            repeats = std::atoi(argv[++argument]);
            Check(repeats > 0, "--repeats must be positive");
         }
         else
         {
            throw std::runtime_error("invalid adversarial test argument");
         }
      }

      std::cout << std::setprecision(17);
      for( const std::string& name : cases )
      {
         for( int repeat = 0; repeat < repeats; ++repeat )
         {
            const RunResult result = RunCase(name);
            Validate(name, result);
            std::cout
               << "case=" << name
               << ",repeat=" << repeat
               << ",status=" << static_cast<int>(result.status)
               << ",ms=" << result.milliseconds
               << ",iterations=" << result.iterations
               << ",trajectory=" << result.trajectory_points
               << ",restoration=" << result.restoration_points
               << ",objective=" << result.objective
               << ",solution_error=" << result.solution_error
               << ",constraint_error=" << result.constraint_error
               << '\n';
         }
      }
   }
   catch( const std::exception& error )
   {
      std::cerr << "ipopt_cxx23_adversarial_nlp_test: "
                << error.what() << '\n';
      return 1;
   }
   return 0;
}
