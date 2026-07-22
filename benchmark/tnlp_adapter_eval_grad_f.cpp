// Copyright (C) 2026 Ipopt contributors.
// All Rights Reserved.
// This file is distributed under the Eclipse Public License.

#include "IpIpoptApplication.hpp"
#include "IpDenseVector.hpp"
#include "IpTNLPAdapter.hpp"
#include "IpVector.hpp"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <system_error>
#include <vector>

using namespace Ipopt;

namespace
{
class FixedVariableNLP final : public TNLP
{
public:
   bool get_nlp_info(
      Index& n,
      Index& m,
      Index& nnz_jac_g,
      Index& nnz_h_lag,
      IndexStyleEnum& style
   ) override
   {
      n = 3;
      m = 0;
      nnz_jac_g = 0;
      nnz_h_lag = 3;
      style = C_STYLE;
      return true;
   }

   bool get_bounds_info(
      Index,
      Number* x_l,
      Number* x_u,
      Index,
      Number*,
      Number*
   ) override
   {
      x_l[0] = 0.;
      x_u[0] = 0.;
      x_l[1] = -1.;
      x_u[1] = 1.;
      x_l[2] = -1.;
      x_u[2] = 1.;
      return true;
   }

   bool get_starting_point(
      Index,
      bool init_x,
      Number* x,
      bool,
      Number*,
      Number*,
      Index,
      bool,
      Number*
   ) override
   {
      if( init_x )
      {
         x[0] = x[1] = x[2] = 0.;
      }
      return true;
   }

   bool eval_f(Index, const Number* x, bool, Number& value) override
   {
      value = x[1] * x[1] + x[2] * x[2];
      return true;
   }

   bool eval_grad_f(Index, const Number* x, bool, Number* gradient) override
   {
      ++gradient_evaluations_;
      gradient[0] = 0.;
      gradient[1] = 2. * x[1];
      gradient[2] = 2. * x[2];
      return true;
   }

   bool eval_g(Index, const Number*, bool, Index, Number*) override
   {
      return true;
   }

   bool eval_jac_g(
      Index,
      const Number*,
      bool,
      Index,
      Index,
      Index*,
      Index*,
      Number*
   ) override
   {
      return true;
   }

   bool eval_h(
      Index,
      const Number*,
      bool,
      Number obj_factor,
      Index,
      const Number*,
      bool,
      Index,
      Index* i_row,
      Index* j_col,
      Number* values
   ) override
   {
      if( values == nullptr )
      {
         for( Index i = 0; i < 3; ++i )
         {
            i_row[i] = i;
            j_col[i] = i;
         }
      }
      else
      {
         values[0] = 0.;
         values[1] = 2. * obj_factor;
         values[2] = 2. * obj_factor;
      }
      return true;
   }

   void finalize_solution(
      SolverReturn,
      Index,
      const Number*,
      const Number*,
      const Number*,
      Index,
      const Number*,
      const Number*,
      Number,
      const IpoptData*,
      IpoptCalculatedQuantities*
   ) override
   {
   }

   std::uint64_t GradientEvaluations() const
   {
      return gradient_evaluations_;
   }

private:
   std::uint64_t gradient_evaluations_ = 0;
};

struct Sample
{
   double ns_per_call;
   std::uint64_t gradient_evaluations;
};

Sample Run(std::size_t iterations, std::size_t warmup)
{
   SmartPtr<IpoptApplication> application = new IpoptApplication();
   application->Options()->SetIntegerValue("print_level", 0);
   if( application->Initialize() != Solve_Succeeded )
   {
      throw std::runtime_error("Ipopt initialization failed");
   }

   SmartPtr<FixedVariableNLP> concrete_problem = new FixedVariableNLP();
   SmartPtr<TNLP> problem = concrete_problem;
   SmartPtr<TNLPAdapter> adapter = new TNLPAdapter(problem, application->Jnlst());
   if( !adapter->ProcessOptions(*application->Options(), "") )
   {
      throw std::runtime_error("TNLPAdapter option processing failed");
   }

   SmartPtr<const VectorSpace> x_space;
   SmartPtr<const VectorSpace> c_space;
   SmartPtr<const VectorSpace> d_space;
   SmartPtr<const VectorSpace> x_l_space;
   SmartPtr<const MatrixSpace> px_l_space;
   SmartPtr<const VectorSpace> x_u_space;
   SmartPtr<const MatrixSpace> px_u_space;
   SmartPtr<const VectorSpace> d_l_space;
   SmartPtr<const MatrixSpace> pd_l_space;
   SmartPtr<const VectorSpace> d_u_space;
   SmartPtr<const MatrixSpace> pd_u_space;
   SmartPtr<const MatrixSpace> jac_c_space;
   SmartPtr<const MatrixSpace> jac_d_space;
   SmartPtr<const SymMatrixSpace> h_space;
   if( !adapter->GetSpaces(
          x_space, c_space, d_space, x_l_space, px_l_space, x_u_space, px_u_space,
          d_l_space, pd_l_space, d_u_space, pd_u_space, jac_c_space, jac_d_space,
          h_space) )
   {
      throw std::runtime_error("TNLPAdapter space construction failed");
   }

   SmartPtr<Vector> x = x_space->MakeNew();
   SmartPtr<Vector> gradient = x_space->MakeNew();
   x->Set(0.25);
   for( std::size_t i = 0; i < warmup; ++i )
   {
      if( !adapter->Eval_grad_f(*x, *gradient) )
      {
         throw std::runtime_error("TNLPAdapter gradient warm-up failed");
      }
   }

   const auto start = std::chrono::steady_clock::now();
   for( std::size_t i = 0; i < iterations; ++i )
   {
      if( !adapter->Eval_grad_f(*x, *gradient) )
      {
         throw std::runtime_error("TNLPAdapter gradient evaluation failed");
      }
   }
   const auto stop = std::chrono::steady_clock::now();

   const std::uint64_t expected_evaluations = iterations + warmup;
   if( concrete_problem->GradientEvaluations() != expected_evaluations )
   {
      throw std::runtime_error("unexpected gradient callback count");
   }
   const DenseVector* dense_gradient = static_cast<const DenseVector*>(GetRawPtr(gradient));
   if( dense_gradient->IsHomogeneous() || dense_gradient->Values()[0] != 0.5 ||
       dense_gradient->Values()[1] != 0.5 )
   {
      throw std::runtime_error("fixed-variable gradient mapping is incorrect");
   }

   return {
      std::chrono::duration<double, std::nano>(stop - start).count() /
         static_cast<double>(iterations),
      concrete_problem->GradientEvaluations()
   };
}

std::size_t ParsePositive(std::string_view text, std::string_view option)
{
   std::size_t result = 0;
   const auto conversion = std::from_chars(text.data(), text.data() + text.size(), result);
   if( conversion.ec != std::errc() || conversion.ptr != text.data() + text.size() || result == 0 )
   {
      throw std::invalid_argument(std::string(option) + " expects a positive integer");
   }
   return result;
}

void PrintUsage(const char* executable)
{
   std::cout << "Usage: " << executable
             << " [--iterations N] [--repetitions N] [--warmup N]\n";
}
} // namespace

int main(int argc, char** argv)
{
   try
   {
      std::size_t iterations = 20000000;
      std::size_t repetitions = 7;
      std::size_t warmup = 1000;

      for( int i = 1; i < argc; ++i )
      {
         const std::string_view argument(argv[i]);
         if( argument == "--help" )
         {
            PrintUsage(argv[0]);
            return EXIT_SUCCESS;
         }
         if( i + 1 >= argc )
         {
            throw std::invalid_argument(std::string(argument) + " requires a value");
         }
         const std::string_view value(argv[++i]);
         if( argument == "--iterations" )
         {
            iterations = ParsePositive(value, argument);
         }
         else if( argument == "--repetitions" )
         {
            repetitions = ParsePositive(value, argument);
         }
         else if( argument == "--warmup" )
         {
            warmup = ParsePositive(value, argument);
         }
         else
         {
            throw std::invalid_argument("unknown option: " + std::string(argument));
         }
      }

      if( repetitions < 3 || repetitions % 2 == 0 )
      {
         throw std::invalid_argument("--repetitions must be odd and at least 3");
      }

      std::vector<double> samples;
      samples.reserve(repetitions);
      std::uint64_t callback_evaluations = 0;
      for( std::size_t repetition = 0; repetition < repetitions; ++repetition )
      {
         const Sample sample = Run(iterations, warmup);
         samples.push_back(sample.ns_per_call);
         callback_evaluations += sample.gradient_evaluations;
      }
      std::sort(samples.begin(), samples.end());

      std::cout << std::setprecision(17)
                << "{\"schema_version\":1"
                << ",\"name\":\"tnlp_adapter_eval_grad_f_fixed_variable\""
                << ",\"metric\":\"nanoseconds_per_call\""
                << ",\"lower_is_better\":true"
                << ",\"median\":" << samples[samples.size() / 2]
                << ",\"iterations\":" << iterations
                << ",\"warmup\":" << warmup
                << ",\"repetitions\":" << repetitions
                << ",\"callback_evaluations\":" << callback_evaluations
                << ",\"cplusplus\":" << __cplusplus
                << ",\"compiler\":\"" << __VERSION__ << "\""
                << ",\"samples\":[";
      for( std::size_t i = 0; i < samples.size(); ++i )
      {
         if( i != 0 )
         {
            std::cout << ',';
         }
         std::cout << samples[i];
      }
      std::cout << "]}\n";
   }
   catch( const std::exception& error )
   {
      std::cerr << "tnlp_adapter_eval_grad_f_benchmark: " << error.what() << '\n';
      return EXIT_FAILURE;
   }
   return EXIT_SUCCESS;
}
