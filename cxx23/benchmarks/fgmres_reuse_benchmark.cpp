// Copyright (C) 2026 Ipopt contributors.
// All Rights Reserved.
// This file is distributed under the Eclipse Public License.

#include <ipopt/cxx23/fgmres.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <system_error>
#include <vector>

namespace
{
using namespace Ipopt::Cxx23;

inline void DoNotOptimize(const void* value)
{
#if defined(__GNUC__) || defined(__clang__)
   asm volatile("" : : "r"(value) : "memory");
#else
   (void)value;
   std::atomic_signal_fence(std::memory_order_seq_cst);
#endif
}

struct TridiagonalEight
{
   EvaluationResult operator()(std::span<const Number> input, std::span<Number> output) const
   {
      for( Index i = 0; i < input.size(); ++i )
      {
         output[i] = 4. * input[i];
         if( i > 0 )
         {
            output[i] -= input[i - 1];
         }
         if( i + 1 < input.size() )
         {
            output[i] -= input[i + 1];
         }
      }
      return {};
   }
};

struct Sample
{
   double nanoseconds_per_solve;
   std::size_t krylov_iterations;
   Number checksum;
};

FgmresOptions Options()
{
   FgmresOptions options;
   options.restart = 8;
   options.maximum_iterations = 8;
   options.relative_tolerance = 1e-12;
   options.absolute_tolerance = 1e-14;
   return options;
}

Sample Run(std::size_t solves, std::size_t warmup, bool reconstruct)
{
   constexpr Index dimension = 8;
   TridiagonalEight matrix;
   IdentityPreconditioner preconditioner;
   const FgmresOptions options = Options();
   FgmresSolver reused_solver(dimension, options);
   std::array<Number, dimension> exact_solution{};
   for( Index i = 0; i < dimension; ++i )
   {
      exact_solution[i] = static_cast<Number>(i + 1);
   }
   std::array<Number, dimension> base_right_hand_side{};
   if( EvaluationResult evaluated = matrix(exact_solution, base_right_hand_side); !evaluated )
   {
      throw std::runtime_error(evaluated.error().message);
   }

   std::array<Number, dimension> right_hand_side{};
   std::array<Number, dimension> solution{};
   std::size_t krylov_iterations = 0;
   Number checksum = 0.;
   const auto solve_once = [&](std::size_t sequence)
   {
      const Number scale = 1. + 1e-6 * static_cast<Number>(sequence % 17);
      for( Index i = 0; i < dimension; ++i )
      {
         right_hand_side[i] = scale * base_right_hand_side[i];
      }
      std::ranges::fill(solution, 0.);

      EvaluationValue<FgmresResult> result;
      if( reconstruct )
      {
         FgmresSolver solver(dimension, options);
         result = solver.solve(matrix, preconditioner, right_hand_side, solution);
      }
      else
      {
         result = reused_solver.solve(matrix, preconditioner, right_hand_side, solution);
      }
      if( !result )
      {
         throw std::runtime_error(result.error().message);
      }
      if( !result->converged() )
      {
         throw std::runtime_error("FGMRES benchmark solve did not converge");
      }
      krylov_iterations += result->iterations;
      checksum += solution[sequence % dimension];
      DoNotOptimize(solution.data());
   };

   for( std::size_t i = 0; i < warmup; ++i )
   {
      solve_once(i);
   }
   krylov_iterations = 0;
   checksum = 0.;
   const auto start = std::chrono::steady_clock::now();
   for( std::size_t i = 0; i < solves; ++i )
   {
      solve_once(i + warmup);
   }
   const auto stop = std::chrono::steady_clock::now();

   if( !std::isfinite(checksum) || checksum <= 0. )
   {
      throw std::runtime_error("invalid FGMRES benchmark checksum");
   }
   return {
      std::chrono::duration<double, std::nano>(stop - start).count() /
         static_cast<double>(solves),
      krylov_iterations,
      checksum
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

double Median(std::vector<double>& samples)
{
   std::ranges::sort(samples);
   return samples[samples.size() / 2];
}

void PrintSamples(const std::vector<double>& samples)
{
   std::cout << '[';
   for( std::size_t i = 0; i < samples.size(); ++i )
   {
      if( i != 0 )
      {
         std::cout << ',';
      }
      std::cout << samples[i];
   }
   std::cout << ']';
}
} // namespace

int main(int argc, char** argv)
{
   try
   {
      std::size_t solves = 20000;
      std::size_t repetitions = 7;
      std::size_t warmup = 100;
      for( int i = 1; i < argc; ++i )
      {
         const std::string_view argument(argv[i]);
         if( argument == "--help" )
         {
            std::cout << "Usage: " << argv[0]
                      << " [--solves N] [--repetitions N] [--warmup N]\n";
            return EXIT_SUCCESS;
         }
         if( i + 1 >= argc )
         {
            throw std::invalid_argument(std::string(argument) + " requires a value");
         }
         const std::string_view value(argv[++i]);
         if( argument == "--solves" )
         {
            solves = ParsePositive(value, argument);
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

      std::vector<double> reused_samples;
      std::vector<double> reconstructed_samples;
      reused_samples.reserve(repetitions);
      reconstructed_samples.reserve(repetitions);
      std::size_t expected_iterations = 0;
      for( std::size_t repetition = 0; repetition < repetitions; ++repetition )
      {
         const auto record = [&](const Sample& reused, const Sample& reconstructed)
         {
            if( reused.krylov_iterations != reconstructed.krylov_iterations )
            {
               throw std::runtime_error("benchmark paths used different Krylov iterations");
            }
            if( std::abs(reused.checksum - reconstructed.checksum) > 1e-10 )
            {
               throw std::runtime_error("benchmark paths produced different solutions");
            }
            if( expected_iterations == 0 )
            {
               expected_iterations = reused.krylov_iterations;
            }
            else if( expected_iterations != reused.krylov_iterations )
            {
               throw std::runtime_error("benchmark iteration count is unstable");
            }
            reused_samples.push_back(reused.nanoseconds_per_solve);
            reconstructed_samples.push_back(reconstructed.nanoseconds_per_solve);
         };
         if( repetition % 2 == 0 )
         {
            const Sample reused = Run(solves, warmup, false);
            const Sample reconstructed = Run(solves, warmup, true);
            record(reused, reconstructed);
         }
         else
         {
            const Sample reconstructed = Run(solves, warmup, true);
            const Sample reused = Run(solves, warmup, false);
            record(reused, reconstructed);
         }
      }

      const double reused_median = Median(reused_samples);
      const double reconstructed_median = Median(reconstructed_samples);
      std::cout << std::setprecision(17)
                << "{\"schema_version\":1"
                << ",\"name\":\"cxx23_fgmres_workspace_reuse\""
                << ",\"metric\":\"nanoseconds_per_solve\""
                << ",\"dimension\":8"
                << ",\"restart\":8"
                << ",\"maximum_iterations\":8"
                << ",\"solves\":" << solves
                << ",\"warmup\":" << warmup
                << ",\"repetitions\":" << repetitions
                << ",\"krylov_iterations_per_repetition\":" << expected_iterations
                << ",\"reused_median\":" << reused_median
                << ",\"reconstructed_median\":" << reconstructed_median
                << ",\"saved_ns\":" << reconstructed_median - reused_median
                << ",\"speedup\":" << reconstructed_median / reused_median
                << ",\"reused_samples\":";
      PrintSamples(reused_samples);
      std::cout << ",\"reconstructed_samples\":";
      PrintSamples(reconstructed_samples);
      std::cout << "}\n";
   }
   catch( const std::exception& error )
   {
      std::cerr << "ipopt_cxx23_fgmres_reuse_benchmark: " << error.what() << '\n';
      return EXIT_FAILURE;
   }
   return EXIT_SUCCESS;
}
