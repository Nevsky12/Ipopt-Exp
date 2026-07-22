// Copyright (C) 2026 Ipopt contributors.
// All Rights Reserved.
// This file is distributed under the Eclipse Public License.

#include <ipopt/cxx23/any_nlp_problem.hpp>

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

struct NativeFourByFourModel
{
   NlpStructure structure() const
   {
      return {4, 4, 4, 4, 1};
   }

   EvaluationValue<Number> eval_objective(std::span<const Number> x)
   {
      return x[0] + x[1] + x[2] + x[3];
   }

   EvaluationResult eval_gradient(std::span<const Number>, std::span<Number> result)
   {
      std::ranges::fill(result, 1.);
      return {};
   }

   EvaluationResult eval_constraints(std::span<const Number> x, std::span<Number> result)
   {
      std::ranges::copy(x, result.begin());
      return {};
   }

   EvaluationResult eval_jacobian_structure(std::span<Index> rows, std::span<Index> columns)
   {
      for( Index i = 0; i < 4; ++i )
      {
         rows[i] = i;
         columns[i] = i;
      }
      return {};
   }

   EvaluationResult eval_jacobian_values(std::span<const Number> x, std::span<Number> values)
   {
      for( Index i = 0; i < 4; ++i )
      {
         values[i] = x[i] + static_cast<Number>(i + 1);
      }
      return {};
   }

   EvaluationResult eval_hessian_structure(std::span<Index> rows, std::span<Index> columns)
   {
      for( Index i = 0; i < 4; ++i )
      {
         rows[i] = i;
         columns[i] = i;
      }
      return {};
   }

   EvaluationResult eval_hessian_values(
      std::span<const Number>,
      Number                  objective_factor,
      std::span<const Number>,
      std::span<Number>       values
   )
   {
      std::ranges::fill(values, objective_factor);
      return {};
   }

   EvaluationResult eval_jacobian_product(
      std::span<const Number> x,
      std::span<const Number> direction,
      std::span<Number>       result
   )
   {
      for( Index i = 0; i < 4; ++i )
      {
         result[i] = (x[i] + static_cast<Number>(i + 1)) * direction[i];
      }
      return {};
   }

   EvaluationResult eval_jacobian_transpose_product(
      std::span<const Number> x,
      std::span<const Number> direction,
      std::span<Number>       result
   )
   {
      return eval_jacobian_product(x, direction, result);
   }

   EvaluationResult eval_hessian_product(
      std::span<const Number>,
      Number                  objective_factor,
      std::span<const Number>,
      std::span<const Number> direction,
      std::span<Number>       result
   )
   {
      for( Index i = 0; i < 4; ++i )
      {
         result[i] = objective_factor * direction[i];
      }
      return {};
   }
};

template <class Product>
double Run(Product&& product, std::size_t iterations, std::size_t warmup)
{
   const std::array<Number, 4> x{1., 2., 3., 4.};
   std::array<Number, 4> direction{0.5, 1.5, 2.5, 3.5};
   std::array<Number, 4> result{};
   const auto invoke = [&]()
   {
      direction[0] += 1e-12;
      if( EvaluationResult evaluated = product(x, direction, result); !evaluated )
      {
         throw std::runtime_error(evaluated.error().message);
      }
      DoNotOptimize(result.data());
   };

   for( std::size_t i = 0; i < warmup; ++i )
   {
      invoke();
   }
   const auto start = std::chrono::steady_clock::now();
   for( std::size_t i = 0; i < iterations; ++i )
   {
      invoke();
   }
   const auto stop = std::chrono::steady_clock::now();

   const Number checksum = result[0] + result[1] + result[2] + result[3];
   if( !std::isfinite(checksum) || checksum <= 0. )
   {
      throw std::runtime_error("invalid benchmark checksum");
   }
   return std::chrono::duration<double, std::nano>(stop - start).count() /
      static_cast<double>(iterations);
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
      std::size_t iterations = 20000000;
      std::size_t repetitions = 7;
      std::size_t warmup = 1000;
      for( int i = 1; i < argc; ++i )
      {
         const std::string_view argument(argv[i]);
         if( argument == "--help" )
         {
            std::cout << "Usage: " << argv[0]
                      << " [--iterations N] [--repetitions N] [--warmup N]\n";
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

      NlpModelAdapter direct(NativeFourByFourModel{});
      AnyNlpProblem erased = MakeNlpProblem(NativeFourByFourModel{});
      const auto direct_product = [&](std::span<const Number> x,
                                      std::span<const Number> direction,
                                      std::span<Number>       result)
      {
         return direct.nlp_jacobian_product(x, direction, result);
      };
      const auto erased_product = [&](std::span<const Number> x,
                                      std::span<const Number> direction,
                                      std::span<Number>       result)
      {
         return erased.nlp_jacobian_product(x, direction, result);
      };

      std::vector<double> direct_samples;
      std::vector<double> erased_samples;
      direct_samples.reserve(repetitions);
      erased_samples.reserve(repetitions);
      for( std::size_t repetition = 0; repetition < repetitions; ++repetition )
      {
         if( repetition % 2 == 0 )
         {
            direct_samples.push_back(Run(direct_product, iterations, warmup));
            erased_samples.push_back(Run(erased_product, iterations, warmup));
         }
         else
         {
            erased_samples.push_back(Run(erased_product, iterations, warmup));
            direct_samples.push_back(Run(direct_product, iterations, warmup));
         }
      }

      const double direct_median = Median(direct_samples);
      const double erased_median = Median(erased_samples);
      std::cout << std::setprecision(17)
                << "{\"schema_version\":1"
                << ",\"name\":\"cxx23_anyany_native_jacobian_product\""
                << ",\"metric\":\"nanoseconds_per_call\""
                << ",\"iterations\":" << iterations
                << ",\"warmup\":" << warmup
                << ",\"repetitions\":" << repetitions
                << ",\"direct_median\":" << direct_median
                << ",\"anyany_median\":" << erased_median
                << ",\"overhead_ns\":" << erased_median - direct_median
                << ",\"ratio\":" << erased_median / direct_median
                << ",\"direct_samples\":";
      PrintSamples(direct_samples);
      std::cout << ",\"anyany_samples\":";
      PrintSamples(erased_samples);
      std::cout << "}\n";
   }
   catch( const std::exception& error )
   {
      std::cerr << "ipopt_cxx23_anyany_jacobian_product_benchmark: "
                << error.what() << '\n';
      return EXIT_FAILURE;
   }
   return EXIT_SUCCESS;
}
