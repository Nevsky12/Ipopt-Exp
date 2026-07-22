// Copyright (C) 2026 Ipopt contributors.
// All Rights Reserved.
// This file is distributed under the Eclipse Public License.

#include <ipopt/cxx23/block_tridiagonal_solver.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <span>
#include <string_view>
#include <vector>

namespace
{
using namespace Ipopt::Cxx23;
using Clock = std::chrono::steady_clock;

struct Matrix
{
   std::vector<Index> sizes;
   std::vector<Number> diagonal;
   std::vector<Number> lower;
};

std::uint64_t Next(std::uint64_t& state)
{
   state ^= state << 13U;
   state ^= state >> 7U;
   state ^= state << 17U;
   return state;
}

Number RandomCoupling(std::uint64_t& state)
{
   return 0.025 *
      (static_cast<Number>(Next(state) % 2001U) / 1000. - 1.);
}

Matrix MakeMatrix(Index stages, Index block_size)
{
   Matrix matrix;
   matrix.sizes.assign(stages, block_size);
   matrix.diagonal.reserve(stages * block_size * block_size);
   matrix.lower.reserve((stages - 1) * block_size * block_size);
   std::uint64_t random = 0x243f6a8885a308d3ULL;

   std::vector<Number> previous_schur(block_size * block_size, 0.);
   for( Index diagonal = 0; diagonal < block_size; ++diagonal )
   {
      previous_schur[diagonal * block_size + diagonal] =
         diagonal % 5U == 0U ? -3. : 3. + 0.1 * static_cast<Number>(diagonal);
   }
   matrix.diagonal.insert(
      matrix.diagonal.end(), previous_schur.begin(), previous_schur.end());

   for( Index stage = 1; stage < stages; ++stage )
   {
      std::vector<Number> schur(block_size * block_size, 0.);
      for( Index diagonal = 0; diagonal < block_size; ++diagonal )
      {
         schur[diagonal * block_size + diagonal] =
            (stage + diagonal) % 5U == 0U
            ? -3. - 0.01 * static_cast<Number>(stage)
            : 3. + 0.1 * static_cast<Number>(diagonal);
      }
      std::vector<Number> multiplier(block_size * block_size);
      for( Number& value : multiplier )
      {
         value = RandomCoupling(random);
      }
      std::vector<Number> lower(block_size * block_size, 0.);
      for( Index row = 0; row < block_size; ++row )
      {
         for( Index column = 0; column < block_size; ++column )
         {
            for( Index inner = 0; inner < block_size; ++inner )
            {
               lower[row * block_size + column] +=
                  multiplier[row * block_size + inner] *
                  previous_schur[inner * block_size + column];
            }
         }
      }
      std::vector<Number> diagonal = schur;
      for( Index row = 0; row < block_size; ++row )
      {
         for( Index column = 0; column < block_size; ++column )
         {
            for( Index inner = 0; inner < block_size; ++inner )
            {
               diagonal[row * block_size + column] +=
                  multiplier[row * block_size + inner] *
                  lower[column * block_size + inner];
            }
         }
      }
      matrix.lower.insert(matrix.lower.end(), lower.begin(), lower.end());
      matrix.diagonal.insert(matrix.diagonal.end(), diagonal.begin(), diagonal.end());
      previous_schur = std::move(schur);
   }
   return matrix;
}

Matrix MakePivotedMatrix(Index stages, Index block_size)
{
   Matrix matrix;
   matrix.sizes.assign(stages, block_size);
   matrix.diagonal.reserve(stages * block_size * block_size);
   matrix.lower.reserve((stages - 1) * block_size * block_size);
   std::uint64_t random = 0x9e3779b97f4a7c15ULL;

   std::vector<std::vector<Number>> schur(stages);
   std::vector<std::vector<Number>> multipliers(stages - 1);
   for( Index stage = 0; stage < stages; ++stage )
   {
      std::vector<Number>& block = schur[stage];
      block.resize(block_size * block_size);
      for( Index row = 0; row < block_size; ++row )
      {
         for( Index column = 0; column < block_size; ++column )
         {
            block[row * block_size + column] =
               0.0025 / static_cast<Number>(1 + stage + row + column);
         }
         const Index anti_diagonal = block_size - row - 1;
         block[row * block_size + anti_diagonal] +=
            1. + 0.0625 * static_cast<Number>(
               std::min(row, anti_diagonal));
      }
      if( stage > 0 )
      {
         std::vector<Number>& multiplier = multipliers[stage - 1];
         multiplier.resize(block_size * block_size);
         for( Number& value : multiplier )
         {
            value = RandomCoupling(random);
         }
      }
   }

   for( Index stage = 0; stage < stages; ++stage )
   {
      std::vector<Number> diagonal = schur[stage];
      if( stage > 0 )
      {
         const std::vector<Number>& multiplier = multipliers[stage - 1];
         const std::vector<Number>& previous_schur = schur[stage - 1];
         std::vector<Number> lower(block_size * block_size, 0.);
         for( Index row = 0; row < block_size; ++row )
         {
            for( Index column = 0; column < block_size; ++column )
            {
               for( Index inner = 0; inner < block_size; ++inner )
               {
                  lower[row * block_size + column] +=
                     multiplier[row * block_size + inner] *
                     previous_schur[inner * block_size + column];
               }
            }
         }
         for( Index row = 0; row < block_size; ++row )
         {
            for( Index column = 0; column <= row; ++column )
            {
               Number update = 0.;
               for( Index inner = 0; inner < block_size; ++inner )
               {
                  update += multiplier[row * block_size + inner] *
                     lower[column * block_size + inner];
               }
               diagonal[row * block_size + column] += update;
               diagonal[column * block_size + row] =
                  diagonal[row * block_size + column];
            }
         }
         matrix.lower.insert(
            matrix.lower.end(), lower.begin(), lower.end());
      }
      matrix.diagonal.insert(
         matrix.diagonal.end(), diagonal.begin(), diagonal.end());
   }
   return matrix;
}

template <class Operation>
double MeasureMicroseconds(Index iterations, Operation&& operation)
{
   const auto begin = Clock::now();
   for( Index iteration = 0; iteration < iterations; ++iteration )
   {
      operation();
   }
   const auto end = Clock::now();
   return std::chrono::duration<double, std::micro>(end - begin).count() /
      static_cast<double>(iterations);
}

[[noreturn]] void Fail(std::string_view message)
{
   std::cerr << message << '\n';
   std::exit(1);
}

void RunCase(
   Index stages,
   Index block_size,
   Index rhs_count,
   bool certified_inertia,
   bool pivoted_numeric = false
)
{
   const Matrix matrix = pivoted_numeric
      ? MakePivotedMatrix(stages, block_size)
      : MakeMatrix(stages, block_size);
   BlockTridiagonalFactorizationOptions options;
   options.require_certified_inertia = certified_inertia;
   SymmetricBlockTridiagonalSolver solver(matrix.sizes, rhs_count, options);
   if( !solver.factorize(matrix.diagonal, matrix.lower) )
   {
      Fail("initial structured factorization failed");
   }
   const Index dimension = solver.storage().dimension;
   std::vector<Number> fused_rhs(dimension * rhs_count);
   std::vector<Number> fused_solution(dimension * rhs_count);
   std::vector<std::vector<Number>> sequential_rhs(
      rhs_count, std::vector<Number>(dimension));
   std::vector<std::vector<Number>> sequential_solution(
      rhs_count, std::vector<Number>(dimension));
   std::vector<Number> apply_input(dimension);
   std::vector<Number> apply_output(dimension);
   for( Index row = 0; row < dimension; ++row )
   {
      apply_input[row] = std::cos(
         0.013 * static_cast<Number>(row + 1));
      for( Index rhs = 0; rhs < rhs_count; ++rhs )
      {
         const Number value = std::sin(
            0.01 * static_cast<Number>((row + 1) * (rhs + 2)));
         fused_rhs[row * rhs_count + rhs] = value;
         sequential_rhs[rhs][row] = value;
      }
   }

   for( Index rhs = 0; rhs < rhs_count; ++rhs )
   {
      if( !solver.solve_rhs(sequential_rhs[rhs], sequential_solution[rhs]) )
      {
         Fail("sequential RHS warmup failed");
      }
   }
   if( !solver.solve_many(fused_rhs, rhs_count, fused_solution) )
   {
      Fail("fused RHS warmup failed");
   }
   if( !solver.apply(apply_input, apply_output) )
   {
      Fail("matrix-application warmup failed");
   }

   const Index factor_iterations = std::max<Index>(16, 4096 / stages);
   const Index solve_iterations = std::max<Index>(128, 32768 / stages);
   const Index apply_iterations = std::max<Index>(64, 16384 / stages);
   const double factor_us = MeasureMicroseconds(factor_iterations, [&] {
      if( !solver.factorize(matrix.diagonal, matrix.lower) )
      {
         Fail("timed structured factorization failed");
      }
   });
   const double sequential_us = MeasureMicroseconds(solve_iterations, [&] {
      for( Index rhs = 0; rhs < rhs_count; ++rhs )
      {
         if( !solver.solve_rhs(sequential_rhs[rhs], sequential_solution[rhs]) )
         {
            Fail("timed sequential RHS solve failed");
         }
      }
   });
   const double fused_us = MeasureMicroseconds(solve_iterations, [&] {
      if( !solver.solve_many(fused_rhs, rhs_count, fused_solution) )
      {
         Fail("timed fused RHS solve failed");
      }
   });
   const double apply_us = MeasureMicroseconds(apply_iterations, [&] {
      if( !solver.apply(apply_input, apply_output) )
      {
         Fail("timed matrix application failed");
      }
   });

   Number maximum_difference = 0.;
   Number checksum = 0.;
   Number apply_checksum = 0.;
   for( Index row = 0; row < dimension; ++row )
   {
      apply_checksum += apply_output[row] * static_cast<Number>(row + 1);
      for( Index rhs = 0; rhs < rhs_count; ++rhs )
      {
         const Number fused = fused_solution[row * rhs_count + rhs];
         maximum_difference = std::max(
            maximum_difference,
            std::abs(fused - sequential_solution[rhs][row]));
         checksum += fused * static_cast<Number>(1 + row + rhs);
      }
   }
   if( maximum_difference > 1e-11 || !std::isfinite(checksum) ||
       !std::isfinite(apply_checksum) )
   {
      Fail("fused and sequential structured solves disagree");
   }

   std::cout << (pivoted_numeric
                    ? "numeric_pivoted"
                    : (certified_inertia ? "certified" : "numeric")) << ','
             << stages << ',' << block_size << ',' << dimension << ','
             << rhs_count << ',' << factor_us << ',' << sequential_us << ','
             << fused_us << ',' << sequential_us / fused_us << ','
             << checksum << ',' << apply_us << ',' << apply_checksum << '\n';
}
} // namespace

int main()
{
   std::cout << std::setprecision(10);
   std::cout << "mode,stages,block_size,dimension,rhs_count,factor_us,"
                "sequential_rhs_us,fused_rhs_us,fused_speedup,checksum,"
                "apply_us,apply_checksum\n";
   for( Index block_size : {5U, 8U, 12U, 14U} )
   {
      for( Index stages : {16U, 32U, 64U, 128U, 256U} )
      {
         RunCase(stages, block_size, 8, true);
         RunCase(stages, block_size, 8, false);
      }
   }
   for( Index block_size : {12U, 14U} )
   {
      for( Index stages : {16U, 32U, 64U, 128U, 256U} )
      {
         RunCase(stages, block_size, 8, false, true);
      }
   }
   return 0;
}
