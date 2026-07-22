// Copyright (C) 2026 Ipopt contributors.
// All Rights Reserved.
// This file is distributed under the Eclipse Public License.

#include <ipopt/cxx23/block_tridiagonal_solver.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace
{
using namespace Ipopt::Cxx23;

void Check(bool condition, std::string_view message)
{
   if( !condition )
   {
      throw std::runtime_error(std::string(message));
   }
}

void CheckNear(Number actual, Number expected, Number tolerance, std::string_view message)
{
   if( std::abs(actual - expected) > tolerance )
   {
      throw std::runtime_error(
         std::string(message) + ": actual=" + std::to_string(actual) +
         " expected=" + std::to_string(expected));
   }
}

struct GeneratedSystem
{
   std::vector<Index> sizes;
   std::vector<Number> diagonal;
   std::vector<Number> lower;
   Index positive = 0;
   Index negative = 0;
};

std::uint64_t NextRandom(std::uint64_t& state)
{
   state ^= state << 13U;
   state ^= state >> 7U;
   state ^= state << 17U;
   return state;
}

Number RandomSmall(std::uint64_t& state)
{
   const Number unit = static_cast<Number>(NextRandom(state) % 2001U) / 1000.;
   return 0.12 * (unit - 1.);
}

/** Build A = L diag(S_k) L^T with arbitrary rectangular first subdiagonal. */
GeneratedSystem MakeSystem(std::vector<Index> sizes, std::uint64_t seed)
{
   GeneratedSystem result;
   result.sizes = std::move(sizes);
   std::vector<std::vector<Number>> schur;
   std::vector<std::vector<Number>> multipliers;
   schur.reserve(result.sizes.size());
   multipliers.reserve(result.sizes.size() - 1);

   for( Index stage = 0; stage < result.sizes.size(); ++stage )
   {
      const Index size = result.sizes[stage];
      std::vector<Number> block(size * size, 0.);
      for( Index diagonal = 0; diagonal < size; ++diagonal )
      {
         const bool negative = ((stage + diagonal) % 4U) == 1U;
         block[diagonal * size + diagonal] =
            (negative ? -1. : 1.) * (2. + 0.3 * static_cast<Number>(diagonal + stage));
         if( negative )
         {
            ++result.negative;
         }
         else
         {
            ++result.positive;
         }
      }
      schur.push_back(std::move(block));
      if( stage > 0 )
      {
         const Index previous = result.sizes[stage - 1];
         std::vector<Number> multiplier(size * previous);
         for( Number& value : multiplier )
         {
            value = RandomSmall(seed);
         }
         multipliers.push_back(std::move(multiplier));
      }
   }

   for( Index stage = 0; stage < result.sizes.size(); ++stage )
   {
      const Index size = result.sizes[stage];
      std::vector<Number> block = schur[stage];
      if( stage > 0 )
      {
         const Index previous = result.sizes[stage - 1];
         const std::vector<Number>& multiplier = multipliers[stage - 1];
         const std::vector<Number>& previous_schur = schur[stage - 1];
         std::vector<Number> lower(size * previous, 0.);
         for( Index row = 0; row < size; ++row )
         {
            for( Index column = 0; column < previous; ++column )
            {
               for( Index inner = 0; inner < previous; ++inner )
               {
                  lower[row * previous + column] +=
                     multiplier[row * previous + inner] *
                     previous_schur[inner * previous + column];
               }
            }
         }
         for( Index row = 0; row < size; ++row )
         {
            for( Index column = 0; column < size; ++column )
            {
               for( Index inner = 0; inner < previous; ++inner )
               {
                  block[row * size + column] +=
                     multiplier[row * previous + inner] *
                     lower[column * previous + inner];
               }
            }
         }
         result.lower.insert(result.lower.end(), lower.begin(), lower.end());
      }
      result.diagonal.insert(result.diagonal.end(), block.begin(), block.end());
   }
   return result;
}

/** Build a numeric-path system whose Schur blocks require repeated row pivots. */
GeneratedSystem MakePivotedNumericSystem(
   std::vector<Index> sizes,
   std::uint64_t      seed
)
{
   GeneratedSystem result;
   result.sizes = std::move(sizes);
   std::vector<std::vector<Number>> schur;
   std::vector<std::vector<Number>> multipliers;
   schur.reserve(result.sizes.size());
   multipliers.reserve(result.sizes.size() - 1);

   for( Index stage = 0; stage < result.sizes.size(); ++stage )
   {
      const Index size = result.sizes[stage];
      std::vector<Number> block(size * size);
      for( Index row = 0; row < size; ++row )
      {
         for( Index column = 0; column < size; ++column )
         {
            block[row * size + column] =
               0.0025 / static_cast<Number>(1 + stage + row + column);
         }
         const Index anti_diagonal = size - row - 1;
         block[row * size + anti_diagonal] +=
            1. + 0.0625 * static_cast<Number>(
               std::min(row, anti_diagonal));
      }
      schur.push_back(std::move(block));
      if( stage > 0 )
      {
         const Index previous = result.sizes[stage - 1];
         std::vector<Number> multiplier(size * previous);
         for( Number& value : multiplier )
         {
            value = RandomSmall(seed);
         }
         multipliers.push_back(std::move(multiplier));
      }
   }

   for( Index stage = 0; stage < result.sizes.size(); ++stage )
   {
      const Index size = result.sizes[stage];
      std::vector<Number> block = schur[stage];
      if( stage > 0 )
      {
         const Index previous = result.sizes[stage - 1];
         const std::vector<Number>& multiplier = multipliers[stage - 1];
         const std::vector<Number>& previous_schur = schur[stage - 1];
         std::vector<Number> lower(size * previous, 0.);
         for( Index row = 0; row < size; ++row )
         {
            for( Index column = 0; column < previous; ++column )
            {
               for( Index inner = 0; inner < previous; ++inner )
               {
                  lower[row * previous + column] +=
                     multiplier[row * previous + inner] *
                     previous_schur[inner * previous + column];
               }
            }
         }
         for( Index row = 0; row < size; ++row )
         {
            for( Index column = 0; column <= row; ++column )
            {
               Number update = 0.;
               for( Index inner = 0; inner < previous; ++inner )
               {
                  update += multiplier[row * previous + inner] *
                     lower[column * previous + inner];
               }
               block[row * size + column] += update;
               block[column * size + row] = block[row * size + column];
            }
         }
         result.lower.insert(result.lower.end(), lower.begin(), lower.end());
      }
      result.diagonal.insert(result.diagonal.end(), block.begin(), block.end());
   }
   return result;
}

std::vector<Index> BlockOffsets(std::span<const Index> sizes)
{
   std::vector<Index> offsets{0};
   for( Index size : sizes )
   {
      offsets.push_back(offsets.back() + size);
   }
   return offsets;
}

std::vector<Number> Apply(
   const GeneratedSystem& system,
   std::span<const Number> input
)
{
   const std::vector<Index> offsets = BlockOffsets(system.sizes);
   std::vector<Number> result(input.size(), 0.);
   Index diagonal_offset = 0;
   Index lower_offset = 0;
   for( Index stage = 0; stage < system.sizes.size(); ++stage )
   {
      const Index size = system.sizes[stage];
      for( Index row = 0; row < size; ++row )
      {
         for( Index column = 0; column < size; ++column )
         {
            result[offsets[stage] + row] +=
               system.diagonal[diagonal_offset + row * size + column] *
               input[offsets[stage] + column];
         }
      }
      diagonal_offset += size * size;
      if( stage > 0 )
      {
         const Index previous = system.sizes[stage - 1];
         for( Index row = 0; row < size; ++row )
         {
            for( Index column = 0; column < previous; ++column )
            {
               const Number value = system.lower[lower_offset + row * previous + column];
               result[offsets[stage] + row] += value * input[offsets[stage - 1] + column];
               result[offsets[stage - 1] + column] += value * input[offsets[stage] + row];
            }
         }
         lower_offset += size * previous;
      }
   }
   return result;
}

void CheckSolution(
   SymmetricBlockTridiagonalSolver& solver,
   const GeneratedSystem&          system,
   Index                           rhs_count
)
{
   const Index dimension = solver.storage().dimension;
   std::vector<Number> expected(dimension * rhs_count);
   std::vector<Number> rhs(dimension * rhs_count);
   for( Index right_hand_side = 0; right_hand_side < rhs_count; ++right_hand_side )
   {
      std::vector<Number> one_solution(dimension);
      for( Index row = 0; row < dimension; ++row )
      {
         one_solution[row] =
            0.15 * static_cast<Number>(row + 1) -
            0.08 * static_cast<Number>(right_hand_side + 1);
         expected[row * rhs_count + right_hand_side] = one_solution[row];
      }
      const std::vector<Number> one_rhs = Apply(system, one_solution);
      for( Index row = 0; row < dimension; ++row )
      {
         rhs[row * rhs_count + right_hand_side] = one_rhs[row];
      }
   }

   std::vector<Number> solution(dimension * rhs_count, -999.);
   EvaluationResult solved = rhs_count == 1
      ? solver.solve_rhs(rhs, solution)
      : solver.solve_many(rhs, rhs_count, solution);
   Check(solved.has_value(), "valid block-tridiagonal solve failed");
   for( Index entry = 0; entry < solution.size(); ++entry )
   {
      CheckNear(solution[entry], expected[entry], 2e-11, "wrong structured solution");
   }
}

void TestVariableBlocksInertiaAndSolveMany()
{
   const GeneratedSystem system = MakeSystem({3, 1, 4, 2}, 0x123456789abcdef0ULL);
   SymmetricBlockTridiagonalSolver solver(system.sizes, 4);
   Check(solver.configured(), "valid block topology was rejected");
   Check(solver.storage().dimension == 10, "wrong total block dimension");
   Check(
      solver.storage().diagonal_values == system.diagonal.size() &&
         solver.storage().lower_values == system.lower.size(),
      "wrong packed storage counts");
   const BlockTridiagonalWorkspaceProfile prepared_workspace =
      solver.workspace_profile();

   EvaluationValue<BlockTridiagonalFactorizationReport> factorized =
      solver.factorize(system.diagonal, system.lower);
   Check(factorized.has_value(), "valid block matrix factorization failed");
   Check(factorized->inertia.exact, "successful inertia was not certified");
   Check(
      factorized->inertia.positive_eigenvalues == system.positive &&
         factorized->inertia.negative_eigenvalues == system.negative &&
         factorized->inertia.zero_eigenvalues == 0,
      "wrong block-LDL inertia");
   Check(
      factorized->inertia.minimum_separation > 0.,
      "inertia certificate has no separation from zero");

   CheckSolution(solver, system, 1);
   CheckSolution(solver, system, 4);
   Check(
      solver.statistics().factorization_attempts == 1 &&
         solver.statistics().successful_factorizations == 1,
      "solve-many refactorized the matrix");
   Check(
      solver.statistics().solve_calls == 2 &&
         solver.statistics().solved_right_hand_sides == 5,
      "solve-many work accounting is wrong");
   Check(
      solver.workspace_profile() == prepared_workspace,
      "factor/solve-many resized constructor-owned workspace");

   std::vector<Number> vector(solver.storage().dimension);
   for( Index row = 0; row < vector.size(); ++row )
   {
      vector[row] = 0.2 * static_cast<Number>(row + 1);
   }
   const std::vector<Number> expected = Apply(system, vector);
   EvaluationResult applied = solver.apply(vector, vector);
   Check(applied.has_value(), "alias-safe block apply failed");
   for( Index row = 0; row < vector.size(); ++row )
   {
      CheckNear(vector[row], expected[row], 1e-12, "wrong block matrix product");
   }
}

void TestGeneratedTopologies()
{
   const std::vector<std::vector<Index>> topologies{
      {1}, {1, 2, 1}, {2, 3, 2, 4}, {4, 1, 3, 2, 1}
   };
   std::uint64_t seed = 0xfedcba9876543210ULL;
   for( const std::vector<Index>& topology : topologies )
   {
      const GeneratedSystem system = MakeSystem(topology, NextRandom(seed));
      SymmetricBlockTridiagonalSolver solver(topology, 3);
      const auto factorized = solver.factorize(system.diagonal, system.lower);
      Check(factorized.has_value(), "generated topology failed factorization");
      Check(
         factorized->inertia.positive_eigenvalues == system.positive &&
            factorized->inertia.negative_eigenvalues == system.negative,
         "generated topology has wrong inertia");
      CheckSolution(solver, system, 3);
   }
}

void TestUncertifiedNumericFactor()
{
   const GeneratedSystem system = MakeSystem({5, 4, 5, 3}, 0x55aa55aaULL);
   BlockTridiagonalFactorizationOptions options;
   options.require_certified_inertia = false;
   SymmetricBlockTridiagonalSolver solver(system.sizes, 3, options);
   const BlockTridiagonalWorkspaceProfile prepared_workspace =
      solver.workspace_profile();
   const auto factorized = solver.factorize(system.diagonal, system.lower);
   Check(factorized.has_value(), "valid numeric block factorization failed");
   Check(
      !factorized->inertia.exact &&
         factorized->inertia.zero_eigenvalues == 0 &&
         factorized->inertia.minimum_separation == 0. &&
         std::isinf(factorized->inertia.certificate_radius),
      "numeric block factor was incorrectly promoted to exact inertia");
   CheckSolution(solver, system, 1);
   CheckSolution(solver, system, 3);
   Check(
      solver.workspace_profile() == prepared_workspace,
      "numeric factor/solve resized constructor-owned workspace");

   SymmetricBlockTridiagonalSolver singular({2}, 1, options);
   Check(
      !singular.factorize(std::vector<Number>{1., 1., 1., 1.}, {}).has_value(),
      "singular numeric Schur block was accepted");
   Check(!singular.factorized(), "failed numeric factor retained stale factors");

   SymmetricBlockTridiagonalSolver pivoted({4}, 1, options);
   const std::vector<Number> pivoted_diagonal{
      0., 2., 0., 0.,
      2., 0., 0., 0.,
      0., 0., 0., 3.,
      0., 0., 3., 0.
   };
   Check(
      pivoted.factorize(pivoted_diagonal, {}).has_value(),
      "numeric factor rejected a nonsingular block requiring row pivots");
   const std::vector<Number> pivoted_expected{1., -2., 3., -4.};
   std::vector<Number> pivoted_rhs(4);
   for( Index row = 0; row < 4; ++row )
   {
      for( Index column = 0; column < 4; ++column )
      {
         pivoted_rhs[row] +=
            pivoted_diagonal[row * 4 + column] * pivoted_expected[column];
      }
   }
   std::vector<Number> pivoted_solution(4);
   Check(
      pivoted.solve_rhs(pivoted_rhs, pivoted_solution).has_value(),
      "numeric solve failed after multiple row pivots");
   for( Index entry = 0; entry < 4; ++entry )
   {
      CheckNear(
         pivoted_solution[entry], pivoted_expected[entry], 1e-13,
         "wrong solution after numeric row pivots");
   }

   for( Index size : {12U, 14U} )
   {
      SymmetricBlockTridiagonalSolver fixed_pivoted({size}, 1, options);
      std::vector<Number> diagonal(size * size, 0.);
      std::vector<Number> diagonal_expected(size);
      std::vector<Number> diagonal_rhs(size);
      for( Index entry = 0; entry < size; ++entry )
      {
         const Number value = entry % 3U == 0U
            ? -2. - 0.125 * static_cast<Number>(entry)
            : 2. + 0.125 * static_cast<Number>(entry);
         diagonal[entry * size + entry] = value;
         diagonal_expected[entry] =
            0.125 * static_cast<Number>(entry + 1) - 0.5;
         diagonal_rhs[entry] = value * diagonal_expected[entry];
      }
      Check(
         fixed_pivoted.factorize(diagonal, {}).has_value(),
         "fixed-size numeric factor rejected a diagonal block");
      std::vector<Number> diagonal_solution(size);
      Check(
         fixed_pivoted.solve_rhs(diagonal_rhs, diagonal_solution).has_value(),
         "fixed-size numeric solve failed for a diagonal block");
      for( Index entry = 0; entry < size; ++entry )
      {
         CheckNear(
            diagonal_solution[entry], diagonal_expected[entry], 1e-13,
            "wrong fixed-size diagonal solution");
      }

      std::vector<Number> anti_diagonal(size * size, 0.);
      std::vector<Number> expected(size);
      std::vector<Number> rhs(size, 0.);
      for( Index row = 0; row < size; ++row )
      {
         for( Index column = 0; column < size; ++column )
         {
            anti_diagonal[row * size + column] =
               0.0025 / static_cast<Number>(1 + row + column);
         }
         const Index column = size - row - 1;
         const Number value = 1. + 0.125 * static_cast<Number>(
            std::min(row, column));
         anti_diagonal[row * size + column] += value;
         expected[row] = 0.25 * static_cast<Number>(row + 1) - 1.;
      }
      for( Index row = 0; row < size; ++row )
      {
         for( Index column = 0; column < size; ++column )
         {
            rhs[row] += anti_diagonal[row * size + column] * expected[column];
         }
      }
      Check(
         fixed_pivoted.factorize(anti_diagonal, {}).has_value(),
         "fixed-size numeric factor rejected repeated row pivots");
      std::vector<Number> solution(size);
      Check(
         fixed_pivoted.solve_rhs(rhs, solution).has_value(),
         "fixed-size numeric solve failed after repeated row pivots");
      for( Index entry = 0; entry < size; ++entry )
      {
         CheckNear(
            solution[entry], expected[entry], 1e-13,
            "wrong fixed-size solution after repeated row pivots");
      }
      Check(
         fixed_pivoted.factorize(diagonal, {}).has_value(),
         "fixed-size numeric factor retained stale pivoted inverse data");
      std::ranges::fill(diagonal_solution, 0.);
      Check(
         fixed_pivoted.solve_rhs(diagonal_rhs, diagonal_solution).has_value(),
         "fixed-size numeric solve failed after a pivot-pattern change");
      for( Index entry = 0; entry < size; ++entry )
      {
         CheckNear(
            diagonal_solution[entry], diagonal_expected[entry], 1e-13,
            "stale inverse data changed a fixed-size diagonal solution");
      }
   }
}

void TestPivotedNumericSchurUpdates()
{
   std::uint64_t seed = 0x9e3779b97f4a7c15ULL;
   for( const std::vector<Index>& sizes : {
           std::vector<Index>{12, 14, 12},
           std::vector<Index>{14, 14, 14}} )
   {
      const GeneratedSystem system = MakePivotedNumericSystem(sizes, seed);
      seed = NextRandom(seed);
      BlockTridiagonalFactorizationOptions options;
      options.require_certified_inertia = false;
      SymmetricBlockTridiagonalSolver solver(system.sizes, 3, options);
      const BlockTridiagonalWorkspaceProfile prepared_workspace =
         solver.workspace_profile();
      const auto factorized = solver.factorize(system.diagonal, system.lower);
      Check(
         factorized.has_value(),
         "pivoted multi-stage numeric factorization failed");
      CheckSolution(solver, system, 1);
      CheckSolution(solver, system, 3);
      Check(
         solver.workspace_profile() == prepared_workspace,
         "pivoted numeric factor/solve resized constructor-owned workspace");
   }
}

void TestFailuresAreTransactional()
{
   SymmetricBlockTridiagonalSolver invalid({}, 1);
   Check(!invalid.configured(), "empty block topology was accepted");
   Check(
      !invalid.factorize({}, {}).has_value(),
      "invalid topology factorization was accepted");

   SymmetricBlockTridiagonalSolver zero_stage({2, 0, 1}, 1);
   Check(!zero_stage.configured(), "zero-sized stage was accepted");
   SymmetricBlockTridiagonalSolver zero_rhs({1}, 0);
   Check(!zero_rhs.configured(), "zero RHS capacity was accepted");
   BlockTridiagonalFactorizationOptions unsafe_margin;
   unsafe_margin.relative_inertia_tolerance = 0.;
   SymmetricBlockTridiagonalSolver unsafe_certificate({1}, 1, unsafe_margin);
   Check(
      !unsafe_certificate.configured(),
      "inertia tolerance below the floating-point safety floor was accepted");

   const GeneratedSystem system = MakeSystem({2, 2}, 42);
   SymmetricBlockTridiagonalSolver solver(system.sizes, 2);
   std::vector<Number> output(4, 77.);
   Check(
      !solver.solve_rhs(std::vector<Number>(4, 1.), output).has_value(),
      "solve before factorization was accepted");
   Check(std::ranges::all_of(output, [](Number value) { return value == 77.; }),
      "failed solve modified output");

   std::vector<Number> asymmetric = system.diagonal;
   asymmetric[1] += 1e-4;
   const auto asymmetric_result = solver.factorize(asymmetric, system.lower);
   Check(!asymmetric_result.has_value(), "asymmetric diagonal block was accepted");
   Check(
      asymmetric_result.error().code == EvaluationErrorCode::invalid_layout,
      "asymmetry returned the wrong error");

   std::vector<Number> nonfinite_diagonal = system.diagonal;
   nonfinite_diagonal[0] = std::numeric_limits<Number>::quiet_NaN();
   const auto nonfinite_diagonal_result =
      solver.factorize(nonfinite_diagonal, system.lower);
   Check(
      !nonfinite_diagonal_result.has_value(),
      "nonfinite diagonal block was accepted");
   Check(
      nonfinite_diagonal_result.error().code ==
         EvaluationErrorCode::nonfinite_output,
      "nonfinite diagonal block returned the wrong error");

   std::vector<Number> nonfinite = system.lower;
   nonfinite[0] = std::numeric_limits<Number>::quiet_NaN();
   const auto nonfinite_result = solver.factorize(system.diagonal, nonfinite);
   Check(!nonfinite_result.has_value(), "nonfinite lower block was accepted");

   BlockTridiagonalFactorizationOptions numeric_options;
   numeric_options.require_certified_inertia = false;
   SymmetricBlockTridiagonalSolver maximum_finite({1}, 1, numeric_options);
   Check(
      maximum_finite.factorize(
         std::array<Number, 1>{std::numeric_limits<Number>::max()},
         {}).has_value(),
      "finite maximum diagonal overflowed during symmetric projection");
   std::array<Number, 1> maximum_solution{};
   Check(
      maximum_finite.solve_rhs(std::array<Number, 1>{1.}, maximum_solution)
         .has_value() &&
         std::isfinite(maximum_solution[0]) && maximum_solution[0] > 0.,
      "finite maximum diagonal did not produce a finite usable factor");

   BlockTridiagonalFactorizationOptions projected_options = numeric_options;
   projected_options.symmetry_tolerance = 1e-3;
   SymmetricBlockTridiagonalSolver projected({2}, 1, projected_options);
   Check(
      projected.factorize(
         std::array<Number, 4>{3., 1.0001, .9999, 2.}, {}).has_value(),
      "within-tolerance asymmetric block was not projected onto symmetry");
   std::array<Number, 2> projected_product{};
   Check(
      projected.apply(
         std::array<Number, 2>{.5, -.25}, projected_product).has_value(),
      "projected symmetric block application failed");
   CheckNear(projected_product[0], 1.25, 1e-14,
      "edge-once application did not use the projected off-diagonal");
   CheckNear(projected_product[1], 0., 1e-14,
      "edge-once application produced the wrong symmetric contribution");

   Check(
      solver.factorize(system.diagonal, system.lower).has_value(),
      "valid matrix was rejected after failed attempts");
   std::vector<Number> too_many_input(12, 1.);
   std::vector<Number> too_many_output(12, 88.);
   Check(
      !solver.solve_many(too_many_input, 3, too_many_output).has_value(),
      "RHS batch above capacity was accepted");
   Check(
      std::ranges::all_of(too_many_output, [](Number value) { return value == 88.; }),
      "rejected RHS batch modified output");

   std::vector<Number> singular_diagonal{0.};
   SymmetricBlockTridiagonalSolver singular({1});
   Check(
      !singular.factorize(singular_diagonal, {}).has_value(),
      "singular Schur block received an inertia certificate");

   BlockTridiagonalFactorizationOptions strict;
   strict.absolute_inertia_tolerance = 1e-8;
   SymmetricBlockTridiagonalSolver uncertified({1}, 1, strict);
   std::vector<Number> tiny_diagonal{1e-10};
   Check(
      !uncertified.factorize(tiny_diagonal, {}).has_value(),
      "eigenvalue interval containing zero was certified");

   std::vector<Number> bad_diagonal = system.diagonal;
   bad_diagonal[0] = 0.;
   bad_diagonal[1] = 0.;
   bad_diagonal[2] = 0.;
   bad_diagonal[3] = 0.;
   Check(
      !solver.factorize(bad_diagonal, system.lower).has_value(),
      "failed refactorization was unexpectedly accepted");
   Check(!solver.factorized(), "failed refactorization retained stale factors");
   Check(
      !solver.solve_rhs(std::vector<Number>(4, 1.), output).has_value(),
      "stale factors were used after a failed factorization");
}

void TestRefinementGate()
{
   const GeneratedSystem system = MakeSystem({3, 2, 3}, 20260722);
   SymmetricBlockTridiagonalSolver solver(system.sizes);
   Check(
      solver.factorize(system.diagonal, system.lower).has_value(),
      "refinement matrix factorization failed");
   const BlockTridiagonalWorkspaceProfile prepared_workspace =
      solver.workspace_profile();
   const Index dimension = solver.storage().dimension;
   std::vector<Number> expected(dimension);
   for( Index row = 0; row < dimension; ++row )
   {
      expected[row] = std::sin(static_cast<Number>(row + 1));
   }
   const std::vector<Number> rhs = Apply(system, expected);
   std::vector<Number> solution(dimension, -13.);
   const auto refined = solver.solve_refined_rhs(
      rhs, solution, BlockRefinementOptions{1e-12, 4});
   Check(refined.has_value(), "valid refinement returned an evaluation error");
   Check(
      refined->status == BlockRefinementStatus::converged,
      "refinement did not satisfy the residual gate");
   Check(
      refined->final_relative_residual <= 1e-12,
      "refinement accepted a residual above tolerance");
   for( Index row = 0; row < dimension; ++row )
   {
      CheckNear(solution[row], expected[row], 2e-11, "wrong refined solution");
   }

   std::ranges::fill(solution, -17.);
   const auto limited = solver.solve_refined_rhs(
      rhs,
      solution,
      BlockRefinementOptions{std::numeric_limits<Number>::denorm_min(), 0});
   Check(limited.has_value(), "refinement limit returned an evaluation error");
   Check(
      limited->status == BlockRefinementStatus::iteration_limit,
      "zero-step refinement did not report its limit");
   Check(
      std::ranges::all_of(solution, [](Number value) { return value == -17.; }),
      "nonconverged refinement committed a solution");

   const auto invalid = solver.solve_refined_rhs(
      rhs, solution, BlockRefinementOptions{-1., 1});
   Check(!invalid.has_value(), "negative refinement tolerance was accepted");
   Check(
      solver.workspace_profile() == prepared_workspace,
      "refinement resized constructor-owned workspace");
}
} // namespace

int main()
{
   try
   {
      TestVariableBlocksInertiaAndSolveMany();
      TestGeneratedTopologies();
      TestUncertifiedNumericFactor();
      TestPivotedNumericSchurUpdates();
      TestFailuresAreTransactional();
      TestRefinementGate();
   }
   catch( const std::exception& exception )
   {
      std::cerr << exception.what() << '\n';
      return 1;
   }
   return 0;
}
