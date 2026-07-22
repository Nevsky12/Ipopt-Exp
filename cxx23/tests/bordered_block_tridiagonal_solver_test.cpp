// Copyright (C) 2026 Ipopt contributors.
// All Rights Reserved.
// This file is distributed under the Eclipse Public License.

#include <ipopt/cxx23/bordered_block_tridiagonal_solver.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
using namespace Ipopt::Cxx23;

void Check(bool condition, const char* message)
{
   if( !condition )
   {
      throw std::runtime_error(message);
   }
}

void CheckNear(
   Number      actual,
   Number      expected,
   Number      tolerance,
   const char* message
)
{
   if( std::abs(actual - expected) > tolerance )
   {
      throw std::runtime_error(
         std::string(message) + ": actual=" + std::to_string(actual) +
         " expected=" + std::to_string(expected));
   }
}

struct ArrowheadSystem
{
   std::vector<Index> blocks{2, 1, 2};
   std::vector<Number> diagonal{
      4., .3, .3, 3.,
      2.5,
      3., .1, .1, 2.
   };
   std::vector<Number> lower{
      .2, -.1,
      .15, -.05
   };
   std::vector<Number> border{
      .4, -.2,
      .1, .3,
      -.25, .15,
      .2, .05,
      -.1, .35
   };
   std::vector<Number> border_diagonal{
      2.8, .2,
      .2, -2.4
   };
};

std::vector<Number> Apply(
   const ArrowheadSystem&    system,
   std::span<const Number>   input
)
{
   const Index stage_dimension = 5;
   const Index border_dimension = 2;
   std::vector<Number> result(stage_dimension + border_dimension, 0.);
   const std::array<Index, 4> offsets{{0, 2, 3, 5}};
   Index diagonal_offset = 0;
   Index lower_offset = 0;
   for( Index stage = 0; stage < system.blocks.size(); ++stage )
   {
      const Index size = system.blocks[stage];
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
         const Index previous = system.blocks[stage - 1];
         for( Index row = 0; row < size; ++row )
         {
            for( Index column = 0; column < previous; ++column )
            {
               const Number value =
                  system.lower[lower_offset + row * previous + column];
               result[offsets[stage] + row] +=
                  value * input[offsets[stage - 1] + column];
               result[offsets[stage - 1] + column] +=
                  value * input[offsets[stage] + row];
            }
         }
         lower_offset += size * previous;
      }
   }
   for( Index stage = 0; stage < stage_dimension; ++stage )
   {
      for( Index border = 0; border < border_dimension; ++border )
      {
         const Number value =
            system.border[stage * border_dimension + border];
         result[stage] += value * input[stage_dimension + border];
         result[stage_dimension + border] += value * input[stage];
      }
   }
   for( Index row = 0; row < border_dimension; ++row )
   {
      for( Index column = 0; column < border_dimension; ++column )
      {
         result[stage_dimension + row] +=
            system.border_diagonal[row * border_dimension + column] *
            input[stage_dimension + column];
      }
   }
   return result;
}

void TestSolveApplyInertiaAndWorkspace()
{
   const ArrowheadSystem system;
   SymmetricBorderedBlockTridiagonalSolver solver(system.blocks, 2);
   Check(solver.configured(), "valid bordered topology was rejected");
   const BorderedBlockTridiagonalStorage storage = solver.storage();
   Check(
      storage.stage_dimension == 5 && storage.border_dimension == 2 &&
         storage.dimension == 7 && storage.diagonal_values == 9 &&
         storage.lower_values == 4 && storage.border_values == 10 &&
         storage.border_diagonal_values == 4,
      "wrong bordered packed-storage profile");
   const BorderedBlockTridiagonalWorkspaceProfile workspace =
      solver.workspace_profile();

   const auto factorized = solver.factorize(
      system.diagonal,
      system.lower,
      system.border,
      system.border_diagonal);
   Check(factorized.has_value(), "valid arrowhead factorization failed");
   Check(
      factorized->stage.inertia.exact && factorized->schur.inertia.exact,
      "component inertia certificates are missing");
   Check(
      !factorized->numeric_inertia.exact &&
         factorized->numeric_inertia.positive_eigenvalues == 6 &&
         factorized->numeric_inertia.negative_eigenvalues == 1 &&
         factorized->numeric_inertia.zero_eigenvalues == 0,
      "computed arrowhead signs were incorrectly certified or counted");
   Check(
      factorized->relative_schur_asymmetry < 1e-14,
      "well-conditioned Schur complement lost symmetry");
   Check(
      solver.statistics().fused_border_right_hand_sides == 2,
      "border columns were not solved as one fused batch");

   std::vector<Number> expected{.2, -.35, .4, -.1, .55, .3, -.25};
   const std::vector<Number> rhs = Apply(system, expected);
   std::vector<Number> solution(expected.size(), -99.);
   Check(solver.solve_rhs(rhs, solution).has_value(), "arrowhead solve failed");
   for( Index row = 0; row < solution.size(); ++row )
   {
      CheckNear(solution[row], expected[row], 2e-12, "wrong arrowhead solution");
   }

   std::vector<Number> alias = expected;
   const std::vector<Number> applied = Apply(system, alias);
   Check(solver.apply(alias, alias).has_value(), "alias-safe arrowhead apply failed");
   for( Index row = 0; row < alias.size(); ++row )
   {
      CheckNear(alias[row], applied[row], 2e-14, "wrong arrowhead product");
   }

   std::ranges::fill(solution, -77.);
   const auto refined = solver.solve_refined_rhs(
      rhs, solution, BlockRefinementOptions{1e-13, 4});
   Check(refined.has_value(), "arrowhead refinement failed");
   Check(
      refined->status == BlockRefinementStatus::converged &&
         refined->final_relative_residual <= 1e-13,
      "arrowhead true-residual gate did not converge");
   for( Index row = 0; row < solution.size(); ++row )
   {
      CheckNear(solution[row], expected[row], 2e-12, "wrong refined solution");
   }
   Check(
      solver.statistics().matrix_applications == 2,
      "public apply and internal refinement application were miscounted");
   Check(
      solver.workspace_profile() == workspace,
      "bordered factor/solve/refinement resized constructor-owned workspace");
}

void TestFailuresAreTransactional()
{
   SymmetricBorderedBlockTridiagonalSolver no_border({1}, 0);
   Check(!no_border.configured(), "zero-sized border was accepted");

   BorderedBlockTridiagonalOptions invalid_options;
   invalid_options.relative_schur_symmetry_tolerance = 0.;
   SymmetricBorderedBlockTridiagonalSolver invalid_tolerance(
      {1}, 1, invalid_options);
   Check(
      !invalid_tolerance.configured(),
      "unsafe Schur symmetry tolerance was accepted");

   const ArrowheadSystem system;
   SymmetricBorderedBlockTridiagonalSolver solver(system.blocks, 2);
   std::vector<Number> output(7, 44.);
   Check(
      !solver.solve_rhs(std::vector<Number>(7, 1.), output).has_value(),
      "bordered solve before factorization was accepted");
   Check(
      std::ranges::all_of(output, [](Number value) { return value == 44.; }),
      "failed bordered solve modified output");

   std::vector<Number> asymmetric = system.border_diagonal;
   asymmetric[1] += 1e-3;
   const auto asymmetric_result = solver.factorize(
      system.diagonal, system.lower, system.border, asymmetric);
   Check(!asymmetric_result.has_value(), "asymmetric border block was accepted");
   Check(
      asymmetric_result.error().code == EvaluationErrorCode::invalid_layout,
      "border asymmetry returned the wrong error");

   std::vector<Number> nonfinite = system.border;
   nonfinite[0] = std::numeric_limits<Number>::quiet_NaN();
   Check(
      !solver.factorize(
         system.diagonal,
         system.lower,
         nonfinite,
         system.border_diagonal).has_value(),
      "nonfinite border was accepted");
   Check(
      solver.factorize(
         system.diagonal,
         system.lower,
         system.border,
         system.border_diagonal).has_value(),
      "valid arrowhead was rejected after failed attempts");

   std::vector<Number> overflowing_input(
      7, std::numeric_limits<Number>::max());
   std::vector<Number> transactional_output(7, 66.);
   Check(
      !solver.apply(overflowing_input, transactional_output).has_value(),
      "nonfinite bordered matrix product was accepted");
   Check(
      std::ranges::all_of(
         transactional_output, [](Number value) { return value == 66.; }),
      "failed bordered matrix application modified caller output");

   std::vector<Number> wrong_rhs(6, 1.);
   std::vector<Number> wrong_output(6, 55.);
   Check(
      !solver.solve_rhs(wrong_rhs, wrong_output).has_value(),
      "wrong bordered RHS dimension was accepted");
   Check(
      std::ranges::all_of(
         wrong_output, [](Number value) { return value == 55.; }),
      "dimension failure modified bordered output");

   SymmetricBorderedBlockTridiagonalSolver singular({1}, 1);
   Check(
      !singular.factorize(
         std::array<Number, 1>{1.},
         {},
         std::array<Number, 1>{1.},
         std::array<Number, 1>{1.}).has_value(),
      "singular bordered Schur complement was accepted");
   Check(!singular.factorized(), "failed Schur factor retained stale factors");

   std::vector<Number> singular_border_diagonal = system.border_diagonal;
   singular_border_diagonal[0] = std::numeric_limits<Number>::quiet_NaN();
   Check(
      !solver.factorize(
         system.diagonal,
         system.lower,
         system.border,
         singular_border_diagonal).has_value(),
      "failed bordered refactorization was accepted");
   Check(!solver.factorized(), "failed bordered refactor retained stale factors");
   Check(
      !solver.solve_rhs(std::vector<Number>(7, 1.), output).has_value(),
      "stale bordered factors were used after failure");
}
} // namespace

int main()
{
   try
   {
      TestSolveApplyInertiaAndWorkspace();
      TestFailuresAreTransactional();
   }
   catch( const std::exception& exception )
   {
      std::cerr << exception.what() << '\n';
      return 1;
   }
   return 0;
}
