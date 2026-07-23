// Copyright (C) 2026 Ipopt contributors.
// All Rights Reserved.
// This file is distributed under the Eclipse Public License.

#include <IpCompoundVector.hpp>
#include <IpDenseVector.hpp>
#include <IpRestoIterateInitializer.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

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

void CheckNear(
   Stable::Number actual,
   Stable::Number expected,
   Stable::Number relative_tolerance,
   std::string_view message
)
{
   const Stable::Number scale =
      std::max({Stable::Number{1.}, std::abs(actual), std::abs(expected)});
   if( std::abs(actual - expected) > relative_tolerance * scale )
   {
      throw std::runtime_error(
         std::string(message) + ": actual=" + std::to_string(actual) +
         " expected=" + std::to_string(expected));
   }
}

void TestCompensatedDenseSum()
{
   Stable::SmartPtr<Stable::DenseVectorSpace> space =
      new Stable::DenseVectorSpace(3);
   Stable::SmartPtr<Stable::DenseVector> values = space->MakeNewDenseVector();
   const std::array<Stable::Number, 3> input{{1e16, 1., -1e16}};
   values->SetValues(input.data());
   Check(values->Sum() == 1., "dense sum lost a representable cancellation");
}

void TestCompoundNormRange()
{
   Stable::SmartPtr<Stable::DenseVectorSpace> first_space =
      new Stable::DenseVectorSpace(1);
   Stable::SmartPtr<Stable::DenseVectorSpace> second_space =
      new Stable::DenseVectorSpace(1);
   Stable::SmartPtr<Stable::CompoundVectorSpace> compound_space =
      new Stable::CompoundVectorSpace(2, 2);
   compound_space->SetCompSpace(0, *first_space);
   compound_space->SetCompSpace(1, *second_space);
   Stable::SmartPtr<Stable::CompoundVector> values =
      compound_space->MakeNewCompoundVector();
   const Stable::Number component =
      std::numeric_limits<Stable::Number>::max() / 4.;
   values->GetCompNonConst(0)->Set(component);
   values->GetCompNonConst(1)->Set(component);
   const Stable::Number expected = std::hypot(component, component);
   const Stable::Number actual = values->Nrm2();
   Check(std::isfinite(actual), "compound norm overflowed a finite result");
   CheckNear(
      actual,
      expected,
      4. * std::numeric_limits<Stable::Number>::epsilon(),
      "compound norm is inaccurate");
}

void TestRestorationSlackRange()
{
   Stable::SmartPtr<Stable::DenseVectorSpace> space =
      new Stable::DenseVectorSpace(7);
   Stable::SmartPtr<Stable::DenseVector> residual =
      space->MakeNewDenseVector();
   Stable::SmartPtr<Stable::DenseVector> negative =
      space->MakeNewDenseVector();
   Stable::SmartPtr<Stable::DenseVector> positive =
      space->MakeNewDenseVector();
   const std::array<Stable::Number, 7> input{{
      1e200, -1e200, 1e12, -1e12, 1e-200, -1e-200, 0.
   }};
   residual->SetValues(input.data());
   Stable::RestoIterateInitializer::ComputeSlackVariables(
      1., *residual, *negative, *positive);

   const Stable::Number* n = negative->Values();
   const Stable::Number* p = positive->Values();
   const Stable::Number tolerance =
      16. * std::numeric_limits<Stable::Number>::epsilon();
   for( Stable::Index i = 0; i < residual->Dim(); ++i )
   {
      Check(
         n[i] > 0. && p[i] > 0. &&
            std::isfinite(n[i]) && std::isfinite(p[i]),
         "restoration slack is not positive and finite");
      const Stable::Number half_magnitude = 0.5 * std::abs(input[i]);
      const Stable::Number denominator =
         std::hypot(1., half_magnitude) + half_magnitude;
      const Stable::Number expected_smaller = 1. + 1. / denominator;
      const Stable::Number actual_smaller = input[i] >= 0. ? n[i] : p[i];
      CheckNear(
         actual_smaller,
         expected_smaller,
         tolerance,
         "small restoration slack is inaccurate");
      CheckNear(
         p[i] - n[i],
         input[i],
         tolerance,
         "restoration slacks violate p-n=residual");
   }
}
} // namespace

int main()
{
   try
   {
      TestCompensatedDenseSum();
      TestCompoundNormRange();
      TestRestorationSlackRange();
   }
   catch( const std::exception& error )
   {
      std::cerr << "ipopt_cxx23_numerical_primitives_test: "
                << error.what() << '\n';
      return 1;
   }
   std::cout << "ipopt_cxx23_numerical_primitives_test: passed\n";
   return 0;
}
