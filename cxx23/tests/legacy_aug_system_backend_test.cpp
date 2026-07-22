// Copyright (C) 2026 Ipopt contributors.
// All Rights Reserved.
// This file is distributed under the Eclipse Public License.

#include <ipopt/cxx23/legacy_aug_system_backend.hpp>
#include <ipopt/cxx23/legacy_matrix_snapshot.hpp>
#include <ipopt/cxx23/solve_session.hpp>

#include "IpDenseVector.hpp"
#include "IpGenTMatrix.hpp"
#include "IpSymTMatrix.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace
{
namespace Cxx = Ipopt::Cxx23;

constexpr Cxx::Number StableTolerance(
   Cxx::Number double_precision_tolerance,
   Cxx::Number stable_epsilon_multiplier
) noexcept
{
   return std::max(
      double_precision_tolerance,
      stable_epsilon_multiplier *
         static_cast<Cxx::Number>(
            std::numeric_limits<Ipopt::Number>::epsilon()));
}

void Check(bool condition, std::string_view message)
{
   if( !condition )
   {
      throw std::runtime_error(std::string(message));
   }
}

void CheckNear(
   std::span<const Cxx::Number> actual,
   std::span<const Cxx::Number> expected,
   Cxx::Number                 tolerance,
   std::string_view            message
)
{
   Check(actual.size() == expected.size(), "comparison dimensions differ");
   for( Cxx::Index i = 0; i < actual.size(); ++i )
   {
      if( std::abs(actual[i] - expected[i]) > tolerance )
      {
         throw std::runtime_error(
            std::string(message) + " at " + std::to_string(i) +
            ": actual=" + std::to_string(actual[i]) +
            " expected=" + std::to_string(expected[i]));
      }
   }
}

struct FullSystemModel
{
   Cxx::NlpStructure structure() const
   {
      return {2, 2, 4, 3, 901};
   }

   Cxx::EvaluationValue<Cxx::Number> eval_objective(
      std::span<const Cxx::Number> x
   )
   {
      return 2. * x[0] * x[0] + x[0] * x[1] + 1.5 * x[1] * x[1];
   }

   Cxx::EvaluationResult eval_gradient(
      std::span<const Cxx::Number> x,
      std::span<Cxx::Number>       result
   )
   {
      result[0] = 4. * x[0] + x[1];
      result[1] = x[0] + 3. * x[1];
      return {};
   }

   Cxx::EvaluationResult eval_constraints(
      std::span<const Cxx::Number> x,
      std::span<Cxx::Number>       result
   )
   {
      result[0] = 2. * x[0] - x[1];
      result[1] = .5 * x[0] + 2. * x[1];
      return {};
   }

   Cxx::EvaluationResult eval_jacobian_structure(
      std::span<Cxx::Index> rows,
      std::span<Cxx::Index> columns
   )
   {
      rows[0] = 0;
      columns[0] = 0;
      rows[1] = 0;
      columns[1] = 1;
      rows[2] = 1;
      columns[2] = 0;
      rows[3] = 1;
      columns[3] = 1;
      return {};
   }

   Cxx::EvaluationResult eval_jacobian_values(
      std::span<const Cxx::Number>,
      std::span<Cxx::Number> values
   )
   {
      values[0] = 2.;
      values[1] = -1.;
      values[2] = .5;
      values[3] = 2.;
      return {};
   }

   Cxx::EvaluationResult eval_hessian_structure(
      std::span<Cxx::Index> rows,
      std::span<Cxx::Index> columns
   )
   {
      rows[0] = 0;
      columns[0] = 0;
      rows[1] = 1;
      columns[1] = 0;
      rows[2] = 1;
      columns[2] = 1;
      return {};
   }

   Cxx::EvaluationResult eval_hessian_values(
      std::span<const Cxx::Number>,
      Cxx::Number objective_factor,
      std::span<const Cxx::Number>,
      std::span<Cxx::Number> values
   )
   {
      values[0] = 4. * objective_factor;
      values[1] = objective_factor;
      values[2] = 3. * objective_factor;
      return {};
   }

   Cxx::EvaluationResult eval_jacobian_product(
      std::span<const Cxx::Number>,
      std::span<const Cxx::Number> direction,
      std::span<Cxx::Number>       result
   )
   {
      result[0] = 2. * direction[0] - direction[1];
      result[1] = .5 * direction[0] + 2. * direction[1];
      return {};
   }

   Cxx::EvaluationResult eval_jacobian_transpose_product(
      std::span<const Cxx::Number>,
      std::span<const Cxx::Number> direction,
      std::span<Cxx::Number>       result
   )
   {
      result[0] = 2. * direction[0] + .5 * direction[1];
      result[1] = -direction[0] + 2. * direction[1];
      return {};
   }

   Cxx::EvaluationResult eval_hessian_product(
      std::span<const Cxx::Number>,
      Cxx::Number objective_factor,
      std::span<const Cxx::Number>,
      std::span<const Cxx::Number> direction,
      std::span<Cxx::Number>       result
   )
   {
      result[0] = objective_factor * (4. * direction[0] + direction[1]);
      result[1] = objective_factor * (direction[0] + 3. * direction[1]);
      return {};
   }
};

struct StableSolverCounters
{
   Cxx::Index multi_solve_calls = 0;
   Cxx::Index factorizations = 0;
   Cxx::Index backsolves = 0;
   Cxx::Index increase_quality_calls = 0;
   bool last_checked_inertia = false;
   Ipopt::Index last_expected_negative = 0;
   std::vector<Ipopt::Number> last_rhs;
};

class DenseRecordingAugSystemSolver final : public Ipopt::AugSystemSolver
{
public:
   explicit DenseRecordingAugSystemSolver(StableSolverCounters& counters)
      : counters_(counters)
   {
   }

   bool InitializeImpl(
      const Ipopt::OptionsList&,
      const std::string&
   ) override
   {
      return true;
   }

   Ipopt::ESymSolverStatus MultiSolve(
      const Ipopt::SymMatrix*                         W,
      Ipopt::Number                                   W_factor,
      const Ipopt::Vector*                            D_x,
      Ipopt::Number                                   delta_x,
      const Ipopt::Vector*                            D_s,
      Ipopt::Number                                   delta_s,
      const Ipopt::Matrix*                            J_c,
      const Ipopt::Vector*                            D_c,
      Ipopt::Number                                   delta_c,
      const Ipopt::Matrix*                            J_d,
      const Ipopt::Vector*                            D_d,
      Ipopt::Number                                   delta_d,
      std::vector<Ipopt::SmartPtr<const Ipopt::Vector> >& rhs_x,
      std::vector<Ipopt::SmartPtr<const Ipopt::Vector> >& rhs_s,
      std::vector<Ipopt::SmartPtr<const Ipopt::Vector> >& rhs_c,
      std::vector<Ipopt::SmartPtr<const Ipopt::Vector> >& rhs_d,
      std::vector<Ipopt::SmartPtr<Ipopt::Vector> >&       sol_x,
      std::vector<Ipopt::SmartPtr<Ipopt::Vector> >&       sol_s,
      std::vector<Ipopt::SmartPtr<Ipopt::Vector> >&       sol_c,
      std::vector<Ipopt::SmartPtr<Ipopt::Vector> >&       sol_d,
      bool                                             check_negative,
      Ipopt::Index                                     expected_negative
   ) override
   {
      ++counters_.multi_solve_calls;
      counters_.last_checked_inertia = check_negative;
      counters_.last_expected_negative = expected_negative;
      if( call_again_remaining_ > 0 )
      {
         --call_again_remaining_;
         return Ipopt::SYMSOLVER_CALL_AGAIN;
      }
      if( throw_on_call_ )
      {
         throw std::runtime_error("injected stable solver exception");
      }
      if( next_status_ != Ipopt::SYMSOLVER_SUCCESS )
      {
         const Ipopt::ESymSolverStatus result = next_status_;
         next_status_ = Ipopt::SYMSOLVER_SUCCESS;
         return result;
      }

      const auto* triplet_w = dynamic_cast<const Ipopt::SymTMatrix*>(W);
      const auto* triplet_jc = dynamic_cast<const Ipopt::GenTMatrix*>(J_c);
      const auto* triplet_jd = dynamic_cast<const Ipopt::GenTMatrix*>(J_d);
      if( triplet_w == nullptr || triplet_jc == nullptr || triplet_jd == nullptr ||
          rhs_x.empty() || rhs_s.size() != rhs_x.size() ||
          rhs_c.size() != rhs_x.size() || rhs_d.size() != rhs_x.size() )
      {
         return Ipopt::SYMSOLVER_FATAL_ERROR;
      }

      const Ipopt::Index nx = rhs_x[0]->Dim();
      const Ipopt::Index ns = rhs_s[0]->Dim();
      const Ipopt::Index nc = rhs_c[0]->Dim();
      const Ipopt::Index nd = rhs_d[0]->Dim();
      const Ipopt::Index dimension = nx + ns + nc + nd;
      std::vector<Ipopt::Number> matrix(
         static_cast<std::size_t>(dimension * dimension), 0.);
      const auto at = [dimension, &matrix](Ipopt::Index row, Ipopt::Index column)
         -> Ipopt::Number&
      {
         return matrix[static_cast<std::size_t>(row * dimension + column)];
      };

      for( Ipopt::Index entry = 0; entry < triplet_w->Nonzeros(); ++entry )
      {
         const Ipopt::Index row = triplet_w->Irows()[entry] - 1;
         const Ipopt::Index column = triplet_w->Jcols()[entry] - 1;
         const Ipopt::Number value = W_factor * triplet_w->Values()[entry];
         at(row, column) += value;
         if( row != column )
         {
            at(column, row) += value;
         }
      }

      AddDiagonal(D_x, delta_x, 0, nx, at);
      AddDiagonal(D_s, delta_s, nx, ns, at);
      AddDiagonal(D_c, -delta_c, nx + ns, nc, at);
      AddDiagonal(D_d, -delta_d, nx + ns + nc, nd, at);

      for( Ipopt::Index entry = 0; entry < triplet_jc->Nonzeros(); ++entry )
      {
         const Ipopt::Index row = triplet_jc->Irows()[entry] - 1;
         const Ipopt::Index column = triplet_jc->Jcols()[entry] - 1;
         const Ipopt::Number value = triplet_jc->Values()[entry];
         at(nx + ns + row, column) += value;
         at(column, nx + ns + row) += value;
      }
      for( Ipopt::Index entry = 0; entry < triplet_jd->Nonzeros(); ++entry )
      {
         const Ipopt::Index row = triplet_jd->Irows()[entry] - 1;
         const Ipopt::Index column = triplet_jd->Jcols()[entry] - 1;
         const Ipopt::Number value = triplet_jd->Values()[entry];
         at(nx + ns + nc + row, column) += value;
         at(column, nx + ns + nc + row) += value;
      }
      for( Ipopt::Index row = 0; row < nd; ++row )
      {
         at(nx + ns + nc + row, nx + row) = -1.;
         at(nx + row, nx + ns + nc + row) = -1.;
      }

      if( matrix != matrix_ )
      {
         matrix_ = matrix;
         if( !Factorize(static_cast<std::size_t>(dimension)) )
         {
            return Ipopt::SYMSOLVER_SINGULAR;
         }
         ++counters_.factorizations;
      }
      if( check_negative && expected_negative != reported_negative_eigenvalues_ )
      {
         return Ipopt::SYMSOLVER_WRONG_INERTIA;
      }

      for( std::size_t right_hand_side = 0;
           right_hand_side < rhs_x.size(); ++right_hand_side )
      {
         std::vector<Ipopt::Number> values(static_cast<std::size_t>(dimension));
         GetVector(*rhs_x[right_hand_side],
            std::span<Ipopt::Number>(values).subspan(0, static_cast<std::size_t>(nx)));
         GetVector(*rhs_s[right_hand_side],
            std::span<Ipopt::Number>(values).subspan(
               static_cast<std::size_t>(nx), static_cast<std::size_t>(ns)));
         GetVector(*rhs_c[right_hand_side],
            std::span<Ipopt::Number>(values).subspan(
               static_cast<std::size_t>(nx + ns), static_cast<std::size_t>(nc)));
         GetVector(*rhs_d[right_hand_side],
            std::span<Ipopt::Number>(values).subspan(
               static_cast<std::size_t>(nx + ns + nc), static_cast<std::size_t>(nd)));
         counters_.last_rhs = values;
         SolveFactorized(values);
         if( return_nonfinite_ )
         {
            values[0] = std::numeric_limits<Ipopt::Number>::quiet_NaN();
         }
         PutVector(
            std::span<const Ipopt::Number>(values).subspan(
               0, static_cast<std::size_t>(nx)), *sol_x[right_hand_side]);
         PutVector(
            std::span<const Ipopt::Number>(values).subspan(
               static_cast<std::size_t>(nx), static_cast<std::size_t>(ns)),
            *sol_s[right_hand_side]);
         PutVector(
            std::span<const Ipopt::Number>(values).subspan(
               static_cast<std::size_t>(nx + ns), static_cast<std::size_t>(nc)),
            *sol_c[right_hand_side]);
         PutVector(
            std::span<const Ipopt::Number>(values).subspan(
               static_cast<std::size_t>(nx + ns + nc), static_cast<std::size_t>(nd)),
            *sol_d[right_hand_side]);
         ++counters_.backsolves;
      }
      return Ipopt::SYMSOLVER_SUCCESS;
   }

   Ipopt::Index NumberOfNegEVals() const override
   {
      return reported_negative_eigenvalues_;
   }

   bool ProvidesInertia() const override
   {
      return provides_inertia_;
   }

   bool IncreaseQuality() override
   {
      ++counters_.increase_quality_calls;
      return true;
   }

   void set_reported_negative_eigenvalues(Ipopt::Index value)
   {
      reported_negative_eigenvalues_ = value;
   }

   void set_provides_inertia(bool value)
   {
      provides_inertia_ = value;
   }

   void set_next_status(Ipopt::ESymSolverStatus status)
   {
      next_status_ = status;
   }

   void set_return_nonfinite(bool value)
   {
      return_nonfinite_ = value;
   }

   void set_throw_on_call(bool value)
   {
      throw_on_call_ = value;
   }

   void set_call_again_remaining(Cxx::Index value)
   {
      call_again_remaining_ = value;
   }

private:
   template <class Accessor>
   static void AddDiagonal(
      const Ipopt::Vector* diagonal,
      Ipopt::Number        scalar,
      Ipopt::Index         offset,
      Ipopt::Index         dimension,
      Accessor&&           at
   )
   {
      std::vector<Ipopt::Number> values(static_cast<std::size_t>(dimension), 0.);
      if( diagonal != nullptr )
      {
         GetVector(*diagonal, values);
      }
      for( Ipopt::Index i = 0; i < dimension; ++i )
      {
         at(offset + i, offset + i) += values[static_cast<std::size_t>(i)] + scalar;
      }
   }

   static void GetVector(
      const Ipopt::Vector& vector,
      std::span<Ipopt::Number> values
   )
   {
      Ipopt::TripletHelper::FillValuesFromVector(
         static_cast<Ipopt::Index>(values.size()), vector, values.data());
   }

   static void PutVector(
      std::span<const Ipopt::Number> values,
      Ipopt::Vector&                 vector
   )
   {
      Ipopt::TripletHelper::PutValuesInVector(
         static_cast<Ipopt::Index>(values.size()), values.data(), vector);
   }

   bool Factorize(std::size_t dimension)
   {
      lu_ = matrix_;
      if( dimension == 0 || lu_.size() != dimension * dimension )
      {
         return false;
      }
      pivots_.assign(dimension, 0);
      for( std::size_t column = 0; column < dimension; ++column )
      {
         std::size_t pivot = column;
         for( std::size_t row = column + 1; row < dimension; ++row )
         {
            if( std::abs(lu_[row * dimension + column]) >
                std::abs(lu_[pivot * dimension + column]) )
            {
               pivot = row;
            }
         }
         if( std::abs(lu_[pivot * dimension + column]) < 1e-14 )
         {
            return false;
         }
         pivots_[column] = pivot;
         if( pivot != column )
         {
            for( std::size_t entry = 0; entry < dimension; ++entry )
            {
               std::swap(
                  lu_[column * dimension + entry],
                  lu_[pivot * dimension + entry]);
            }
         }
         for( std::size_t row = column + 1; row < dimension; ++row )
         {
            lu_[row * dimension + column] /= lu_[column * dimension + column];
            for( std::size_t entry = column + 1; entry < dimension; ++entry )
            {
               lu_[row * dimension + entry] -=
                  lu_[row * dimension + column] *
                  lu_[column * dimension + entry];
            }
         }
      }
      return true;
   }

   void SolveFactorized(std::vector<Ipopt::Number>& values) const
   {
      const std::size_t dimension = pivots_.size();
      for( std::size_t column = 0; column < dimension; ++column )
      {
         if( pivots_[column] != column )
         {
            std::swap(values[column], values[pivots_[column]]);
         }
      }
      for( std::size_t row = 1; row < dimension; ++row )
      {
         for( std::size_t column = 0; column < row; ++column )
         {
            values[row] -= lu_[row * dimension + column] * values[column];
         }
      }
      for( std::size_t reverse = 0; reverse < dimension; ++reverse )
      {
         const std::size_t row = dimension - 1 - reverse;
         for( std::size_t column = row + 1; column < dimension; ++column )
         {
            values[row] -= lu_[row * dimension + column] * values[column];
         }
         values[row] /= lu_[row * dimension + row];
      }
   }

   StableSolverCounters& counters_;
   std::vector<Ipopt::Number> matrix_;
   std::vector<Ipopt::Number> lu_;
   std::vector<std::size_t> pivots_;
   Ipopt::Index reported_negative_eigenvalues_ = 2;
   Ipopt::ESymSolverStatus next_status_ = Ipopt::SYMSOLVER_SUCCESS;
   Cxx::Index call_again_remaining_ = 0;
   bool provides_inertia_ = true;
   bool return_nonfinite_ = false;
   bool throw_on_call_ = false;
};

struct Fixture
{
   Fixture()
   {
      x_space = new Ipopt::DenseVectorSpace(2);
      s_space = new Ipopt::DenseVectorSpace(1);
      c_space = new Ipopt::DenseVectorSpace(1);
      d_space = new Ipopt::DenseVectorSpace(1);
      x = x_space->MakeNewDenseVector();
      s = s_space->MakeNewDenseVector();
      c = c_space->MakeNewDenseVector();
      d = d_space->MakeNewDenseVector();
      sigma_x = x_space->MakeNewDenseVector();
      sigma_s = s_space->MakeNewDenseVector();
      x->Set(0.);
      s->Set(0.);
      c->Set(0.);
      d->Set(0.);
      sigma_x->Values()[0] = z_lower[0] / slack_x_lower[0];
      sigma_x->Values()[1] = z_upper[0] / slack_x_upper[0];
      sigma_s->Values()[0] =
         v_lower[0] / slack_s_lower[0] + v_upper[0] / slack_s_upper[0];

      const Ipopt::Index hessian_rows[] = {1, 2, 2};
      const Ipopt::Index hessian_columns[] = {1, 1, 2};
      hessian_space = new Ipopt::SymTMatrixSpace(
         2, 3, hessian_rows, hessian_columns);
      hessian = hessian_space->MakeNewSymTMatrix();
      const Ipopt::Number hessian_values[] = {4., 1., 3.};
      hessian->SetValues(hessian_values);

      const Ipopt::Index jacobian_rows[] = {1, 1};
      const Ipopt::Index jacobian_columns[] = {1, 2};
      equality_jacobian_space = new Ipopt::GenTMatrixSpace(
         1, 2, 2, jacobian_rows, jacobian_columns);
      equality_jacobian = equality_jacobian_space->MakeNewGenTMatrix();
      const Ipopt::Number equality_values[] = {2., -1.};
      equality_jacobian->SetValues(equality_values);

      inequality_jacobian_space = new Ipopt::GenTMatrixSpace(
         1, 2, 2, jacobian_rows, jacobian_columns);
      inequality_jacobian = inequality_jacobian_space->MakeNewGenTMatrix();
      const Ipopt::Number inequality_values[] = {.5, 2.};
      inequality_jacobian->SetValues(inequality_values);

      solver = new DenseRecordingAugSystemSolver(counters);
   }

   Cxx::PrimalDualKktOperator MakeKkt() const
   {
      return Cxx::PrimalDualKktOperator(
         Cxx::MakeNlpProblem(FullSystemModel{}),
         {
            .equality_constraints = {0},
            .inequality_constraints = {1},
            .primal_lower_bounds = {0},
            .primal_upper_bounds = {1},
            .slack_lower_bounds = {0},
            .slack_upper_bounds = {0}
         });
   }

   Cxx::PrimalDualState State(std::uint64_t numeric_revision = 77) const
   {
      return {
         .nlp = {x_state, 1., multipliers},
         .z_lower = z_lower,
         .z_upper = z_upper,
         .v_lower = v_lower,
         .v_upper = v_upper,
         .slack_x_lower = slack_x_lower,
         .slack_x_upper = slack_x_upper,
         .slack_s_lower = slack_s_lower,
         .slack_s_upper = slack_s_upper,
         .regularization = {.2, .3, .1, .15},
         .numeric_revision = numeric_revision
      };
   }

   Cxx::LegacyAugSystemViews Views() const
   {
      return {
         .solver = Ipopt::GetRawPtr(solver),
         .hessian = Ipopt::GetRawPtr(hessian),
         .jacobian_equalities = Ipopt::GetRawPtr(equality_jacobian),
         .jacobian_inequalities = Ipopt::GetRawPtr(inequality_jacobian),
         .x_prototype = Ipopt::GetRawPtr(x),
         .s_prototype = Ipopt::GetRawPtr(s),
         .equality_prototype = Ipopt::GetRawPtr(c),
         .inequality_prototype = Ipopt::GetRawPtr(d)
      };
   }

   Cxx::LegacyAugSystemViews ViewsWithLiveDiagonals() const
   {
      Cxx::LegacyAugSystemViews views = Views();
      views.complementarity_x_diagonal = Ipopt::GetRawPtr(sigma_x);
      views.complementarity_s_diagonal = Ipopt::GetRawPtr(sigma_s);
      return views;
   }

   StableSolverCounters counters;
   Ipopt::SmartPtr<Ipopt::DenseVectorSpace> x_space;
   Ipopt::SmartPtr<Ipopt::DenseVectorSpace> s_space;
   Ipopt::SmartPtr<Ipopt::DenseVectorSpace> c_space;
   Ipopt::SmartPtr<Ipopt::DenseVectorSpace> d_space;
   Ipopt::SmartPtr<Ipopt::DenseVector> x;
   Ipopt::SmartPtr<Ipopt::DenseVector> s;
   Ipopt::SmartPtr<Ipopt::DenseVector> c;
   Ipopt::SmartPtr<Ipopt::DenseVector> d;
   Ipopt::SmartPtr<Ipopt::DenseVector> sigma_x;
   Ipopt::SmartPtr<Ipopt::DenseVector> sigma_s;
   Ipopt::SmartPtr<Ipopt::SymTMatrixSpace> hessian_space;
   Ipopt::SmartPtr<Ipopt::SymTMatrix> hessian;
   Ipopt::SmartPtr<Ipopt::GenTMatrixSpace> equality_jacobian_space;
   Ipopt::SmartPtr<Ipopt::GenTMatrix> equality_jacobian;
   Ipopt::SmartPtr<Ipopt::GenTMatrixSpace> inequality_jacobian_space;
   Ipopt::SmartPtr<Ipopt::GenTMatrix> inequality_jacobian;
   Ipopt::SmartPtr<DenseRecordingAugSystemSolver> solver;

   const std::array<Cxx::Number, 2> x_state{{.2, -.3}};
   const std::array<Cxx::Number, 2> multipliers{{.4, -.2}};
   const std::array<Cxx::Number, 1> z_lower{{1.2}};
   const std::array<Cxx::Number, 1> z_upper{{.7}};
   const std::array<Cxx::Number, 1> v_lower{{.9}};
   const std::array<Cxx::Number, 1> v_upper{{1.1}};
   const std::array<Cxx::Number, 1> slack_x_lower{{2.}};
   const std::array<Cxx::Number, 1> slack_x_upper{{1.5}};
   const std::array<Cxx::Number, 1> slack_s_lower{{.8}};
   const std::array<Cxx::Number, 1> slack_s_upper{{1.4}};
};

void CheckPackedRightHandSide(
   const Fixture&                    fixture,
   std::span<const Cxx::Number>      full_rhs
)
{
   std::array<Cxx::Number, 5> expected{{
      full_rhs[0] + full_rhs[5] / fixture.slack_x_lower[0],
      full_rhs[1] - full_rhs[6] / fixture.slack_x_upper[0],
      full_rhs[2] + full_rhs[7] / fixture.slack_s_lower[0] -
         full_rhs[8] / fixture.slack_s_upper[0],
      full_rhs[3],
      full_rhs[4]
   }};
   std::array<Cxx::Number, 5> actual{};
   Check(fixture.counters.last_rhs.size() == actual.size(), "reduced RHS size is wrong");
   std::ranges::transform(
      fixture.counters.last_rhs, actual.begin(),
      [](Ipopt::Number value) { return static_cast<Cxx::Number>(value); });
   CheckNear(actual, expected, StableTolerance(1e-13, 128.), "reduced RHS");
}

void TestEndToEndPreparedReuse()
{
   Fixture fixture;
   Cxx::PrimalDualKktOperator kkt = fixture.MakeKkt();
   Check(kkt.valid(), "full-system KKT layout was rejected");
   const Cxx::PrimalDualState state = fixture.State();

   Cxx::EvaluationValue<Cxx::PreparedDirectPreconditioner> prepared_result =
      Cxx::PrepareLegacyAugSystemPreconditioner(
         fixture.Views(), kkt, state);
   Check(prepared_result.has_value(), "legacy augmented-system preparation failed");
   Cxx::PreparedDirectPreconditioner preconditioner =
      std::move(*prepared_result);
   Check(fixture.counters.factorizations == 1, "stable backend was not factorized once");
   Check(fixture.counters.backsolves == 1, "factor preparation did not use one zero RHS");
   Check(fixture.counters.last_checked_inertia, "factor preparation skipped inertia");
   Check(fixture.counters.last_expected_negative == 2, "wrong inertia target");

   Cxx::FgmresOptions options;
   options.restart = 9;
   options.maximum_iterations = 9;
   options.relative_tolerance = StableTolerance(1e-12, 256.);
   options.absolute_tolerance = StableTolerance(1e-14, 128.);
   Cxx::EvaluationValue<Cxx::PrimalDualSolveSession> session_result =
      Cxx::MakePrimalDualSolveSession(kkt, options);
   Check(session_result.has_value(), "full-system solve session creation failed");
   Cxx::PrimalDualSolveSession session = std::move(*session_result);

   const std::array<Cxx::Number, 9> first_exact{{
      .3, -.8, .4, -.2, .7, -.5, .6, -.9, .35
   }};
   std::array<Cxx::Number, 9> rhs{};
   Check(kkt.apply_flat(state, first_exact, rhs).has_value(), "first KKT product failed");
   std::array<Cxx::Number, 9> direct_solution{};
   Check(
      preconditioner(0, rhs, direct_solution).has_value(),
      "direct legacy full-system solve failed");
   CheckNear(
      direct_solution, first_exact,
      StableTolerance(2e-12, 512.), "direct full-system direction");
   CheckPackedRightHandSide(fixture, rhs);

   std::array<Cxx::Number, 9> solution{};
   Cxx::EvaluationValue<Cxx::FgmresResult> first = session.solve(
      kkt, state, preconditioner, rhs, solution);
   Check(first.has_value() && first->converged(), "first legacy-preconditioned solve failed");
   Check(first->iterations == 1, "exact legacy preconditioner needed more than one step");
   CheckNear(
      solution, first_exact,
      StableTolerance(2e-12, 512.), "first full-system direction");

   std::array<Cxx::Number, 9> reconstructed_rhs{};
   Check(
      kkt.apply_flat(state, solution, reconstructed_rhs).has_value(),
      "full-system residual product failed");
   CheckNear(
      reconstructed_rhs, rhs,
      StableTolerance(3e-12, 512.), "first full-system residual");

   const std::array<Cxx::Number, 9> second_exact{{
      -1.1, .25, -.6, .75, -.2, .4, -.3, .5, -.45
   }};
   Check(kkt.apply_flat(state, second_exact, rhs).has_value(), "second KKT product failed");
   std::ranges::fill(solution, 0.);
   Cxx::EvaluationValue<Cxx::FgmresResult> second = session.solve(
      kkt, state, preconditioner, rhs, solution);
   Check(second.has_value() && second->converged(), "second legacy-preconditioned solve failed");
   Check(second->iterations == 1, "second exact solve needed more than one step");
   CheckNear(
      solution, second_exact,
      StableTolerance(2e-12, 512.), "second full-system direction");
   Check(fixture.counters.factorizations == 1, "second RHS refactorized stable backend");
   Check(fixture.counters.backsolves == 4, "stable backsolve count is wrong");
   Check(
      preconditioner.statistics().factorizations == 1 &&
         preconditioner.statistics().successful_solves == 3,
      "legacy prepared statistics are wrong");
   Check(fixture.counters.increase_quality_calls == 0,
      "adapter stole IncreaseQuality policy from PDFullSpaceSolver");
}

void TestMatrixSnapshot()
{
   Fixture fixture;
   Cxx::EvaluationValue<Cxx::AnyNlpProblem> snapshot_result =
      Cxx::MakeLegacyMatrixSnapshotProblem({
         .hessian = Ipopt::GetRawPtr(fixture.hessian),
         .jacobian_equalities = Ipopt::GetRawPtr(fixture.equality_jacobian),
         .jacobian_inequalities = Ipopt::GetRawPtr(fixture.inequality_jacobian),
         .structural_revision = 701
      });
   Check(snapshot_result.has_value(), "stable matrix snapshot failed");
   Cxx::AnyNlpProblem snapshot = std::move(*snapshot_result);
   Check(
      snapshot.nlp_structure() == Cxx::NlpStructure{2, 2, 4, 3, 701},
      "stable matrix snapshot structure is wrong");
   Check(
      snapshot.nlp_jacobian_product_capabilities() ==
         Cxx::JacobianProductCapabilities{true, true},
      "stable matrix snapshot Jacobian capabilities are wrong");
   Check(
      snapshot.nlp_hessian_product_capabilities() ==
         Cxx::HessianProductCapabilities{true},
      "stable matrix snapshot Hessian capability is wrong");

   const std::array<Cxx::Number, 2> x{{.2, -.3}};
   const std::array<Cxx::Number, 2> direction{{.3, -.7}};
   std::array<Cxx::Number, 2> product{};
   Check(
      snapshot.nlp_jacobian_product(x, direction, product).has_value(),
      "stable matrix snapshot forward product failed");
   const std::array<Cxx::Number, 2> expected_forward{{1.3, -1.25}};
   CheckNear(
      product, expected_forward,
      StableTolerance(1e-14, 64.), "snapshot forward product");

   const std::array<Cxx::Number, 2> transpose_direction{{.4, -.6}};
   Check(
      snapshot.nlp_jacobian_transpose_product(
         x, transpose_direction, product).has_value(),
      "stable matrix snapshot transpose product failed");
   const std::array<Cxx::Number, 2> expected_transpose{{.5, -1.6}};
   CheckNear(
      product, expected_transpose,
      StableTolerance(1e-14, 64.), "snapshot transpose product");

   const std::array<Cxx::Number, 2> multipliers{{.4, -.2}};
   Check(
      snapshot.nlp_hessian_product(
         x, 1., multipliers, direction, product).has_value(),
      "stable matrix snapshot Hessian product failed");
   const std::array<Cxx::Number, 2> expected_hessian{{.5, -1.8}};
   CheckNear(
      product, expected_hessian,
      StableTolerance(1e-14, 64.), "snapshot Hessian product");

   std::array<Cxx::Number, 2> aliased = direction;
   Check(
      snapshot.nlp_jacobian_product(x, aliased, aliased).has_value(),
      "stable matrix snapshot alias-safe product failed");
   CheckNear(
      aliased, expected_forward,
      StableTolerance(1e-14, 64.), "snapshot aliased product");

   const std::array<Cxx::Number, 2> overflowing{{
      std::numeric_limits<Cxx::Number>::max(),
      std::numeric_limits<Cxx::Number>::max()
   }};
   product = {17., 19.};
   const std::array<Cxx::Number, 2> sentinel = product;
   Cxx::EvaluationResult nonfinite = snapshot.nlp_hessian_product(
      x, 1., multipliers, overflowing, product);
   Check(!nonfinite.has_value(), "nonfinite snapshot product was accepted");
   Check(
      nonfinite.error().code == Cxx::EvaluationErrorCode::nonfinite_output,
      "nonfinite snapshot product returned the wrong error");
   Check(product == sentinel, "nonfinite snapshot product modified output");

   Cxx::EvaluationValue<Cxx::AnyNlpProblem> missing =
      Cxx::MakeLegacyMatrixSnapshotProblem({});
   Check(!missing.has_value(), "missing stable snapshot matrices were accepted");
}

void TestValidationAndFailures()
{
   {
      Fixture fixture;
      Cxx::PrimalDualKktOperator kkt = fixture.MakeKkt();
      const Cxx::PrimalDualState state = fixture.State();
      Cxx::EvaluationValue<Cxx::PreparedDirectPreconditioner> prepared_result =
         Cxx::PrepareLegacyAugSystemPreconditioner(
            fixture.ViewsWithLiveDiagonals(), kkt, state);
      Check(prepared_result.has_value(), "matching live sigma vectors were rejected");
      Cxx::PreparedDirectPreconditioner preconditioner =
         std::move(*prepared_result);
      const std::array<Cxx::Number, 9> input{{
         1., 2., 3., 4., 5., 6., 7., 8., 9.
      }};
      std::array<Cxx::Number, 9> output{{
         11., 12., 13., 14., 15., 16., 17., 18., 19.
      }};
      const std::array<Cxx::Number, 9> sentinel = output;
      fixture.sigma_x->Values()[0] += .25;
      const Cxx::Index calls_before = fixture.counters.multi_solve_calls;
      Cxx::EvaluationResult stale = preconditioner(0, input, output);
      Check(!stale.has_value(), "changed live sigma tag was accepted");
      Check(
         stale.error().code == Cxx::EvaluationErrorCode::numeric_mismatch,
         "changed live sigma tag returned the wrong error");
      Check(output == sentinel, "changed live sigma tag modified output");
      Check(
         fixture.counters.multi_solve_calls == calls_before,
         "changed live sigma tag entered the solver");
   }
   {
      Fixture fixture;
      fixture.sigma_s->Values()[0] += .25;
      Cxx::PrimalDualKktOperator kkt = fixture.MakeKkt();
      Cxx::EvaluationValue<Cxx::AnyDirectSolverBackend> mismatch =
         Cxx::MakeLegacyAugSystemDirectBackend(
            fixture.ViewsWithLiveDiagonals(), kkt, fixture.State());
      Check(!mismatch.has_value(), "inconsistent live sigma values were accepted");
      Check(
         mismatch.error().code == Cxx::EvaluationErrorCode::numeric_mismatch,
         "inconsistent live sigma values returned the wrong error");
   }
   {
      Fixture fixture;
      Cxx::PrimalDualKktOperator kkt = fixture.MakeKkt();
      Cxx::PrimalDualState invalid_state = fixture.State();
      const std::array<Cxx::Number, 1> zero_slack{{0.}};
      invalid_state.slack_x_lower = zero_slack;
      Cxx::EvaluationValue<Cxx::AnyDirectSolverBackend> invalid =
         Cxx::MakeLegacyAugSystemDirectBackend(
            fixture.Views(), kkt, invalid_state);
      Check(!invalid.has_value(), "zero complementarity slack was accepted");
   }
   {
      Fixture fixture;
      fixture.solver->set_provides_inertia(false);
      Cxx::PrimalDualKktOperator kkt = fixture.MakeKkt();
      Cxx::EvaluationValue<Cxx::AnyDirectSolverBackend> rejected =
         Cxx::MakeLegacyAugSystemDirectBackend(
            fixture.Views(), kkt, fixture.State());
      Check(!rejected.has_value(), "inertia-less solver accepted an inertia check");

      Cxx::EvaluationValue<Cxx::PreparedDirectPreconditioner> accepted =
         Cxx::PrepareLegacyAugSystemPreconditioner(
            fixture.Views(), kkt, fixture.State(), {.check_inertia = false});
      Check(accepted.has_value(), "explicit inertia-free preparation failed");
   }
   {
      Fixture fixture;
      fixture.solver->set_reported_negative_eigenvalues(1);
      Cxx::PrimalDualKktOperator kkt = fixture.MakeKkt();
      Cxx::EvaluationValue<Cxx::PreparedDirectPreconditioner> wrong_inertia =
         Cxx::PrepareLegacyAugSystemPreconditioner(
            fixture.Views(), kkt, fixture.State());
      Check(!wrong_inertia.has_value(), "wrong stable inertia was accepted");
      Check(fixture.counters.factorizations == 1, "wrong inertia skipped factorization");
   }
   {
      Fixture fixture;
      fixture.solver->set_call_again_remaining(2);
      Cxx::PrimalDualKktOperator kkt = fixture.MakeKkt();
      Cxx::EvaluationValue<Cxx::PreparedDirectPreconditioner> retried =
         Cxx::PrepareLegacyAugSystemPreconditioner(
            fixture.Views(), kkt, fixture.State());
      Check(retried.has_value(), "bounded stable call-again retry failed");
      Check(fixture.counters.multi_solve_calls == 3, "call-again retry count is wrong");
      Check(fixture.counters.factorizations == 1, "call-again path refactorized");
   }
   {
      Fixture fixture;
      Cxx::PrimalDualKktOperator kkt = fixture.MakeKkt();
      const Cxx::PrimalDualState state = fixture.State();
      Cxx::EvaluationValue<Cxx::PreparedDirectPreconditioner> prepared_result =
         Cxx::PrepareLegacyAugSystemPreconditioner(
            fixture.Views(), kkt, state);
      Check(prepared_result.has_value(), "failure fixture preparation failed");
      Cxx::PreparedDirectPreconditioner preconditioner =
         std::move(*prepared_result);
      const std::array<Cxx::Number, 9> input{{
         1., 2., 3., 4., 5., 6., 7., 8., 9.
      }};
      std::array<Cxx::Number, 9> output{{
         11., 12., 13., 14., 15., 16., 17., 18., 19.
      }};
      const std::array<Cxx::Number, 9> sentinel = output;

      fixture.solver->set_next_status(Ipopt::SYMSOLVER_SINGULAR);
      Cxx::EvaluationResult singular = preconditioner(0, input, output);
      Check(!singular.has_value(), "stable singular status was ignored");
      Check(output == sentinel, "stable singular solve modified output");

      fixture.solver->set_throw_on_call(true);
      Cxx::EvaluationResult thrown = preconditioner(0, input, output);
      Check(!thrown.has_value(), "stable exception escaped as success");
      Check(output == sentinel, "stable exception modified output");
      fixture.solver->set_throw_on_call(false);

      fixture.solver->set_return_nonfinite(true);
      Cxx::EvaluationResult nonfinite = preconditioner(0, input, output);
      Check(!nonfinite.has_value(), "stable nonfinite result was accepted");
      Check(
         nonfinite.error().code == Cxx::EvaluationErrorCode::nonfinite_output,
         "stable nonfinite result returned the wrong error");
      Check(output == sentinel, "stable nonfinite result modified output");
      fixture.solver->set_return_nonfinite(false);

      fixture.hessian->Values()[0] += .25;
      const Cxx::Index calls_before = fixture.counters.multi_solve_calls;
      Cxx::EvaluationResult stale = preconditioner(0, input, output);
      Check(!stale.has_value(), "changed stable matrix tag was accepted");
      Check(
         stale.error().code == Cxx::EvaluationErrorCode::numeric_mismatch,
         "changed stable matrix returned the wrong error");
      Check(output == sentinel, "changed stable matrix modified output");
      Check(
         fixture.counters.multi_solve_calls == calls_before,
         "changed stable matrix entered the solver");
   }
}
} // namespace

int main()
{
   try
   {
      TestEndToEndPreparedReuse();
      TestMatrixSnapshot();
      TestValidationAndFailures();
   }
   catch( const std::exception& exception )
   {
      std::cerr << exception.what() << '\n';
      return 1;
   }
   return 0;
}
