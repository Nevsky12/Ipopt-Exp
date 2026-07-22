// Copyright (C) 2026 Ipopt contributors.
// All Rights Reserved.
// This file is distributed under the Eclipse Public License.

#include <ipopt/cxx23/legacy_ipopt_bridge.hpp>

#include <IpIpoptApplication.hpp>
#include <IpTimingStatistics.hpp>

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
namespace Stable = ::Ipopt;
namespace Modern = ::Ipopt::Cxx23;

constexpr Modern::Number kStableValueTolerance = std::max(
   Modern::Number{1e-12},
   Modern::Number{64.} *
      static_cast<Modern::Number>(std::numeric_limits<Stable::Number>::epsilon()));

class BridgeTnlp final : public Stable::TNLP
{
public:
   explicit BridgeTnlp(bool fortran_style = false)
      : fortran_style_(fortran_style)
   {
   }

   bool get_nlp_info(
      Stable::Index&          n,
      Stable::Index&          m,
      Stable::Index&          jacobian_nonzeros,
      Stable::Index&          hessian_nonzeros,
      IndexStyleEnum&         index_style
   ) override
   {
      n = 4;
      m = 3;
      jacobian_nonzeros = 6;
      hessian_nonzeros = 4;
      index_style = fortran_style_ ? FORTRAN_STYLE : C_STYLE;
      return true;
   }

   bool get_bounds_info(
      Stable::Index,
      Stable::Number* x_lower,
      Stable::Number* x_upper,
      Stable::Index,
      Stable::Number* g_lower,
      Stable::Number* g_upper
   ) override
   {
      constexpr std::array<Stable::Number, 4> expected_x_lower{{-10., 2., -10., -10.}};
      constexpr std::array<Stable::Number, 4> expected_x_upper{{10., 2., 10., 10.}};
      constexpr std::array<Stable::Number, 3> expected_g_lower{{5., -1., -3.}};
      constexpr std::array<Stable::Number, 3> expected_g_upper{{5., 9., -3.}};
      std::ranges::copy(expected_x_lower, x_lower);
      std::ranges::copy(expected_x_upper, x_upper);
      std::ranges::copy(expected_g_lower, g_lower);
      std::ranges::copy(expected_g_upper, g_upper);
      return true;
   }

   bool get_starting_point(
      Stable::Index,
      bool            init_x,
      Stable::Number* x,
      bool,
      Stable::Number*,
      Stable::Number*,
      Stable::Index,
      bool,
      Stable::Number*
   ) override
   {
      if( init_x )
      {
         constexpr std::array<Stable::Number, 4> initial{{0., 2., 0., 0.}};
         std::ranges::copy(initial, x);
      }
      return true;
   }

   bool eval_f(
      Stable::Index,
      const Stable::Number* x,
      bool,
      Stable::Number& objective
   ) override
   {
      objective = x[0] + x[1] + x[2] + x[3];
      return true;
   }

   bool eval_grad_f(
      Stable::Index,
      const Stable::Number*,
      bool            new_x,
      Stable::Number* gradient
   ) override
   {
      last_gradient_new_x = new_x;
      if( fail_gradient )
      {
         gradient[0] = 12345.;
         return false;
      }
      std::fill_n(gradient, 4, 1.);
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
      constraints[1] = x[1] + x[2];
      constraints[2] = x[2] + x[3];
      return true;
   }

   bool eval_jac_g(
      Stable::Index,
      const Stable::Number*,
      bool            new_x,
      Stable::Index,
      Stable::Index,
      Stable::Index* rows,
      Stable::Index* columns,
      Stable::Number* values
   ) override
   {
      if( values == nullptr )
      {
         constexpr std::array<Stable::Index, 6> expected_rows{{0, 0, 1, 1, 2, 2}};
         constexpr std::array<Stable::Index, 6> expected_columns{{0, 1, 1, 2, 2, 3}};
         for( Stable::Index i = 0; i < 6; ++i )
         {
            rows[i] = expected_rows[i] + (fortran_style_ ? 1 : 0);
            columns[i] = expected_columns[i] + (fortran_style_ ? 1 : 0);
         }
      }
      else
      {
         last_jacobian_new_x = new_x;
         std::fill_n(values, 6, 1.);
      }
      return true;
   }

   bool eval_h(
      Stable::Index,
      const Stable::Number*,
      bool            new_x,
      Stable::Number objective_factor,
      Stable::Index,
      const Stable::Number*,
      bool            new_lambda,
      Stable::Index,
      Stable::Index* rows,
      Stable::Index* columns,
      Stable::Number* values
   ) override
   {
      if( values == nullptr )
      {
         constexpr std::array<Stable::Index, 4> diagonal{{0, 1, 2, 3}};
         for( Stable::Index i = 0; i < 4; ++i )
         {
            rows[i] = diagonal[i] + (fortran_style_ ? 1 : 0);
            columns[i] = diagonal[i] + (fortran_style_ ? 1 : 0);
         }
      }
      else
      {
         last_hessian_new_x = new_x;
         last_hessian_new_lambda = new_lambda;
         std::fill_n(values, 4, objective_factor);
      }
      return true;
   }

   bool get_scaling_parameters(
      Stable::Number& objective,
      bool&           use_x_scaling,
      Stable::Index,
      Stable::Number* x_scaling,
      bool&           use_g_scaling,
      Stable::Index,
      Stable::Number* g_scaling
   ) override
   {
      objective = -2.;
      use_x_scaling = true;
      use_g_scaling = true;
      constexpr std::array<Stable::Number, 4> expected_x{{2., 3., 4., 5.}};
      constexpr std::array<Stable::Number, 3> expected_g{{7., 11., 13.}};
      std::ranges::copy(expected_x, x_scaling);
      std::ranges::copy(expected_g, g_scaling);
      return true;
   }

   void finalize_solution(
      Stable::SolverReturn,
      Stable::Index,
      const Stable::Number*,
      const Stable::Number*,
      const Stable::Number*,
      Stable::Index,
      const Stable::Number*,
      const Stable::Number*,
      Stable::Number,
      const Stable::IpoptData*,
      Stable::IpoptCalculatedQuantities*
   ) override
   {
   }

   bool fail_gradient = false;
   bool last_gradient_new_x = false;
   bool last_jacobian_new_x = false;
   bool last_hessian_new_x = false;
   bool last_hessian_new_lambda = false;

private:
   bool fortran_style_;
};

class BridgeUserScaling final : public Stable::StandardScalingBase
{
public:
   explicit BridgeUserScaling(const Stable::SmartPtr<const Stable::NLP>& nlp)
      : nlp_(nlp)
   {
   }

protected:
   void DetermineScalingParametersImpl(
      const Stable::SmartPtr<const Stable::VectorSpace> x_space,
      const Stable::SmartPtr<const Stable::VectorSpace> c_space,
      const Stable::SmartPtr<const Stable::VectorSpace> d_space,
      const Stable::SmartPtr<const Stable::MatrixSpace>,
      const Stable::SmartPtr<const Stable::MatrixSpace>,
      const Stable::SmartPtr<const Stable::SymMatrixSpace>,
      const Stable::Matrix&,
      const Stable::Vector&,
      const Stable::Matrix&,
      const Stable::Vector&,
      Stable::Number& objective,
      Stable::SmartPtr<Stable::Vector>& x_scaling,
      Stable::SmartPtr<Stable::Vector>& c_scaling,
      Stable::SmartPtr<Stable::Vector>& d_scaling
   ) override
   {
      nlp_->GetScalingParameters(
         x_space, c_space, d_space,
         objective, x_scaling, c_scaling, d_scaling);
   }

private:
   Stable::SmartPtr<const Stable::NLP> nlp_;
};

class StableFixture
{
public:
   explicit StableFixture(const std::string& fixed_treatment)
      : application_(new Stable::IpoptApplication(false)),
        tnlp_(new BridgeTnlp()),
        adapter_(new Stable::TNLPAdapter(tnlp_, application_->Jnlst())),
        scaling_(new BridgeUserScaling(Stable::ConstPtr(adapter_))),
        timing_(),
        orig_nlp_(new Stable::OrigIpoptNLP(
           Stable::ConstPtr(application_->Jnlst()), adapter_, scaling_, timing_))
   {
      if( !application_->Options()->SetStringValue(
             "fixed_variable_treatment", fixed_treatment) )
      {
         throw std::runtime_error("failed to set fixed_variable_treatment");
      }
      if( !orig_nlp_->Initialize(
             *application_->Jnlst(), *application_->Options(), "") )
      {
         throw std::runtime_error("OrigIpoptNLP initialization failed");
      }
      if( !orig_nlp_->InitializeStructures(
             x_, true,
             y_c_, false,
             y_d_, false,
             z_lower_, false,
             z_upper_, false,
             v_lower_, v_upper_) )
      {
         throw std::runtime_error("OrigIpoptNLP structure initialization failed");
      }
   }

   Stable::TNLPAdapter& adapter()
   {
      return *adapter_;
   }

   Stable::OrigIpoptNLP& orig_nlp()
   {
      return *orig_nlp_;
   }

   Stable::SmartPtr<Stable::Vector> make_x() const
   {
      return x_->MakeNew();
   }

   Stable::SmartPtr<Stable::Vector> make_y_c() const
   {
      return y_c_->MakeNew();
   }

   Stable::SmartPtr<Stable::Vector> make_y_d() const
   {
      return y_d_->MakeNew();
   }

private:
   Stable::SmartPtr<Stable::IpoptApplication> application_;
   Stable::SmartPtr<BridgeTnlp> tnlp_;
   Stable::SmartPtr<Stable::TNLPAdapter> adapter_;
   Stable::SmartPtr<Stable::NLPScalingObject> scaling_;
   Stable::TimingStatistics timing_;
   Stable::SmartPtr<Stable::OrigIpoptNLP> orig_nlp_;
   Stable::SmartPtr<Stable::Vector> x_;
   Stable::SmartPtr<Stable::Vector> y_c_;
   Stable::SmartPtr<Stable::Vector> y_d_;
   Stable::SmartPtr<Stable::Vector> z_lower_;
   Stable::SmartPtr<Stable::Vector> z_upper_;
   Stable::SmartPtr<Stable::Vector> v_lower_;
   Stable::SmartPtr<Stable::Vector> v_upper_;
};

void Check(bool condition, std::string_view message)
{
   if( !condition )
   {
      throw std::runtime_error(std::string(message));
   }
}

void CheckNear(
   const std::vector<Modern::Number>& actual,
   const std::vector<Modern::Number>& expected,
   std::string_view                   message
)
{
   Check(actual.size() == expected.size(), message);
   for( Modern::Index i = 0; i < actual.size(); ++i )
   {
      if( std::abs(actual[i] - expected[i]) > kStableValueTolerance )
      {
         throw std::runtime_error(
            std::string(message) + " at index " + std::to_string(i));
      }
   }
}

std::vector<Modern::Number> StableVectorValues(const Stable::Vector& vector)
{
   const Modern::Index dimension = static_cast<Modern::Index>(vector.Dim());
   std::vector<Stable::Number> stable_values(dimension);
   Stable::TripletHelper::FillValuesFromVector(
      vector.Dim(), vector, stable_values.data());
   std::vector<Modern::Number> values(dimension);
   for( Modern::Index i = 0; i < dimension; ++i )
   {
      values[i] = static_cast<Modern::Number>(stable_values[i]);
   }
   return values;
}

struct StableTriplet
{
   std::vector<Modern::Index> rows;
   std::vector<Modern::Index> columns;
   std::vector<Modern::Number> values;
};

void AppendStableMatrix(
   const Stable::Matrix& matrix,
   Stable::Index         row_offset,
   StableTriplet&        result
)
{
   const Stable::Index stable_nonzeros =
      Stable::TripletHelper::GetNumberEntries(matrix);
   const Modern::Index nonzeros =
      static_cast<Modern::Index>(stable_nonzeros);
   std::vector<Stable::Index> stable_rows(nonzeros);
   std::vector<Stable::Index> stable_columns(nonzeros);
   std::vector<Stable::Number> stable_values(nonzeros);
   Stable::TripletHelper::FillRowCol(
      stable_nonzeros, matrix,
      stable_rows.data(), stable_columns.data(), row_offset, 0);
   Stable::TripletHelper::FillValues(
      stable_nonzeros, matrix, stable_values.data());
   for( Modern::Index i = 0; i < nonzeros; ++i )
   {
      Check(
         stable_rows[i] > 0 && stable_columns[i] > 0,
         "stable matrix did not use one-based triplet indices");
      result.rows.push_back(static_cast<Modern::Index>(stable_rows[i] - 1));
      result.columns.push_back(static_cast<Modern::Index>(stable_columns[i] - 1));
      result.values.push_back(static_cast<Modern::Number>(stable_values[i]));
   }
}

void CheckMap(
   const Modern::LegacyIpoptBridgeData& bridge,
   const std::vector<Modern::Index>&    variables,
   const std::vector<Modern::Number>&   variable_template,
   const std::vector<Modern::FixedVariableEquality>& fixed_equalities,
   const std::vector<Modern::Number>&   variable_scaling,
   const std::vector<Modern::Number>&   equality_scaling
)
{
   Check(
      bridge.coordinate_map.internal_to_full_variables == variables,
      "variable map mismatch");
   CheckNear(
      bridge.coordinate_map.full_variable_template,
      variable_template,
      "full variable template mismatch");
   Check(
      bridge.coordinate_map.equality_from_full_constraints ==
         std::vector<Modern::Index>({0, 2}),
      "equality map mismatch");
   CheckNear(
      bridge.coordinate_map.equality_rhs,
      {5., -3.},
      "equality RHS mismatch");
   Check(
      bridge.coordinate_map.fixed_variable_equalities == fixed_equalities,
      "fixed equality mismatch");
   Check(
      bridge.coordinate_map.inequality_from_full_constraints ==
         std::vector<Modern::Index>({1}),
      "inequality map mismatch");
   Check(
      std::abs(bridge.scaling.objective + 2.) < kStableValueTolerance,
      "objective scaling mismatch");
   CheckNear(bridge.scaling.variables, variable_scaling, "variable scaling mismatch");
   CheckNear(bridge.scaling.equalities, equality_scaling, "equality scaling mismatch");
   CheckNear(bridge.scaling.inequalities, {11.}, "inequality scaling mismatch");
}

Modern::LegacyIpoptBridgeData Export(StableFixture& fixture)
{
   Modern::EvaluationValue<Modern::LegacyIpoptBridgeData> result =
      Modern::ExportLegacyIpoptBridgeData(
         fixture.adapter(), fixture.orig_nlp());
   Check(result.has_value(), "live legacy bridge export failed");
   return std::move(*result);
}

void TestTnlpBoundary()
{
   Stable::SmartPtr<BridgeTnlp> concrete_tnlp = new BridgeTnlp(true);
   Stable::SmartPtr<Stable::TNLP> tnlp = concrete_tnlp;
   Modern::EvaluationValue<Modern::AnyNlpProblem> problem_result =
      Modern::MakeLegacyTnlpProblem(tnlp, 93);
   Check(problem_result.has_value(), "FORTRAN_STYLE TNLP adapter failed");
   Modern::AnyNlpProblem problem = std::move(*problem_result);
   Check(
      problem.nlp_structure() == Modern::NlpStructure{4, 3, 6, 4, 93},
      "TNLP adapter structure mismatch");

   std::array<Modern::Index, 6> jacobian_rows{};
   std::array<Modern::Index, 6> jacobian_columns{};
   Check(
      problem.nlp_jacobian_structure(
         jacobian_rows, jacobian_columns).has_value(),
      "FORTRAN_STYLE Jacobian conversion failed");
   Check(
      jacobian_rows == std::array<Modern::Index, 6>{{0, 0, 1, 1, 2, 2}} &&
         jacobian_columns ==
            std::array<Modern::Index, 6>{{0, 1, 1, 2, 2, 3}},
      "FORTRAN_STYLE Jacobian conversion is wrong");
   std::array<Modern::Index, 4> hessian_rows{};
   std::array<Modern::Index, 4> hessian_columns{};
   Check(
      problem.nlp_hessian_structure(
         hessian_rows, hessian_columns).has_value(),
      "FORTRAN_STYLE Hessian conversion failed");
   Check(
      hessian_rows == std::array<Modern::Index, 4>{{0, 1, 2, 3}} &&
         hessian_columns == std::array<Modern::Index, 4>{{0, 1, 2, 3}},
      "FORTRAN_STYLE Hessian conversion is wrong");

   const std::array<Modern::Number, 4> x{{1., 2., 3., 4.}};
   std::array<Modern::Number, 4> gradient{};
   Check(
      problem.nlp_gradient(x, gradient).has_value(),
      "TNLP gradient callback failed");
   Check(concrete_tnlp->last_gradient_new_x, "TNLP gradient received stale new_x=false");
   std::array<Modern::Number, 6> jacobian_values{};
   Check(
      problem.nlp_jacobian_values(x, jacobian_values).has_value(),
      "TNLP Jacobian callback failed");
   Check(concrete_tnlp->last_jacobian_new_x, "TNLP Jacobian received stale new_x=false");
   const std::array<Modern::Number, 3> multipliers{{.5, 1., 1.5}};
   std::array<Modern::Number, 4> hessian_values{};
   Check(
      problem.nlp_hessian_values(
         x, 2., multipliers, hessian_values).has_value(),
      "TNLP Hessian callback failed");
   Check(
      concrete_tnlp->last_hessian_new_x &&
         concrete_tnlp->last_hessian_new_lambda,
      "TNLP Hessian received a stale cache flag");

   concrete_tnlp->fail_gradient = true;
   std::array<Modern::Number, 4> untouched{{101., 102., 103., 104.}};
   Modern::EvaluationResult failed = problem.nlp_gradient(x, untouched);
   Check(!failed.has_value(), "TNLP callback failure was ignored");
   Check(
      failed.error().code == Modern::EvaluationErrorCode::model_failure,
      "TNLP callback failure returned the wrong error");
   Check(
      untouched == std::array<Modern::Number, 4>{{101., 102., 103., 104.}},
      "failed TNLP callback modified caller output");
}

void TestAllFixedVariableTreatments()
{
   StableFixture parameter("make_parameter");
   CheckMap(
      Export(parameter),
      {0, 2, 3},
      {0., 2., 0., 0.},
      {},
      {2., 4., 5.},
      {7., 13.});

   StableFixture parameter_no_dual("make_parameter_nodual");
   CheckMap(
      Export(parameter_no_dual),
      {0, 2, 3},
      {0., 2., 0., 0.},
      {},
      {2., 4., 5.},
      {7., 13.});

   StableFixture constraint("make_constraint");
   CheckMap(
      Export(constraint),
      {0, 1, 2, 3},
      {0., 0., 0., 0.},
      {{1, 2.}},
      {2., 3., 4., 5.},
      {7., 13., 1.});

   StableFixture relaxed("relax_bounds");
   CheckMap(
      Export(relaxed),
      {0, 1, 2, 3},
      {0., 0., 0., 0.},
      {},
      {2., 3., 4., 5.},
      {7., 13.});
}

void TestCoordinateFactoryAndOwnershipCheck()
{
   StableFixture fixture("make_parameter");
   Modern::EvaluationValue<Modern::AnyNlpProblem> problem_result =
      Modern::MakeLegacyIpoptCoordinateProblem(
         fixture.adapter(),
         fixture.orig_nlp(),
         71);
   Check(problem_result.has_value(), "live coordinate factory failed");
   Modern::AnyNlpProblem problem = std::move(*problem_result);
   const Modern::NlpStructure structure = problem.nlp_structure();
   Check(
      structure.variables == 3 && structure.constraints == 3 &&
         structure.jacobian_nonzeros == 4 && structure.hessian_nonzeros == 3 &&
         structure.revision != 0,
      "live coordinate factory structure mismatch");

   const std::array<Modern::Number, 3> scaled_x{{4., 8., 10.}};
   Modern::EvaluationValue<Modern::Number> objective =
      problem.nlp_objective(scaled_x);
   Check(
      objective.has_value() &&
         std::abs(*objective + 16.) < kStableValueTolerance,
      "live coordinate objective mismatch");
   std::array<Modern::Number, 3> constraints{};
   Check(
      problem.nlp_constraints(scaled_x, constraints).has_value(),
      "live coordinate constraints failed");
   const std::array<Modern::Number, 3> expected_constraints{{-7., 91., 44.}};
   Check(
      std::ranges::equal(constraints, expected_constraints),
      "live coordinate constraints mismatch");

   Stable::SmartPtr<Stable::Vector> stable_x = fixture.make_x();
   const std::array<Stable::Number, 3> stable_scaled_x{{4., 8., 10.}};
   Stable::TripletHelper::PutValuesInVector(
      stable_x->Dim(), stable_scaled_x.data(), *stable_x);
   const Stable::Number stable_objective = fixture.orig_nlp().f(*stable_x);
   Check(
      std::abs(*objective - static_cast<Modern::Number>(stable_objective)) <
         kStableValueTolerance,
      "modern objective differs from OrigIpoptNLP");
   Stable::SmartPtr<const Stable::Vector> stable_c =
      fixture.orig_nlp().c(*stable_x);
   Stable::SmartPtr<const Stable::Vector> stable_d =
      fixture.orig_nlp().d(*stable_x);
   std::vector<Modern::Number> stable_constraints = StableVectorValues(*stable_c);
   const std::vector<Modern::Number> stable_d_values = StableVectorValues(*stable_d);
   stable_constraints.insert(
      stable_constraints.end(), stable_d_values.begin(), stable_d_values.end());
   CheckNear(
      std::vector<Modern::Number>(constraints.begin(), constraints.end()),
      stable_constraints,
      "modern constraints differ from OrigIpoptNLP");

   std::vector<Modern::Number> gradient(structure.variables);
   Check(
      problem.nlp_gradient(scaled_x, gradient).has_value(),
      "live coordinate gradient failed");
   Stable::SmartPtr<const Stable::Vector> stable_gradient =
      fixture.orig_nlp().grad_f(*stable_x);
   CheckNear(
      gradient,
      StableVectorValues(*stable_gradient),
      "modern gradient differs from OrigIpoptNLP");

   std::vector<Modern::Index> jacobian_rows(structure.jacobian_nonzeros);
   std::vector<Modern::Index> jacobian_columns(structure.jacobian_nonzeros);
   std::vector<Modern::Number> jacobian_values(structure.jacobian_nonzeros);
   Check(
      problem.nlp_jacobian_structure(
         jacobian_rows, jacobian_columns).has_value(),
      "live coordinate Jacobian structure failed");
   Check(
      problem.nlp_jacobian_values(scaled_x, jacobian_values).has_value(),
      "live coordinate Jacobian values failed");
   Stable::SmartPtr<const Stable::Matrix> stable_jacobian_c =
      fixture.orig_nlp().jac_c(*stable_x);
   Stable::SmartPtr<const Stable::Matrix> stable_jacobian_d =
      fixture.orig_nlp().jac_d(*stable_x);
   StableTriplet stable_jacobian;
   AppendStableMatrix(*stable_jacobian_c, 0, stable_jacobian);
   AppendStableMatrix(*stable_jacobian_d, stable_c->Dim(), stable_jacobian);
   Check(jacobian_rows == stable_jacobian.rows, "live Jacobian row order mismatch");
   Check(
      jacobian_columns == stable_jacobian.columns,
      "live Jacobian column order mismatch");
   CheckNear(
      jacobian_values,
      stable_jacobian.values,
      "modern Jacobian differs from OrigIpoptNLP");

   constexpr Modern::Number objective_factor = 1.5;
   const std::array<Modern::Number, 3> multipliers{{.5, -.25, 1.5}};
   std::vector<Modern::Index> hessian_rows(structure.hessian_nonzeros);
   std::vector<Modern::Index> hessian_columns(structure.hessian_nonzeros);
   std::vector<Modern::Number> hessian_values(structure.hessian_nonzeros);
   Check(
      problem.nlp_hessian_structure(
         hessian_rows, hessian_columns).has_value(),
      "live coordinate Hessian structure failed");
   Check(
      problem.nlp_hessian_values(
         scaled_x, objective_factor, multipliers, hessian_values).has_value(),
      "live coordinate Hessian values failed");
   Stable::SmartPtr<Stable::Vector> stable_y_c = fixture.make_y_c();
   Stable::SmartPtr<Stable::Vector> stable_y_d = fixture.make_y_d();
   const std::array<Stable::Number, 2> stable_y_c_values{{.5, -.25}};
   const std::array<Stable::Number, 1> stable_y_d_values{{1.5}};
   Stable::TripletHelper::PutValuesInVector(
      stable_y_c->Dim(), stable_y_c_values.data(), *stable_y_c);
   Stable::TripletHelper::PutValuesInVector(
      stable_y_d->Dim(), stable_y_d_values.data(), *stable_y_d);
   Stable::SmartPtr<const Stable::SymMatrix> stable_hessian =
      fixture.orig_nlp().h(
         *stable_x, objective_factor, *stable_y_c, *stable_y_d);
   StableTriplet stable_hessian_triplet;
   AppendStableMatrix(*stable_hessian, 0, stable_hessian_triplet);
   Check(hessian_rows == stable_hessian_triplet.rows, "live Hessian row order mismatch");
   Check(
      hessian_columns == stable_hessian_triplet.columns,
      "live Hessian column order mismatch");
   CheckNear(
      hessian_values,
      stable_hessian_triplet.values,
      "modern Hessian differs from OrigIpoptNLP");

   StableFixture other("make_parameter");
   Modern::EvaluationValue<Modern::LegacyIpoptBridgeData> mismatch =
      Modern::ExportLegacyIpoptBridgeData(
         fixture.adapter(), other.orig_nlp());
   Check(!mismatch.has_value(), "mismatched stable owner was accepted");
   Check(
      mismatch.error().code == Modern::EvaluationErrorCode::structure_mismatch,
      "mismatched stable owner returned the wrong error");
}
} // namespace

int main()
{
   try
   {
      TestTnlpBoundary();
      TestAllFixedVariableTreatments();
      TestCoordinateFactoryAndOwnershipCheck();
      std::cout << "legacy Ipopt bridge tests passed\n";
      return 0;
   }
   catch( const Stable::IpoptException& error )
   {
      std::cerr << error.Message() << '\n';
      return 1;
   }
   catch( const std::exception& error )
   {
      std::cerr << error.what() << '\n';
      return 1;
   }
}
