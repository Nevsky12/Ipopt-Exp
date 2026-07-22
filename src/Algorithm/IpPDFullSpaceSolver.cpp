// Copyright (C) 2004, 2007 International Business Machines and others.
// All Rights Reserved.
// This code is published under the Eclipse Public License.
//
// Authors:  Carl Laird, Andreas Waechter     IBM    2004-08-13

#include "IpPDFullSpaceSolver.hpp"
#include "IpDebug.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <mutex>
#include <vector>

namespace Ipopt
{

#if IPOPT_VERBOSITY > 0
static const Index dbg_verbosity = 0;
#endif

PDFullSpaceSolverRefinementStatistics::PDFullSpaceSolverRefinementStatistics()
{
   Reset();
}

void PDFullSpaceSolverRefinementStatistics::Reset()
{
   solve_calls = 0;
   direct_solve_calls = 0;
   factorization_attempts = 0;
   refinement_phases = 0;
   refinement_iterations = 0;
   refinement_backsolves = 0;
   kkt_applications = 0;
   converged_refinements = 0;
   failed_refinements = 0;
   quality_improvements = 0;
   singularity_retries = 0;
   preconditioner_cache_misses = 0;
   refinement_wallclock_time = 0.;
   maximum_initial_residual_ratio = 0.;
   maximum_final_residual_ratio = 0.;
   maximum_delta_x = 0.;
   maximum_delta_s = 0.;
   maximum_delta_c = 0.;
   maximum_delta_d = 0.;
   last_initial_residual_ratio = 0.;
   last_final_residual_ratio = 0.;
}

class PDFullSpaceSolverSystem
{
public:
   PDFullSpaceSolverSystem(
      PDFullSpaceSolverData& refinement_data,
      const SymMatrix& W,
      const Matrix&    J_c,
      const Matrix&    J_d,
      const Matrix&    Px_L,
      const Matrix&    Px_U,
      const Matrix&    Pd_L,
      const Matrix&    Pd_U,
      const Vector&    z_L,
      const Vector&    z_U,
      const Vector&    v_L,
      const Vector&    v_U,
      const Vector&    slack_x_L,
      const Vector&    slack_x_U,
      const Vector&    slack_s_L,
      const Vector&    slack_s_U,
      const Vector&    sigma_x,
      const Vector&    sigma_s
   )
      : refinement_data(refinement_data),
        W(W),
        J_c(J_c),
        J_d(J_d),
        Px_L(Px_L),
        Px_U(Px_U),
        Pd_L(Pd_L),
        Pd_U(Pd_U),
        z_L(z_L),
        z_U(z_U),
        v_L(v_L),
        v_U(v_U),
        slack_x_L(slack_x_L),
        slack_x_U(slack_x_U),
        slack_s_L(slack_s_L),
        slack_s_U(slack_s_U),
        sigma_x(sigma_x),
        sigma_s(sigma_s)
   { }

   PDFullSpaceSolverData& refinement_data;
   const SymMatrix& W;
   const Matrix& J_c;
   const Matrix& J_d;
   const Matrix& Px_L;
   const Matrix& Px_U;
   const Matrix& Pd_L;
   const Matrix& Pd_U;
   const Vector& z_L;
   const Vector& z_U;
   const Vector& v_L;
   const Vector& v_U;
   const Vector& slack_x_L;
   const Vector& slack_x_U;
   const Vector& slack_s_L;
   const Vector& slack_s_U;
   const Vector& sigma_x;
   const Vector& sigma_s;
};

class PDFullSpaceSolverData
{
public:
   PDFullSpaceSolverData()
      : method(PDFULLSPACE_ITERATIVE_REFINEMENT),
        fgmres_restart(10),
        fgmres_breakdown_tolerance(
           64. * std::numeric_limits<Number>::epsilon()),
        fgmres_reorthogonalize(true)
   { }

   void PrepareWorkspace(
      const IteratesVector& prototype
   )
   {
      const Index basis_count = fgmres_restart + 1;
      const bool rebuild = basis.size() != static_cast<size_t>(basis_count)
         || basis.empty() || basis[0]->Dim() != prototype.Dim();
      if( rebuild )
      {
         basis.clear();
         preconditioned_basis.clear();
         for( Index i = 0; i < basis_count; ++i )
         {
            basis.push_back(prototype.MakeNewIteratesVector(true));
         }
         for( Index i = 0; i < fgmres_restart; ++i )
         {
            preconditioned_basis.push_back(
               prototype.MakeNewIteratesVector(true));
         }
         base_solution = prototype.MakeNewIteratesVector(true);
         arnoldi_work = prototype.MakeNewIteratesVector(true);
      }
      hessenberg.resize((fgmres_restart + 1) * fgmres_restart);
      givens_cosines.resize(fgmres_restart);
      givens_sines.resize(fgmres_restart);
      projected_rhs.resize(fgmres_restart + 1);
      coefficients.resize(fgmres_restart);
   }

   Number& Hessenberg(
      Index row,
      Index column
   )
   {
      return hessenberg[column * (fgmres_restart + 1) + row];
   }

   void ResetCycle()
   {
      std::fill(hessenberg.begin(), hessenberg.end(), 0.);
      std::fill(givens_cosines.begin(), givens_cosines.end(), 0.);
      std::fill(givens_sines.begin(), givens_sines.end(), 0.);
      std::fill(projected_rhs.begin(), projected_rhs.end(), 0.);
      std::fill(coefficients.begin(), coefficients.end(), 0.);
   }

   PDFullSpaceSolverRefinementMethod method;
   Index fgmres_restart;
   Number fgmres_breakdown_tolerance;
   bool fgmres_reorthogonalize;
   PDFullSpaceSolverRefinementStatistics statistics;
   std::vector<SmartPtr<IteratesVector> > basis;
   std::vector<SmartPtr<IteratesVector> > preconditioned_basis;
   SmartPtr<IteratesVector> base_solution;
   SmartPtr<IteratesVector> arnoldi_work;
   std::vector<Number> hessenberg;
   std::vector<Number> givens_cosines;
   std::vector<Number> givens_sines;
   std::vector<Number> projected_rhs;
   std::vector<Number> coefficients;
};

namespace
{
class PDFullSpaceSolverDataRegistry
{
public:
   ~PDFullSpaceSolverDataRegistry()
   {
      for( std::map<const PDFullSpaceSolver*, PDFullSpaceSolverData*>::iterator
              entry = instances.begin();
           entry != instances.end();
           ++entry )
      {
         delete entry->second;
      }
   }

   std::mutex mutex;
   std::map<const PDFullSpaceSolver*, PDFullSpaceSolverData*> instances;
};

PDFullSpaceSolverDataRegistry& RefinementDataRegistry()
{
   static PDFullSpaceSolverDataRegistry registry;
   return registry;
}

void RegisterRefinementData(
   const PDFullSpaceSolver* solver
)
{
   PDFullSpaceSolverData* data = new PDFullSpaceSolverData();
   PDFullSpaceSolverDataRegistry& registry = RefinementDataRegistry();
   const std::lock_guard<std::mutex> lock(registry.mutex);
   try
   {
      const std::pair<
         std::map<const PDFullSpaceSolver*, PDFullSpaceSolverData*>::iterator,
         bool> inserted = registry.instances.insert(
            std::make_pair(solver, data));
      if( !inserted.second )
      {
         delete inserted.first->second;
         inserted.first->second = data;
      }
   }
   catch( ... )
   {
      delete data;
      throw;
   }
}

void UnregisterRefinementData(
   const PDFullSpaceSolver* solver
)
{
   PDFullSpaceSolverData* data = NULL;
   PDFullSpaceSolverDataRegistry& registry = RefinementDataRegistry();
   {
      const std::lock_guard<std::mutex> lock(registry.mutex);
      std::map<const PDFullSpaceSolver*, PDFullSpaceSolverData*>::iterator found =
         registry.instances.find(solver);
      if( found != registry.instances.end() )
      {
         data = found->second;
         registry.instances.erase(found);
      }
   }
   delete data;
}

PDFullSpaceSolverData& LookupRefinementData(
   const PDFullSpaceSolver* solver
)
{
   PDFullSpaceSolverDataRegistry& registry = RefinementDataRegistry();
   const std::lock_guard<std::mutex> lock(registry.mutex);
   std::map<const PDFullSpaceSolver*, PDFullSpaceSolverData*>::iterator found =
      registry.instances.find(solver);
   DBG_ASSERT(found != registry.instances.end());
   return *found->second;
}

Number StableHypot(
   Number first,
   Number second
)
{
   first = std::abs(first);
   second = std::abs(second);
   const Number scale = Max(first, second);
   if( scale == 0. )
   {
      return 0.;
   }
   first /= scale;
   second /= scale;
   return scale * std::sqrt(first * first + second * second);
}

void RecordPerturbation(
   PDFullSpaceSolverRefinementStatistics& statistics,
   Number delta_x,
   Number delta_s,
   Number delta_c,
   Number delta_d
)
{
   statistics.maximum_delta_x = Max(
      statistics.maximum_delta_x, std::abs(delta_x));
   statistics.maximum_delta_s = Max(
      statistics.maximum_delta_s, std::abs(delta_s));
   statistics.maximum_delta_c = Max(
      statistics.maximum_delta_c, std::abs(delta_c));
   statistics.maximum_delta_d = Max(
      statistics.maximum_delta_d, std::abs(delta_d));
}
} // namespace

PDFullSpaceSolver::PDFullSpaceSolver(
   AugSystemSolver&       augSysSolver,
   PDPerturbationHandler& perturbHandler
)
   : PDSystemSolver(),
     augSysSolver_(&augSysSolver),
     perturbHandler_(&perturbHandler),
     dummy_cache_(1)
{
   DBG_START_METH("PDFullSpaceSolver::PDFullSpaceSolver", dbg_verbosity);
   RegisterRefinementData(this);
}

PDFullSpaceSolver::~PDFullSpaceSolver()
{
   DBG_START_METH("PDFullSpaceSolver::~PDFullSpaceSolver()", dbg_verbosity);
   UnregisterRefinementData(this);
}

PDFullSpaceSolverData& PDFullSpaceSolver::RefinementData() const
{
   return LookupRefinementData(this);
}

const PDFullSpaceSolverRefinementStatistics&
PDFullSpaceSolver::RefinementStatistics() const
{
   return RefinementData().statistics;
}

void PDFullSpaceSolver::ResetRefinementStatistics()
{
   RefinementData().statistics.Reset();
}

PDFullSpaceSolverRefinementMethod PDFullSpaceSolver::RefinementMethod() const
{
   return RefinementData().method;
}

void PDFullSpaceSolver::RegisterOptions(
   SmartPtr<RegisteredOptions> roptions
)
{
   roptions->AddLowerBoundedIntegerOption(
      "min_refinement_steps",
      "Minimum number of iterative refinement steps per linear system solve.",
      0,
      1,
      "Iterative refinement (on the full unsymmetric system) is performed for each right hand side. "
      "This option determines the minimum number of iterative refinements "
      "(i.e. at least \"min_refinement_steps\" iterative refinement steps are enforced per right hand side.)");
   roptions->AddLowerBoundedIntegerOption(
      "max_refinement_steps",
      "Maximum number of iterative refinement steps per linear system solve.",
      0,
      10,
      "Iterative refinement (on the full unsymmetric system) is performed for each right hand side. "
      "This option determines the maximum number of iterative refinement steps.");
   roptions->AddLowerBoundedNumberOption(
      "residual_ratio_max",
      "Iterative refinement tolerance",
      0., true,
      1e-10,
      "Iterative refinement is performed until the residual test ratio is less than this tolerance "
      "(or until \"max_refinement_steps\" refinement steps are performed).",
      true);
   roptions->AddLowerBoundedNumberOption(
      "residual_ratio_singular",
      "Threshold for declaring linear system singular after failed iterative refinement.",
      0., true,
      1e-5,
      "If the residual test ratio is larger than this value after failed iterative refinement, "
      "the algorithm pretends that the linear system is singular.",
      true);
   // ToDo Think about following option - are the correct norms used?
   roptions->AddLowerBoundedNumberOption(
      "residual_improvement_factor",
      "Minimal required reduction of residual test ratio in iterative refinement.",
      0., true,
      0.999999999,
      "If the improvement of the residual test ratio made by one iterative refinement step is not better than this factor, "
      "iterative refinement is aborted.",
      true);
   roptions->AddStringOption3(
      "linear_system_refinement",
      "Method used to refine the direct full-space primal-dual step.",
      "iterative-refinement",
      "iterative-refinement", "use the classic stationary iterative refinement",
      "fgmres-kdelta", "use FGMRES with K_delta as operator and right preconditioner",
      "fgmres-k", "use unperturbed K as operator and factorized K_delta as right preconditioner",
      "Both FGMRES variants start from the direct solution and reuse its current factorization. "
      "They do not perform a second factorization.",
      true);
   roptions->AddLowerBoundedIntegerOption(
      "fgmres_restart",
      "Restart length for full-space FGMRES refinement.",
      1,
      10,
      "The total number of FGMRES iterations is still limited by max_refinement_steps.",
      true);
   roptions->AddLowerBoundedNumberOption(
      "fgmres_breakdown_tolerance",
      "Relative Arnoldi breakdown tolerance for full-space FGMRES refinement.",
      0., true,
      64. * std::numeric_limits<Number>::epsilon(),
      "A happy breakdown is accepted only after checking the full KKT residual.",
      true);
   roptions->AddBoolOption(
      "fgmres_reorthogonalization",
      "Use a second modified Gram-Schmidt pass in full-space FGMRES refinement.",
      true,
      "Reorthogonalization is more robust but requires additional vector dot products and axpy operations.",
      true);
   roptions->AddLowerBoundedNumberOption(
      "neg_curv_test_tol",
      "Tolerance for heuristic to ignore wrong inertia.",
      0.0, false,
      0.0,
      "If nonzero, incorrect inertia in the augmented system is ignored, and "
      "Ipopt tests if the direction is a direction of positive curvature. "
      "This tolerance is alpha_n in the paper by Zavala and Chiang (2014) and "
      "it determines when the direction is considered to be sufficiently positive. "
      "A value in the range of [1e-12, 1e-11] is recommended.");
   roptions->AddStringOption2(
      "neg_curv_test_reg",
      "Whether to do the curvature test with the primal regularization (see Zavala and Chiang, 2014).",
      "yes",
      "yes", "use primal regularization with the inertia-free curvature test",
      "no",  "use original IPOPT approach, in which the primal regularization is ignored");
}

bool PDFullSpaceSolver::InitializeImpl(
   const OptionsList& options,
   const std::string& prefix
)
{
   PDFullSpaceSolverData& refinement_data = RefinementData();

   // Check for the algorithm options
   options.GetIntegerValue("min_refinement_steps", min_refinement_steps_, prefix);
   options.GetIntegerValue("max_refinement_steps", max_refinement_steps_, prefix);
   ASSERT_EXCEPTION(max_refinement_steps_ >= min_refinement_steps_, OPTION_INVALID,
                    "Option \"max_refinement_steps\": This value must be larger than or equal to min_refinement_steps (default 1)");

   options.GetNumericValue("residual_ratio_max", residual_ratio_max_, prefix);
   options.GetNumericValue("residual_ratio_singular", residual_ratio_singular_, prefix);
   ASSERT_EXCEPTION(residual_ratio_singular_ >= residual_ratio_max_, OPTION_INVALID,
                    "Option \"residual_ratio_singular\": This value must be not smaller than residual_ratio_max.");
   options.GetNumericValue("residual_improvement_factor", residual_improvement_factor_, prefix);
   Index refinement_method;
   options.GetEnumValue("linear_system_refinement", refinement_method, prefix);
   refinement_data.method =
      static_cast<PDFullSpaceSolverRefinementMethod>(refinement_method);
   options.GetIntegerValue("fgmres_restart", refinement_data.fgmres_restart, prefix);
   options.GetNumericValue(
      "fgmres_breakdown_tolerance",
      refinement_data.fgmres_breakdown_tolerance,
      prefix);
   options.GetBoolValue(
      "fgmres_reorthogonalization",
      refinement_data.fgmres_reorthogonalize,
      prefix);
   options.GetNumericValue("neg_curv_test_tol", neg_curv_test_tol_, prefix);
   options.GetBoolValue("neg_curv_test_reg", neg_curv_test_reg_, prefix);

   // Reset internal flags and data
   augsys_improved_ = false;
   refinement_data.statistics.Reset();

   if( !augSysSolver_->Initialize(Jnlst(), IpNLP(), IpData(), IpCq(), options, prefix) )
   {
      return false;
   }

   return perturbHandler_->Initialize(Jnlst(), IpNLP(), IpData(), IpCq(), options, prefix);
}

bool PDFullSpaceSolver::Solve(
   Number                alpha,
   Number                beta,
   const IteratesVector& rhs,
   IteratesVector&       res,
   bool                  allow_inexact,
   bool                  improve_solution /* = false */
)
{
   DBG_START_METH("PDFullSpaceSolver::Solve", dbg_verbosity);
   DBG_ASSERT(!allow_inexact || !improve_solution);
   DBG_ASSERT(!improve_solution || beta == 0.);

   // Timing of PDSystem solver starts here
   IpData().TimingStats().PDSystemSolverTotal().Start();

   DBG_PRINT_VECTOR(2, "rhs_x", *rhs.x());
   DBG_PRINT_VECTOR(2, "rhs_s", *rhs.s());
   DBG_PRINT_VECTOR(2, "rhs_c", *rhs.y_c());
   DBG_PRINT_VECTOR(2, "rhs_d", *rhs.y_d());
   DBG_PRINT_VECTOR(2, "rhs_zL", *rhs.z_L());
   DBG_PRINT_VECTOR(2, "rhs_zU", *rhs.z_U());
   DBG_PRINT_VECTOR(2, "rhs_vL", *rhs.v_L());
   DBG_PRINT_VECTOR(2, "rhs_vU", *rhs.v_U());
   DBG_PRINT_VECTOR(2, "res_x in", *res.x());
   DBG_PRINT_VECTOR(2, "res_s in", *res.s());
   DBG_PRINT_VECTOR(2, "res_c in", *res.y_c());
   DBG_PRINT_VECTOR(2, "res_d in", *res.y_d());
   DBG_PRINT_VECTOR(2, "res_zL in", *res.z_L());
   DBG_PRINT_VECTOR(2, "res_zU in", *res.z_U());
   DBG_PRINT_VECTOR(2, "res_vL in", *res.v_L());
   DBG_PRINT_VECTOR(2, "res_vU in", *res.v_U());

   // if beta is nonzero, keep a copy of the incoming values in res_ */
   SmartPtr<IteratesVector> copy_res;
   if( beta != 0. )
   {
      copy_res = res.MakeNewIteratesVectorCopy();
   }

   // Receive data about matrix
//   SmartPtr<const Vector> x = IpData().curr()->x();
//   SmartPtr<const Vector> s = IpData().curr()->s();
   SmartPtr<const SymMatrix> W = IpData().W();
   SmartPtr<const Matrix> J_c = IpCq().curr_jac_c();
   SmartPtr<const Matrix> J_d = IpCq().curr_jac_d();
   SmartPtr<const Matrix> Px_L = IpNLP().Px_L();
   SmartPtr<const Matrix> Px_U = IpNLP().Px_U();
   SmartPtr<const Matrix> Pd_L = IpNLP().Pd_L();
   SmartPtr<const Matrix> Pd_U = IpNLP().Pd_U();
   SmartPtr<const Vector> z_L = IpData().curr()->z_L();
   SmartPtr<const Vector> z_U = IpData().curr()->z_U();
   SmartPtr<const Vector> v_L = IpData().curr()->v_L();
   SmartPtr<const Vector> v_U = IpData().curr()->v_U();
   SmartPtr<const Vector> slack_x_L = IpCq().curr_slack_x_L();
   SmartPtr<const Vector> slack_x_U = IpCq().curr_slack_x_U();
   SmartPtr<const Vector> slack_s_L = IpCq().curr_slack_s_L();
   SmartPtr<const Vector> slack_s_U = IpCq().curr_slack_s_U();
   SmartPtr<const Vector> sigma_x = IpCq().curr_sigma_x();
   SmartPtr<const Vector> sigma_s = IpCq().curr_sigma_s();
   DBG_PRINT_VECTOR(2, "Sigma_x", *sigma_x);
   DBG_PRINT_VECTOR(2, "Sigma_s", *sigma_s);

   PDFullSpaceSolverData& refinement_data = RefinementData();
   const PDFullSpaceSolverSystem system(
      refinement_data,
      *W, *J_c, *J_d, *Px_L, *Px_U, *Pd_L, *Pd_U,
      *z_L, *z_U, *v_L, *v_U,
      *slack_x_L, *slack_x_U, *slack_s_L, *slack_s_U,
      *sigma_x, *sigma_s);
   const bool include_perturbation =
      refinement_data.method != PDFULLSPACE_FGMRES_K;
   ++refinement_data.statistics.solve_calls;

   bool done = false;
   // The following flag is set to true, if we asked the linear
   // solver to improve the quality of the solution in
   // the next solve
   bool resolve_with_better_quality = false;
   // the following flag is set to true, if iterative refinement
   // failed and we want to try if a modified system is able to
   // remedy that problem by pretending the matrix is singular
   bool pretend_singular = false;
   bool pretend_singular_last_time = false;

   // Beginning of loop for solving the system (including all
   // modifications for the linear system to ensure good solution
   // quality)
   while( !done )
   {

      // if improve_solution is true, we are given already a solution
      // from the calling function, so we can skip the first solve
      bool solve_retval = true;
      if( !improve_solution )
      {
         ++refinement_data.statistics.direct_solve_calls;
         solve_retval = SolveOnce(resolve_with_better_quality, pretend_singular, *W, *J_c, *J_d, *Px_L, *Px_U, *Pd_L,
                                  *Pd_U, *z_L, *z_U, *v_L, *v_U, *slack_x_L, *slack_x_U, *slack_s_L, *slack_s_U, *sigma_x, *sigma_s, 1., 0.,
                                  rhs, res);
         resolve_with_better_quality = false;
         pretend_singular = false;
      }
      improve_solution = false;

      if( !solve_retval )
      {
         // If system seems not to be solvable, we return with false
         // and let the calling routine deal with it.
         IpData().TimingStats().PDSystemSolverTotal().End();
         return false;
      }

      if( allow_inexact )
      {
         // no safety checks required
         if( Jnlst().ProduceOutput(J_MOREDETAILED, J_LINEAR_ALGEBRA) )
         {
            SmartPtr<IteratesVector> resid = res.MakeNewIteratesVector(true);
            ComputeResiduals(
               system, include_perturbation, rhs, res, *resid);
         }
         break;
      }

      // Get space for the residual
      SmartPtr<IteratesVector> resid = res.MakeNewIteratesVector(true);

      const Number refinement_start = WallclockTime();
      ++refinement_data.statistics.refinement_phases;

      // ToDo don't to that after max refinement?
      ComputeResiduals(system, include_perturbation, rhs, res, *resid);

      Number residual_ratio = ComputeResidualRatio(rhs, res, *resid);
      Jnlst().Printf(J_DETAILED, J_LINEAR_ALGEBRA,
                     "residual_ratio = %e\n", residual_ratio);
      refinement_data.statistics.last_initial_residual_ratio = residual_ratio;
      refinement_data.statistics.maximum_initial_residual_ratio = Max(
         refinement_data.statistics.maximum_initial_residual_ratio,
         residual_ratio);
      Number residual_ratio_old = residual_ratio;

      Index num_iter_ref = 0;
      bool refinement_failed = false;
      if( refinement_data.method == PDFULLSPACE_ITERATIVE_REFINEMENT )
      {
         while( !refinement_failed
                && (num_iter_ref < min_refinement_steps_
                    || residual_ratio > residual_ratio_max_) )
         {
            ++refinement_data.statistics.refinement_backsolves;
            solve_retval = SolveOnce(
               resolve_with_better_quality, false,
               *W, *J_c, *J_d, *Px_L, *Px_U, *Pd_L, *Pd_U,
               *z_L, *z_U, *v_L, *v_U,
               *slack_x_L, *slack_x_U, *slack_s_L, *slack_s_U,
               *sigma_x, *sigma_s, -1., 1., *resid, res);
            ASSERT_EXCEPTION(
               solve_retval,
               INTERNAL_ABORT,
               "SolveOnce returns false during iterative refinement.");

            ComputeResiduals(
               system, include_perturbation, rhs, res, *resid);
            residual_ratio = ComputeResidualRatio(rhs, res, *resid);
            Jnlst().Printf(
               J_DETAILED, J_LINEAR_ALGEBRA,
               "residual_ratio = %e\n", residual_ratio);

            ++num_iter_ref;
            refinement_failed = residual_ratio > residual_ratio_max_
               && num_iter_ref > min_refinement_steps_
               && (num_iter_ref > max_refinement_steps_
                   || residual_ratio
                      > residual_improvement_factor_ * residual_ratio_old);
            residual_ratio_old = residual_ratio;
         }
      }
      else
      {
         solve_retval = SolveWithFgmres(
            system,
            rhs,
            res,
            *resid,
            residual_ratio,
            num_iter_ref,
            refinement_failed);
         ASSERT_EXCEPTION(
            solve_retval,
            INTERNAL_ABORT,
            "Current K_delta factorization could not apply the FGMRES preconditioner.");
      }

      refinement_data.statistics.refinement_iterations += num_iter_ref;
      refinement_data.statistics.last_final_residual_ratio = residual_ratio;
      refinement_data.statistics.maximum_final_residual_ratio = Max(
         refinement_data.statistics.maximum_final_residual_ratio,
         residual_ratio);
      refinement_data.statistics.refinement_wallclock_time +=
         WallclockTime() - refinement_start;

      if( !refinement_failed )
      {
         ++refinement_data.statistics.converged_refinements;
      }
      else
      {
         ++refinement_data.statistics.failed_refinements;
         Jnlst().Printf(
            J_DETAILED, J_LINEAR_ALGEBRA,
            "%s refinement failed with residual_ratio = %e\n",
            refinement_data.method == PDFULLSPACE_ITERATIVE_REFINEMENT
               ? "Iterative"
               : "FGMRES",
            residual_ratio);

         // Pretend singularity only once - if it didn't help, we
         // have to live with what we got so far
         resolve_with_better_quality = false;
         DBG_PRINT((1, "pretend_singular = %d\n", pretend_singular));
         if( !pretend_singular_last_time )
         {
            // First try if we can ask the augmented system solver to
            // improve the quality of the solution (only if that hasn't
            // been done before for this linear system)
            if( !augsys_improved_ )
            {
               Jnlst().Printf(J_DETAILED, J_LINEAR_ALGEBRA,
                              "Asking augmented system solver to improve quality of its solutions.\n");
               augsys_improved_ = augSysSolver_->IncreaseQuality();
               if( augsys_improved_ )
               {
                  ++refinement_data.statistics.quality_improvements;
                  IpData().Append_info_string("q");
                  resolve_with_better_quality = true;
               }
               else
               {
                  // solver said it cannot improve quality, so let
                  // possibly conclude that the current modification is
                  // singular
                  pretend_singular = true;
               }
            }
            else
            {
               // we had already asked the solver before to improve the
               // quality of the solution, so let's now pretend that the
               // modification is possibly singular
               pretend_singular = true;
            }
            pretend_singular_last_time = pretend_singular;
            if( pretend_singular )
            {
               // let's only conclude that the current linear system
               // including modifications is singular, if the residual is
               // quite bad
               if( residual_ratio < residual_ratio_singular_ )
               {
                  pretend_singular = false;
                  IpData().Append_info_string("S");
                  Jnlst().Printf(J_DETAILED, J_LINEAR_ALGEBRA,
                                 "Just accept current solution.\n");
               }
               else
               {
                  ++refinement_data.statistics.singularity_retries;
                  IpData().Append_info_string("s");
                  Jnlst().Printf(J_DETAILED, J_LINEAR_ALGEBRA,
                                 "Pretend that the current system (including modifications) is singular.\n");
               }
            }
         }
         else
         {
            pretend_singular = false;
            DBG_PRINT((1, "Resetting pretend_singular to false.\n"));
         }
      }

      done = !(resolve_with_better_quality) && !(pretend_singular);

   } // End of loop for solving the linear system (incl. modifications)

   // Finally let's assemble the res result vectors
   if( alpha != 0. )
   {
      res.Scal(alpha);
   }

   if( beta != 0. )
   {
      res.Axpy(beta, *copy_res);
   }

   DBG_PRINT_VECTOR(2, "res_x", *res.x());
   DBG_PRINT_VECTOR(2, "res_s", *res.s());
   DBG_PRINT_VECTOR(2, "res_c", *res.y_c());
   DBG_PRINT_VECTOR(2, "res_d", *res.y_d());
   DBG_PRINT_VECTOR(2, "res_zL", *res.z_L());
   DBG_PRINT_VECTOR(2, "res_zU", *res.z_U());
   DBG_PRINT_VECTOR(2, "res_vL", *res.v_L());
   DBG_PRINT_VECTOR(2, "res_vU", *res.v_U());

   IpData().TimingStats().PDSystemSolverTotal().End();

   return true;
}

bool PDFullSpaceSolver::SolveOnce(
   bool                  resolve_with_better_quality,
   bool                  pretend_singular,
   const SymMatrix&      W,
   const Matrix&         J_c,
   const Matrix&         J_d,
   const Matrix&         Px_L,
   const Matrix&         Px_U,
   const Matrix&         Pd_L,
   const Matrix&         Pd_U,
   const Vector&         z_L,
   const Vector&         z_U,
   const Vector&         v_L,
   const Vector&         v_U,
   const Vector&         slack_x_L,
   const Vector&         slack_x_U,
   const Vector&         slack_s_L,
   const Vector&         slack_s_U,
   const Vector&         sigma_x,
   const Vector&         sigma_s,
   Number                alpha,
   Number                beta,
   const IteratesVector& rhs,
   IteratesVector&       res
)
{
   // TO DO LIST:
   //
   // 1. decide for reasonable return codes (e.g. fatal error, too
   //    ill-conditioned...)
   // 2. Make constants parameters that can be set from the outside
   // 3. Get Information out of Ipopt structures
   // 4. add heuristic for structurally singular problems
   // 5. see if it makes sense to distinguish delta_x and delta_s,
   //    or delta_c and delta_d
   // 6. increase pivot tolerance if number of get evals so too small
   DBG_START_METH("PDFullSpaceSolver::SolveOnce", dbg_verbosity);
   PDFullSpaceSolverData& refinement_data = RefinementData();

   IpData().TimingStats().PDSystemSolverSolveOnce().Start();

   // Compute the right hand side for the augmented system formulation
   SmartPtr<Vector> augRhs_x = rhs.x()->MakeNewCopy();
   Px_L.AddMSinvZ(1.0, slack_x_L, *rhs.z_L(), *augRhs_x);
   Px_U.AddMSinvZ(-1.0, slack_x_U, *rhs.z_U(), *augRhs_x);

   SmartPtr<Vector> augRhs_s = rhs.s()->MakeNewCopy();
   Pd_L.AddMSinvZ(1.0, slack_s_L, *rhs.v_L(), *augRhs_s);
   Pd_U.AddMSinvZ(-1.0, slack_s_U, *rhs.v_U(), *augRhs_s);

   // Get space into which we can put the solution of the augmented system
   SmartPtr<IteratesVector> sol = res.MakeNewIteratesVector(true);

   // Now check whether any data has changed
   std::vector<const TaggedObject*> deps(13);
   deps[0] = &W;
   deps[1] = &J_c;
   deps[2] = &J_d;
   deps[3] = &z_L;
   deps[4] = &z_U;
   deps[5] = &v_L;
   deps[6] = &v_U;
   deps[7] = &slack_x_L;
   deps[8] = &slack_x_U;
   deps[9] = &slack_s_L;
   deps[10] = &slack_s_U;
   deps[11] = &sigma_x;
   deps[12] = &sigma_s;
   void* dummy = NULL;
   bool uptodate = dummy_cache_.GetCachedResult(dummy, deps);
   if( !uptodate )
   {
      dummy_cache_.AddCachedResult(dummy, deps);
      augsys_improved_ = false;
   }
   // improve_current_solution can only be true, if that system has
   // been solved before
   DBG_ASSERT((!resolve_with_better_quality && !pretend_singular) || uptodate);
   (void) resolve_with_better_quality;

   ESymSolverStatus retval;
   if( uptodate && !pretend_singular )
   {

      // Get the perturbation values
      Number delta_x;
      Number delta_s;
      Number delta_c;
      Number delta_d;
      perturbHandler_->CurrentPerturbation(delta_x, delta_s, delta_c, delta_d);
      RecordPerturbation(
         refinement_data.statistics,
         delta_x, delta_s, delta_c, delta_d);

      // No need to go through the pain of finding the appropriate
      // values for the deltas, because the matrix hasn't changed since
      // the last call.  So, just call the Solve Method
      //
      // Note: resolve_with_better_quality is true, then the Solve
      // method has already asked the augSysSolver to increase the
      // quality at the end solve, and we are now getting the solution
      // with that better quality
      retval = augSysSolver_->Solve(&W, 1.0, &sigma_x, delta_x, &sigma_s, delta_s, &J_c, NULL, delta_c, &J_d, NULL,
                                    delta_d, *augRhs_x, *augRhs_s, *rhs.y_c(), *rhs.y_d(), *sol->x_NonConst(), *sol->s_NonConst(),
                                    *sol->y_c_NonConst(), *sol->y_d_NonConst(), false, 0);
      if( retval != SYMSOLVER_SUCCESS )
      {
         IpData().TimingStats().PDSystemSolverSolveOnce().End();
         return false;
      }
   }
   else
   {
      const Index numberOfEVals = rhs.y_c()->Dim() + rhs.y_d()->Dim();
      // counter for the number of trial evaluations
      // ToDo is not at the correct place
      Index count = 0;

      // Get the very first perturbation values from the perturbation
      // Handler
      Number delta_x;
      Number delta_s;
      Number delta_c;
      Number delta_d;
      perturbHandler_->ConsiderNewSystem(delta_x, delta_s, delta_c, delta_d);

      retval = SYMSOLVER_SINGULAR;
      while( retval != SYMSOLVER_SUCCESS )
      {

         if( pretend_singular )
         {
            retval = SYMSOLVER_SINGULAR;
            pretend_singular = false;
         }
         else
         {
            count++;
            ++refinement_data.statistics.factorization_attempts;
            Jnlst().Printf(J_MOREDETAILED, J_LINEAR_ALGEBRA,
                           "Solving system with delta_x=%e delta_s=%e\n                    delta_c=%e delta_d=%e\n", delta_x,
                           delta_s, delta_c, delta_d);
            bool check_inertia = true;
            if( neg_curv_test_tol_ > 0. )
            {
               check_inertia = false;
            }
            retval = augSysSolver_->Solve(&W, 1.0, &sigma_x, delta_x, &sigma_s, delta_s, &J_c, NULL, delta_c, &J_d,
                                          NULL, delta_d, *augRhs_x, *augRhs_s, *rhs.y_c(), *rhs.y_d(), *sol->x_NonConst(), *sol->s_NonConst(),
                                          *sol->y_c_NonConst(), *sol->y_d_NonConst(), check_inertia, numberOfEVals);
         }
         if( retval == SYMSOLVER_FATAL_ERROR )
         {
            return false;
         }
         if( retval == SYMSOLVER_SINGULAR && (rhs.y_c()->Dim() + rhs.y_d()->Dim() > 0) )
         {

            // Get new perturbation factors from the perturbation
            // handlers for the singular case
            bool pert_return = perturbHandler_->PerturbForSingularity(delta_x, delta_s, delta_c, delta_d);
            if( !pert_return )
            {
               Jnlst().Printf(J_DETAILED, J_LINEAR_ALGEBRA,
                              "PerturbForSingularity can't be done\n");
               IpData().TimingStats().PDSystemSolverSolveOnce().End();
               return false;
            }
         }
         else if( retval == SYMSOLVER_WRONG_INERTIA && augSysSolver_->NumberOfNegEVals() < numberOfEVals )
         {
            Jnlst().Printf(J_DETAILED, J_LINEAR_ALGEBRA,
                           "Number of negative eigenvalues too small!\n");
            // If the number of negative eigenvalues is too small, then
            // we first try to remedy this by asking for better quality
            // solution (e.g. increasing pivot tolerance), and if that
            // doesn't help, we assume that the system is singular
            bool assume_singular = true;
            if( !augsys_improved_ )
            {
               Jnlst().Printf(J_DETAILED, J_LINEAR_ALGEBRA,
                              "Asking augmented system solver to improve quality of its solutions.\n");
               augsys_improved_ = augSysSolver_->IncreaseQuality();
               if( augsys_improved_ )
               {
                  IpData().Append_info_string("q");
                  assume_singular = false;
               }
               else
               {
                  Jnlst().Printf(J_DETAILED, J_LINEAR_ALGEBRA,
                                 "Quality could not be improved\n");
               }
            }
            if( assume_singular )
            {
               bool pert_return = perturbHandler_->PerturbForSingularity(delta_x, delta_s, delta_c, delta_d);
               if( !pert_return )
               {
                  Jnlst().Printf(J_DETAILED, J_LINEAR_ALGEBRA,
                                 "PerturbForSingularity can't be done for assume singular.\n");
                  IpData().TimingStats().PDSystemSolverSolveOnce().End();
                  return false;
               }
               IpData().Append_info_string("a");
            }
         }
         else if( retval == SYMSOLVER_WRONG_INERTIA || retval == SYMSOLVER_SINGULAR )
         {
            // Get new perturbation factors from the perturbation
            // handlers for the case of wrong inertia
            bool pert_return = perturbHandler_->PerturbForWrongInertia(delta_x, delta_s, delta_c, delta_d);
            if( !pert_return )
            {
               Jnlst().Printf(J_DETAILED, J_LINEAR_ALGEBRA,
                              "PerturbForWrongInertia can't be done for wrong interia or singular.\n");
               IpData().TimingStats().PDSystemSolverSolveOnce().End();
               return false;
            }
         }
         else if (neg_curv_test_tol_ > 0.)
         {
            DBG_ASSERT(augSysSolver_->ProvidesInertia());
            // we now check if the inertia is possible wrong
            Index neg_values = augSysSolver_->NumberOfNegEVals();
            if (neg_values != numberOfEVals)
            {
               // check if we have a direction of sufficient positive curvature
               SmartPtr<Vector> x_tmp = sol->x()->MakeNew();
               W.MultVector(1., *sol->x(), 0., *x_tmp);
               Number xWx = x_tmp->Dot(*sol->x());
               x_tmp->Copy(*sol->x());
               x_tmp->ElementWiseMultiply(sigma_x);
               xWx += x_tmp->Dot(*sol->x());
               SmartPtr<Vector> s_tmp = sol->s()->MakeNewCopy();
               s_tmp->ElementWiseMultiply(sigma_s);
               xWx += s_tmp->Dot(*sol->s());
               if (neg_curv_test_reg_)
               {
                  x_tmp->Copy(*sol->x());
                  x_tmp->Scal(delta_x);
                  xWx += x_tmp->Dot(*sol->x());

                  s_tmp->Copy(*sol->s());
                  s_tmp->Scal(delta_s);
                  xWx += s_tmp->Dot(*sol->s());
               }
               Number xs_nrmsq = std::pow(sol->x()->Nrm2(), 2) + std::pow(sol->s()->Nrm2(), 2);
               Jnlst().Printf(J_DETAILED, J_LINEAR_ALGEBRA,
                              "In inertia heuristic: xWx = %e xx = %e\n",
                              xWx, xs_nrmsq);
               if (xWx < neg_curv_test_tol_ * xs_nrmsq)
               {
                  Jnlst().Printf(J_DETAILED, J_LINEAR_ALGEBRA,
                                 "    -> Redo with modified matrix.\n");
                  bool pert_return = perturbHandler_->PerturbForWrongInertia(delta_x, delta_s,
                                     delta_c, delta_d);
                  if (!pert_return)
                  {
                     Jnlst().Printf(J_DETAILED, J_LINEAR_ALGEBRA,
                                    "PerturbForWrongInertia can't be done for inertia heuristic.\n");
                     IpData().TimingStats().PDSystemSolverSolveOnce().End();
                     return false;
                  }
                  retval = SYMSOLVER_WRONG_INERTIA;
               }
            }
         }
      } // while (retval!=SYMSOLVER_SUCCESS)

      // Some output
      Jnlst().Printf(J_DETAILED, J_LINEAR_ALGEBRA,
                     "Number of trial factorizations performed: %" IPOPT_INDEX_FORMAT "\n", count);
      Jnlst().Printf(J_DETAILED, J_LINEAR_ALGEBRA,
                     "Perturbation parameters: delta_x=%e delta_s=%e\n                         delta_c=%e delta_d=%e\n", delta_x,
                     delta_s, delta_c, delta_d);
      // Set the perturbation values in the Data object
      IpData().setPDPert(delta_x, delta_s, delta_c, delta_d);
   }

   // Compute the remaining sol Vectors
   Px_L.SinvBlrmZMTdBr(-1., slack_x_L, *rhs.z_L(), z_L, *sol->x(), *sol->z_L_NonConst());
   Px_U.SinvBlrmZMTdBr(1., slack_x_U, *rhs.z_U(), z_U, *sol->x(), *sol->z_U_NonConst());
   Pd_L.SinvBlrmZMTdBr(-1., slack_s_L, *rhs.v_L(), v_L, *sol->s(), *sol->v_L_NonConst());
   Pd_U.SinvBlrmZMTdBr(1., slack_s_U, *rhs.v_U(), v_U, *sol->s(), *sol->v_U_NonConst());

   // Finally let's assemble the res result vectors
   res.AddOneVector(alpha, *sol, beta);

   IpData().TimingStats().PDSystemSolverSolveOnce().End();

   return true;
}

bool PDFullSpaceSolver::SolveWithCurrentFactorization(
   const PDFullSpaceSolverSystem& system,
   const IteratesVector&          rhs,
   IteratesVector&                res
)
{
   IpData().TimingStats().PDSystemSolverSolveOnce().Start();

   std::vector<const TaggedObject*> deps(13);
   deps[0] = &system.W;
   deps[1] = &system.J_c;
   deps[2] = &system.J_d;
   deps[3] = &system.z_L;
   deps[4] = &system.z_U;
   deps[5] = &system.v_L;
   deps[6] = &system.v_U;
   deps[7] = &system.slack_x_L;
   deps[8] = &system.slack_x_U;
   deps[9] = &system.slack_s_L;
   deps[10] = &system.slack_s_U;
   deps[11] = &system.sigma_x;
   deps[12] = &system.sigma_s;
   void* dummy = NULL;
   if( !dummy_cache_.GetCachedResult(dummy, deps) )
   {
      ++system.refinement_data.statistics.preconditioner_cache_misses;
      IpData().TimingStats().PDSystemSolverSolveOnce().End();
      return false;
   }

   SmartPtr<Vector> augRhs_x = rhs.x()->MakeNewCopy();
   system.Px_L.AddMSinvZ(
      1.0, system.slack_x_L, *rhs.z_L(), *augRhs_x);
   system.Px_U.AddMSinvZ(
      -1.0, system.slack_x_U, *rhs.z_U(), *augRhs_x);

   SmartPtr<Vector> augRhs_s = rhs.s()->MakeNewCopy();
   system.Pd_L.AddMSinvZ(
      1.0, system.slack_s_L, *rhs.v_L(), *augRhs_s);
   system.Pd_U.AddMSinvZ(
      -1.0, system.slack_s_U, *rhs.v_U(), *augRhs_s);

   Number delta_x;
   Number delta_s;
   Number delta_c;
   Number delta_d;
   perturbHandler_->CurrentPerturbation(
      delta_x, delta_s, delta_c, delta_d);
   RecordPerturbation(
      system.refinement_data.statistics,
      delta_x, delta_s, delta_c, delta_d);

   ++system.refinement_data.statistics.refinement_backsolves;
   const ESymSolverStatus retval = augSysSolver_->Solve(
      &system.W, 1.0,
      &system.sigma_x, delta_x,
      &system.sigma_s, delta_s,
      &system.J_c, NULL, delta_c,
      &system.J_d, NULL, delta_d,
      *augRhs_x, *augRhs_s, *rhs.y_c(), *rhs.y_d(),
      *res.x_NonConst(), *res.s_NonConst(),
      *res.y_c_NonConst(), *res.y_d_NonConst(),
      false, 0);
   if( retval != SYMSOLVER_SUCCESS )
   {
      IpData().TimingStats().PDSystemSolverSolveOnce().End();
      return false;
   }

   system.Px_L.SinvBlrmZMTdBr(
      -1., system.slack_x_L, *rhs.z_L(), system.z_L,
      *res.x(), *res.z_L_NonConst());
   system.Px_U.SinvBlrmZMTdBr(
      1., system.slack_x_U, *rhs.z_U(), system.z_U,
      *res.x(), *res.z_U_NonConst());
   system.Pd_L.SinvBlrmZMTdBr(
      -1., system.slack_s_L, *rhs.v_L(), system.v_L,
      *res.s(), *res.v_L_NonConst());
   system.Pd_U.SinvBlrmZMTdBr(
      1., system.slack_s_U, *rhs.v_U(), system.v_U,
      *res.s(), *res.v_U_NonConst());

   IpData().TimingStats().PDSystemSolverSolveOnce().End();
   return true;
}

bool PDFullSpaceSolver::SolveWithFgmres(
   const PDFullSpaceSolverSystem& system,
   const IteratesVector&          rhs,
   IteratesVector&                res,
   IteratesVector&                resid,
   Number&                        residual_ratio,
   Index&                         num_iterations,
   bool&                          refinement_failed
)
{
   num_iterations = 0;
   refinement_failed = false;
   if( min_refinement_steps_ == 0
       && residual_ratio <= residual_ratio_max_ )
   {
      return true;
   }

   PDFullSpaceSolverData& data = system.refinement_data;
   data.PrepareWorkspace(res);
   const bool include_perturbation =
      data.method == PDFULLSPACE_FGMRES_KDELTA;
   const Index iteration_limit = Max(
      min_refinement_steps_, max_refinement_steps_ + 1);
   Number residual_ratio_old = residual_ratio;

   while( num_iterations < iteration_limit )
   {
      const Number residual_norm = resid.Nrm2();
      if( !IsFiniteNumber(residual_norm) )
      {
         refinement_failed = true;
         return true;
      }
      if( residual_norm == 0. )
      {
         return true;
      }

      data.base_solution->Copy(res);
      data.ResetCycle();
      data.basis[0]->Copy(resid);
      data.basis[0]->Scal(-1. / residual_norm);
      data.projected_rhs[0] = residual_norm;

      const Index remaining = iteration_limit - num_iterations;
      const Index cycle_steps = Min(data.fgmres_restart, remaining);
      for( Index step = 0; step < cycle_steps; ++step )
      {
         if( !SolveWithCurrentFactorization(
                system,
                *data.basis[step],
                *data.preconditioned_basis[step]) )
         {
            return false;
         }

         ApplyKKT(
            system,
            include_perturbation,
            *data.preconditioned_basis[step],
            *data.arnoldi_work);
         ++num_iterations;

         const Number unorthogonalized_norm = data.arnoldi_work->Nrm2();
         if( !IsFiniteNumber(unorthogonalized_norm) )
         {
            refinement_failed = true;
            return true;
         }

         const Index orthogonalization_passes =
            data.fgmres_reorthogonalize ? 2 : 1;
         for( Index pass = 0; pass < orthogonalization_passes; ++pass )
         {
            for( Index basis_index = 0;
                 basis_index <= step;
                 ++basis_index )
            {
               const Number projection = data.arnoldi_work->Dot(
                  *data.basis[basis_index]);
               data.Hessenberg(basis_index, step) += projection;
               data.arnoldi_work->Axpy(
                  -projection, *data.basis[basis_index]);
            }
         }

         const Number next_basis_norm = data.arnoldi_work->Nrm2();
         if( !IsFiniteNumber(next_basis_norm) )
         {
            refinement_failed = true;
            return true;
         }
         data.Hessenberg(step + 1, step) = next_basis_norm;
         const bool happy_breakdown = next_basis_norm
            <= data.fgmres_breakdown_tolerance * unorthogonalized_norm;
         if( !happy_breakdown )
         {
            data.basis[step + 1]->Copy(*data.arnoldi_work);
            data.basis[step + 1]->Scal(1. / next_basis_norm);
         }

         for( Index row = 0; row < step; ++row )
         {
            const Number upper = data.Hessenberg(row, step);
            const Number lower = data.Hessenberg(row + 1, step);
            data.Hessenberg(row, step) =
               data.givens_cosines[row] * upper
               + data.givens_sines[row] * lower;
            data.Hessenberg(row + 1, step) =
               -data.givens_sines[row] * upper
               + data.givens_cosines[row] * lower;
         }

         const Number diagonal = data.Hessenberg(step, step);
         const Number subdiagonal = data.Hessenberg(step + 1, step);
         const Number rotation_norm = StableHypot(diagonal, subdiagonal);
         const Number rotation_scale = Max(
            unorthogonalized_norm,
            std::numeric_limits<Number>::min());
         if( !IsFiniteNumber(rotation_norm)
             || rotation_norm
                <= data.fgmres_breakdown_tolerance * rotation_scale )
         {
            refinement_failed = residual_ratio > residual_ratio_max_;
            return true;
         }

         data.givens_cosines[step] = diagonal / rotation_norm;
         data.givens_sines[step] = subdiagonal / rotation_norm;
         data.Hessenberg(step, step) = rotation_norm;
         data.Hessenberg(step + 1, step) = 0.;
         data.projected_rhs[step + 1] =
            -data.givens_sines[step] * data.projected_rhs[step];
         data.projected_rhs[step] *= data.givens_cosines[step];

         bool candidate_valid = true;
         for( Index reverse = step + 1; reverse > 0; --reverse )
         {
            const Index row = reverse - 1;
            Number value = data.projected_rhs[row];
            Number row_scale = 0.;
            for( Index column = row + 1; column <= step; ++column )
            {
               value -= data.Hessenberg(row, column)
                  * data.coefficients[column];
               row_scale = Max(
                  row_scale,
                  std::abs(data.Hessenberg(row, column)));
            }
            const Number row_diagonal = data.Hessenberg(row, row);
            row_scale = Max(row_scale, std::abs(row_diagonal));
            const Number minimum_diagonal =
               data.fgmres_breakdown_tolerance
               * Max(row_scale, std::numeric_limits<Number>::min());
            if( !IsFiniteNumber(row_diagonal)
                || std::abs(row_diagonal) <= minimum_diagonal )
            {
               candidate_valid = false;
               break;
            }
            data.coefficients[row] = value / row_diagonal;
            if( !IsFiniteNumber(data.coefficients[row]) )
            {
               candidate_valid = false;
               break;
            }
         }
         if( !candidate_valid )
         {
            refinement_failed = residual_ratio > residual_ratio_max_;
            return true;
         }

         res.Copy(*data.base_solution);
         for( Index index = 0; index <= step; ++index )
         {
            res.Axpy(
               data.coefficients[index],
               *data.preconditioned_basis[index]);
         }

         ComputeResiduals(
            system, include_perturbation, rhs, res, resid);
         residual_ratio = ComputeResidualRatio(rhs, res, resid);
         Jnlst().Printf(
            J_DETAILED, J_LINEAR_ALGEBRA,
            "FGMRES residual_ratio = %e after %" IPOPT_INDEX_FORMAT
            " iterations\n",
            residual_ratio,
            num_iterations);
         if( !IsFiniteNumber(residual_ratio) )
         {
            refinement_failed = true;
            return true;
         }

         if( num_iterations >= min_refinement_steps_
             && residual_ratio <= residual_ratio_max_ )
         {
            return true;
         }

         refinement_failed = residual_ratio > residual_ratio_max_
            && num_iterations > min_refinement_steps_
            && (num_iterations > max_refinement_steps_
                || residual_ratio
                   > residual_improvement_factor_ * residual_ratio_old);
         residual_ratio_old = residual_ratio;
         if( refinement_failed )
         {
            return true;
         }
         if( happy_breakdown )
         {
            refinement_failed = residual_ratio > residual_ratio_max_;
            return true;
         }
      }
   }

   refinement_failed = num_iterations < min_refinement_steps_
      || residual_ratio > residual_ratio_max_;
   return true;
}

void PDFullSpaceSolver::ApplyKKT(
   const PDFullSpaceSolverSystem& system,
   bool                           include_perturbation,
   const IteratesVector&          input,
   IteratesVector&                output
)
{
   ++system.refinement_data.statistics.kkt_applications;

   Number delta_x = 0.;
   Number delta_s = 0.;
   Number delta_c = 0.;
   Number delta_d = 0.;
   if( include_perturbation )
   {
      perturbHandler_->CurrentPerturbation(
         delta_x, delta_s, delta_c, delta_d);
      RecordPerturbation(
         system.refinement_data.statistics,
         delta_x, delta_s, delta_c, delta_d);
   }

   SmartPtr<Vector> tmp;

   system.W.MultVector(1., *input.x(), 0., *output.x_NonConst());
   system.J_c.TransMultVector(
      1., *input.y_c(), 1., *output.x_NonConst());
   system.J_d.TransMultVector(
      1., *input.y_d(), 1., *output.x_NonConst());
   system.Px_L.MultVector(
      -1., *input.z_L(), 1., *output.x_NonConst());
   system.Px_U.MultVector(
      1., *input.z_U(), 1., *output.x_NonConst());
   if( delta_x != 0. )
   {
      output.x_NonConst()->Axpy(delta_x, *input.x());
   }

   system.Pd_U.MultVector(
      1., *input.v_U(), 0., *output.s_NonConst());
   system.Pd_L.MultVector(
      -1., *input.v_L(), 1., *output.s_NonConst());
   output.s_NonConst()->Axpy(-1., *input.y_d());
   if( delta_s != 0. )
   {
      output.s_NonConst()->Axpy(delta_s, *input.s());
   }

   system.J_c.MultVector(
      1., *input.x(), 0., *output.y_c_NonConst());
   if( delta_c != 0. )
   {
      output.y_c_NonConst()->Axpy(-delta_c, *input.y_c());
   }

   system.J_d.MultVector(
      1., *input.x(), 0., *output.y_d_NonConst());
   output.y_d_NonConst()->Axpy(-1., *input.s());
   if( delta_d != 0. )
   {
      output.y_d_NonConst()->Axpy(-delta_d, *input.y_d());
   }

   output.z_L_NonConst()->Copy(*input.z_L());
   output.z_L_NonConst()->ElementWiseMultiply(system.slack_x_L);
   tmp = system.z_L.MakeNew();
   system.Px_L.TransMultVector(1., *input.x(), 0., *tmp);
   tmp->ElementWiseMultiply(system.z_L);
   output.z_L_NonConst()->Axpy(1., *tmp);

   output.z_U_NonConst()->Copy(*input.z_U());
   output.z_U_NonConst()->ElementWiseMultiply(system.slack_x_U);
   tmp = system.z_U.MakeNew();
   system.Px_U.TransMultVector(1., *input.x(), 0., *tmp);
   tmp->ElementWiseMultiply(system.z_U);
   output.z_U_NonConst()->Axpy(-1., *tmp);

   output.v_L_NonConst()->Copy(*input.v_L());
   output.v_L_NonConst()->ElementWiseMultiply(system.slack_s_L);
   tmp = system.v_L.MakeNew();
   system.Pd_L.TransMultVector(1., *input.s(), 0., *tmp);
   tmp->ElementWiseMultiply(system.v_L);
   output.v_L_NonConst()->Axpy(1., *tmp);

   output.v_U_NonConst()->Copy(*input.v_U());
   output.v_U_NonConst()->ElementWiseMultiply(system.slack_s_U);
   tmp = system.v_U.MakeNew();
   system.Pd_U.TransMultVector(1., *input.s(), 0., *tmp);
   tmp->ElementWiseMultiply(system.v_U);
   output.v_U_NonConst()->Axpy(-1., *tmp);
}

void PDFullSpaceSolver::ComputeResiduals(
   const PDFullSpaceSolverSystem& system,
   bool                           include_perturbation,
   const IteratesVector&          rhs,
   const IteratesVector&          res,
   IteratesVector&                resid
)
{
   DBG_START_METH("PDFullSpaceSolver::ComputeResiduals", dbg_verbosity);

   DBG_PRINT_VECTOR(2, "res", res);
   IpData().TimingStats().ComputeResiduals().Start();

   ++system.refinement_data.statistics.kkt_applications;

   Number delta_x = 0.;
   Number delta_s = 0.;
   Number delta_c = 0.;
   Number delta_d = 0.;
   if( include_perturbation )
   {
      perturbHandler_->CurrentPerturbation(
         delta_x, delta_s, delta_c, delta_d);
      RecordPerturbation(
         system.refinement_data.statistics,
         delta_x, delta_s, delta_c, delta_d);
   }

   SmartPtr<Vector> tmp;

   // x
   system.W.MultVector(1., *res.x(), 0., *resid.x_NonConst());
   system.J_c.TransMultVector(1., *res.y_c(), 1., *resid.x_NonConst());
   system.J_d.TransMultVector(1., *res.y_d(), 1., *resid.x_NonConst());
   system.Px_L.MultVector(-1., *res.z_L(), 1., *resid.x_NonConst());
   system.Px_U.MultVector(1., *res.z_U(), 1., *resid.x_NonConst());
   resid.x_NonConst()->AddTwoVectors(delta_x, *res.x(), -1., *rhs.x(), 1.);

   // s
   system.Pd_U.MultVector(1., *res.v_U(), 0., *resid.s_NonConst());
   system.Pd_L.MultVector(-1., *res.v_L(), 1., *resid.s_NonConst());
   resid.s_NonConst()->AddTwoVectors(-1., *res.y_d(), -1., *rhs.s(), 1.);
   if( delta_s != 0. )
   {
      resid.s_NonConst()->Axpy(delta_s, *res.s());
   }

   // c
   system.J_c.MultVector(1., *res.x(), 0., *resid.y_c_NonConst());
   resid.y_c_NonConst()->AddTwoVectors(-delta_c, *res.y_c(), -1., *rhs.y_c(), 1.);

   // d
   system.J_d.MultVector(1., *res.x(), 0., *resid.y_d_NonConst());
   resid.y_d_NonConst()->AddTwoVectors(-1., *res.s(), -1., *rhs.y_d(), 1.);
   if( delta_d != 0. )
   {
      resid.y_d_NonConst()->Axpy(-delta_d, *res.y_d());
   }

   // zL
   resid.z_L_NonConst()->Copy(*res.z_L());
   resid.z_L_NonConst()->ElementWiseMultiply(system.slack_x_L);
   tmp = system.z_L.MakeNew();
   system.Px_L.TransMultVector(1., *res.x(), 0., *tmp);
   tmp->ElementWiseMultiply(system.z_L);
   resid.z_L_NonConst()->AddTwoVectors(1., *tmp, -1., *rhs.z_L(), 1.);

   // zU
   resid.z_U_NonConst()->Copy(*res.z_U());
   resid.z_U_NonConst()->ElementWiseMultiply(system.slack_x_U);
   tmp = system.z_U.MakeNew();
   system.Px_U.TransMultVector(1., *res.x(), 0., *tmp);
   tmp->ElementWiseMultiply(system.z_U);
   resid.z_U_NonConst()->AddTwoVectors(-1., *tmp, -1., *rhs.z_U(), 1.);

   // vL
   resid.v_L_NonConst()->Copy(*res.v_L());
   resid.v_L_NonConst()->ElementWiseMultiply(system.slack_s_L);
   tmp = system.v_L.MakeNew();
   system.Pd_L.TransMultVector(1., *res.s(), 0., *tmp);
   tmp->ElementWiseMultiply(system.v_L);
   resid.v_L_NonConst()->AddTwoVectors(1., *tmp, -1., *rhs.v_L(), 1.);

   // vU
   resid.v_U_NonConst()->Copy(*res.v_U());
   resid.v_U_NonConst()->ElementWiseMultiply(system.slack_s_U);
   tmp = system.v_U.MakeNew();
   system.Pd_U.TransMultVector(1., *res.s(), 0., *tmp);
   tmp->ElementWiseMultiply(system.v_U);
   resid.v_U_NonConst()->AddTwoVectors(-1., *tmp, -1., *rhs.v_U(), 1.);

   DBG_PRINT_VECTOR(2, "resid", resid);

   if( Jnlst().ProduceOutput(J_MOREVECTOR, J_LINEAR_ALGEBRA) )
   {
      resid.Print(Jnlst(), J_MOREVECTOR, J_LINEAR_ALGEBRA, "resid");
   }

   if( Jnlst().ProduceOutput(J_MOREDETAILED, J_LINEAR_ALGEBRA) )
   {
      Jnlst().Printf(J_MOREDETAILED, J_LINEAR_ALGEBRA,
                     "max-norm resid_x  %e\n", resid.x()->Amax());
      Jnlst().Printf(J_MOREDETAILED, J_LINEAR_ALGEBRA,
                     "max-norm resid_s  %e\n", resid.s()->Amax());
      Jnlst().Printf(J_MOREDETAILED, J_LINEAR_ALGEBRA,
                     "max-norm resid_c  %e\n", resid.y_c()->Amax());
      Jnlst().Printf(J_MOREDETAILED, J_LINEAR_ALGEBRA,
                     "max-norm resid_d  %e\n", resid.y_d()->Amax());
      Jnlst().Printf(J_MOREDETAILED, J_LINEAR_ALGEBRA,
                     "max-norm resid_zL %e\n", resid.z_L()->Amax());
      Jnlst().Printf(J_MOREDETAILED, J_LINEAR_ALGEBRA,
                     "max-norm resid_zU %e\n", resid.z_U()->Amax());
      Jnlst().Printf(J_MOREDETAILED, J_LINEAR_ALGEBRA,
                     "max-norm resid_vL %e\n", resid.v_L()->Amax());
      Jnlst().Printf(J_MOREDETAILED, J_LINEAR_ALGEBRA,
                     "max-norm resid_vU %e\n", resid.v_U()->Amax());
   }
   IpData().TimingStats().ComputeResiduals().End();
}

Number PDFullSpaceSolver::ComputeResidualRatio(
   const IteratesVector& rhs,
   const IteratesVector& res,
   const IteratesVector& resid
)
{
   DBG_START_METH("PDFullSpaceSolver::ComputeResidualRatio", dbg_verbosity);

   Number nrm_rhs = rhs.Amax();
   Number nrm_res = res.Amax();
   Number nrm_resid = resid.Amax();
   Jnlst().Printf(J_MOREDETAILED, J_LINEAR_ALGEBRA,
                  "nrm_rhs = %8.2e nrm_sol = %8.2e nrm_resid = %8.2e\n", nrm_rhs, nrm_res, nrm_resid);

   if( nrm_rhs + nrm_res == 0. )
   {
      return nrm_resid;  // this should be zero
   }
   else
   {
      // ToDo: determine how to include norm of matrix, and what
      // safeguard to use against incredibly large solution vectors
      Number max_cond = 1e6;
      return nrm_resid / (Min(nrm_res, max_cond * nrm_rhs) + nrm_rhs);
   }
}

} // namespace Ipopt
