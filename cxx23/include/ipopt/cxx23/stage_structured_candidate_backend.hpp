// Copyright (C) 2026 Ipopt contributors.
// All Rights Reserved.
// This file is distributed under the Eclipse Public License.

#ifndef IPOPT_CXX23_STAGE_STRUCTURED_CANDIDATE_BACKEND_HPP
#define IPOPT_CXX23_STAGE_STRUCTURED_CANDIDATE_BACKEND_HPP

#include <ipopt/cxx23/block_tridiagonal_solver.hpp>
#include <ipopt/cxx23/candidate_first_backend.hpp>

#include <algorithm>
#include <cmath>
#include <concepts>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace Ipopt::Cxx23
{
/** Static topology of a structured representation of one full KKT family. */
struct StageStructuredLayout
{
   std::vector<Index> block_sizes;
   Index full_direction_dimension = 0;
   StructureFingerprint kkt_fingerprint{0, 0};
   /** Dimension whose inertia must match the stable augmented-system check.
    *
    * Zero preserves the original contract and means full_direction_dimension.
    * A full eight-block assembler may eliminate complementarity directions
    * algebraically and set this to the symmetric [x,s,y_c,y_d] dimension while
    * still reconstructing a full eight-block direction.
    */
   Index inertia_dimension = 0;
};

/** Exact inertia contribution removed by a congruent condensation. */
struct CongruenceInertiaContribution
{
   Index positive_eigenvalues = 0;
   Index negative_eigenvalues = 0;
   bool exact = false;
};

/** Work performed outside the generic block factorization. */
struct StageStructuredWork
{
   Index factorizations = 0;
   Index backsolves = 0;
   Index kkt_applications = 0;
   Index derivative_product_requests = 0;
   Index refinement_steps = 0;
   Index quality_improvements = 0;
};

/** Regularization family that may make a missing independent proof hold. */
enum class StageStructuredInertiaRetryHint
{
   both,
   primal,
   dual
};

struct StageStructuredAssemblyReport
{
   CongruenceInertiaContribution eliminated_inertia;
   /** Optional theorem- or capability-based proof for layout.inertia_dimension.
    * When present, this supersedes local Schur-block inertia, but never the
    * numerical-factorization and true-residual gates. */
   std::optional<CertifiedInertia> independent_full_inertia;
   /** Conservative retry direction when the optional proof is absent. */
   StageStructuredInertiaRetryHint inertia_retry_hint =
      StageStructuredInertiaRetryHint::both;
   StageStructuredWork work;
};

/**
 * An assembler maps the current full eight-block request to a symmetric
 * block-tridiagonal system and reconstructs the full direction afterward.
 *
 * assemble() fills complete row-major diagonal blocks, lower blocks, and one
 * reduced RHS. If it condenses variables, eliminated_inertia must be an exact
 * inertia contribution from the same congruence transformation. reconstruct()
 * must undo the permutation/condensation into full Ipopt flat ordering. Neither
 * method may retain the request or any supplied span.
 */
template <class Assembler>
concept StageStructuredAssembler = requires(
   Assembler&                         assembler,
   const Assembler&                   const_assembler,
   CandidateFirstSolveRequest         request,
   PrimalDualRegularization           regularization,
   std::span<Number>                  diagonal,
   std::span<Number>                  lower,
   std::span<Number>                  rhs,
   std::span<const Number>            structured_solution,
   std::span<Number>                  full_direction
)
{
   { const_assembler.stage_structured_layout() }
      -> std::same_as<StageStructuredLayout>;
   {
      assembler.assemble_stage_system(
         request, regularization, diagonal, lower, rhs)
   } -> std::same_as<EvaluationValue<StageStructuredAssemblyReport>>;
   {
      assembler.reconstruct_stage_direction(
         request, regularization, structured_solution, full_direction)
   } -> std::same_as<EvaluationValue<StageStructuredWork>>;
};

struct StageStructuredCandidateOptions
{
   BlockTridiagonalFactorizationOptions factorization;
   BlockRefinementOptions refinement;
   Index maximum_factorization_attempts = 8;
   Number initial_primal_regularization = 1e-8;
   Number initial_dual_regularization = 1e-8;
   Number regularization_growth = 10.;
   Number maximum_regularization = 1e16;
};

/** Movable allocation bundle for a stage backend whose topology is known
 * before the live KKT/assembler becomes available.
 *
 * Construction performs every topology-sized allocation. Binding consumes
 * the workspace and validates its block sizes and full dimension against the
 * assembler. The fingerprint in the preparation layout is intentionally not
 * used: the live assembler remains authoritative for structural identity.
 */
class StageStructuredCandidateWorkspace
{
public:
   explicit StageStructuredCandidateWorkspace(
      StageStructuredLayout          layout,
      StageStructuredCandidateOptions options = {}
   )
      : options_(options),
        solver_(layout.block_sizes, 1, options.factorization),
        diagonal_(solver_.storage().diagonal_values),
        lower_(solver_.storage().lower_values),
        structured_rhs_(solver_.storage().dimension),
        structured_solution_(solver_.storage().dimension),
        full_direction_(layout.full_direction_dimension)
   {
   }

   StageStructuredCandidateWorkspace(
      const StageStructuredCandidateWorkspace&
   ) = delete;
   StageStructuredCandidateWorkspace& operator=(
      const StageStructuredCandidateWorkspace&
   ) = delete;
   StageStructuredCandidateWorkspace(
      StageStructuredCandidateWorkspace&&
   ) = default;
   StageStructuredCandidateWorkspace& operator=(
      StageStructuredCandidateWorkspace&&
   ) = default;

private:
   template <StageStructuredAssembler>
   friend class StageStructuredCandidateBackend;

   StageStructuredCandidateOptions options_;
   SymmetricBlockTridiagonalSolver solver_;
   std::vector<Number> diagonal_;
   std::vector<Number> lower_;
   std::vector<Number> structured_rhs_;
   std::vector<Number> structured_solution_;
   std::vector<Number> full_direction_;
};

/** Independent candidate-first backend for explicitly stage-structured KKT.
 *
 * Each attempt assembles and factorizes without a stable Ipopt solve. Too many
 * negative eigenvalues increase primal regularization; too few increase dual
 * regularization. Singular/uncertified factors and failed refinement increase
 * both. A direction is returned only after certified full inertia, converged
 * structured true-residual refinement, successful reconstruction, and finite
 * output.
 */
template <StageStructuredAssembler Assembler>
class StageStructuredCandidateBackend
{
public:
   explicit StageStructuredCandidateBackend(
      Assembler assembler,
      StageStructuredCandidateOptions options = {}
   )
      : assembler_(std::move(assembler)),
        layout_(assembler_.stage_structured_layout()),
        options_(options),
        solver_(layout_.block_sizes, 1, options_.factorization),
        diagonal_(solver_.storage().diagonal_values),
        lower_(solver_.storage().lower_values),
        structured_rhs_(solver_.storage().dimension),
        structured_solution_(solver_.storage().dimension),
        full_direction_(layout_.full_direction_dimension)
   {
      configuration_error_ = ValidateConfiguration();
   }

   StageStructuredCandidateBackend(
      Assembler                         assembler,
      StageStructuredCandidateWorkspace workspace
   )
      : assembler_(std::move(assembler)),
        layout_(assembler_.stage_structured_layout()),
        options_(workspace.options_),
        solver_(std::move(workspace.solver_)),
        diagonal_(std::move(workspace.diagonal_)),
        lower_(std::move(workspace.lower_)),
        structured_rhs_(std::move(workspace.structured_rhs_)),
        structured_solution_(std::move(workspace.structured_solution_)),
        full_direction_(std::move(workspace.full_direction_))
   {
      configuration_error_ = ValidateConfiguration();
   }

   EvaluationValue<CandidateFirstSolveResult> solve(
      CandidateFirstSolveRequest request
   )
   {
      if( !configuration_error_.empty() )
      {
         return Failure(
            EvaluationErrorCode::invalid_layout, configuration_error_);
      }
      if( request.rhs.size() != layout_.full_direction_dimension ||
          request.kkt.flat_dimension() != layout_.full_direction_dimension )
      {
         return Failure(
            EvaluationErrorCode::dimension_mismatch,
            "stage-structured backend full direction dimension does not match the request");
      }
      const Index inertia_dimension = InertiaDimension();
      if( request.required_negative_eigenvalues > inertia_dimension )
      {
         return Failure(
            EvaluationErrorCode::invalid_layout,
            "requested negative-eigenvalue count exceeds the configured inertia dimension");
      }
      if( EvaluationResult state = request.kkt.validate_state(request.state); !state )
      {
         return std::unexpected(state.error());
      }
      StructureFingerprintResult fingerprint = request.kkt.structure_fingerprint();
      if( !fingerprint )
      {
         return std::unexpected(fingerprint.error());
      }
      if( *fingerprint != layout_.kkt_fingerprint )
      {
         return Failure(
            EvaluationErrorCode::structure_mismatch,
            "stage-structured assembler does not match the requested KKT structure");
      }
      if( !ValidRegularization(request.state.regularization) )
      {
         return Failure(
            EvaluationErrorCode::invalid_layout,
            "candidate-first regularization must be finite and nonnegative");
      }

      PrimalDualRegularization regularization = request.state.regularization;
      CandidateFirstWorkStatistics work;
      std::string last_failure = "stage-structured factorization attempts were exhausted";
      for( Index attempt = 0;
           attempt < options_.maximum_factorization_attempts;
           ++attempt )
      {
         EvaluationValue<StageStructuredAssemblyReport> assembled =
            assembler_.assemble_stage_system(
               request, regularization, diagonal_, lower_, structured_rhs_);
         if( !assembled )
         {
            return std::unexpected(assembled.error());
         }
         AddWork(work, assembled->work);

         EvaluationValue<BlockTridiagonalFactorizationReport> factorized =
            solver_.factorize(diagonal_, lower_);
         SaturatingAdd(work.factorizations, 1);
         if( !factorized )
         {
            last_failure = factorized.error().message;
            if( factorized.error().code != EvaluationErrorCode::model_failure )
            {
               return std::unexpected(factorized.error());
            }
            if( !PrepareRetry(regularization, RetryTarget::both, work) )
            {
               break;
            }
            continue;
         }

         EvaluationValue<CertifiedInertia> full_inertia = [&]()
         {
            if( assembled->independent_full_inertia )
            {
               return ValidateIndependentInertia(
                  *assembled->independent_full_inertia,
                  inertia_dimension);
            }
            return ComposeInertia(
               factorized->inertia,
               assembled->eliminated_inertia,
               inertia_dimension);
         }();
         if( !full_inertia )
         {
            if( !assembled->independent_full_inertia &&
                !factorized->inertia.exact &&
                PrepareRetry(regularization, RetryTarget::both, work) )
            {
               last_failure =
                  "numeric stage factor requires an independent exact inertia proof";
               continue;
            }
            return std::unexpected(full_inertia.error());
         }
         if( full_inertia->negative_eigenvalues !=
             request.required_negative_eigenvalues )
         {
            const RetryTarget target =
               full_inertia->negative_eigenvalues > request.required_negative_eigenvalues
               ? RetryTarget::primal
               : RetryTarget::dual;
            last_failure = "stage-structured factorization has " +
               std::to_string(full_inertia->negative_eigenvalues) +
               " negative eigenvalues, expected " +
               std::to_string(request.required_negative_eigenvalues);
            if( !PrepareRetry(regularization, target, work) )
            {
               break;
            }
            continue;
         }

         const BlockTridiagonalStatistics before_refinement = solver_.statistics();
         EvaluationValue<BlockRefinementReport> refined =
            solver_.solve_refined_rhs(
               structured_rhs_, structured_solution_, options_.refinement);
         const BlockTridiagonalStatistics after_refinement = solver_.statistics();
         SaturatingAdd(
            work.backsolves,
            after_refinement.solved_right_hand_sides -
               before_refinement.solved_right_hand_sides);
         SaturatingAdd(
            work.refinement_steps,
            after_refinement.refinement_steps -
               before_refinement.refinement_steps);
         SaturatingAdd(
            work.kkt_applications,
            after_refinement.matrix_applications -
               before_refinement.matrix_applications);
         if( !refined )
         {
            last_failure = refined.error().message;
            if( !PrepareRetry(regularization, RetryTarget::both, work) )
            {
               break;
            }
            continue;
         }
         if( refined->status != BlockRefinementStatus::converged )
         {
            last_failure = refined->status == BlockRefinementStatus::residual_increase
               ? "stage-structured iterative refinement residual did not decrease"
               : "stage-structured iterative refinement reached its iteration limit";
            if( !PrepareRetry(regularization, RetryTarget::both, work) )
            {
               break;
            }
            continue;
         }

         std::ranges::fill(full_direction_, 0.);
         EvaluationValue<StageStructuredWork> reconstructed =
            assembler_.reconstruct_stage_direction(
               request, regularization, structured_solution_, full_direction_);
         if( !reconstructed )
         {
            return std::unexpected(reconstructed.error());
         }
         AddWork(work, *reconstructed);
         if( !std::ranges::all_of(
                full_direction_, [](Number value) { return std::isfinite(value); }) )
         {
            return Failure(
               EvaluationErrorCode::nonfinite_output,
               "stage-structured reconstruction produced a nonfinite direction");
         }

         CandidateFirstSolveResult result;
         result.direction = full_direction_;
         result.accepted_regularization = regularization;
         result.inertia = {
            CandidateFirstInertiaCertainty::exact,
            full_inertia->negative_eigenvalues
         };
         result.work = work;
         result.converged = true;
         return result;
      }
      return Failure(EvaluationErrorCode::model_failure, std::move(last_failure));
   }

   const StageStructuredLayout& layout() const noexcept
   {
      return layout_;
   }

private:
   enum class RetryTarget
   {
      primal,
      dual,
      both
   };

   std::string ValidateConfiguration() const
   {
      if( !solver_.configured() )
      {
         return "stage-structured block topology is invalid";
      }
      if( solver_.block_sizes() != layout_.block_sizes ||
          diagonal_.size() != solver_.storage().diagonal_values ||
          lower_.size() != solver_.storage().lower_values ||
          structured_rhs_.size() != solver_.storage().dimension ||
          structured_solution_.size() != solver_.storage().dimension ||
          full_direction_.size() != layout_.full_direction_dimension )
      {
         return "prepared stage workspace does not match the assembler topology";
      }
      if( layout_.full_direction_dimension == 0 ||
          InertiaDimension() == 0 ||
          solver_.storage().dimension > InertiaDimension() ||
          InertiaDimension() > layout_.full_direction_dimension )
      {
         return "stage-structured reduced/inertia/full dimensions are inconsistent";
      }
      if( options_.maximum_factorization_attempts == 0 ||
          !std::isfinite(options_.initial_primal_regularization) ||
          !(options_.initial_primal_regularization > 0.) ||
          !std::isfinite(options_.initial_dual_regularization) ||
          !(options_.initial_dual_regularization > 0.) ||
          !std::isfinite(options_.regularization_growth) ||
          !(options_.regularization_growth > 1.) ||
          !std::isfinite(options_.maximum_regularization) ||
          options_.maximum_regularization < options_.initial_primal_regularization ||
          options_.maximum_regularization < options_.initial_dual_regularization ||
          !std::isfinite(options_.refinement.relative_tolerance) ||
          options_.refinement.relative_tolerance < 0. )
      {
         return "stage-structured retry/refinement options are invalid";
      }
      return {};
   }

   Index InertiaDimension() const noexcept
   {
      return layout_.inertia_dimension == 0
         ? layout_.full_direction_dimension
         : layout_.inertia_dimension;
   }

   static bool ValidRegularization(PrimalDualRegularization regularization) noexcept
   {
      return std::isfinite(regularization.x) && regularization.x >= 0. &&
         std::isfinite(regularization.s) && regularization.s >= 0. &&
         std::isfinite(regularization.c) && regularization.c >= 0. &&
         std::isfinite(regularization.d) && regularization.d >= 0.;
   }

   static EvaluationValue<CandidateFirstSolveResult> Failure(
      EvaluationErrorCode code,
      std::string         message
   )
   {
      return std::unexpected(EvaluationError{code, std::move(message)});
   }

   static void AddWork(
      CandidateFirstWorkStatistics& target,
      const StageStructuredWork&    source
   ) noexcept
   {
      SaturatingAdd(target.factorizations, source.factorizations);
      SaturatingAdd(target.backsolves, source.backsolves);
      SaturatingAdd(target.kkt_applications, source.kkt_applications);
      SaturatingAdd(
         target.derivative_product_requests,
         source.derivative_product_requests);
      SaturatingAdd(target.refinement_steps, source.refinement_steps);
      SaturatingAdd(target.quality_improvements, source.quality_improvements);
   }

   static void SaturatingAdd(Index& target, Index increment) noexcept
   {
      target = increment > std::numeric_limits<Index>::max() - target
         ? std::numeric_limits<Index>::max()
         : target + increment;
   }

   static bool CheckedAdd(Index left, Index right, Index& result) noexcept
   {
      if( right > std::numeric_limits<Index>::max() - left )
      {
         return false;
      }
      result = left + right;
      return true;
   }

   static EvaluationValue<CertifiedInertia> ComposeInertia(
      CertifiedInertia                    structured,
      CongruenceInertiaContribution       eliminated,
      Index                               full_dimension
   )
   {
      if( !structured.exact || !eliminated.exact )
      {
         return std::unexpected(EvaluationError{
            EvaluationErrorCode::model_failure,
            "stage-structured assembler did not provide exact eliminated inertia"
         });
      }
      Index structured_dimension = 0;
      Index full_positive = 0;
      Index full_negative = 0;
      Index composed_dimension = 0;
      if( !CheckedAdd(
             structured.positive_eigenvalues,
             structured.negative_eigenvalues,
             structured_dimension) ||
          !CheckedAdd(
             structured.positive_eigenvalues,
             eliminated.positive_eigenvalues,
             full_positive) ||
          !CheckedAdd(
             structured.negative_eigenvalues,
             eliminated.negative_eigenvalues,
             full_negative) ||
          !CheckedAdd(full_positive, full_negative, composed_dimension) )
      {
         return std::unexpected(EvaluationError{
            EvaluationErrorCode::invalid_layout,
            "stage-structured inertia contribution overflows Index"
         });
      }
      structured.positive_eigenvalues = full_positive;
      structured.negative_eigenvalues = full_negative;
      if( structured_dimension > full_dimension ||
          composed_dimension != full_dimension )
      {
         return std::unexpected(EvaluationError{
            EvaluationErrorCode::invalid_layout,
            "structured and eliminated inertia do not cover the configured inertia dimension"
         });
      }
      structured.zero_eigenvalues = 0;
      structured.exact = true;
      return structured;
   }

   static EvaluationValue<CertifiedInertia> ValidateIndependentInertia(
      CertifiedInertia inertia,
      Index            full_dimension
   )
   {
      Index nonzero_dimension = 0;
      if( !inertia.exact || inertia.zero_eigenvalues != 0 ||
          !CheckedAdd(
             inertia.positive_eigenvalues,
             inertia.negative_eigenvalues,
             nonzero_dimension) ||
          nonzero_dimension != full_dimension )
      {
         return std::unexpected(EvaluationError{
            EvaluationErrorCode::invalid_layout,
            "independent inertia certificate does not cover the configured inertia target"
         });
      }
      return inertia;
   }

   bool PrepareRetry(
      PrimalDualRegularization&       regularization,
      RetryTarget                    target,
      CandidateFirstWorkStatistics&  work
   ) const noexcept
   {
      bool changed = false;
      if( target == RetryTarget::primal || target == RetryTarget::both )
      {
         changed = IncreaseRegularization(
            regularization.x, options_.initial_primal_regularization) || changed;
         changed = IncreaseRegularization(
            regularization.s, options_.initial_primal_regularization) || changed;
      }
      if( target == RetryTarget::dual || target == RetryTarget::both )
      {
         changed = IncreaseRegularization(
            regularization.c, options_.initial_dual_regularization) || changed;
         changed = IncreaseRegularization(
            regularization.d, options_.initial_dual_regularization) || changed;
      }
      if( changed )
      {
         SaturatingAdd(work.quality_improvements, 1);
      }
      return changed;
   }

   bool IncreaseRegularization(Number& value, Number initial) const noexcept
   {
      const Number previous = value;
      if( value < initial )
      {
         value = initial;
      }
      else if( value < options_.maximum_regularization )
      {
         value = std::min(
            options_.maximum_regularization,
            value * options_.regularization_growth);
      }
      return value > previous;
   }

   Assembler assembler_;
   StageStructuredLayout layout_;
   StageStructuredCandidateOptions options_;
   SymmetricBlockTridiagonalSolver solver_;
   std::vector<Number> diagonal_;
   std::vector<Number> lower_;
   std::vector<Number> structured_rhs_;
   std::vector<Number> structured_solution_;
   std::vector<Number> full_direction_;
   std::string configuration_error_;
};

template <StageStructuredAssembler Assembler>
SharedCandidateFirstBackend MakeStageStructuredCandidateBackend(
   Assembler assembler,
   StageStructuredCandidateOptions options = {}
)
{
   return MakeCandidateFirstBackend(
      StageStructuredCandidateBackend<Assembler>(
         std::move(assembler), options));
}
} // namespace Ipopt::Cxx23

#endif
