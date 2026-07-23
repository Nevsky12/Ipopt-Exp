// Copyright (C) 2026 Ipopt contributors.
// All Rights Reserved.
// This file is distributed under the Eclipse Public License.

#ifndef IPOPT_CXX23_BORDERED_STAGE_STRUCTURED_CANDIDATE_BACKEND_HPP
#define IPOPT_CXX23_BORDERED_STAGE_STRUCTURED_CANDIDATE_BACKEND_HPP

#include <ipopt/cxx23/bordered_block_tridiagonal_solver.hpp>
#include <ipopt/cxx23/stage_structured_candidate_backend.hpp>

#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace Ipopt::Cxx23
{
/** Static topology for a reduced stage chain plus a small dense border. */
struct BorderedStageStructuredLayout
{
   std::vector<Index> block_sizes;
   Index border_dimension = 0;
   Index full_direction_dimension = 0;
   StructureFingerprint kkt_fingerprint{0, 0};
   /** Inertia target after exact complementarity condensation. */
   Index inertia_dimension = 0;
   /** The bound assembler proved that reconstruction writes every full entry.
    * Uncertified assemblers retain runtime NaN poisoning before reconstruction.
    */
   bool full_direction_overwrite_certified = false;
};

/** Maps one full KKT request to [A B;B^T C] and overwrites its full direction. */
template <class Assembler>
concept BorderedStageStructuredAssembler = requires(
   Assembler&                         assembler,
   const Assembler&                   const_assembler,
   CandidateFirstSolveRequest         request,
   PrimalDualRegularization           regularization,
   std::span<Number>                  diagonal,
   std::span<Number>                  lower,
   std::span<Number>                  border,
   std::span<Number>                  border_diagonal,
   std::span<Number>                  rhs,
   std::span<const Number>            structured_solution,
   std::span<Number>                  full_direction
)
{
   { const_assembler.bordered_stage_structured_layout() }
      -> std::same_as<BorderedStageStructuredLayout>;
   {
      assembler.assemble_bordered_stage_system(
         request,
         regularization,
         diagonal,
         lower,
         border,
         border_diagonal,
         rhs)
   } -> std::same_as<EvaluationValue<StageStructuredAssemblyReport>>;
   {
      assembler.reconstruct_bordered_stage_direction(
         request, regularization, structured_solution, full_direction)
   } -> std::same_as<EvaluationValue<StageStructuredWork>>;
};

template <class Assembler>
concept BorderedStageStructuredRightHandSideAssembler = requires(
   Assembler&                  assembler,
   CandidateFirstSolveRequest request,
   std::span<Number>          rhs
)
{
   { assembler.assemble_bordered_stage_rhs(request, rhs) }
      -> std::same_as<EvaluationResult>;
};

struct BorderedStageStructuredCandidateOptions
{
   BorderedBlockTridiagonalOptions factorization;
   BlockRefinementOptions refinement;
   Index maximum_factorization_attempts = 8;
   Number initial_primal_regularization = 1e-8;
   Number initial_dual_regularization = 1e-8;
   Number regularization_growth = 10.;
   Number maximum_regularization = 1e16;
   /** Start a new numeric revision at the last fully accepted perturbation. */
   bool reuse_accepted_regularization = false;
   /** Adaptive full-KKT correction using the current bordered factor. */
   FgmresOptions full_kkt_refinement{
      .restart = 20,
      .maximum_iterations = 20,
      .relative_tolerance = 1e-11,
      .absolute_tolerance = 1e-13,
      .breakdown_tolerance = 1e-14,
      .reorthogonalize = true,
      .check_finite = true
   };
};

/** Candidate-first backend for an explicitly bordered stage KKT.
 *
 * The arrowhead kernel never claims exact inertia for a floating-point Schur
 * complement. Acceptance therefore requires the assembler to provide an
 * independent exact certificate for the complete reduced KKT. If the proof is
 * unavailable at the current regularization, both regularizations grow and
 * assembly is retried. The numerical factor and original-arrowhead residual
 * gate remain mandatory even when the theorem-level certificate is present.
 */
template <BorderedStageStructuredAssembler Assembler>
class BorderedStageStructuredCandidateBackend
{
public:
   explicit BorderedStageStructuredCandidateBackend(
      Assembler                                 assembler,
      BorderedStageStructuredCandidateOptions  options = {}
   )
      : assembler_(std::move(assembler)),
        layout_(assembler_.bordered_stage_structured_layout()),
        options_(options),
        solver_(
           layout_.block_sizes,
           layout_.border_dimension,
           options_.factorization),
        diagonal_(solver_.storage().diagonal_values),
        lower_(solver_.storage().lower_values),
        border_(solver_.storage().border_values),
        border_diagonal_(solver_.storage().border_diagonal_values),
        structured_rhs_(solver_.storage().dimension),
        structured_solution_(solver_.storage().dimension),
        full_direction_(layout_.full_direction_dimension),
        full_kkt_fgmres_(
           layout_.full_direction_dimension,
           options_.full_kkt_refinement)
   {
      configuration_error_ = ValidateConfiguration();
   }

   EvaluationValue<CandidateFirstSolveResult> solve(
      CandidateFirstSolveRequest request
   )
   {
      current_factor_.reset();
      if( !configuration_error_.empty() )
      {
         return Failure(
            EvaluationErrorCode::invalid_layout, configuration_error_);
      }
      if constexpr( requires {
         {
            assembler_.prepare_reusable_bordered_stage_storage(
               diagonal_, lower_, border_, border_diagonal_)
         } -> std::same_as<EvaluationResult>;
      } )
      {
         if( EvaluationResult prepared =
                assembler_.prepare_reusable_bordered_stage_storage(
                   diagonal_, lower_, border_, border_diagonal_);
             !prepared )
         {
            return std::unexpected(prepared.error());
         }
      }
      if( request.rhs.size() != layout_.full_direction_dimension ||
          request.kkt.flat_dimension() != layout_.full_direction_dimension ||
          (!request.direction_output.empty() &&
           request.direction_output.size() !=
              layout_.full_direction_dimension) )
      {
         return Failure(
            EvaluationErrorCode::dimension_mismatch,
            "bordered stage backend full direction dimension does not match the request");
      }
      if( request.required_negative_eigenvalues > layout_.inertia_dimension )
      {
         return Failure(
            EvaluationErrorCode::invalid_layout,
            "requested negative-eigenvalue count exceeds the bordered inertia dimension");
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
            "bordered stage assembler does not match the requested KKT structure");
      }
      if( !ValidRegularization(request.state.regularization) )
      {
         return Failure(
            EvaluationErrorCode::invalid_layout,
            "bordered candidate regularization must be finite and nonnegative");
      }

      PrimalDualRegularization regularization = request.state.regularization;
      if( options_.reuse_accepted_regularization &&
          last_accepted_regularization_ )
      {
         regularization.x = std::max(
            regularization.x, last_accepted_regularization_->x);
         regularization.s = std::max(
            regularization.s, last_accepted_regularization_->s);
         regularization.c = std::max(
            regularization.c, last_accepted_regularization_->c);
         regularization.d = std::max(
            regularization.d, last_accepted_regularization_->d);
      }
      CandidateFirstWorkStatistics work;
      std::string last_failure =
         "bordered stage factorization attempts were exhausted";
      for( Index attempt = 0;
           attempt < options_.maximum_factorization_attempts;
           ++attempt )
      {
         EvaluationValue<StageStructuredAssemblyReport> assembled =
            assembler_.assemble_bordered_stage_system(
               request,
               regularization,
               diagonal_,
               lower_,
               border_,
               border_diagonal_,
               structured_rhs_);
         if( !assembled )
         {
            return std::unexpected(assembled.error());
         }
         AddWork(work, assembled->work);

         if( !assembled->independent_full_inertia )
         {
            last_failure =
               "bordered numeric Schur factor requires an independent exact inertia proof";
            if( !PrepareRetry(
                   regularization,
                   RetryTargetForHint(assembled->inertia_retry_hint),
                   work) )
            {
               break;
            }
            continue;
         }
         EvaluationValue<CertifiedInertia> full_inertia =
            ValidateIndependentInertia(
               *assembled->independent_full_inertia,
               layout_.inertia_dimension);
         if( !full_inertia )
         {
            return std::unexpected(full_inertia.error());
         }
         if( full_inertia->negative_eigenvalues !=
             request.required_negative_eigenvalues )
         {
            const RetryTarget target =
               full_inertia->negative_eigenvalues >
                  request.required_negative_eigenvalues
               ? RetryTarget::primal
               : RetryTarget::dual;
            last_failure = "bordered stage certificate has " +
               std::to_string(full_inertia->negative_eigenvalues) +
               " negative eigenvalues, expected " +
               std::to_string(request.required_negative_eigenvalues);
            if( !PrepareRetry(regularization, target, work) )
            {
               break;
            }
            continue;
         }

         EvaluationValue<BorderedBlockTridiagonalFactorizationReport> factorized =
            solver_.factorize(diagonal_, lower_, border_, border_diagonal_);
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

         const BorderedBlockTridiagonalStatistics before = solver_.statistics();
         EvaluationValue<BlockRefinementReport> refined =
            solver_.solve_refined_rhs(
               structured_rhs_, structured_solution_, options_.refinement);
         const BorderedBlockTridiagonalStatistics after = solver_.statistics();
         SaturatingAdd(
            work.backsolves,
            after.solved_right_hand_sides - before.solved_right_hand_sides);
         SaturatingAdd(
            work.refinement_steps,
            after.refinement_steps - before.refinement_steps);
         SaturatingAdd(
            work.kkt_applications,
            after.matrix_applications - before.matrix_applications);
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
            last_failure = refined->status ==
                  BlockRefinementStatus::residual_increase
               ? "bordered stage iterative-refinement residual did not decrease"
               : "bordered stage iterative refinement reached its iteration limit";
            if( !PrepareRetry(regularization, RetryTarget::both, work) )
            {
               break;
            }
            continue;
         }

         const bool use_request_direction_output =
            !request.direction_output.empty();
         std::span<Number> full_direction = use_request_direction_output
            ? request.direction_output
            : std::span<Number>(full_direction_);
#ifndef NDEBUG
         std::ranges::fill(
            full_direction, std::numeric_limits<Number>::quiet_NaN());
#else
         if( !layout_.full_direction_overwrite_certified )
         {
            std::ranges::fill(
               full_direction, std::numeric_limits<Number>::quiet_NaN());
         }
#endif
         EvaluationValue<StageStructuredWork> reconstructed =
            assembler_.reconstruct_bordered_stage_direction(
               request,
               regularization,
               structured_solution_,
               full_direction);
         if( !reconstructed )
         {
            return std::unexpected(reconstructed.error());
         }
         AddWork(work, *reconstructed);
         if( !std::ranges::all_of(
                full_direction, [](Number value) { return std::isfinite(value); }) )
         {
            return Failure(
               EvaluationErrorCode::nonfinite_output,
               "bordered stage reconstruction produced a nonfinite direction");
         }

         CandidateFirstSolveResult result;
         if( use_request_direction_output )
         {
            result.direction_written_to_request_output = true;
         }
         else
         {
            result.direction = full_direction_;
         }
         result.accepted_regularization = regularization;
         result.inertia = {
            CandidateFirstInertiaCertainty::exact,
            full_inertia->negative_eigenvalues
         };
         result.work = work;
         result.converged = true;
         current_factor_ = CurrentFactor{
            .numeric_revision = request.state.numeric_revision,
            .regularization = regularization,
            .restoration_problem = request.restoration_problem
         };
         if( options_.reuse_accepted_regularization )
         {
            last_accepted_regularization_ = regularization;
         }
         return result;
      }
      return Failure(EvaluationErrorCode::model_failure, std::move(last_failure));
   }

   /** Refine the last direct step with full K_delta and the current factor. */
   EvaluationValue<CandidateFirstRefinementResult> refine(
      CandidateFirstRefinementRequest request
   ) requires BorderedStageStructuredRightHandSideAssembler<Assembler>
   {
      CandidateFirstRefinementResult result;
      result.supported = true;
      if( !current_factor_ )
      {
         return RefinementFailure(
            EvaluationErrorCode::numeric_mismatch,
            "bordered full-KKT refinement has no current direct factor");
      }
      if( request.rhs.size() != layout_.full_direction_dimension ||
          request.direction.size() != layout_.full_direction_dimension ||
          request.kkt.flat_dimension() != layout_.full_direction_dimension )
      {
         return RefinementFailure(
            EvaluationErrorCode::dimension_mismatch,
            "bordered full-KKT refinement has the wrong dimension");
      }
      if( request.state.numeric_revision != current_factor_->numeric_revision ||
          request.restoration_problem != current_factor_->restoration_problem ||
          !SameRegularization(
             request.state.regularization,
             current_factor_->regularization) )
      {
         return RefinementFailure(
            EvaluationErrorCode::numeric_mismatch,
            "bordered full-KKT refinement state does not match the current factor");
      }
      if( EvaluationResult valid = request.kkt.validate_state(request.state);
          !valid )
      {
         return std::unexpected(valid.error());
      }
      StructureFingerprintResult fingerprint =
         request.kkt.structure_fingerprint();
      if( !fingerprint )
      {
         return std::unexpected(fingerprint.error());
      }
      if( *fingerprint != layout_.kkt_fingerprint )
      {
         return RefinementFailure(
            EvaluationErrorCode::structure_mismatch,
            "bordered full-KKT refinement structure does not match its factor");
      }
      if( !AllFinite(request.rhs) || !AllFinite(request.direction) )
      {
         return RefinementFailure(
            EvaluationErrorCode::nonfinite_output,
            "bordered full-KKT refinement input is nonfinite");
      }

      CandidateFirstWorkStatistics callback_work;
      const auto apply_operator = [&](std::span<const Number> input,
                                      std::span<Number> output)
      {
         return request.kkt.apply_flat(request.state, input, output);
      };
      const auto apply_preconditioner = [&]([[maybe_unused]] Index iteration,
                                             std::span<const Number> input,
                                             std::span<Number> output)
      {
         CandidateFirstSolveRequest correction_request{
            request.kkt,
            request.state,
            input,
            0,
            request.restoration_problem
         };
         if( EvaluationResult assembled =
                assembler_.assemble_bordered_stage_rhs(
                   correction_request, structured_rhs_);
             !assembled )
         {
            return assembled;
         }
         if( EvaluationResult solved = solver_.solve_rhs(
                structured_rhs_, structured_solution_);
             !solved )
         {
            return solved;
         }
         PoisonDirection(full_direction_);
         EvaluationValue<StageStructuredWork> reconstructed =
            assembler_.reconstruct_bordered_stage_direction(
               correction_request,
               request.state.regularization,
               structured_solution_,
               full_direction_);
         if( !reconstructed )
         {
            return EvaluationResult(std::unexpected(reconstructed.error()));
         }
         AddWork(callback_work, *reconstructed);
         if( !AllFinite(full_direction_) )
         {
            return EvaluationResult(std::unexpected(EvaluationError{
               EvaluationErrorCode::nonfinite_output,
               "bordered full-KKT preconditioner produced a nonfinite direction"
            }));
         }
         std::ranges::copy(full_direction_, output.begin());
         return EvaluationResult{};
      };

      EvaluationValue<FgmresResult> refined = full_kkt_fgmres_.solve(
         apply_operator,
         apply_preconditioner,
         request.rhs,
         request.direction);
      if( !refined )
      {
         return std::unexpected(refined.error());
      }
      result.work = callback_work;
      SaturatingAdd(
         result.work.backsolves, refined->preconditioner_evaluations);
      SaturatingAdd(
         result.work.kkt_applications, refined->operator_evaluations);
      SaturatingAdd(result.work.refinement_steps, refined->iterations);
      SaturatingAddProduct(
         result.work.derivative_product_requests,
         refined->operator_evaluations,
         3);
      result.converged = refined->converged();
      return result;
   }

   const BorderedStageStructuredLayout& layout() const noexcept
   {
      return layout_;
   }

private:
   struct CurrentFactor
   {
      std::uint64_t numeric_revision = 0;
      PrimalDualRegularization regularization{0., 0., 0., 0.};
      bool restoration_problem = false;
   };

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
         return "bordered stage block topology is invalid";
      }
      const BorderedBlockTridiagonalStorage storage = solver_.storage();
      if( solver_.block_sizes() != layout_.block_sizes ||
          storage.border_dimension != layout_.border_dimension ||
          diagonal_.size() != storage.diagonal_values ||
          lower_.size() != storage.lower_values ||
          border_.size() != storage.border_values ||
          border_diagonal_.size() != storage.border_diagonal_values ||
          structured_rhs_.size() != storage.dimension ||
          structured_solution_.size() != storage.dimension ||
          full_direction_.size() != layout_.full_direction_dimension )
      {
         return "bordered prepared workspace does not match the assembler topology";
      }
      if( layout_.border_dimension == 0 ||
          layout_.full_direction_dimension == 0 ||
          layout_.inertia_dimension == 0 ||
          storage.dimension > layout_.inertia_dimension ||
          layout_.inertia_dimension > layout_.full_direction_dimension )
      {
         return "bordered reduced/inertia/full dimensions are inconsistent";
      }
      if( options_.maximum_factorization_attempts == 0 ||
          !std::isfinite(options_.initial_primal_regularization) ||
          !(options_.initial_primal_regularization > 0.) ||
          !std::isfinite(options_.initial_dual_regularization) ||
          !(options_.initial_dual_regularization > 0.) ||
          !std::isfinite(options_.regularization_growth) ||
          !(options_.regularization_growth > 1.) ||
          !std::isfinite(options_.maximum_regularization) ||
          options_.maximum_regularization <
             options_.initial_primal_regularization ||
          options_.maximum_regularization <
             options_.initial_dual_regularization ||
          !std::isfinite(options_.refinement.relative_tolerance) ||
          options_.refinement.relative_tolerance < 0. ||
          full_kkt_fgmres_.dimension() != layout_.full_direction_dimension ||
          options_.full_kkt_refinement.restart == 0 ||
          !std::isfinite(options_.full_kkt_refinement.relative_tolerance) ||
          options_.full_kkt_refinement.relative_tolerance < 0. ||
          !std::isfinite(options_.full_kkt_refinement.absolute_tolerance) ||
          options_.full_kkt_refinement.absolute_tolerance < 0. ||
          !std::isfinite(options_.full_kkt_refinement.breakdown_tolerance) ||
          !(options_.full_kkt_refinement.breakdown_tolerance > 0.) )
      {
         return "bordered retry/refinement options are invalid";
      }
      return {};
   }

   static bool ValidRegularization(PrimalDualRegularization value) noexcept
   {
      return std::isfinite(value.x) && value.x >= 0. &&
         std::isfinite(value.s) && value.s >= 0. &&
         std::isfinite(value.c) && value.c >= 0. &&
         std::isfinite(value.d) && value.d >= 0.;
   }

   static bool SameRegularization(
      PrimalDualRegularization left,
      PrimalDualRegularization right
   ) noexcept
   {
      return left.x == right.x && left.s == right.s &&
         left.c == right.c && left.d == right.d;
   }

   static bool AllFinite(std::span<const Number> values) noexcept
   {
      return std::ranges::all_of(
         values, [](Number value) { return std::isfinite(value); });
   }

   void PoisonDirection(std::span<Number> direction) const noexcept
   {
#ifndef NDEBUG
      std::ranges::fill(
         direction, std::numeric_limits<Number>::quiet_NaN());
#else
      if( !layout_.full_direction_overwrite_certified )
      {
         std::ranges::fill(
            direction, std::numeric_limits<Number>::quiet_NaN());
      }
#endif
   }

   static EvaluationValue<CandidateFirstSolveResult> Failure(
      EvaluationErrorCode code,
      std::string         message
   )
   {
      return std::unexpected(EvaluationError{code, std::move(message)});
   }

   static EvaluationValue<CandidateFirstRefinementResult> RefinementFailure(
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

   static void SaturatingAddProduct(
      Index& target,
      Index  factor,
      Index  multiplier
   ) noexcept
   {
      const Index increment = factor != 0 &&
         multiplier > std::numeric_limits<Index>::max() / factor
         ? std::numeric_limits<Index>::max()
         : factor * multiplier;
      SaturatingAdd(target, increment);
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

   static EvaluationValue<CertifiedInertia> ValidateIndependentInertia(
      CertifiedInertia inertia,
      Index            full_dimension
   )
   {
      Index dimension = 0;
      if( !inertia.exact || inertia.zero_eigenvalues != 0 ||
          !CheckedAdd(
             inertia.positive_eigenvalues,
             inertia.negative_eigenvalues,
             dimension) ||
          dimension != full_dimension )
      {
         return std::unexpected(EvaluationError{
            EvaluationErrorCode::invalid_layout,
            "bordered independent inertia proof does not cover the configured target"
         });
      }
      return inertia;
   }

   static RetryTarget RetryTargetForHint(
      StageStructuredInertiaRetryHint hint
   ) noexcept
   {
      switch( hint )
      {
         case StageStructuredInertiaRetryHint::primal:
            return RetryTarget::primal;
         case StageStructuredInertiaRetryHint::dual:
            return RetryTarget::dual;
         case StageStructuredInertiaRetryHint::both:
            return RetryTarget::both;
      }
      return RetryTarget::both;
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
   BorderedStageStructuredLayout layout_;
   BorderedStageStructuredCandidateOptions options_;
   SymmetricBorderedBlockTridiagonalSolver solver_;
   std::vector<Number> diagonal_;
   std::vector<Number> lower_;
   std::vector<Number> border_;
   std::vector<Number> border_diagonal_;
   std::vector<Number> structured_rhs_;
   std::vector<Number> structured_solution_;
   std::vector<Number> full_direction_;
   FgmresSolver full_kkt_fgmres_;
   std::string configuration_error_;
   std::optional<PrimalDualRegularization> last_accepted_regularization_;
   std::optional<CurrentFactor> current_factor_;
};

template <BorderedStageStructuredAssembler Assembler>
SharedCandidateFirstBackend MakeBorderedStageStructuredCandidateBackend(
   Assembler                                 assembler,
   BorderedStageStructuredCandidateOptions  options = {}
)
{
   return MakeCandidateFirstBackend(
      BorderedStageStructuredCandidateBackend<Assembler>(
         std::move(assembler), options));
}
} // namespace Ipopt::Cxx23

#endif
