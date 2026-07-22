// Copyright (C) 2026 Ipopt contributors.
// All Rights Reserved.
// This file is distributed under the Eclipse Public License.

#include <ipopt/cxx23/stage_structured_candidate_backend.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

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

void CheckNear(Number actual, Number expected, std::string_view message)
{
   if( std::abs(actual - expected) > 2e-11 )
   {
      throw std::runtime_error(
         std::string(message) + ": actual=" + std::to_string(actual) +
         " expected=" + std::to_string(expected));
   }
}

struct TinyStageModel
{
   Number h00;
   Number h01;
   Number h11;
   Number j0;
   Number j1;
   std::uint64_t revision = 91;

   NlpStructure structure() const
   {
      return {2, 1, 2, 3, revision};
   }

   EvaluationValue<Number> eval_objective(std::span<const Number>)
   {
      return 0.;
   }

   EvaluationResult eval_gradient(std::span<const Number>, std::span<Number> result)
   {
      std::ranges::fill(result, 0.);
      return {};
   }

   EvaluationResult eval_constraints(
      std::span<const Number> x,
      std::span<Number>       result
   )
   {
      result[0] = j0 * x[0] + j1 * x[1];
      return {};
   }

   EvaluationResult eval_jacobian_structure(
      std::span<Index> rows,
      std::span<Index> columns
   )
   {
      rows[0] = 0;
      columns[0] = 0;
      rows[1] = 0;
      columns[1] = 1;
      return {};
   }

   EvaluationResult eval_jacobian_values(
      std::span<const Number>,
      std::span<Number> values
   )
   {
      values[0] = j0;
      values[1] = j1;
      return {};
   }

   EvaluationResult eval_hessian_structure(
      std::span<Index> rows,
      std::span<Index> columns
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

   EvaluationResult eval_hessian_values(
      std::span<const Number>,
      Number objective_factor,
      std::span<const Number>,
      std::span<Number> values
   )
   {
      values[0] = objective_factor * h00;
      values[1] = objective_factor * h01;
      values[2] = objective_factor * h11;
      return {};
   }

   EvaluationResult eval_jacobian_product(
      std::span<const Number>,
      std::span<const Number> direction,
      std::span<Number>       result
   )
   {
      result[0] = j0 * direction[0] + j1 * direction[1];
      return {};
   }

   EvaluationResult eval_jacobian_transpose_product(
      std::span<const Number>,
      std::span<const Number> direction,
      std::span<Number>       result
   )
   {
      result[0] = j0 * direction[0];
      result[1] = j1 * direction[0];
      return {};
   }

   EvaluationResult eval_hessian_product(
      std::span<const Number>,
      Number objective_factor,
      std::span<const Number>,
      std::span<const Number> direction,
      std::span<Number>       result
   )
   {
      result[0] = objective_factor * (h00 * direction[0] + h01 * direction[1]);
      result[1] = objective_factor * (h01 * direction[0] + h11 * direction[1]);
      return {};
   }
};

class TinyStageAssembler
{
public:
   TinyStageAssembler(
      StageStructuredLayout layout,
      TinyStageModel        model,
      bool                  condense_first_primal = false
   )
      : layout_(std::move(layout)),
        model_(model),
        condense_first_primal_(condense_first_primal)
   {
   }

   StageStructuredLayout stage_structured_layout() const
   {
      return layout_;
   }

   EvaluationValue<StageStructuredAssemblyReport> assemble_stage_system(
      CandidateFirstSolveRequest request,
      PrimalDualRegularization   regularization,
      std::span<Number>          diagonal,
      std::span<Number>          lower,
      std::span<Number>          rhs
   )
   {
      if( fail_assembly )
      {
         return std::unexpected(EvaluationError{
            EvaluationErrorCode::model_failure,
            "injected stage assembly failure"
         });
      }
      StageStructuredAssemblyReport report;
      report.eliminated_inertia.exact = !uncertified_eliminated_inertia;
      if( condense_first_primal_ )
      {
         if( diagonal.size() != 4 || !lower.empty() || rhs.size() != 2 )
         {
            return DimensionFailure("condensed stage storage has the wrong size");
         }
         if( model_.h01 != 0. || model_.j0 != 0. )
         {
            return DimensionFailure("test condensation requires a decoupled first primal");
         }
         const Number eliminated_pivot = model_.h00 + regularization.x;
         if( eliminated_pivot == 0. )
         {
            return std::unexpected(EvaluationError{
               EvaluationErrorCode::model_failure,
               "test condensation pivot is singular"
            });
         }
         diagonal[0] = model_.h11 + regularization.x;
         diagonal[1] = model_.j1;
         diagonal[2] = model_.j1;
         diagonal[3] = -regularization.c;
         rhs[0] = request.rhs[1];
         rhs[1] = request.rhs[2];
         if( eliminated_pivot > 0. )
         {
            report.eliminated_inertia.positive_eigenvalues = 1;
         }
         else
         {
            report.eliminated_inertia.negative_eigenvalues = 1;
         }
      }
      else
      {
         if( diagonal.size() != 5 || lower.size() != 2 || rhs.size() != 3 )
         {
            return DimensionFailure("full stage storage has the wrong size");
         }
         diagonal[0] = model_.h00 + regularization.x;
         diagonal[1] = model_.h01;
         diagonal[2] = model_.h01;
         diagonal[3] = model_.h11 + regularization.x;
         diagonal[4] = -regularization.c;
         lower[0] = model_.j0;
         lower[1] = model_.j1;
         std::ranges::copy(request.rhs, rhs.begin());
      }
      if( wrong_eliminated_count )
      {
         ++report.eliminated_inertia.positive_eigenvalues;
      }
      report.independent_full_inertia = independent_full_inertia;
      return report;
   }

   EvaluationValue<StageStructuredWork> reconstruct_stage_direction(
      CandidateFirstSolveRequest request,
      PrimalDualRegularization   regularization,
      std::span<const Number>    structured_solution,
      std::span<Number>          full_direction
   )
   {
      if( fail_reconstruction )
      {
         return std::unexpected(EvaluationError{
            EvaluationErrorCode::model_failure,
            "injected stage reconstruction failure"
         });
      }
      StageStructuredWork work;
      if( condense_first_primal_ )
      {
         if( structured_solution.size() != 2 || full_direction.size() != 3 )
         {
            return std::unexpected(EvaluationError{
               EvaluationErrorCode::dimension_mismatch,
               "condensed reconstruction has the wrong size"
            });
         }
         full_direction[0] = request.rhs[0] / (model_.h00 + regularization.x);
         full_direction[1] = structured_solution[0];
         full_direction[2] = structured_solution[1];
         work.backsolves = 1;
      }
      else
      {
         if( incomplete_reconstruction )
         {
            std::ranges::copy(
               structured_solution.first(structured_solution.size() - 1),
               full_direction.begin());
         }
         else
         {
            std::ranges::copy(structured_solution, full_direction.begin());
         }
      }
      if( nonfinite_reconstruction )
      {
         full_direction[0] = std::numeric_limits<Number>::quiet_NaN();
      }
      return work;
   }

   bool fail_assembly = false;
   bool fail_reconstruction = false;
   bool incomplete_reconstruction = false;
   bool nonfinite_reconstruction = false;
   bool uncertified_eliminated_inertia = false;
   bool wrong_eliminated_count = false;
   std::optional<CertifiedInertia> independent_full_inertia;

private:
   static EvaluationValue<StageStructuredAssemblyReport> DimensionFailure(
      std::string message
   )
   {
      return std::unexpected(EvaluationError{
         EvaluationErrorCode::dimension_mismatch,
         std::move(message)
      });
   }

   StageStructuredLayout layout_;
   TinyStageModel model_;
   bool condense_first_primal_;
};

PrimalDualKktOperator MakeKkt(TinyStageModel model)
{
   return PrimalDualKktOperator(
      MakeNlpProblem(model),
      {
         .equality_constraints = {0},
         .inequality_constraints = {},
         .primal_lower_bounds = {},
         .primal_upper_bounds = {},
         .slack_lower_bounds = {},
         .slack_upper_bounds = {}
      });
}

PrimalDualState MakeState(
   const std::array<Number, 2>& x,
   const std::array<Number, 1>& multipliers,
   PrimalDualRegularization     regularization = {0., 0., 0., 0.}
)
{
   return {
      .nlp = {x, 1., multipliers},
      .z_lower = {},
      .z_upper = {},
      .v_lower = {},
      .v_upper = {},
      .slack_x_lower = {},
      .slack_x_upper = {},
      .slack_s_lower = {},
      .slack_s_upper = {},
      .regularization = regularization,
      .numeric_revision = 1
   };
}

StageStructuredLayout Layout(
   PrimalDualKktOperator& kkt,
   std::vector<Index>     block_sizes
)
{
   const StructureFingerprintResult fingerprint = kkt.structure_fingerprint();
   Check(fingerprint.has_value(), "test KKT fingerprint failed");
   return {std::move(block_sizes), kkt.flat_dimension(), *fingerprint};
}

struct LazyFactoryState
{
   Index creations = 0;
   Index failures_remaining = 0;
};

class CountingStageBackendFactory
{
public:
   CountingStageBackendFactory(
      TinyStageModel                    model,
      std::shared_ptr<LazyFactoryState> state
   )
      : model_(model), state_(std::move(state))
   {
   }

   EvaluationValue<StageStructuredCandidateBackend<TinyStageAssembler>>
   operator()(PrimalDualKktOperator& kkt, bool restoration_problem)
   {
      ++state_->creations;
      if( restoration_problem )
      {
         return std::unexpected(EvaluationError{
            EvaluationErrorCode::invalid_layout,
            "test lazy factory rejects restoration"
         });
      }
      if( state_->failures_remaining != 0 )
      {
         --state_->failures_remaining;
         return std::unexpected(EvaluationError{
            EvaluationErrorCode::model_failure,
            "injected lazy backend creation failure"
         });
      }
      return StageStructuredCandidateBackend(
         TinyStageAssembler(Layout(kkt, {2, 1}), model_));
   }

private:
   TinyStageModel model_;
   std::shared_ptr<LazyFactoryState> state_;
};

void CheckTrueResidual(
   PrimalDualKktOperator&    kkt,
   PrimalDualState           state,
   std::span<const Number>   direction,
   PrimalDualRegularization  regularization,
   std::span<const Number>   rhs
)
{
   state.regularization = regularization;
   std::array<Number, 3> applied{};
   Check(
      kkt.apply_flat(state, direction, applied).has_value(),
      "accepted structured direction could not be applied");
   for( Index row = 0; row < applied.size(); ++row )
   {
      CheckNear(applied[row], rhs[row], "structured direction fails the true KKT");
   }
}

void CheckTrueResidual(
   PrimalDualKktOperator&           kkt,
   PrimalDualState                  state,
   const CandidateFirstSolveResult& result,
   std::span<const Number>          rhs
)
{
   CheckTrueResidual(
      kkt, state, result.direction, result.accepted_regularization, rhs);
}

void TestHappyPathAndErasure()
{
   const TinyStageModel model{4., 1., 3., 1., 2.};
   PrimalDualKktOperator kkt = MakeKkt(model);
   const auto layout = Layout(kkt, {2, 1});
   const std::array<Number, 2> x{{0., 0.}};
   const std::array<Number, 1> multipliers{{0.}};
   const PrimalDualState state = MakeState(x, multipliers);
   const std::array<Number, 3> rhs{{2., -1., 3.}};
   CandidateFirstSolveRequest request{kkt, state, rhs, 1, false};

   StageStructuredCandidateBackend backend(
      TinyStageAssembler(layout, model));
   const auto solved = backend.solve(request);
   Check(solved.has_value() && solved->converged, "stage candidate happy path failed");
   Check(
      solved->inertia.certainty == CandidateFirstInertiaCertainty::exact &&
         solved->inertia.negative_eigenvalues == 1,
      "stage candidate returned the wrong inertia certificate");
   Check(
      solved->work.factorizations == 1 && solved->work.backsolves == 1 &&
         solved->work.quality_improvements == 0,
      "stage candidate happy-path accounting is wrong");
   CheckTrueResidual(kkt, state, *solved, rhs);

   SharedCandidateFirstBackend erased = MakeStageStructuredCandidateBackend(
      TinyStageAssembler(layout, model));
   const auto erased_solve = erased->candidate_first_solve(request);
   Check(
      erased_solve.has_value() && erased_solve->converged,
      "AnyAny-erased stage candidate failed");
   CheckTrueResidual(kkt, state, *erased_solve, rhs);

   std::array<Number, 3> direction_output;
   std::ranges::fill(
      direction_output, std::numeric_limits<Number>::quiet_NaN());
   const auto output_solve = erased->candidate_first_solve({
      kkt, state, rhs, 1, false, direction_output
   });
   Check(
      output_solve.has_value() && output_solve->direction.empty() &&
         output_solve->direction_written_to_request_output,
      "AnyAny-erased stage candidate did not select caller-owned output");
   CheckTrueResidual(
      kkt,
      state,
      direction_output,
      output_solve->accepted_regularization,
      rhs);
   const std::array<Number, 3> retained_output = direction_output;
   Check(
      erased->candidate_first_solve(request).has_value() &&
         direction_output == retained_output,
      "stage candidate retained caller-owned output past the solve");

   std::array<Number, 2> short_output{};
   const auto wrong_output = erased->candidate_first_solve({
      kkt, state, rhs, 1, false, short_output
   });
   Check(
      !wrong_output.has_value() &&
         wrong_output.error().code == EvaluationErrorCode::dimension_mismatch,
      "stage candidate accepted a wrong-size caller-owned output");
}

void TestPreparedWorkspaceBinding()
{
   const TinyStageModel model{4., 1., 3., 1., 2.};
   PrimalDualKktOperator kkt = MakeKkt(model);
   const auto layout = Layout(kkt, {2, 1});
   const std::array<Number, 2> x{{0., 0.}};
   const std::array<Number, 1> multipliers{{0.}};
   const PrimalDualState state = MakeState(x, multipliers);
   const std::array<Number, 3> rhs{{2., -1., 3.}};

   StageStructuredCandidateWorkspace workspace(layout);
   StageStructuredCandidateBackend backend(
      TinyStageAssembler(layout, model), std::move(workspace));
   const auto solved = backend.solve({kkt, state, rhs, 1, false});
   Check(solved.has_value(), "prepared stage workspace solve failed");
   CheckTrueResidual(kkt, state, *solved, rhs);

   StageStructuredLayout wrong_layout = layout;
   ++wrong_layout.full_direction_dimension;
   StageStructuredCandidateWorkspace wrong_workspace(wrong_layout);
   StageStructuredCandidateBackend mismatch(
      TinyStageAssembler(layout, model), std::move(wrong_workspace));
   const auto rejected = mismatch.solve({kkt, state, rhs, 1, false});
   Check(
      !rejected.has_value() &&
         rejected.error().code == EvaluationErrorCode::invalid_layout,
      "mismatched prepared stage workspace was accepted");
}

void TestInertiaRetryChangesPrimalRegularization()
{
   const TinyStageModel model{-1., 0., 2., 0., 1.};
   PrimalDualKktOperator kkt = MakeKkt(model);
   const auto layout = Layout(kkt, {2, 1});
   const std::array<Number, 2> x{{0., 0.}};
   const std::array<Number, 1> multipliers{{0.}};
   const PrimalDualState state = MakeState(x, multipliers);
   const std::array<Number, 3> rhs{{1., 2., -3.}};

   StageStructuredCandidateOptions options;
   options.maximum_factorization_attempts = 2;
   options.initial_primal_regularization = 2.;
   StageStructuredCandidateBackend backend(
      TinyStageAssembler(layout, model), options);
   const auto solved = backend.solve({kkt, state, rhs, 1, false});
   Check(solved.has_value(), "inertia retry did not recover");
   CheckNear(
      solved->accepted_regularization.x, 2.,
      "inertia retry chose the wrong primal regularization");
   Check(
      solved->work.factorizations == 2 &&
         solved->work.quality_improvements == 1 &&
         solved->work.backsolves == 1,
      "inertia retry work accounting is wrong");
   CheckTrueResidual(kkt, state, *solved, rhs);
}

void TestCongruentCondensationCertificate()
{
   const TinyStageModel model{3., 0., 2., 0., 1.};
   PrimalDualKktOperator kkt = MakeKkt(model);
   const auto layout = Layout(kkt, {2});
   const std::array<Number, 2> x{{0., 0.}};
   const std::array<Number, 1> multipliers{{0.}};
   const PrimalDualState state = MakeState(x, multipliers);
   const std::array<Number, 3> rhs{{6., -2., 1.}};

   StageStructuredCandidateBackend backend(
      TinyStageAssembler(layout, model, true));
   const auto solved = backend.solve({kkt, state, rhs, 1, false});
   Check(solved.has_value(), "congruent condensed stage solve failed");
   Check(
      solved->inertia.negative_eigenvalues == 1 &&
         solved->work.backsolves == 2,
      "condensed inertia/reconstruction accounting is wrong");
   CheckTrueResidual(kkt, state, *solved, rhs);
}

void TestIndependentInertiaProofGate()
{
   const TinyStageModel model{4., 1., 3., 1., 2.};
   PrimalDualKktOperator kkt = MakeKkt(model);
   const auto layout = Layout(kkt, {2, 1});
   const std::array<Number, 2> x{{0., 0.}};
   const std::array<Number, 1> multipliers{{0.}};
   const PrimalDualState state = MakeState(x, multipliers);
   const std::array<Number, 3> rhs{{2., -1., 3.}};
   StageStructuredCandidateOptions options;
   options.factorization.require_certified_inertia = false;
   options.maximum_factorization_attempts = 1;

   StageStructuredCandidateBackend missing_proof(
      TinyStageAssembler(layout, model), options);
   const auto missing = missing_proof.solve({kkt, state, rhs, 1, false});
   Check(
      !missing.has_value() &&
         missing.error().code == EvaluationErrorCode::model_failure,
      "numeric stage factor without an independent proof was accepted");

   TinyStageAssembler valid_assembler(layout, model);
   valid_assembler.independent_full_inertia = CertifiedInertia{
      .positive_eigenvalues = 2,
      .negative_eigenvalues = 1,
      .zero_eigenvalues = 0,
      .certificate_radius = 0.,
      .minimum_separation = 0.,
      .exact = true
   };
   StageStructuredCandidateBackend valid(
      std::move(valid_assembler), options);
   const auto solved = valid.solve({kkt, state, rhs, 1, false});
   Check(
      solved.has_value() && solved->inertia.negative_eigenvalues == 1,
      "valid independent full-KKT proof was rejected");
   CheckTrueResidual(kkt, state, *solved, rhs);

   const std::array<CertifiedInertia, 3> invalid_proofs{{
      {
         .positive_eigenvalues = 2,
         .negative_eigenvalues = 1,
         .zero_eigenvalues = 0,
         .exact = false
      },
      {
         .positive_eigenvalues = 1,
         .negative_eigenvalues = 1,
         .zero_eigenvalues = 0,
         .exact = true
      },
      {
         .positive_eigenvalues = 1,
         .negative_eigenvalues = 1,
         .zero_eigenvalues = 1,
         .exact = true
      }
   }};
   for( const CertifiedInertia& proof : invalid_proofs )
   {
      TinyStageAssembler invalid_assembler(layout, model);
      invalid_assembler.independent_full_inertia = proof;
      StageStructuredCandidateBackend invalid(
         std::move(invalid_assembler), options);
      const auto result = invalid.solve({kkt, state, rhs, 1, false});
      Check(
         !result.has_value() &&
            result.error().code == EvaluationErrorCode::invalid_layout,
         "invalid independent full-KKT proof was accepted");
   }
}

void TestContractFailures()
{
   const TinyStageModel model{4., .25, 3., 1., 2.};
   PrimalDualKktOperator kkt = MakeKkt(model);
   const auto layout = Layout(kkt, {2, 1});
   const std::array<Number, 2> x{{0., 0.}};
   const std::array<Number, 1> multipliers{{0.}};
   const PrimalDualState state = MakeState(x, multipliers);
   const std::array<Number, 3> rhs{{1., 2., 3.}};

   StageStructuredLayout wrong_fingerprint = layout;
   ++wrong_fingerprint.kkt_fingerprint.low;
   StageStructuredCandidateBackend mismatch(
      TinyStageAssembler(wrong_fingerprint, model));
   const auto mismatch_result = mismatch.solve({kkt, state, rhs, 1, false});
   Check(
      !mismatch_result.has_value() &&
         mismatch_result.error().code == EvaluationErrorCode::structure_mismatch,
      "wrong stage/KKT fingerprint was accepted");

   TinyStageAssembler bad_count_assembler(layout, model);
   bad_count_assembler.wrong_eliminated_count = true;
   StageStructuredCandidateBackend bad_count(std::move(bad_count_assembler));
   const auto bad_count_result = bad_count.solve({kkt, state, rhs, 1, false});
   Check(
      !bad_count_result.has_value() &&
         bad_count_result.error().code == EvaluationErrorCode::invalid_layout,
      "inertia contribution with the wrong dimension was accepted");

   TinyStageAssembler uncertified_assembler(layout, model);
   uncertified_assembler.uncertified_eliminated_inertia = true;
   StageStructuredCandidateBackend uncertified(
      std::move(uncertified_assembler));
   Check(
      !uncertified.solve({kkt, state, rhs, 1, false}).has_value(),
      "uncertified eliminated inertia was accepted");

   TinyStageAssembler reconstruction_assembler(layout, model);
   reconstruction_assembler.fail_reconstruction = true;
   StageStructuredCandidateBackend reconstruction(
      std::move(reconstruction_assembler));
   Check(
      !reconstruction.solve({kkt, state, rhs, 1, false}).has_value(),
      "failed direction reconstruction was accepted");

   TinyStageAssembler incomplete_assembler(layout, model);
   incomplete_assembler.incomplete_reconstruction = true;
   StageStructuredCandidateBackend incomplete(std::move(incomplete_assembler));
   const auto incomplete_result =
      incomplete.solve({kkt, state, rhs, 1, false});
   Check(
      !incomplete_result.has_value() &&
         incomplete_result.error().code ==
            EvaluationErrorCode::nonfinite_output,
      "incomplete direction reconstruction was accepted");

   TinyStageAssembler nonfinite_assembler(layout, model);
   nonfinite_assembler.nonfinite_reconstruction = true;
   StageStructuredCandidateBackend nonfinite(std::move(nonfinite_assembler));
   const auto nonfinite_result = nonfinite.solve({kkt, state, rhs, 1, false});
   Check(
      !nonfinite_result.has_value() &&
         nonfinite_result.error().code == EvaluationErrorCode::nonfinite_output,
      "nonfinite reconstructed direction was accepted");

   StageStructuredCandidateOptions exhausted_options;
   exhausted_options.maximum_factorization_attempts = 2;
   exhausted_options.initial_primal_regularization = 1.;
   StageStructuredCandidateBackend exhausted(
      TinyStageAssembler(layout, model), exhausted_options);
   Check(
      !exhausted.solve({kkt, state, rhs, 0, false}).has_value(),
      "wrong-inertia retries unexpectedly produced a certificate");

   StageStructuredCandidateOptions invalid_options;
   invalid_options.maximum_factorization_attempts = 0;
   StageStructuredCandidateBackend invalid(
      TinyStageAssembler(layout, model), invalid_options);
   const auto invalid_result = invalid.solve({kkt, state, rhs, 1, false});
   Check(
      !invalid_result.has_value() &&
         invalid_result.error().code == EvaluationErrorCode::invalid_layout,
      "invalid retry configuration was accepted");

   StageStructuredLayout too_small_inertia = layout;
   too_small_inertia.inertia_dimension = 2;
   StageStructuredCandidateBackend too_small(
      TinyStageAssembler(too_small_inertia, model));
   Check(
      !too_small.solve({kkt, state, rhs, 1, false}).has_value(),
      "inertia dimension smaller than the structured system was accepted");

   StageStructuredLayout too_large_inertia = layout;
   too_large_inertia.inertia_dimension = 4;
   StageStructuredCandidateBackend too_large(
      TinyStageAssembler(too_large_inertia, model));
   Check(
      !too_large.solve({kkt, state, rhs, 1, false}).has_value(),
      "inertia dimension larger than the full direction was accepted");
}

void TestLazyFactoryLifecycle()
{
   const TinyStageModel model{4., 1., 3., 1., 2.};
   TinyStageModel changed_model = model;
   changed_model.revision = 92;
   PrimalDualKktOperator first_kkt = MakeKkt(model);
   PrimalDualKktOperator changed_kkt = MakeKkt(changed_model);
   const std::array<Number, 2> x{{0., 0.}};
   const std::array<Number, 1> multipliers{{0.}};
   const PrimalDualState state = MakeState(x, multipliers);
   const std::array<Number, 3> rhs{{2., -1., 3.}};
   const auto factory_state = std::make_shared<LazyFactoryState>();

   LazyCandidateFirstBackend lazy(
      CountingStageBackendFactory(model, factory_state));
   Check(
      lazy.solve({first_kkt, state, rhs, 1, false}).has_value(),
      "lazy backend initial solve failed");
   Check(
      lazy.solve({first_kkt, state, rhs, 1, false}).has_value() &&
         factory_state->creations == 1,
      "lazy backend did not reuse an unchanged KKT structure");

   factory_state->failures_remaining = 1;
   const auto failed_reconfiguration =
      lazy.solve({changed_kkt, state, rhs, 1, false});
   Check(
      !failed_reconfiguration.has_value() &&
         factory_state->creations == 2,
      "lazy backend did not report the injected reconfiguration failure");
   Check(
      lazy.solve({first_kkt, state, rhs, 1, false}).has_value() &&
         factory_state->creations == 2,
      "failed reconfiguration discarded the prior backend");
   Check(
      lazy.solve({changed_kkt, state, rhs, 1, false}).has_value() &&
         factory_state->creations == 3,
      "lazy backend did not rebuild after a structural change");

   const auto restoration = lazy.solve({changed_kkt, state, rhs, 1, true});
   Check(
      !restoration.has_value() && factory_state->creations == 4,
      "lazy factory did not distinguish restoration role");
   Check(
      lazy.solve({changed_kkt, state, rhs, 1, false}).has_value() &&
         factory_state->creations == 4,
      "restoration rejection discarded the main-problem backend");

   SharedCandidateFirstBackend erased = MakeLazyCandidateFirstBackend(
      CountingStageBackendFactory(model, factory_state));
   Check(
      erased->candidate_first_solve({first_kkt, state, rhs, 1, false})
         .has_value(),
      "AnyAny-erased lazy backend failed");
}
} // namespace

int main()
{
   try
   {
      TestHappyPathAndErasure();
      TestPreparedWorkspaceBinding();
      TestInertiaRetryChangesPrimalRegularization();
      TestCongruentCondensationCertificate();
      TestIndependentInertiaProofGate();
      TestContractFailures();
      TestLazyFactoryLifecycle();
   }
   catch( const std::exception& exception )
   {
      std::cerr << exception.what() << '\n';
      return 1;
   }
   return 0;
}
