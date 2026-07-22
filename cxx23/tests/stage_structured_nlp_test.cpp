// Copyright (C) 2026 Ipopt contributors.
// All Rights Reserved.
// This file is distributed under the Eclipse Public License.

#include <ipopt/cxx23/stage_structured_nlp.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
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

std::vector<OptimalControlStageDimensions> Dimensions()
{
   return {
      {2, 3, 1, 2},
      {1, 3, 0, 1},
      {0, 3, 2, 0}
   };
}

StageNlpOrdering Ordering()
{
   return {
      .primal = {2, 0, 1, 3, 4, 8, 5, 6, 7, 11, 9, 10},
      .dynamics = {3, 4, 5, 0, 1, 2},
      .path_equalities = {6, 8, 10},
      .path_inequalities = {7, 9, 11}
   };
}

NlpStructure SourceStructure()
{
   return {12, 12, 0, 0, 17};
}

StageNlpTopology Topology()
{
   return StageNlpTopology(SourceStructure(), Dimensions(), Ordering(), 23);
}

void CheckOffsets(std::span<const Index> actual, std::span<const Index> expected)
{
   Check(
      std::ranges::equal(actual, expected),
      "stage topology produced incorrect packed offsets");
}

void TestTopologyAndFingerprint()
{
   const StageNlpTopology topology = Topology();
   Check(topology.configured(), "valid OCP topology was rejected");
   Check(topology.stages().size() == 3, "wrong OCP stage count");
   Check(
      topology.dynamics_constraints() == 6 &&
         topology.path_equality_constraints() == 3 &&
         topology.path_inequality_constraints() == 3,
      "wrong OCP constraint counts");
   const StageDerivativeStorage storage = topology.derivative_storage();
   Check(
      storage.stage_hessians == 50 &&
         storage.dynamics_jacobians_transposed == 27 &&
         storage.path_equality_jacobians_transposed == 11 &&
         storage.path_inequality_jacobians_transposed == 14,
      "wrong packed OCP derivative storage");

   const std::array<Index, 4> primal_offsets{{0, 5, 9, 12}};
   const std::array<Index, 3> dynamics_offsets{{0, 3, 6}};
   const std::array<Index, 4> path_equality_offsets{{0, 1, 1, 3}};
   const std::array<Index, 4> path_inequality_offsets{{0, 2, 3, 3}};
   const std::array<Index, 4> hessian_offsets{{0, 25, 41, 50}};
   const std::array<Index, 3> dynamics_jacobian_offsets{{0, 15, 27}};
   const std::array<Index, 4> equality_jacobian_offsets{{0, 5, 5, 11}};
   const std::array<Index, 4> inequality_jacobian_offsets{{0, 10, 14, 14}};
   CheckOffsets(topology.primal_offsets(), primal_offsets);
   CheckOffsets(topology.dynamics_offsets(), dynamics_offsets);
   CheckOffsets(topology.path_equality_offsets(), path_equality_offsets);
   CheckOffsets(topology.path_inequality_offsets(), path_inequality_offsets);
   CheckOffsets(topology.hessian_offsets(), hessian_offsets);
   CheckOffsets(topology.dynamics_jacobian_offsets(), dynamics_jacobian_offsets);
   CheckOffsets(
      topology.path_equality_jacobian_offsets(), equality_jacobian_offsets);
   CheckOffsets(
      topology.path_inequality_jacobian_offsets(), inequality_jacobian_offsets);

   const StructureFingerprintResult first = topology.fingerprint();
   const StructureFingerprintResult same = Topology().fingerprint();
   Check(first.has_value() && same == first, "equal stage topologies hash differently");

   StageNlpOrdering changed_order = Ordering();
   std::swap(changed_order.primal[0], changed_order.primal[1]);
   const StageNlpTopology reordered(
      SourceStructure(), Dimensions(), std::move(changed_order), 23);
   const StructureFingerprintResult reordered_fingerprint = reordered.fingerprint();
   Check(
      reordered_fingerprint.has_value() && *reordered_fingerprint != *first,
      "stage ordering was omitted from the fingerprint");
   const StageNlpTopology revised(
      SourceStructure(), Dimensions(), Ordering(), 24);
   Check(
      revised.fingerprint().has_value() && revised.fingerprint() != first,
      "stage revision was omitted from the fingerprint");
}

struct ProviderState
{
   Index calls = 0;
   bool fail = false;
   bool nonfinite = false;
};

class DenseStageProvider
{
public:
   DenseStageProvider(
      StageNlpTopology topology,
      std::shared_ptr<ProviderState> state
   )
      : topology_(std::move(topology)),
        state_(std::move(state))
   {
   }

   const StageNlpTopology& stage_nlp_topology() const noexcept
   {
      return topology_;
   }

   EvaluationResult eval_stage_derivatives(
      KktEvaluationPoint,
      StageDerivativeBuffers buffers
   )
   {
      ++state_->calls;
      Number next = 1.;
      for( std::span<Number> values : {
              buffers.stage_hessians,
              buffers.dynamics_jacobians_transposed,
              buffers.path_equality_jacobians_transposed,
              buffers.path_inequality_jacobians_transposed} )
      {
         for( Number& value : values )
         {
            value = next;
            next += 1.;
         }
      }
      if( state_->nonfinite )
      {
         buffers.stage_hessians[0] =
            std::numeric_limits<Number>::quiet_NaN();
      }
      if( state_->fail )
      {
         return std::unexpected(EvaluationError{
            EvaluationErrorCode::model_failure,
            "injected stage derivative failure"
         });
      }
      return {};
   }

private:
   StageNlpTopology topology_;
   std::shared_ptr<ProviderState> state_;
};

static_assert(StageDerivativeProvider<DenseStageProvider>);

void TestPreparedDerivativeProvider()
{
   const auto state = std::make_shared<ProviderState>();
   PreparedStageDerivativeProvider prepared = PrepareStageDerivativeProvider(
      DenseStageProvider(Topology(), state));
   const std::array<Number, 12> x{};
   const std::array<Number, 12> multipliers{};
   const EvaluationValue<StageDerivativeView> evaluated =
      prepared.evaluate({x, 1., multipliers});
   Check(evaluated.has_value(), "valid stage derivative evaluation failed");
   Check(
      evaluated->stage_hessians.size() == 50 &&
         evaluated->dynamics_jacobians_transposed.size() == 27 &&
         evaluated->path_equality_jacobians_transposed.size() == 11 &&
         evaluated->path_inequality_jacobians_transposed.size() == 14,
      "stage derivative view has the wrong sizes");
   Check(
      evaluated->stage_hessians.front() == 1. &&
         evaluated->dynamics_jacobians_transposed.front() == 51. &&
         evaluated->path_equality_jacobians_transposed.front() == 78. &&
         evaluated->path_inequality_jacobians_transposed.back() == 102.,
      "stage derivative packing was corrupted");
   Check(state->calls == 1, "stage provider was not called exactly once");

   const std::array<Number, 11> short_x{};
   const auto wrong_dimension = prepared.evaluate({short_x, 1., multipliers});
   Check(
      !wrong_dimension.has_value() && state->calls == 1,
      "wrong stage point dimensions entered the provider");

   state->fail = true;
   const auto failed = prepared.evaluate({x, 1., multipliers});
   Check(
      !failed.has_value() &&
         failed.error().code == EvaluationErrorCode::model_failure,
      "stage provider failure was swallowed");
   state->fail = false;
   state->nonfinite = true;
   const auto nonfinite = prepared.evaluate({x, 1., multipliers});
   Check(
      !nonfinite.has_value() &&
         nonfinite.error().code == EvaluationErrorCode::nonfinite_output,
      "nonfinite stage derivative was accepted");
}

void TestInvalidTopologies()
{
   const StageNlpTopology empty(SourceStructure(), {}, {}, 1);
   Check(!empty.configured(), "empty stage topology was accepted");
   Check(!empty.fingerprint().has_value(), "invalid topology was fingerprinted");

   const StageNlpTopology zero_revision(
      SourceStructure(), Dimensions(), Ordering(), 0);
   Check(!zero_revision.configured(), "zero stage revision was accepted");

   std::vector<OptimalControlStageDimensions> zero_state = Dimensions();
   zero_state[1].states = 0;
   const StageNlpTopology invalid_state(
      SourceStructure(), std::move(zero_state), Ordering(), 1);
   Check(!invalid_state.configured(), "zero-dimensional state was accepted");

   NlpStructure wrong_source = SourceStructure();
   ++wrong_source.variables;
   const StageNlpTopology wrong_totals(
      wrong_source, Dimensions(), Ordering(), 1);
   Check(!wrong_totals.configured(), "wrong generic NLP totals were accepted");

   StageNlpOrdering duplicate_primal = Ordering();
   duplicate_primal.primal[0] = duplicate_primal.primal[1];
   const StageNlpTopology duplicate_variable(
      SourceStructure(), Dimensions(), std::move(duplicate_primal), 1);
   Check(!duplicate_variable.configured(), "duplicate primal map was accepted");

   StageNlpOrdering duplicate_constraint = Ordering();
   duplicate_constraint.path_inequalities[0] =
      duplicate_constraint.path_equalities[0];
   const StageNlpTopology duplicate_row(
      SourceStructure(), Dimensions(), std::move(duplicate_constraint), 1);
   Check(!duplicate_row.configured(), "duplicate constraint row was accepted");

   std::vector<OptimalControlStageDimensions> overflowing{
      {std::numeric_limits<Index>::max(), 1, 0, 0}
   };
   const StageNlpTopology overflow(
      {0, 0, 0, 0, 1}, std::move(overflowing), {}, 1);
   Check(!overflow.configured(), "overflowing stage dimensions were accepted");

   const auto invalid_state_storage = std::make_shared<ProviderState>();
   PreparedStageDerivativeProvider invalid_prepared =
      PrepareStageDerivativeProvider(
         DenseStageProvider(empty, invalid_state_storage));
   const std::array<Number, 12> x{};
   const std::array<Number, 12> multipliers{};
   Check(
      !invalid_prepared.evaluate({x, 1., multipliers}).has_value() &&
         invalid_state_storage->calls == 0,
      "invalid topology entered the derivative provider");
}
} // namespace

int main()
{
   try
   {
      TestTopologyAndFingerprint();
      TestPreparedDerivativeProvider();
      TestInvalidTopologies();
   }
   catch( const std::exception& exception )
   {
      std::cerr << exception.what() << '\n';
      return 1;
   }
   return 0;
}
