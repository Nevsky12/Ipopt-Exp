// Copyright (C) 2026 Ipopt contributors.
// All Rights Reserved.
// This file is distributed under the Eclipse Public License.

#ifndef IPOPT_CXX23_DIRECT_PRECONDITIONER_HPP
#define IPOPT_CXX23_DIRECT_PRECONDITIONER_HPP

#include <ipopt/cxx23/primal_dual_kkt_operator.hpp>

#include <anyany/anyany.hpp>
#include <anyany/anyany_macro.hpp>

#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstdint>
#include <expected>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace Ipopt::Cxx23
{
template <class Backend>
concept DirectSolverBackend = requires(
   Backend&                backend,
   const Backend&          const_backend,
   std::span<const Number> input,
   std::span<Number>       output
)
{
   { const_backend.dimension() } noexcept -> std::same_as<Index>;
   { const_backend.structure_fingerprint() } noexcept
      -> std::same_as<StructureFingerprint>;
   { const_backend.numeric_revision() } noexcept -> std::same_as<std::uint64_t>;
   { backend.factorize() } -> std::same_as<EvaluationResult>;
   { backend.solve_rhs(input, output) } -> std::same_as<EvaluationResult>;
};

template <DirectSolverBackend Backend>
class DirectSolverBackendAdapter
{
public:
   explicit DirectSolverBackendAdapter(Backend backend)
      : backend_(std::move(backend))
   {
   }

   Index direct_solver_dimension() const noexcept
   {
      return backend_.dimension();
   }

   StructureFingerprint direct_solver_structure_fingerprint() const noexcept
   {
      return backend_.structure_fingerprint();
   }

   std::uint64_t direct_solver_numeric_revision() const noexcept
   {
      return backend_.numeric_revision();
   }

   EvaluationResult direct_solver_factorize()
   {
      return backend_.factorize();
   }

   EvaluationResult direct_solver_solve_rhs(
      std::span<const Number> input,
      std::span<Number>       output
   )
   {
      return backend_.solve_rhs(input, output);
   }

private:
   Backend backend_;
};

anyany_method(direct_solver_dimension,
   (const& self) requires(self.direct_solver_dimension())->Index);
anyany_method(direct_solver_structure_fingerprint,
   (const& self) requires(self.direct_solver_structure_fingerprint())->StructureFingerprint);
anyany_method(direct_solver_numeric_revision,
   (const& self) requires(self.direct_solver_numeric_revision())->std::uint64_t);
anyany_method(direct_solver_factorize,
   (&self) requires(self.direct_solver_factorize())->EvaluationResult);
anyany_method(direct_solver_solve_rhs,
   (&self, std::span<const Number> input, std::span<Number> output)
      requires(self.direct_solver_solve_rhs(input, output))->EvaluationResult);

using AnyDirectSolverBackend = aa::any_with<
   aa::move,
   aa::type_info,
   direct_solver_dimension,
   direct_solver_structure_fingerprint,
   direct_solver_numeric_revision,
   direct_solver_factorize,
   direct_solver_solve_rhs>;

template <DirectSolverBackend Backend>
AnyDirectSolverBackend MakeDirectSolverBackend(Backend backend)
{
   return DirectSolverBackendAdapter<Backend>(std::move(backend));
}

struct DirectPreconditionerStatistics
{
   Index factorizations = 0;
   Index solve_requests = 0;
   Index successful_solves = 0;
   Index evaluation_failures = 0;
};

/** A factor-once, solve-many right preconditioner.
 *
 * The backend is already bound to one numeric KKT state. Construction is only
 * possible through Prepare(), which validates structure and numeric revision
 * and performs exactly one factorization. Every Krylov application thereafter
 * is solve_rhs only. Output is committed only after a successful finite solve.
 */
class PreparedDirectPreconditioner
{
public:
   PreparedDirectPreconditioner(const PreparedDirectPreconditioner&) = delete;
   PreparedDirectPreconditioner& operator=(const PreparedDirectPreconditioner&) = delete;
   PreparedDirectPreconditioner(PreparedDirectPreconditioner&&) = default;
   PreparedDirectPreconditioner& operator=(PreparedDirectPreconditioner&&) = delete;

   static EvaluationValue<PreparedDirectPreconditioner> Prepare(
      PrimalDualKktOperator& kkt,
      PrimalDualState        state,
      AnyDirectSolverBackend backend
   )
   {
      StructureFingerprintResult kkt_fingerprint = kkt.structure_fingerprint();
      if( !kkt_fingerprint )
      {
         return std::unexpected(kkt_fingerprint.error());
      }
      if( EvaluationResult valid_state = kkt.validate_state(state); !valid_state )
      {
         return std::unexpected(valid_state.error());
      }
      if( backend.direct_solver_dimension() != kkt.flat_dimension() )
      {
         return std::unexpected(EvaluationError{
            EvaluationErrorCode::dimension_mismatch,
            "direct backend dimension does not match the KKT operator"
         });
      }
      if( backend.direct_solver_structure_fingerprint() != *kkt_fingerprint )
      {
         return std::unexpected(EvaluationError{
            EvaluationErrorCode::structure_mismatch,
            "direct backend structure does not match the KKT operator"
         });
      }
      if( state.numeric_revision == 0 )
      {
         return std::unexpected(EvaluationError{
            EvaluationErrorCode::numeric_mismatch,
            "prepared direct preconditioning requires a nonzero numeric revision"
         });
      }
      if( backend.direct_solver_numeric_revision() != state.numeric_revision )
      {
         return std::unexpected(EvaluationError{
            EvaluationErrorCode::numeric_mismatch,
            "direct backend numeric revision does not match the KKT state"
         });
      }
      if( EvaluationResult factorized = backend.direct_solver_factorize(); !factorized )
      {
         return std::unexpected(factorized.error());
      }
      return PreparedDirectPreconditioner(
         std::move(backend), kkt.flat_dimension(), *kkt_fingerprint,
         state.numeric_revision);
   }

   Index dimension() const noexcept
   {
      return dimension_;
   }

   StructureFingerprint structure_fingerprint() const noexcept
   {
      return structure_fingerprint_;
   }

   std::uint64_t numeric_revision() const noexcept
   {
      return numeric_revision_;
   }

   const DirectPreconditionerStatistics& statistics() const noexcept
   {
      return statistics_;
   }

   EvaluationResult validate(
      PrimalDualKktOperator& kkt,
      PrimalDualState        state
   )
   {
      StructureFingerprintResult current = kkt.structure_fingerprint();
      if( !current )
      {
         return std::unexpected(current.error());
      }
      if( kkt.flat_dimension() != dimension_ || *current != structure_fingerprint_ )
      {
         return std::unexpected(EvaluationError{
            EvaluationErrorCode::structure_mismatch,
            "prepared direct preconditioner does not match the KKT structure"
         });
      }
      if( state.numeric_revision == 0 || state.numeric_revision != numeric_revision_ )
      {
         return std::unexpected(EvaluationError{
            EvaluationErrorCode::numeric_mismatch,
            "prepared direct preconditioner does not match the KKT numeric revision"
         });
      }
      return {};
   }

   EvaluationResult operator()(
      Index,
      std::span<const Number> input,
      std::span<Number>       output
   )
   {
      if( input.size() != dimension_ )
      {
         return detail::DimensionMismatch("preconditioner input", input.size(), dimension_);
      }
      if( output.size() != dimension_ )
      {
         return detail::DimensionMismatch("preconditioner output", output.size(), dimension_);
      }

      ++statistics_.solve_requests;
      if( EvaluationResult solved = backend_.direct_solver_solve_rhs(input, output_scratch_); !solved )
      {
         ++statistics_.evaluation_failures;
         return solved;
      }
      if( !std::ranges::all_of(
             output_scratch_, [](Number value) { return std::isfinite(value); }) )
      {
         ++statistics_.evaluation_failures;
         return std::unexpected(EvaluationError{
            EvaluationErrorCode::nonfinite_output,
            "direct preconditioner returned a nonfinite solution"
         });
      }
      std::ranges::copy(output_scratch_, output.begin());
      ++statistics_.successful_solves;
      return {};
   }

private:
   PreparedDirectPreconditioner(
      AnyDirectSolverBackend backend,
      Index                  dimension,
      StructureFingerprint   structure_fingerprint,
      std::uint64_t          numeric_revision
   )
      : backend_(std::move(backend)),
        dimension_(dimension),
        structure_fingerprint_(structure_fingerprint),
        numeric_revision_(numeric_revision),
        output_scratch_(dimension)
   {
      statistics_.factorizations = 1;
   }

   AnyDirectSolverBackend backend_;
   const Index dimension_;
   const StructureFingerprint structure_fingerprint_;
   const std::uint64_t numeric_revision_;
   std::vector<Number> output_scratch_;
   DirectPreconditionerStatistics statistics_;
};

inline EvaluationValue<PreparedDirectPreconditioner> PrepareDirectPreconditioner(
   PrimalDualKktOperator& kkt,
   PrimalDualState        state,
   AnyDirectSolverBackend backend
)
{
   return PreparedDirectPreconditioner::Prepare(kkt, state, std::move(backend));
}
} // namespace Ipopt::Cxx23

#endif
