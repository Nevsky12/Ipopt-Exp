// Copyright (C) 2026 Ipopt contributors.
// All Rights Reserved.
// This file is distributed under the Eclipse Public License.

#ifndef IPOPT_CXX23_LEGACY_AUG_SYSTEM_BACKEND_HPP
#define IPOPT_CXX23_LEGACY_AUG_SYSTEM_BACKEND_HPP

#include <ipopt/cxx23/direct_preconditioner.hpp>

#include "IpAugSystemSolver.hpp"
#include "IpException.hpp"
#include "IpTripletHelper.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <exception>
#include <expected>
#include <new>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace Ipopt::Cxx23
{
/** Stable-ABI objects needed to reuse an existing AugSystemSolver.
 *
 * The factory retains Ipopt SmartPtrs to every object. The corresponding
 * matrix and vector spaces must keep their normal Ipopt lifetime.
 */
struct LegacyAugSystemViews
{
   ::Ipopt::AugSystemSolver* solver = nullptr;
   const ::Ipopt::SymMatrix* hessian = nullptr;
   const ::Ipopt::Matrix* jacobian_equalities = nullptr;
   const ::Ipopt::Matrix* jacobian_inequalities = nullptr;
   const ::Ipopt::Vector* x_prototype = nullptr;
   const ::Ipopt::Vector* s_prototype = nullptr;
   const ::Ipopt::Vector* equality_prototype = nullptr;
   const ::Ipopt::Vector* inequality_prototype = nullptr;
   /** Optional live IpCq sigma vectors. Supplying both preserves their tags. */
   const ::Ipopt::Vector* complementarity_x_diagonal = nullptr;
   const ::Ipopt::Vector* complementarity_s_diagonal = nullptr;
};

struct LegacyAugSystemOptions
{
   /** Match PDFullSpaceSolver's accepted-inertia check while preparing. */
   bool check_inertia = true;

   /** Bound an unexpected SYMSOLVER_CALL_AGAIN loop. */
   Index call_again_limit = 16;
};

namespace legacy_aug_system_detail
{
inline EvaluationResult Failure(EvaluationErrorCode code, std::string message)
{
   return std::unexpected(EvaluationError{code, std::move(message)});
}

inline bool SameDimension(::Ipopt::Index stable, Index cxx23) noexcept
{
   if( !std::in_range<::Ipopt::Index>(cxx23) )
   {
      return false;
   }
   return stable == static_cast<::Ipopt::Index>(cxx23);
}

inline bool Finite(std::span<const Number> values) noexcept
{
   return std::ranges::all_of(
      values, [](Number value) { return std::isfinite(value); });
}

inline bool PositiveFinite(std::span<const Number> values) noexcept
{
   return std::ranges::all_of(
      values, [](Number value) { return value > 0. && std::isfinite(value); });
}
} // namespace legacy_aug_system_detail

/** Adapts the stable four-block AugSystemSolver to the C++23 eight-block
 * direct-backend contract.
 *
 * factorize() performs one zero-RHS MultiSolve because the stable interface
 * does not expose factorization separately. Unchanged matrix tags then make
 * every solve_rhs() a backsolve for StdAugSystemSolver/TSymLinearSolver and
 * GenAugSystemSolver implementations that honor their existing new-matrix
 * contract.
 *
 * Perturbation selection, inertia retries, and IncreaseQuality remain owned by
 * PDFullSpaceSolver. This backend is bound to one already selected set of
 * regularization values and rejects stable matrix tag changes.
 */
class LegacyAugSystemDirectBackend
{
public:
   LegacyAugSystemDirectBackend(const LegacyAugSystemDirectBackend&) = delete;
   LegacyAugSystemDirectBackend& operator=(const LegacyAugSystemDirectBackend&) = delete;
   LegacyAugSystemDirectBackend(LegacyAugSystemDirectBackend&&) = default;
   LegacyAugSystemDirectBackend& operator=(LegacyAugSystemDirectBackend&&) = delete;

   static EvaluationValue<LegacyAugSystemDirectBackend> Create(
      LegacyAugSystemViews          views,
      PrimalDualKktOperator&        kkt,
      PrimalDualState               state,
      LegacyAugSystemOptions        options = {}
   )
   {
      if( EvaluationResult valid = ValidateViews(views); !valid )
      {
         return std::unexpected(valid.error());
      }
      if( EvaluationResult valid = kkt.validate_state(state); !valid )
      {
         return std::unexpected(valid.error());
      }
      if( state.numeric_revision == 0 )
      {
         return std::unexpected(EvaluationError{
            EvaluationErrorCode::numeric_mismatch,
            "legacy augmented-system preparation requires a nonzero numeric revision"
         });
      }
      if( options.call_again_limit == 0 )
      {
         return std::unexpected(EvaluationError{
            EvaluationErrorCode::invalid_layout,
            "legacy augmented-system call-again limit must be positive"
         });
      }

      const PrimalDualDimensions dimensions = kkt.dimensions();
      if( EvaluationResult valid = ValidateStableDimensions(views, dimensions); !valid )
      {
         return std::unexpected(valid.error());
      }
      if( options.check_inertia && !views.solver->ProvidesInertia() )
      {
         return std::unexpected(EvaluationError{
            EvaluationErrorCode::model_failure,
            "legacy augmented-system solver cannot provide the requested inertia"
         });
      }
      if( !views.hessian->HasValidNumbers() ||
          !views.jacobian_equalities->HasValidNumbers() ||
          !views.jacobian_inequalities->HasValidNumbers() )
      {
         return std::unexpected(EvaluationError{
            EvaluationErrorCode::nonfinite_output,
            "legacy augmented-system matrix contains a nonfinite value"
         });
      }
      if( EvaluationResult valid = ValidateNumerics(state); !valid )
      {
         return std::unexpected(valid.error());
      }

      StructureFingerprintResult fingerprint = kkt.structure_fingerprint();
      if( !fingerprint )
      {
         return std::unexpected(fingerprint.error());
      }

      std::vector<Number> sigma_x(dimensions.x, 0.);
      std::vector<Number> sigma_s(dimensions.s, 0.);
      const PrimalDualLayout& layout = kkt.layout();
      for( Index i = 0; i < dimensions.z_lower; ++i )
      {
         sigma_x[layout.primal_lower_bounds[i]] +=
            state.z_lower[i] / state.slack_x_lower[i];
      }
      for( Index i = 0; i < dimensions.z_upper; ++i )
      {
         sigma_x[layout.primal_upper_bounds[i]] +=
            state.z_upper[i] / state.slack_x_upper[i];
      }
      for( Index i = 0; i < dimensions.v_lower; ++i )
      {
         sigma_s[layout.slack_lower_bounds[i]] +=
            state.v_lower[i] / state.slack_s_lower[i];
      }
      for( Index i = 0; i < dimensions.v_upper; ++i )
      {
         sigma_s[layout.slack_upper_bounds[i]] +=
            state.v_upper[i] / state.slack_s_upper[i];
      }
      if( !legacy_aug_system_detail::Finite(sigma_x) ||
          !legacy_aug_system_detail::Finite(sigma_s) )
      {
         return std::unexpected(EvaluationError{
            EvaluationErrorCode::nonfinite_output,
            "legacy augmented-system complementarity diagonal is nonfinite"
         });
      }
      const auto stable_representable = [](Number value)
      {
         return std::isfinite(static_cast<::Ipopt::Number>(value));
      };
      if( !std::ranges::all_of(sigma_x, stable_representable) ||
          !std::ranges::all_of(sigma_s, stable_representable) )
      {
         return std::unexpected(EvaluationError{
            EvaluationErrorCode::nonfinite_output,
            "legacy augmented-system complementarity diagonal is not representable by Ipopt::Number"
         });
      }
      if( EvaluationResult valid = ValidateProvidedDiagonals(
             views, sigma_x, sigma_s, dimensions);
          !valid )
      {
         return std::unexpected(valid.error());
      }

      return LegacyAugSystemDirectBackend(
         views, dimensions, layout, state, *fingerprint, std::move(sigma_x),
         std::move(sigma_s), options);
   }

   Index dimension() const noexcept
   {
      return dimensions_.total();
   }

   StructureFingerprint structure_fingerprint() const noexcept
   {
      return fingerprint_;
   }

   std::uint64_t numeric_revision() const noexcept
   {
      return numeric_revision_;
   }

   EvaluationResult factorize()
   {
      if( EvaluationResult valid = ValidateStableInputs(); !valid )
      {
         return valid;
      }
      if( factorized_ )
      {
         return {};
      }
      SetStableVectorsToZero();
      if( EvaluationResult solved = CallStableSolver(options_.check_inertia); !solved )
      {
         return solved;
      }
      if( !sol_x_->HasValidNumbers() || !sol_s_->HasValidNumbers() ||
          !sol_c_->HasValidNumbers() || !sol_d_->HasValidNumbers() )
      {
         return legacy_aug_system_detail::Failure(
            EvaluationErrorCode::nonfinite_output,
            "legacy augmented-system factor preparation returned a nonfinite solution");
      }
      factorized_ = true;
      return {};
   }

   EvaluationResult solve_rhs(
      std::span<const Number> input,
      std::span<Number>       output
   )
   {
      if( input.size() != dimension() )
      {
         return detail::DimensionMismatch("legacy direct RHS", input.size(), dimension());
      }
      if( output.size() != dimension() )
      {
         return detail::DimensionMismatch(
            "legacy direct solution", output.size(), dimension());
      }
      if( !factorized_ )
      {
         return legacy_aug_system_detail::Failure(
            EvaluationErrorCode::model_failure,
            "legacy augmented-system solve requested before factorization");
      }
      if( EvaluationResult valid = ValidateStableInputs(); !valid )
      {
         return valid;
      }

      const std::span<const Number> safe_input = detail::PreserveAliasedInput(
         input, output, input_scratch_);
      if( !legacy_aug_system_detail::Finite(safe_input) )
      {
         return legacy_aug_system_detail::Failure(
            EvaluationErrorCode::model_failure,
            "legacy augmented-system RHS contains a nonfinite value");
      }
      if( EvaluationResult packed = PackReducedRightHandSide(safe_input); !packed )
      {
         return packed;
      }
      SetStableSolutionsToZero();
      if( EvaluationResult solved = CallStableSolver(false); !solved )
      {
         return solved;
      }
      if( EvaluationResult unpacked = UnpackFullSolution(safe_input); !unpacked )
      {
         return unpacked;
      }
      std::ranges::copy(output_scratch_, output.begin());
      return {};
   }

private:
   LegacyAugSystemDirectBackend(
      LegacyAugSystemViews     views,
      PrimalDualDimensions     dimensions,
      const PrimalDualLayout&  layout,
      PrimalDualState          state,
      StructureFingerprint    fingerprint,
      std::vector<Number>      sigma_x,
      std::vector<Number>      sigma_s,
      LegacyAugSystemOptions   options
   )
      : solver_(views.solver),
        hessian_(views.hessian),
        jacobian_equalities_(views.jacobian_equalities),
        jacobian_inequalities_(views.jacobian_inequalities),
        x_prototype_(views.x_prototype),
        s_prototype_(views.s_prototype),
        equality_prototype_(views.equality_prototype),
        inequality_prototype_(views.inequality_prototype),
        hessian_tag_(views.hessian->GetTag()),
        equality_jacobian_tag_(views.jacobian_equalities->GetTag()),
        inequality_jacobian_tag_(views.jacobian_inequalities->GetTag()),
        dimensions_(dimensions),
        layout_(layout),
        fingerprint_(fingerprint),
        numeric_revision_(state.numeric_revision),
        options_(options),
        z_lower_(state.z_lower.begin(), state.z_lower.end()),
        z_upper_(state.z_upper.begin(), state.z_upper.end()),
        v_lower_(state.v_lower.begin(), state.v_lower.end()),
        v_upper_(state.v_upper.begin(), state.v_upper.end()),
        slack_x_lower_(state.slack_x_lower.begin(), state.slack_x_lower.end()),
        slack_x_upper_(state.slack_x_upper.begin(), state.slack_x_upper.end()),
        slack_s_lower_(state.slack_s_lower.begin(), state.slack_s_lower.end()),
        slack_s_upper_(state.slack_s_upper.begin(), state.slack_s_upper.end()),
        stable_x_values_(dimensions.x),
        stable_s_values_(dimensions.s),
        stable_c_values_(dimensions.y_c),
        stable_d_values_(dimensions.y_d),
        augmented_x_(dimensions.x),
        augmented_s_(dimensions.s),
        input_scratch_(dimensions.total()),
        output_scratch_(dimensions.total())
   {
      delta_x_ = static_cast<::Ipopt::Number>(state.regularization.x);
      delta_s_ = static_cast<::Ipopt::Number>(state.regularization.s);
      delta_c_ = static_cast<::Ipopt::Number>(state.regularization.c);
      delta_d_ = static_cast<::Ipopt::Number>(state.regularization.d);
      expected_negative_eigenvalues_ = static_cast<::Ipopt::Index>(
         dimensions.y_c + dimensions.y_d);

      rhs_x_ = x_prototype_->MakeNew();
      rhs_s_ = s_prototype_->MakeNew();
      rhs_c_ = equality_prototype_->MakeNew();
      rhs_d_ = inequality_prototype_->MakeNew();
      sol_x_ = x_prototype_->MakeNew();
      sol_s_ = s_prototype_->MakeNew();
      sol_c_ = equality_prototype_->MakeNew();
      sol_d_ = inequality_prototype_->MakeNew();
      if( views.complementarity_x_diagonal != nullptr )
      {
         sigma_x_ = views.complementarity_x_diagonal;
         sigma_s_ = views.complementarity_s_diagonal;
      }
      else
      {
         ::Ipopt::SmartPtr<::Ipopt::Vector> owned_sigma_x = x_prototype_->MakeNew();
         ::Ipopt::SmartPtr<::Ipopt::Vector> owned_sigma_s = s_prototype_->MakeNew();
         PutStableVector(sigma_x, stable_x_values_, *owned_sigma_x);
         PutStableVector(sigma_s, stable_s_values_, *owned_sigma_s);
         sigma_x_ = ::Ipopt::ConstPtr(owned_sigma_x);
         sigma_s_ = ::Ipopt::ConstPtr(owned_sigma_s);
      }
      sigma_x_tag_ = sigma_x_->GetTag();
      sigma_s_tag_ = sigma_s_->GetTag();

      rhs_x_vectors_[0] = ::Ipopt::ConstPtr(rhs_x_);
      rhs_s_vectors_[0] = ::Ipopt::ConstPtr(rhs_s_);
      rhs_c_vectors_[0] = ::Ipopt::ConstPtr(rhs_c_);
      rhs_d_vectors_[0] = ::Ipopt::ConstPtr(rhs_d_);
      sol_x_vectors_[0] = sol_x_;
      sol_s_vectors_[0] = sol_s_;
      sol_c_vectors_[0] = sol_c_;
      sol_d_vectors_[0] = sol_d_;
   }

   static EvaluationResult ValidateViews(LegacyAugSystemViews views)
   {
      if( views.solver == nullptr || views.hessian == nullptr ||
          views.jacobian_equalities == nullptr ||
          views.jacobian_inequalities == nullptr ||
          views.x_prototype == nullptr || views.s_prototype == nullptr ||
          views.equality_prototype == nullptr ||
          views.inequality_prototype == nullptr )
      {
         return legacy_aug_system_detail::Failure(
            EvaluationErrorCode::invalid_layout,
            "legacy augmented-system view contains a null object");
      }
      if( (views.complementarity_x_diagonal == nullptr) !=
          (views.complementarity_s_diagonal == nullptr) )
      {
         return legacy_aug_system_detail::Failure(
            EvaluationErrorCode::invalid_layout,
            "legacy augmented-system view must provide both complementarity diagonals or neither");
      }
      return {};
   }

   static EvaluationResult ValidateStableDimensions(
      LegacyAugSystemViews      views,
      PrimalDualDimensions      dimensions
   )
   {
      const bool valid =
         legacy_aug_system_detail::SameDimension(views.hessian->Dim(), dimensions.x) &&
         legacy_aug_system_detail::SameDimension(
            views.jacobian_equalities->NCols(), dimensions.x) &&
         legacy_aug_system_detail::SameDimension(
            views.jacobian_equalities->NRows(), dimensions.y_c) &&
         legacy_aug_system_detail::SameDimension(
            views.jacobian_inequalities->NCols(), dimensions.x) &&
         legacy_aug_system_detail::SameDimension(
            views.jacobian_inequalities->NRows(), dimensions.y_d) &&
         dimensions.s == dimensions.y_d &&
         legacy_aug_system_detail::SameDimension(
            views.x_prototype->Dim(), dimensions.x) &&
         legacy_aug_system_detail::SameDimension(
            views.s_prototype->Dim(), dimensions.s) &&
         legacy_aug_system_detail::SameDimension(
            views.equality_prototype->Dim(), dimensions.y_c) &&
         legacy_aug_system_detail::SameDimension(
            views.inequality_prototype->Dim(), dimensions.y_d);
      const bool valid_diagonal_dimensions =
         views.complementarity_x_diagonal == nullptr ||
         (legacy_aug_system_detail::SameDimension(
             views.complementarity_x_diagonal->Dim(), dimensions.x) &&
          legacy_aug_system_detail::SameDimension(
             views.complementarity_s_diagonal->Dim(), dimensions.s));
      if( !valid || !valid_diagonal_dimensions )
      {
         return legacy_aug_system_detail::Failure(
            EvaluationErrorCode::dimension_mismatch,
            "legacy augmented-system objects do not match the C++23 KKT dimensions");
      }
      return {};
   }

   static EvaluationResult ValidateProvidedDiagonals(
      LegacyAugSystemViews      views,
      std::span<const Number>   sigma_x,
      std::span<const Number>   sigma_s,
      PrimalDualDimensions      dimensions
   )
   {
      if( views.complementarity_x_diagonal == nullptr )
      {
         return {};
      }
      if( !views.complementarity_x_diagonal->HasValidNumbers() ||
          !views.complementarity_s_diagonal->HasValidNumbers() )
      {
         return legacy_aug_system_detail::Failure(
            EvaluationErrorCode::nonfinite_output,
            "provided legacy complementarity diagonal is nonfinite");
      }
      std::vector<::Ipopt::Number> stable_x(dimensions.x);
      std::vector<::Ipopt::Number> stable_s(dimensions.s);
      ::Ipopt::TripletHelper::FillValuesFromVector(
         views.complementarity_x_diagonal->Dim(),
         *views.complementarity_x_diagonal, stable_x.data());
      ::Ipopt::TripletHelper::FillValuesFromVector(
         views.complementarity_s_diagonal->Dim(),
         *views.complementarity_s_diagonal, stable_s.data());
      const auto agrees = [](Number expected, ::Ipopt::Number stable)
      {
         constexpr Number tolerance_factor = 64.;
         const Number actual = static_cast<Number>(stable);
         const Number arithmetic_epsilon = std::max(
            std::numeric_limits<Number>::epsilon(),
            static_cast<Number>(
               std::numeric_limits<::Ipopt::Number>::epsilon()));
         const Number tolerance = tolerance_factor *
            arithmetic_epsilon *
            std::max({Number{1.}, std::abs(expected), std::abs(actual)});
         return std::isfinite(actual) && std::abs(actual - expected) <= tolerance;
      };
      for( Index i = 0; i < dimensions.x; ++i )
      {
         if( !agrees(sigma_x[i], stable_x[i]) )
         {
            return legacy_aug_system_detail::Failure(
               EvaluationErrorCode::numeric_mismatch,
               "provided legacy x complementarity diagonal disagrees with the KKT state");
         }
      }
      for( Index i = 0; i < dimensions.s; ++i )
      {
         if( !agrees(sigma_s[i], stable_s[i]) )
         {
            return legacy_aug_system_detail::Failure(
               EvaluationErrorCode::numeric_mismatch,
               "provided legacy s complementarity diagonal disagrees with the KKT state");
         }
      }
      return {};
   }

   static EvaluationResult ValidateNumerics(PrimalDualState state)
   {
      const Number regularization[] = {
         state.regularization.x, state.regularization.s,
         state.regularization.c, state.regularization.d
      };
      if( !legacy_aug_system_detail::Finite(regularization) ||
          !legacy_aug_system_detail::Finite(state.z_lower) ||
          !legacy_aug_system_detail::Finite(state.z_upper) ||
          !legacy_aug_system_detail::Finite(state.v_lower) ||
          !legacy_aug_system_detail::Finite(state.v_upper) ||
          !legacy_aug_system_detail::PositiveFinite(state.slack_x_lower) ||
          !legacy_aug_system_detail::PositiveFinite(state.slack_x_upper) ||
          !legacy_aug_system_detail::PositiveFinite(state.slack_s_lower) ||
          !legacy_aug_system_detail::PositiveFinite(state.slack_s_upper) )
      {
         return legacy_aug_system_detail::Failure(
            EvaluationErrorCode::nonfinite_output,
            "legacy augmented-system state has nonfinite data or nonpositive slacks");
      }

      const auto representable = [](Number value)
      {
         return std::isfinite(static_cast<::Ipopt::Number>(value));
      };
      if( !std::ranges::all_of(regularization, representable) )
      {
         return legacy_aug_system_detail::Failure(
            EvaluationErrorCode::nonfinite_output,
            "legacy augmented-system regularization is not representable by Ipopt::Number");
      }
      return {};
   }

   EvaluationResult ValidateStableInputs() const
   {
      if( hessian_->GetTag() != hessian_tag_ ||
          jacobian_equalities_->GetTag() != equality_jacobian_tag_ ||
          jacobian_inequalities_->GetTag() != inequality_jacobian_tag_ ||
          sigma_x_->GetTag() != sigma_x_tag_ ||
          sigma_s_->GetTag() != sigma_s_tag_ )
      {
         return legacy_aug_system_detail::Failure(
            EvaluationErrorCode::numeric_mismatch,
            "legacy augmented-system matrix changed after backend preparation");
      }
      if( !hessian_->HasValidNumbers() ||
          !jacobian_equalities_->HasValidNumbers() ||
          !jacobian_inequalities_->HasValidNumbers() ||
          !sigma_x_->HasValidNumbers() || !sigma_s_->HasValidNumbers() )
      {
         return legacy_aug_system_detail::Failure(
            EvaluationErrorCode::nonfinite_output,
            "legacy augmented-system matrix became nonfinite");
      }
      return {};
   }

   void SetStableVectorsToZero()
   {
      rhs_x_->Set(0.);
      rhs_s_->Set(0.);
      rhs_c_->Set(0.);
      rhs_d_->Set(0.);
      SetStableSolutionsToZero();
   }

   void SetStableSolutionsToZero()
   {
      sol_x_->Set(0.);
      sol_s_->Set(0.);
      sol_c_->Set(0.);
      sol_d_->Set(0.);
   }

   EvaluationResult CallStableSolver(bool check_inertia)
   {
      for( Index attempt = 0; attempt < options_.call_again_limit; ++attempt )
      {
         ::Ipopt::ESymSolverStatus status;
         try
         {
            status = solver_->MultiSolve(
               ::Ipopt::GetRawPtr(hessian_), 1.,
               ::Ipopt::GetRawPtr(sigma_x_), delta_x_,
               ::Ipopt::GetRawPtr(sigma_s_), delta_s_,
               ::Ipopt::GetRawPtr(jacobian_equalities_), nullptr, delta_c_,
               ::Ipopt::GetRawPtr(jacobian_inequalities_), nullptr, delta_d_,
               rhs_x_vectors_, rhs_s_vectors_, rhs_c_vectors_, rhs_d_vectors_,
               sol_x_vectors_, sol_s_vectors_, sol_c_vectors_, sol_d_vectors_,
               check_inertia,
               check_inertia ? expected_negative_eigenvalues_ : 0);
         }
         catch( const std::bad_alloc& )
         {
            throw;
         }
         catch( const ::Ipopt::IpoptException& exception )
         {
            return legacy_aug_system_detail::Failure(
               EvaluationErrorCode::model_failure,
               "legacy augmented-system solver threw: " + exception.Message());
         }
         catch( const std::exception& exception )
         {
            return legacy_aug_system_detail::Failure(
               EvaluationErrorCode::model_failure,
               "legacy augmented-system solver threw: " + std::string(exception.what()));
         }
         catch( ... )
         {
            return legacy_aug_system_detail::Failure(
               EvaluationErrorCode::model_failure,
               "legacy augmented-system solver threw an unknown exception");
         }

         if( status == ::Ipopt::SYMSOLVER_SUCCESS )
         {
            return {};
         }
         if( status == ::Ipopt::SYMSOLVER_CALL_AGAIN )
         {
            continue;
         }

         std::string message = "legacy augmented-system solve failed with status " +
            std::to_string(static_cast<int>(status));
         if( status == ::Ipopt::SYMSOLVER_WRONG_INERTIA && solver_->ProvidesInertia() )
         {
            message += ", negative eigenvalues=" +
               std::to_string(solver_->NumberOfNegEVals());
         }
         return legacy_aug_system_detail::Failure(
            EvaluationErrorCode::model_failure, std::move(message));
      }
      return legacy_aug_system_detail::Failure(
         EvaluationErrorCode::model_failure,
         "legacy augmented-system solver exceeded the call-again limit");
   }

   EvaluationResult PackReducedRightHandSide(std::span<const Number> input)
   {
      const Index x_offset = 0;
      const Index s_offset = x_offset + dimensions_.x;
      const Index c_offset = s_offset + dimensions_.s;
      const Index d_offset = c_offset + dimensions_.y_c;
      const Index z_lower_offset = d_offset + dimensions_.y_d;
      const Index z_upper_offset = z_lower_offset + dimensions_.z_lower;
      const Index v_lower_offset = z_upper_offset + dimensions_.z_upper;
      const Index v_upper_offset = v_lower_offset + dimensions_.v_lower;

      std::ranges::copy(
         input.subspan(x_offset, dimensions_.x), augmented_x_.begin());
      std::ranges::copy(
         input.subspan(s_offset, dimensions_.s), augmented_s_.begin());
      for( Index i = 0; i < dimensions_.z_lower; ++i )
      {
         augmented_x_[layout_.primal_lower_bounds[i]] +=
            input[z_lower_offset + i] / slack_x_lower_[i];
      }
      for( Index i = 0; i < dimensions_.z_upper; ++i )
      {
         augmented_x_[layout_.primal_upper_bounds[i]] -=
            input[z_upper_offset + i] / slack_x_upper_[i];
      }
      for( Index i = 0; i < dimensions_.v_lower; ++i )
      {
         augmented_s_[layout_.slack_lower_bounds[i]] +=
            input[v_lower_offset + i] / slack_s_lower_[i];
      }
      for( Index i = 0; i < dimensions_.v_upper; ++i )
      {
         augmented_s_[layout_.slack_upper_bounds[i]] -=
            input[v_upper_offset + i] / slack_s_upper_[i];
      }
      if( !legacy_aug_system_detail::Finite(augmented_x_) ||
          !legacy_aug_system_detail::Finite(augmented_s_) )
      {
         return legacy_aug_system_detail::Failure(
            EvaluationErrorCode::nonfinite_output,
            "legacy augmented-system reduced RHS is nonfinite");
      }

      if( !PutStableVector(augmented_x_, stable_x_values_, *rhs_x_) ||
          !PutStableVector(augmented_s_, stable_s_values_, *rhs_s_) ||
          !PutStableVector(
             input.subspan(c_offset, dimensions_.y_c), stable_c_values_, *rhs_c_) ||
          !PutStableVector(
             input.subspan(d_offset, dimensions_.y_d), stable_d_values_, *rhs_d_) )
      {
         return legacy_aug_system_detail::Failure(
            EvaluationErrorCode::nonfinite_output,
            "legacy augmented-system RHS is not representable by Ipopt::Number");
      }
      return {};
   }

   EvaluationResult UnpackFullSolution(std::span<const Number> input)
   {
      const Index x_offset = 0;
      const Index s_offset = x_offset + dimensions_.x;
      const Index c_offset = s_offset + dimensions_.s;
      const Index d_offset = c_offset + dimensions_.y_c;
      const Index z_lower_offset = d_offset + dimensions_.y_d;
      const Index z_upper_offset = z_lower_offset + dimensions_.z_lower;
      const Index v_lower_offset = z_upper_offset + dimensions_.z_upper;
      const Index v_upper_offset = v_lower_offset + dimensions_.v_lower;

      if( !GetStableVector(*sol_x_, stable_x_values_,
             std::span<Number>(output_scratch_).subspan(x_offset, dimensions_.x)) ||
          !GetStableVector(*sol_s_, stable_s_values_,
             std::span<Number>(output_scratch_).subspan(s_offset, dimensions_.s)) ||
          !GetStableVector(*sol_c_, stable_c_values_,
             std::span<Number>(output_scratch_).subspan(c_offset, dimensions_.y_c)) ||
          !GetStableVector(*sol_d_, stable_d_values_,
             std::span<Number>(output_scratch_).subspan(d_offset, dimensions_.y_d)) )
      {
         return legacy_aug_system_detail::Failure(
            EvaluationErrorCode::nonfinite_output,
            "legacy augmented-system reduced solution is not finite in C++23 Number");
      }

      for( Index i = 0; i < dimensions_.z_lower; ++i )
      {
         output_scratch_[z_lower_offset + i] =
            (input[z_lower_offset + i] - z_lower_[i] *
               output_scratch_[x_offset + layout_.primal_lower_bounds[i]]) /
            slack_x_lower_[i];
      }
      for( Index i = 0; i < dimensions_.z_upper; ++i )
      {
         output_scratch_[z_upper_offset + i] =
            (input[z_upper_offset + i] + z_upper_[i] *
               output_scratch_[x_offset + layout_.primal_upper_bounds[i]]) /
            slack_x_upper_[i];
      }
      for( Index i = 0; i < dimensions_.v_lower; ++i )
      {
         output_scratch_[v_lower_offset + i] =
            (input[v_lower_offset + i] - v_lower_[i] *
               output_scratch_[s_offset + layout_.slack_lower_bounds[i]]) /
            slack_s_lower_[i];
      }
      for( Index i = 0; i < dimensions_.v_upper; ++i )
      {
         output_scratch_[v_upper_offset + i] =
            (input[v_upper_offset + i] + v_upper_[i] *
               output_scratch_[s_offset + layout_.slack_upper_bounds[i]]) /
            slack_s_upper_[i];
      }
      if( !legacy_aug_system_detail::Finite(output_scratch_) )
      {
         return legacy_aug_system_detail::Failure(
            EvaluationErrorCode::nonfinite_output,
            "legacy augmented-system full solution is nonfinite");
      }
      return {};
   }

   static bool PutStableVector(
      std::span<const Number>          source,
      std::vector<::Ipopt::Number>&   conversion,
      ::Ipopt::Vector&                destination
   )
   {
      for( Index i = 0; i < source.size(); ++i )
      {
         conversion[i] = static_cast<::Ipopt::Number>(source[i]);
         if( !std::isfinite(conversion[i]) )
         {
            return false;
         }
      }
      ::Ipopt::TripletHelper::PutValuesInVector(
         static_cast<::Ipopt::Index>(source.size()), conversion.data(), destination);
      return true;
   }

   static bool GetStableVector(
      const ::Ipopt::Vector&          source,
      std::vector<::Ipopt::Number>&   conversion,
      std::span<Number>               destination
   )
   {
      ::Ipopt::TripletHelper::FillValuesFromVector(
         static_cast<::Ipopt::Index>(destination.size()), source, conversion.data());
      for( Index i = 0; i < destination.size(); ++i )
      {
         destination[i] = static_cast<Number>(conversion[i]);
         if( !std::isfinite(destination[i]) )
         {
            return false;
         }
      }
      return true;
   }

   ::Ipopt::SmartPtr<::Ipopt::AugSystemSolver> solver_;
   ::Ipopt::SmartPtr<const ::Ipopt::SymMatrix> hessian_;
   ::Ipopt::SmartPtr<const ::Ipopt::Matrix> jacobian_equalities_;
   ::Ipopt::SmartPtr<const ::Ipopt::Matrix> jacobian_inequalities_;
   ::Ipopt::SmartPtr<const ::Ipopt::Vector> x_prototype_;
   ::Ipopt::SmartPtr<const ::Ipopt::Vector> s_prototype_;
   ::Ipopt::SmartPtr<const ::Ipopt::Vector> equality_prototype_;
   ::Ipopt::SmartPtr<const ::Ipopt::Vector> inequality_prototype_;

   const ::Ipopt::TaggedObject::Tag hessian_tag_;
   const ::Ipopt::TaggedObject::Tag equality_jacobian_tag_;
   const ::Ipopt::TaggedObject::Tag inequality_jacobian_tag_;
   const PrimalDualDimensions dimensions_;
   const PrimalDualLayout layout_;
   const StructureFingerprint fingerprint_;
   const std::uint64_t numeric_revision_;
   const LegacyAugSystemOptions options_;

   std::vector<Number> z_lower_;
   std::vector<Number> z_upper_;
   std::vector<Number> v_lower_;
   std::vector<Number> v_upper_;
   std::vector<Number> slack_x_lower_;
   std::vector<Number> slack_x_upper_;
   std::vector<Number> slack_s_lower_;
   std::vector<Number> slack_s_upper_;

   ::Ipopt::Number delta_x_ = 0.;
   ::Ipopt::Number delta_s_ = 0.;
   ::Ipopt::Number delta_c_ = 0.;
   ::Ipopt::Number delta_d_ = 0.;
   ::Ipopt::Index expected_negative_eigenvalues_ = 0;

   ::Ipopt::SmartPtr<::Ipopt::Vector> rhs_x_;
   ::Ipopt::SmartPtr<::Ipopt::Vector> rhs_s_;
   ::Ipopt::SmartPtr<::Ipopt::Vector> rhs_c_;
   ::Ipopt::SmartPtr<::Ipopt::Vector> rhs_d_;
   ::Ipopt::SmartPtr<::Ipopt::Vector> sol_x_;
   ::Ipopt::SmartPtr<::Ipopt::Vector> sol_s_;
   ::Ipopt::SmartPtr<::Ipopt::Vector> sol_c_;
   ::Ipopt::SmartPtr<::Ipopt::Vector> sol_d_;
   ::Ipopt::SmartPtr<const ::Ipopt::Vector> sigma_x_;
   ::Ipopt::SmartPtr<const ::Ipopt::Vector> sigma_s_;
   ::Ipopt::TaggedObject::Tag sigma_x_tag_ = 0;
   ::Ipopt::TaggedObject::Tag sigma_s_tag_ = 0;

   std::vector<::Ipopt::SmartPtr<const ::Ipopt::Vector> > rhs_x_vectors_{1};
   std::vector<::Ipopt::SmartPtr<const ::Ipopt::Vector> > rhs_s_vectors_{1};
   std::vector<::Ipopt::SmartPtr<const ::Ipopt::Vector> > rhs_c_vectors_{1};
   std::vector<::Ipopt::SmartPtr<const ::Ipopt::Vector> > rhs_d_vectors_{1};
   std::vector<::Ipopt::SmartPtr<::Ipopt::Vector> > sol_x_vectors_{1};
   std::vector<::Ipopt::SmartPtr<::Ipopt::Vector> > sol_s_vectors_{1};
   std::vector<::Ipopt::SmartPtr<::Ipopt::Vector> > sol_c_vectors_{1};
   std::vector<::Ipopt::SmartPtr<::Ipopt::Vector> > sol_d_vectors_{1};

   std::vector<::Ipopt::Number> stable_x_values_;
   std::vector<::Ipopt::Number> stable_s_values_;
   std::vector<::Ipopt::Number> stable_c_values_;
   std::vector<::Ipopt::Number> stable_d_values_;
   std::vector<Number> augmented_x_;
   std::vector<Number> augmented_s_;
   std::vector<Number> input_scratch_;
   std::vector<Number> output_scratch_;
   bool factorized_ = false;
};

inline EvaluationValue<AnyDirectSolverBackend> MakeLegacyAugSystemDirectBackend(
   LegacyAugSystemViews   views,
   PrimalDualKktOperator& kkt,
   PrimalDualState        state,
   LegacyAugSystemOptions options = {}
)
{
   EvaluationValue<LegacyAugSystemDirectBackend> backend =
      LegacyAugSystemDirectBackend::Create(views, kkt, state, options);
   if( !backend )
   {
      return std::unexpected(backend.error());
   }
   return MakeDirectSolverBackend(std::move(*backend));
}

inline EvaluationValue<PreparedDirectPreconditioner>
PrepareLegacyAugSystemPreconditioner(
   LegacyAugSystemViews   views,
   PrimalDualKktOperator& kkt,
   PrimalDualState        state,
   LegacyAugSystemOptions options = {}
)
{
   EvaluationValue<AnyDirectSolverBackend> backend =
      MakeLegacyAugSystemDirectBackend(views, kkt, state, options);
   if( !backend )
   {
      return std::unexpected(backend.error());
   }
   return PrepareDirectPreconditioner(kkt, state, std::move(*backend));
}
} // namespace Ipopt::Cxx23

#endif
