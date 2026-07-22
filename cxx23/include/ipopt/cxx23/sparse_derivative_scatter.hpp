// Copyright (C) 2026 Ipopt contributors.
// All Rights Reserved.
// This file is distributed under the Eclipse Public License.

#ifndef IPOPT_CXX23_SPARSE_DERIVATIVE_SCATTER_HPP
#define IPOPT_CXX23_SPARSE_DERIVATIVE_SCATTER_HPP

#include <ipopt/cxx23/any_nlp_problem.hpp>

#include <algorithm>
#include <span>
#include <vector>

namespace Ipopt::Cxx23::detail
{
/** Immutable sparse-to-packed scatter with a unique-target fast path.
 *
 * Duplicate sparse coordinates are legal: those plans retain the original
 * zero-then-accumulate operation order. A unique plan can overwrite every
 * mapped target and only has to reset the structural holes.
 */
class SparseDerivativeScatterPlan
{
public:
   struct Entry
   {
      Index source = 0;
      Index target = 0;
   };

   void push_back(Entry entry)
   {
      entries_.push_back(entry);
   }

   bool prepare(Index target_size)
   {
      std::vector<bool> seen(target_size, false);
      unique_targets_ = true;
      structural_zeros_.clear();
      for( const Entry entry : entries_ )
      {
         if( entry.target >= target_size )
         {
            return false;
         }
         if( seen[entry.target] )
         {
            unique_targets_ = false;
         }
         else
         {
            seen[entry.target] = true;
         }
      }
      if( unique_targets_ )
      {
         for( Index target = 0; target < target_size; ++target )
         {
            if( !seen[target] )
            {
               structural_zeros_.push_back(target);
            }
         }
      }
      return true;
   }

   void write(
      std::span<const Number> source,
      std::span<Number>       target
   ) const noexcept
   {
      if( !unique_targets_ )
      {
         std::ranges::fill(target, Number{0.});
         for( const Entry entry : entries_ )
         {
            target[entry.target] += source[entry.source];
         }
         return;
      }
      for( const Index target_index : structural_zeros_ )
      {
         target[target_index] = Number{0.};
      }
      for( const Entry entry : entries_ )
      {
         // Match the former +0 followed by +=, including signed-zero rules.
         target[entry.target] = Number{0.} + source[entry.source];
      }
   }

   bool unique_targets() const noexcept
   {
      return unique_targets_;
   }

private:
   std::vector<Entry> entries_;
   std::vector<Index> structural_zeros_;
   bool unique_targets_ = false;
};
} // namespace Ipopt::Cxx23::detail

#endif
