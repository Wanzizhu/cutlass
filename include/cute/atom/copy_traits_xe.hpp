#pragma once

#include <cute/arch/copy.hpp>

#include <cute/layout.hpp>

namespace cute
{

////////////////////////////////////////////////////////////////////////////////////////////////////

// Element copy selector
template <class SrcTensor, class DstTensor>
CUTE_HOST_DEVICE constexpr
auto
select_elementwise_copy(SrcTensor const&, DstTensor const&)
{
  using SrcType = typename SrcTensor::value_type;
  using DstType = typename DstTensor::value_type;

  return UniversalCopy<SrcType,DstType>{};
}

}
