/***************************************************************************************************
 * Copyright (c) 2017-2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted
 * provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright notice, this list of
 *       conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright notice, this list of
 *       conditions and the following disclaimer in the documentation and/or other materials
 *       provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the names of its contributors may be used
 *       to endorse or promote products derived from this software without specific prior written
 *       permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TOR (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
/* \file
   \brief Defines operations for reduction operation in CUTLASS Library.
*/
  
#include "cutlass/cutlass.h"
#include "cutlass/library/library.h"
#include "cutlass/library/manifest.h"

#include "reduction_operation.h"

namespace cutlass {
namespace library {

// naming convention initialize_reduce_[ReductionOp]_[EpilogueOp]_[ElementWorkspace]_[ElementAccumulator]_[ElementOutput]

void initialize_reduce_add_linear_combination_f32_f32_f16(Manifest &manifest) {

  using ElementWorkspace = float; 
  using ElementAccumulator = float;
  using ElementOutput = cutlass::half_t;
  using ElementCompute = float;

  using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombination<
    ElementOutput,
    128 / cutlass::sizeof_bits<ElementOutput>::value,
    ElementAccumulator,
    ElementCompute
  >;

  using ReductionOp = cutlass::reduction::thread::ReduceAdd<
    ElementAccumulator, 
    typename EpilogueOutputOp::ElementAccumulator,
    EpilogueOutputOp::kCount
  >;

  using Operation_reduce_add_linear_combination_f32_f32_f16 = cutlass::reduction::device::ReduceSplitK<
    cutlass::reduction::kernel::ReduceSplitK<
      cutlass::MatrixShape<4, 32 * EpilogueOutputOp::kCount>,
      EpilogueOutputOp,
      ReductionOp
    >
  >;

  manifest.append(new ReductionOperation<
    Operation_reduce_add_linear_combination_f32_f32_f16>(
      "reduce_add_linear_combination_f32_f32_f16"
  ));
}


void initialize_reduce_add_linear_combination_f32_f32_f32(Manifest &manifest) {

  using ElementWorkspace = float; 
  using ElementAccumulator = float;
  using ElementOutput = float;
  using ElementCompute = float;

  using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombination<
    ElementOutput,
    128 / cutlass::sizeof_bits<ElementOutput>::value,
    ElementAccumulator,
    ElementCompute
  >;

  using ReductionOp = cutlass::reduction::thread::ReduceAdd<
    ElementAccumulator, 
    typename EpilogueOutputOp::ElementAccumulator,
    EpilogueOutputOp::kCount
  >;

  using Operation_reduce_add_linear_combination_f32_f32_f32 = cutlass::reduction::device::ReduceSplitK<
    cutlass::reduction::kernel::ReduceSplitK<
      cutlass::MatrixShape<4, 32 * EpilogueOutputOp::kCount>,
      EpilogueOutputOp,
      ReductionOp
    >
  >;

  manifest.append(new ReductionOperation<
    Operation_reduce_add_linear_combination_f32_f32_f32>(
      "reduce_add_linear_combination_f32_f32_f32"
  ));
}

void initialize_reduce_add_linear_combination_cf32_cf32_cf32(Manifest &manifest) {

  using ElementWorkspace = cutlass::complex<float>; 
  using ElementAccumulator = cutlass::complex<float>;
  using ElementOutput = cutlass::complex<float>;
  using ElementCompute = cutlass::complex<float>;

  using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombination<
    ElementOutput,
    128 / cutlass::sizeof_bits<ElementOutput>::value,
    ElementAccumulator,
    ElementCompute
  >;

  using ReductionOp = cutlass::reduction::thread::ReduceAdd<
    ElementAccumulator, 
    typename EpilogueOutputOp::ElementAccumulator,
    EpilogueOutputOp::kCount
  >;

  using Operation_reduce_add_linear_combination_cf32_cf32_cf32 = cutlass::reduction::device::ReduceSplitK<
    cutlass::reduction::kernel::ReduceSplitK<
      cutlass::MatrixShape<4, 32 * EpilogueOutputOp::kCount>,
      EpilogueOutputOp,
      ReductionOp
    >
  >;

  manifest.append(new ReductionOperation<
    Operation_reduce_add_linear_combination_cf32_cf32_cf32>(
      "reduce_add_linear_combination_cf32_cf32_cf32"
  ));
}


} 
}
