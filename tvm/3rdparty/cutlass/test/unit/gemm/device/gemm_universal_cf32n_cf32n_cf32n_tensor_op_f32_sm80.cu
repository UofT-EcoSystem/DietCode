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
/*! \file
    \brief Tests for device-wide GEMM interface
*/

#include <iostream>

#include "../../common/cutlass_unit_test.h"
#include "cutlass/cutlass.h"

#include "cutlass/gemm/device/gemm_universal.h"

#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/host/gemm.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/tensor_view_io.h"

#include "testbed_universal.h"

////////////////////////////////////////////////////////////////////////////////

#if defined(CUTLASS_ARCH_MMA_SM80_SUPPORTED)

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_Device_GemmUniversal_cf32n_cf32t_cf32n_tensor_op_f32, 64x64x16_32x32x16) {

  using Element = cutlass::complex<float>;

  using Gemm = cutlass::gemm::device::GemmUniversal<
    Element, 
    cutlass::layout::ColumnMajor, 
    Element,
    cutlass::layout::RowMajor, 
    Element, 
    cutlass::layout::ColumnMajor,
    Element, 
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<32, 32, 16>,
    cutlass::gemm::GemmShape<16, 16, 16>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    cutlass::epilogue::thread::LinearCombination<
      Element, 
      1,
      Element, 
      Element>,
    cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle, 
    3,
    1,
    1,
    cutlass::arch::OpMultiplyAddComplex,
    cutlass::ComplexTransform::kNone,
    cutlass::ComplexTransform::kNone
  >;

  EXPECT_TRUE(test::gemm::device::TestAllGemmUniversal<Gemm>());
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_Device_GemmUniversal_cf32n_cf32h_cf32n_tensor_op_f32, 64x64x16_32x32x16) {

  using Element = cutlass::complex<float>;

  using Gemm = cutlass::gemm::device::GemmUniversal<
    Element, 
    cutlass::layout::ColumnMajor, 
    Element,
    cutlass::layout::RowMajor, 
    Element, 
    cutlass::layout::ColumnMajor,
    Element, 
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<32, 32, 16>,
    cutlass::gemm::GemmShape<16, 16, 16>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    cutlass::epilogue::thread::LinearCombination<
      Element, 
      1,
      Element, 
      Element>,
    cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle, 
    3,
    1,
    1,
    cutlass::arch::OpMultiplyAddComplex,
    cutlass::ComplexTransform::kNone,
    cutlass::ComplexTransform::kConjugate
  >;

  EXPECT_TRUE(test::gemm::device::TestAllGemmUniversal<Gemm>());
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_Device_GemmUniversal_cf32h_cf32t_cf32n_tensor_op_f32, 64x64x16_32x32x16) {

  using Element = cutlass::complex<float>;

  using Gemm = cutlass::gemm::device::GemmUniversal<
    Element, 
    cutlass::layout::RowMajor, 
    Element,
    cutlass::layout::RowMajor, 
    Element, 
    cutlass::layout::ColumnMajor,
    Element, 
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<32, 32, 16>,
    cutlass::gemm::GemmShape<16, 16, 16>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    cutlass::epilogue::thread::LinearCombination<
      Element, 
      1,
      Element, 
      Element>,
    cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle, 
    3,
    1,
    1,
    cutlass::arch::OpMultiplyAddComplex,
    cutlass::ComplexTransform::kConjugate,
    cutlass::ComplexTransform::kNone
  >;

  EXPECT_TRUE(test::gemm::device::TestAllGemmUniversal<Gemm>());
}
////////////////////////////////////////////////////////////////////////////////

TEST(SM80_Device_GemmUniversal_cf32h_cf32c_cf32n_tensor_op_f32, 64x64x16_32x32x16) {

  using Element = cutlass::complex<float>;

  using Gemm = cutlass::gemm::device::GemmUniversal<
    Element, 
    cutlass::layout::RowMajor, 
    Element,
    cutlass::layout::ColumnMajor, 
    Element, 
    cutlass::layout::ColumnMajor,
    Element, 
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<32, 32, 16>,
    cutlass::gemm::GemmShape<16, 16, 16>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    cutlass::epilogue::thread::LinearCombination<
      Element, 
      1,
      Element, 
      Element>,
    cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle, 
    3,
    1,
    1,
    cutlass::arch::OpMultiplyAddComplex,
    cutlass::ComplexTransform::kConjugate,
    cutlass::ComplexTransform::kConjugate
  >;

  EXPECT_TRUE(test::gemm::device::TestAllGemmUniversal<Gemm>());
}

////////////////////////////////////////////////////////////////////////////////

#endif // #if defined(CUTLASS_ARCH_MMA_SM80_SUPPORTED)

////////////////////////////////////////////////////////////////////////////////

