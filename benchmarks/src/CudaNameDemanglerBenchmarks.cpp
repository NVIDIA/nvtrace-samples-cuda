/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <NvTrace/Offline/Exporter/NvTraceDemanglerConvenience.h>

#include <benchmark/benchmark.h>

#include <cstddef>
#include <cstring>

namespace nv::trace {

    static constexpr size_t kMaxDemangledNameLength = 16 * 1024;

    static constexpr char kShortPointerName[] = "_Z6squarePi";

    static constexpr char kCublasGemvName[] =
        "_ZN8internal5gemvx6kernelIii13__nv_bfloat16S2_S2_fLb0ELb1ELb1ELb0ELi7ELb0E"
        "18cublasGemvParamsExIi30cublasGemvTensorStridedBatchedIKS2_ES6_S4_IS2_EfEEENSt9"
        "enable_ifIXntT5_EvE4typeET11_";

    static constexpr char kPytorchUnrolledName[] =
        "_ZN2at6native27unrolled_elementwise_kernelIZZZNS0_23direct_copy_kernel_cudaERNS_"
        "18TensorIteratorBaseEENKUlvE1_clEvENKUlvE5_clEvEUlfE_St5arrayIPcLm2EELi4E23"
        "TrivialOffsetCalculatorILi1EjESB_NS0_6memory12LoadWithCastILi1EEENSC_13StoreWithCast"
        "ILi1EEEEEviT_T0_T2_T3_T4_T5_";

    static constexpr char kPytorchVectorizedName[] =
        "_ZN2at6native29vectorized_elementwise_kernelILi4EZNS0_51_GLOBAL__N__7b48bba2_12_"
        "PowKernel_cu_2f7922d3_2269929pow_tensor_scalar_kernel_implIffEEvRNS_18"
        "TensorIteratorBaseET0_EUlfE_St5arrayIPcLm2EEEEviS6_T1_";

    static constexpr char kPytorchElementwiseName[] =
        "_ZN2at6native18elementwise_kernelILi128ELi4EZNS0_22gpu_kernel_impl_nocastINS0_13"
        "BinaryFunctorIN3c108BFloat16ES5_S5_NS0_15binary_internal10MulFunctorIfEEEEEEvRNS_"
        "18TensorIteratorBaseERKT_EUliE_EEviT1_";

    static constexpr char kCuTileConstraintName[] =
        "cutile_python_generated_kernel_with_representative_constraints_Kt2_"
        "A2u8_3i4v8_g0iu_A1i64_w_T2Si32T1Sf32_Lg0iA1i32_1t1_F40091eb851eb851f_I_7_B1";

    static void BM_DemangleKernelName(benchmark::State& benchmarkState, const char* pMangledName)
    {
        const size_t mangledNameLength = std::strlen(pMangledName);
        KernelNameDemangler kernelNameDemangler;
        kernelNameDemangler.Initialize(SIZE_MAX, kMaxDemangledNameLength, mangledNameLength, kMaxDemangledNameLength);

        const char* pValidationOutput = DemangleKernelName(kernelNameDemangler, pMangledName, mangledNameLength);
        if (pValidationOutput == pMangledName || kernelNameDemangler.stagedKernelName.empty())
        {
            benchmarkState.SkipWithError("Representative name failed to demangle");
            return;
        }
        const size_t demangledNameLength = kernelNameDemangler.stagedKernelName.length();
        const uint8_t* pStableScratchSpace = kernelNameDemangler.scratchSpace.data();
        const char* pStableOutputSpace = kernelNameDemangler.stagedKernelName.c_str();

        for (auto benchmarkIteration : benchmarkState)
        {
            static_cast<void>(benchmarkIteration);
            const char* pDemangledName = DemangleKernelName(kernelNameDemangler, pMangledName, mangledNameLength);
            benchmark::DoNotOptimize(pDemangledName);
            benchmark::DoNotOptimize(kernelNameDemangler.stagedKernelName.c_str());
            benchmark::ClobberMemory();
        }

        if (kernelNameDemangler.scratchSpace.data() != pStableScratchSpace
            || kernelNameDemangler.stagedKernelName.c_str() != pStableOutputSpace)
        {
            benchmarkState.SkipWithError("A reusable benchmark buffer was reallocated");
            return;
        }

        benchmarkState.SetItemsProcessed(benchmarkState.iterations());
        benchmarkState.SetBytesProcessed(benchmarkState.iterations() * static_cast<int64_t>(mangledNameLength));
        benchmarkState.counters["input_bytes"] = static_cast<double>(mangledNameLength);
        benchmarkState.counters["output_bytes"] = static_cast<double>(demangledNameLength);
    }

    BENCHMARK_CAPTURE(BM_DemangleKernelName, ShortPointer, kShortPointerName);
    BENCHMARK_CAPTURE(BM_DemangleKernelName, CublasGemv, kCublasGemvName);
    BENCHMARK_CAPTURE(BM_DemangleKernelName, PytorchUnrolled, kPytorchUnrolledName);
    BENCHMARK_CAPTURE(BM_DemangleKernelName, PytorchVectorized, kPytorchVectorizedName);
    BENCHMARK_CAPTURE(BM_DemangleKernelName, PytorchElementwise, kPytorchElementwiseName);
    BENCHMARK_CAPTURE(BM_DemangleKernelName, CuTileConstraints, kCuTileConstraintName);

} // namespace nv::trace

BENCHMARK_MAIN();
