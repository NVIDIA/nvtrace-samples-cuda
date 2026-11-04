/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstdio>
#include <cstdlib>

#include <cuda.h>
#include <cuda_runtime.h>

#include <NvTrace/Target/NvTraceScopeExitGuard.h>

namespace {

    constexpr size_t kRuntimeMemsetBytes = 16 * 1024 * 1024;
    constexpr size_t kDriverMemsetBytes = 16 * 1024 * 1024;
    constexpr size_t kDriver2DWidthBytes = 1 * 1024 * 1024;
    constexpr size_t kDriver2DHeight = 2;

    bool CheckRuntimeResult(cudaError_t result, const char* call)
    {
        if (result == cudaSuccess)
        {
            return true;
        }

        std::fprintf(stderr, "%s failed: %s: %s\n", call, cudaGetErrorName(result), cudaGetErrorString(result));
        return false;
    }

    bool CheckDriverResult(CUresult result, const char* call)
    {
        if (result == CUDA_SUCCESS)
        {
            return true;
        }

        const char* name = nullptr;
        const char* message = nullptr;
        cuGetErrorName(result, &name);
        cuGetErrorString(result, &message);
        std::fprintf(stderr, "%s failed: %s: %s\n", call, name ? name : "unknown", message ? message : "unknown");
        return false;
    }

    bool RuntimeMemset()
    {
        void* dptr = nullptr;
        cudaStream_t stream = nullptr;

        if (!CheckRuntimeResult(cudaStreamCreate(&stream), "cudaStreamCreate(&stream)"))
        {
            return false;
        }
        auto streamGuard = nv::trace::ScopeExitGuard([&]() {
            CheckRuntimeResult(cudaStreamDestroy(stream), "cudaStreamDestroy(stream)");
        });

        if (!CheckRuntimeResult(cudaMalloc(&dptr, kRuntimeMemsetBytes), "cudaMalloc(&dptr, kRuntimeMemsetBytes)"))
        {
            return false;
        }
        auto memoryGuard = nv::trace::ScopeExitGuard([&]() {
            CheckRuntimeResult(cudaFree(dptr), "cudaFree(dptr)");
        });

        if (!CheckRuntimeResult(
                cudaMemsetAsync(dptr, 0x5a, kRuntimeMemsetBytes, stream),
                "cudaMemsetAsync(dptr, 0x5a, kRuntimeMemsetBytes, stream)"))
        {
            return false;
        }
        return CheckRuntimeResult(cudaStreamSynchronize(stream), "cudaStreamSynchronize(stream)");
    }

    bool DriverMemset1D()
    {
        CUdeviceptr dptr = 0;
        CUstream stream = nullptr;

        if (!CheckDriverResult(cuStreamCreate(&stream, CU_STREAM_DEFAULT), "cuStreamCreate(&stream, CU_STREAM_DEFAULT)"))
        {
            return false;
        }
        auto streamGuard = nv::trace::ScopeExitGuard([&]() {
            CheckDriverResult(cuStreamDestroy(stream), "cuStreamDestroy(stream)");
        });

        if (!CheckDriverResult(cuMemAlloc(&dptr, kDriverMemsetBytes), "cuMemAlloc(&dptr, kDriverMemsetBytes)"))
        {
            return false;
        }
        auto memoryGuard = nv::trace::ScopeExitGuard([&]() {
            CheckDriverResult(cuMemFree(dptr), "cuMemFree(dptr)");
        });

        if (!CheckDriverResult(
                cuMemsetD8Async(dptr, 0xa5, kDriverMemsetBytes, stream),
                "cuMemsetD8Async(dptr, 0xa5, kDriverMemsetBytes, stream)"))
        {
            return false;
        }
        return CheckDriverResult(cuStreamSynchronize(stream), "cuStreamSynchronize(stream)");
    }

    bool DriverMemset2D()
    {
        CUdeviceptr dptr = 0;
        CUstream stream = nullptr;
        const size_t pitch = kDriver2DWidthBytes;
        const size_t allocationBytes = kDriver2DWidthBytes * kDriver2DHeight;

        if (!CheckDriverResult(cuStreamCreate(&stream, CU_STREAM_DEFAULT), "cuStreamCreate(&stream, CU_STREAM_DEFAULT)"))
        {
            return false;
        }
        auto streamGuard = nv::trace::ScopeExitGuard([&]() {
            CheckDriverResult(cuStreamDestroy(stream), "cuStreamDestroy(stream)");
        });

        if (!CheckDriverResult(cuMemAlloc(&dptr, allocationBytes), "cuMemAlloc(&dptr, allocationBytes)"))
        {
            return false;
        }
        auto memoryGuard = nv::trace::ScopeExitGuard([&]() {
            CheckDriverResult(cuMemFree(dptr), "cuMemFree(dptr)");
        });

        if (!CheckDriverResult(
                cuMemsetD2D8Async(dptr, pitch, 0x3c, kDriver2DWidthBytes, kDriver2DHeight, stream),
                "cuMemsetD2D8Async(dptr, pitch, 0x3c, kDriver2DWidthBytes, kDriver2DHeight, stream)"))
        {
            return false;
        }
        return CheckDriverResult(cuStreamSynchronize(stream), "cuStreamSynchronize(stream)");
    }

} // namespace

int ImmediateMemset(void)
{
    if (!CheckRuntimeResult(cudaSetDevice(0), "cudaSetDevice(0)"))
    {
        return EXIT_FAILURE;
    }
    if (!CheckRuntimeResult(cudaFree(nullptr), "cudaFree(nullptr)"))
    {
        return EXIT_FAILURE;
    }

    CUcontext currentContext = nullptr;
    if (!CheckDriverResult(cuCtxGetCurrent(&currentContext), "cuCtxGetCurrent(&currentContext)"))
    {
        return EXIT_FAILURE;
    }
    if (currentContext == nullptr)
    {
        std::fprintf(stderr, "No current CUDA context after runtime initialization.\n");
        return EXIT_FAILURE;
    }

    if (!RuntimeMemset() || !DriverMemset1D() || !DriverMemset2D())
    {
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
