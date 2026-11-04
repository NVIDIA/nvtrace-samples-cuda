/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <atomic>
#include <cstdio>
#include <cstdlib>

#include <cuda.h>
#include <cuda_runtime.h>

#define CUDA_DRIVER_API
#include <helper_cuda.h>

#include <NvTrace/Target/NvTraceScopeExitGuard.h>

namespace {

    void CUDART_CB EagerRuntimeHostFn(void* userData)
    {
        auto* pRan = static_cast<std::atomic<bool>*>(userData);
        pRan->store(true, std::memory_order_release);
    }

    void CUDAAPI EagerDriverHostFn(void* userData)
    {
        auto* pRan = static_cast<std::atomic<bool>*>(userData);
        pRan->store(true, std::memory_order_release);
    }

    void CUDART_CB GraphHostFn(void* userData)
    {
        auto* pRan = static_cast<std::atomic<bool>*>(userData);
        pRan->store(true, std::memory_order_release);
    }

    // Eager path: cudaLaunchHostFunc outside of graph capture.
    bool EagerRuntimeHostCallback()
    {
        std::atomic<bool> hostRan{false};
        cudaStream_t stream = nullptr;

        if (checkCudaErrorsReturn(cudaStreamCreate(&stream)))
        {
            return false;
        }
        auto streamGuard = nv::trace::ScopeExitGuard([&]() {
            checkCudaErrorsReturn(cudaStreamDestroy(stream));
        });

        if (checkCudaErrorsReturn(cudaLaunchHostFunc(stream, EagerRuntimeHostFn, &hostRan)))
        {
            return false;
        }
        if (checkCudaErrorsReturn(cudaStreamSynchronize(stream)))
        {
            return false;
        }

        if (!hostRan.load(std::memory_order_acquire))
        {
            std::fprintf(stderr, "Eager runtime host callback did not run.\n");
            return false;
        }

        std::printf("Eager runtime host callback completed.\n");
        return true;
    }

    // Eager path: cuLaunchHostFunc (driver API) outside of graph capture.
    bool EagerDriverHostCallback()
    {
        std::atomic<bool> hostRan{false};
        CUstream stream = nullptr;

        if (checkCudaErrorsReturn(cuStreamCreate(&stream, CU_STREAM_DEFAULT)))
        {
            return false;
        }
        auto streamGuard = nv::trace::ScopeExitGuard([&]() {
            checkCudaErrorsReturn(cuStreamDestroy(stream));
        });

        if (checkCudaErrorsReturn(cuLaunchHostFunc(stream, EagerDriverHostFn, &hostRan)))
        {
            return false;
        }
        if (checkCudaErrorsReturn(cuStreamSynchronize(stream)))
        {
            return false;
        }

        if (!hostRan.load(std::memory_order_acquire))
        {
            std::fprintf(stderr, "Eager driver host callback did not run.\n");
            return false;
        }

        std::printf("Eager driver host callback completed.\n");
        return true;
    }

    // Graph path: cudaGraphAddHostNode + instantiate + launch.
    bool GraphHostNodeCallback()
    {
        std::atomic<bool> hostRan{false};
        cudaGraph_t graph = nullptr;
        cudaGraphExec_t graphExec = nullptr;
        cudaStream_t stream = nullptr;

        if (checkCudaErrorsReturn(cudaGraphCreate(&graph, 0)))
        {
            return false;
        }
        auto graphGuard = nv::trace::ScopeExitGuard([&]() {
            if (graph != nullptr)
            {
                checkCudaErrorsReturn(cudaGraphDestroy(graph));
            }
        });

        cudaHostNodeParams hostParams = {};
        hostParams.fn = GraphHostFn;
        hostParams.userData = &hostRan;

        cudaGraphNode_t hostNode = nullptr;
        if (checkCudaErrorsReturn(cudaGraphAddHostNode(&hostNode, graph, nullptr, 0, &hostParams)))
        {
            return false;
        }

        if (checkCudaErrorsReturn(cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0)))
        {
            return false;
        }
        auto graphExecGuard = nv::trace::ScopeExitGuard([&]() {
            if (graphExec != nullptr)
            {
                checkCudaErrorsReturn(cudaGraphExecDestroy(graphExec));
            }
        });

        if (checkCudaErrorsReturn(cudaStreamCreate(&stream)))
        {
            return false;
        }
        auto streamGuard = nv::trace::ScopeExitGuard([&]() {
            checkCudaErrorsReturn(cudaStreamDestroy(stream));
        });

        if (checkCudaErrorsReturn(cudaGraphLaunch(graphExec, stream)))
        {
            return false;
        }
        if (checkCudaErrorsReturn(cudaStreamSynchronize(stream)))
        {
            return false;
        }

        if (!hostRan.load(std::memory_order_acquire))
        {
            std::fprintf(stderr, "Graph host node callback did not run.\n");
            return false;
        }

        std::printf("Graph host node callback completed.\n");
        return true;
    }

} // namespace

int HostCallbacks(void)
{
    if (checkCudaErrorsReturn(cudaSetDevice(0)))
    {
        return EXIT_FAILURE;
    }
    // Force CUDA runtime context creation so the driver API path has a current context.
    if (checkCudaErrorsReturn(cudaFree(nullptr)))
    {
        return EXIT_FAILURE;
    }

    CUcontext currentContext = nullptr;
    if (checkCudaErrorsReturn(cuCtxGetCurrent(&currentContext)))
    {
        return EXIT_FAILURE;
    }
    if (currentContext == nullptr)
    {
        std::fprintf(stderr, "No current CUDA context after runtime initialization.\n");
        return EXIT_FAILURE;
    }

    if (!EagerRuntimeHostCallback() || !EagerDriverHostCallback() || !GraphHostNodeCallback())
    {
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
