/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <chrono>
#include <cstdio>
#include <functional>
#include <inttypes.h>
#include <set>

#include "VectorAddLauncher.h"

#define CUDA_SAFE_CALL(call_, errorMsg_)                      \
    do                                                        \
    {                                                         \
        cudaError_t err = cudaSuccess;                        \
        call_;                                                \
        err = cudaGetLastError();                             \
        if (err != cudaSuccess)                               \
        {                                                     \
            errorMsg_ = std::string(cudaGetErrorString(err)); \
            return false;                                     \
        }                                                     \
    } while (0)

#define CU_SAFE_CALL(call_, errorMsg_)                            \
    do                                                            \
    {                                                             \
        CUresult err = call_;                                     \
        if (err != CUDA_SUCCESS)                                  \
        {                                                         \
            const char* errStr = nullptr;                         \
            cuGetErrorString(err, &errStr);                       \
            errorMsg_ = std::string(errStr ? errStr : "Unknown"); \
            return false;                                         \
        }                                                         \
    } while (0)

__global__ void EmptyKernel()
{
}

__device__ uint64_t GlobalTimer64()
{
    uint64_t timestamp;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(timestamp));
    return timestamp;
}

__global__ void WaitNsecKernel(const uint64_t timeNsec)
{
    uint64_t startTimestamp = GlobalTimer64();
    uint64_t endTimestamp;
    uint64_t elapsedNsec;

    // Spin until the requested time has elapsed.
    do
    {
        endTimestamp = GlobalTimer64();
        elapsedNsec = endTimestamp - startTimestamp;
    } while (elapsedNsec < timeNsec);

    return;
}

// ================================================================================
// Easy CUDA Launcher
// ================================================================================
bool EasyCudaLauncher::LaunchEmptyKernel()
{
    int threadsPerBlock = 256;
    int blocksPerGrid = 1;
    CUDA_SAFE_CALL((EmptyKernel<<<blocksPerGrid, threadsPerBlock>>>()), m_errorMsg);

    return true;
}

bool EasyCudaLauncher::LaunchWaitKernel(uint64_t timeNsec)
{
    int threadsPerBlock = 256;
    int blocksPerGrid = 1;
    CUDA_SAFE_CALL((WaitNsecKernel<<<blocksPerGrid, threadsPerBlock>>>(timeNsec)), m_errorMsg);

    return true;
}

bool EasyCudaLauncher::CudaDriverGetVersion()
{
    int driverVersion = 0;
    CUDA_SAFE_CALL((cudaDriverGetVersion(&driverVersion)), m_errorMsg);

    return true;
}

bool EasyCudaLauncher::CuDriverGetVersion()
{
    int driverVersion = 0;
    CU_SAFE_CALL((cuDriverGetVersion(&driverVersion)), m_errorMsg);

    return true;
}

bool EasyCudaLauncher::CuDeviceGetCount()
{
    int count = 0;
    CU_SAFE_CALL((cuDeviceGetCount(&count)), m_errorMsg);

    return true;
}

bool EasyCudaLauncher::CudaStreamTest()
{
    cudaStream_t stream;
    CUDA_SAFE_CALL((cudaStreamCreate(&stream)), m_errorMsg);
    CUDA_SAFE_CALL((cudaStreamDestroy(stream)), m_errorMsg);

    return true;
}

bool EasyCudaLauncher::CuModuleTest()
{
    CUmodule module;
    cuModuleLoad(&module, "kernel.cu"); // intentionally not checking because it's not a real kernel
    cuModuleUnload(module);

    return true;
}

/**
 * CUDA Kernel Device code
 *
 * Computes the vector addition of A and B into C. The 3 vectors have the same
 * number of elements numElements.
 */
__global__ void VectorAddKernel(const float* A, const float* B, float* C, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements)
    {
        C[i] = A[i] + B[i] + 0.0f;
    }
}

__global__ void VectorAddTanhKernel(const float* A, const float* B, float* C, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements)
    {
        // Start with an initial value
        float result = A[i] + B[i];

        for (int j = 0; j < 200; j++)
        {
            result = tanhf(result);
        }

        C[i] = result;
    }
}

// ================================================================================
// VectorAddLauncher
// ================================================================================
VectorAddLauncher::~VectorAddLauncher()
{
    FreeHostMemory();
    FreeDeviceMemory();
}

bool VectorAddLauncher::AllocHostMemory()
{
    m_pHostBufferA = (float*)malloc(m_size);
    m_pHostBufferB = (float*)malloc(m_size);
    m_pHostBufferC = (float*)malloc(m_size);

    if (!m_pHostBufferA || !m_pHostBufferB || !m_pHostBufferC)
    {
        m_errorMsg = "Failed to allocate host vectors!";
        return false;
    }

    for (int i = 0; i < m_numElements; ++i)
    {
        m_pHostBufferA[i] = rand() / (float)RAND_MAX;
        m_pHostBufferB[i] = rand() / (float)RAND_MAX;
    }

    return true;
}

bool VectorAddLauncher::FreeHostMemory()
{
    auto safeFree = [](float*& ptr) -> void {
        if (ptr)
        {
            free(ptr);
            ptr = nullptr;
        }
    };

    safeFree(m_pHostBufferA);
    safeFree(m_pHostBufferB);
    safeFree(m_pHostBufferC);

    return true;
}

bool VectorAddLauncher::AllocAndCopyDeviceMemory()
{
    m_pDeviceBufferA = nullptr;
    CUDA_SAFE_CALL((cudaMalloc((void**)&m_pDeviceBufferA, m_size)), m_errorMsg);
    m_pDeviceBufferB = nullptr;
    CUDA_SAFE_CALL((cudaMalloc((void**)&m_pDeviceBufferB, m_size)), m_errorMsg);
    m_pDeviceBufferC = nullptr;
    CUDA_SAFE_CALL((cudaMalloc((void**)&m_pDeviceBufferC, m_size)), m_errorMsg);

    CUDA_SAFE_CALL((cudaMemcpy(m_pDeviceBufferA, m_pHostBufferA, m_size, cudaMemcpyHostToDevice)), m_errorMsg);
    CUDA_SAFE_CALL((cudaMemcpy(m_pDeviceBufferB, m_pHostBufferB, m_size, cudaMemcpyHostToDevice)), m_errorMsg);

    IncrementCounterIfInBenchmark(CounterId::CudaApiCount, 6);

    return true;
}

bool VectorAddLauncher::FreeDeviceMemory()
{
    auto safeCudaFree = [](float*& ptr) -> void {
        if (ptr)
        {
            cudaFree(ptr);
            ptr = nullptr;
        }
    };

    safeCudaFree(m_pDeviceBufferA);
    safeCudaFree(m_pDeviceBufferB);
    safeCudaFree(m_pDeviceBufferC);

    IncrementCounterIfInBenchmark(CounterId::CudaApiCount, 3);

    return true;
}

bool VectorAddLauncher::LaunchKernel()
{
    int threadsPerBlock = 256;
    int blocksPerGrid = (m_numElements + threadsPerBlock - 1) / threadsPerBlock;
    CUDA_SAFE_CALL((VectorAddKernel<<<blocksPerGrid, threadsPerBlock>>>(m_pDeviceBufferA, m_pDeviceBufferB, m_pDeviceBufferC, m_numElements)), m_errorMsg);

    IncrementCounterIfInBenchmark(CounterId::KernelLaunchCount);

    return true;
}

bool VectorAddLauncher::LaunchAddTanhKernel()
{
    int threadsPerBlock = 256;
    int blocksPerGrid = (m_numElements + threadsPerBlock - 1) / threadsPerBlock;
    CUDA_SAFE_CALL((VectorAddTanhKernel<<<blocksPerGrid, threadsPerBlock>>>(m_pDeviceBufferA, m_pDeviceBufferB, m_pDeviceBufferC, m_numElements)), m_errorMsg);

    IncrementCounterIfInBenchmark(CounterId::KernelLaunchCount);

    return true;
}

bool VectorAddLauncher::VerifyComputeResult()
{
    CUDA_SAFE_CALL((cudaMemcpy(m_pHostBufferC, m_pDeviceBufferC, m_size, cudaMemcpyDeviceToHost)), m_errorMsg);

    for (int i = 0; i < m_numElements; ++i)
    {
        if (fabs(m_pHostBufferA[i] + m_pHostBufferB[i] - m_pHostBufferC[i]) > 1e-5)
        {
            m_errorMsg = "Result verification failed at element" + std::to_string(i);
            return false;
        }
    }

    return true;
}

int64_t VectorAddLauncher::QueryCount(int id)
{
    if (id >= CounterId::CounterIdNum)
    {
        printf("Error: Query ID %d is bigger than the CounterId nums %d.\n", id, CounterId::CounterIdNum);
        return 0;
    }

    return m_counters[id];
}

// ================================================================================
// GraphLauncher
// ================================================================================
bool GraphLauncher::Initialize()
{
    CUDA_SAFE_CALL((cudaStreamCreate(&m_stream)), m_errorMsg);

    return true;
}

void GraphLauncher::Reset()
{
    cudaStreamSynchronize(m_stream);
    cudaGraphDestroy(m_graph);
    cudaGraphExecDestroy(m_graphExec);
    cudaStreamDestroy(m_stream);
}

bool GraphLauncher::setEmptyKernel()
{
    CUDA_SAFE_CALL((cudaStreamBeginCapture(m_stream, cudaStreamCaptureModeGlobal)), m_errorMsg);

    constexpr int threadsPerBlock = 256;
    constexpr int blocksPerGrid = 1;
    EmptyKernel<<<blocksPerGrid, threadsPerBlock, 0, m_stream>>>();

    CUDA_SAFE_CALL((cudaStreamEndCapture(m_stream, &m_graph)), m_errorMsg);
    CUDA_SAFE_CALL((cudaGraphInstantiate(&m_graphExec, m_graph, nullptr, nullptr, 0)), m_errorMsg);

    return true;
}

bool GraphLauncher::setWaitKernel(uint64_t timeNsec)
{
    CUDA_SAFE_CALL((cudaStreamBeginCapture(m_stream, cudaStreamCaptureModeGlobal)), m_errorMsg);

    constexpr int threadsPerBlock = 256;
    constexpr int blocksPerGrid = 1;
    WaitNsecKernel<<<blocksPerGrid, threadsPerBlock, 0, m_stream>>>(timeNsec);

    CUDA_SAFE_CALL((cudaStreamEndCapture(m_stream, &m_graph)), m_errorMsg);
    CUDA_SAFE_CALL((cudaGraphInstantiate(&m_graphExec, m_graph, nullptr, nullptr, 0)), m_errorMsg);

    return true;
}

bool GraphLauncher::LaunchGraph()
{
    CUDA_SAFE_CALL((cudaGraphLaunch(m_graphExec, m_stream)), m_errorMsg);

    return true;
}
