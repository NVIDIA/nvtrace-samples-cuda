/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <chrono>
#include <cstdint>
#include <cstring>
#include <cuda.h>
#include <cuda_runtime.h>
#include <string>

class EasyCudaLauncher
{
public:
    ~EasyCudaLauncher() = default;
    EasyCudaLauncher() = default;

    bool LaunchEmptyKernel();
    bool LaunchWaitKernel(uint64_t timeNsec);
    bool CudaDriverGetVersion();
    bool CuDriverGetVersion();
    bool CuDeviceGetCount();
    bool CudaStreamTest();
    bool CuModuleTest();

    const char* GetErrorMsg()
    {
        return m_errorMsg.c_str();
    }


private:
    std::string m_errorMsg = "";
};

class VectorAddLauncher
{
public:
    enum CounterId
    {
        CudaApiCount,
        KernelLaunchCount,
        CounterIdNum
    };

    ~VectorAddLauncher();
    VectorAddLauncher() = default;

    void Initialize(int num)
    {
        m_numElements = num;
        m_size = m_numElements * sizeof(float);
        memset(m_counters, 0, sizeof(int64_t) * CounterId::CounterIdNum);
    }

    bool AllocHostMemory();
    bool FreeHostMemory();

    bool AllocAndCopyDeviceMemory();
    bool FreeDeviceMemory();

    bool LaunchKernel();
    bool LaunchAddTanhKernel();
    bool VerifyComputeResult();

    void BenchmarkEnter()
    {
        m_inBenchmark = true;
    }

    void BenchmarkExit()
    {
        m_inBenchmark = false;
    }

    const char* GetErrorMsg()
    {
        return m_errorMsg.c_str();
    }

    int64_t QueryCount(int id);

private:
    // data
    int m_numElements = 1;
    size_t m_size = 1 * sizeof(float);
    // host memory
    float* m_pHostBufferA = nullptr;
    float* m_pHostBufferB = nullptr;
    float* m_pHostBufferC = nullptr;
    // deivce memory
    float* m_pDeviceBufferA = nullptr;
    float* m_pDeviceBufferB = nullptr;
    float* m_pDeviceBufferC = nullptr;

    int64_t m_counters[CounterId::CounterIdNum];
    bool m_inBenchmark = false;
    std::string m_errorMsg = "";

    void IncrementCounterIfInBenchmark(CounterId counterId, uint64_t addend = 1)
    {
        if (m_inBenchmark)
        {
            m_counters[counterId] += addend;
        }
    }
};

class GraphLauncher
{
public:
    ~GraphLauncher() = default;
    GraphLauncher() = default;

    bool Initialize();
    void Reset();

    bool setEmptyKernel();
    bool setWaitKernel(uint64_t timeNsec);
    bool LaunchGraph();

    const char* GetErrorMsg()
    {
        return m_errorMsg.c_str();
    }

private:
    cudaStream_t m_stream;
    cudaGraph_t m_graph;
    cudaGraphExec_t m_graphExec;
    std::string m_errorMsg = "";
};
