/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <cstring>
#include <cuda.h>
#include <cuda_runtime.h>
#include <string>

class MemsetMemcpyLauncher
{
public:
    enum CounterId
    {
        CudaApiCount,
        GraphLaunchCount,
        CounterIdNum
    };

    ~MemsetMemcpyLauncher();
    MemsetMemcpyLauncher() = default;

    void Initialize(size_t numBytes)
    {
        m_size = numBytes;
        memset(m_counters, 0, sizeof(int64_t) * CounterId::CounterIdNum);
    }

    bool AllocHostMemory();
    bool FreeHostMemory();
    bool AllocDeviceMemory();
    bool FreeDeviceMemory();

    bool MemsetDevice();
    bool MemcpyH2D();
    bool MemcpyD2H();
    bool MemcpyD2D();
    bool MemcpyH2H();
    bool MemOpWait32();
    bool MemOpWait64();
    bool MemOpWrite32();
    bool MemOpWrite64();

    bool InitializeGraph();
    void ResetGraph();
    bool RecordMemsetDeviceGraph();
    bool RecordMemcpyH2DGraph();
    bool RecordMemcpyD2HGraph();
    bool RecordMemcpyD2DGraph();
    bool RecordMemcpyH2HGraph();
    bool RecordMemsetConvertedGraph();
    bool RecordMemcpyH2DConvertedGraph();
    bool RecordMemcpyD2HConvertedGraph();
    bool RecordMemcpyD2DConvertedGraph();
    bool LaunchGraph();

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
    size_t m_size = 0;
    void* m_pHostBufferSrc = nullptr;
    void* m_pHostBufferDst = nullptr;
    void* m_pPinnedBufferSrc = nullptr;
    void* m_pPinnedBufferDst = nullptr;
    void* m_pDeviceBufferSrc = nullptr;
    void* m_pDeviceBufferDst = nullptr;

    cudaStream_t m_stream = nullptr;
    cudaGraph_t m_graph = nullptr;
    cudaGraphExec_t m_graphExec = nullptr;

    int64_t m_counters[CounterId::CounterIdNum] = {};
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
