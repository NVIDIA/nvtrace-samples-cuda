/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "MemsetMemcpyLauncher.h"

#include <cstdio>

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

namespace {

    bool CreateConditionalGraphWithBody(CUgraph& graph, CUgraph& bodyGraph, std::string& errorMsg)
    {
        CUcontext ctx = nullptr;
        CU_SAFE_CALL(cuCtxGetCurrent(&ctx), errorMsg);

        CUgraph createdGraph = nullptr;
        CU_SAFE_CALL(cuGraphCreate(&createdGraph, 0), errorMsg);

        CUgraphConditionalHandle conditionalHandle = 0;
        CU_SAFE_CALL(
            cuGraphConditionalHandleCreate(&conditionalHandle, createdGraph, ctx, 1, CU_GRAPH_COND_ASSIGN_DEFAULT),
            errorMsg);

        CUgraphNodeParams conditionalNodeParams = {};
        conditionalNodeParams.type = CU_GRAPH_NODE_TYPE_CONDITIONAL;
        conditionalNodeParams.conditional.handle = conditionalHandle;
        conditionalNodeParams.conditional.type = CU_GRAPH_COND_TYPE_IF;
        conditionalNodeParams.conditional.size = 1;
        conditionalNodeParams.conditional.ctx = ctx;

        CUgraphNode conditionalNode = nullptr;
        CU_SAFE_CALL(cuGraphAddNode(&conditionalNode, createdGraph, nullptr, nullptr, 0, &conditionalNodeParams), errorMsg);

        graph = createdGraph;
        bodyGraph = conditionalNodeParams.conditional.phGraph_out[0];
        if (bodyGraph == nullptr)
        {
            errorMsg = "Conditional graph body was not created";
            return false;
        }

        return true;
    }

    bool InstantiateDriverGraph(CUgraph graph, cudaGraphExec_t& graphExec, std::string& errorMsg)
    {
        CUgraphExec createdGraphExec = nullptr;
        CU_SAFE_CALL(cuGraphInstantiate(&createdGraphExec, graph, 0), errorMsg);
        graphExec = createdGraphExec;
        return true;
    }

} // namespace

MemsetMemcpyLauncher::~MemsetMemcpyLauncher()
{
    ResetGraph();
    FreeHostMemory();
    FreeDeviceMemory();
}

bool MemsetMemcpyLauncher::AllocHostMemory()
{
    m_pHostBufferSrc = malloc(m_size);
    m_pHostBufferDst = malloc(m_size);

    if (!m_pHostBufferSrc || !m_pHostBufferDst)
    {
        m_errorMsg = "Failed to allocate host buffers!";
        return false;
    }

    CUDA_SAFE_CALL((cudaMallocHost(&m_pPinnedBufferSrc, m_size)), m_errorMsg);
    CUDA_SAFE_CALL((cudaMallocHost(&m_pPinnedBufferDst, m_size)), m_errorMsg);

    return true;
}

bool MemsetMemcpyLauncher::FreeHostMemory()
{
    auto safeFree = [](void*& ptr) {
        if (ptr)
        {
            free(ptr);
            ptr = nullptr;
        }
    };

    auto safeCudaFreeHost = [](void*& ptr) {
        if (ptr)
        {
            cudaFreeHost(ptr);
            ptr = nullptr;
        }
    };

    safeFree(m_pHostBufferSrc);
    safeFree(m_pHostBufferDst);
    safeCudaFreeHost(m_pPinnedBufferSrc);
    safeCudaFreeHost(m_pPinnedBufferDst);

    return true;
}

bool MemsetMemcpyLauncher::AllocDeviceMemory()
{
    m_pDeviceBufferSrc = nullptr;
    CUDA_SAFE_CALL((cudaMalloc(&m_pDeviceBufferSrc, m_size)), m_errorMsg);
    m_pDeviceBufferDst = nullptr;
    CUDA_SAFE_CALL((cudaMalloc(&m_pDeviceBufferDst, m_size)), m_errorMsg);

    CUDA_SAFE_CALL((cudaMemset(m_pDeviceBufferSrc, 0, m_size)), m_errorMsg);
    CUDA_SAFE_CALL((cudaMemset(m_pDeviceBufferDst, 0, m_size)), m_errorMsg);

    IncrementCounterIfInBenchmark(CudaApiCount, 4);
    return true;
}

bool MemsetMemcpyLauncher::FreeDeviceMemory()
{
    auto safeCudaFree = [](void*& ptr) {
        if (ptr)
        {
            cudaFree(ptr);
            ptr = nullptr;
        }
    };

    safeCudaFree(m_pDeviceBufferSrc);
    safeCudaFree(m_pDeviceBufferDst);

    IncrementCounterIfInBenchmark(CudaApiCount, 2);

    return true;
}

bool MemsetMemcpyLauncher::MemsetDevice()
{
    CUDA_SAFE_CALL((cudaMemset(m_pDeviceBufferDst, 0, m_size)), m_errorMsg);
    IncrementCounterIfInBenchmark(CudaApiCount);
    return true;
}

bool MemsetMemcpyLauncher::MemcpyH2D()
{
    CUDA_SAFE_CALL((cudaMemcpy(m_pDeviceBufferDst, m_pHostBufferSrc, m_size, cudaMemcpyHostToDevice)), m_errorMsg);
    IncrementCounterIfInBenchmark(CudaApiCount);
    return true;
}

bool MemsetMemcpyLauncher::MemcpyD2H()
{
    CUDA_SAFE_CALL((cudaMemcpy(m_pHostBufferDst, m_pDeviceBufferSrc, m_size, cudaMemcpyDeviceToHost)), m_errorMsg);
    IncrementCounterIfInBenchmark(CudaApiCount);
    return true;
}

bool MemsetMemcpyLauncher::MemcpyD2D()
{
    CUDA_SAFE_CALL((cudaMemcpy(m_pDeviceBufferDst, m_pDeviceBufferSrc, m_size, cudaMemcpyDeviceToDevice)), m_errorMsg);
    IncrementCounterIfInBenchmark(CudaApiCount);
    return true;
}

bool MemsetMemcpyLauncher::MemcpyH2H()
{
    // cudaMemcpy with cudaMemcpyHostToHost is handled entirely inside libcudart (plain memcpy),
    // never reaching the driver's H2H path (cuimemcpy.c:2475). cuMemcpy via the driver API with
    // page-locked buffers (which have valid CUDA virtual addresses) exercises that path.
    CUresult result = cuMemcpy(
        reinterpret_cast<CUdeviceptr>(m_pPinnedBufferDst),
        reinterpret_cast<CUdeviceptr>(m_pPinnedBufferSrc),
        m_size);
    if (result != CUDA_SUCCESS)
    {
        const char* errStr = nullptr;
        cuGetErrorString(result, &errStr);
        m_errorMsg = errStr ? errStr : "cuMemcpy failed";
        return false;
    }
    IncrementCounterIfInBenchmark(CudaApiCount);
    return true;
}

bool MemsetMemcpyLauncher::MemOpWait32()
{
    CUstream stream = nullptr;
    const uint64_t value = 0;
    CUresult result = cuStreamWaitValue32(stream, (CUdeviceptr)m_pDeviceBufferDst, value, 0);
    if (result != CUDA_SUCCESS)
    {
        const char* errStr = nullptr;
        cuGetErrorString(result, &errStr);
        m_errorMsg = errStr ? errStr : "cuStreamWaitValue32 failed";
        return false;
    }
    IncrementCounterIfInBenchmark(CudaApiCount);
    return true;
}

bool MemsetMemcpyLauncher::MemOpWait64()
{
    CUstream stream = nullptr;
    const uint64_t value = 0;
    CUresult result = cuStreamWaitValue64(stream, (CUdeviceptr)m_pDeviceBufferDst, value, 0);
    if (result != CUDA_SUCCESS)
    {
        const char* errStr = nullptr;
        cuGetErrorString(result, &errStr);
        m_errorMsg = errStr ? errStr : "cuStreamWaitValue64 failed";
        return false;
    }
    IncrementCounterIfInBenchmark(CudaApiCount);
    return true;
}

bool MemsetMemcpyLauncher::MemOpWrite32()
{
    CUstream stream = nullptr;
    const uint32_t value = 0x76543210ull;
    CUresult result = cuStreamWriteValue32(stream, (CUdeviceptr)m_pDeviceBufferDst, value, 0);
    if (result != CUDA_SUCCESS)
    {
        const char* errStr = nullptr;
        cuGetErrorString(result, &errStr);
        m_errorMsg = errStr ? errStr : "cuStreamWriteValue32 failed";
        return false;
    }
    IncrementCounterIfInBenchmark(CudaApiCount);
    return true;
}

bool MemsetMemcpyLauncher::MemOpWrite64()
{
    CUstream stream = nullptr;
    const uint64_t value = 0xFEDCBA9876543210ull;
    CUresult result = cuStreamWriteValue64(stream, (CUdeviceptr)m_pDeviceBufferDst, value, 0);
    if (result != CUDA_SUCCESS)
    {
        const char* errStr = nullptr;
        cuGetErrorString(result, &errStr);
        m_errorMsg = errStr ? errStr : "cuStreamWriteValue64 failed";
        return false;
    }
    IncrementCounterIfInBenchmark(CudaApiCount);
    return true;
}

bool MemsetMemcpyLauncher::InitializeGraph()
{
    CUDA_SAFE_CALL((cudaStreamCreate(&m_stream)), m_errorMsg);
    return true;
}

void MemsetMemcpyLauncher::ResetGraph()
{
    if (m_stream)
    {
        cudaStreamSynchronize(m_stream);
    }
    if (m_graphExec)
    {
        cudaGraphExecDestroy(m_graphExec);
        m_graphExec = nullptr;
    }
    if (m_graph)
    {
        cudaGraphDestroy(m_graph);
        m_graph = nullptr;
    }
    if (m_stream)
    {
        cudaStreamDestroy(m_stream);
        m_stream = nullptr;
    }
}

bool MemsetMemcpyLauncher::RecordMemsetDeviceGraph()
{
    CUDA_SAFE_CALL((cudaStreamBeginCapture(m_stream, cudaStreamCaptureModeGlobal)), m_errorMsg);
    CUDA_SAFE_CALL((cudaMemsetAsync(m_pDeviceBufferDst, 0, m_size, m_stream)), m_errorMsg);
    CUDA_SAFE_CALL((cudaStreamEndCapture(m_stream, &m_graph)), m_errorMsg);
    CUDA_SAFE_CALL((cudaGraphInstantiate(&m_graphExec, m_graph, nullptr, nullptr, 0)), m_errorMsg);
    return true;
}

bool MemsetMemcpyLauncher::RecordMemcpyH2DGraph()
{
    CUDA_SAFE_CALL((cudaStreamBeginCapture(m_stream, cudaStreamCaptureModeGlobal)), m_errorMsg);
    CUDA_SAFE_CALL((cudaMemcpyAsync(m_pDeviceBufferDst, m_pHostBufferSrc, m_size, cudaMemcpyHostToDevice, m_stream)), m_errorMsg);
    CUDA_SAFE_CALL((cudaStreamEndCapture(m_stream, &m_graph)), m_errorMsg);
    CUDA_SAFE_CALL((cudaGraphInstantiate(&m_graphExec, m_graph, nullptr, nullptr, 0)), m_errorMsg);
    return true;
}

bool MemsetMemcpyLauncher::RecordMemcpyD2HGraph()
{
    CUDA_SAFE_CALL((cudaStreamBeginCapture(m_stream, cudaStreamCaptureModeGlobal)), m_errorMsg);
    CUDA_SAFE_CALL((cudaMemcpyAsync(m_pHostBufferDst, m_pDeviceBufferSrc, m_size, cudaMemcpyDeviceToHost, m_stream)), m_errorMsg);
    CUDA_SAFE_CALL((cudaStreamEndCapture(m_stream, &m_graph)), m_errorMsg);
    CUDA_SAFE_CALL((cudaGraphInstantiate(&m_graphExec, m_graph, nullptr, nullptr, 0)), m_errorMsg);
    return true;
}

bool MemsetMemcpyLauncher::RecordMemcpyD2DGraph()
{
    CUDA_SAFE_CALL((cudaStreamBeginCapture(m_stream, cudaStreamCaptureModeGlobal)), m_errorMsg);
    CUDA_SAFE_CALL((cudaMemcpyAsync(m_pDeviceBufferDst, m_pDeviceBufferSrc, m_size, cudaMemcpyDeviceToDevice, m_stream)), m_errorMsg);
    CUDA_SAFE_CALL((cudaStreamEndCapture(m_stream, &m_graph)), m_errorMsg);
    CUDA_SAFE_CALL((cudaGraphInstantiate(&m_graphExec, m_graph, nullptr, nullptr, 0)), m_errorMsg);
    return true;
}

bool MemsetMemcpyLauncher::RecordMemcpyH2HGraph()
{
    // Match the direct H2H benchmark: use the driver API with page-locked buffers so the
    // memcpy node exercises the driver's H2H path.
    CUDA_SAFE_CALL((cudaStreamBeginCapture(m_stream, cudaStreamCaptureModeGlobal)), m_errorMsg);

    CUresult result = cuMemcpyAsync(
        reinterpret_cast<CUdeviceptr>(m_pPinnedBufferDst),
        reinterpret_cast<CUdeviceptr>(m_pPinnedBufferSrc),
        m_size,
        m_stream);
    if (result != CUDA_SUCCESS)
    {
        const char* errStr = nullptr;
        cuGetErrorString(result, &errStr);
        m_errorMsg = errStr ? errStr : "cuMemcpyAsync failed";
        cudaStreamEndCapture(m_stream, &m_graph);
        return false;
    }

    CUDA_SAFE_CALL((cudaStreamEndCapture(m_stream, &m_graph)), m_errorMsg);
    CUDA_SAFE_CALL((cudaGraphInstantiate(&m_graphExec, m_graph, nullptr, nullptr, 0)), m_errorMsg);
    return true;
}

bool MemsetMemcpyLauncher::RecordMemsetConvertedGraph()
{
    CUgraph graph = nullptr;
    CUgraph bodyGraph = nullptr;
    if (!CreateConditionalGraphWithBody(graph, bodyGraph, m_errorMsg))
    {
        return false;
    }

    CUcontext ctx = nullptr;
    CU_SAFE_CALL(cuCtxGetCurrent(&ctx), m_errorMsg);

    CUgraphNodeParams memsetNodeParams = {};
    memsetNodeParams.type = CU_GRAPH_NODE_TYPE_MEMSET;
    memsetNodeParams.memset.dst = reinterpret_cast<CUdeviceptr>(m_pDeviceBufferDst);
    memsetNodeParams.memset.pitch = m_size;
    memsetNodeParams.memset.value = 0;
    memsetNodeParams.memset.elementSize = 1;
    memsetNodeParams.memset.width = m_size;
    memsetNodeParams.memset.height = 1;
    memsetNodeParams.memset.ctx = ctx;

    CUgraphNode memsetNode = nullptr;
    CU_SAFE_CALL(cuGraphAddNode(&memsetNode, bodyGraph, nullptr, nullptr, 0, &memsetNodeParams), m_errorMsg);

    m_graph = graph;
    return InstantiateDriverGraph(graph, m_graphExec, m_errorMsg);
}

bool MemsetMemcpyLauncher::RecordMemcpyH2DConvertedGraph()
{
    CUgraph graph = nullptr;
    CUgraph bodyGraph = nullptr;
    if (!CreateConditionalGraphWithBody(graph, bodyGraph, m_errorMsg))
    {
        return false;
    }

    CUcontext ctx = nullptr;
    CU_SAFE_CALL(cuCtxGetCurrent(&ctx), m_errorMsg);

    // Explicit graph memcpy nodes require page-locked host memory; pageable memory is rejected
    // with CUDA_ERROR_INVALID_VALUE at instantiation, so use the pinned host buffer here.
    CUDA_MEMCPY3D copyParams = {};
    copyParams.srcMemoryType = CU_MEMORYTYPE_HOST;
    copyParams.dstMemoryType = CU_MEMORYTYPE_DEVICE;
    copyParams.srcHost = m_pPinnedBufferSrc;
    copyParams.dstDevice = reinterpret_cast<CUdeviceptr>(m_pDeviceBufferDst);
    copyParams.WidthInBytes = m_size;
    copyParams.Height = 1;
    copyParams.Depth = 1;

    CUgraphNodeParams memcpyNodeParams = {};
    memcpyNodeParams.type = CU_GRAPH_NODE_TYPE_MEMCPY;
    memcpyNodeParams.memcpy.copyCtx = ctx;
    memcpyNodeParams.memcpy.copyParams = copyParams;

    CUgraphNode memcpyNode = nullptr;
    CU_SAFE_CALL(cuGraphAddNode(&memcpyNode, bodyGraph, nullptr, nullptr, 0, &memcpyNodeParams), m_errorMsg);

    m_graph = graph;
    return InstantiateDriverGraph(graph, m_graphExec, m_errorMsg);
}

bool MemsetMemcpyLauncher::RecordMemcpyD2HConvertedGraph()
{
    CUgraph graph = nullptr;
    CUgraph bodyGraph = nullptr;
    if (!CreateConditionalGraphWithBody(graph, bodyGraph, m_errorMsg))
    {
        return false;
    }

    CUcontext ctx = nullptr;
    CU_SAFE_CALL(cuCtxGetCurrent(&ctx), m_errorMsg);

    // Explicit graph memcpy nodes require page-locked host memory; pageable memory is rejected
    // with CUDA_ERROR_INVALID_VALUE at instantiation, so use the pinned host buffer here.
    CUDA_MEMCPY3D copyParams = {};
    copyParams.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    copyParams.dstMemoryType = CU_MEMORYTYPE_HOST;
    copyParams.srcDevice = reinterpret_cast<CUdeviceptr>(m_pDeviceBufferSrc);
    copyParams.dstHost = m_pPinnedBufferDst;
    copyParams.WidthInBytes = m_size;
    copyParams.Height = 1;
    copyParams.Depth = 1;

    CUgraphNodeParams memcpyNodeParams = {};
    memcpyNodeParams.type = CU_GRAPH_NODE_TYPE_MEMCPY;
    memcpyNodeParams.memcpy.copyCtx = ctx;
    memcpyNodeParams.memcpy.copyParams = copyParams;

    CUgraphNode memcpyNode = nullptr;
    CU_SAFE_CALL(cuGraphAddNode(&memcpyNode, bodyGraph, nullptr, nullptr, 0, &memcpyNodeParams), m_errorMsg);

    m_graph = graph;
    return InstantiateDriverGraph(graph, m_graphExec, m_errorMsg);
}

bool MemsetMemcpyLauncher::RecordMemcpyD2DConvertedGraph()
{
    CUgraph graph = nullptr;
    CUgraph bodyGraph = nullptr;
    if (!CreateConditionalGraphWithBody(graph, bodyGraph, m_errorMsg))
    {
        return false;
    }

    CUcontext ctx = nullptr;
    CU_SAFE_CALL(cuCtxGetCurrent(&ctx), m_errorMsg);

    CUDA_MEMCPY3D copyParams = {};
    copyParams.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    copyParams.dstMemoryType = CU_MEMORYTYPE_DEVICE;
    copyParams.srcDevice = reinterpret_cast<CUdeviceptr>(m_pDeviceBufferSrc);
    copyParams.dstDevice = reinterpret_cast<CUdeviceptr>(m_pDeviceBufferDst);
    copyParams.WidthInBytes = m_size;
    copyParams.Height = 1;
    copyParams.Depth = 1;

    CUgraphNodeParams memcpyNodeParams = {};
    memcpyNodeParams.type = CU_GRAPH_NODE_TYPE_MEMCPY;
    memcpyNodeParams.memcpy.copyCtx = ctx;
    memcpyNodeParams.memcpy.copyParams = copyParams;

    CUgraphNode memcpyNode = nullptr;
    CU_SAFE_CALL(cuGraphAddNode(&memcpyNode, bodyGraph, nullptr, nullptr, 0, &memcpyNodeParams), m_errorMsg);

    m_graph = graph;
    return InstantiateDriverGraph(graph, m_graphExec, m_errorMsg);
}

bool MemsetMemcpyLauncher::LaunchGraph()
{
    CUDA_SAFE_CALL((cudaGraphLaunch(m_graphExec, m_stream)), m_errorMsg);
    CUDA_SAFE_CALL((cudaStreamSynchronize(m_stream)), m_errorMsg);
    IncrementCounterIfInBenchmark(GraphLaunchCount);
    return true;
}

int64_t MemsetMemcpyLauncher::QueryCount(int id)
{
    if (id >= CounterId::CounterIdNum)
    {
        printf("Error: Query ID %d is bigger than the CounterId nums %d.\n", id, CounterId::CounterIdNum);
        return 0;
    }

    return m_counters[id];
}
