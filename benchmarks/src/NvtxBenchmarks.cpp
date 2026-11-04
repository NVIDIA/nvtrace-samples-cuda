/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * To have a trustable benchmark result, please fix the CPU frequency.
 * Otherwise, starting NvTrace tends to maximize the CPU frequency, and the comparison between NvTrace and no NvTrace becomes unreliable.
 *
 * cpupower frequency-info -o proc
 * sudo cpupower frequency-set -d 3600000 -u 3600000
 * sudo cpupower frequency-set --governor performance
 */

#include "cxxopts.hpp"

#include <nvtx3/nvToolsExt.h>

#include <NvTrace/Target/NvTraceOutOfProcessService.h>

#include <benchmark/benchmark.h>
#include <cstdio>
#include <cwchar>
#include <iostream>
#include <vector>

static nv::trace::OutOfProcessService g_nvTraceService;

struct NvtxBenchmarkOptions
{
    bool runWithNvTrace = true;
    size_t stringLength = 64;
};

static NvtxBenchmarkOptions g_BenchmarkOptions;
static nvtxDomainHandle_t g_benchmarkDomain = nullptr;
static constexpr int64_t kNestedDepth = 100;
static constexpr int kNumThreads = 4;

namespace nv::trace {

    class NvTraceNvtxBenchmarkFixture : public benchmark::Fixture
    {
    public:
        void SetUp(benchmark::State& state) override
        {
        }

        void TearDown(benchmark::State& state) override
        {
        }
    };

    // ===============================
    // RangePush/Pop variants
    // ===============================

    BENCHMARK_DEFINE_F(NvTraceNvtxBenchmarkFixture, RangePushPopA)
    (benchmark::State& state)
    {
        const std::string rangeStr(g_BenchmarkOptions.stringLength, 'x');

        for (auto _ : state)
        {
            nvtxRangePushA(rangeStr.c_str());
            nvtxRangePop();
        }

        state.SetItemsProcessed(state.iterations());
    }

    BENCHMARK_DEFINE_F(NvTraceNvtxBenchmarkFixture, RangePushPopW)
    (benchmark::State& state)
    {
        const std::wstring rangeStr(g_BenchmarkOptions.stringLength, L'x');

        for (auto _ : state)
        {
            nvtxRangePushW(rangeStr.c_str());
            nvtxRangePop();
        }

        state.SetItemsProcessed(state.iterations());
    }

    BENCHMARK_DEFINE_F(NvTraceNvtxBenchmarkFixture, RangePushPopExA)
    (benchmark::State& state)
    {
        const std::string rangeStr(g_BenchmarkOptions.stringLength, 'x');

        for (auto _ : state)
        {
            nvtxEventAttributes_t attr = {};
            attr.version = NVTX_VERSION;
            attr.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
            attr.messageType = NVTX_MESSAGE_TYPE_ASCII;
            attr.message.ascii = rangeStr.c_str();
            nvtxRangePushEx(&attr);
            nvtxRangePop();
        }

        state.SetItemsProcessed(state.iterations());
    }

    BENCHMARK_DEFINE_F(NvTraceNvtxBenchmarkFixture, RangePushPopExW)
    (benchmark::State& state)
    {
        const std::wstring rangeStr(g_BenchmarkOptions.stringLength, L'x');

        for (auto _ : state)
        {
            nvtxEventAttributes_t attr = {};
            attr.version = NVTX_VERSION;
            attr.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
            attr.messageType = NVTX_MESSAGE_TYPE_UNICODE;
            attr.message.unicode = rangeStr.c_str();
            nvtxRangePushEx(&attr);
            nvtxRangePop();
        }

        state.SetItemsProcessed(state.iterations());
    }

    BENCHMARK_DEFINE_F(NvTraceNvtxBenchmarkFixture, RangePushPopExRegistered)
    (benchmark::State& state)
    {
        const std::string rangeStr(g_BenchmarkOptions.stringLength, 'x');
        nvtxStringHandle_t handle = nvtxDomainRegisterStringA(nullptr, rangeStr.c_str());

        for (auto _ : state)
        {
            nvtxEventAttributes_t attr = {};
            attr.version = NVTX_VERSION;
            attr.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
            attr.messageType = NVTX_MESSAGE_TYPE_REGISTERED;
            attr.message.registered = handle;
            nvtxRangePushEx(&attr);
            nvtxRangePop();
        }

        state.SetItemsProcessed(state.iterations());
    }

    BENCHMARK_DEFINE_F(NvTraceNvtxBenchmarkFixture, DomainRangePushPopExA)
    (benchmark::State& state)
    {
        const std::string rangeStr(g_BenchmarkOptions.stringLength, 'x');

        for (auto _ : state)
        {
            nvtxEventAttributes_t attr = {};
            attr.version = NVTX_VERSION;
            attr.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
            attr.messageType = NVTX_MESSAGE_TYPE_ASCII;
            attr.message.ascii = rangeStr.c_str();
            nvtxDomainRangePushEx(g_benchmarkDomain, &attr);
            nvtxDomainRangePop(g_benchmarkDomain);
        }

        state.SetItemsProcessed(state.iterations());
    }

    BENCHMARK_DEFINE_F(NvTraceNvtxBenchmarkFixture, DomainRangePushPopExW)
    (benchmark::State& state)
    {
        const std::wstring rangeStr(g_BenchmarkOptions.stringLength, L'x');

        for (auto _ : state)
        {
            nvtxEventAttributes_t attr = {};
            attr.version = NVTX_VERSION;
            attr.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
            attr.messageType = NVTX_MESSAGE_TYPE_UNICODE;
            attr.message.unicode = rangeStr.c_str();
            nvtxDomainRangePushEx(g_benchmarkDomain, &attr);
            nvtxDomainRangePop(g_benchmarkDomain);
        }

        state.SetItemsProcessed(state.iterations());
    }

    BENCHMARK_DEFINE_F(NvTraceNvtxBenchmarkFixture, DomainRangePushPopExRegistered)
    (benchmark::State& state)
    {
        const std::string rangeStr(g_BenchmarkOptions.stringLength, 'x');
        nvtxStringHandle_t handle = nvtxDomainRegisterStringA(g_benchmarkDomain, rangeStr.c_str());

        for (auto _ : state)
        {
            nvtxEventAttributes_t attr = {};
            attr.version = NVTX_VERSION;
            attr.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
            attr.messageType = NVTX_MESSAGE_TYPE_REGISTERED;
            attr.message.registered = handle;
            nvtxDomainRangePushEx(g_benchmarkDomain, &attr);
            nvtxDomainRangePop(g_benchmarkDomain);
        }

        state.SetItemsProcessed(state.iterations());
    }

    // ===============================
    // Nested RangePush/Pop variants
    // ===============================

    BENCHMARK_DEFINE_F(NvTraceNvtxBenchmarkFixture, NestedRangePushPopA)
    (benchmark::State& state)
    {
        const std::string rangeStr(g_BenchmarkOptions.stringLength, 'x');
        const int maxDepth = static_cast<int>(state.range(0));

        for (auto _ : state)
        {
            for (int depth = 0; depth < maxDepth; depth++)
            {
                nvtxRangePushA(rangeStr.c_str());
            }
            for (int depth = 0; depth < maxDepth; depth++)
            {
                nvtxRangePop();
            }
        }

        state.SetItemsProcessed(state.iterations() * maxDepth);
    }

    BENCHMARK_DEFINE_F(NvTraceNvtxBenchmarkFixture, NestedRangePushPopW)
    (benchmark::State& state)
    {
        const std::wstring rangeStr(g_BenchmarkOptions.stringLength, L'x');
        const int maxDepth = static_cast<int>(state.range(0));

        for (auto _ : state)
        {
            for (int depth = 0; depth < maxDepth; depth++)
            {
                nvtxRangePushW(rangeStr.c_str());
            }
            for (int depth = 0; depth < maxDepth; depth++)
            {
                nvtxRangePop();
            }
        }

        state.SetItemsProcessed(state.iterations() * maxDepth);
    }

    BENCHMARK_DEFINE_F(NvTraceNvtxBenchmarkFixture, NestedRangePushPopExA)
    (benchmark::State& state)
    {
        const std::string rangeStr(g_BenchmarkOptions.stringLength, 'x');
        const int maxDepth = static_cast<int>(state.range(0));

        for (auto _ : state)
        {
            for (int depth = 0; depth < maxDepth; depth++)
            {
                nvtxEventAttributes_t attr = {};
                attr.version = NVTX_VERSION;
                attr.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
                attr.messageType = NVTX_MESSAGE_TYPE_ASCII;
                attr.message.ascii = rangeStr.c_str();
                nvtxRangePushEx(&attr);
            }
            for (int depth = 0; depth < maxDepth; depth++)
            {
                nvtxRangePop();
            }
        }

        state.SetItemsProcessed(state.iterations() * maxDepth);
    }

    BENCHMARK_DEFINE_F(NvTraceNvtxBenchmarkFixture, NestedRangePushPopExW)
    (benchmark::State& state)
    {
        const std::wstring rangeStr(g_BenchmarkOptions.stringLength, L'x');
        const int maxDepth = static_cast<int>(state.range(0));

        for (auto _ : state)
        {
            for (int depth = 0; depth < maxDepth; depth++)
            {
                nvtxEventAttributes_t attr = {};
                attr.version = NVTX_VERSION;
                attr.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
                attr.messageType = NVTX_MESSAGE_TYPE_UNICODE;
                attr.message.unicode = rangeStr.c_str();
                nvtxRangePushEx(&attr);
            }
            for (int depth = 0; depth < maxDepth; depth++)
            {
                nvtxRangePop();
            }
        }

        state.SetItemsProcessed(state.iterations() * maxDepth);
    }

    BENCHMARK_DEFINE_F(NvTraceNvtxBenchmarkFixture, NestedRangePushPopExRegistered)
    (benchmark::State& state)
    {
        const std::string rangeStr(g_BenchmarkOptions.stringLength, 'x');
        nvtxStringHandle_t handle = nvtxDomainRegisterStringA(nullptr, rangeStr.c_str());
        const int maxDepth = static_cast<int>(state.range(0));

        for (auto _ : state)
        {
            for (int depth = 0; depth < maxDepth; depth++)
            {
                nvtxEventAttributes_t attr = {};
                attr.version = NVTX_VERSION;
                attr.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
                attr.messageType = NVTX_MESSAGE_TYPE_REGISTERED;
                attr.message.registered = handle;
                nvtxRangePushEx(&attr);
            }
            for (int depth = 0; depth < maxDepth; depth++)
            {
                nvtxRangePop();
            }
        }

        state.SetItemsProcessed(state.iterations() * maxDepth);
    }

    BENCHMARK_DEFINE_F(NvTraceNvtxBenchmarkFixture, NestedDomainRangePushPopExA)
    (benchmark::State& state)
    {
        const std::string rangeStr(g_BenchmarkOptions.stringLength, 'x');
        const int maxDepth = static_cast<int>(state.range(0));

        for (auto _ : state)
        {
            for (int depth = 0; depth < maxDepth; depth++)
            {
                nvtxEventAttributes_t attr = {};
                attr.version = NVTX_VERSION;
                attr.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
                attr.messageType = NVTX_MESSAGE_TYPE_ASCII;
                attr.message.ascii = rangeStr.c_str();
                nvtxDomainRangePushEx(g_benchmarkDomain, &attr);
            }
            for (int depth = 0; depth < maxDepth; depth++)
            {
                nvtxDomainRangePop(g_benchmarkDomain);
            }
        }

        state.SetItemsProcessed(state.iterations() * maxDepth);
    }

    BENCHMARK_DEFINE_F(NvTraceNvtxBenchmarkFixture, NestedDomainRangePushPopExW)
    (benchmark::State& state)
    {
        const std::wstring rangeStr(g_BenchmarkOptions.stringLength, L'x');
        const int maxDepth = static_cast<int>(state.range(0));

        for (auto _ : state)
        {
            for (int depth = 0; depth < maxDepth; depth++)
            {
                nvtxEventAttributes_t attr = {};
                attr.version = NVTX_VERSION;
                attr.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
                attr.messageType = NVTX_MESSAGE_TYPE_UNICODE;
                attr.message.unicode = rangeStr.c_str();
                nvtxDomainRangePushEx(g_benchmarkDomain, &attr);
            }
            for (int depth = 0; depth < maxDepth; depth++)
            {
                nvtxDomainRangePop(g_benchmarkDomain);
            }
        }

        state.SetItemsProcessed(state.iterations() * maxDepth);
    }

    BENCHMARK_DEFINE_F(NvTraceNvtxBenchmarkFixture, NestedDomainRangePushPopExRegistered)
    (benchmark::State& state)
    {
        const std::string rangeStr(g_BenchmarkOptions.stringLength, 'x');
        const int maxDepth = static_cast<int>(state.range(0));
        nvtxStringHandle_t handle = nvtxDomainRegisterStringA(g_benchmarkDomain, rangeStr.c_str());

        for (auto _ : state)
        {
            for (int depth = 0; depth < maxDepth; depth++)
            {
                nvtxEventAttributes_t attr = {};
                attr.version = NVTX_VERSION;
                attr.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
                attr.messageType = NVTX_MESSAGE_TYPE_REGISTERED;
                attr.message.registered = handle;
                nvtxDomainRangePushEx(g_benchmarkDomain, &attr);
            }
            for (int depth = 0; depth < maxDepth; depth++)
            {
                nvtxDomainRangePop(g_benchmarkDomain);
            }
        }

        state.SetItemsProcessed(state.iterations() * maxDepth);
    }

    // ===============================
    // nvtxMark variants
    // ===============================

    BENCHMARK_DEFINE_F(NvTraceNvtxBenchmarkFixture, MarkA)
    (benchmark::State& state)
    {
        const std::string msg(g_BenchmarkOptions.stringLength, 'x');

        for (auto _ : state)
        {
            nvtxMarkA(msg.c_str());
        }

        state.SetItemsProcessed(state.iterations());
    }

    BENCHMARK_DEFINE_F(NvTraceNvtxBenchmarkFixture, MarkW)
    (benchmark::State& state)
    {
        const std::wstring msg(g_BenchmarkOptions.stringLength, L'x');

        for (auto _ : state)
        {
            nvtxMarkW(msg.c_str());
        }

        state.SetItemsProcessed(state.iterations());
    }

    BENCHMARK_DEFINE_F(NvTraceNvtxBenchmarkFixture, MarkExA)
    (benchmark::State& state)
    {
        const std::string msg(g_BenchmarkOptions.stringLength, 'x');

        for (auto _ : state)
        {
            nvtxEventAttributes_t attr = {};
            attr.version = NVTX_VERSION;
            attr.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
            attr.messageType = NVTX_MESSAGE_TYPE_ASCII;
            attr.message.ascii = msg.c_str();
            nvtxMarkEx(&attr);
        }

        state.SetItemsProcessed(state.iterations());
    }

    BENCHMARK_DEFINE_F(NvTraceNvtxBenchmarkFixture, MarkExW)
    (benchmark::State& state)
    {
        const std::wstring msg(g_BenchmarkOptions.stringLength, L'x');

        for (auto _ : state)
        {
            nvtxEventAttributes_t attr = {};
            attr.version = NVTX_VERSION;
            attr.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
            attr.messageType = NVTX_MESSAGE_TYPE_UNICODE;
            attr.message.unicode = msg.c_str();
            nvtxMarkEx(&attr);
        }

        state.SetItemsProcessed(state.iterations());
    }

    BENCHMARK_DEFINE_F(NvTraceNvtxBenchmarkFixture, MarkExRegistered)
    (benchmark::State& state)
    {
        const std::string msg(g_BenchmarkOptions.stringLength, 'x');
        nvtxStringHandle_t handle = nvtxDomainRegisterStringA(nullptr, msg.c_str());

        for (auto _ : state)
        {
            nvtxEventAttributes_t attr = {};
            attr.version = NVTX_VERSION;
            attr.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
            attr.messageType = NVTX_MESSAGE_TYPE_REGISTERED;
            attr.message.registered = handle;
            nvtxMarkEx(&attr);
        }

        state.SetItemsProcessed(state.iterations());
    }

    BENCHMARK_DEFINE_F(NvTraceNvtxBenchmarkFixture, DomainMarkExA)
    (benchmark::State& state)
    {
        const std::string msg(g_BenchmarkOptions.stringLength, 'x');

        for (auto _ : state)
        {
            nvtxEventAttributes_t attr = {};
            attr.version = NVTX_VERSION;
            attr.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
            attr.messageType = NVTX_MESSAGE_TYPE_ASCII;
            attr.message.ascii = msg.c_str();
            nvtxDomainMarkEx(g_benchmarkDomain, &attr);
        }

        state.SetItemsProcessed(state.iterations());
    }

    BENCHMARK_DEFINE_F(NvTraceNvtxBenchmarkFixture, DomainMarkExW)
    (benchmark::State& state)
    {
        const std::wstring msg(g_BenchmarkOptions.stringLength, L'x');

        for (auto _ : state)
        {
            nvtxEventAttributes_t attr = {};
            attr.version = NVTX_VERSION;
            attr.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
            attr.messageType = NVTX_MESSAGE_TYPE_UNICODE;
            attr.message.unicode = msg.c_str();
            nvtxDomainMarkEx(g_benchmarkDomain, &attr);
        }

        state.SetItemsProcessed(state.iterations());
    }

    BENCHMARK_DEFINE_F(NvTraceNvtxBenchmarkFixture, DomainMarkExRegistered)
    (benchmark::State& state)
    {
        const std::string msg(g_BenchmarkOptions.stringLength, 'x');
        nvtxStringHandle_t handle = nvtxDomainRegisterStringA(g_benchmarkDomain, msg.c_str());

        for (auto _ : state)
        {
            nvtxEventAttributes_t attr = {};
            attr.version = NVTX_VERSION;
            attr.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
            attr.messageType = NVTX_MESSAGE_TYPE_REGISTERED;
            attr.message.registered = handle;
            nvtxDomainMarkEx(g_benchmarkDomain, &attr);
        }

        state.SetItemsProcessed(state.iterations());
    }

    // ===============================
    // Registered string variants
    // ===============================

    BENCHMARK_DEFINE_F(NvTraceNvtxBenchmarkFixture, RegisterStringA)
    (benchmark::State& state)
    {
        const std::string str(g_BenchmarkOptions.stringLength, 'x');

        for (auto _ : state)
        {
            nvtxStringHandle_t handle = nvtxDomainRegisterStringA(nullptr, str.c_str());
            benchmark::DoNotOptimize(handle);
        }

        state.SetItemsProcessed(state.iterations());
    }

    BENCHMARK_DEFINE_F(NvTraceNvtxBenchmarkFixture, RegisterStringW)
    (benchmark::State& state)
    {
        const std::wstring str(g_BenchmarkOptions.stringLength, L'x');

        for (auto _ : state)
        {
            nvtxStringHandle_t handle = nvtxDomainRegisterStringW(nullptr, str.c_str());
            benchmark::DoNotOptimize(handle);
        }

        state.SetItemsProcessed(state.iterations());
    }

    // ===============================
    // Multi-domain PushPop
    // ===============================

    BENCHMARK_DEFINE_F(NvTraceNvtxBenchmarkFixture, FourDomainRangePushPop)
    (benchmark::State& state)
    {
        const std::string rangeStr(g_BenchmarkOptions.stringLength, 'x');
        nvtxDomainHandle_t domainA = nvtxDomainCreateA("benchmark_domain_a");
        nvtxDomainHandle_t domainB = nvtxDomainCreateA("benchmark_domain_b");
        nvtxDomainHandle_t domainC = nvtxDomainCreateA("benchmark_domain_c");
        nvtxDomainHandle_t domainD = nvtxDomainCreateA("benchmark_domain_d");

        nvtxEventAttributes_t attr = {};
        attr.version = NVTX_VERSION;
        attr.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
        attr.messageType = NVTX_MESSAGE_TYPE_ASCII;
        attr.message.ascii = rangeStr.c_str();

        for (auto _ : state)
        {
            nvtxDomainRangePushEx(domainA, &attr);
            nvtxDomainRangePushEx(domainB, &attr);
            nvtxDomainRangePushEx(domainC, &attr);
            nvtxDomainRangePushEx(domainD, &attr);
            nvtxDomainRangePop(domainA);
            nvtxDomainRangePop(domainB);
            nvtxDomainRangePop(domainC);
            nvtxDomainRangePop(domainD);
        }

        state.SetItemsProcessed(state.iterations() * 4);
    }

    // ===============================
    // sprintf + Mark (unique strings)
    // ===============================

    BENCHMARK_DEFINE_F(NvTraceNvtxBenchmarkFixture, SprintfMarkA)
    (benchmark::State& state)
    {
        const std::string prefix(g_BenchmarkOptions.stringLength, 'x');
        const size_t bufSize = g_BenchmarkOptions.stringLength + 21;
        std::vector<char> buf(bufSize);
        unsigned long long counter = 0;

        for (auto _ : state)
        {
            snprintf(buf.data(), bufSize, "%s%llu", prefix.c_str(), counter++);
            nvtxMarkA(buf.data());
        }

        state.SetItemsProcessed(state.iterations());
    }

    BENCHMARK_DEFINE_F(NvTraceNvtxBenchmarkFixture, SprintfMarkW)
    (benchmark::State& state)
    {
        const std::wstring prefix(g_BenchmarkOptions.stringLength, L'x');
        const size_t bufSize = g_BenchmarkOptions.stringLength + 21;
        std::vector<wchar_t> buf(bufSize);
        unsigned long long counter = 0;

        for (auto _ : state)
        {
            swprintf(buf.data(), bufSize, L"%ls%llu", prefix.c_str(), counter++);
            nvtxMarkW(buf.data());
        }

        state.SetItemsProcessed(state.iterations());
    }

    // ===============================
    // Benchmark registration
    // ===============================

    BENCHMARK_REGISTER_F(NvTraceNvtxBenchmarkFixture, RangePushPopA);
    BENCHMARK_REGISTER_F(NvTraceNvtxBenchmarkFixture, RangePushPopW);
    BENCHMARK_REGISTER_F(NvTraceNvtxBenchmarkFixture, RangePushPopExA);
    BENCHMARK_REGISTER_F(NvTraceNvtxBenchmarkFixture, RangePushPopExW);
    BENCHMARK_REGISTER_F(NvTraceNvtxBenchmarkFixture, RangePushPopExRegistered);
    BENCHMARK_REGISTER_F(NvTraceNvtxBenchmarkFixture, DomainRangePushPopExA);
    BENCHMARK_REGISTER_F(NvTraceNvtxBenchmarkFixture, DomainRangePushPopExW);
    BENCHMARK_REGISTER_F(NvTraceNvtxBenchmarkFixture, DomainRangePushPopExRegistered);

    BENCHMARK_REGISTER_F(NvTraceNvtxBenchmarkFixture, NestedRangePushPopA)
        ->ArgName("depth")
        ->Arg(kNestedDepth);
    BENCHMARK_REGISTER_F(NvTraceNvtxBenchmarkFixture, NestedRangePushPopW)
        ->ArgName("depth")
        ->Arg(kNestedDepth);
    BENCHMARK_REGISTER_F(NvTraceNvtxBenchmarkFixture, NestedRangePushPopExA)
        ->ArgName("depth")
        ->Arg(kNestedDepth);
    BENCHMARK_REGISTER_F(NvTraceNvtxBenchmarkFixture, NestedRangePushPopExW)
        ->ArgName("depth")
        ->Arg(kNestedDepth);
    BENCHMARK_REGISTER_F(NvTraceNvtxBenchmarkFixture, NestedRangePushPopExRegistered)
        ->ArgName("depth")
        ->Arg(kNestedDepth);
    BENCHMARK_REGISTER_F(NvTraceNvtxBenchmarkFixture, NestedDomainRangePushPopExA)
        ->ArgName("depth")
        ->Arg(kNestedDepth);
    BENCHMARK_REGISTER_F(NvTraceNvtxBenchmarkFixture, NestedDomainRangePushPopExW)
        ->ArgName("depth")
        ->Arg(kNestedDepth);
    BENCHMARK_REGISTER_F(NvTraceNvtxBenchmarkFixture, NestedDomainRangePushPopExRegistered)
        ->ArgName("depth")
        ->Arg(kNestedDepth);

    BENCHMARK_REGISTER_F(NvTraceNvtxBenchmarkFixture, MarkA);
    BENCHMARK_REGISTER_F(NvTraceNvtxBenchmarkFixture, MarkW);
    BENCHMARK_REGISTER_F(NvTraceNvtxBenchmarkFixture, MarkExA);
    BENCHMARK_REGISTER_F(NvTraceNvtxBenchmarkFixture, MarkExW);
    BENCHMARK_REGISTER_F(NvTraceNvtxBenchmarkFixture, MarkExRegistered);
    BENCHMARK_REGISTER_F(NvTraceNvtxBenchmarkFixture, DomainMarkExA);
    BENCHMARK_REGISTER_F(NvTraceNvtxBenchmarkFixture, DomainMarkExW);
    BENCHMARK_REGISTER_F(NvTraceNvtxBenchmarkFixture, DomainMarkExRegistered);

    BENCHMARK_REGISTER_F(NvTraceNvtxBenchmarkFixture, RegisterStringA);
    BENCHMARK_REGISTER_F(NvTraceNvtxBenchmarkFixture, RegisterStringW);

    BENCHMARK_REGISTER_F(NvTraceNvtxBenchmarkFixture, RangePushPopA)
        ->Threads(kNumThreads);
    BENCHMARK_REGISTER_F(NvTraceNvtxBenchmarkFixture, NestedRangePushPopA)
        ->ArgName("depth")
        ->Arg(kNestedDepth)
        ->Threads(kNumThreads);

    BENCHMARK_REGISTER_F(NvTraceNvtxBenchmarkFixture, FourDomainRangePushPop);

    BENCHMARK_REGISTER_F(NvTraceNvtxBenchmarkFixture, SprintfMarkA);
    BENCHMARK_REGISTER_F(NvTraceNvtxBenchmarkFixture, SprintfMarkW);

} // namespace nv::trace

int main(int argc, char** argv)
{
    std::vector<char*> updatedArgv{argv[0]};

    cxxopts::Options options("NVTX Benchmark Options");
    options.add_options()("nvtrace", "Run benchmarks with NvTrace (default true)", cxxopts::value<bool>()->default_value("true"));
    options.add_options()("strlen", "Length of NVTX strings used in benchmarks (default 64)", cxxopts::value<int64_t>()->default_value("64"));
    options.add_options()("h,help", "Display this help message");
    options.allow_unrecognised_options();

    auto result = options.parse(argc, argv);
    if (result.count("help"))
    {
        std::cout << options.help() << std::endl;
        updatedArgv.push_back(const_cast<char*>("--help"));
    }

    g_BenchmarkOptions.runWithNvTrace = result["nvtrace"].as<bool>();
    const int64_t stringLength = result["strlen"].as<int64_t>();
    if (stringLength <= 0)
    {
        std::cerr << "Error: --strlen must be a positive integer" << std::endl;
        return 1;
    }
    g_BenchmarkOptions.stringLength = static_cast<size_t>(stringLength);

    for (const auto& unmatched : result.unmatched())
    {
        updatedArgv.push_back(const_cast<char*>(unmatched.c_str()));
    }
    int updatedArgc = static_cast<int>(updatedArgv.size());

    ::benchmark::Initialize(&updatedArgc, updatedArgv.data());
    if (::benchmark::ReportUnrecognizedArguments(updatedArgc, updatedArgv.data()))
    {
        return 1;
    }

    if (g_BenchmarkOptions.runWithNvTrace)
    {
        std::cout << "Starting out-of-process nvidia-nsight-perf service for NVTX benchmarks" << std::endl;
        if (!g_nvTraceService.Initialize())
        {
            std::cerr << "Failed to initialize out-of-process nvidia-nsight-perf service" << std::endl;
            return 1;
        }
    }

    g_benchmarkDomain = nvtxDomainCreateA("benchmark_domain");

    ::benchmark::RunSpecifiedBenchmarks();
    ::benchmark::Shutdown();

    if (g_BenchmarkOptions.runWithNvTrace)
    {
        g_nvTraceService.Reset();
    }

    return 0;
}
