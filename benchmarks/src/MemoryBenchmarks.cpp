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

#include "MemsetMemcpyLauncher.h"
#include "cxxopts.hpp"

#include <NvTrace/Target/NvTraceOutOfProcessService.h>
#include <NvTrace/Target/NvTraceScopeExitGuard.h>

#include <benchmark/benchmark.h>
#include <iostream>

template <typename TLauncher>
bool BenchmarkCheckSuccess(bool condition, TLauncher& launcher, benchmark::State& state)
{
    if (!condition)
    {
        state.SkipWithError(launcher.GetErrorMsg());
        return false;
    }
    return true;
}

static nv::trace::OutOfProcessService g_nvTraceService;

struct MemoryBenchmarkOptions
{
    bool runWithNvTrace = true;
};

static MemoryBenchmarkOptions g_BenchmarkOptions;

namespace nv::trace {

    class NvTraceCudaBenchmarkFeature : public benchmark::Fixture
    {
    public:
        void SetMemoryBenchmarkCounters(benchmark::State& state, MemsetMemcpyLauncher& launcher)
        {
            state.SetBytesProcessed(state.iterations() * state.range(0));
            state.SetItemsProcessed(state.iterations());
            state.counters["api_call"] = benchmark::Counter(
                (double)launcher.QueryCount(MemsetMemcpyLauncher::CounterId::CudaApiCount),
                benchmark::Counter::kIsRate);
        }

        void SetGraphMemoryBenchmarkCounters(benchmark::State& state, MemsetMemcpyLauncher& launcher)
        {
            state.SetBytesProcessed(state.iterations() * state.range(0));
            state.SetItemsProcessed(state.iterations());
            state.counters["graph_launch"] = benchmark::Counter(
                (double)launcher.QueryCount(MemsetMemcpyLauncher::CounterId::GraphLaunchCount),
                benchmark::Counter::kIsRate);
        }
    };

    // ===============================
    // Memset
    // ===============================

    BENCHMARK_DEFINE_F(NvTraceCudaBenchmarkFeature, Memset_Device)
    (benchmark::State& state)
    {
        MemsetMemcpyLauncher launcher;
        launcher.Initialize(static_cast<size_t>(state.range(0)));
        if (!BenchmarkCheckSuccess(launcher.AllocDeviceMemory(), launcher, state))
        {
            return;
        }

        {
            launcher.BenchmarkEnter();
            auto exitGuard = ScopeExitGuard([&]() { launcher.BenchmarkExit(); });
            for (auto _ : state)
            {
                if (!BenchmarkCheckSuccess(launcher.MemsetDevice(), launcher, state))
                {
                    return;
                }
            }
        }

        if (!BenchmarkCheckSuccess(launcher.FreeDeviceMemory(), launcher, state))
        {
            return;
        }

        SetMemoryBenchmarkCounters(state, launcher);
    }

    // ===============================
    // Memcpy
    // ===============================

    BENCHMARK_DEFINE_F(NvTraceCudaBenchmarkFeature, Memcpy_H2D)
    (benchmark::State& state)
    {
        MemsetMemcpyLauncher launcher;
        launcher.Initialize(static_cast<size_t>(state.range(0)));
        if (!BenchmarkCheckSuccess(launcher.AllocHostMemory(), launcher, state))
        {
            return;
        }
        if (!BenchmarkCheckSuccess(launcher.AllocDeviceMemory(), launcher, state))
        {
            return;
        }

        {
            launcher.BenchmarkEnter();
            auto exitGuard = ScopeExitGuard([&]() { launcher.BenchmarkExit(); });
            for (auto _ : state)
            {
                if (!BenchmarkCheckSuccess(launcher.MemcpyH2D(), launcher, state))
                {
                    return;
                }
            }
        }

        if (!BenchmarkCheckSuccess(launcher.FreeDeviceMemory(), launcher, state))
        {
            return;
        }
        if (!BenchmarkCheckSuccess(launcher.FreeHostMemory(), launcher, state))
        {
            return;
        }

        SetMemoryBenchmarkCounters(state, launcher);
    }

    BENCHMARK_DEFINE_F(NvTraceCudaBenchmarkFeature, Memcpy_D2H)
    (benchmark::State& state)
    {
        MemsetMemcpyLauncher launcher;
        launcher.Initialize(static_cast<size_t>(state.range(0)));
        if (!BenchmarkCheckSuccess(launcher.AllocHostMemory(), launcher, state))
        {
            return;
        }
        if (!BenchmarkCheckSuccess(launcher.AllocDeviceMemory(), launcher, state))
        {
            return;
        }

        {
            launcher.BenchmarkEnter();
            auto exitGuard = ScopeExitGuard([&]() { launcher.BenchmarkExit(); });
            for (auto _ : state)
            {
                if (!BenchmarkCheckSuccess(launcher.MemcpyD2H(), launcher, state))
                {
                    return;
                }
            }
        }

        if (!BenchmarkCheckSuccess(launcher.FreeDeviceMemory(), launcher, state))
        {
            return;
        }
        if (!BenchmarkCheckSuccess(launcher.FreeHostMemory(), launcher, state))
        {
            return;
        }

        SetMemoryBenchmarkCounters(state, launcher);
    }

    BENCHMARK_DEFINE_F(NvTraceCudaBenchmarkFeature, Memcpy_D2D)
    (benchmark::State& state)
    {
        MemsetMemcpyLauncher launcher;
        launcher.Initialize(static_cast<size_t>(state.range(0)));
        if (!BenchmarkCheckSuccess(launcher.AllocHostMemory(), launcher, state))
        {
            return;
        }
        if (!BenchmarkCheckSuccess(launcher.AllocDeviceMemory(), launcher, state))
        {
            return;
        }

        {
            launcher.BenchmarkEnter();
            auto exitGuard = ScopeExitGuard([&]() { launcher.BenchmarkExit(); });
            for (auto _ : state)
            {
                if (!BenchmarkCheckSuccess(launcher.MemcpyD2D(), launcher, state))
                {
                    return;
                }
            }
        }

        if (!BenchmarkCheckSuccess(launcher.FreeDeviceMemory(), launcher, state))
        {
            return;
        }
        if (!BenchmarkCheckSuccess(launcher.FreeHostMemory(), launcher, state))
        {
            return;
        }

        SetMemoryBenchmarkCounters(state, launcher);
    }

    BENCHMARK_DEFINE_F(NvTraceCudaBenchmarkFeature, Memcpy_H2H)
    (benchmark::State& state)
    {
        MemsetMemcpyLauncher launcher;
        launcher.Initialize(static_cast<size_t>(state.range(0)));
        if (!BenchmarkCheckSuccess(launcher.AllocHostMemory(), launcher, state))
        {
            return;
        }

        {
            launcher.BenchmarkEnter();
            auto exitGuard = ScopeExitGuard([&]() { launcher.BenchmarkExit(); });
            for (auto _ : state)
            {
                if (!BenchmarkCheckSuccess(launcher.MemcpyH2H(), launcher, state))
                {
                    return;
                }
            }
        }

        if (!BenchmarkCheckSuccess(launcher.FreeHostMemory(), launcher, state))
        {
            return;
        }

        SetMemoryBenchmarkCounters(state, launcher);
    }

    BENCHMARK_DEFINE_F(NvTraceCudaBenchmarkFeature, MemOp_Write32)
    (benchmark::State& state)
    {
        MemsetMemcpyLauncher launcher;
        launcher.Initialize(static_cast<size_t>(state.range(0)));
        if (!BenchmarkCheckSuccess(launcher.AllocDeviceMemory(), launcher, state))
        {
            return;
        }

        {
            launcher.BenchmarkEnter();
            auto exitGuard = ScopeExitGuard([&]() { launcher.BenchmarkExit(); });
            for (auto _ : state)
            {
                if (!BenchmarkCheckSuccess(launcher.MemOpWrite32(), launcher, state))
                {
                    return;
                }
            }
        }

        if (!BenchmarkCheckSuccess(launcher.FreeDeviceMemory(), launcher, state))
        {
            return;
        }

        SetMemoryBenchmarkCounters(state, launcher);
    }

    BENCHMARK_DEFINE_F(NvTraceCudaBenchmarkFeature, MemOp_Write64)
    (benchmark::State& state)
    {
        MemsetMemcpyLauncher launcher;
        launcher.Initialize(static_cast<size_t>(state.range(0)));
        if (!BenchmarkCheckSuccess(launcher.AllocDeviceMemory(), launcher, state))
        {
            return;
        }

        {
            launcher.BenchmarkEnter();
            auto exitGuard = ScopeExitGuard([&]() { launcher.BenchmarkExit(); });
            for (auto _ : state)
            {
                if (!BenchmarkCheckSuccess(launcher.MemOpWrite64(), launcher, state))
                {
                    return;
                }
            }
        }

        if (!BenchmarkCheckSuccess(launcher.FreeDeviceMemory(), launcher, state))
        {
            return;
        }

        SetMemoryBenchmarkCounters(state, launcher);
    }

    BENCHMARK_DEFINE_F(NvTraceCudaBenchmarkFeature, MemOp_WaitValue32)
    (benchmark::State& state)
    {
        MemsetMemcpyLauncher launcher;
        launcher.Initialize(static_cast<size_t>(state.range(0)));
        if (!BenchmarkCheckSuccess(launcher.AllocDeviceMemory(), launcher, state))
        {
            return;
        }

        {
            launcher.BenchmarkEnter();
            auto exitGuard = ScopeExitGuard([&]() { launcher.BenchmarkExit(); });
            for (auto _ : state)
            {
                if (!BenchmarkCheckSuccess(launcher.MemOpWait32(), launcher, state))
                {
                    return;
                }
            }
        }

        if (!BenchmarkCheckSuccess(launcher.FreeDeviceMemory(), launcher, state))
        {
            return;
        }

        SetMemoryBenchmarkCounters(state, launcher);
    }

    BENCHMARK_DEFINE_F(NvTraceCudaBenchmarkFeature, MemOp_WaitValue64)
    (benchmark::State& state)
    {
        MemsetMemcpyLauncher launcher;
        launcher.Initialize(static_cast<size_t>(state.range(0)));
        if (!BenchmarkCheckSuccess(launcher.AllocDeviceMemory(), launcher, state))
        {
            return;
        }

        {
            launcher.BenchmarkEnter();
            auto exitGuard = ScopeExitGuard([&]() { launcher.BenchmarkExit(); });
            for (auto _ : state)
            {
                if (!BenchmarkCheckSuccess(launcher.MemOpWait64(), launcher, state))
                {
                    return;
                }
            }
        }

        if (!BenchmarkCheckSuccess(launcher.FreeDeviceMemory(), launcher, state))
        {
            return;
        }

        SetMemoryBenchmarkCounters(state, launcher);
    }

    // ===============================
    // Graph Memset / Memcpy (stream capture)
    // ===============================

    BENCHMARK_DEFINE_F(NvTraceCudaBenchmarkFeature, MemsetGraph_Device)
    (benchmark::State& state)
    {
        MemsetMemcpyLauncher launcher;
        launcher.Initialize(static_cast<size_t>(state.range(0)));
        if (!BenchmarkCheckSuccess(launcher.AllocDeviceMemory(), launcher, state))
        {
            return;
        }
        if (!BenchmarkCheckSuccess(launcher.InitializeGraph(), launcher, state))
        {
            return;
        }
        if (!BenchmarkCheckSuccess(launcher.RecordMemsetDeviceGraph(), launcher, state))
        {
            return;
        }

        {
            launcher.BenchmarkEnter();
            auto exitGuard = ScopeExitGuard([&]() { launcher.BenchmarkExit(); });
            for (auto _ : state)
            {
                if (!BenchmarkCheckSuccess(launcher.LaunchGraph(), launcher, state))
                {
                    return;
                }
            }
        }

        launcher.ResetGraph();
        if (!BenchmarkCheckSuccess(launcher.FreeDeviceMemory(), launcher, state))
        {
            return;
        }

        SetGraphMemoryBenchmarkCounters(state, launcher);
    }

    BENCHMARK_DEFINE_F(NvTraceCudaBenchmarkFeature, MemcpyGraph_H2D)
    (benchmark::State& state)
    {
        MemsetMemcpyLauncher launcher;
        launcher.Initialize(static_cast<size_t>(state.range(0)));
        if (!BenchmarkCheckSuccess(launcher.AllocHostMemory(), launcher, state))
        {
            return;
        }
        if (!BenchmarkCheckSuccess(launcher.AllocDeviceMemory(), launcher, state))
        {
            return;
        }
        if (!BenchmarkCheckSuccess(launcher.InitializeGraph(), launcher, state))
        {
            return;
        }
        if (!BenchmarkCheckSuccess(launcher.RecordMemcpyH2DGraph(), launcher, state))
        {
            return;
        }

        {
            launcher.BenchmarkEnter();
            auto exitGuard = ScopeExitGuard([&]() { launcher.BenchmarkExit(); });
            for (auto _ : state)
            {
                if (!BenchmarkCheckSuccess(launcher.LaunchGraph(), launcher, state))
                {
                    return;
                }
            }
        }

        launcher.ResetGraph();
        if (!BenchmarkCheckSuccess(launcher.FreeDeviceMemory(), launcher, state))
        {
            return;
        }
        if (!BenchmarkCheckSuccess(launcher.FreeHostMemory(), launcher, state))
        {
            return;
        }

        SetGraphMemoryBenchmarkCounters(state, launcher);
    }

    BENCHMARK_DEFINE_F(NvTraceCudaBenchmarkFeature, MemcpyGraph_D2H)
    (benchmark::State& state)
    {
        MemsetMemcpyLauncher launcher;
        launcher.Initialize(static_cast<size_t>(state.range(0)));
        if (!BenchmarkCheckSuccess(launcher.AllocHostMemory(), launcher, state))
        {
            return;
        }
        if (!BenchmarkCheckSuccess(launcher.AllocDeviceMemory(), launcher, state))
        {
            return;
        }
        if (!BenchmarkCheckSuccess(launcher.InitializeGraph(), launcher, state))
        {
            return;
        }
        if (!BenchmarkCheckSuccess(launcher.RecordMemcpyD2HGraph(), launcher, state))
        {
            return;
        }

        {
            launcher.BenchmarkEnter();
            auto exitGuard = ScopeExitGuard([&]() { launcher.BenchmarkExit(); });
            for (auto _ : state)
            {
                if (!BenchmarkCheckSuccess(launcher.LaunchGraph(), launcher, state))
                {
                    return;
                }
            }
        }

        launcher.ResetGraph();
        if (!BenchmarkCheckSuccess(launcher.FreeDeviceMemory(), launcher, state))
        {
            return;
        }
        if (!BenchmarkCheckSuccess(launcher.FreeHostMemory(), launcher, state))
        {
            return;
        }

        SetGraphMemoryBenchmarkCounters(state, launcher);
    }

    BENCHMARK_DEFINE_F(NvTraceCudaBenchmarkFeature, MemcpyGraph_D2D)
    (benchmark::State& state)
    {
        MemsetMemcpyLauncher launcher;
        launcher.Initialize(static_cast<size_t>(state.range(0)));
        if (!BenchmarkCheckSuccess(launcher.AllocHostMemory(), launcher, state))
        {
            return;
        }
        if (!BenchmarkCheckSuccess(launcher.AllocDeviceMemory(), launcher, state))
        {
            return;
        }
        if (!BenchmarkCheckSuccess(launcher.InitializeGraph(), launcher, state))
        {
            return;
        }
        if (!BenchmarkCheckSuccess(launcher.RecordMemcpyD2DGraph(), launcher, state))
        {
            return;
        }

        {
            launcher.BenchmarkEnter();
            auto exitGuard = ScopeExitGuard([&]() { launcher.BenchmarkExit(); });
            for (auto _ : state)
            {
                if (!BenchmarkCheckSuccess(launcher.LaunchGraph(), launcher, state))
                {
                    return;
                }
            }
        }

        launcher.ResetGraph();
        if (!BenchmarkCheckSuccess(launcher.FreeDeviceMemory(), launcher, state))
        {
            return;
        }
        if (!BenchmarkCheckSuccess(launcher.FreeHostMemory(), launcher, state))
        {
            return;
        }

        SetGraphMemoryBenchmarkCounters(state, launcher);
    }

    BENCHMARK_DEFINE_F(NvTraceCudaBenchmarkFeature, MemcpyGraph_H2H)
    (benchmark::State& state)
    {
        MemsetMemcpyLauncher launcher;
        launcher.Initialize(static_cast<size_t>(state.range(0)));
        if (!BenchmarkCheckSuccess(launcher.AllocHostMemory(), launcher, state))
        {
            return;
        }
        if (!BenchmarkCheckSuccess(launcher.InitializeGraph(), launcher, state))
        {
            return;
        }
        if (!BenchmarkCheckSuccess(launcher.RecordMemcpyH2HGraph(), launcher, state))
        {
            return;
        }

        {
            launcher.BenchmarkEnter();
            auto exitGuard = ScopeExitGuard([&]() { launcher.BenchmarkExit(); });
            for (auto _ : state)
            {
                if (!BenchmarkCheckSuccess(launcher.LaunchGraph(), launcher, state))
                {
                    return;
                }
            }
        }

        launcher.ResetGraph();
        if (!BenchmarkCheckSuccess(launcher.FreeHostMemory(), launcher, state))
        {
            return;
        }

        SetGraphMemoryBenchmarkCounters(state, launcher);
    }

    // ===============================
    // Graph Memset / Memcpy (driver lowers memory nodes to kernels)
    // ===============================

    BENCHMARK_DEFINE_F(NvTraceCudaBenchmarkFeature, MemsetGraphConverted_Device)
    (benchmark::State& state)
    {
        MemsetMemcpyLauncher launcher;
        launcher.Initialize(static_cast<size_t>(state.range(0)));
        if (!BenchmarkCheckSuccess(launcher.AllocDeviceMemory(), launcher, state))
        {
            return;
        }
        if (!BenchmarkCheckSuccess(launcher.InitializeGraph(), launcher, state))
        {
            return;
        }
        if (!BenchmarkCheckSuccess(launcher.RecordMemsetConvertedGraph(), launcher, state))
        {
            return;
        }

        {
            launcher.BenchmarkEnter();
            auto exitGuard = ScopeExitGuard([&]() { launcher.BenchmarkExit(); });
            for (auto _ : state)
            {
                if (!BenchmarkCheckSuccess(launcher.LaunchGraph(), launcher, state))
                {
                    return;
                }
            }
        }

        launcher.ResetGraph();
        if (!BenchmarkCheckSuccess(launcher.FreeDeviceMemory(), launcher, state))
        {
            return;
        }

        SetGraphMemoryBenchmarkCounters(state, launcher);
    }

    BENCHMARK_DEFINE_F(NvTraceCudaBenchmarkFeature, MemcpyGraphConverted_H2D)
    (benchmark::State& state)
    {
        MemsetMemcpyLauncher launcher;
        launcher.Initialize(static_cast<size_t>(state.range(0)));
        if (!BenchmarkCheckSuccess(launcher.AllocHostMemory(), launcher, state))
        {
            return;
        }
        if (!BenchmarkCheckSuccess(launcher.AllocDeviceMemory(), launcher, state))
        {
            return;
        }
        if (!BenchmarkCheckSuccess(launcher.InitializeGraph(), launcher, state))
        {
            return;
        }
        if (!BenchmarkCheckSuccess(launcher.RecordMemcpyH2DConvertedGraph(), launcher, state))
        {
            return;
        }

        {
            launcher.BenchmarkEnter();
            auto exitGuard = ScopeExitGuard([&]() { launcher.BenchmarkExit(); });
            for (auto _ : state)
            {
                if (!BenchmarkCheckSuccess(launcher.LaunchGraph(), launcher, state))
                {
                    return;
                }
            }
        }

        launcher.ResetGraph();
        if (!BenchmarkCheckSuccess(launcher.FreeDeviceMemory(), launcher, state))
        {
            return;
        }
        if (!BenchmarkCheckSuccess(launcher.FreeHostMemory(), launcher, state))
        {
            return;
        }

        SetGraphMemoryBenchmarkCounters(state, launcher);
    }

    BENCHMARK_DEFINE_F(NvTraceCudaBenchmarkFeature, MemcpyGraphConverted_D2H)
    (benchmark::State& state)
    {
        MemsetMemcpyLauncher launcher;
        launcher.Initialize(static_cast<size_t>(state.range(0)));
        if (!BenchmarkCheckSuccess(launcher.AllocHostMemory(), launcher, state))
        {
            return;
        }
        if (!BenchmarkCheckSuccess(launcher.AllocDeviceMemory(), launcher, state))
        {
            return;
        }
        if (!BenchmarkCheckSuccess(launcher.InitializeGraph(), launcher, state))
        {
            return;
        }
        if (!BenchmarkCheckSuccess(launcher.RecordMemcpyD2HConvertedGraph(), launcher, state))
        {
            return;
        }

        {
            launcher.BenchmarkEnter();
            auto exitGuard = ScopeExitGuard([&]() { launcher.BenchmarkExit(); });
            for (auto _ : state)
            {
                if (!BenchmarkCheckSuccess(launcher.LaunchGraph(), launcher, state))
                {
                    return;
                }
            }
        }

        launcher.ResetGraph();
        if (!BenchmarkCheckSuccess(launcher.FreeDeviceMemory(), launcher, state))
        {
            return;
        }
        if (!BenchmarkCheckSuccess(launcher.FreeHostMemory(), launcher, state))
        {
            return;
        }

        SetGraphMemoryBenchmarkCounters(state, launcher);
    }

    BENCHMARK_DEFINE_F(NvTraceCudaBenchmarkFeature, MemcpyGraphConverted_D2D)
    (benchmark::State& state)
    {
        MemsetMemcpyLauncher launcher;
        launcher.Initialize(static_cast<size_t>(state.range(0)));
        if (!BenchmarkCheckSuccess(launcher.AllocHostMemory(), launcher, state))
        {
            return;
        }
        if (!BenchmarkCheckSuccess(launcher.AllocDeviceMemory(), launcher, state))
        {
            return;
        }
        if (!BenchmarkCheckSuccess(launcher.InitializeGraph(), launcher, state))
        {
            return;
        }
        if (!BenchmarkCheckSuccess(launcher.RecordMemcpyD2DConvertedGraph(), launcher, state))
        {
            return;
        }

        {
            launcher.BenchmarkEnter();
            auto exitGuard = ScopeExitGuard([&]() { launcher.BenchmarkExit(); });
            for (auto _ : state)
            {
                if (!BenchmarkCheckSuccess(launcher.LaunchGraph(), launcher, state))
                {
                    return;
                }
            }
        }

        launcher.ResetGraph();
        if (!BenchmarkCheckSuccess(launcher.FreeDeviceMemory(), launcher, state))
        {
            return;
        }
        if (!BenchmarkCheckSuccess(launcher.FreeHostMemory(), launcher, state))
        {
            return;
        }

        SetGraphMemoryBenchmarkCounters(state, launcher);
    }

    // Register benchmarks
    BENCHMARK_REGISTER_F(NvTraceCudaBenchmarkFeature, MemOp_Write32)->Arg(4);
    BENCHMARK_REGISTER_F(NvTraceCudaBenchmarkFeature, MemOp_Write64)->Arg(8);
    BENCHMARK_REGISTER_F(NvTraceCudaBenchmarkFeature, MemOp_WaitValue32)->Arg(4);
    BENCHMARK_REGISTER_F(NvTraceCudaBenchmarkFeature, MemOp_WaitValue64)->Arg(8);
    BENCHMARK_REGISTER_F(NvTraceCudaBenchmarkFeature, Memset_Device)->Arg(1024)->Arg(65536)->Arg(1048576)->Arg(16777216);
    BENCHMARK_REGISTER_F(NvTraceCudaBenchmarkFeature, Memcpy_H2D)->Arg(1024)->Arg(65536)->Arg(1048576)->Arg(16777216);
    BENCHMARK_REGISTER_F(NvTraceCudaBenchmarkFeature, Memcpy_D2H)->Arg(1024)->Arg(65536)->Arg(1048576)->Arg(16777216);
    BENCHMARK_REGISTER_F(NvTraceCudaBenchmarkFeature, Memcpy_D2D)->Arg(1024)->Arg(65536)->Arg(1048576)->Arg(16777216);
    BENCHMARK_REGISTER_F(NvTraceCudaBenchmarkFeature, Memcpy_H2H)->Arg(1024)->Arg(65536)->Arg(1048576)->Arg(16777216);
    BENCHMARK_REGISTER_F(NvTraceCudaBenchmarkFeature, MemsetGraph_Device)->Arg(1024)->Arg(65536)->Arg(1048576)->Arg(16777216);
    BENCHMARK_REGISTER_F(NvTraceCudaBenchmarkFeature, MemcpyGraph_H2D)->Arg(1024)->Arg(65536)->Arg(1048576)->Arg(16777216);
    BENCHMARK_REGISTER_F(NvTraceCudaBenchmarkFeature, MemcpyGraph_D2H)->Arg(1024)->Arg(65536)->Arg(1048576)->Arg(16777216);
    BENCHMARK_REGISTER_F(NvTraceCudaBenchmarkFeature, MemcpyGraph_D2D)->Arg(1024)->Arg(65536)->Arg(1048576)->Arg(16777216);
    BENCHMARK_REGISTER_F(NvTraceCudaBenchmarkFeature, MemcpyGraph_H2H)->Arg(1024)->Arg(65536)->Arg(1048576)->Arg(16777216);
    BENCHMARK_REGISTER_F(NvTraceCudaBenchmarkFeature, MemsetGraphConverted_Device)->Arg(1024)->Arg(65536)->Arg(1048576)->Arg(16777216);
    BENCHMARK_REGISTER_F(NvTraceCudaBenchmarkFeature, MemcpyGraphConverted_H2D)->Arg(1024)->Arg(65536)->Arg(1048576)->Arg(16777216);
    BENCHMARK_REGISTER_F(NvTraceCudaBenchmarkFeature, MemcpyGraphConverted_D2H)->Arg(1024)->Arg(65536)->Arg(1048576)->Arg(16777216);
    BENCHMARK_REGISTER_F(NvTraceCudaBenchmarkFeature, MemcpyGraphConverted_D2D)->Arg(1024)->Arg(65536)->Arg(1048576)->Arg(16777216);

} // namespace nv::trace

int main(int argc, char** argv)
{
    std::vector<char*> updatedArgv{argv[0]};

    cxxopts::Options options("CUDA Memory Benchmark Options");
    options.add_options()("nvtrace", "Run benchmarks with NvTrace (default true)", cxxopts::value<bool>()->default_value("true"));
    options.add_options()("h,help", "Display this help message");
    options.allow_unrecognised_options();

    auto result = options.parse(argc, argv);
    if (result.count("help"))
    {
        std::cout << options.help() << std::endl;
        updatedArgv.push_back(const_cast<char*>("--help"));
    }

    g_BenchmarkOptions.runWithNvTrace = result["nvtrace"].as<bool>();

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
        std::cout << "Starting nvidia-nsight-perf service and running benchmarks with NvTrace" << std::endl;
        nv::trace::OutOfProcessService::Options serviceOptions = {};
        if (!g_nvTraceService.Initialize(serviceOptions))
        {
            std::cerr << "Failed to initialize out-of-process nvidia-nsight-perf service" << std::endl;
            return 1;
        }
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
    }

    ::benchmark::RunSpecifiedBenchmarks();
    ::benchmark::Shutdown();

    if (g_BenchmarkOptions.runWithNvTrace)
    {
        g_nvTraceService.Reset();
    }

    return 0;
}
