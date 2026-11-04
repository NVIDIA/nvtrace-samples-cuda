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
 *
 * Please use the provided script RunCudaBenchmarks.sh
 */

#include "VectorAddLauncher.h"
#include "cxxopts.hpp"

#include <NvTrace/Target/NvTraceOutOfProcessService.h>
#include <NvTrace/Target/NvTraceScopeExitGuard.h>

#include <benchmark/benchmark.h>
#include <iostream>


#define BENCHMARK_CHECK_SUCCESS(condition_, launcher_, state_) \
    do                                                         \
    {                                                          \
        if (!(condition_))                                     \
        {                                                      \
            state_.SkipWithError(launcher_.GetErrorMsg());     \
            return;                                            \
        }                                                      \
    } while (0)

static nv::trace::OutOfProcessService g_nvTraceService;

struct EndToEndBenchmarkOptions
{
    bool runWithNvTrace = true;
};

static EndToEndBenchmarkOptions g_BenchmarkOptions;

static void BusyWait(uint64_t timeNsec)
{
    auto time_start = std::chrono::high_resolution_clock::now();
    while (std::chrono::high_resolution_clock::now() - time_start < std::chrono::nanoseconds(timeNsec))
    {
        // busy wait
    }
}

namespace nv::trace {

    class NvTraceCudaBenchmarkFeature : public benchmark::Fixture
    {
    public:
        void SetVectorAddBenchmarkCounters(benchmark::State& state, VectorAddLauncher& launcher)
        {
            state.SetItemsProcessed(state.iterations() * state.range(0));
            state.SetBytesProcessed(state.iterations() * state.range(0) * sizeof(float));
            state.counters["api_call"] = benchmark::Counter((double)launcher.QueryCount(VectorAddLauncher::CounterId::CudaApiCount), benchmark::Counter::kIsRate);
            state.counters["cu_launch"] = benchmark::Counter((double)launcher.QueryCount(VectorAddLauncher::CounterId::KernelLaunchCount), benchmark::Counter::kIsRate);
        }
    };

    // ===============================
    // VectorAdd
    // ===============================

    BENCHMARK_DEFINE_F(NvTraceCudaBenchmarkFeature, EndToEnd)
    (benchmark::State& state)
    {
        VectorAddLauncher launcher;
        launcher.Initialize(static_cast<int>(state.range(0)));
        BENCHMARK_CHECK_SUCCESS(launcher.AllocHostMemory(), launcher, state);
        BENCHMARK_CHECK_SUCCESS(launcher.AllocAndCopyDeviceMemory(), launcher, state);
        BENCHMARK_CHECK_SUCCESS(launcher.LaunchKernel(), launcher, state);
        BENCHMARK_CHECK_SUCCESS(launcher.VerifyComputeResult(), launcher, state);
        BENCHMARK_CHECK_SUCCESS(launcher.FreeDeviceMemory(), launcher, state);

        {
            launcher.BenchmarkEnter();
            auto exitGuard = ScopeExitGuard([&]() { launcher.BenchmarkExit(); });
            for (auto _ : state)
            {
                BENCHMARK_CHECK_SUCCESS(launcher.AllocAndCopyDeviceMemory(), launcher, state);
                BENCHMARK_CHECK_SUCCESS(launcher.LaunchKernel(), launcher, state);
                BENCHMARK_CHECK_SUCCESS(launcher.FreeDeviceMemory(), launcher, state);
            }
        }

        BENCHMARK_CHECK_SUCCESS(launcher.FreeDeviceMemory(), launcher, state);
        BENCHMARK_CHECK_SUCCESS(launcher.FreeHostMemory(), launcher, state);

        SetVectorAddBenchmarkCounters(state, launcher);
    }

    BENCHMARK_DEFINE_F(NvTraceCudaBenchmarkFeature, KernelAdd)
    (benchmark::State& state)
    {
        VectorAddLauncher launcher;
        launcher.Initialize(static_cast<int>(state.range(0)));
        BENCHMARK_CHECK_SUCCESS(launcher.AllocHostMemory(), launcher, state);
        BENCHMARK_CHECK_SUCCESS(launcher.AllocAndCopyDeviceMemory(), launcher, state);
        BENCHMARK_CHECK_SUCCESS(launcher.LaunchKernel(), launcher, state);
        BENCHMARK_CHECK_SUCCESS(launcher.VerifyComputeResult(), launcher, state);

        {
            launcher.BenchmarkEnter();
            auto exitGuard = ScopeExitGuard([&]() { launcher.BenchmarkExit(); });
            for (auto _ : state)
            {
                BENCHMARK_CHECK_SUCCESS(launcher.LaunchKernel(), launcher, state);
            }
        }

        BENCHMARK_CHECK_SUCCESS(launcher.FreeDeviceMemory(), launcher, state);
        BENCHMARK_CHECK_SUCCESS(launcher.FreeHostMemory(), launcher, state);

        SetVectorAddBenchmarkCounters(state, launcher);
    }

    BENCHMARK_DEFINE_F(NvTraceCudaBenchmarkFeature, KernelLargeWorkload)
    (benchmark::State& state)
    {
        VectorAddLauncher launcher;
        launcher.Initialize(static_cast<int>(state.range(0)));
        BENCHMARK_CHECK_SUCCESS(launcher.AllocHostMemory(), launcher, state);
        BENCHMARK_CHECK_SUCCESS(launcher.AllocAndCopyDeviceMemory(), launcher, state);

        {
            launcher.BenchmarkEnter();
            auto exitGuard = ScopeExitGuard([&]() { launcher.BenchmarkExit(); });
            for (auto _ : state)
            {
                BENCHMARK_CHECK_SUCCESS(launcher.LaunchAddTanhKernel(), launcher, state);
            }
        }

        BENCHMARK_CHECK_SUCCESS(launcher.FreeDeviceMemory(), launcher, state);
        BENCHMARK_CHECK_SUCCESS(launcher.FreeHostMemory(), launcher, state);

        SetVectorAddBenchmarkCounters(state, launcher);
    }

    // ===============================
    // CUDA Launcher
    // ===============================
    BENCHMARK_DEFINE_F(NvTraceCudaBenchmarkFeature, SyncKernel_Empty)
    (benchmark::State& state)
    {
        EasyCudaLauncher launcher;

        for (auto _ : state)
        {
            BENCHMARK_CHECK_SUCCESS(launcher.LaunchEmptyKernel(), launcher, state);
            cudaDeviceSynchronize();
        }

        state.SetItemsProcessed(state.iterations());
    }

    BENCHMARK_DEFINE_F(NvTraceCudaBenchmarkFeature, SyncKernel_WaitforNsec)
    (benchmark::State& state)
    {
        EasyCudaLauncher launcher;
        uint64_t gpuWaitDuration = state.range(0);

        {
            for (auto _ : state)
            {
                BENCHMARK_CHECK_SUCCESS(launcher.LaunchWaitKernel(gpuWaitDuration), launcher, state);
                cudaDeviceSynchronize();
            }
        }

        state.SetItemsProcessed(state.iterations());
    }

    BENCHMARK_DEFINE_F(NvTraceCudaBenchmarkFeature, AsyncDenseKernel_Empty)
    (benchmark::State& state)
    {
        EasyCudaLauncher launcher;

        for (auto _ : state)
        {
            BENCHMARK_CHECK_SUCCESS(launcher.LaunchEmptyKernel(), launcher, state);
        }

        state.SetItemsProcessed(state.iterations());
    }

    BENCHMARK_DEFINE_F(NvTraceCudaBenchmarkFeature, AsyncDenseKernel_WaitforNsec)
    (benchmark::State& state)
    {
        EasyCudaLauncher launcher;
        uint64_t gpuWaitDuration = state.range(0);

        for (auto _ : state)
        {
            BENCHMARK_CHECK_SUCCESS(launcher.LaunchWaitKernel(gpuWaitDuration), launcher, state);
        }

        state.SetItemsProcessed(state.iterations());
    }

    BENCHMARK_DEFINE_F(NvTraceCudaBenchmarkFeature, AsyncSparseKernel_Empty)
    (benchmark::State& state)
    {
        EasyCudaLauncher launcher;

        for (auto _ : state)
        {
            BENCHMARK_CHECK_SUCCESS(launcher.LaunchEmptyKernel(), launcher, state);
            BusyWait(1000);
        }

        state.SetItemsProcessed(state.iterations());
    }

    BENCHMARK_DEFINE_F(NvTraceCudaBenchmarkFeature, AsyncSparseKernel_WaitforNsec)
    (benchmark::State& state)
    {
        EasyCudaLauncher launcher;
        uint64_t gpuWaitDuration = state.range(0);

        for (auto _ : state)
        {
            BENCHMARK_CHECK_SUCCESS(launcher.LaunchWaitKernel(gpuWaitDuration), launcher, state);
            BusyWait(1000);
        }

        state.SetItemsProcessed(state.iterations());
    }

    BENCHMARK_DEFINE_F(NvTraceCudaBenchmarkFeature, cudaDriverGetVersion)
    (benchmark::State& state)
    {
        EasyCudaLauncher launcher;

        for (auto _ : state)
        {

            BENCHMARK_CHECK_SUCCESS(launcher.CudaDriverGetVersion(), launcher, state);
        }

        state.SetItemsProcessed(state.iterations());
    }

    BENCHMARK_DEFINE_F(NvTraceCudaBenchmarkFeature, cuDriverGetVersion)
    (benchmark::State& state)
    {
        EasyCudaLauncher launcher;
        for (auto _ : state)
        {
            BENCHMARK_CHECK_SUCCESS(launcher.CuDriverGetVersion(), launcher, state);
        }

        state.SetItemsProcessed(state.iterations());
    }

    BENCHMARK_DEFINE_F(NvTraceCudaBenchmarkFeature, cuGetDeviceCount)
    (benchmark::State& state)
    {
        EasyCudaLauncher launcher;

        for (auto _ : state)
        {
            BENCHMARK_CHECK_SUCCESS(launcher.CuDeviceGetCount(), launcher, state);
        }

        state.SetItemsProcessed(state.iterations());
    }


    BENCHMARK_DEFINE_F(NvTraceCudaBenchmarkFeature, cudaStreamTest)
    (benchmark::State& state)
    {
        EasyCudaLauncher launcher;

        for (auto _ : state)
        {

            BENCHMARK_CHECK_SUCCESS(launcher.CudaStreamTest(), launcher, state);
        }

        state.SetItemsProcessed(state.iterations());
    }

    BENCHMARK_DEFINE_F(NvTraceCudaBenchmarkFeature, cuModuleTest)
    (benchmark::State& state)
    {
        EasyCudaLauncher launcher;

        for (auto _ : state)
        {
            BENCHMARK_CHECK_SUCCESS(launcher.CuModuleTest(), launcher, state);
        }

        state.SetItemsProcessed(state.iterations());
    }

    BENCHMARK_DEFINE_F(NvTraceCudaBenchmarkFeature, SyntheticTest_Async1us1000TimesThenSync)
    (benchmark::State& state)
    {
        EasyCudaLauncher launcher;
        uint64_t gpuWaitDurationNs = 1000;

        for (auto _ : state)
        {
            for (int i = 0; i < 1000; i++)
            {
                BENCHMARK_CHECK_SUCCESS(launcher.LaunchWaitKernel(gpuWaitDurationNs), launcher, state);
            }
            cudaDeviceSynchronize();
        }

        state.SetItemsProcessed(state.iterations());
    }

    // ===============================
    // GraphLauncher
    // ===============================
    BENCHMARK_DEFINE_F(NvTraceCudaBenchmarkFeature, AsyncGraphKernel_Empty)
    (benchmark::State& state)
    {
        GraphLauncher graphLauncher;

        BENCHMARK_CHECK_SUCCESS(graphLauncher.Initialize(), graphLauncher, state);
        auto exitGuard = ScopeExitGuard([&]() { graphLauncher.Reset(); });
        BENCHMARK_CHECK_SUCCESS(graphLauncher.setEmptyKernel(), graphLauncher, state);

        for (auto _ : state)
        {
            BENCHMARK_CHECK_SUCCESS(graphLauncher.LaunchGraph(), graphLauncher, state);
        }

        state.SetItemsProcessed(state.iterations());
    }

    BENCHMARK_DEFINE_F(NvTraceCudaBenchmarkFeature, AsyncGraphKernel_WaitforNsec)
    (benchmark::State& state)
    {
        uint64_t gpuWaitDuration = state.range(0);

        GraphLauncher graphLauncher;
        BENCHMARK_CHECK_SUCCESS(graphLauncher.Initialize(), graphLauncher, state);
        auto exitGuard = ScopeExitGuard([&]() { graphLauncher.Reset(); });
        BENCHMARK_CHECK_SUCCESS(graphLauncher.setWaitKernel(gpuWaitDuration), graphLauncher, state);

        for (auto _ : state)
        {
            BENCHMARK_CHECK_SUCCESS(graphLauncher.LaunchGraph(), graphLauncher, state);
        }

        state.SetItemsProcessed(state.iterations());
    }

    // Register benchmarks
    BENCHMARK_REGISTER_F(NvTraceCudaBenchmarkFeature, EndToEnd)->Arg(1);
    BENCHMARK_REGISTER_F(NvTraceCudaBenchmarkFeature, KernelAdd)->Arg(1);
    BENCHMARK_REGISTER_F(NvTraceCudaBenchmarkFeature, KernelLargeWorkload)->Arg(1);

    BENCHMARK_REGISTER_F(NvTraceCudaBenchmarkFeature, SyncKernel_Empty)->Arg(1);
    BENCHMARK_REGISTER_F(NvTraceCudaBenchmarkFeature, SyncKernel_WaitforNsec)->Arg(1000)->Arg(10000)->Arg(100000);
    BENCHMARK_REGISTER_F(NvTraceCudaBenchmarkFeature, AsyncDenseKernel_Empty)->Arg(1);
    BENCHMARK_REGISTER_F(NvTraceCudaBenchmarkFeature, AsyncDenseKernel_WaitforNsec)->Arg(1000)->Arg(10000)->Arg(100000);
    BENCHMARK_REGISTER_F(NvTraceCudaBenchmarkFeature, AsyncSparseKernel_Empty)->Arg(1);
    BENCHMARK_REGISTER_F(NvTraceCudaBenchmarkFeature, AsyncSparseKernel_WaitforNsec)->Arg(1000)->Arg(10000)->Arg(100000);
    BENCHMARK_REGISTER_F(NvTraceCudaBenchmarkFeature, AsyncGraphKernel_Empty)->Arg(1);
    BENCHMARK_REGISTER_F(NvTraceCudaBenchmarkFeature, AsyncGraphKernel_WaitforNsec)->Arg(1000)->Arg(10000)->Arg(100000);
    BENCHMARK_REGISTER_F(NvTraceCudaBenchmarkFeature, SyntheticTest_Async1us1000TimesThenSync)->Arg(1);

    BENCHMARK_REGISTER_F(NvTraceCudaBenchmarkFeature, cudaDriverGetVersion)->Arg(1);
    BENCHMARK_REGISTER_F(NvTraceCudaBenchmarkFeature, cuDriverGetVersion)->Arg(1);
    BENCHMARK_REGISTER_F(NvTraceCudaBenchmarkFeature, cuGetDeviceCount)->Arg(1);
    BENCHMARK_REGISTER_F(NvTraceCudaBenchmarkFeature, cudaStreamTest)->Arg(1);
    // BENCHMARK_REGISTER_F(NvTraceCudaBenchmarkFeature, cuModuleTest)->Arg(1); // comment out because of instability

} // namespace nv::trace

int main(int argc, char** argv)
{
    // Parse command line arguments
    std::vector<char*> updatedArgv{argv[0]};

    cxxopts::Options options("CUDA End-to-End Benchmark Options");
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

    // Reconstruct arguments not parsed by cxxopts
    for (const auto& unmatched : result.unmatched())
    {
        updatedArgv.push_back(const_cast<char*>(unmatched.c_str()));
    }
    int updatedArgc = static_cast<int>(updatedArgv.size());

    // Initialize Google Benchmark
    ::benchmark::Initialize(&updatedArgc, updatedArgv.data());
    if (::benchmark::ReportUnrecognizedArguments(updatedArgc, updatedArgv.data()))
    {
        return 1;
    }

    // Initialize the nvidia-nsight-perf service if requested
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

    // Run benchmarks
    ::benchmark::RunSpecifiedBenchmarks();
    ::benchmark::Shutdown();

    // Clean up the nvidia-nsight-perf service if it was started
    if (g_BenchmarkOptions.runWithNvTrace)
    {
        g_nvTraceService.Reset();
    }

    return 0;
}
