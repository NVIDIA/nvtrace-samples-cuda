/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <NvTrace/Target/NvTraceOutOfProcessService.h>

#include <benchmark/benchmark.h>
#include <cuda.h>

namespace nv::trace {

    static void BM_NvTraceServiceInitialization(benchmark::State& state)
    {
        cuInit(0);

        for (auto _ : state)
        {
            OutOfProcessService nvTraceService;
            if (!nvTraceService.Initialize())
            {
                state.SkipWithError("Failed to initialize out-of-process nvidia-nsight-perf service");
                return;
            }
            nvTraceService.Reset();
        }
    }

    BENCHMARK(BM_NvTraceServiceInitialization);

} // namespace nv::trace

BENCHMARK_MAIN();
