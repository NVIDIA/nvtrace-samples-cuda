/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstddef>
#include <gtest/gtest-spi.h>
#include <gtest/gtest.h>

#include "cxxopts.hpp"
#include <NvTrace/Offline/DataModel/NvTraceStreamDataModelBuilder.h>
#include <NvTrace/Offline/DataModel/NvTraceStreamDatabaseReader.h>
#include <NvTrace/Offline/Exporter/NvTracePerfettoTraceExporter.h>
#include <NvTrace/Offline/NvTracePrintHelpers.h>
#include <NvTrace/Target/Consumer/NvTraceStreamingConsumer.h>
#include <NvTrace/Target/NvTraceClientLogger.h>
#include <NvTrace/Target/NvTraceCpuTimestamp.h>
#include <NvTrace/Target/NvTraceScopeExitGuard.h>
#include <NvTrace/Target/NvTraceUuidUtilities.h>
#include <NvTraceGTestThrowListener.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <set>
#include <string>
#include <system_error>
#include <thread>
#include <vector>

#if defined(__linux__)
#include <cstdlib>
#endif

#define EXPECT_NEAR_SIZET(a, b, tolerance) EXPECT_NEAR(static_cast<double>(a), static_cast<double>(b), static_cast<double>(tolerance))

constexpr size_t k_toleranceBuffer = 300;

using namespace nv::trace;

using StreamingConsumerType = StreamingConsumer<TsfDiskDatabaseBufferedStorageWriter>;
using TsfDatabaseReaderType = TraceStreamDatabaseReader<TsfDiskDatabaseBufferedStorageReader>;

// ================================
// DataModelValidatorOptions
// ================================
struct DataModelValidatorOptions
{
    std::string outputFilename = "";

    bool isCudartTraceEnabled = false;
    uint32_t durationMs = 2000;     // streaming consumer duration
    uint32_t launches = 0xFFFFFFFF; // number of events from user application
    uint32_t processN = 1;          // number of processes to collect
};

DataModelValidatorOptions g_dataModelValidatorOptions;

static void ParseDataModelValidatorOptions(int argc, char** argv)
{
    cxxopts::Options options("Options");
    options.add_options()("duration-ms", "Duration in milliseconds", cxxopts::value<uint32_t>()->default_value("2000"));
    options.add_options()("launches", "Number of launches expected", cxxopts::value<uint32_t>()->default_value("0xFFFFFFFF"));
    options.add_options()("processN", "Number of processes expected", cxxopts::value<uint32_t>()->default_value("1"));
    options.add_options()("output", "The filename to export the Perfetto Trace to.", cxxopts::value<std::string>()->default_value(""));
    options.add_options()("enable-cudart-trace", "Enable CUDA Runtime API trace.", cxxopts::value<bool>()->default_value("false"));

    auto result = options.parse(argc, argv);
    if (result.count("help"))
    {
        printf("%s\n", options.help().c_str());
        exit(0);
    }

    g_dataModelValidatorOptions.durationMs = result["duration-ms"].as<uint32_t>();
    g_dataModelValidatorOptions.launches = result["launches"].as<uint32_t>();
    g_dataModelValidatorOptions.processN = result["processN"].as<uint32_t>();
    g_dataModelValidatorOptions.outputFilename = result["output"].as<std::string>();
    g_dataModelValidatorOptions.isCudartTraceEnabled = result["enable-cudart-trace"].as<bool>();
}

static std::filesystem::path CreateTemporaryTsfDatabaseDirectory()
{
    std::error_code errorCode;
    const std::filesystem::path parent = std::filesystem::temp_directory_path(errorCode);
    if (errorCode)
    {
        return {};
    }

#if defined(__linux__)
    const std::string pathTemplate = (parent / "nvtrace-tsf-XXXXXX").string();
    std::vector<char> pathBuffer(pathTemplate.begin(), pathTemplate.end());
    pathBuffer.push_back('\0');
    if (mkdtemp(pathBuffer.data()) == nullptr)
    {
        return {};
    }
    return std::filesystem::path(pathBuffer.data());
#else
    const auto uniqueSuffix = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    const std::filesystem::path path = parent / ("nvtrace-tsf-" + std::to_string(uniqueSuffix));
    if (!std::filesystem::create_directory(path, errorCode) || errorCode)
    {
        return {};
    }
    return path;
#endif
}

// ================================
// Global Stats
// ================================
namespace {

    using RuntimeApiIds = std::set<FB::Provider::CudaApiTrace::CudaRuntimeApiFunctionId>;

    const RuntimeApiIds RuntimeApiFunctionIdsForKernelLaunch = {
        FB::Provider::CudaApiTrace::CudaRuntimeApiFunctionId_cudaLaunchKernel_ptsz_v7000,
        FB::Provider::CudaApiTrace::CudaRuntimeApiFunctionId_cudaLaunchKernel_v7000,
        FB::Provider::CudaApiTrace::CudaRuntimeApiFunctionId_cudaLaunchKernelExC_ptsz_v11060,
        FB::Provider::CudaApiTrace::CudaRuntimeApiFunctionId_cudaLaunchKernelExC_v11060};

    using DriverApiIds = std::set<FB::Provider::CudaApiTrace::CudaDriverApiFunctionId>;

    const DriverApiIds DriverApiFunctionIdsForKernelLaunch = {
        FB::Provider::CudaApiTrace::CudaDriverApiFunctionId_cuLaunchKernel,
        FB::Provider::CudaApiTrace::CudaDriverApiFunctionId_cuLaunchKernel_ptsz,
        FB::Provider::CudaApiTrace::CudaDriverApiFunctionId_cuLaunchKernelEx,
        FB::Provider::CudaApiTrace::CudaDriverApiFunctionId_cuLaunchKernelEx_ptsz};

    struct Stats
    {
        struct RuntimeApiCalls
        {
            size_t cudaLaunchKernelCount = 0;
        };
        RuntimeApiCalls runtimeApiCalls;

        struct DriverApiCalls
        {
            size_t cuLaunchKernelCount = 0;
        };
        DriverApiCalls driverApiCalls;

        struct ImmediateKernelLaunches
        {
            size_t totalCount = 0;
            size_t resolvedCount = 0;
            size_t invalidTimestampCount = 0;
            size_t invalidRecordLocationCount = 0;
            size_t invalidEventTypeCount = 0;
            size_t invalidFunctionIdCount = 0;

            std::string ToString() const
            {
                return SprintfToString("totalCount: %zu, resolvedCount: %zu, invalidTimestampCount: %zu, invalidRecordLocationCount: %zu, invalidEventTypeCount: %zu, invalidFunctionIdCount: %zu", totalCount, resolvedCount, invalidTimestampCount, invalidRecordLocationCount, invalidEventTypeCount, invalidFunctionIdCount);
            }
        };
        ImmediateKernelLaunches immediateKernelLaunches;
    };
} // namespace

Stats g_stats;

template <typename TTraceReader>
static void checkImmediateKernelLaunches(const Codec::DataModel<TTraceReader>& dataModel, const Codec::CudaApiTraceData& cudaApiTrace, const Codec::CudaApiTraceData::CuDeviceData& cudaDevice)
{
    g_stats.immediateKernelLaunches.totalCount += cudaDevice.immediateKernelLaunches.size();

    for (const auto& immediateKernelLaunch : cudaDevice.immediateKernelLaunches)
    {
        const bool gpuStartTsValid = (immediateKernelLaunch.startTimestampGpuNsec != Codec::InvalidTimestamp);
        const bool gpuEndTsValid = (immediateKernelLaunch.endTimestampGpuNsec != Codec::InvalidTimestamp);
        if (!gpuStartTsValid || !gpuEndTsValid)
        {
            g_stats.immediateKernelLaunches.invalidTimestampCount++;
            continue;
        }

        const NvTrace_CUDA_ApiTrace_ImmediateKernelLaunch* pImmediateKernelLaunchEvent = dataModel.GetImmediateKernelLaunchPayload(immediateKernelLaunch.recordLocation);
        if (!pImmediateKernelLaunchEvent)
        {
            g_stats.immediateKernelLaunches.invalidRecordLocationCount++;
            continue;
        }

        const uint64_t cufunctionId = pImmediateKernelLaunchEvent->cufunctionId;
        auto functionItr = cudaApiTrace.functions.find(cufunctionId);
        if (functionItr == cudaApiTrace.functions.end())
        {
            g_stats.immediateKernelLaunches.invalidFunctionIdCount++;
            continue;
        }

        g_stats.immediateKernelLaunches.resolvedCount++;
    }
}

// ================================
// Test
// ================================
TEST(DataModelValidator, ValidateCompleteness)
{
    const std::filesystem::path tsfDatabaseDir = CreateTemporaryTsfDatabaseDirectory();
    ASSERT_FALSE(tsfDatabaseDir.empty()) << "Failed to create a temporary TSF database directory";
    auto removeTsfDatabaseDirGuard = ScopeExitGuard([&]() {
        std::error_code errorCode;
        std::filesystem::remove_all(tsfDatabaseDir, errorCode);
    });

    StreamingConsumerType streamingConsumer;
    typename StreamingConsumerType::Options consumerOptions;
    ASSERT_TRUE(streamingConsumer.Initialize(consumerOptions, [&](TsfDiskDatabaseBufferedStorageWriter& storageWriter) {
        NVTRACE_LOG_INF(50, "Creating TSF database at: %s", tsfDatabaseDir.string().c_str());
        return storageWriter.Initialize(tsfDatabaseDir);
    })) << "StreamingConsumer initialization failed. The nvidia-nsight-perf service may not be running.";

    NVTRACE_LOG_INF(50, "Starting NvTrace streaming consumer...");
    const uint64_t observationStartUtcNsec = streamingConsumer.ConvertTscTickToUtcNsec(ReadCpuTscTimestamp());
    ASSERT_TRUE(streamingConsumer.Start());
    std::this_thread::sleep_for(std::chrono::milliseconds(g_dataModelValidatorOptions.durationMs));
    const uint64_t observationEndUtcNsec = streamingConsumer.ConvertTscTickToUtcNsec(ReadCpuTscTimestamp());
    ASSERT_TRUE(streamingConsumer.StopAtUtc(observationEndUtcNsec));
    streamingConsumer.Reset();

    TsfDatabaseReaderType databaseReader;
    ASSERT_TRUE(databaseReader.Initialize([&](TsfDiskDatabaseBufferedStorageReader& storageReader) {
        return storageReader.Initialize(tsfDatabaseDir);
    })) << "Failed to open TSF database at "
        << tsfDatabaseDir.string();

    typename Codec::DataModelBuilder<TsfDatabaseReaderType>::Options dataModelBuilderOptions;
    dataModelBuilderOptions.observationStartUtcNsec = observationStartUtcNsec;
    dataModelBuilderOptions.observationEndUtcNsec = observationEndUtcNsec;

    Codec::DataModel<TsfDatabaseReaderType> dataModel;
    Codec::DataModelBuilder<TsfDatabaseReaderType>(dataModel).Build(databaseReader, dataModelBuilderOptions);

    // validate cpu clock calibration points
    NVTRACE_LOG_INF(20, "CPU clock calibration points: %zu", dataModel.cpuClockCalibration.calibrationPoints.size());
    ASSERT_GT(dataModel.cpuClockCalibration.calibrationPoints.size(), 0) << "CPU clock calibration points were missing!";

    // Validate runtime/driver API calls
    {
        ASSERT_EQ(dataModel.processes.size(), g_dataModelValidatorOptions.processN) << "Process count failed to match!";

        for (const auto& [processLuid, process] : dataModel.processes)
        {
            for (const auto& [uniqueThreadId, thread] : process.cudaApiTrace.threads)
            {
                for (const auto& runtimeApiCall : thread.runtimeApiCalls)
                {
                    if (RuntimeApiFunctionIdsForKernelLaunch.count(runtimeApiCall.runtimeApiFunctionId))
                    {
                        g_stats.runtimeApiCalls.cudaLaunchKernelCount++;
                    }
                }

                for (const auto& driverApiCall : thread.driverApiCalls)
                {
                    if (DriverApiFunctionIdsForKernelLaunch.count(driverApiCall.driverApiFunctionId))
                    {
                        g_stats.driverApiCalls.cuLaunchKernelCount++;
                    }
                }
            }
        }

        if (g_dataModelValidatorOptions.launches != 0xFFFFFFFF) // if not set, we don't care about the number of launches
        {
            if (g_dataModelValidatorOptions.isCudartTraceEnabled)
            {
                NVTRACE_LOG_INF(20, "CUDA runtime immediate kernel launch samples: %zu", g_stats.runtimeApiCalls.cudaLaunchKernelCount);
                const size_t cudaLaunchKernelExpectedCount = g_dataModelValidatorOptions.launches;
                const size_t cudaLaunchKernelDeltaTolerance = cudaLaunchKernelExpectedCount * 5 / 100 + k_toleranceBuffer; // tolerance: 5%
                EXPECT_NEAR_SIZET(g_stats.runtimeApiCalls.cudaLaunchKernelCount, cudaLaunchKernelExpectedCount, cudaLaunchKernelDeltaTolerance) << "Runtime immediate kernel launch samples failed to match!";
            }

            NVTRACE_LOG_INF(20, "CUDA driver immediate kernel launch samples: %zu", g_stats.driverApiCalls.cuLaunchKernelCount);
            const size_t cuLaunchKernelExpectedCount = g_dataModelValidatorOptions.launches;
            const size_t cuLaunchKernelDeltaTolerance = cuLaunchKernelExpectedCount * 5 / 100 + k_toleranceBuffer; // tolerance: 5%
            EXPECT_NEAR_SIZET(g_stats.driverApiCalls.cuLaunchKernelCount, cuLaunchKernelExpectedCount, cuLaunchKernelDeltaTolerance) << "Driver immediate kernel launch samples failed to match!";
        }
    }


    // Validate GPU events
    {
        for (const auto& [gpuPartitionLuid, gpuPartition] : dataModel.gpuPartitions)
        {
            std::string gpuUuidStr = ToString(gpuPartition.gpuUuid);

            NVTRACE_LOG_INF(20, "GPU clock calibration points: %zu", gpuPartition.gpuClockCalibration.calibrationPoints.size());
            ASSERT_GT(gpuPartition.gpuClockCalibration.calibrationPoints.size(), 0) << "GPU clock calibration points were missing!";

            NVTRACE_LOG_INF(20, "GPU PM counters: %zu", gpuPartition.gpuPmCounters.sampleRecordLocations.size());
            const size_t expectedGpuPmCountersSamples = g_dataModelValidatorOptions.durationMs;        // assume: 1000 Hz
            const size_t deltaTolerance = expectedGpuPmCountersSamples * 15 / 100 + k_toleranceBuffer; // tolerance: 15% (give it a higher tolerance since this is a time-based measurement)
            EXPECT_NEAR_SIZET(gpuPartition.gpuPmCounters.sampleRecordLocations.size(), expectedGpuPmCountersSamples, deltaTolerance) << "GPU PM counters failed to match!";

            // Validate immediate kernel launches
            for (const auto& [processLuid, process] : dataModel.processes)
            {
                for (const auto& cudaDevice : process.cudaApiTrace.devices)
                {
                    if (cudaDevice.gpuUuid == gpuPartition.gpuUuid)
                    {
                        checkImmediateKernelLaunches(dataModel, process.cudaApiTrace, cudaDevice);
                        break;
                    }
                }
            }
        }

        if (g_dataModelValidatorOptions.launches != 0xFFFFFFFF)
        {
            NVTRACE_LOG_INF(20, "Immediate kernel launches resolved samples: %zu", g_stats.immediateKernelLaunches.resolvedCount);
            const size_t expectedImmediateKernelLaunches = g_dataModelValidatorOptions.launches;
            const size_t immediateKernelLaunchesDeltaTolerance = expectedImmediateKernelLaunches * 10 / 100 + k_toleranceBuffer; // tolerance: 10%
            EXPECT_NEAR_SIZET(g_stats.immediateKernelLaunches.resolvedCount, expectedImmediateKernelLaunches, immediateKernelLaunchesDeltaTolerance) << "Immediate kernel launches failed to match! Current stats:\n"
                                                                                                                                                     << g_stats.immediateKernelLaunches.ToString();
        }
    }

    if (!g_dataModelValidatorOptions.outputFilename.empty())
    {
        nv::trace::PerfettoTraceExporterOptions traceExporterOptions = {};
        traceExporterOptions.enableDebugEvents = true;

        nv::trace::PerfettoTraceExporter<TsfDatabaseReaderType> perfettoTraceExporter(dataModel, traceExporterOptions);
        perfettoTraceExporter.Export(g_dataModelValidatorOptions.outputFilename);
    }
}

int main(int argc, char** argv)
{
    // Call InitGooleTest before ParseGlobalTestOptions so that GTest can parse the
    //   --gtest_* options; place those options on the command line before our
    //   custom options; GTest will advance argc as it consumes options
    ::testing::InitGoogleTest(&argc, argv);

    ParseDataModelValidatorOptions(argc, argv);

    AddGTestThrowListener();

    return RUN_ALL_TESTS();
}
