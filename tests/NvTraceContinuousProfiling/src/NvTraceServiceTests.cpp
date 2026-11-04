/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest-spi.h>
#include <gtest/gtest.h>

#include <HostCallbacks.h>
#include <ImmediateMemset.h>
#include <VectorAdd.h>
#include <hmmaCoupledFlops.h>
#include <immaTensorCoreGemm.h>
#include <simpleCudaGraphs.h>
#include <simpleMultiGPU.h>

#include <NvTrace/Offline/DataModel/NvTraceStreamDataModelBuilder.h>
#include <NvTrace/Offline/DataModel/NvTraceStreamDatabaseReader.h>
#include <NvTrace/Offline/Exporter/NvTracePerfettoTraceExporter.h>
#include <NvTrace/Target/Consumer/NvTraceStreamingConsumer.h>
#include <NvTrace/Target/Consumer/NvTraceTsfSnapshotConsumer.h>
#include <NvTrace/Target/NvTraceCpuTimestamp.h>
#include <NvTrace/Target/NvTraceOutOfProcessService.h>
#include <NvTrace/Target/NvTraceScopeExitGuard.h>

#include "GlobalTestOptions.h"

#include <perfetto_trace.pb.h>

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <string>
#include <system_error>
#include <unordered_map>
#include <vector>

#if defined(__linux__)
#include <cstdlib>
#endif

#include <nvtx3/nvtx3.hpp>

namespace nv::trace {

    struct TraceCategoryCounts
    {
        size_t systemInfoEvents = 0;
        size_t gpuGrctxEvents = 0;
        size_t gpuPmEvents = 0;
        size_t gpuKernelEvents = 0;
        size_t gpuDtoHMemcpyEvents = 0;
        size_t nvtxEvents = 0;
        size_t cudaApiEvents = 0;
        size_t totalEvents = 0;
        size_t internedStringValues = 0;
        size_t stringValueIidAnnotations = 0;
        size_t inlineStringAnnotations = 0;
        size_t unresolvedStringValueIids = 0;
        size_t unknownGpuEventNames = 0;
        std::string firstUnknownGpuEventName;
    };

    constexpr uint32_t kCudaDeviceAttributeComputeCapabilityMajor = 75; // CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR
    constexpr int32_t kMinComputeCapabilityMajorForNvTrace = 10;

    using StreamingConsumerType = StreamingConsumer<TsfDiskDatabaseBufferedStorageWriter>;
    using TsfSnapshotConsumerType = TsfSnapshotConsumer<TsfDiskDatabaseBufferedStorageWriter>;
    using TsfDatabaseReaderType = TraceStreamDatabaseReader<TsfDiskDatabaseBufferedStorageReader>;

    struct TsfSnapshotCompletionState
    {
        std::mutex mutex;
        std::condition_variable cv;
        bool completed = false;
        bool succeeded = false;
    };

    void OnTsfSnapshotCompleted(void* pUserData, uint32_t, bool succeeded)
    {
        auto& state = *static_cast<TsfSnapshotCompletionState*>(pUserData);
        {
            std::lock_guard<std::mutex> lock(state.mutex);
            state.succeeded = succeeded;
            state.completed = true;
        }
        state.cv.notify_one();
    }

    std::filesystem::path CreateTemporaryTsfDatabaseDirectory()
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

    template <typename TTraceReader>
    int32_t GetMaxComputeCapabilityMajor(const Codec::DataModel<TTraceReader>& dataModel)
    {
        int32_t maxMajor = 0;

        for (const auto& [_, gpuPartition] : dataModel.gpuPartitions)
        {
            for (const NvTrace_CudaAttribute& attribute : gpuPartition.info.attributes)
            {
                if (attribute.key == kCudaDeviceAttributeComputeCapabilityMajor)
                {
                    maxMajor = std::max(maxMajor, attribute.value);
                }
            }
        }

        return maxMajor;
    }

    void IncrementCategoryCount(TraceCategoryCounts& counts, const std::string& category)
    {
        if (category == "system_info")
        {
            ++counts.systemInfoEvents;
        }
        else if (category == "gpu_grctx")
        {
            ++counts.gpuGrctxEvents;
        }
        else if (category == "gpu_pm")
        {
            ++counts.gpuPmEvents;
        }
        else if (category == "cuda_gpu_kernel_launch")
        {
            ++counts.gpuKernelEvents;
        }
        else if (category.rfind("nvtx", 0) == 0)
        {
            ++counts.nvtxEvents;
        }
        else if (category == "cuda_runtime" || category == "cuda_driver")
        {
            ++counts.cudaApiEvents;
        }
    }

    std::string ResolveInternedCategory(
        const perfetto::protos::TrackEvent& trackEvent,
        const std::unordered_map<uint64_t, std::string>& categoryIids)
    {
        if (trackEvent.category_iids().empty())
        {
            return {};
        }
        const auto it = categoryIids.find(trackEvent.category_iids(0));
        return it != categoryIids.end() ? it->second : std::string{};
    }

    std::string ResolveInternedEventName(
        const perfetto::protos::TrackEvent& trackEvent,
        const std::unordered_map<uint64_t, std::string>& eventNameIids)
    {
        const auto it = eventNameIids.find(trackEvent.name_iid());
        return it != eventNameIids.end() ? it->second : std::string{};
    }

    TraceCategoryCounts CountPerfettoTraceEventsByCategory(const std::vector<uint8_t>& traceBytes)
    {
        TraceCategoryCounts counts;
        std::unordered_map<uint64_t, std::string> categoryIids;
        std::unordered_map<uint64_t, std::string> eventNameIids;
        std::unordered_map<uint64_t, std::string> stringValueIids;

        // The writer streams framed Trace.packet fields; concatenated they are a
        // valid Trace message, so the generated parser is enough for this test.
        perfetto::protos::Trace trace;
        if (!trace.ParseFromArray(traceBytes.data(), static_cast<int>(traceBytes.size())))
        {
            return counts;
        }

        for (const perfetto::protos::TracePacket& packet : trace.packet())
        {
            if (packet.has_interned_data())
            {
                for (const perfetto::protos::EventCategory& category : packet.interned_data().event_categories())
                {
                    categoryIids.emplace(category.iid(), category.name());
                }
                for (const perfetto::protos::EventName& eventName : packet.interned_data().event_names())
                {
                    eventNameIids.emplace(eventName.iid(), eventName.name());
                }
                for (const perfetto::protos::InternedString& stringValue : packet.interned_data().debug_annotation_string_values())
                {
                    stringValueIids.emplace(stringValue.iid(), stringValue.str());
                    ++counts.internedStringValues;
                }
            }

            if (!packet.has_track_event())
            {
                continue;
            }

            const perfetto::protos::TrackEvent& trackEvent = packet.track_event();
            for (const perfetto::protos::DebugAnnotation& annotation : trackEvent.debug_annotations())
            {
                if (annotation.has_string_value())
                {
                    ++counts.inlineStringAnnotations;
                }
                if (annotation.has_string_value_iid())
                {
                    ++counts.stringValueIidAnnotations;
                    if (stringValueIids.find(annotation.string_value_iid()) == stringValueIids.end())
                    {
                        ++counts.unresolvedStringValueIids;
                    }
                }
            }
            switch (trackEvent.type())
            {
                case perfetto::protos::TrackEvent::TYPE_SLICE_END:
                {
                    break;
                }
                case perfetto::protos::TrackEvent::TYPE_COUNTER:
                {
                    ++counts.totalEvents;
                    ++counts.gpuPmEvents;
                    break;
                }
                case perfetto::protos::TrackEvent::TYPE_SLICE_BEGIN:
                case perfetto::protos::TrackEvent::TYPE_INSTANT:
                {
                    ++counts.totalEvents;
                    const std::string category = ResolveInternedCategory(trackEvent, categoryIids);
                    IncrementCategoryCount(counts, category);
                    const std::string eventName = ResolveInternedEventName(trackEvent, eventNameIids);
                    if (category.rfind("cuda_gpu_", 0) == 0)
                    {
                        if (eventName.rfind("MemcpyD2H(", 0) == 0)
                        {
                            ++counts.gpuDtoHMemcpyEvents;
                        }
                        if (eventName.find("??") != std::string::npos)
                        {
                            ++counts.unknownGpuEventNames;
                            if (counts.firstUnknownGpuEventName.empty())
                            {
                                counts.firstUnknownGpuEventName = eventName;
                            }
                        }
                    }
                    break;
                }
                default:
                {
                    break;
                }
            }
        }

        return counts;
    }

    template <typename TTraceReader>
    size_t CountDataModelEvents(const Codec::DataModel<TTraceReader>& dataModel)
    {
        size_t eventCount = 0;

        for (const auto& [_, process] : dataModel.processes)
        {
            for (const auto& domainClique : process.nvtxTrace.domainCliques)
            {
                for (const auto& threadEntry : domainClique.threads)
                {
                    const auto& thread = threadEntry.second;
                    eventCount += thread.marks.size();
                    eventCount += thread.pushPopRanges.size();
                }
            }

            for (const auto& [__, thread] : process.cudaApiTrace.threads)
            {
                eventCount += thread.runtimeApiCalls.size();
                eventCount += thread.driverApiCalls.size();
            }
        }

        return eventCount;
    }

    // Validate the data model before exporting it.
    template <typename TTraceReader>
    void ValidateDataModel(const Codec::DataModel<TTraceReader>& dataModel)
    {
        EXPECT_FALSE(dataModel.processes.empty()) << "Data model has no processes";
        printf("Data model validation completed (events=%zu).\n", CountDataModelEvents(dataModel));
    }

    // Validate the exported Perfetto protobuf trace file.
    void ValidateExportedPerfettoTrace(const char* tracePath, int32_t maxComputeCapabilityMajor)
    {
        std::ifstream input(tracePath, std::ios::binary);
        ASSERT_TRUE(input.is_open()) << "Failed to open Perfetto trace: " << tracePath;

        const std::vector<uint8_t> traceBytes((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
        ASSERT_FALSE(traceBytes.empty()) << "Perfetto trace is empty";

        const TraceCategoryCounts counts = CountPerfettoTraceEventsByCategory(traceBytes);
        printf(
            "Perfetto trace category counts: total=%zu, system_info=%zu, gpu_grctx=%zu, gpu_pm=%zu, "
            "gpu_kernels=%zu, gpu_dtoh_memcpy=%zu, nvtx=%zu, cuda_api=%zu (max compute capability major=%d).\n",
            counts.totalEvents,
            counts.systemInfoEvents,
            counts.gpuGrctxEvents,
            counts.gpuPmEvents,
            counts.gpuKernelEvents,
            counts.gpuDtoHMemcpyEvents,
            counts.nvtxEvents,
            counts.cudaApiEvents,
            maxComputeCapabilityMajor);
        printf(
            "Perfetto annotation string interning: values=%zu, iid uses=%zu, inline uses=%zu, unresolved iids=%zu.\n",
            counts.internedStringValues,
            counts.stringValueIidAnnotations,
            counts.inlineStringAnnotations,
            counts.unresolvedStringValueIids);
        printf("Perfetto GPU slice names containing ??: %zu.\n", counts.unknownGpuEventNames);

        EXPECT_GT(counts.totalEvents, 0u) << "Perfetto trace has no events";
        EXPECT_EQ(counts.unknownGpuEventNames, 0u)
            << "Perfetto trace contains GPU events containing ?? " << ", first: " << counts.firstUnknownGpuEventName;
        EXPECT_GT(counts.internedStringValues, 0u) << "Perfetto trace has no interned annotation string values";
        EXPECT_GT(counts.stringValueIidAnnotations, 0u) << "Perfetto trace has no interned annotation string references";
        EXPECT_EQ(counts.inlineStringAnnotations, 0u) << "Perfetto trace contains non-interned annotation strings";
        EXPECT_EQ(counts.unresolvedStringValueIids, 0u) << "Perfetto trace contains unresolved annotation string references";

        if (maxComputeCapabilityMajor >= kMinComputeCapabilityMajorForNvTrace)
        {
            EXPECT_GT(counts.gpuKernelEvents, 0u) << "Perfetto trace is missing GPU kernel events";
            EXPECT_GT(counts.cudaApiEvents, 0u) << "Perfetto trace is missing CUDA API trace events";
            EXPECT_GT(counts.systemInfoEvents, 0u) << "Perfetto trace is missing System Info events";
            EXPECT_GT(counts.gpuGrctxEvents, 0u) << "Perfetto trace is missing GR Context events";
            EXPECT_GT(counts.gpuPmEvents, 0u) << "Perfetto trace is missing GPU PM counter events";
            if (g_TestOptions.selectedSamples.count(SampleId::VectorAdd))
            {
                EXPECT_GT(counts.gpuDtoHMemcpyEvents, 0u) << "Perfetto trace is missing GPU MemcpyD2H events";
            }
        }
        else
        {
            printf(
                "Skipping checks due to unsupported GPU (max compute capability major %d < %d).\n",
                maxComputeCapabilityMajor,
                kMinComputeCapabilityMajorForNvTrace);
        }
        EXPECT_GT(counts.nvtxEvents, 0u) << "Perfetto trace is missing NVTX events";
    }

    template <typename TTraceReader>
    void ExportAndValidateTrace(Codec::DataModel<TTraceReader>& dataModel)
    {
        ValidateDataModel(dataModel);

        const int32_t maxComputeCapabilityMajor = GetMaxComputeCapabilityMajor(dataModel);

        PerfettoTraceExporterOptions options;
        options.enableDebugEvents = true;
        options.enableTraceArchive = false;
        PerfettoTraceExporter<TTraceReader> perfettoTraceExporter(dataModel, options);
        perfettoTraceExporter.Export(g_TestOptions.traceOutputPath);

        ValidateExportedPerfettoTrace(g_TestOptions.traceOutputPath.c_str(), maxComputeCapabilityMajor);
    }

    void ExportTsfDatabaseAndValidate(
        const std::filesystem::path& tsfDatabaseDir,
        uint64_t observationStartUtcNsec,
        uint64_t observationEndUtcNsec)
    {
        TsfDatabaseReaderType databaseReader;
        ASSERT_TRUE(databaseReader.Initialize([&](TsfDiskDatabaseBufferedStorageReader& storageReader) {
            return storageReader.Initialize(tsfDatabaseDir);
        })) << "Failed to open TSF database at "
            << tsfDatabaseDir.string();

        typename Codec::DataModelBuilder<TsfDatabaseReaderType>::Options dataModelBuilderOptions;
        dataModelBuilderOptions.observationStartUtcNsec = observationStartUtcNsec;
        dataModelBuilderOptions.observationEndUtcNsec = observationEndUtcNsec;

        Codec::DataModel<TsfDatabaseReaderType> dataModel;
        Codec::DataModelBuilder<TsfDatabaseReaderType> dataModelBuilder(dataModel);
        dataModelBuilder.Build(databaseReader, dataModelBuilderOptions);

        ExportAndValidateTrace(dataModel);
    }

    void WriteBinaryFile(const std::string& filename, const std::vector<uint8_t>& buffer)
    {
        std::ofstream outFile(filename, std::ios::binary);
        ASSERT_TRUE(outFile.is_open());
        outFile.write(reinterpret_cast<const char*>(buffer.data()), buffer.size());
        outFile.close();
    }
    int ProfileSample(int (*pfnCudaWorkload)(void), const std::string& eventName)
    {
        nvtxRangePush(eventName.c_str());
        int sampleStatus = pfnCudaWorkload();
        printf(".");
        nvtxRangePop();
        return sampleStatus;
    }

    TEST(NvTraceService, Session_CreateDestroy)
    {
#if 0
    printf("Waiting for keypress ...\n");
    getchar();
#endif

        ////////////////////////////////////////////////////////////////////////////
        // Create an nvidia-nsight-perf service
        OutOfProcessService service;
        if (g_TestOptions.runNvTrace)
        {
            ASSERT_TRUE(service.Initialize());
        }
        auto exitGuard = ScopeExitGuard([&]() {
            if (g_TestOptions.runNvTrace)
            {
                service.Reset();
            }
        });

        // Initialize the NvtxTrace provider.
        nvtxInitialize(nullptr);

        ////////////////////////////////////////////////////////////////////////////
        // Create CUDA resources and run the main loop
        int status = 0;
        // Run a warmup kernel, which causes CUDA resources to be lazily created.
        printf("Launching a warmup kernel: ");
        {
            ProfileSample(VectorAdd, "TestNvTraceCudaContinuousProfiling");
        }
        printf("\n");

        ////////////////////////////////////////////////////////////////////////////
        // Create an Observer Client; creating this after the nvidia-nsight-perf service has been initialized ensures no dropped events.
        StreamingConsumerType streamingConsumer;
        TsfSnapshotConsumerType tsfSnapshotConsumer;
        TsfSnapshotCompletionState snapshotCompletionState;
        std::filesystem::path tsfDatabaseDir;
        std::filesystem::path tsfSnapshotDatabaseDir;
        auto removeTsfDatabaseDirGuard = ScopeExitGuard([&]() {
            if (tsfDatabaseDir.empty())
            {
                return;
            }
            std::error_code errorCode;
            std::filesystem::remove_all(tsfDatabaseDir, errorCode);
        });
        uint64_t observationStartUtcNsec = Codec::InvalidTimestamp;
        uint64_t observationEndUtcNsec = Codec::InvalidTimestamp;
        if (g_TestOptions.runNvTrace)
        {
            tsfDatabaseDir = CreateTemporaryTsfDatabaseDirectory();
            ASSERT_FALSE(tsfDatabaseDir.empty()) << "Failed to create a temporary TSF database directory";

            if (g_TestOptions.useSnapshotConsumer)
            {
                printf("Initializing TsfSnapshotConsumer ...\n");
                const std::filesystem::path primaryDatabaseDir = tsfDatabaseDir / "primary";
                tsfSnapshotDatabaseDir = tsfDatabaseDir / "snapshot";

                typename TsfSnapshotConsumerType::Options consumerOptions;
                consumerOptions.fnOnSnapshotCompleted = OnTsfSnapshotCompleted;
                consumerOptions.pUserDataOnSnapshotCompleted = &snapshotCompletionState;
                ASSERT_TRUE(tsfSnapshotConsumer.Initialize(consumerOptions, [&](TsfDiskDatabaseBufferedStorageWriter& storageWriter) {
                    printf("Creating TSF snapshot primary database at: %s\n", primaryDatabaseDir.string().c_str());
                    return storageWriter.Initialize(primaryDatabaseDir);
                }));
            }
            else
            {
                printf("Initializing StreamingConsumer ...\n");
                typename StreamingConsumerType::Options consumerOptions;
                ASSERT_TRUE(streamingConsumer.Initialize(consumerOptions, [&](TsfDiskDatabaseBufferedStorageWriter& storageWriter) {
                    printf("Creating TSF database at: %s\n", tsfDatabaseDir.string().c_str());
                    return storageWriter.Initialize(tsfDatabaseDir);
                }));
                nvtxRangePush("StreamingConsumer_Start_Stop");
                printf("Starting StreamingConsumer ...\n");
                observationStartUtcNsec = streamingConsumer.ConvertTscTickToUtcNsec(ReadCpuTscTimestamp());
                ASSERT_TRUE(streamingConsumer.Start());
            }
        }

        if (g_TestOptions.selectedSamples.count(SampleId::VectorAdd))
        {
            // Enter the steady state.
            printf("Entering the steady state (VectorAdd): ");
            for (uint32_t iteration = 0; iteration < g_TestOptions.numIterations; ++iteration)
            {
                status |= ProfileSample(VectorAdd, "VectorAdd");
            }
            printf("\n");
        }
        if (g_TestOptions.selectedSamples.count(SampleId::CudaGraphs))
        {
            printf("Entering the steady state (CUDA Graphs): ");
            for (uint32_t iteration = 0; iteration < g_TestOptions.numIterations; ++iteration)
            {
                status |= ProfileSample(simpleCudaGraphs, "simpleCudaGraphs");
            }
            printf("\n");
        }
        if (g_TestOptions.selectedSamples.count(SampleId::MultiGPU))
        {
            printf("Entering the steady state (MultiGPU): ");
            for (uint32_t iteration = 0; iteration < g_TestOptions.numIterations; ++iteration)
            {
                status |= ProfileSample(simpleMultiGPU, "simpleMultiGPU");
            }
            printf("\n");
        }
        if (g_TestOptions.selectedSamples.count(SampleId::Coupled))
        {
            printf("Entering the steady state (Coupled): ");
            for (uint32_t iteration = 0; iteration < g_TestOptions.numIterations; ++iteration)
            {
                status |= ProfileSample(hmmaCoupledFlops, "hmmaCoupledFlops");
            }
            printf("\n");
        }
        if (g_TestOptions.selectedSamples.count(SampleId::TensorCore))
        {
            printf("Entering the steady state (Tensor Cores): ");
            for (uint32_t iteration = 0; iteration < g_TestOptions.numIterations; ++iteration)
            {
                status |= ProfileSample(immaTensorCoreGemm, "immaTensorCoreGemm");
            }
            printf("\n");
        }
        if (g_TestOptions.selectedSamples.count(SampleId::ImmediateMemset))
        {
            printf("Entering the steady state (Immediate Memset): ");
            for (uint32_t iteration = 0; iteration < g_TestOptions.numIterations; ++iteration)
            {
                status |= ProfileSample(ImmediateMemset, "ImmediateMemset");
            }
            printf("\n");
        }
        EXPECT_EQ(0, status);
        if (g_TestOptions.selectedSamples.count(SampleId::HostCallbacks))
        {
            printf("Entering the steady state (Host Callbacks): ");
            for (uint32_t iteration = 0; iteration < g_TestOptions.numIterations; ++iteration)
            {
                status |= ProfileSample(HostCallbacks, "HostCallbacks");
            }
            printf("\n");
        }
        ////////////////////////////////////////////////////////////////////////////
        // Stop the observer and export results
        if (g_TestOptions.runNvTrace)
        {
            if (g_TestOptions.useSnapshotConsumer)
            {
                printf("Triggering TsfSnapshotConsumer snapshot ...\n");
                typename TsfSnapshotConsumerType::SnapshotParam snapshotParam;
                snapshotParam.historyDurationNsec = 0;
                snapshotParam.futureDurationNsec = 0;
                ASSERT_TRUE(tsfSnapshotConsumer.Snapshot(
                    [&](TsfDiskDatabaseBufferedStorageWriter& snapshotStorageWriter) {
                        printf("Creating TSF snapshot database at: %s\n", tsfSnapshotDatabaseDir.string().c_str());
                        return snapshotStorageWriter.Initialize(tsfSnapshotDatabaseDir);
                    },
                    snapshotParam));

                {
                    std::unique_lock<std::mutex> lock(snapshotCompletionState.mutex);
                    ASSERT_TRUE(snapshotCompletionState.cv.wait_for(lock, std::chrono::seconds(60), [&]() {
                        return snapshotCompletionState.completed;
                    })) << "Timed out waiting for the TSF snapshot to complete";
                    ASSERT_TRUE(snapshotCompletionState.succeeded) << "TSF snapshot failed";
                }

                tsfSnapshotConsumer.Reset();
                ExportTsfDatabaseAndValidate(tsfSnapshotDatabaseDir, Codec::InvalidTimestamp, Codec::InvalidTimestamp);
            }
            else
            {
                nvtxRangePop(); // "StreamingConsumer_Start_Stop"
                observationEndUtcNsec = streamingConsumer.ConvertTscTickToUtcNsec(ReadCpuTscTimestamp());
                ASSERT_TRUE(streamingConsumer.StopAtUtc(observationEndUtcNsec));
                streamingConsumer.Reset();
                ExportTsfDatabaseAndValidate(tsfDatabaseDir, observationStartUtcNsec, observationEndUtcNsec);
            }
        }
    }
} // namespace nv::trace
