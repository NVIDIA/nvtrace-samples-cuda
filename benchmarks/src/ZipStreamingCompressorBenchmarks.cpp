/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <NvTrace/Offline/Exporter/NvTracePerfettoTraceWriter.h>
#include <NvTrace/Offline/Exporter/NvTraceZipStreamingCompressor.h>

#include <benchmark/benchmark.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace nv {
namespace trace {
namespace {

    const std::vector<uint8_t>& GetGeneratedPerfettoTrace()
    {
        static const std::vector<uint8_t> traceBytes = [] {
            PerfettoTraceWriter perfettoTraceWriter;
            perfettoTraceWriter.SetTimestampOriginUtcNsec(0);
            const PerfettoTraceWriter::TrackUuid parentTrack = perfettoTraceWriter.CreateNamedTrack("NvTrace - benchmark-node");
            std::array<PerfettoTraceWriter::TrackUuid, 6> counterTracks{};
            for (size_t trackIndex = 0; trackIndex < counterTracks.size(); ++trackIndex)
            {
                const std::string trackName = "GPU Metric " + std::to_string(trackIndex);
                counterTracks[trackIndex] = perfettoTraceWriter.CreateCounterTrack(trackName, parentTrack, "count");
            }
            for (size_t sampleIndex = 0; sampleIndex < 10000; ++sampleIndex)
            {
                const uint64_t timestampNsec = static_cast<uint64_t>(sampleIndex) * 1000000;
                for (size_t trackIndex = 0; trackIndex < counterTracks.size(); ++trackIndex)
                {
                    const double value = static_cast<double>((sampleIndex * (trackIndex + 3)) % 1000);
                    perfettoTraceWriter.AddCounterSample(counterTracks[trackIndex], timestampNsec, value);
                }
            }

            std::vector<uint8_t> result;
            auto sink = [&](const uint8_t* pData, size_t size) {
                result.insert(result.end(), pData, pData + size);
                return true;
            };
            const bool writeSucceeded = perfettoTraceWriter.Write(sink);
            if (!writeSucceeded)
            {
                result.clear();
            }
            return result;
        }();
        return traceBytes;
    }

    static void BenchmarkZipStreamingCompressor(
        benchmark::State& state,
        ZipStreamingCompressor::DeflateStrategy deflateStrategy)
    {
        const std::vector<uint8_t>& traceBytes = GetGeneratedPerfettoTrace();
        std::vector<uint8_t> stagingBuffer(64 * 1024, uint8_t{0});
        const size_t inputChunkSize = static_cast<size_t>(state.range(0));

        if (traceBytes.empty())
        {
            state.SkipWithError("Failed to generate the Perfetto benchmark trace");
            return;
        }

        size_t archiveSize = 0;
        for (auto benchmarkIteration : state)
        {
            static_cast<void>(benchmarkIteration);
            ZipStreamingCompressor compressor;
            const bool initialized = compressor.Initialize(1, deflateStrategy);
            if (!initialized)
            {
                state.SkipWithError("Failed to initialize ZipStreamingCompressor");
                return;
            }

            archiveSize = 0;
            auto acquireBuffer = [&]() {
                ZipStreamingCompressor::OutputBuffer outputBuffer;
                outputBuffer.pData = stagingBuffer.data();
                outputBuffer.capacity = stagingBuffer.size();
                outputBuffer.slot = 0;
                return outputBuffer;
            };
            auto submitBuffer = [&](const ZipStreamingCompressor::OutputBuffer& outputBuffer, size_t size) {
                auto* outputData = outputBuffer.pData;
                benchmark::DoNotOptimize(outputData);
                benchmark::ClobberMemory();
                archiveSize += size;
                return true;
            };
            auto writeArchive = [&](auto& archive) {
                auto entry = archive.OpenEntry("benchmark.perfetto-trace");
                size_t offset = 0;
                while (offset < traceBytes.size())
                {
                    const size_t size = std::min(inputChunkSize, traceBytes.size() - offset);
                    const bool writeSucceeded = entry.Write(traceBytes.data() + offset, size);
                    if (!writeSucceeded)
                    {
                        return false;
                    }
                    offset += size;
                }
                return true;
            };
            const bool writeSucceeded = compressor.WriteArchive(acquireBuffer, submitBuffer, writeArchive);
            if (!writeSucceeded)
            {
                state.SkipWithError("Failed to generate the benchmark ZIP archive");
                return;
            }
            benchmark::DoNotOptimize(archiveSize);
        }

        state.SetBytesProcessed(state.iterations() * static_cast<int64_t>(traceBytes.size()));
        state.counters["compression_ratio"] = static_cast<double>(traceBytes.size()) / static_cast<double>(archiveSize);
        state.counters["input_bytes"] = static_cast<double>(traceBytes.size());
        state.counters["output_bytes"] = static_cast<double>(archiveSize);
    }

    static void BenchmarkZipStreamingCompressorFast(benchmark::State& state)
    {
        BenchmarkZipStreamingCompressor(state, ZipStreamingCompressor::DeflateStrategy::Fast);
    }

    static void BenchmarkZipStreamingCompressorCompact(benchmark::State& state)
    {
        BenchmarkZipStreamingCompressor(state, ZipStreamingCompressor::DeflateStrategy::Compact);
    }

    BENCHMARK(BenchmarkZipStreamingCompressorFast)
        ->ArgName("input_chunk_bytes")
        ->Arg(4 * 1024)
        ->Arg(64 * 1024)
        ->Arg(1024 * 1024);

    BENCHMARK(BenchmarkZipStreamingCompressorCompact)
        ->ArgName("input_chunk_bytes")
        ->Arg(4 * 1024)
        ->Arg(64 * 1024)
        ->Arg(1024 * 1024);

}
}
} // namespace nv::trace

BENCHMARK_MAIN();
