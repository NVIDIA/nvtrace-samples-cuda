/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "NvTraceGTestThrowListener.h"

#include <cstdint>
#include <gtest/gtest-spi.h>
#include <gtest/gtest.h>

#include "GlobalTestOptions.h"
#include "cxxopts.hpp"

#if defined(__linux__)
#include <cstdlib>
#include <unistd.h>

// Ensure XDG_RUNTIME_DIR is set so the out-of-process nvidia-nsight-perf service can bind its
// user-local socket. Headless CI environments may not have a logind session
// and therefore leave XDG_RUNTIME_DIR unset.
static void EnsureXdgRuntimeDirForTests()
{
    const char* xdg = std::getenv("XDG_RUNTIME_DIR");
    if (xdg != nullptr && xdg[0] != '\0')
    {
        return;
    }
    char tmpDir[] = "/tmp/nvtrace-test-XXXXXX";
    if (mkdtemp(tmpDir) != nullptr)
    {
        setenv("XDG_RUNTIME_DIR", tmpDir, 1);
    }
}
#endif

GlobalTestOptions g_TestOptions;

std::string GetSamplesDescription()
{
    std::string description = "0 - All samples\n";

    for (const auto& [sampleId, sampleName] : g_kSupportedSamples)
    {
        description += std::to_string(static_cast<uint32_t>(sampleId) + 1) + " - " + sampleName + "\n";
    }
    return description += "Example: --samples 1,3,5\n";
}

void ParseGlobalTestOptions(int argc, char** argv)
{
    // clang-format off
    cxxopts::Options options("Options");
    options.add_options()
    ("iterations", "Number of iterations", cxxopts::value<uint32_t>()->default_value(std::to_string(GlobalTestOptions().numIterations)))
    ("timeout-ms", "How long to wait for nvidia-nsight-perf service to initialize before timeout (in milliseconds)", cxxopts::value<uint32_t>()->default_value(std::to_string(GlobalTestOptions().serviceInitializationTimeoutMsec)))
    ("samples", GetSamplesDescription(), cxxopts::value<std::vector<uint32_t>>()->default_value("0"))
    ("nvtrace", "Activate/Deactivate NvTrace Profiling", cxxopts::value<bool>()->default_value("true"))
    ("snapshot", "Whether to use TsfSnapshotConsumer instead of StreamingConsumer to consume NvTrace data", cxxopts::value<bool>()->default_value("false"))
    ("trace-output-path", "Output path for the exported Perfetto protobuf trace file", cxxopts::value<std::string>()->default_value(GlobalTestOptions().traceOutputPath))
    ("h,help", "Print usage");
    // clang-format on 

    auto result = options.parse(argc, argv);

    if (result.count("help"))
    {
        std::cout << options.help() << std::endl;
        exit(0);
    }

    g_TestOptions.numIterations = result["iterations"].as<uint32_t>();
    g_TestOptions.serviceInitializationTimeoutMsec = result["timeout-ms"].as<uint32_t>();
    g_TestOptions.runNvTrace = result["nvtrace"].as<bool>();
    g_TestOptions.useSnapshotConsumer = result["snapshot"].as<bool>();
    g_TestOptions.traceOutputPath = result["trace-output-path"].as<std::string>();
    for (uint32_t sampleId : result["samples"].as<std::vector<uint32_t>>())
    {
        if (sampleId == 0) {
            // If sampleId is 0, select all samples
            for (const auto& [id, _] : g_kSupportedSamples) {
                g_TestOptions.selectedSamples.insert(static_cast<SampleId>(id));
            }
            break; // No need to process further
        }
        if (g_kSupportedSamples.count(static_cast<SampleId>(sampleId - 1)) == 0) {
            std::cerr << "Invalid sample id: " << sampleId << ". Valid options are:\n" << GetSamplesDescription();
            exit(1);
        }
        g_TestOptions.selectedSamples.insert(static_cast<SampleId>(sampleId - 1));
    }
    
}

int main(int argc, char** argv)
{
#if defined(__linux__)
    EnsureXdgRuntimeDirForTests();
#endif

    // Call InitGooleTest before ParseGlobalTestOptions so that GTest can parse the
    //   --gtest_* options; place those options on the command line before our
    //   custom options; GTest will advance argc as it consumes options
    ::testing::InitGoogleTest(&argc, argv);
    ParseGlobalTestOptions(argc, argv);

    AddGTestThrowListener();

    return RUN_ALL_TESTS();
}
