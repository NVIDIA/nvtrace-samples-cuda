/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include <cstdint>
#include <map>
#include <string>
#include <unordered_set>
#include <vector>

#include <NvTrace/Offline/DataModel/NvTraceDataModel.h>

// Add an enum for supported sample types, and string names.
enum class SampleId : uint32_t
{
    VectorAdd = 0,
    CudaGraphs = 1,
    MultiGPU = 2,
    Coupled = 3,
    TensorCore = 4,
    ImmediateMemset = 5,
    HostCallbacks = 6
};

// Add string names for the SampleType enum
inline const std::map<SampleId, std::string> g_kSupportedSamples = {
    {      SampleId::VectorAdd,       "vectorAdd"},
    {     SampleId::CudaGraphs,      "cudaGraphs"},
    {       SampleId::MultiGPU,        "multiGPU"},
    {        SampleId::Coupled,         "coupled"},
    {     SampleId::TensorCore,      "tensorCore"},
    {SampleId::ImmediateMemset, "immediateMemset"},
    {  SampleId::HostCallbacks,   "hostCallbacks"}
};

struct GlobalTestOptions
{
    uint32_t numIterations = 5;
    uint32_t serviceInitializationTimeoutMsec = 0xFFFFFFFFu;
    std::unordered_set<SampleId> selectedSamples;
    uint32_t runNvTrace = 1;
    bool useSnapshotConsumer = false;
    std::string traceOutputPath = "nvtrace.perfetto-trace";
};

extern GlobalTestOptions g_TestOptions;
