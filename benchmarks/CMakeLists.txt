# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# NVTX Trace needs nvtx3's CUDA driver injection discovery.
get_target_property(nvtx3IncludeDirectories nvtx3 INTERFACE_INCLUDE_DIRECTORIES)
if(NOT nvtx3IncludeDirectories)
    get_target_property(nvtx3LinkLibraries nvtx3 INTERFACE_LINK_LIBRARIES)
    foreach(nvtx3LinkLibrary IN LISTS nvtx3LinkLibraries)
        if(TARGET ${nvtx3LinkLibrary})
            get_target_property(nvtx3LinkLibraryIncludeDirectories ${nvtx3LinkLibrary} INTERFACE_INCLUDE_DIRECTORIES)
            if(nvtx3LinkLibraryIncludeDirectories)
                set(nvtx3IncludeDirectories ${nvtx3LinkLibraryIncludeDirectories})
                break()
            endif()
        endif()
    endforeach()
endif()
if(NOT nvtx3IncludeDirectories)
    message(FATAL_ERROR "Could not resolve the nvtx3 include directories from the nvtx3 target")
endif()

add_executable(BenchmarkNvTraceCuda
    src/EndToEndBenchmarks.cpp
    src/VectorAddLauncher.cu
)

target_include_directories(BenchmarkNvTraceCuda
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(BenchmarkNvTraceCuda
    PRIVATE
        CUDA::cuda_driver
        CUDA::toolkit
        cxxopts
        googlebenchmark
        nvtrace_client
)
nvtrace_link_cuda_runtime(BenchmarkNvTraceCuda)

add_executable(BenchmarkNvTraceNvtx
    src/NvtxBenchmarks.cpp
)

target_include_directories(BenchmarkNvTraceNvtx
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_include_directories(BenchmarkNvTraceNvtx BEFORE PRIVATE
    ${nvtx3IncludeDirectories}
)

target_link_libraries(BenchmarkNvTraceNvtx
    PRIVATE
        # Keep nvtx3 ahead of the CUDA Toolkit's NVTX. Include precedence follows link order, and
        # this target's NVTX events may not be traced if the CUDA Toolkit's NVTX wins.
        nvtx3
        CUDA::cuda_driver
        CUDA::toolkit
        cxxopts
        googlebenchmark
        nvtrace_client
)

add_executable(BenchmarkNvTraceDemangler
    src/CudaNameDemanglerBenchmarks.cpp
)

target_link_libraries(BenchmarkNvTraceDemangler
    PRIVATE
        googlebenchmark
        nvtrace_client_offline
)

add_executable(BenchmarkNvTraceZipStreamingCompressor
    src/ZipStreamingCompressorBenchmarks.cpp
)

target_link_libraries(BenchmarkNvTraceZipStreamingCompressor
    PRIVATE
        googlebenchmark
        nvtrace_client_offline
)

add_executable(BenchmarkNvTraceServiceInitialization
    src/ServiceInitializationBenchmarks.cpp
)

target_include_directories(BenchmarkNvTraceServiceInitialization
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(BenchmarkNvTraceServiceInitialization
    PRIVATE
        CUDA::cuda_driver
        CUDA::toolkit
        googlebenchmark
        nvtrace_client
)

add_executable(BenchmarkNvTraceMemory
    src/MemoryBenchmarks.cpp
    src/MemsetMemcpyLauncher.cpp
)

target_include_directories(BenchmarkNvTraceMemory
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(BenchmarkNvTraceMemory
    PRIVATE
        CUDA::cuda_driver
        CUDA::toolkit
        cxxopts
        googlebenchmark
        nvtrace_client
)
nvtrace_link_cuda_runtime(BenchmarkNvTraceMemory)

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/scripts/BenchmarkCommon.sh DESTINATION "${CMAKE_BINARY_DIR}/bin")
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/scripts/RunCudaBenchmarks.sh DESTINATION "${CMAKE_BINARY_DIR}/bin")
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/scripts/RunNvtxBenchmarks.sh DESTINATION "${CMAKE_BINARY_DIR}/bin")
