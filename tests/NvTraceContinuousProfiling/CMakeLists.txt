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
set(CudaSamplesSourceDir
    "${CMAKE_CURRENT_SOURCE_DIR}/../../cuda-samples"
)

add_library(CudaSamples OBJECT
    src/hmmaCoupledFlops.cu
    src/ImmediateMemset.cu
    src/HostCallbacks.cpp
    ${CudaSamplesSourceDir}/src/immaTensorCoreGemm.cu
    ${CudaSamplesSourceDir}/src/simpleCudaGraphs.cu
    ${CudaSamplesSourceDir}/src/simpleMultiGPU.cu
    ${CudaSamplesSourceDir}/src/VectorAdd.cu
)

target_include_directories(CudaSamples
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CudaSamplesSourceDir}/include
)
# nvcc keeps its bundled CUDA Toolkit nvtx3 ahead of target_include_directories, so pass nvtx3's include on the CUDA command line instead.
target_compile_options(CudaSamples BEFORE PRIVATE
    "$<$<COMPILE_LANGUAGE:CUDA>:-I${nvtx3IncludeDirectories}>"
)

target_link_libraries(CudaSamples
    PRIVATE
        NvTraceTestUtilities
        # Keep nvtx3 ahead of the CUDA Toolkit's NVTX. Include precedence follows link order, and
        # this target's NVTX events may not be traced if the CUDA Toolkit's NVTX wins.
        nvtx3
        CUDA::cuda_driver
        CUDA::toolkit
        googletest
        nvtrace_client
)

add_executable(TestNvTraceCudaContinuousProfiling
    src/NvTraceServiceTests.cpp
    src/CudaContinuousProfiling.cpp
)

target_include_directories(TestNvTraceCudaContinuousProfiling
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_include_directories(TestNvTraceCudaContinuousProfiling BEFORE PRIVATE
    ${nvtx3IncludeDirectories}
)

target_link_libraries(TestNvTraceCudaContinuousProfiling
    PRIVATE
        NvTraceTestUtilities
        # Keep nvtx3 ahead of the CUDA Toolkit's NVTX. Include precedence follows link order, and
        # this target's NVTX events may not be traced if the CUDA Toolkit's NVTX wins.
        nvtx3
        CUDA::cuda_driver
        CUDA::toolkit
        CudaSamples
        cxxopts
        googletest
        nvtrace_client_offline
)
nvtrace_link_cuda_runtime(TestNvTraceCudaContinuousProfiling)
