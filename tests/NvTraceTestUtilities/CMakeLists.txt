# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(NvTraceTestUtilities INTERFACE)
target_include_directories(NvTraceTestUtilities INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
