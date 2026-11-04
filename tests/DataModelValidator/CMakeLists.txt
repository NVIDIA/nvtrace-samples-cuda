# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_executable(DataModelValidator
    src/DataModelValidator.cpp
)

target_link_libraries(DataModelValidator
    PRIVATE
        NvTraceTestUtilities
        cxxopts
        googletest
        nvtrace_client_offline
)
