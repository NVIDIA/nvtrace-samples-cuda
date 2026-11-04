/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// Workload that exercises both eager cuLaunchHostFunc / cudaLaunchHostFunc and
// CUDA graph host nodes.
int HostCallbacks(void);
