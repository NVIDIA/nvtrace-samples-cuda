#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


# Exit immediately if a command exits with a non-zero status.
# This ensures the script stops if any command fails (e.g., sudo password incorrect).
set -e

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/BenchmarkCommon.sh"

# Check if the argument is valid
if [[ "$1" != "nvtrace" && "$1" != "nsys" ]]; then
    echo "Usage: $0 <nvtrace|nsys> [strlen]"
    echo "  nvtrace           Run benchmarks with NvTrace"
    echo "  nsys          Run benchmarks with nsys"
    echo "  [strlen]        Length of NVTX strings (default 64)"
    exit 1
fi

STRLEN_ARGS=()
if [[ -n "$2" ]]; then
    STRLEN_ARGS=("--strlen=$2")
fi

pin_cpu_frequency "${BENCHMARK_CPU_FREQUENCY_KHZ}"

BM_ARGS=(
    "--benchmark_min_time=0.1"
    "${BENCHMARK_COMMON_ARGS[@]}"
    "--benchmark_repetitions=5"
    "--benchmark_out=bm_nvtx.csv"
    "--benchmark_out_format=csv"
)

case $1 in
    "nvtrace")
        echo "[INFO] Running benchmarks with NvTrace"
        run_benchmark ./BenchmarkNvTraceNvtx "${BM_ARGS[@]}" --nvtrace=true "${STRLEN_ARGS[@]}"
        ;;
    "nsys")
        echo "[INFO] Running benchmarks with nsys..."
        run_benchmark nsys profile --trace=nvtx --sample=none --cpuctxsw=none \
            ./BenchmarkNvTraceNvtx "${BM_ARGS[@]}" --nvtrace=false "${STRLEN_ARGS[@]}"
        ;;
esac
