#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


# Exit immediately if a command exits with a non-zero status.
# This ensures the script stops if any command fails (e.g., sudo password incorrect).
set -e

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/BenchmarkCommon.sh"

# Check if the argument is valid
if [[ "$1" != "base" && "$1" != "nvtrace" && "$1" != "nsys" && "$1" != "nvTraceServiceInit" ]]; then
    echo "Usage: $0 <base|nvtrace|nsys|nvTraceServiceInit>"
    echo "  base             Run benchmarks without NvTrace"
    echo "  nvtrace              Run benchmarks with NvTrace (service + target hooks; no consumer)"
    echo "  nsys             Run benchmarks with nsys"
    echo "  nvTraceServiceInit   Run out-of-process nvidia-nsight-perf service start/stop benchmark"
    exit 1
fi

# NvTrace CUDA API trace Beta benchmark requires NVIDIA driver >= 615.06
MIN_IN_DRIVER_NVTRACE_CUDA_API_TRACE_DRIVER_VERSION="615.06"
if [[ "$1" == "nvtrace" ]]; then
    DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n 1 || true)
    if [[ -z "${DRIVER_VERSION}" ]]; then
        echo "[ERROR] Could not detect NVIDIA driver version. Ensure a compatible driver is installed."
        exit 1
    fi
    if [[ "$(printf '%s\n' "${MIN_IN_DRIVER_NVTRACE_CUDA_API_TRACE_DRIVER_VERSION}" "${DRIVER_VERSION}" | sort -V | tail -n 1)" != "${DRIVER_VERSION}" ]]; then
        echo "[WARNING] Detected driver version (${DRIVER_VERSION}) is older than CUDA 13.4 (${MIN_IN_DRIVER_NVTRACE_CUDA_API_TRACE_DRIVER_VERSION}), please make sure that you have CUDA 13.4 compat package"
        read -r -p "Press Enter to continue..." _ || true
    fi
fi

pin_cpu_frequency "${BENCHMARK_CPU_FREQUENCY_KHZ}"

BM_ARGS=(
    "--benchmark_min_time=0.5"
    "${BENCHMARK_COMMON_ARGS[@]}"
    "--benchmark_repetitions=5"
    "--benchmark_out=bm.json"
    "--benchmark_out_format=json"
)

SERVICE_INIT_BM_ARGS=(
    "--benchmark_min_time=0.5"
    "${BENCHMARK_COMMON_ARGS[@]}"
    "--benchmark_repetitions=1"
    "--benchmark_out=bm_service_init.json"
    "--benchmark_out_format=json"
)

case $1 in
    "base")
        echo "[INFO] Running benchmarks without NvTrace"
        run_benchmark ./BenchmarkNvTraceCuda "${BM_ARGS[@]}" --nvtrace=false
        ;;
    "nvtrace")
        echo "[INFO] Running benchmarks with NvTrace"
        run_benchmark ./BenchmarkNvTraceCuda "${BM_ARGS[@]}" --nvtrace=true
        ;;
    "nsys")
        echo "[INFO] Running benchmarks with nsys..."
        run_benchmark nsys profile --trace=cuda --sample=none --cpuctxsw=none \
            ./BenchmarkNvTraceCuda "${BM_ARGS[@]}" --nvtrace=false
        ;;
    "nvTraceServiceInit")
        echo "[INFO] Running out-of-process nvidia-nsight-perf service initialization benchmark"
        run_benchmark ./BenchmarkNvTraceServiceInitialization "${SERVICE_INIT_BM_ARGS[@]}"
        ;;
esac
