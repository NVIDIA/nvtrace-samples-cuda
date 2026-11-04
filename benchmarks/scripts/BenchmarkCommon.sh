# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Common configuration and CPU frequency control for NvPerf benchmark runners.

BENCHMARK_CPU_SET="${BENCHMARK_CPU_SET:-12-15}"
BENCHMARK_CPU_FREQUENCY_KHZ="3000000"
BENCHMARK_COMMON_ARGS=(
    "--benchmark_min_warmup_time=0.1"
    "--benchmark_counters_tabular=true"
    "--benchmark_time_unit=ns"
)

CPUFREQ_PATH="/sys/devices/system/cpu/cpu0/cpufreq"
CPU_FREQUENCY_RESTORED=true

run_benchmark() {
    taskset -c "${BENCHMARK_CPU_SET}" "$@"
}

select_closest_cpu_frequency() {
    local requested_freq_khz="$1"
    local available_frequencies_file="${CPUFREQ_PATH}/scaling_available_frequencies"
    local -a available_frequencies
    local closest_freq_khz
    local closest_difference
    local candidate_freq_khz
    local difference
    local min_freq_khz
    local max_freq_khz

    if [[ -r "${available_frequencies_file}" ]] &&
       read -r -a available_frequencies < "${available_frequencies_file}" &&
       (( ${#available_frequencies[@]} > 0 )); then
        closest_freq_khz="${available_frequencies[0]}"
        closest_difference=$(( closest_freq_khz - requested_freq_khz ))
        if (( closest_difference < 0 )); then
            closest_difference=$(( -closest_difference ))
        fi

        for candidate_freq_khz in "${available_frequencies[@]:1}"; do
            difference=$(( candidate_freq_khz - requested_freq_khz ))
            if (( difference < 0 )); then
                difference=$(( -difference ))
            fi
            if (( difference < closest_difference ||
                  (difference == closest_difference && candidate_freq_khz < closest_freq_khz) )); then
                closest_freq_khz="${candidate_freq_khz}"
                closest_difference="${difference}"
            fi
        done

        echo "${closest_freq_khz}"
        return
    fi

    min_freq_khz=$(<"${CPUFREQ_PATH}/cpuinfo_min_freq")
    max_freq_khz=$(<"${CPUFREQ_PATH}/cpuinfo_max_freq")
    if (( requested_freq_khz < min_freq_khz )); then
        echo "${min_freq_khz}"
    elif (( requested_freq_khz > max_freq_khz )); then
        echo "${max_freq_khz}"
    else
        echo "${requested_freq_khz}"
    fi
}

set_cpu_governor_if_available() {
    local requested_governor="$1"
    local available_governors_file="${CPUFREQ_PATH}/scaling_available_governors"
    local -a available_governors
    local available_governor

    if [[ -r "${available_governors_file}" ]] &&
       read -r -a available_governors < "${available_governors_file}"; then
        for available_governor in "${available_governors[@]}"; do
            if [[ "${available_governor}" == "${requested_governor}" ]]; then
                echo "[INFO] Setting CPU governor to ${requested_governor}"
                sudo cpupower frequency-set --governor "${requested_governor}" > /dev/null
                return
            fi
        done
    fi

    echo "[WARNING] ${requested_governor} governor is unavailable; keeping ${ORIGINAL_GOVERNOR}"
}

restore_cpu_frequency() {
    if [[ "${CPU_FREQUENCY_RESTORED}" == true ]]; then
        return
    fi
    CPU_FREQUENCY_RESTORED=true

    echo "[INFO] Reverting CPU frequency..."
    sudo cpupower frequency-set --governor "${ORIGINAL_GOVERNOR}" > /dev/null
    sudo cpupower frequency-set -d "${ORIGINAL_MIN_FREQ_KHZ}" -u "${ORIGINAL_MAX_FREQ_KHZ}" > /dev/null
    echo "[INFO] CPU frequency reverted"
}

pin_cpu_frequency() {
    local requested_freq_khz="$1"
    local target_freq_khz

    ORIGINAL_GOVERNOR=$(<"${CPUFREQ_PATH}/scaling_governor")
    ORIGINAL_MIN_FREQ_KHZ=$(<"${CPUFREQ_PATH}/scaling_min_freq")
    ORIGINAL_MAX_FREQ_KHZ=$(<"${CPUFREQ_PATH}/scaling_max_freq")
    CPU_FREQUENCY_RESTORED=false
    trap restore_cpu_frequency INT TERM EXIT

    echo "[INFO] Starting CPU Frequency Controlled Benchmark Script"
    echo "Original Governor:          ${ORIGINAL_GOVERNOR}"
    echo "Original Minimum Frequency: ${ORIGINAL_MIN_FREQ_KHZ} KHz"
    echo "Original Maximum Frequency: ${ORIGINAL_MAX_FREQ_KHZ} KHz"

    target_freq_khz=$(select_closest_cpu_frequency "${requested_freq_khz}")
    echo "[INFO] Fixing CPU frequency to ${target_freq_khz} KHz (closest to ${requested_freq_khz} KHz)"
    sudo cpupower frequency-set -d "${target_freq_khz}" -u "${target_freq_khz}" > /dev/null
    set_cpu_governor_if_available performance
}
