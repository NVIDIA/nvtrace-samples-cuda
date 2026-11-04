# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import time
import select
import signal
import subprocess
import sys

# Start nvidia-nsight-perf and verify it starts correctly
def start_service_host(service_host_command):
    try:
        # Start nvidia-nsight-perf
        service_host_process = subprocess.Popen(service_host_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        expected_messages = ["Initializing", "Creating", "Service created"]
        for expected_message in expected_messages:
            ready, _, _ = select.select([service_host_process.stdout, service_host_process.stderr], [], [], 5)
            if ready:
                line = service_host_process.stdout.readline()
                if not line:
                    line = service_host_process.stderr.readline()
                    if not line:
                        print("nvidia-nsight-perf terminated unexpectedly")
                    else:
                        print(line.strip(), flush=True)
                    sys.exit(1)
                print(line.strip(), flush=True)
                if expected_message not in line:
                    sys.exit(1)
            else:
                print(f"Timeout waiting for nvidia-nsight-perf to initialize")
                sys.exit(1)
        return service_host_process
    except Exception as e:
        print(f"Error starting nvidia-nsight-perf: {e}")
        sys.exit(1)

# Start NvTrace streaming consumer (observer)
def start_nvtrace_streaming_consumer(nvtrace_streaming_consumer_command):
    try:
        # Start the NvTrace streaming consumer
        nvtrace_streaming_consumer_process = subprocess.Popen(nvtrace_streaming_consumer_command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        expected_messages = ["Initializing", "Creating database", "Starting", "Press enter"]
        for expected_message in expected_messages:
            ready, _, _ = select.select([nvtrace_streaming_consumer_process.stdout, nvtrace_streaming_consumer_process.stderr], [], [], 5)
            if ready:
                line = nvtrace_streaming_consumer_process.stdout.readline()
                if not line:
                    line = nvtrace_streaming_consumer_process.stderr.readline()
                    if not line:
                        print("NvTrace streaming consumer terminated unexpectedly")
                    else:
                        print(line.strip(), flush=True)
                    return None
                print(line.strip(), flush=True)
                if expected_message not in line:
                    return None
            else:
                print(f"Timeout waiting for NvTrace streaming consumer to initialize")
                return None
        return nvtrace_streaming_consumer_process
    except Exception as e:
        print(f"Error starting NvTrace streaming consumer: {e}")
        return None
                        

# Run the workload to be profiled
def run_workload(workload_command):
    try:
        # Run the workload process and wait for it to complete
        subprocess.run(workload_command, check=True)
    except Exception as e:
        print(f"Error running workload: {e}")

def main():
    if len(sys.argv) < 2:
        print(f"Usage: python {os.path.basename(__file__)} <workload> [<arg1> <arg2> ...]")
        return

    # nvidia-nsight-perf should be in the same directory as this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    service_host_command = [os.path.join(script_dir, "nvidia-nsight-perf")]

    # Workload command from the remaining command line arguments
    workload_cmd = sys.argv[1:]

    # Start nvidia-nsight-perf
    service_host_process = start_service_host(service_host_command)
    if not service_host_process:
        return

    # Start the observer process
    nvtrace_streaming_consumer_command = [os.path.join(script_dir, "nvtrace_streaming_consumer"), "--stop-on-enter-key"]
    nvtrace_streaming_consumer_process = start_nvtrace_streaming_consumer(nvtrace_streaming_consumer_command)

    if nvtrace_streaming_consumer_process:
        # Run the workload process
        run_workload(workload_cmd)
        print("Workload completed; stopping NvTrace streaming consumer")
        # Send enter to the streaming consumer
        nvtrace_streaming_consumer_process.stdin.write('\n')
        nvtrace_streaming_consumer_process.stdin.flush()
        nvtrace_streaming_consumer_process.wait()

    # Send Ctrl-C to the nvidia-nsight-perf process
    service_host_process.send_signal(signal.SIGINT)
    service_host_process.wait()
    print("nvidia-nsight-perf terminated")

if __name__ == "__main__":
    main()
