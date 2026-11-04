# NvTrace Data Model Validator

## Introduction

`DataModelValidator` collects a live trace from the NVIDIA Nsight Perf service, builds the offline data model, and checks trace completeness using GoogleTest. It can also export the captured data as a Perfetto trace.

## Usage

```shell
./DataModelValidator \
    --duration-ms 10000 \
    --processN 1 \
    --output validation.perfetto-trace
```

Run the CUDA workload in another terminal during the collection window. Choose a duration long enough to cover the workload. Open the optional exported trace in [Perfetto](https://ui.perfetto.dev).

The service must collect CUDA Driver API tracing, GPU kernel events, CPU/GPU clock calibration, and GPU performance counters for the corresponding checks. The counter-sample check assumes 1000 Hz sampling. See the [service configuration documentation](https://github.com/NVIDIA/nsight-perf-daemon/blob/main/README.md).

For a workload with a known number of immediate kernel launches, add `--launches <count>`. Enable CUDA Runtime API tracing in the service and add `--enable-cudart-trace` to also check Runtime API launch counts.

The test uses a temporary Trace Stream Database and removes it when the test finishes.

## Options

- `--duration-ms <ms>`: collection duration; default `2000`.
- `--processN <count>`: expected number of processes in the data model; default `1`.
- `--launches <count>`: expected immediate kernel launch count; default `0xFFFFFFFF`, which skips launch-count comparisons.
- `--enable-cudart-trace`: also check CUDA Runtime API launch counts when `--launches` is set; default `false`. This option does not enable tracing in the service.
- `--output <path>`: export a Perfetto trace with debug events; default is no export.

GoogleTest options are also supported. Place `--gtest_*` options before the validator options.

## Validation

The test checks CPU clock calibration and the expected process count. For each captured GPU partition, it checks GPU clock calibration and the number of GPU performance-counter samples. When `--launches` is set, it also compares API launch counts and resolved immediate kernel launches with the expected count. CUDA Graph launches are not included in these comparisons.

Count comparisons allow a tolerance: 5% plus 300 events for API calls, 10% plus 300 for resolved kernel launches, and 15% plus 300 for counter samples. The expected counter sample count equals the collection duration in milliseconds, assuming 1000 Hz sampling.

GoogleTest reports assertion failures, and the executable returns a nonzero status if validation fails. Check workload timing, the expected process and launch counts, and the service configuration when investigating a failure. Unrelated captured processes can cause the process-count check to fail.
