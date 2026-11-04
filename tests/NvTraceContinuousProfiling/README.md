# NvTrace CUDA Continuous Profiling Test

## Introduction

This sample profiles seven CUDA workloads: vector addition, CUDA Graphs, multi-GPU execution, coupled BF16 HMMA, tensor-core matrix multiplication, immediate memset operations, and host callbacks. Run it as a smoke test after building the SDK.


## CUDA Toolkit Reference

The Coupled BF16 HMMA sample is original to the NVIDIA Nsight Perf SDK. The other illustrated samples can be found in the [CUDA Samples repository](https://github.com/NVIDIA/cuda-samples).

## How to Open the Trace Files

1. Build the SDK package.

2. Launch `./TestNvTraceCudaContinuousProfiling` application
```shell
./TestNvTraceCudaContinuousProfiling
```

3. Open the generated `nvtrace.perfetto-trace` in <https://ui.perfetto.dev>.

## How to Run the Sample

There are 7 samples from which you can **select** which ones you would like to run using the argument `--samples`. You can also select the **number of iterations** in each sample by using the argument `--iterations`.

| Number Selection | Samples      |
| -----------------| ------------ |
| 0 (default)      | All Samples  |
| 1                | VectorAdd    |
| 2                | CUDA Graphs  |
| 3                | MultiGPU     |
| 4                | Coupled BF16 HMMA |
| 5                | Tensor Cores |
| 6                | Immediate Memset |
| 7                | Host Callbacks |

**Examples**

Run only MultiGPU and Coupled BF16 HMMA:

```shell
./TestNvTraceCudaContinuousProfiling --samples 3,4
```

Run all seven workloads 20 times each:

```shell
./TestNvTraceCudaContinuousProfiling --iterations 20
```


Run MultiGPU and Coupled BF16 HMMA 20 times each:
```shell
./TestNvTraceCudaContinuousProfiling --samples 3,4 --iterations 20
```

## Options

- `--iterations <count>`: iterations per selected workload; default `5`.
- `--timeout-ms <ms>`: maximum wait for service initialization.
- `--samples <ids>`: comma-separated workload selections; default `0` for all.
- `--nvtrace <bool>`: enable or disable NvTrace profiling; default `true`.
- `--snapshot <bool>`: use `TsfSnapshotConsumer` instead of `StreamingConsumer`; default `false`.
- `--trace-output-path <path>`: exported Perfetto trace path; default `nvtrace.perfetto-trace`.
- `--help`: print usage.