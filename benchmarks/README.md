# NvTrace CUDA Benchmarks

## Introduction

The benchmarks evaluate the performance impact of NvTrace profiling, focusing on both system-level and end-to-end overhead. Included artifacts are:

* **BenchmarkNvTraceCuda**: End-to-end CUDA benchmarks that test various kernel launch patterns and CUDA API calls
* **BenchmarkNvTraceMemory**: Memory operation benchmarks that measure `cudaMemset` and `cudaMemcpy` (H2D, D2H, D2D, H2H) throughput across multiple buffer sizes
* **BenchmarkNvTraceServiceInitialization**: Measures NVIDIA Nsight Perf service start/stop latency (forks a new service each iteration)
* **RunCudaBenchmarks.sh**: Automated script that ensures consistent benchmarking conditions (e.g., environment setup, warm-up runs, repeatability)

## Usage

### Building

The benchmarks are built as part of the NVIDIA Nsight Perf SDK samples.

### Running Benchmarks

To achieve consistent and reliable results, use the provided script which sets the CPU to a fixed frequency:

```bash
# Run end-to-end CUDA benchmarks without NvTrace (baseline)
sudo ./RunCudaBenchmarks.sh base

# Run end-to-end CUDA benchmarks with NvTrace enabled (service + target hooks)
sudo ./RunCudaBenchmarks.sh nvtrace

# Run end-to-end CUDA benchmarks under Nsight Systems
sudo ./RunCudaBenchmarks.sh nsys

# Run service initialization benchmark (noisy; forks each iteration)
sudo ./RunCudaBenchmarks.sh nvTraceServiceInit
```

**Note**: sudo privileges are required for the script to set the CPU clock to a fixed frequency

## Benchmark Categories

### CUDA Benchmarks

* **CPU-limited scenarios**: When the GPU kernel is small and CPU cycles are fully occupied, profiling overhead becomes measurable. This is often observed in applications that launch many tiny kernels back-to-back.
* **GPU-limited scenarios**: When the GPU kernel is large and the CPU is mostly waiting for GPU execution, profiling overhead is typically lower.

| Benchmark Name | Description | Workload Type | Purpose |
|---|---|---|---|
| **EndToEnd** | Complete vector addition workflow (memory allocation, kernel launch) | Vector Operations | I/O Bandwidth limited scenarios |
| **KernelAdd** | Vector addition kernel execution only | Vector Operations | CPU limited scenarios |
| **KernelLargeWorkload** | Complex vector operations with tanh function | Vector Operations | GPU limited scenarios |
| **SyncKernel_Empty** | Synchronous empty kernel launches | Immediate Kernel Launch | CPU limited scenarios |
| **SyncKernel_WaitforNsec** | Synchronous kernels with specified wait times (1μs, 10μs, 100μs) | Immediate Kernel Launch | from CPU-limited to GPU-limited |
| **AsyncDenseKernel_Empty** | Asynchronous empty kernel launches | Immediate Kernel Launch | CPU limited scenarios |
| **AsyncDenseKernel_WaitforNsec** | Asynchronous kernels with specified wait times (1μs, 10μs, 100μs) | Immediate Kernel Launch | from CPU-limited to GPU-limited |
| **AsyncSparseKernel_Empty** | Sparse asynchronous kernel launches | Immediate Kernel Launch | CPU limited scenarios |
| **AsyncSparseKernel_WaitforNsec** | Sparse asynchronous kernels with specified wait times (1μs, 10μs, 100μs) | Immediate Kernel Launch | from CPU-limited to GPU-limited |
| **AsyncGraphKernel_Empty** | CUDA Graph-based kernel launches | Graph Kernel Launch | |
| **AsyncGraphKernel_WaitforNsec** | CUDA Graph kernels with specified wait times (1μs, 10μs, 100μs) | Graph Kernel Launch | |
| **cudaDriverGetVersion** | CUDA Runtime API version query | Runtime API | Isolates runtime API trace overhead for a very fast function |
| **cuDriverGetVersion** | CUDA Driver API version query | Driver API | Isolates driver API trace overhead for a very fast function |
| **cuGetDeviceCount** | Device enumeration | Driver API | Isolates driver API trace overhead for a very fast function |
| **cudaStreamTest** | CUDA stream operations | Runtime API | Isolates runtime API trace overhead for a very fast function |
| **SyntheticTest_Async1us1000TimesThenSync** | 1000 asynchronous 1μs kernels followed by synchronization | Synthetic | |

### Memory Benchmarks

`BenchmarkNvTraceMemory` measures raw memory-operation throughput for `cudaMemset` and `cudaMemcpy` across four buffer sizes (1KB, 64KB, 1MB, 16MB). Run with `--nvtrace=false` for baseline bandwidth or `--nvtrace=true` to measure NvTrace overhead on memory operations.

**Direct API benchmarks** call `cudaMemset` / `cudaMemcpy` (or `cuMemcpy` for H2H) synchronously on each iteration.

**Graph benchmarks** record a single memset or memcpy node via CUDA stream capture (`cudaStreamBeginCapture` / `cudaStreamEndCapture`), instantiate the graph once, then time repeated `cudaGraphLaunch` calls.

**Converted graph benchmarks** build explicit `CU_GRAPH_NODE_TYPE_MEMSET` / `CU_GRAPH_NODE_TYPE_MEMCPY` nodes inside a conditional graph body (matching `CudaApiTraceInDriverGraphNodeTest`). During `cuGraphInstantiate`, the driver lowers these memory nodes to internal kernel nodes.

| Benchmark Name | Description | Transfer Direction |
|---|---|---|
| **Memset_Device** | `cudaMemset` on a device buffer | N/A |
| **Memcpy_H2D** | Host-to-device copy | Host → Device |
| **Memcpy_D2H** | Device-to-host copy | Device → Host |
| **Memcpy_D2D** | Device-to-device copy | Device → Device |
| **Memcpy_H2H** | Host-to-host copy via CUDA driver API (`cuMemcpy`) | Host → Host |
| **MemsetGraph_Device** | `cudaMemsetAsync` captured into a graph, launched via `cudaGraphLaunch` | N/A |
| **MemcpyGraph_H2D** | H2D `cudaMemcpyAsync` graph node | Host → Device |
| **MemcpyGraph_D2H** | D2H `cudaMemcpyAsync` graph node | Device → Host |
| **MemcpyGraph_D2D** | D2D `cudaMemcpyAsync` graph node | Device → Device |
| **MemcpyGraph_H2H** | H2H `cuMemcpyAsync` graph node (pinned buffers) | Host → Host |
| **MemsetGraphConverted_Device** | Memset node lowered to kernel during graph instantiation | N/A |
| **MemcpyGraphConverted_H2D** | H2D memcpy node lowered to kernel during graph instantiation | Host → Device |
| **MemcpyGraphConverted_D2H** | D2H memcpy node lowered to kernel during graph instantiation | Device → Host |
| **MemcpyGraphConverted_D2D** | D2D memcpy node lowered to kernel during graph instantiation | Device → Device |

```bash
# Run memory benchmarks without NvTrace (baseline bandwidth)
./BenchmarkNvTraceMemory --nvtrace=false

# Run memory benchmarks with NvTrace enabled
./BenchmarkNvTraceMemory --nvtrace=true
```

### Service initialization benchmark

| Benchmark Name | Binary | Description |
|---|---|---|
| **BM_NvTraceServiceInitialization** | `BenchmarkNvTraceServiceInitialization` | Service creation and destruction time per iteration |

