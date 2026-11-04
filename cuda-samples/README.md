# CUDA Samples

This directory contains selected source files imported or adapted from [NVIDIA CUDA Samples](https://github.com/NVIDIA/cuda-samples) for use in the NVTrace CUDA profiling tests.

## Compatibility

Compatibility with the complete upstream CUDA Samples repository is not guaranteed.

## License

The imported files retain their original NVIDIA copyright notices and license terms. See [LICENSE](LICENSE) and the notices in each file.

The Apache 2.0 license used elsewhere in this repository does not apply to these imported files.

## Contents

- `include/` contains the CUDA Samples helper headers.
- `src/` contains the vector addition, integer Tensor Core GEMM, CUDA graphs, and multi-GPU workloads.

Some files have been adapted for integration with the NVTrace tests, including changes to entry points, tracing, and test assertions. This directory contains a selected subset of CUDA Samples, rather than a complete import.
