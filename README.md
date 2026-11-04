# NvTrace CUDA Samples

This directory contains CUDA examples, tests, and benchmarks for the NVIDIA Nsight Perf SDK:

- [`benchmarks`](benchmarks/README.md) measures profiling overhead.
- [`tests/DataModelValidator`](tests/DataModelValidator/README.md) validates captured trace data.
- [`tests/NvTraceContinuousProfiling`](tests/NvTraceContinuousProfiling/README.md) demonstrates continuous profiling of CUDA workloads.
- [`cuda-samples`](cuda-samples/README.md) contains selected NVIDIA CUDA Samples sources used by the examples.

## Compatibility

The sample source and test behavior themselves have the lowest stability level and may change between SDK releases. They are not compatibility interfaces.

## Contributions

Contributions require signed-off commits and the Developer Certificate of Origin. See [CONTRIBUTING.md](CONTRIBUTING.md) before submitting a change.

## Security

Do not report potential security vulnerabilities through public GitHub issues. Follow the private reporting process in [SECURITY.md](SECURITY.md).

## License

NVIDIA-authored SDK source is licensed under Apache-2.0; see [LICENSE](LICENSE). Imported/adapted files under `cuda-samples/` retain their BSD-style terms in [cuda-samples/LICENSE](cuda-samples/LICENSE). See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for dependency licenses and attribution.
