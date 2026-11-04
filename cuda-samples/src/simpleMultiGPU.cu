/* Copyright (c) 2022-2026, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/*
 * This application demonstrates how to use the CUDA API to use multiple GPUs,
 * with an emphasis on simple illustration of the techniques (not on
 * performance).
 *
 * Note that in order to detect multiple GPUs in your system you have to disable
 * SLI in the nvidia control panel. Otherwise only one GPU is visible to the
 * application. On the other side, you can still extend your desktop to screens
 * attached to both GPUs.
 */

// System includes
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>

#include "NvTraceCudaGTestHelpers.h"

#ifndef MAX
#define MAX(a, b) (a > b ? a : b)
#endif

typedef struct
{
    // Host-side input data
    int    dataN;
    float *h_Data;

    // Partial sum for this GPU
    float *h_Sum;

    // Device buffers
    float *d_Data, *d_Sum;

    // Reduction copied back from GPU
    float *h_Sum_from_device;

    // Stream for asynchronous command execution
    cudaStream_t stream;

} TGPUplan;

////////////////////////////////////////////////////////////////////////////////
// Data configuration
////////////////////////////////////////////////////////////////////////////////
const int MAX_GPU_COUNT = 32;
const int DATA_N        = 1048576;

////////////////////////////////////////////////////////////////////////////////
// Simple reduction kernel.
// Refer to the 'reduction' CUDA Sample describing
// reduction optimization strategies
////////////////////////////////////////////////////////////////////////////////
__global__ static void reduceKernel(float *d_Result, float *d_Input, int N)
{
    const int tid     = blockIdx.x * blockDim.x + threadIdx.x;
    const int threadN = gridDim.x * blockDim.x;
    float     sum     = 0;

    for (int pos = tid; pos < N; pos += threadN)
        sum += d_Input[pos];

    d_Result[tid] = sum;
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int simpleMultiGPU()
{
    // Solver config
    TGPUplan plan[MAX_GPU_COUNT];

    // GPU reduction results
    float h_SumGPU[MAX_GPU_COUNT];

    float  sumGPU;
    double sumCPU, diff;

    int i, j, gpuBase, GPU_N;

    const int BLOCK_N  = 32;
    const int THREAD_N = 256;
    const int ACCUM_N  = BLOCK_N * THREAD_N;

    printf("Starting simpleMultiGPU\n");
    cudaError_t err = cudaGetDeviceCount(&GPU_N);
    ASSERT_CUDART(err);
    if (err != cudaSuccess)
    {
        return EXIT_FAILURE;
    }
    if (GPU_N == 0)
    {
        fprintf(stderr, "No CUDA-capable devices found.\n");
        return EXIT_FAILURE;
    }

    if (GPU_N > MAX_GPU_COUNT) {
        GPU_N = MAX_GPU_COUNT;
    }

    printf("CUDA-capable device count: %i\n", GPU_N);

    printf("Generating input data...\n\n");

    // Subdividing input data across GPUs
    // Get data sizes for each GPU
    for (i = 0; i < GPU_N; i++) {
        plan[i].dataN = DATA_N / GPU_N;
    }

    // Take into account "odd" data sizes
    for (i = 0; i < DATA_N % GPU_N; i++) {
        plan[i].dataN++;
    }

    // Assign data ranges to GPUs
    gpuBase = 0;

    for (i = 0; i < GPU_N; i++) {
        plan[i].h_Sum = h_SumGPU + i;
        gpuBase += plan[i].dataN;
    }

    // Create streams for issuing GPU command asynchronously and allocate memory
    // (GPU and System page-locked)
    for (i = 0; i < GPU_N; i++) {
        ASSERT_CUDART(cudaSetDevice(i));
        err = cudaStreamCreate(&plan[i].stream);
        ASSERT_CUDART(err);
        if (err != cudaSuccess)
        {
            return EXIT_FAILURE;
        }
        // Allocate memory
        err = cudaMalloc((void **)&plan[i].d_Data, plan[i].dataN * sizeof(float));
        ASSERT_CUDART(err);
        if (err != cudaSuccess)
        {
            return EXIT_FAILURE;
        }
        err = cudaMalloc((void **)&plan[i].d_Sum, ACCUM_N * sizeof(float));
        ASSERT_CUDART(err);
        if (err != cudaSuccess)
        {
            return EXIT_FAILURE;
        }
        err = cudaMallocHost((void **)&plan[i].h_Sum_from_device, ACCUM_N * sizeof(float));
        ASSERT_CUDART(err);
        if (err != cudaSuccess)
        {
            return EXIT_FAILURE;
        }
        err = cudaMallocHost((void **)&plan[i].h_Data, plan[i].dataN * sizeof(float));
        ASSERT_CUDART(err);
        if (err != cudaSuccess)
        {
            return EXIT_FAILURE;
        }

        for (j = 0; j < plan[i].dataN; j++) {
            plan[i].h_Data[j] = (float)rand() / (float)RAND_MAX;
        }
    }

    // Start timing and compute on GPU(s)
    printf("Computing with %d GPUs...\n", GPU_N);

    // Copy data to GPU, launch the kernel and copy data back. All asynchronously
    for (i = 0; i < GPU_N; i++) {
        // Set device
        ASSERT_CUDART(cudaSetDevice(i));

        // Copy input data from CPU
        ASSERT_CUDART(cudaMemcpyAsync(
            plan[i].d_Data, plan[i].h_Data, plan[i].dataN * sizeof(float), cudaMemcpyHostToDevice, plan[i].stream));

        // Perform GPU computations
        reduceKernel<<<BLOCK_N, THREAD_N, 0, plan[i].stream>>>(plan[i].d_Sum, plan[i].d_Data, plan[i].dataN);
        // getLastCudaError("reduceKernel() execution failed.\n");

        // Read back GPU results
        ASSERT_CUDART(cudaMemcpyAsync(
            plan[i].h_Sum_from_device, plan[i].d_Sum, ACCUM_N * sizeof(float), cudaMemcpyDeviceToHost, plan[i].stream));
    }

    // Process GPU results
    for (i = 0; i < GPU_N; i++) {
        float sum;

        // Set device
        ASSERT_CUDART(cudaSetDevice(i));

        // Wait for all operations to finish
        cudaStreamSynchronize(plan[i].stream);

        // Finalize GPU reduction for current subvector
        sum = 0;

        for (j = 0; j < ACCUM_N; j++) {
            sum += plan[i].h_Sum_from_device[j];
        }

        *(plan[i].h_Sum) = (float)sum;

        // Shut down this GPU
        ASSERT_CUDART(cudaFreeHost(plan[i].h_Sum_from_device));
        ASSERT_CUDART(cudaFree(plan[i].d_Sum));
        ASSERT_CUDART(cudaFree(plan[i].d_Data));
        ASSERT_CUDART(cudaStreamDestroy(plan[i].stream));
    }

    sumGPU = 0;

    for (i = 0; i < GPU_N; i++) {
        sumGPU += h_SumGPU[i];
    }

    // Compute on Host CPU
    printf("Computing with Host CPU...\n\n");

    sumCPU = 0;

    for (i = 0; i < GPU_N; i++) {
        for (j = 0; j < plan[i].dataN; j++) {
            sumCPU += plan[i].h_Data[j];
        }
    }

    // Compare GPU and CPU results
    printf("Comparing GPU and Host CPU results...\n");
    diff = fabs(sumCPU - sumGPU) / fabs(sumCPU);
    printf("  GPU sum: %f\n  CPU sum: %f\n", sumGPU, sumCPU);
    printf("  Relative difference: %E \n\n", diff);

    // Cleanup and shutdown
    for (i = 0; i < GPU_N; i++) {
        ASSERT_CUDART(cudaSetDevice(i));
        ASSERT_CUDART(cudaFreeHost(plan[i].h_Data));
    }

    EXPECT_GE(1e-5, diff);
    ASSERT_CUDART(cudaSetDevice(0));

    return 0;
}