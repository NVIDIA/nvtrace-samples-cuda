/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Minimal BF16 HMMA Tensor Core Implementation for coupled FLOPs

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <stdio.h>

#include <helper_cuda.h>
#include <helper_functions.h>


#define checkKernelErrors(expr)                                                               \
    do                                                                                        \
    {                                                                                         \
        expr;                                                                                 \
                                                                                              \
        cudaError_t __err = cudaGetLastError();                                               \
        if (__err != cudaSuccess)                                                             \
        {                                                                                     \
            printf("Line %d: '%s' failed: %s\n", __LINE__, #expr, cudaGetErrorString(__err)); \
            abort();                                                                          \
        }                                                                                     \
    } while (0)

using namespace nvcuda;

// tile sizes
#define M 16
#define N 16
#define K 16

// matrix dimensions
#define MATRIX_M 512
#define MATRIX_N 512
#define MATRIX_K 512

// BF16 HMMA kernel for coupled tensor operations
__global__ void bf16_hmma_coupled_mm(__nv_bfloat16* __restrict__ A, __nv_bfloat16* __restrict__ B, __nv_bfloat16* __restrict__ C, __nv_bfloat16* __restrict__ D, float* __restrict__ Result, int M_dim, int N_dim, int K_dim)
{

    // each block handles one 16x16 tile
    int tile_row = blockIdx.y * M;
    int tile_col = blockIdx.x * N;

    if (tile_row >= M_dim || tile_col >= N_dim)
    {
        return;
    }

    // fragment declarations for BF16 HMMA - A*B operation
    wmma::fragment<wmma::matrix_a, M, N, K, __nv_bfloat16, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, M, N, K, __nv_bfloat16, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, M, N, K, float> acc_frag_ab;

    // fragment declarations for C*D operation (coupled)
    wmma::fragment<wmma::matrix_a, M, N, K, __nv_bfloat16, wmma::row_major> c_frag;
    wmma::fragment<wmma::matrix_b, M, N, K, __nv_bfloat16, wmma::row_major> d_frag;
    wmma::fragment<wmma::accumulator, M, N, K, float> acc_frag_cd;

    // initialize accumulators to zero
    wmma::fill_fragment(acc_frag_ab, 0.0f);
    wmma::fill_fragment(acc_frag_cd, 0.0f);

    for (int k = 0; k < K_dim; k += K)
    {
        if (k + K <= K_dim)
        {\
            wmma::load_matrix_sync(a_frag, A + tile_row * K_dim + k, K_dim);
            wmma::load_matrix_sync(b_frag, B + k * N_dim + tile_col, N_dim);

            wmma::load_matrix_sync(c_frag, C + tile_row * K_dim + k, K_dim);
            wmma::load_matrix_sync(d_frag, D + k * N_dim + tile_col, N_dim);

            // coupled HMMA operations: A*B and C*D
            wmma::mma_sync(acc_frag_ab, a_frag, b_frag, acc_frag_ab);
            wmma::mma_sync(acc_frag_cd, c_frag, d_frag, acc_frag_cd);
        }
    }

    // couple results: combine A*B + C*D
    for (int i = 0; i < acc_frag_ab.num_elements; i++)
    {
        acc_frag_ab.x[i] += acc_frag_cd.x[i];
    }

    wmma::store_matrix_sync(Result + tile_row * N_dim + tile_col, acc_frag_ab, N_dim, wmma::mem_row_major);
}

int hmmaCoupledFlops(void)
{
    checkCudaErrors(cudaSetDevice(0));

    printf("Initializing...\n");

    printf("BF16 HMMA Matrix dimensions: M=%d, N=%d, K=%d\n", MATRIX_M, MATRIX_N, MATRIX_K);

    const size_t size_A = MATRIX_M * MATRIX_K * sizeof(__nv_bfloat16);
    const size_t size_B = MATRIX_K * MATRIX_N * sizeof(__nv_bfloat16);
    const size_t size_C = MATRIX_M * MATRIX_K * sizeof(__nv_bfloat16);
    const size_t size_D = MATRIX_K * MATRIX_N * sizeof(__nv_bfloat16);
    const size_t size_Result = MATRIX_M * MATRIX_N * sizeof(float);

    // allocate host memory
    __nv_bfloat16* h_A = (__nv_bfloat16*)malloc(size_A);
    __nv_bfloat16* h_B = (__nv_bfloat16*)malloc(size_B);
    __nv_bfloat16* h_C = (__nv_bfloat16*)malloc(size_C);
    __nv_bfloat16* h_D = (__nv_bfloat16*)malloc(size_D);

    // allocate space for strategic validation
    const int num_validation_points = 25;
    int validation_points[num_validation_points][2];
    float* h_gpu_results = (float*)malloc(num_validation_points * sizeof(float));
    float* h_cpu_results = (float*)malloc(num_validation_points * sizeof(float));

    if (!h_A || !h_B || !h_C || !h_D || !h_gpu_results || !h_cpu_results)
    {
        printf("Host memory allocation failed\n");
        if (h_A)
        {
            free(h_A);
        }
        if (h_B)
        {
            free(h_B);
        }
        if (h_C)
        {
            free(h_C);
        }
        if (h_D)
        {
            free(h_D);
        }
        if (h_gpu_results)
        {
            free(h_gpu_results);
        }
        if (h_cpu_results)
        {
            free(h_cpu_results);
        }
        return -1;
    }

    // generate validation points - 5x5 grid
    int point_idx = 0;
    for (int i = 0; i < 5 && point_idx < num_validation_points; i++)
    {
        for (int j = 0; j < 5 && point_idx < num_validation_points; j++)
        {
            int row = i * (MATRIX_M - 1) / 4; // 0, 127, 255, 383, 511
            int col = j * (MATRIX_N - 1) / 4; // 0, 127, 255, 383, 511
            validation_points[point_idx][0] = row;
            validation_points[point_idx][1] = col;
            point_idx++;
        }
    }

    // initialize with position-dependent values to ensure different results across matrix
    for (int row = 0; row < MATRIX_M; row++)
    {
        for (int col = 0; col < MATRIX_K; col++)
        {
            // A: varies with row (0.1 to 0.6 range)
            h_A[row * MATRIX_K + col] = __float2bfloat16(0.1f + 0.5f * row / MATRIX_M);
            // C: varies with both row and col (0.2 to 0.4 range)
            h_C[row * MATRIX_K + col] = __float2bfloat16(0.2f + 0.2f * (row + col) / (MATRIX_M + MATRIX_K));
        }
    }
    for (int row = 0; row < MATRIX_K; row++)
    {
        for (int col = 0; col < MATRIX_N; col++)
        {
            // B: varies with col (0.1 to 0.6 range)
            h_B[row * MATRIX_N + col] = __float2bfloat16(0.1f + 0.5f * col / MATRIX_N);
            // D: varies with both row and col (0.3 to 0.5 range)
            h_D[row * MATRIX_N + col] = __float2bfloat16(0.3f + 0.2f * (row + col) / (MATRIX_K + MATRIX_N));
        }
    }

    // allocate device memory
    __nv_bfloat16 *d_A, *d_B, *d_C, *d_D;
    float* d_Result;

    checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_A), size_A));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_B), size_B));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_C), size_C));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_D), size_D));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_Result), size_Result));

    // Copy data to device
    checkCudaErrors(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_C, h_C, size_C, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_D, h_D, size_D, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemset(d_Result, 0, size_Result));

    // setup kernel launch parameters - use warp size for WMMA
    dim3 block(32, 1); // 32 threads per block for WMMA operations
    dim3 grid((MATRIX_N + N - 1) / N, (MATRIX_M + M - 1) / M);

    printf("Computing using BF16 HMMA coupled kernel: A*B + C*D\n");

    // Performance measurement
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    checkCudaErrors(cudaEventRecord(start));

    const int num_iterations = 5;
    for (int i = 0; i < num_iterations; i++)
    {
        checkKernelErrors((bf16_hmma_coupled_mm<<<grid, block>>>(d_A, d_B, d_C, d_D, d_Result, MATRIX_M, MATRIX_N, MATRIX_K)));
    }

    checkCudaErrors(cudaEventRecord(stop));
    checkCudaErrors(cudaEventSynchronize(stop));

    float milliseconds = 0;
    checkCudaErrors(cudaEventElapsedTime(&milliseconds, start, stop));

    double total_ops = 4.0 * MATRIX_M * MATRIX_N * MATRIX_K * num_iterations;
    printf("Time: %f ms\n", milliseconds);
    printf("TOPS: %.2f\n", (total_ops / (milliseconds / 1000.0)) / 1e12);

    // copy strategic sample points from GPU result for validation
    for (int i = 0; i < point_idx; i++)
    {
        int row = validation_points[i][0];
        int col = validation_points[i][1];
        checkCudaErrors(cudaMemcpy(&h_gpu_results[i], d_Result + row * MATRIX_N + col, sizeof(float), cudaMemcpyDeviceToHost));
    }
    printf("GPU results sampled from %d strategic points\n", point_idx);

    // CPU verification for strategic sample points: Validate A*B + C*D coupling
    printf("Computing CPU reference for %d strategic validation points...\n", point_idx);
    for (int p = 0; p < point_idx; p++)
    {
        int row = validation_points[p][0];
        int col = validation_points[p][1];

        float ab_sum = 0.0f;
        float cd_sum = 0.0f;

        for (int k = 0; k < MATRIX_K; k++)
        {
            ab_sum += __bfloat162float(h_A[row * MATRIX_K + k]) * __bfloat162float(h_B[k * MATRIX_N + col]);
        }

        for (int k = 0; k < MATRIX_K; k++)
        {
            cd_sum += __bfloat162float(h_C[row * MATRIX_K + k]) * __bfloat162float(h_D[k * MATRIX_N + col]);
        }

        h_cpu_results[p] = ab_sum + cd_sum;
    }

    // compare GPU and CPU results
    int passed_elements = 0;
    float max_error = 0.0f;

    for (int p = 0; p < point_idx; p++)
    {
        float error = fabsf(h_gpu_results[p] - h_cpu_results[p]);
        max_error = fmaxf(max_error, error);

        if (error < 1e-3f)
        {
            passed_elements++;
        }
    }

    printf("Validation Summary: %d/%d points passed (%.1f%%)\n", passed_elements, point_idx, 100.0f * passed_elements / point_idx);
    printf("Maximum error: %.6f\n", max_error);
    printf("Coverage: %.4f%% of output matrix\n", 100.0f * point_idx / (MATRIX_M * MATRIX_N));

    if (passed_elements == point_idx)
    {
        printf("VALIDATION PASSED - Coupled tensor operations verified!\n");
    }
    else
    {
        printf("VALIDATION FAILED - %d points failed\n", point_idx - passed_elements);
    }

    // Cleanup (properly destroy events to avoid CUDA context pollution)
    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(stop));

    free(h_A);
    free(h_B);
    free(h_C);
    free(h_D);
    free(h_gpu_results);
    free(h_cpu_results);
    checkCudaErrors(cudaFree(reinterpret_cast<void*>(d_A)));
    checkCudaErrors(cudaFree(reinterpret_cast<void*>(d_B)));
    checkCudaErrors(cudaFree(reinterpret_cast<void*>(d_C)));
    checkCudaErrors(cudaFree(reinterpret_cast<void*>(d_D)));
    checkCudaErrors(cudaFree(reinterpret_cast<void*>(d_Result)));

    return 0;
}
