/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// This header contains policy-free error-checking macros that extend GTest ASSERT / EXPECT.
// These are expected to be used with a "throw listener" that throws an exception on GTest ASSERT.

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <gtest/gtest.h>


inline const char* GetCuResultName(CUresult status)
{
    const char* pStr;
    if (CUDA_SUCCESS == cuGetErrorName(status, &pStr))
    {
        return pStr;
    }
    return "???";
}

inline const char* GetCuResultString(CUresult status)
{
    const char* pStr;
    if (CUDA_SUCCESS == cuGetErrorString(status, &pStr))
    {
        return pStr;
    }
    return "???";
}

template <class TFunc>
void ExpectCudaDriverApiCall(const char* pExpr, TFunc&& func)
{
    CUresult status = func();
    GTEST_ASSERT_(::testing::internal::EqHelper::Compare("CUDA_SUCCESS", pExpr, CUDA_SUCCESS, status), GTEST_NONFATAL_FAILURE_)
        << "CUresult=" << GetCuResultName(status) << " : " << GetCuResultString(status);
}

template <class TFunc>
void AssertCudaDriverApiCall(const char* pExpr, TFunc&& func)
{
    CUresult status = func();
    GTEST_ASSERT_(::testing::internal::EqHelper::Compare("CUDA_SUCCESS", pExpr, CUDA_SUCCESS, status), GTEST_FATAL_FAILURE_)
        << "CUresult=" << GetCuResultName(status) << " : " << GetCuResultString(status);
}

template <class TFunc>
void ExpectCudaRuntimeCall(const char* pExpr, TFunc&& func)
{
    cudaError_t error = func();
    GTEST_ASSERT_(::testing::internal::EqHelper::Compare("cudaSuccess", pExpr, cudaSuccess, error), GTEST_NONFATAL_FAILURE_)
        << "cudaError=" << cudaGetErrorName(error) << " : " << cudaGetErrorString(error);
}

template <class TFunc>
void AssertCudaRuntimeCall(const char* pExpr, TFunc&& func)
{
    cudaError_t error = func();
    GTEST_ASSERT_(::testing::internal::EqHelper::Compare("cudaSuccess", pExpr, cudaSuccess, error), GTEST_FATAL_FAILURE_)
        << "cudaError=" << cudaGetErrorName(error) << " : " << cudaGetErrorString(error);
}

#define EXPECT_CURESULT(x_) AssertCudaDriverApiCall(#x_, [&]() { return x_; })
#define ASSERT_CURESULT(x_) AssertCudaDriverApiCall(#x_, [&]() { return x_; })

#define EXPECT_CUDART(x_) AssertCudaRuntimeCall(#x_, [&]() { return x_; })
#define ASSERT_CUDART(x_) AssertCudaRuntimeCall(#x_, [&]() { return x_; })
