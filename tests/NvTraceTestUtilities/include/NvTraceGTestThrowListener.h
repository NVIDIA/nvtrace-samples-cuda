/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <gtest/gtest.h>

/// Support for failing/skipping & terminating a test from a *subroutine*.
/// Must be called after InitGoogleTest() in main().
inline void AddGTestThrowListener()
{
    /// boilerplate from https://google.github.io/googletest/advanced.html#asserting-on-subroutines-with-an-exception
    class ThrowListener : public ::testing::EmptyTestEventListener
    {
        void OnTestPartResult(const ::testing::TestPartResult& result) override
        {
            if (result.type() == ::testing::TestPartResult::kFatalFailure || result.type() == ::testing::TestPartResult::kSkip)
            {
                throw ::testing::AssertionException(result);
            }
        }
    };

    ::testing::UnitTest::GetInstance()->listeners().Append(new ThrowListener);
}
