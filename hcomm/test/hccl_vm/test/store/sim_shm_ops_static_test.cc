/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <gtest/gtest.h>

#include "store_sim_shm_ops.cc"
#include "store_sim_shm_ops.h"

class SimShmOpsStaticTest : public testing::Test {};

TEST_F(SimShmOpsStaticTest, Swap32_BasicValue)
{
    uint32_t input = 0x01020304;
    uint32_t expected = 0x04030201;
    EXPECT_EQ(Swap32(input), expected);
}

TEST_F(SimShmOpsStaticTest, Swap32_Zero) { EXPECT_EQ(Swap32(0u), 0u); }

TEST_F(SimShmOpsStaticTest, Swap32_MaxValue)
{
    uint32_t input = 0xFFFFFFFF;
    EXPECT_EQ(Swap32(input), 0xFFFFFFFF);
}

TEST_F(SimShmOpsStaticTest, Swap32_SameByteValue)
{
    uint32_t input = 0xAAAAAAAA;
    EXPECT_EQ(Swap32(input), 0xAAAAAAAA);
}

TEST_F(SimShmOpsStaticTest, Swap32_AllBitsSet)
{
    uint32_t input = 0x12345678;
    uint32_t expected = 0x78563412;
    EXPECT_EQ(Swap32(input), expected);
}

TEST_F(SimShmOpsStaticTest, Swap64_BasicValue)
{
    uint64_t input = 0x0102030405060708ULL;
    uint64_t expected = 0x0807060504030201ULL;
    EXPECT_EQ(Swap64(input), expected);
}

TEST_F(SimShmOpsStaticTest, Swap64_Zero) { EXPECT_EQ(Swap64(0ULL), 0ULL); }

TEST_F(SimShmOpsStaticTest, Swap64_MaxValue)
{
    uint64_t input = 0xFFFFFFFFFFFFFFFFULL;
    EXPECT_EQ(Swap64(input), 0xFFFFFFFFFFFFFFFFULL);
}

TEST_F(SimShmOpsStaticTest, Swap64_SameByteValue)
{
    uint64_t input = 0xAAAAAAAAAAAAAAAAULL;
    EXPECT_EQ(Swap64(input), 0xAAAAAAAAAAAAAAAAULL);
}

TEST_F(SimShmOpsStaticTest, Swap64_AllBitsSet)
{
    uint64_t input = 0x1234567890ABCDEFULL;
    uint64_t expected = 0xEFCDAB9078563412ULL;
    EXPECT_EQ(Swap64(input), expected);
}

TEST_F(SimShmOpsStaticTest, IsBigEndian_ReturnsBoolean)
{
    int result = IsBigEndian();
    EXPECT_TRUE(result == 0 || result == 1);
}

TEST_F(SimShmOpsStaticTest, ToLe32_OnLittleEndian_ReturnsSameValue)
{
    uint32_t input = 0x12345678;
    EXPECT_EQ(ToLe32(input), input);
}

TEST_F(SimShmOpsStaticTest, ToLe64_OnLittleEndian_ReturnsSameValue)
{
    uint64_t input = 0x1234567890ABCDEFULL;
    EXPECT_EQ(ToLe64(input), input);
}

TEST_F(SimShmOpsStaticTest, FromLe32_OnLittleEndian_ReturnsSameValue)
{
    uint32_t input = 0x12345678;
    EXPECT_EQ(FromLe32(input), input);
}

TEST_F(SimShmOpsStaticTest, FromLe64_OnLittleEndian_ReturnsSameValue)
{
    uint64_t input = 0x1234567890ABCDEFULL;
    EXPECT_EQ(FromLe64(input), input);
}

TEST_F(SimShmOpsStaticTest, Swap32_RoundTrip_Identity)
{
    uint32_t original = 0xDEADBEEF;
    EXPECT_EQ(Swap32(Swap32(original)), original);
}

TEST_F(SimShmOpsStaticTest, Swap64_RoundTrip_Identity)
{
    uint64_t original = 0xDEADBEEFCAFEBABEULL;
    EXPECT_EQ(Swap64(Swap64(original)), original);
}

TEST_F(SimShmOpsStaticTest, ToLe32_Zero_ReturnsZero) { EXPECT_EQ(ToLe32(0u), 0u); }

TEST_F(SimShmOpsStaticTest, ToLe64_Zero_ReturnsZero) { EXPECT_EQ(ToLe64(0ULL), 0ULL); }

TEST_F(SimShmOpsStaticTest, FromLe32_Zero_ReturnsZero) { EXPECT_EQ(FromLe32(0u), 0u); }

TEST_F(SimShmOpsStaticTest, FromLe64_Zero_ReturnsZero) { EXPECT_EQ(FromLe64(0ULL), 0ULL); }
