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

extern "C" {
uint32_t TsdCapabilityGet(const uint32_t logicDeviceId, const int32_t type, const uint64_t ptr);
}

class TsdClientStubTest : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(TsdClientStubTest, TsdCapabilityGet_BasicCall)
{
    uint32_t result = TsdCapabilityGet(0, 0, 0);
    EXPECT_EQ(result, 0u);
}

TEST_F(TsdClientStubTest, TsdCapabilityGet_DifferentDeviceId)
{
    EXPECT_EQ(TsdCapabilityGet(0, 0, 0), 0u);
    EXPECT_EQ(TsdCapabilityGet(1, 0, 0), 0u);
    EXPECT_EQ(TsdCapabilityGet(100, 0, 0), 0u);
    EXPECT_EQ(TsdCapabilityGet(UINT32_MAX, 0, 0), 0u);
}

TEST_F(TsdClientStubTest, TsdCapabilityGet_DifferentType)
{
    EXPECT_EQ(TsdCapabilityGet(0, 0, 0), 0u);
    EXPECT_EQ(TsdCapabilityGet(0, 1, 0), 0u);
    EXPECT_EQ(TsdCapabilityGet(0, -1, 0), 0u);
    EXPECT_EQ(TsdCapabilityGet(0, INT32_MAX, 0), 0u);
    EXPECT_EQ(TsdCapabilityGet(0, INT32_MIN, 0), 0u);
}

TEST_F(TsdClientStubTest, TsdCapabilityGet_DifferentPtr)
{
    EXPECT_EQ(TsdCapabilityGet(0, 0, 0), 0u);
    EXPECT_EQ(TsdCapabilityGet(0, 0, 1), 0u);
    EXPECT_EQ(TsdCapabilityGet(0, 0, 12345), 0u);
    EXPECT_EQ(TsdCapabilityGet(0, 0, UINT64_MAX), 0u);
}

TEST_F(TsdClientStubTest, TsdCapabilityGet_AllCombinations)
{
    for (uint32_t deviceId = 0; deviceId < 5; deviceId++) {
        for (int32_t type = 0; type < 5; type++) {
            for (uint64_t ptr = 0; ptr < 5; ptr++) {
                uint32_t result = TsdCapabilityGet(deviceId, type, ptr);
                EXPECT_EQ(result, 0u);
            }
        }
    }
}

TEST_F(TsdClientStubTest, TsdCapabilityGet_BoundaryValues)
{
    EXPECT_EQ(TsdCapabilityGet(0, 0, 0), 0u);
    EXPECT_EQ(TsdCapabilityGet(UINT32_MAX, INT32_MAX, UINT64_MAX), 0u);
    EXPECT_EQ(TsdCapabilityGet(UINT32_MAX, INT32_MIN, 0), 0u);
}
