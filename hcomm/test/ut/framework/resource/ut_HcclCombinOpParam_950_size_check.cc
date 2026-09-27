/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include "mc2_type.h"
using namespace Hccl;

class HcclCombinOpParam950StructTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(HcclCombinOpParam950StructTest, TestHcclCombinOpParamSize) { EXPECT_EQ(sizeof(HcclCombinOpParam), 9320u); }

TEST_F(HcclCombinOpParam950StructTest, TestHcclCombinOpParamFieldOffsets)
{
    EXPECT_EQ(offsetof(HcclCombinOpParam, workSpace), 0u);
    EXPECT_EQ(offsetof(HcclCombinOpParam, workSpaceSize), offsetof(HcclCombinOpParam, workSpace) + 8u);
    EXPECT_EQ(offsetof(HcclCombinOpParam, rankId), offsetof(HcclCombinOpParam, workSpaceSize) + 8u);
    EXPECT_EQ(offsetof(HcclCombinOpParam, rankDim), offsetof(HcclCombinOpParam, rankId) + 4u);
    EXPECT_EQ(offsetof(HcclCombinOpParam, winSize), offsetof(HcclCombinOpParam, rankDim) + 4u);
    EXPECT_EQ(offsetof(HcclCombinOpParam, windowsIn), offsetof(HcclCombinOpParam, winSize) + 8u);
    EXPECT_EQ(offsetof(HcclCombinOpParam, windowsOut), offsetof(HcclCombinOpParam, windowsIn) + 512u);
    EXPECT_EQ(offsetof(HcclCombinOpParam, xnAddr), offsetof(HcclCombinOpParam, windowsOut) + 512u);
    EXPECT_EQ(offsetof(HcclCombinOpParam, ckeAddr), offsetof(HcclCombinOpParam, xnAddr) + 8u);
    EXPECT_EQ(offsetof(HcclCombinOpParam, msAddr), offsetof(HcclCombinOpParam, ckeAddr) + 8u);
    EXPECT_EQ(offsetof(HcclCombinOpParam, msSize), offsetof(HcclCombinOpParam, msAddr) + 8u);
    EXPECT_EQ(offsetof(HcclCombinOpParam, opType), offsetof(HcclCombinOpParam, msSize) + 8u);
    EXPECT_EQ(offsetof(HcclCombinOpParam, algorithmType), offsetof(HcclCombinOpParam, opType) + 32u);
    EXPECT_EQ(offsetof(HcclCombinOpParam, wq), offsetof(HcclCombinOpParam, algorithmType) + 8u);
    EXPECT_EQ(offsetof(HcclCombinOpParam, cq), offsetof(HcclCombinOpParam, wq) + 5120u);
}

TEST_F(HcclCombinOpParam950StructTest, TestHcclCombinOpParamFieldSizes)
{
    EXPECT_EQ(sizeof(HcclAiRMAWQ), 80u);
    EXPECT_EQ(sizeof(HcclAiRMACQ), 48u);
}
