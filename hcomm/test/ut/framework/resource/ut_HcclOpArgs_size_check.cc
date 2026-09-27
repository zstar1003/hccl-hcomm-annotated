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

class HcclOpArgsStructTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(HcclOpArgsStructTest, TestHcclOpArgsSize) { EXPECT_EQ(sizeof(HcclOpArgs), 160u); }

TEST_F(HcclOpArgsStructTest, TestHcclOpArgsFieldOffsets)
{
    EXPECT_EQ(offsetof(HcclOpArgs, srcDataType), 0u);
    EXPECT_EQ(offsetof(HcclOpArgs, dstDataType), offsetof(HcclOpArgs, srcDataType) + 1u);
    EXPECT_EQ(offsetof(HcclOpArgs, reduceType), offsetof(HcclOpArgs, dstDataType) + 1u);
    EXPECT_EQ(offsetof(HcclOpArgs, count), offsetof(HcclOpArgs, reduceType) + 6u);
    EXPECT_EQ(offsetof(HcclOpArgs, algConfig), offsetof(HcclOpArgs, count) + 8u);
    EXPECT_EQ(offsetof(HcclOpArgs, commEngine), offsetof(HcclOpArgs, algConfig) + 128u);
    EXPECT_EQ(offsetof(HcclOpArgs, reverse), offsetof(HcclOpArgs, commEngine) + 8u);
}

TEST_F(HcclOpArgsStructTest, TestHcclOpArgsFieldSizes)
{
    EXPECT_EQ(sizeof(DataType), 1u);
    EXPECT_EQ(sizeof(ReduceOp), 1u);
    EXPECT_EQ(sizeof(HcclAccelerator), 1u);
}

TEST_F(HcclOpArgsStructTest, TestHcclOpArgsAlgConfigSize)
{
    HcclOpArgs opArgs;
    EXPECT_EQ(sizeof(opArgs.algConfig), 128u);
}

TEST_F(HcclOpArgsStructTest, TestHcclOpArgsInitFunction)
{
    HcclOpArgs opArgs;
    opArgs.Init();

    EXPECT_EQ(opArgs.srcDataType, DataType::FP16);
    EXPECT_EQ(opArgs.dstDataType, DataType::FP16);
    EXPECT_EQ(opArgs.reduceType, ReduceOp::SUM);
    EXPECT_EQ(opArgs.count, 0u);
}
