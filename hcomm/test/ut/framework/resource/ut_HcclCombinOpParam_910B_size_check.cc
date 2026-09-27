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
#include "aicpu_operator_pub.h"

class HcclCombinOpParam910BStructTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(HcclCombinOpParam910BStructTest, TestHcclCombinOpParamSize) { EXPECT_EQ(sizeof(HcclCombinOpParam), 8928u); }

TEST_F(HcclCombinOpParam910BStructTest, TestHcclCombinOpParamFieldOffsets)
{
    EXPECT_EQ(offsetof(HcclCombinOpParam, mc2WorkSpace), 0u);
    EXPECT_EQ(offsetof(HcclCombinOpParam, rankId), offsetof(HcclCombinOpParam, mc2WorkSpace) + 16u);
    EXPECT_EQ(offsetof(HcclCombinOpParam, rankNum), offsetof(HcclCombinOpParam, rankId) + 4u);
    EXPECT_EQ(offsetof(HcclCombinOpParam, winSize), offsetof(HcclCombinOpParam, rankNum) + 4u);
    EXPECT_EQ(offsetof(HcclCombinOpParam, windowsIn), offsetof(HcclCombinOpParam, winSize) + 8u);
    EXPECT_EQ(offsetof(HcclCombinOpParam, windowsOut), offsetof(HcclCombinOpParam, windowsIn) + 256u);
    EXPECT_EQ(offsetof(HcclCombinOpParam, hcomId), offsetof(HcclCombinOpParam, windowsOut) + 256u);
    EXPECT_EQ(offsetof(HcclCombinOpParam, streamInfo), offsetof(HcclCombinOpParam, hcomId) + 128u);
    EXPECT_EQ(offsetof(HcclCombinOpParam, signalInfo), offsetof(HcclCombinOpParam, streamInfo) + 512u);
    EXPECT_EQ(offsetof(HcclCombinOpParam, config), offsetof(HcclCombinOpParam, signalInfo) + 7264u);
    EXPECT_EQ(offsetof(HcclCombinOpParam, overFlowAddr), offsetof(HcclCombinOpParam, config) + 48u);
    EXPECT_EQ(offsetof(HcclCombinOpParam, onlyRead), offsetof(HcclCombinOpParam, overFlowAddr) + 8u);
    EXPECT_EQ(offsetof(HcclCombinOpParam, kfcControlTransferH2DParams), offsetof(HcclCombinOpParam, onlyRead) + 8u);
    EXPECT_EQ(
        offsetof(HcclCombinOpParam, kfcStatusTransferD2HParams),
        offsetof(HcclCombinOpParam, kfcControlTransferH2DParams) + 40u);
    EXPECT_EQ(offsetof(HcclCombinOpParam, padding), offsetof(HcclCombinOpParam, kfcStatusTransferD2HParams) + 40u);
    EXPECT_EQ(offsetof(HcclCombinOpParam, winExpSize), offsetof(HcclCombinOpParam, padding) + 16u);
    EXPECT_EQ(offsetof(HcclCombinOpParam, windowsExp), offsetof(HcclCombinOpParam, winExpSize) + 8u);
    EXPECT_EQ(offsetof(HcclCombinOpParam, multiServerFlag), offsetof(HcclCombinOpParam, windowsExp) + 256u);
    EXPECT_EQ(offsetof(HcclCombinOpParam, ibverbsData), offsetof(HcclCombinOpParam, multiServerFlag) + 8u);
    EXPECT_EQ(offsetof(HcclCombinOpParam, ibverbsDataSize), offsetof(HcclCombinOpParam, ibverbsData) + 8u);
    EXPECT_EQ(offsetof(HcclCombinOpParam, sizeOfAiRMAInfo), offsetof(HcclCombinOpParam, ibverbsDataSize) + 8u);
    EXPECT_EQ(offsetof(HcclCombinOpParam, aiRMAInfo), offsetof(HcclCombinOpParam, sizeOfAiRMAInfo) + 8u);
    EXPECT_EQ(offsetof(HcclCombinOpParam, capabilityPtr), offsetof(HcclCombinOpParam, aiRMAInfo) + 8u);
    EXPECT_EQ(offsetof(HcclCombinOpParam, capabilitySize), offsetof(HcclCombinOpParam, capabilityPtr) + 8u);
}

TEST_F(HcclCombinOpParam910BStructTest, TestHcclCombinOpParamFieldSizes)
{
    EXPECT_EQ(sizeof(HcclMC2WorkSpace), 16u);
    EXPECT_EQ(sizeof(HcclStreamInfo), 16u);
    EXPECT_EQ(sizeof(HcclCombinOpSignalParam), 7264u);
    EXPECT_EQ(sizeof(HcclOpConfig), 48u);
    EXPECT_EQ(sizeof(HcclAiRMAWQ), 64u);
    EXPECT_EQ(sizeof(HcclAiRMACQ), 56u);
}
