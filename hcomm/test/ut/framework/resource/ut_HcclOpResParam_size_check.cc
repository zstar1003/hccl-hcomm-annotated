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

class HcclHcclOpResParamStructTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(HcclHcclOpResParamStructTest, TestHcclOpResParamSize) { EXPECT_EQ(sizeof(HcclOpResParam), 2122720u); }

TEST_F(HcclHcclOpResParamStructTest, TestHcclOpResParamFieldOffsets)
{
    EXPECT_EQ(offsetof(HcclOpResParam, mc2WorkSpace), 0u);
    EXPECT_EQ(offsetof(HcclOpResParam, localUsrRankId), offsetof(HcclOpResParam, mc2WorkSpace) + 16u);
    EXPECT_EQ(offsetof(HcclOpResParam, rankSize), offsetof(HcclOpResParam, localUsrRankId) + 4u);
    EXPECT_EQ(offsetof(HcclOpResParam, winSize), offsetof(HcclOpResParam, rankSize) + 4u);
    EXPECT_EQ(offsetof(HcclOpResParam, localWindowsIn), offsetof(HcclOpResParam, winSize) + 8u);
    EXPECT_EQ(offsetof(HcclOpResParam, localWindowsOut), offsetof(HcclOpResParam, localWindowsIn) + 8u);
    EXPECT_EQ(offsetof(HcclOpResParam, hcomId), offsetof(HcclOpResParam, localWindowsOut) + 8u);
    EXPECT_EQ(offsetof(HcclOpResParam, winExpSize), offsetof(HcclOpResParam, hcomId) + 128u);
    EXPECT_EQ(offsetof(HcclOpResParam, localWindowsExp), offsetof(HcclOpResParam, winExpSize) + 8u);
    EXPECT_EQ(offsetof(HcclOpResParam, rWinStart), offsetof(HcclOpResParam, localWindowsExp) + 8u);
    EXPECT_EQ(offsetof(HcclOpResParam, rWinOffset), offsetof(HcclOpResParam, rWinStart) + 4u);
    EXPECT_EQ(offsetof(HcclOpResParam, version), offsetof(HcclOpResParam, rWinOffset) + 4u);
    EXPECT_EQ(offsetof(HcclOpResParam, reservedStrcut), offsetof(HcclOpResParam, version) + 8u);
    EXPECT_EQ(offsetof(HcclOpResParam, topoInfo), offsetof(HcclOpResParam, reservedStrcut) + 2472u);
    EXPECT_EQ(offsetof(HcclOpResParam, config), offsetof(HcclOpResParam, topoInfo) + 160u);
    EXPECT_EQ(offsetof(HcclOpResParam, hostStateInfo), offsetof(HcclOpResParam, config) + 48u);
    EXPECT_EQ(offsetof(HcclOpResParam, aicpuStateInfo), offsetof(HcclOpResParam, hostStateInfo) + 8u);
    EXPECT_EQ(offsetof(HcclOpResParam, lockAddr), offsetof(HcclOpResParam, aicpuStateInfo) + 8u);
    EXPECT_EQ(offsetof(HcclOpResParam, rsv), offsetof(HcclOpResParam, lockAddr) + 8u);
    EXPECT_EQ(offsetof(HcclOpResParam, notifysize), offsetof(HcclOpResParam, rsv) + 64u);
    EXPECT_EQ(offsetof(HcclOpResParam, remoteResNum), offsetof(HcclOpResParam, notifysize) + 4u);
    EXPECT_EQ(offsetof(HcclOpResParam, remoteRes), offsetof(HcclOpResParam, remoteResNum) + 4u);
    EXPECT_EQ(offsetof(HcclOpResParam, kfcControlTransferH2DParams), offsetof(HcclOpResParam, remoteRes) + 2097152u);
    EXPECT_EQ(
        offsetof(HcclOpResParam, kfcStatusTransferD2HParams),
        offsetof(HcclOpResParam, kfcControlTransferH2DParams) + 40u);
    EXPECT_EQ(offsetof(HcclOpResParam, tinyMem), offsetof(HcclOpResParam, kfcStatusTransferD2HParams) + 40u);
    EXPECT_EQ(offsetof(HcclOpResParam, tinyMemSize), offsetof(HcclOpResParam, tinyMem) + 8u);
    EXPECT_EQ(offsetof(HcclOpResParam, zeroCopyHeadPtr), offsetof(HcclOpResParam, tinyMemSize) + 8u);
    EXPECT_EQ(offsetof(HcclOpResParam, zeroCopyTailPtr), offsetof(HcclOpResParam, zeroCopyHeadPtr) + 8u);
    EXPECT_EQ(offsetof(HcclOpResParam, zeroCopyRingBuffer), offsetof(HcclOpResParam, zeroCopyTailPtr) + 8u);
    EXPECT_EQ(offsetof(HcclOpResParam, zeroCopyIpcPtrs), offsetof(HcclOpResParam, zeroCopyRingBuffer) + 8u);
    EXPECT_EQ(offsetof(HcclOpResParam, zeroCopyDevicePhyId), offsetof(HcclOpResParam, zeroCopyIpcPtrs) + 256u);
    EXPECT_EQ(offsetof(HcclOpResParam, utraceStatusFlag), offsetof(HcclOpResParam, zeroCopyDevicePhyId) + 128u);
    EXPECT_EQ(offsetof(HcclOpResParam, opCounterInfo), offsetof(HcclOpResParam, utraceStatusFlag) + 8u);
    EXPECT_EQ(offsetof(HcclOpResParam, hierarchicalAlgInfo), offsetof(HcclOpResParam, opCounterInfo) + 32u);
    EXPECT_EQ(offsetof(HcclOpResParam, localRes), offsetof(HcclOpResParam, hierarchicalAlgInfo) + 32u);
    EXPECT_EQ(offsetof(HcclOpResParam, debugConfig), offsetof(HcclOpResParam, localRes) + 3464u);
    EXPECT_EQ(offsetof(HcclOpResParam, aicpuCustomParamAddr), offsetof(HcclOpResParam, debugConfig) + 8u);
    EXPECT_EQ(offsetof(HcclOpResParam, aicpuCustomParamSize), offsetof(HcclOpResParam, aicpuCustomParamAddr) + 8u);
    EXPECT_EQ(offsetof(HcclOpResParam, userMemRes), offsetof(HcclOpResParam, aicpuCustomParamSize) + 8u);
    EXPECT_EQ(offsetof(HcclOpResParam, userMemType), offsetof(HcclOpResParam, userMemRes) + 18432u);
    EXPECT_EQ(offsetof(HcclOpResParam, aicpuOrderStreamParam), offsetof(HcclOpResParam, userMemType) + 8u);
    EXPECT_EQ(offsetof(HcclOpResParam, aicpuOrderNotifyAddr), offsetof(HcclOpResParam, aicpuOrderStreamParam) + 32u);
    EXPECT_EQ(offsetof(HcclOpResParam, aicpuOrderNotifySize), offsetof(HcclOpResParam, aicpuOrderNotifyAddr) + 8u);
    EXPECT_EQ(
        offsetof(HcclOpResParam, multiSuperPodDiffDeviceNumMode), offsetof(HcclOpResParam, aicpuOrderNotifySize) + 8u);
    EXPECT_EQ(offsetof(HcclOpResParam, isARSDoubleRing), offsetof(HcclOpResParam, multiSuperPodDiffDeviceNumMode) + 4u);
    EXPECT_EQ(offsetof(HcclOpResParam, opEntry), offsetof(HcclOpResParam, isARSDoubleRing) + 1u);
    EXPECT_EQ(offsetof(HcclOpResParam, hcclSdmaQos), offsetof(HcclOpResParam, opEntry) + 3u);
    EXPECT_EQ(offsetof(HcclOpResParam, sizeOfAiRMAInfo), offsetof(HcclOpResParam, hcclSdmaQos) + 8u);
    EXPECT_EQ(offsetof(HcclOpResParam, aiRMAInfo), offsetof(HcclOpResParam, sizeOfAiRMAInfo) + 8u);
}

TEST_F(HcclHcclOpResParamStructTest, TestHcclOpResParamFieldSizes)
{
    /*正式方案上库后修改*/
    // EXPECT_EQ(sizeof(HcclMC2WorkSpace), 16u);
    EXPECT_EQ(sizeof(ReservedStruct), 2472u);
    EXPECT_EQ(sizeof(AlgoTopoInfo), 160u);
    EXPECT_EQ(sizeof(HcclOpConfig), 48u);
    EXPECT_EQ(sizeof(RemoteResPtr), 16u);
    EXPECT_EQ(sizeof(hccl::HDCommunicateParams), 40u);
    EXPECT_EQ(sizeof(OpCounterInfo), 32u);
    EXPECT_EQ(sizeof(HierarchicalAlgInfo), 32u);
    /*正式方案上库后修改*/
    // EXPECT_EQ(sizeof(LocalResInfoV2), 3464u);
    EXPECT_EQ(sizeof(MemDetails), 24u);
    EXPECT_EQ(sizeof(HcclStreamParam), 32u);
}
