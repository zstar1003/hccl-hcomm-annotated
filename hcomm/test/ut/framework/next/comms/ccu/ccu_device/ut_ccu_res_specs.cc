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
#include <mockcpp/mockcpp.hpp>

#include "ccu_res_specs.h"
#include "hccl_common.h"

#define private public
#define protected public

using namespace hcomm;

constexpr int32_t fakeDevLogicId = 45; // 选用一个设备id，避免单例在内存在影响其他用例

static void SetupSpecs(int32_t devLogicId, CcuVersion ccuVersion)
{
    auto& specs = CcuResSpecifications::GetInstance(devLogicId);
    specs.initFlag_ = true;
    specs.ccuVersion_ = ccuVersion;
    specs.serveMode_ = ServeMode::NORMAL;
    specs.dieEnableFlags_[0] = true;
    specs.dieEnableFlags_[1] = true;
    specs.resSpecs_[0].msId = 100;
    specs.resSpecs_[0].missionKey = 200;
    specs.resSpecs_[0].resourceAddr = 0xE7FFBF800000;
    specs.resSpecs_[0].loopEngineNum = 200;
    specs.resSpecs_[0].missionNum = 16;
    specs.resSpecs_[0].instructionNum = 32768;

    if (ccuVersion == CcuVersion::CCU_V2) {
        specs.resSpecs_[0].xnNum = 12288;
        specs.resSpecs_[0].gsaNum = 0;
        specs.resSpecs_[0].channelNum = 1024;
        specs.resSpecs_[0].pfeNum = 20;
        specs.resSpecs_[0].countXnNum = 4096;
        specs.resSpecs_[0].loopCkeNum = 512;
        specs.resSpecs_[0].channelJettyMap = CcuChannelJettyMap(8, 1);
    } else {
        specs.resSpecs_[0].xnNum = 3072;
        specs.resSpecs_[0].gsaNum = 3072;
        specs.resSpecs_[0].channelNum = 128;
        specs.resSpecs_[0].pfeNum = 10;
        specs.resSpecs_[0].countXnNum = 0;
        specs.resSpecs_[0].loopCkeNum = 0;
        specs.resSpecs_[0].channelJettyMap = CcuChannelJettyMap(1, 1);
    }

    specs.resSpecs_[0].msNum = 1536;
    specs.resSpecs_[0].ckeNum = 1024;
    specs.resSpecs_[0].jettyNum = 128;
    specs.resSpecs_[0].wqeBBNum = 4096;

    specs.resSpecs_[1] = specs.resSpecs_[0];
}

static void SetupSpecsV2(int32_t devLogicId) { SetupSpecs(devLogicId, CcuVersion::CCU_V2); }

static void SetupSpecsV1(int32_t devLogicId) { SetupSpecs(devLogicId, CcuVersion::CCU_V1); }

class CcuResSpecificationsTest : public testing::Test {
protected:
    void SetUp() override { SetupSpecsV2(fakeDevLogicId); }
    void TearDown() override {}
};

TEST_F(CcuResSpecificationsTest, Ut_GetServeMode_Expect_ReturnNORMAL)
{
    EXPECT_EQ(CcuResSpecifications::GetInstance(fakeDevLogicId).GetServeMode(), ServeMode::NORMAL);
}

TEST_F(CcuResSpecificationsTest, Ut_GetCcuVersion_Expect_ReturnV2)
{
    EXPECT_EQ(CcuResSpecifications::GetInstance(fakeDevLogicId).GetCcuVersion(), CcuVersion::CCU_V2);
}

TEST_F(CcuResSpecificationsTest, Ut_GetDieEnableFlag_When_DieEnabled_Expect_True)
{
    bool flag = false;
    EXPECT_EQ(CcuResSpecifications::GetInstance(fakeDevLogicId).GetDieEnableFlag(0, flag), HcclResult::HCCL_SUCCESS);
    EXPECT_TRUE(flag);
}

TEST_F(CcuResSpecificationsTest, Ut_GetDieEnableFlag_When_DieIdInvalid_Expect_ReturnHCCL_E_PARA)
{
    bool flag = false;
    EXPECT_EQ(CcuResSpecifications::GetInstance(fakeDevLogicId).GetDieEnableFlag(5, flag), HcclResult::HCCL_E_PARA);
}

TEST_F(CcuResSpecificationsTest, Ut_GetResourceAddr_When_Normal_Expect_ReturnSuccess)
{
    uint64_t addr = 0;
    EXPECT_EQ(CcuResSpecifications::GetInstance(fakeDevLogicId).GetResourceAddr(0, addr), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(addr, 0xE7FFBF800000ULL);
}

TEST_F(CcuResSpecificationsTest, Ut_GetResourceAddr_When_DieIdInvalid_Expect_ReturnHCCL_E_PARA)
{
    uint64_t addr = 0;
    EXPECT_EQ(CcuResSpecifications::GetInstance(fakeDevLogicId).GetResourceAddr(5, addr), HcclResult::HCCL_E_PARA);
}

TEST_F(CcuResSpecificationsTest, Ut_GetResourceAddr_When_DieDisabled_Expect_ReturnHCCL_E_PARA)
{
    CcuResSpecifications::GetInstance(fakeDevLogicId).dieEnableFlags_[1] = false;
    uint64_t addr = 0;
    EXPECT_EQ(CcuResSpecifications::GetInstance(fakeDevLogicId).GetResourceAddr(1, addr), HcclResult::HCCL_E_PARA);
}

TEST_F(CcuResSpecificationsTest, Ut_GetXnBaseAddr_When_V2_Expect_ReturnSuccess)
{
    uint64_t addr = 0;
    EXPECT_EQ(CcuResSpecifications::GetInstance(fakeDevLogicId).GetXnBaseAddr(0, addr), HcclResult::HCCL_SUCCESS);
    EXPECT_NE(addr, 0ULL);
}

TEST_F(CcuResSpecificationsTest, Ut_GetXnBaseAddr_When_V1_Expect_ReturnSuccess)
{
    SetupSpecsV1(1);
    uint64_t addr = 0;
    EXPECT_EQ(CcuResSpecifications::GetInstance(1).GetXnBaseAddr(0, addr), HcclResult::HCCL_SUCCESS);
    EXPECT_NE(addr, 0ULL);
}

TEST_F(CcuResSpecificationsTest, Ut_GetXnBaseAddr_When_ResourceAddrZero_Expect_ReturnHCCL_E_INTERNAL)
{
    CcuResSpecifications::GetInstance(fakeDevLogicId).resSpecs_[0].resourceAddr = 0;
    uint64_t addr = 0;
    EXPECT_EQ(CcuResSpecifications::GetInstance(fakeDevLogicId).GetXnBaseAddr(0, addr), HcclResult::HCCL_E_INTERNAL);
}

TEST_F(CcuResSpecificationsTest, Ut_GetCkeBaseAddr_When_V2_Expect_ReturnSuccess)
{
    uint64_t addr = 0;
    EXPECT_EQ(CcuResSpecifications::GetInstance(fakeDevLogicId).GetCkeBaseAddr(0, addr), HcclResult::HCCL_SUCCESS);
    EXPECT_NE(addr, 0ULL);
}

TEST_F(CcuResSpecificationsTest, Ut_GetCkeBaseAddr_When_V1_Expect_ReturnSuccess)
{
    SetupSpecsV1(1);
    uint64_t addr = 0;
    EXPECT_EQ(CcuResSpecifications::GetInstance(1).GetCkeBaseAddr(0, addr), HcclResult::HCCL_SUCCESS);
    EXPECT_NE(addr, 0ULL);
}

TEST_F(CcuResSpecificationsTest, Ut_GetXnOffsetCcumBaseAddr_When_V2_Expect_NonZero)
{
    uint64_t addr = CcuResSpecifications::GetInstance(fakeDevLogicId).GetXnOffsetCcumBaseAddr(0);
    EXPECT_NE(addr, 0ULL);
}

TEST_F(CcuResSpecificationsTest, Ut_GetXnOffsetCcumBaseAddr_When_V1_Expect_NonZero)
{
    SetupSpecsV1(1);
    uint64_t addr = CcuResSpecifications::GetInstance(1).GetXnOffsetCcumBaseAddr(0);
    EXPECT_NE(addr, 0ULL);
}

TEST_F(CcuResSpecificationsTest, Ut_GetXnOffsetCcumBaseAddr_When_DieIdInvalid_Expect_InvalidAddr)
{
    EXPECT_EQ(CcuResSpecifications::GetInstance(fakeDevLogicId).GetXnOffsetCcumBaseAddr(5), INVALID_ADDR);
}

TEST_F(CcuResSpecificationsTest, Ut_GetCkeOffsetCcumBaseAddr_When_V2_Expect_NonZero)
{
    uint64_t addr = CcuResSpecifications::GetInstance(fakeDevLogicId).GetCkeOffsetCcumBaseAddr(0);
    EXPECT_NE(addr, 0ULL);
}

TEST_F(CcuResSpecificationsTest, Ut_GetCkeOffsetCcumBaseAddr_When_V1_Expect_NonZero)
{
    SetupSpecsV1(1);
    uint64_t addr = CcuResSpecifications::GetInstance(1).GetCkeOffsetCcumBaseAddr(0);
    EXPECT_NE(addr, 0ULL);
}

TEST_F(CcuResSpecificationsTest, Ut_GetXnOffsetCcumAddrById_When_Normal_Expect_ReturnSuccess)
{
    uint64_t addr = 0;
    EXPECT_EQ(
        CcuResSpecifications::GetInstance(fakeDevLogicId).GetXnOffsetCcumAddrById(0, 10, addr),
        HcclResult::HCCL_SUCCESS);
    EXPECT_NE(addr, 0ULL);
}

TEST_F(CcuResSpecificationsTest, Ut_GetCkeOffsetCcumAddrById_When_Normal_Expect_ReturnSuccess)
{
    uint64_t addr = 0;
    EXPECT_EQ(
        CcuResSpecifications::GetInstance(fakeDevLogicId).GetCkeOffsetCcumAddrById(0, 10, addr),
        HcclResult::HCCL_SUCCESS);
    EXPECT_NE(addr, 0ULL);
}

TEST_F(CcuResSpecificationsTest, Ut_GetMsId_When_Normal_Expect_ReturnSuccess)
{
    uint32_t msId = 0;
    EXPECT_EQ(CcuResSpecifications::GetInstance(fakeDevLogicId).GetMsId(0, msId), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(msId, 100U);
}

TEST_F(CcuResSpecificationsTest, Ut_GetMissionKey_When_Normal_Expect_ReturnSuccess)
{
    uint32_t key = 0;
    EXPECT_EQ(CcuResSpecifications::GetInstance(fakeDevLogicId).GetMissionKey(0, key), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(key, 200U);
}

TEST_F(CcuResSpecificationsTest, Ut_GetInstructionNum_When_Normal_Expect_ReturnSuccess)
{
    uint32_t num = 0;
    EXPECT_EQ(CcuResSpecifications::GetInstance(fakeDevLogicId).GetInstructionNum(0, num), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(num, 32768U);
}

TEST_F(CcuResSpecificationsTest, Ut_GetMissionNum_When_Normal_Expect_ReturnSuccess)
{
    uint32_t num = 0;
    EXPECT_EQ(CcuResSpecifications::GetInstance(fakeDevLogicId).GetMissionNum(0, num), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(num, 16U);
}

TEST_F(CcuResSpecificationsTest, Ut_GetLoopEngineNum_When_Normal_Expect_ReturnSuccess)
{
    uint32_t num = 0;
    EXPECT_EQ(CcuResSpecifications::GetInstance(fakeDevLogicId).GetLoopEngineNum(0, num), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(num, 200U);
}

TEST_F(CcuResSpecificationsTest, Ut_GetCkeNum_When_Normal_Expect_ReturnSuccess)
{
    uint32_t num = 0;
    EXPECT_EQ(CcuResSpecifications::GetInstance(fakeDevLogicId).GetCkeNum(0, num), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(num, 1024U);
}

TEST_F(CcuResSpecificationsTest, Ut_GetXnNum_When_Normal_Expect_ReturnSuccess)
{
    uint32_t num = 0;
    EXPECT_EQ(CcuResSpecifications::GetInstance(fakeDevLogicId).GetXnNum(0, num), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(num, 12288U);
}

TEST_F(CcuResSpecificationsTest, Ut_GetGsaNum_When_Normal_Expect_ReturnSuccess)
{
    uint32_t num = 0;
    EXPECT_EQ(CcuResSpecifications::GetInstance(fakeDevLogicId).GetGsaNum(0, num), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(num, 0U);
}

TEST_F(CcuResSpecificationsTest, Ut_GetMsNum_When_Normal_Expect_ReturnSuccess)
{
    uint32_t num = 0;
    EXPECT_EQ(CcuResSpecifications::GetInstance(fakeDevLogicId).GetMsNum(0, num), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(num, 1536U);
}

TEST_F(CcuResSpecificationsTest, Ut_GetLoopCkeNum_When_Normal_Expect_ReturnSuccess)
{
    uint32_t num = 0;
    EXPECT_EQ(CcuResSpecifications::GetInstance(fakeDevLogicId).GetLoopCkeNum(0, num), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(num, 512U);
}

TEST_F(CcuResSpecificationsTest, Ut_GetCountXnNum_When_Normal_Expect_ReturnSuccess)
{
    uint32_t num = 0;
    EXPECT_EQ(CcuResSpecifications::GetInstance(fakeDevLogicId).GetCountXnNum(0, num), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(num, 4096U);
}

TEST_F(CcuResSpecificationsTest, Ut_GetChannelNum_When_Normal_Expect_ReturnSuccess)
{
    uint32_t num = 0;
    EXPECT_EQ(CcuResSpecifications::GetInstance(fakeDevLogicId).GetChannelNum(0, num), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(num, 1024U);
}

TEST_F(CcuResSpecificationsTest, Ut_GetJettyNum_When_Normal_Expect_ReturnSuccess)
{
    uint32_t num = 0;
    EXPECT_EQ(CcuResSpecifications::GetInstance(fakeDevLogicId).GetJettyNum(0, num), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(num, 128U);
}

TEST_F(CcuResSpecificationsTest, Ut_GetPfeReservedNum_When_V2_Expect_Return20)
{
    uint32_t num = 0;
    EXPECT_EQ(CcuResSpecifications::GetInstance(fakeDevLogicId).GetPfeReservedNum(0, num), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(num, CCU_V2_PER_DIE_PFE_RESERVED_NUM);
}

TEST_F(CcuResSpecificationsTest, Ut_GetPfeReservedNum_When_V1_Expect_Return16)
{
    SetupSpecsV1(1);
    uint32_t num = 0;
    EXPECT_EQ(CcuResSpecifications::GetInstance(1).GetPfeReservedNum(0, num), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(num, CCU_V1_PER_DIE_PFE_RESERVED_NUM);
}

TEST_F(CcuResSpecificationsTest, Ut_GetPfeNum_When_Normal_Expect_ReturnSuccess)
{
    uint32_t num = 0;
    EXPECT_EQ(CcuResSpecifications::GetInstance(fakeDevLogicId).GetPfeNum(0, num), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(num, 20U);
}

TEST_F(CcuResSpecificationsTest, Ut_GetWqeBBNum_When_Normal_Expect_ReturnSuccess)
{
    uint32_t num = 0;
    EXPECT_EQ(CcuResSpecifications::GetInstance(fakeDevLogicId).GetWqeBBNum(0, num), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(num, 4096U);
}

TEST_F(CcuResSpecificationsTest, Ut_GetChannelJettyMap_When_Normal_Expect_ReturnSuccess)
{
    CcuChannelJettyMap map(1, 1);
    EXPECT_EQ(CcuResSpecifications::GetInstance(fakeDevLogicId).GetChannelJettyMap(0, map), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(map.channelNum, 8U);
    EXPECT_EQ(map.jettyNum, 1U);
}

TEST_F(CcuResSpecificationsTest, Ut_Deinit_When_Normal_Expect_ReturnSuccess)
{
    EXPECT_EQ(CcuResSpecifications::GetInstance(fakeDevLogicId).Deinit(), HcclResult::HCCL_SUCCESS);
    EXPECT_FALSE(CcuResSpecifications::GetInstance(fakeDevLogicId).initFlag_);
}
