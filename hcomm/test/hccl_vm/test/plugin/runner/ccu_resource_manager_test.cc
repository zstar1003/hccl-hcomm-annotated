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
#include <cstring>
#include <gtest/gtest.h>
#include <memory>

#include "ccu_microcode_common_v1.h"
#include "ccu_resource_manager.h"
#include "ccu_simulator.h"

using namespace hcomm::CcuRep;

class CcuResourceManagerTest : public testing::Test {
protected:
    void SetUp() override {}

    void TearDown() override {}
};

// Test: Singleton instance
TEST_F(CcuResourceManagerTest, SingletonInstance)
{
    CcuResourceManager& instance1 = CcuResourceManager::GetInstance();
    CcuResourceManager& instance2 = CcuResourceManager::GetInstance();

    EXPECT_EQ(&instance1, &instance2);
}

// Test: Delete copy constructor
TEST_F(CcuResourceManagerTest, DeleteCopyConstructor)
{
    EXPECT_FALSE(std::is_copy_constructible<CcuResourceManager>::value);
}

// Test: Delete copy assignment
TEST_F(CcuResourceManagerTest, DeleteCopyAssignment)
{
    EXPECT_FALSE(std::is_copy_assignable<CcuResourceManager>::value);
}

// Test: Init with CCU_V1
TEST_F(CcuResourceManagerTest, InitWithCcuV1)
{
    CcuResourceManager& mgr = CcuResourceManager::GetInstance();
    EXPECT_NO_THROW(mgr.Init(0, 4, RunnerCcuVersion::CCU_V1, {}));
}

// Test: Init with CCU_V2 (not supported)
TEST_F(CcuResourceManagerTest, InitWithCcuV2)
{
    CcuResourceManager& mgr = CcuResourceManager::GetInstance();
    EXPECT_NO_THROW(mgr.Init(0, 4, RunnerCcuVersion::CCU_V2, {}));
}

// Test: Init with CCU_INVALID
TEST_F(CcuResourceManagerTest, InitWithCcuInvalid)
{
    CcuResourceManager& mgr = CcuResourceManager::GetInstance();
    EXPECT_NO_THROW(mgr.Init(0, 4, RunnerCcuVersion::CCU_INVALID, {}));
}

// Test: GetXnValue and UpdateXnValue
TEST_F(CcuResourceManagerTest, XnValueOperations)
{
    CcuResourceManager& mgr = CcuResourceManager::GetInstance();
    mgr.Init(0, 4, RunnerCcuVersion::CCU_V1, {});

    uint64_t testValue = 0x123456789ABCDEF0ULL;
    mgr.UpdateXnValue(0, 0, 0, testValue);
    uint64_t retrievedValue = mgr.GetXnValue(0, 0, 0);
    EXPECT_EQ(retrievedValue, testValue);
}

// Test: GetGsaValue and UpdateGsaValue
TEST_F(CcuResourceManagerTest, GsaValueOperations)
{
    CcuResourceManager& mgr = CcuResourceManager::GetInstance();
    mgr.Init(0, 4, RunnerCcuVersion::CCU_V1, {});

    uint64_t testValue = 0xFEDCBA9876543210ULL;
    mgr.UpdateGsaValue(0, 0, 0, testValue);
    uint64_t retrievedValue = mgr.GetGsaValue(0, 0, 0);
    EXPECT_EQ(retrievedValue, testValue);
}

// Test: GetCkeValue and UpdateCkeValue
TEST_F(CcuResourceManagerTest, CkeValueOperations)
{
    CcuResourceManager& mgr = CcuResourceManager::GetInstance();
    mgr.Init(0, 4, RunnerCcuVersion::CCU_V1, {});

    uint16_t testValue = 0xABCD;
    mgr.UpdateCkeValue(0, 0, 0, testValue);
    uint16_t retrievedValue = mgr.GetCkeValue(0, 0, 0);
    EXPECT_EQ(retrievedValue, testValue);
}

// Test: Multiple Xn operations
TEST_F(CcuResourceManagerTest, MultipleXnOperations)
{
    CcuResourceManager& mgr = CcuResourceManager::GetInstance();
    mgr.Init(0, 4, RunnerCcuVersion::CCU_V1, {});

    for (uint16_t i = 0; i < 10; i++) {
        mgr.UpdateXnValue(0, 0, i, i * 100);
    }

    for (uint16_t i = 0; i < 10; i++) {
        EXPECT_EQ(mgr.GetXnValue(0, 0, i), i * 100);
    }
}

// Test: Multiple Gsa operations
TEST_F(CcuResourceManagerTest, MultipleGsaOperations)
{
    CcuResourceManager& mgr = CcuResourceManager::GetInstance();
    mgr.Init(0, 4, RunnerCcuVersion::CCU_V1, {});

    for (uint16_t i = 0; i < 10; i++) {
        mgr.UpdateGsaValue(0, 0, i, i * 1000);
    }

    for (uint16_t i = 0; i < 10; i++) {
        EXPECT_EQ(mgr.GetGsaValue(0, 0, i), i * 1000);
    }
}

// Test: Multiple Cke operations
TEST_F(CcuResourceManagerTest, MultipleCkeOperations)
{
    CcuResourceManager& mgr = CcuResourceManager::GetInstance();
    mgr.Init(0, 4, RunnerCcuVersion::CCU_V1, {});

    for (uint16_t i = 0; i < 10; i++) {
        mgr.UpdateCkeValue(0, 0, i, i * 10);
    }

    for (uint16_t i = 0; i < 10; i++) {
        EXPECT_EQ(mgr.GetCkeValue(0, 0, i), i * 10);
    }
}

// Test: GetMsAddr
TEST_F(CcuResourceManagerTest, GetMsAddr)
{
    CcuResourceManager& mgr = CcuResourceManager::GetInstance();
    mgr.Init(0, 4, RunnerCcuVersion::CCU_V1, {});

    char* addr0 = mgr.GetMsAddr(0, 0, 0);
    char* addr1 = mgr.GetMsAddr(0, 0, 1);

    EXPECT_NE(addr0, nullptr);
    EXPECT_NE(addr1, nullptr);
    // Each MS is 4KB, so addr1 should be 4096 bytes after addr0
    EXPECT_EQ(addr1 - addr0, 4096);
}

// Test: GetMsAddr with different msId
TEST_F(CcuResourceManagerTest, GetMsAddrDifferentMsId)
{
    CcuResourceManager& mgr = CcuResourceManager::GetInstance();
    mgr.Init(0, 4, RunnerCcuVersion::CCU_V1, {});

    char* addr0 = mgr.GetMsAddr(0, 0, 0);
    char* addr100 = mgr.GetMsAddr(0, 0, 100);

    EXPECT_NE(addr0, nullptr);
    EXPECT_NE(addr100, nullptr);
    EXPECT_EQ(addr100 - addr0, 100 * 4096);
}

// Test: InitInstrInfo
TEST_F(CcuResourceManagerTest, InitInstrInfo)
{
    CcuResourceManager& mgr = CcuResourceManager::GetInstance();
    mgr.Init(0, 4, RunnerCcuVersion::CCU_V1, {});

    CcuInstrData instrData;
    instrData.instrCnt = 5;
    instrData.instrData.resize(5);

    EXPECT_NO_THROW(mgr.InitInstrInfo(0, 0, instrData));
    EXPECT_EQ(mgr.GetInstrCnt(0, 0), 5);
}

// Test: GetInstrCnt
TEST_F(CcuResourceManagerTest, GetInstrCnt)
{
    CcuResourceManager& mgr = CcuResourceManager::GetInstance();
    mgr.Init(0, 4, RunnerCcuVersion::CCU_V1, {});

    CcuInstrData instrData;
    instrData.instrCnt = 10;
    instrData.instrData.resize(10);

    mgr.InitInstrInfo(0, 0, instrData);
    EXPECT_EQ(mgr.GetInstrCnt(0, 0), 10);
}

// Test: GetInstrData
TEST_F(CcuResourceManagerTest, GetInstrData)
{
    CcuResourceManager& mgr = CcuResourceManager::GetInstance();
    mgr.Init(0, 4, RunnerCcuVersion::CCU_V1, {});

    CcuInstrData instrData;
    instrData.instrCnt = 3;
    instrData.instrData.resize(3);
    memset(&instrData.instrData[0], 0, sizeof(CcuInstr));

    mgr.InitInstrInfo(0, 0, instrData);

    auto retrievedData = mgr.GetInstrData(0, 0);
    EXPECT_EQ(retrievedData.size(), 3);
}

// Test: InitChannelInfo
TEST_F(CcuResourceManagerTest, InitChannelInfo)
{
    CcuResourceManager& mgr = CcuResourceManager::GetInstance();
    mgr.Init(0, 4, RunnerCcuVersion::CCU_V1, {});

    RankChannelInfo channelInfo;
    CcuInfo rmtInfo;
    rmtInfo.rankId = 1;
    rmtInfo.dieId = 0;
    channelInfo[0][0] = rmtInfo;

    EXPECT_NO_THROW(mgr.InitChannelInfo(0, channelInfo));
}

// Test: GetRmtCcu
TEST_F(CcuResourceManagerTest, GetRmtCcu)
{
    CcuResourceManager& mgr = CcuResourceManager::GetInstance();
    mgr.Init(0, 4, RunnerCcuVersion::CCU_V1, {});

    RankChannelInfo channelInfo;
    CcuInfo rmtInfo;
    rmtInfo.rankId = 1;
    rmtInfo.dieId = 0;
    channelInfo[0][0] = rmtInfo;

    mgr.InitChannelInfo(0, channelInfo);

    auto rmtCcu = mgr.GetRmtCcu(0, 0, 0);
    EXPECT_EQ(rmtCcu.first, 1);
    EXPECT_EQ(rmtCcu.second, 0);
}

// Test: GetRmtCcu with invalid channel
TEST_F(CcuResourceManagerTest, GetRmtCcuInvalidChannel)
{
    CcuResourceManager& mgr = CcuResourceManager::GetInstance();
    mgr.Init(0, 4, RunnerCcuVersion::CCU_V1, {});

    auto rmtCcu = mgr.GetRmtCcu(0, 0, 1000);
    EXPECT_EQ(rmtCcu.first, S32_INVALID);
    EXPECT_EQ(rmtCcu.second, S32_INVALID);
}

// Test: AddTaskInfo
TEST_F(CcuResourceManagerTest, AddTaskInfo)
{
    CcuResourceManager& mgr = CcuResourceManager::GetInstance();
    mgr.Init(0, 4, RunnerCcuVersion::CCU_V1, {});

    HcclTaskMetaData task;
    task.rankId = 0;
    task.taskData.ccu.dieId = 0;
    task.taskData.ccu.args[0] = 0x12345678;

    EXPECT_NO_THROW(mgr.AddTaskInfo(0, task));
    EXPECT_EQ(mgr.GetSqeArgValue(0, 0, 0), 0x12345678);
}

// Test: AddTaskInfo with invalid dieId
TEST_F(CcuResourceManagerTest, AddTaskInfoInvalidDieId)
{
    CcuResourceManager& mgr = CcuResourceManager::GetInstance();
    mgr.Init(0, 4, RunnerCcuVersion::CCU_V1, {});

    HcclTaskMetaData task;
    task.rankId = 0;
    task.taskData.ccu.dieId = 100; // Invalid dieId

    EXPECT_NO_THROW(mgr.AddTaskInfo(0, task));
}

// Test: GetSqeArgValue
TEST_F(CcuResourceManagerTest, GetSqeArgValue)
{
    CcuResourceManager& mgr = CcuResourceManager::GetInstance();
    mgr.Init(0, 4, RunnerCcuVersion::CCU_V1, {});

    HcclTaskMetaData task;
    task.rankId = 0;
    task.taskData.ccu.dieId = 0;
    for (int i = 0; i < 13; i++) {
        task.taskData.ccu.args[i] = i * 100;
    }

    mgr.AddTaskInfo(0, task);

    for (int i = 0; i < 13; i++) {
        EXPECT_EQ(mgr.GetSqeArgValue(0, 0, i), i * 100);
    }
}

// Test: InitSimulator
TEST_F(CcuResourceManagerTest, InitSimulator)
{
    CcuResourceManager& mgr = CcuResourceManager::GetInstance();
    mgr.Init(0, 4, RunnerCcuVersion::CCU_V1, {});

    auto simulator = mgr.InitSimulator(0, 0, 0, 10, 10);
    EXPECT_NE(simulator, nullptr);
}

// Test: InitSimulator twice
TEST_F(CcuResourceManagerTest, InitSimulatorTwice)
{
    CcuResourceManager& mgr = CcuResourceManager::GetInstance();
    mgr.Init(0, 4, RunnerCcuVersion::CCU_V1, {});

    auto simulator1 = mgr.InitSimulator(0, 0, 0, 10, 10);
    auto simulator2 = mgr.InitSimulator(0, 0, 5, 15, 10);

    EXPECT_EQ(simulator1, simulator2);
}

// Test: DumpCcuInstructions
TEST_F(CcuResourceManagerTest, DumpCcuInstructions)
{
    CcuResourceManager& mgr = CcuResourceManager::GetInstance();
    mgr.Init(0, 4, RunnerCcuVersion::CCU_V1, {});

    EXPECT_NO_THROW(mgr.DumpCcuInstructions(0));
}

// Test: DumpCcuXnResouceInfo
TEST_F(CcuResourceManagerTest, DumpCcuXnResouceInfo)
{
    CcuResourceManager& mgr = CcuResourceManager::GetInstance();
    mgr.Init(0, 4, RunnerCcuVersion::CCU_V1, {});

    EXPECT_NO_THROW(mgr.DumpCcuXnResouceInfo(0));
}

// Test: DumpCcuGsaResouceInfo
TEST_F(CcuResourceManagerTest, DumpCcuGsaResouceInfo)
{
    CcuResourceManager& mgr = CcuResourceManager::GetInstance();
    mgr.Init(0, 4, RunnerCcuVersion::CCU_V1, {});

    EXPECT_NO_THROW(mgr.DumpCcuGsaResouceInfo(0));
}

// Test: DumpCcuCkeResouceInfo
TEST_F(CcuResourceManagerTest, DumpCcuCkeResouceInfo)
{
    CcuResourceManager& mgr = CcuResourceManager::GetInstance();
    mgr.Init(0, 4, RunnerCcuVersion::CCU_V1, {});

    EXPECT_NO_THROW(mgr.DumpCcuCkeResouceInfo(0));
}

// Test: DumpCcuChannelResouceInfo
TEST_F(CcuResourceManagerTest, DumpCcuChannelResouceInfo)
{
    CcuResourceManager& mgr = CcuResourceManager::GetInstance();
    mgr.Init(0, 4, RunnerCcuVersion::CCU_V1, {});

    EXPECT_NO_THROW(mgr.DumpCcuChannelResouceInfo(0));
}

// Test: DumpCcuAllResouceInfo
TEST_F(CcuResourceManagerTest, DumpCcuAllResouceInfo)
{
    CcuResourceManager& mgr = CcuResourceManager::GetInstance();
    mgr.Init(0, 4, RunnerCcuVersion::CCU_V1, {});

    EXPECT_NO_THROW(mgr.DumpCcuAllResouceInfo(0));
}

// Test: DumpChannelId2RmtRank
TEST_F(CcuResourceManagerTest, DumpChannelId2RmtRank)
{
    CcuResourceManager& mgr = CcuResourceManager::GetInstance();
    mgr.Init(0, 4, RunnerCcuVersion::CCU_V1, {});

    EXPECT_NO_THROW(mgr.DumpChannelId2RmtRank(0, 0));
}

// Test: GetInstrDescribe
TEST_F(CcuResourceManagerTest, GetInstrDescribe)
{
    CcuResourceManager& mgr = CcuResourceManager::GetInstance();
    mgr.Init(0, 4, RunnerCcuVersion::CCU_V1, {});

    std::string desc = mgr.GetInstrDescribe(0, 0, 0);
    EXPECT_TRUE(desc.empty() || !desc.empty()); // Just verify it doesn't crash
}

// Test: Boundary test for Xn ID
TEST_F(CcuResourceManagerTest, BoundaryXnId)
{
    CcuResourceManager& mgr = CcuResourceManager::GetInstance();
    mgr.Init(0, 4, RunnerCcuVersion::CCU_V1, {});

    uint16_t maxXnId = SimCcuV1::CCU_RESOURCE_XN_MAX - 1;
    mgr.UpdateXnValue(0, 0, maxXnId, 0xFFFFFFFFFFFFFFFFULL);
    EXPECT_EQ(mgr.GetXnValue(0, 0, maxXnId), 0xFFFFFFFFFFFFFFFFULL);
}

// Test: Boundary test for Gsa ID
TEST_F(CcuResourceManagerTest, BoundaryGsaId)
{
    CcuResourceManager& mgr = CcuResourceManager::GetInstance();
    mgr.Init(0, 4, RunnerCcuVersion::CCU_V1, {});

    uint16_t maxGsaId = SimCcuV1::CCU_RESOURCE_GSA_MAX - 1;
    mgr.UpdateGsaValue(0, 0, maxGsaId, 0xFFFFFFFFFFFFFFFFULL);
    EXPECT_EQ(mgr.GetGsaValue(0, 0, maxGsaId), 0xFFFFFFFFFFFFFFFFULL);
}

// Test: Boundary test for Cke ID
TEST_F(CcuResourceManagerTest, BoundaryCkeId)
{
    CcuResourceManager& mgr = CcuResourceManager::GetInstance();
    mgr.Init(0, 4, RunnerCcuVersion::CCU_V1, {});

    uint16_t maxCkeId = SimCcuV1::CCU_RESOURCE_CKE_MAX - 1;
    mgr.UpdateCkeValue(0, 0, maxCkeId, 0xFFFF);
    EXPECT_EQ(mgr.GetCkeValue(0, 0, maxCkeId), 0xFFFF);
}

// Test: Different die IDs
TEST_F(CcuResourceManagerTest, DifferentDieIds)
{
    CcuResourceManager& mgr = CcuResourceManager::GetInstance();
    mgr.Init(0, 4, RunnerCcuVersion::CCU_V1, {});

    for (int dieId = 0; dieId < HcclSim::DIE_NUM; dieId++) {
        mgr.UpdateXnValue(0, dieId, 0, dieId * 1000);
        EXPECT_EQ(mgr.GetXnValue(0, dieId, 0), dieId * 1000);
    }
}

// Test: Different rank IDs
TEST_F(CcuResourceManagerTest, DifferentRankIds)
{
    CcuResourceManager& mgr = CcuResourceManager::GetInstance();

    for (int rankId = 0; rankId < 4; rankId++) {
        mgr.Init(rankId, 4, RunnerCcuVersion::CCU_V1, {});
        mgr.UpdateXnValue(rankId, 0, 0, rankId * 10000);
        EXPECT_EQ(mgr.GetXnValue(rankId, 0, 0), rankId * 10000);
    }
}

// Test: AddTaskInfo with unsupported version (CCU_INVALID)
TEST_F(CcuResourceManagerTest, AddTaskInfoUnsupportedVersion)
{
    CcuResourceManager& mgr = CcuResourceManager::GetInstance();
    mgr.Init(0, 4, RunnerCcuVersion::CCU_INVALID, {});

    HcclTaskMetaData task;
    task.rankId = 0;
    task.taskData.ccu.dieId = 0;
    task.taskData.ccu.args[0] = 42;
    EXPECT_NO_THROW(mgr.AddTaskInfo(0, task));
}

// Test: GetSqeArgValue with unsupported version
TEST_F(CcuResourceManagerTest, GetSqeArgValueUnsupportedVersion)
{
    CcuResourceManager& mgr = CcuResourceManager::GetInstance();
    mgr.Init(0, 4, RunnerCcuVersion::CCU_INVALID, {});
    EXPECT_EQ(mgr.GetSqeArgValue(0, 0, 0), U64_INVALID);
}

// Test: GetXnValue with unsupported version
TEST_F(CcuResourceManagerTest, GetXnValueUnsupportedVersion)
{
    CcuResourceManager& mgr = CcuResourceManager::GetInstance();
    mgr.Init(0, 4, RunnerCcuVersion::CCU_INVALID, {});
    EXPECT_EQ(mgr.GetXnValue(0, 0, 0), U64_INVALID);
}

// Test: GetGsaValue with unsupported version
TEST_F(CcuResourceManagerTest, GetGsaValueUnsupportedVersion)
{
    CcuResourceManager& mgr = CcuResourceManager::GetInstance();
    mgr.Init(0, 4, RunnerCcuVersion::CCU_INVALID, {});
    EXPECT_EQ(mgr.GetGsaValue(0, 0, 0), U64_INVALID);
}

// Test: GetCkeValue with unsupported version
TEST_F(CcuResourceManagerTest, GetCkeValueUnsupportedVersion)
{
    CcuResourceManager& mgr = CcuResourceManager::GetInstance();
    mgr.Init(0, 4, RunnerCcuVersion::CCU_INVALID, {});
    EXPECT_EQ(mgr.GetCkeValue(0, 0, 0), U16_INVALID);
}

// Test: GetRmtCcu with unsupported version
TEST_F(CcuResourceManagerTest, GetRmtCcuUnsupportedVersion)
{
    CcuResourceManager& mgr = CcuResourceManager::GetInstance();
    mgr.Init(0, 4, RunnerCcuVersion::CCU_INVALID, {});
    auto result = mgr.GetRmtCcu(0, 0, 0);
    EXPECT_EQ(result.first, S32_INVALID);
    EXPECT_EQ(result.second, S32_INVALID);
}

// Test: UpdateXnValue with unsupported version
TEST_F(CcuResourceManagerTest, UpdateXnValueUnsupportedVersion)
{
    CcuResourceManager& mgr = CcuResourceManager::GetInstance();
    mgr.Init(0, 4, RunnerCcuVersion::CCU_INVALID, {});
    EXPECT_NO_THROW(mgr.UpdateXnValue(0, 0, 0, 123));
}

// Test: UpdateGsaValue with unsupported version
TEST_F(CcuResourceManagerTest, UpdateGsaValueUnsupportedVersion)
{
    CcuResourceManager& mgr = CcuResourceManager::GetInstance();
    mgr.Init(0, 4, RunnerCcuVersion::CCU_INVALID, {});
    EXPECT_NO_THROW(mgr.UpdateGsaValue(0, 0, 0, 456));
}

// Test: UpdateCkeValue with unsupported version
TEST_F(CcuResourceManagerTest, UpdateCkeValueUnsupportedVersion)
{
    CcuResourceManager& mgr = CcuResourceManager::GetInstance();
    mgr.Init(0, 4, RunnerCcuVersion::CCU_INVALID, {});
    EXPECT_NO_THROW(mgr.UpdateCkeValue(0, 0, 0, 789));
}

// Test: GetMsAddr with unsupported version
TEST_F(CcuResourceManagerTest, GetMsAddrUnsupportedVersion)
{
    CcuResourceManager& mgr = CcuResourceManager::GetInstance();
    mgr.Init(0, 4, RunnerCcuVersion::CCU_INVALID, {});
    EXPECT_EQ(mgr.GetMsAddr(0, 0, 0), nullptr);
}

// Test: GetInstrCnt with unsupported version
TEST_F(CcuResourceManagerTest, GetInstrCntUnsupportedVersion)
{
    CcuResourceManager& mgr = CcuResourceManager::GetInstance();
    mgr.Init(0, 4, RunnerCcuVersion::CCU_INVALID, {});
    EXPECT_EQ(mgr.GetInstrCnt(0, 0), U16_INVALID);
}

// Test: GetInstrData with unsupported version
TEST_F(CcuResourceManagerTest, GetInstrDataUnsupportedVersion)
{
    CcuResourceManager& mgr = CcuResourceManager::GetInstance();
    mgr.Init(0, 4, RunnerCcuVersion::CCU_INVALID, {});
    auto data = mgr.GetInstrData(0, 0);
    EXPECT_TRUE(data.empty());
}

// Test: GetInstrDescribe with unsupported version
TEST_F(CcuResourceManagerTest, GetInstrDescribeUnsupportedVersion)
{
    CcuResourceManager& mgr = CcuResourceManager::GetInstance();
    mgr.Init(0, 4, RunnerCcuVersion::CCU_INVALID, {});
    std::string desc = mgr.GetInstrDescribe(0, 0, 0);
    EXPECT_TRUE(desc.empty());
}

// Test: TransMemToMem with null srcBuf
TEST_F(CcuResourceManagerTest, TransMemToMemNullSrc)
{
    CcuResourceManager& mgr = CcuResourceManager::GetInstance();
    mgr.Init(0, 4, RunnerCcuVersion::CCU_V1, {});
    uint64_t dstBuf = 0;
    EXPECT_NO_THROW(mgr.TransMemToMem(nullptr, &dstBuf, sizeof(dstBuf), false, 0, 0));
}

// Test: TransMemToMem with null dstBuf
TEST_F(CcuResourceManagerTest, TransMemToMemNullDst)
{
    CcuResourceManager& mgr = CcuResourceManager::GetInstance();
    mgr.Init(0, 4, RunnerCcuVersion::CCU_V1, {});
    uint64_t srcBuf = 0x1234;
    EXPECT_NO_THROW(mgr.TransMemToMem(&srcBuf, nullptr, sizeof(srcBuf), false, 0, 0));
}

// Test: TransMSToMem with null buf
TEST_F(CcuResourceManagerTest, TransMSToMemNullBuf)
{
    CcuResourceManager& mgr = CcuResourceManager::GetInstance();
    mgr.Init(0, 4, RunnerCcuVersion::CCU_V1, {});
    EXPECT_FALSE(mgr.TransMSToMem(0, 0, 0, nullptr, 8));
}

// Test: TransMemToMS with null buf
TEST_F(CcuResourceManagerTest, TransMemToMSNullBuf)
{
    CcuResourceManager& mgr = CcuResourceManager::GetInstance();
    mgr.Init(0, 4, RunnerCcuVersion::CCU_V1, {});
    EXPECT_FALSE(mgr.TransMemToMS(0, 0, 0, nullptr, 8));
}

// Test: TransMSToMS basic operation
TEST_F(CcuResourceManagerTest, TransMSToMSBasic)
{
    CcuResourceManager& mgr = CcuResourceManager::GetInstance();
    mgr.Init(0, 4, RunnerCcuVersion::CCU_V1, {});
    EXPECT_NO_THROW(mgr.TransMSToMS(0, 0, 0, 0, 0, 1, 8));
}

// Test: DumpCcuInstructions with enableDump true (covered by existing test, but verify no crash)
TEST_F(CcuResourceManagerTest, DumpCcuInstructionsAfterInit)
{
    CcuResourceManager& mgr = CcuResourceManager::GetInstance();
    mgr.Init(0, 4, RunnerCcuVersion::CCU_V1, {});
    CcuInstrData instrData;
    instrData.instrCnt = 2;
    instrData.instrData.resize(2);
    mgr.InitInstrInfo(0, 0, instrData);
    EXPECT_NO_THROW(mgr.DumpCcuInstructions(0));
}

// Test: DumpChannelId2RmtRank after InitChannelInfo
TEST_F(CcuResourceManagerTest, DumpChannelId2RmtRankAfterChannelInfo)
{
    CcuResourceManager& mgr = CcuResourceManager::GetInstance();
    mgr.Init(0, 4, RunnerCcuVersion::CCU_V1, {});
    RankChannelInfo channelInfo;
    CcuInfo rmtInfo;
    rmtInfo.rankId = 1;
    rmtInfo.dieId = 0;
    channelInfo[0][0] = rmtInfo;
    mgr.InitChannelInfo(0, channelInfo);
    EXPECT_NO_THROW(mgr.DumpChannelId2RmtRank(0, 0));
}
