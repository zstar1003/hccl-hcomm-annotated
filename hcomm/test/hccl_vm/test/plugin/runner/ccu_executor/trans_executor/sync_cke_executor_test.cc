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

#include "ccu_executor_base.h"
#include "ccu_microcode_common_v1.h"
#include "ccu_resource_manager.h"
#include "ccu_simulator.h"
#include "sync_cke_executor.h"

using namespace hcomm::CcuRep;

class SyncCkeExecutorTest : public testing::Test {
protected:
    void SetUp() override
    {
        auto& mgr = CcuResourceManager::GetInstance();
        mgr.Init(0, 2, RunnerCcuVersion::CCU_V1, {});
        mgr.Init(1, 2, RunnerCcuVersion::CCU_V1, {});
    }
    void TearDown() override {}
};

// Test: SyncCkeExecutor struct size check
TEST_F(SyncCkeExecutorTest, StructSize) { EXPECT_GT(sizeof(SyncCkeExecutor), 0); }

// Test: SyncCkeExecutor default constructor
TEST_F(SyncCkeExecutorTest, DefaultConstructor)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    SyncCkeExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Describe());
}

// Test: SyncCkeExecutor parameterized constructor
TEST_F(SyncCkeExecutorTest, ParameterizedConstructor)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    SyncCkeExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Describe());
}

// Test: SyncCkeExecutor Parser with zero values
TEST_F(SyncCkeExecutorTest, ParserZeroValues)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    SyncCkeExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Parser());
    EXPECT_NO_THROW(executor.Describe());
}

// Test: SyncCkeExecutor Parser with max values
TEST_F(SyncCkeExecutorTest, ParserMaxValues)
{
    CcuInstr instr;
    memset(&instr, 0xFF, sizeof(instr));

    SyncCkeExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Parser());
    EXPECT_NO_THROW(executor.Describe());
}

// Test: SyncCkeExecutor Parser with specific sync parameters
TEST_F(SyncCkeExecutorTest, ParserSyncParameters)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    instr.v1.syncCKE.rmtCKEId = 10;
    instr.v1.syncCKE.locCKEId = 20;
    instr.v1.syncCKE.locCKEMask = 0xFF00;
    instr.v1.syncCKE.channelId = 5;
    instr.v1.syncCKE.clearType = 1;

    SyncCkeExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Parser());
    std::string desc = executor.Describe();
    EXPECT_FALSE(desc.empty());
}

// Test: SyncCkeExecutor with different channel IDs
TEST_F(SyncCkeExecutorTest, DifferentChannelIds)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    uint16_t channels[] = {0, 1, 64, 127, SimCcuV1::MAX_CCU_CHANNEL_NUM - 1};

    for (auto ch : channels) {
        instr.v1.syncCKE.channelId = ch;
        SyncCkeExecutor executor(0, 0, 0, instr, nullptr);
        EXPECT_NO_THROW(executor.Parser());
        EXPECT_NO_THROW(executor.Describe());
    }
}

// Test: SyncCkeExecutor Describe contains expected keywords
TEST_F(SyncCkeExecutorTest, DescribeContent)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    instr.v1.syncCKE.rmtCKEId = 10;
    instr.v1.syncCKE.locCKEId = 20;
    instr.v1.syncCKE.channelId = 5;

    SyncCkeExecutor executor(0, 0, 0, instr, nullptr);
    executor.Parser();
    std::string desc = executor.Describe();
    EXPECT_FALSE(desc.empty());
    EXPECT_NE(desc.find("Sync LocCKE"), std::string::npos);
}

// Test: SyncCkeExecutor inheritance check
TEST_F(SyncCkeExecutorTest, InheritanceCheck)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    SyncCkeExecutor executor(0, 0, 0, instr, nullptr);
    CcuExecutorBase* base = &executor;
    EXPECT_NE(base, nullptr);
}

// Test: SyncCkeExecutor with various local and remote CKE combinations
TEST_F(SyncCkeExecutorTest, VariousCkeCombinations)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    uint16_t ckeIds[] = {0, 100, 0x7FFF, 0xFFFF};

    for (auto locId : ckeIds) {
        for (auto rmtId : ckeIds) {
            instr.v1.syncCKE.locCKEId = locId;
            instr.v1.syncCKE.rmtCKEId = rmtId;
            SyncCkeExecutor executor(0, 0, 0, instr, nullptr);
            EXPECT_NO_THROW(executor.Parser());
            EXPECT_NO_THROW(executor.Describe());
        }
    }
}

TEST_F(SyncCkeExecutorTest, ProcessWithValidChannel)
{
    auto& mgr = CcuResourceManager::GetInstance();
    mgr.UpdateCkeValue(0, 0, 0, 0x00FF);
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    instr.v1.syncCKE.channelId = 0;
    instr.v1.syncCKE.locCKEId = 0;
    instr.v1.syncCKE.rmtCKEId = 0;
    instr.v1.syncCKE.locCKEMask = 0xFFFF;
    instr.v1.syncCKE.setCKEId = 0;
    instr.v1.syncCKE.setCKEMask = 0;
    CcuSimulator sim(0, 0, 0, 1, 1, RunnerCcuVersion::CCU_V1);
    SyncCkeExecutor executor(0, 0, 0, instr, &sim);
    executor.Parser();
    EXPECT_NO_THROW(executor.Process(mgr));
}

TEST_F(SyncCkeExecutorTest, RunWithCkeNotSatisfied)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    instr.v1.syncCKE.channelId = 0;
    instr.v1.syncCKE.waitCKEId = 0;
    instr.v1.syncCKE.waitCKEMask = 0x0001;
    instr.v1.syncCKE.clearType = 0;
    CcuSimulator sim(0, 0, 0, 1, 1, RunnerCcuVersion::CCU_V1);
    SyncCkeExecutor executor(0, 0, 0, instr, &sim);
    executor.Parser();
    EXPECT_NO_THROW(executor.Run());
}
}

TEST_F(SyncCkeExecutorTest, ProcessWithInvalidChannelId)
{
    auto& mgr = CcuResourceManager::GetInstance();
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    instr.v1.syncCKE.channelId = SimCcuV1::MAX_CCU_CHANNEL_NUM;
    instr.v1.syncCKE.locCKEId = 0;
    instr.v1.syncCKE.rmtCKEId = 0;
    instr.v1.syncCKE.setCKEId = 0;
    instr.v1.syncCKE.setCKEMask = 0;
    instr.v1.syncCKE.setRmtCKEId = 0;
    instr.v1.syncCKE.setRmtCKEMask = 0;
    CcuSimulator sim(0, 0, 0, 1, 1, RunnerCcuVersion::CCU_V1);
    SyncCkeExecutor executor(0, 0, 0, instr, &sim);
    executor.Parser();
    EXPECT_NO_THROW(executor.Process(mgr));
}

TEST_F(SyncCkeExecutorTest, RunWithCkeNotSatisfied)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    instr.v1.syncCKE.channelId = 0;
    instr.v1.syncCKE.waitCKEId = 0;
    instr.v1.syncCKE.waitCKEMask = 0x0001;
    instr.v1.syncCKE.clearType = 0;
    CcuSimulator sim(0, 0, 0, 1, 1, RunnerCcuVersion::CCU_V1);
    SyncCkeExecutor executor(0, 0, 0, instr, &sim);
    executor.Parser();
    executor.Run();
}
}
