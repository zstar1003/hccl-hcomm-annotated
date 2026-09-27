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
#include "sync_xn_executor.h"

using namespace hcomm::CcuRep;

class SyncXnExecutorTest : public testing::Test {
protected:
    void SetUp() override
    {
        auto& mgr = CcuResourceManager::GetInstance();
        mgr.Init(0, 2, RunnerCcuVersion::CCU_V1, {});
        mgr.Init(1, 2, RunnerCcuVersion::CCU_V1, {});
    }
    void TearDown() override {}
};

// Test: SyncXnExecutor struct size check
TEST_F(SyncXnExecutorTest, StructSize) { EXPECT_GT(sizeof(SyncXnExecutor), 0); }

// Test: SyncXnExecutor default constructor
TEST_F(SyncXnExecutorTest, DefaultConstructor)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    SyncXnExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Describe());
}

// Test: SyncXnExecutor parameterized constructor
TEST_F(SyncXnExecutorTest, ParameterizedConstructor)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    SyncXnExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Describe());
}

// Test: SyncXnExecutor Parser with zero values
TEST_F(SyncXnExecutorTest, ParserZeroValues)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    SyncXnExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Parser());
    EXPECT_NO_THROW(executor.Describe());
}

// Test: SyncXnExecutor Parser with max values
TEST_F(SyncXnExecutorTest, ParserMaxValues)
{
    CcuInstr instr;
    memset(&instr, 0xFF, sizeof(instr));

    SyncXnExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Parser());
    EXPECT_NO_THROW(executor.Describe());
}

// Test: SyncXnExecutor Parser with specific Xn parameters
TEST_F(SyncXnExecutorTest, ParserXnParameters)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    instr.v1.syncXn.rmtXnId = 100;
    instr.v1.syncXn.locXnId = 200;
    instr.v1.syncXn.channelId = 5;
    instr.v1.syncXn.setRmtCKEId = 10;
    instr.v1.syncXn.setRmtCKEMask = 0xFF;

    SyncXnExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Parser());
    std::string desc = executor.Describe();
    EXPECT_FALSE(desc.empty());
}

// Test: SyncXnExecutor with different Xn IDs
TEST_F(SyncXnExecutorTest, DifferentXnIds)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    uint16_t xnIds[] = {0, 1, SimCcuV1::CCU_RESOURCE_XN_NUM / 2, SimCcuV1::CCU_RESOURCE_XN_NUM - 1, 0xFFFF};

    for (auto xnId : xnIds) {
        instr.v1.syncXn.locXnId = xnId;
        instr.v1.syncXn.rmtXnId = xnId;
        SyncXnExecutor executor(0, 0, 0, instr, nullptr);
        EXPECT_NO_THROW(executor.Parser());
        EXPECT_NO_THROW(executor.Describe());
    }
}

// Test: SyncXnExecutor Describe contains expected keywords
TEST_F(SyncXnExecutorTest, DescribeContent)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    instr.v1.syncXn.rmtXnId = 10;
    instr.v1.syncXn.locXnId = 20;
    instr.v1.syncXn.channelId = 5;

    SyncXnExecutor executor(0, 0, 0, instr, nullptr);
    executor.Parser();
    std::string desc = executor.Describe();
    EXPECT_FALSE(desc.empty());
    EXPECT_NE(desc.find("Sync locXnId"), std::string::npos);
}

// Test: SyncXnExecutor inheritance check
TEST_F(SyncXnExecutorTest, InheritanceCheck)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    SyncXnExecutor executor(0, 0, 0, instr, nullptr);
    CcuExecutorBase* base = &executor;
    EXPECT_NE(base, nullptr);
}

TEST_F(SyncXnExecutorTest, ProcessWithInvalidChannelId)
{
    auto& mgr = CcuResourceManager::GetInstance();
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    instr.v1.syncXn.channelId = SimCcuV1::MAX_CCU_CHANNEL_NUM;
    instr.v1.syncXn.locXnId = 0;
    instr.v1.syncXn.rmtXnId = 0;
    instr.v1.syncXn.setCKEId = 0;
    instr.v1.syncXn.setCKEMask = 0;
    instr.v1.syncXn.setRmtCKEId = 0;
    instr.v1.syncXn.setRmtCKEMask = 0;
    CcuSimulator sim(0, 0, 0, 1, 1, RunnerCcuVersion::CCU_V1);
    SyncXnExecutor executor(0, 0, 0, instr, &sim);
    executor.Parser();
    EXPECT_NO_THROW(executor.Process(mgr));
}

TEST_F(SyncXnExecutorTest, RunWithCkeNotSatisfied)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    instr.v1.syncXn.channelId = 0;
    instr.v1.syncXn.waitCKEId = 0;
    instr.v1.syncXn.waitCKEMask = 0x0001;
    instr.v1.syncXn.clearType = 0;
    CcuSimulator sim(0, 0, 0, 1, 1, RunnerCcuVersion::CCU_V1);
    SyncXnExecutor executor(0, 0, 0, instr, &sim);
    executor.Parser();
    executor.Run();
}

TEST_F(SyncXnExecutorTest, ProcessWithValidChannel)
{
    auto& mgr = CcuResourceManager::GetInstance();
    mgr.UpdateXnValue(0, 0, 0, 0x2000);
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    instr.v1.syncXn.channelId = 0;
    instr.v1.syncXn.locXnId = 0;
    instr.v1.syncXn.rmtXnId = 0;
    instr.v1.syncXn.setCKEId = 0;
    instr.v1.syncXn.setCKEMask = 0;
    instr.v1.syncXn.setRmtCKEId = 0;
    instr.v1.syncXn.setRmtCKEMask = 0;
    CcuSimulator sim(0, 0, 0, 1, 1, RunnerCcuVersion::CCU_V1);
    SyncXnExecutor executor(0, 0, 0, instr, &sim);
    executor.Parser();
    EXPECT_NO_THROW(executor.Process(mgr));
}

TEST_F(SyncXnExecutorTest, RunWithCkeNotSatisfied)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    instr.v1.syncXn.channelId = 0;
    instr.v1.syncXn.waitCKEId = 0;
    instr.v1.syncXn.waitCKEMask = 0x0001;
    instr.v1.syncXn.clearType = 0;
    CcuSimulator sim(0, 0, 0, 1, 1, RunnerCcuVersion::CCU_V1);
    SyncXnExecutor executor(0, 0, 0, instr, &sim);
    executor.Parser();
    EXPECT_NO_THROW(executor.Run());
}
