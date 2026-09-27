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
#include "sync_gsa_executor.h"

using namespace hcomm::CcuRep;

class SyncGsaExecutorTest : public testing::Test {
protected:
    void SetUp() override
    {
        auto& mgr = CcuResourceManager::GetInstance();
        mgr.Init(0, 2, RunnerCcuVersion::CCU_V1, {});
        mgr.Init(1, 2, RunnerCcuVersion::CCU_V1, {});
    }
    void TearDown() override {}
};

// Test: SyncGsaExecutor struct size check
TEST_F(SyncGsaExecutorTest, StructSize) { EXPECT_GT(sizeof(SyncGsaExecutor), 0); }

// Test: SyncGsaExecutor default constructor
TEST_F(SyncGsaExecutorTest, DefaultConstructor)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    SyncGsaExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Describe());
}

// Test: SyncGsaExecutor parameterized constructor
TEST_F(SyncGsaExecutorTest, ParameterizedConstructor)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    SyncGsaExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Describe());
}

// Test: SyncGsaExecutor Parser with zero values
TEST_F(SyncGsaExecutorTest, ParserZeroValues)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    SyncGsaExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Parser());
    EXPECT_NO_THROW(executor.Describe());
}

// Test: SyncGsaExecutor Parser with max values
TEST_F(SyncGsaExecutorTest, ParserMaxValues)
{
    CcuInstr instr;
    memset(&instr, 0xFF, sizeof(instr));

    SyncGsaExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Parser());
    EXPECT_NO_THROW(executor.Describe());
}

// Test: SyncGsaExecutor Parser with specific GSA parameters
TEST_F(SyncGsaExecutorTest, ParserGsaParameters)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    instr.v1.syncGSA.rmtGSAId = 100;
    instr.v1.syncGSA.locGSAId = 200;
    instr.v1.syncGSA.channelId = 5;
    instr.v1.syncGSA.setRmtCKEId = 10;
    instr.v1.syncGSA.setRmtCKEMask = 0xFF;

    SyncGsaExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Parser());
    std::string desc = executor.Describe();
    EXPECT_FALSE(desc.empty());
}

// Test: SyncGsaExecutor with different GSA IDs
TEST_F(SyncGsaExecutorTest, DifferentGsaIds)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    uint16_t gsaIds[] = {0, 1, SimCcuV1::CCU_RESOURCE_GSA_NUM / 2, SimCcuV1::CCU_RESOURCE_GSA_NUM - 1, 0xFFFF};

    for (auto gsaId : gsaIds) {
        instr.v1.syncGSA.locGSAId = gsaId;
        instr.v1.syncGSA.rmtGSAId = gsaId;
        SyncGsaExecutor executor(0, 0, 0, instr, nullptr);
        EXPECT_NO_THROW(executor.Parser());
        EXPECT_NO_THROW(executor.Describe());
    }
}

// Test: SyncGsaExecutor Describe contains expected keywords
TEST_F(SyncGsaExecutorTest, DescribeContent)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    instr.v1.syncGSA.rmtGSAId = 10;
    instr.v1.syncGSA.locGSAId = 20;
    instr.v1.syncGSA.channelId = 5;

    SyncGsaExecutor executor(0, 0, 0, instr, nullptr);
    executor.Parser();
    std::string desc = executor.Describe();
    EXPECT_FALSE(desc.empty());
    EXPECT_NE(desc.find("Sync locGSAId"), std::string::npos);
}

// Test: SyncGsaExecutor inheritance check
TEST_F(SyncGsaExecutorTest, InheritanceCheck)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    SyncGsaExecutor executor(0, 0, 0, instr, nullptr);
    CcuExecutorBase* base = &executor;
    EXPECT_NE(base, nullptr);
}

// Test: SyncGsaExecutor with various channel configurations
TEST_F(SyncGsaExecutorTest, VariousChannelConfigurations)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    for (uint16_t ch = 0; ch < 8; ch++) {
        instr.v1.syncGSA.channelId = ch;
        SyncGsaExecutor executor(0, 0, 0, instr, nullptr);
        EXPECT_NO_THROW(executor.Parser());
        EXPECT_NO_THROW(executor.Describe());
    }
}

TEST_F(SyncGsaExecutorTest, ProcessWithValidChannel)
{
    auto& mgr = CcuResourceManager::GetInstance();
    mgr.UpdateGsaValue(0, 0, 0, 0x1000);
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    instr.v1.syncGSA.channelId = 0;
    instr.v1.syncGSA.locGSAId = 0;
    instr.v1.syncGSA.rmtGSAId = 0;
    instr.v1.syncGSA.setCKEId = 0;
    instr.v1.syncGSA.setCKEMask = 0;
    instr.v1.syncGSA.setRmtCKEId = 0;
    instr.v1.syncGSA.setRmtCKEMask = 0;
    CcuSimulator sim(0, 0, 0, 1, 1, RunnerCcuVersion::CCU_V1);
    SyncGsaExecutor executor(0, 0, 0, instr, &sim);
    executor.Parser();
    EXPECT_NO_THROW(executor.Process(mgr));
}

TEST_F(SyncGsaExecutorTest, RunWithCkeNotSatisfied)
{
    auto& mgr = CcuResourceManager::GetInstance();
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    instr.v1.syncGSA.channelId = 0;
    instr.v1.syncGSA.waitCKEId = 0;
    instr.v1.syncGSA.waitCKEMask = 0x0001;
    instr.v1.syncGSA.clearType = 0;
    CcuSimulator sim(0, 0, 0, 1, 1, RunnerCcuVersion::CCU_V1);
    SyncGsaExecutor executor(0, 0, 0, instr, &sim);
    executor.Parser();
    executor.Run();
}
