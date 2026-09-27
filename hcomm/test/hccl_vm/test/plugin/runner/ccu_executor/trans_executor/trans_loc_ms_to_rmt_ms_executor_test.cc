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
#include "trans_loc_ms_to_rmt_ms_executor.h"

using namespace hcomm::CcuRep;

class TransLocMSToRmtMSExecutorTest : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

// Test: TransLocMSToRmtMSExecutor struct size check
TEST_F(TransLocMSToRmtMSExecutorTest, StructSize) { EXPECT_GT(sizeof(TransLocMSToRmtMSExecutor), 0); }

// Test: TransLocMSToRmtMSExecutor default constructor
TEST_F(TransLocMSToRmtMSExecutorTest, DefaultConstructor)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    TransLocMSToRmtMSExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Describe());
}

// Test: TransLocMSToRmtMSExecutor parameterized constructor
TEST_F(TransLocMSToRmtMSExecutorTest, ParameterizedConstructor)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    TransLocMSToRmtMSExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Describe());
}

// Test: TransLocMSToRmtMSExecutor Parser with zero values
TEST_F(TransLocMSToRmtMSExecutorTest, ParserZeroValues)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    TransLocMSToRmtMSExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Parser());
    EXPECT_NO_THROW(executor.Describe());
}

// Test: TransLocMSToRmtMSExecutor Parser with max values
TEST_F(TransLocMSToRmtMSExecutorTest, ParserMaxValues)
{
    CcuInstr instr;
    memset(&instr, 0xFF, sizeof(instr));

    TransLocMSToRmtMSExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Parser());
    EXPECT_NO_THROW(executor.Describe());
}

// Test: TransLocMSToRmtMSExecutor Parser with specific parameters
TEST_F(TransLocMSToRmtMSExecutorTest, ParserSpecificParameters)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    instr.v1.transLocMSToRmtMS.rmtMSId = 100;
    instr.v1.transLocMSToRmtMS.locMSId = 50;
    instr.v1.transLocMSToRmtMS.lengthXnId = 5;
    instr.v1.transLocMSToRmtMS.channelId = 10;
    instr.v1.transLocMSToRmtMS.setRmtCKEId = 20;
    instr.v1.transLocMSToRmtMS.setRmtCKEMask = 0xFF;

    TransLocMSToRmtMSExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Parser());
    std::string desc = executor.Describe();
    EXPECT_FALSE(desc.empty());
}

// Test: TransLocMSToRmtMSExecutor with different channel IDs
TEST_F(TransLocMSToRmtMSExecutorTest, DifferentChannelIds)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    for (uint16_t ch = 0; ch < 16; ch++) {
        instr.v1.transLocMSToRmtMS.channelId = ch;
        TransLocMSToRmtMSExecutor executor(0, 0, 0, instr, nullptr);
        EXPECT_NO_THROW(executor.Parser());
        EXPECT_NO_THROW(executor.Describe());
    }
}

// Test: TransLocMSToRmtMSExecutor Describe contains expected keywords
TEST_F(TransLocMSToRmtMSExecutorTest, DescribeContent)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    instr.v1.transLocMSToRmtMS.locMSId = 10;
    instr.v1.transLocMSToRmtMS.rmtMSId = 100;
    instr.v1.transLocMSToRmtMS.channelId = 5;

    TransLocMSToRmtMSExecutor executor(0, 0, 0, instr, nullptr);
    executor.Parser();
    std::string desc = executor.Describe();
    EXPECT_FALSE(desc.empty());
    EXPECT_NE(desc.find("Trans LocMS"), std::string::npos);
}

// Test: TransLocMSToRmtMSExecutor inheritance check
TEST_F(TransLocMSToRmtMSExecutorTest, InheritanceCheck)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    TransLocMSToRmtMSExecutor executor(0, 0, 0, instr, nullptr);
    CcuExecutorBase* base = &executor;
    EXPECT_NE(base, nullptr);
}

TEST_F(TransLocMSToRmtMSExecutorTest, ProcessWithInvalidChannelId)
{
    auto& mgr = CcuResourceManager::GetInstance();
    mgr.Init(0, 2, RunnerCcuVersion::CCU_V1, {});
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    instr.v1.transLocMSToRmtMS.channelId = SimCcuV1::MAX_CCU_CHANNEL_NUM;
    instr.v1.transLocMSToRmtMS.locMSId = 0;
    instr.v1.transLocMSToRmtMS.rmtMSId = 0;
    instr.v1.transLocMSToRmtMS.lengthEn = 0;
    instr.v1.transLocMSToRmtMS.setCKEId = 0;
    instr.v1.transLocMSToRmtMS.setCKEMask = 0;
    instr.v1.transLocMSToRmtMS.setRmtCKEId = 0;
    instr.v1.transLocMSToRmtMS.setRmtCKEMask = 0;
    CcuSimulator sim(0, 0, 0, 1, 1, RunnerCcuVersion::CCU_V1);
    TransLocMSToRmtMSExecutor executor(0, 0, 0, instr, &sim);
    executor.Parser();
    EXPECT_NO_THROW(executor.Process(mgr));
}

TEST_F(TransLocMSToRmtMSExecutorTest, RunWithCkeNotSatisfied)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    instr.v1.transLocMSToRmtMS.waitCKEId = 0;
    instr.v1.transLocMSToRmtMS.waitCKEMask = 0x0001;
    instr.v1.transLocMSToRmtMS.clearType = 0;
    CcuSimulator sim(0, 0, 0, 1, 1, RunnerCcuVersion::CCU_V1);
    TransLocMSToRmtMSExecutor executor(0, 0, 0, instr, &sim);
    executor.Parser();
    executor.Run();
}

TEST_F(TransLocMSToRmtMSExecutorTest, ProcessWithDieIdMismatch)
{
    auto& mgr = CcuResourceManager::GetInstance();
    mgr.Init(0, 2, RunnerCcuVersion::CCU_V1, {});
    mgr.Init(1, 2, RunnerCcuVersion::CCU_V1, {});
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    instr.v1.transLocMSToRmtMS.channelId = 0;
    instr.v1.transLocMSToRmtMS.rmtMSId = 0x8000;
    instr.v1.transLocMSToRmtMS.locMSId = 0;
    instr.v1.transLocMSToRmtMS.waitCKEMask = 0;
    CcuSimulator sim(0, 0, 0, 1, 1, RunnerCcuVersion::CCU_V1);
    TransLocMSToRmtMSExecutor executor(0, 0, 0, instr, &sim);
    executor.Parser();
    EXPECT_NO_THROW(executor.Process(mgr));
}

TEST_F(TransLocMSToRmtMSExecutorTest, ProcessWithValidChannel)
{
    auto& mgr = CcuResourceManager::GetInstance();
    mgr.Init(0, 2, RunnerCcuVersion::CCU_V1, {});
    mgr.Init(1, 2, RunnerCcuVersion::CCU_V1, {});
    mgr.UpdateXnValue(0, 0, 0, 4096);
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    instr.v1.transLocMSToRmtMS.channelId = 0;
    instr.v1.transLocMSToRmtMS.rmtMSId = 0;
    instr.v1.transLocMSToRmtMS.locMSId = 0;
    instr.v1.transLocMSToRmtMS.lengthEn = 1;
    instr.v1.transLocMSToRmtMS.lengthXnId = 0;
    instr.v1.transLocMSToRmtMS.waitCKEMask = 0;
    CcuSimulator sim(0, 0, 0, 1, 1, RunnerCcuVersion::CCU_V1);
    TransLocMSToRmtMSExecutor executor(0, 0, 0, instr, &sim);
    executor.Parser();
    EXPECT_NO_THROW(executor.Process(mgr));
}
