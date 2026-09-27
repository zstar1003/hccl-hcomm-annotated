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
#include "trans_loc_ms_to_loc_mem_executor.h"

using namespace hcomm::CcuRep;

class TransLocMSToLocMemExecutorTest : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

// Test: TransLocMSToLocMemExecutor struct size check
TEST_F(TransLocMSToLocMemExecutorTest, StructSize) { EXPECT_GT(sizeof(TransLocMSToLocMemExecutor), 0); }

// Test: TransLocMSToLocMemExecutor default constructor
TEST_F(TransLocMSToLocMemExecutorTest, DefaultConstructor)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    TransLocMSToLocMemExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Describe());
}

// Test: TransLocMSToLocMemExecutor parameterized constructor
TEST_F(TransLocMSToLocMemExecutorTest, ParameterizedConstructor)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    TransLocMSToLocMemExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Describe());
}

// Test: TransLocMSToLocMemExecutor Parser with zero values
TEST_F(TransLocMSToLocMemExecutorTest, ParserZeroValues)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    TransLocMSToLocMemExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Parser());
    EXPECT_NO_THROW(executor.Describe());
}

// Test: TransLocMSToLocMemExecutor Parser with max values
TEST_F(TransLocMSToLocMemExecutorTest, ParserMaxValues)
{
    CcuInstr instr;
    memset(&instr, 0xFF, sizeof(instr));

    TransLocMSToLocMemExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Parser());
    EXPECT_NO_THROW(executor.Describe());
}

// Test: TransLocMSToLocMemExecutor Parser with specific parameters
TEST_F(TransLocMSToLocMemExecutorTest, ParserSpecificParameters)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    instr.v1.transLocMSToLocMem.locGSAId = 100;
    instr.v1.transLocMSToLocMem.locXnId = 50;
    instr.v1.transLocMSToLocMem.locMSId = 0x8000 | 10;
    instr.v1.transLocMSToLocMem.lengthXnId = 5;
    instr.v1.transLocMSToLocMem.channelId = 0;
    instr.v1.transLocMSToLocMem.lengthEn = 1;

    TransLocMSToLocMemExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Parser());
    std::string desc = executor.Describe();
    EXPECT_FALSE(desc.empty());
}

// Test: TransLocMSToLocMemExecutor with different lengthEn values
TEST_F(TransLocMSToLocMemExecutorTest, DifferentLengthEnValues)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    for (uint16_t lenEn = 0; lenEn <= 1; lenEn++) {
        instr.v1.transLocMSToLocMem.lengthEn = lenEn;
        TransLocMSToLocMemExecutor executor(0, 0, 0, instr, nullptr);
        EXPECT_NO_THROW(executor.Parser());
        EXPECT_NO_THROW(executor.Describe());
    }
}

// Test: TransLocMSToLocMemExecutor with different MS IDs
TEST_F(TransLocMSToLocMemExecutorTest, DifferentMsIds)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    uint16_t msIds[] = {0, 1, 100, SimCcuV1::CCU_RESOURCE_MS_NUM - 1, 0x7FFF, 0xFFFF};

    for (auto msId : msIds) {
        instr.v1.transLocMSToLocMem.locMSId = msId;
        TransLocMSToLocMemExecutor executor(0, 0, 0, instr, nullptr);
        EXPECT_NO_THROW(executor.Parser());
        EXPECT_NO_THROW(executor.Describe());
    }
}

// Test: TransLocMSToLocMemExecutor Describe contains expected keywords
TEST_F(TransLocMSToLocMemExecutorTest, DescribeContent)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    instr.v1.transLocMSToLocMem.locGSAId = 10;
    instr.v1.transLocMSToLocMem.locMSId = 100;

    TransLocMSToLocMemExecutor executor(0, 0, 0, instr, nullptr);
    executor.Parser();
    std::string desc = executor.Describe();
    EXPECT_FALSE(desc.empty());
    EXPECT_NE(desc.find("LocMSToLocMem"), std::string::npos);
}

// Test: TransLocMSToLocMemExecutor inheritance check
TEST_F(TransLocMSToLocMemExecutorTest, InheritanceCheck)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    TransLocMSToLocMemExecutor executor(0, 0, 0, instr, nullptr);
    CcuExecutorBase* base = &executor;
    EXPECT_NE(base, nullptr);
}

TEST_F(TransLocMSToLocMemExecutorTest, ProcessWithZeroValues)
{
    auto& mgr = CcuResourceManager::GetInstance();
    mgr.Init(0, 2, RunnerCcuVersion::CCU_V1, {});
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    instr.v1.transLocMSToLocMem.locGSAId = 0;
    instr.v1.transLocMSToLocMem.locMSId = 0;
    instr.v1.transLocMSToLocMem.lengthEn = 0;
    instr.v1.transLocMSToLocMem.setCKEId = 0;
    instr.v1.transLocMSToLocMem.setCKEMask = 0;
    CcuSimulator sim(0, 0, 0, 1, 1, RunnerCcuVersion::CCU_V1);
    TransLocMSToLocMemExecutor executor(0, 0, 0, instr, &sim);
    executor.Parser();
    EXPECT_NO_THROW(executor.Process(mgr));
}

TEST_F(TransLocMSToLocMemExecutorTest, RunWithCkeNotSatisfied)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    instr.v1.transLocMSToLocMem.waitCKEId = 0;
    instr.v1.transLocMSToLocMem.waitCKEMask = 0x0001;
    instr.v1.transLocMSToLocMem.clearType = 0;
    CcuSimulator sim(0, 0, 0, 1, 1, RunnerCcuVersion::CCU_V1);
    TransLocMSToLocMemExecutor executor(0, 0, 0, instr, &sim);
    executor.Parser();
    executor.Run();
}

TEST_F(TransLocMSToLocMemExecutorTest, ProcessWithNullAddressFails)
{
    auto& mgr = CcuResourceManager::GetInstance();
    mgr.Init(0, 2, RunnerCcuVersion::CCU_V1, {});
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    instr.v1.transLocMSToLocMem.waitCKEMask = 0;
    CcuSimulator sim(0, 0, 0, 1, 1, RunnerCcuVersion::CCU_V1);
    TransLocMSToLocMemExecutor executor(0, 0, 0, instr, &sim);
    executor.Parser();
    EXPECT_NO_THROW(executor.Process(mgr));
}

TEST_F(TransLocMSToLocMemExecutorTest, ProcessWithLengthEnEnabled)
{
    auto& mgr = CcuResourceManager::GetInstance();
    mgr.Init(0, 2, RunnerCcuVersion::CCU_V1, {});
    mgr.UpdateXnValue(0, 0, 0, 4096);
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    instr.v1.transLocMSToLocMem.lengthEn = 1;
    instr.v1.transLocMSToLocMem.lengthXnId = 0;
    instr.v1.transLocMSToLocMem.waitCKEMask = 0;
    CcuSimulator sim(0, 0, 0, 1, 1, RunnerCcuVersion::CCU_V1);
    TransLocMSToLocMemExecutor executor(0, 0, 0, instr, &sim);
    executor.Parser();
    EXPECT_NO_THROW(executor.Process(mgr));
}
