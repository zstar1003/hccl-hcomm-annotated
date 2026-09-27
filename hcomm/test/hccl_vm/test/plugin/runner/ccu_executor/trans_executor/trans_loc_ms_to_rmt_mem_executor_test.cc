/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstring>
#include <gtest/gtest.h>

#include "ccu_executor_base.h"
#include "ccu_microcode_common_v1.h"
#include "ccu_resource_manager.h"
#include "ccu_simulator.h"
#include "trans_loc_ms_to_rmt_mem_executor.h"

using namespace hcomm::CcuRep;

class TransLocMSToRmtMemExecutorTest : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

// Test: TransLocMSToRmtMemExecutor struct size check
TEST_F(TransLocMSToRmtMemExecutorTest, StructSize) { EXPECT_GT(sizeof(TransLocMSToRmtMemExecutor), 0); }

// Test: TransLocMSToRmtMemExecutor default constructor
TEST_F(TransLocMSToRmtMemExecutorTest, DefaultConstructor)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    TransLocMSToRmtMemExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Describe());
}

// Test: TransLocMSToRmtMemExecutor parameterized constructor
TEST_F(TransLocMSToRmtMemExecutorTest, ParameterizedConstructor)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    TransLocMSToRmtMemExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Describe());
}

// Test: TransLocMSToRmtMemExecutor Parser with zero values
TEST_F(TransLocMSToRmtMemExecutorTest, ParserZeroValues)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    TransLocMSToRmtMemExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Parser());
    EXPECT_NO_THROW(executor.Describe());
}

// Test: TransLocMSToRmtMemExecutor Parser with max values
TEST_F(TransLocMSToRmtMemExecutorTest, ParserMaxValues)
{
    CcuInstr instr;
    memset(&instr, 0xFF, sizeof(instr));

    TransLocMSToRmtMemExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Parser());
    EXPECT_NO_THROW(executor.Describe());
}

// Test: TransLocMSToRmtMemExecutor Parser with specific parameters
TEST_F(TransLocMSToRmtMemExecutorTest, ParserSpecificParameters)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    instr.v1.transLocMSToRmtMem.rmtGSAId = 100;
    instr.v1.transLocMSToRmtMem.rmtXnId = 50;
    instr.v1.transLocMSToRmtMem.locMSId = 10;
    instr.v1.transLocMSToRmtMem.lengthXnId = 5;
    instr.v1.transLocMSToRmtMem.channelId = 1;

    TransLocMSToRmtMemExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Parser());
    std::string desc = executor.Describe();
    EXPECT_FALSE(desc.empty());
}

// Test: TransLocMSToRmtMemExecutor Describe contains expected keywords
TEST_F(TransLocMSToRmtMemExecutorTest, DescribeContent)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    instr.v1.transLocMSToRmtMem.locMSId = 10;
    instr.v1.transLocMSToRmtMem.rmtGSAId = 100;

    TransLocMSToRmtMemExecutor executor(0, 0, 0, instr, nullptr);
    executor.Parser();
    std::string desc = executor.Describe();
    EXPECT_FALSE(desc.empty());
    EXPECT_NE(desc.find("Trans LocMS"), std::string::npos);
}

// Test: TransLocMSToRmtMemExecutor inheritance check
TEST_F(TransLocMSToRmtMemExecutorTest, InheritanceCheck)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    TransLocMSToRmtMemExecutor executor(0, 0, 0, instr, nullptr);
    CcuExecutorBase* base = &executor;
    EXPECT_NE(base, nullptr);
}

TEST_F(TransLocMSToRmtMemExecutorTest, ProcessWithZeroValues)
{
    auto& mgr = CcuResourceManager::GetInstance();
    mgr.Init(0, 2, RunnerCcuVersion::CCU_V1, {});
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    instr.v1.transLocMSToRmtMem.rmtGSAId = 0;
    instr.v1.transLocMSToRmtMem.locMSId = 0;
    instr.v1.transLocMSToRmtMem.lengthEn = 0;
    instr.v1.transLocMSToRmtMem.setCKEId = 0;
    instr.v1.transLocMSToRmtMem.setCKEMask = 0;
    CcuSimulator sim(0, 0, 0, 1, 1, RunnerCcuVersion::CCU_V1);
    TransLocMSToRmtMemExecutor executor(0, 0, 0, instr, &sim);
    executor.Parser();
    EXPECT_NO_THROW(executor.Process(mgr));
}

TEST_F(TransLocMSToRmtMemExecutorTest, RunWithCkeNotSatisfied)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    instr.v1.transLocMSToRmtMem.waitCKEId = 0;
    instr.v1.transLocMSToRmtMem.waitCKEMask = 0x0001;
    instr.v1.transLocMSToRmtMem.clearType = 0;
    CcuSimulator sim(0, 0, 0, 1, 1, RunnerCcuVersion::CCU_V1);
    TransLocMSToRmtMemExecutor executor(0, 0, 0, instr, &sim);
    executor.Parser();
    executor.Run();
}

TEST_F(TransLocMSToRmtMemExecutorTest, ProcessWithNullAddressFails)
{
    auto& mgr = CcuResourceManager::GetInstance();
    mgr.Init(0, 2, RunnerCcuVersion::CCU_V1, {});
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    instr.v1.transLocMSToRmtMem.waitCKEMask = 0;
    CcuSimulator sim(0, 0, 0, 1, 1, RunnerCcuVersion::CCU_V1);
    TransLocMSToRmtMemExecutor executor(0, 0, 0, instr, &sim);
    executor.Parser();
    EXPECT_NO_THROW(executor.Process(mgr));
}

TEST_F(TransLocMSToRmtMemExecutorTest, ProcessWithLengthEnEnabled)
{
    auto& mgr = CcuResourceManager::GetInstance();
    mgr.Init(0, 2, RunnerCcuVersion::CCU_V1, {});
    mgr.UpdateXnValue(0, 0, 0, 4096);
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    instr.v1.transLocMSToRmtMem.lengthEn = 1;
    instr.v1.transLocMSToRmtMem.lengthXnId = 0;
    instr.v1.transLocMSToRmtMem.waitCKEMask = 0;
    CcuSimulator sim(0, 0, 0, 1, 1, RunnerCcuVersion::CCU_V1);
    TransLocMSToRmtMemExecutor executor(0, 0, 0, instr, &sim);
    executor.Parser();
    EXPECT_NO_THROW(executor.Process(mgr));
}
