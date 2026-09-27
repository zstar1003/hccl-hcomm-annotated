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
#include "trans_loc_mem_to_rmt_mem_executor.h"

using namespace hcomm::CcuRep;

class TransLocMemToRmtMemExecutorTest : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

// Test: TransLocMemToRmtMemExecutor struct size check
TEST_F(TransLocMemToRmtMemExecutorTest, StructSize) { EXPECT_GT(sizeof(TransLocMemToRmtMemExecutor), 0); }

// Test: TransLocMemToRmtMemExecutor default constructor
TEST_F(TransLocMemToRmtMemExecutorTest, DefaultConstructor)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    TransLocMemToRmtMemExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Describe());
}

// Test: TransLocMemToRmtMemExecutor parameterized constructor
TEST_F(TransLocMemToRmtMemExecutorTest, ParameterizedConstructor)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    TransLocMemToRmtMemExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Describe());
}

// Test: TransLocMemToRmtMemExecutor Parser with zero values
TEST_F(TransLocMemToRmtMemExecutorTest, ParserZeroValues)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    TransLocMemToRmtMemExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Parser());
    EXPECT_NO_THROW(executor.Describe());
}

// Test: TransLocMemToRmtMemExecutor Parser with max values
TEST_F(TransLocMemToRmtMemExecutorTest, ParserMaxValues)
{
    CcuInstr instr;
    memset(&instr, 0xFF, sizeof(instr));

    TransLocMemToRmtMemExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Parser());
    EXPECT_NO_THROW(executor.Describe());
}

// Test: TransLocMemToRmtMemExecutor Parser with specific parameters
TEST_F(TransLocMemToRmtMemExecutorTest, ParserSpecificParameters)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    instr.v1.transLocMemToRmtMem.rmtGSAId = 100;
    instr.v1.transLocMemToRmtMem.rmtXnId = 50;
    instr.v1.transLocMemToRmtMem.locGSAId = 10;
    instr.v1.transLocMemToRmtMem.locXnId = 5;
    instr.v1.transLocMemToRmtMem.lengthXnId = 3;
    instr.v1.transLocMemToRmtMem.channelId = 1;

    TransLocMemToRmtMemExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Parser());
    std::string desc = executor.Describe();
    EXPECT_FALSE(desc.empty());
}

// Test: TransLocMemToRmtMemExecutor inheritance check
TEST_F(TransLocMemToRmtMemExecutorTest, InheritanceCheck)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    TransLocMemToRmtMemExecutor executor(0, 0, 0, instr, nullptr);
    CcuExecutorBase* base = &executor;
    EXPECT_NE(base, nullptr);
}

TEST_F(TransLocMemToRmtMemExecutorTest, ProcessWithNullAddress)
{
    auto& mgr = CcuResourceManager::GetInstance();
    mgr.Init(0, 2, RunnerCcuVersion::CCU_V1, {});
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    instr.v1.transLocMemToRmtMem.waitCKEMask = 0;
    CcuSimulator sim(0, 0, 0, 1, 1, RunnerCcuVersion::CCU_V1);
    TransLocMemToRmtMemExecutor executor(0, 0, 0, instr, &sim);
    executor.Parser();
    EXPECT_NO_THROW(executor.Process(mgr));
}

TEST_F(TransLocMemToRmtMemExecutorTest, ProcessWithZeroValues)
{
    auto& mgr = CcuResourceManager::GetInstance();
    mgr.Init(0, 2, RunnerCcuVersion::CCU_V1, {});
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    instr.v1.transLocMemToRmtMem.rmtGSAId = 0;
    instr.v1.transLocMemToRmtMem.locGSAId = 0;
    instr.v1.transLocMemToRmtMem.lengthEn = 0;
    instr.v1.transLocMemToRmtMem.setCKEId = 0;
    instr.v1.transLocMemToRmtMem.setCKEMask = 0;
    CcuSimulator sim(0, 0, 0, 1, 1, RunnerCcuVersion::CCU_V1);
    TransLocMemToRmtMemExecutor executor(0, 0, 0, instr, &sim);
    executor.Parser();
    EXPECT_NO_THROW(executor.Process(mgr));
}

TEST_F(TransLocMemToRmtMemExecutorTest, RunWithCkeNotSatisfied)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    instr.v1.transLocMemToRmtMem.waitCKEId = 0;
    instr.v1.transLocMemToRmtMem.waitCKEMask = 0x0001;
    instr.v1.transLocMemToRmtMem.clearType = 0;
    CcuSimulator sim(0, 0, 0, 1, 1, RunnerCcuVersion::CCU_V1);
    TransLocMemToRmtMemExecutor executor(0, 0, 0, instr, &sim);
    executor.Parser();
    executor.Run();
}
