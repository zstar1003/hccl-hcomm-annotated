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
#include "trans_loc_mem_to_loc_ms_executor.h"

using namespace hcomm::CcuRep;

class TransLocMemToLocMSExecutorTest : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

// Test: TransLocMemToLocMSExecutor struct size check
TEST_F(TransLocMemToLocMSExecutorTest, StructSize) { EXPECT_GT(sizeof(TransLocMemToLocMSExecutor), 0); }

// Test: TransLocMemToLocMSExecutor default constructor
TEST_F(TransLocMemToLocMSExecutorTest, DefaultConstructor)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    TransLocMemToLocMSExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Describe());
}

// Test: TransLocMemToLocMSExecutor parameterized constructor
TEST_F(TransLocMemToLocMSExecutorTest, ParameterizedConstructor)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    TransLocMemToLocMSExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Describe());
}

// Test: TransLocMemToLocMSExecutor Parser with zero values
TEST_F(TransLocMemToLocMSExecutorTest, ParserZeroValues)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    TransLocMemToLocMSExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Parser());
    EXPECT_NO_THROW(executor.Describe());
}

// Test: TransLocMemToLocMSExecutor Parser with max values
TEST_F(TransLocMemToLocMSExecutorTest, ParserMaxValues)
{
    CcuInstr instr;
    memset(&instr, 0xFF, sizeof(instr));

    TransLocMemToLocMSExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Parser());
    EXPECT_NO_THROW(executor.Describe());
}

// Test: TransLocMemToLocMSExecutor Parser with specific parameters
TEST_F(TransLocMemToLocMSExecutorTest, ParserSpecificParameters)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    instr.v1.transLocMemToLocMS.locGSAId = 100;
    instr.v1.transLocMemToLocMS.locXnId = 50;
    instr.v1.transLocMemToLocMS.locMSId = 0x8000 | 10;
    instr.v1.transLocMemToLocMS.lengthXnId = 5;
    instr.v1.transLocMemToLocMS.lengthEn = 0;

    TransLocMemToLocMSExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Parser());
    std::string desc = executor.Describe();
    EXPECT_FALSE(desc.empty());
}

// Test: TransLocMemToLocMSExecutor with different GSA IDs
TEST_F(TransLocMemToLocMSExecutorTest, DifferentGsaIds)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    uint16_t gsaIds[] = {0, 1, SimCcuV1::CCU_RESOURCE_GSA_NUM / 2, SimCcuV1::CCU_RESOURCE_GSA_NUM - 1};

    for (auto gsaId : gsaIds) {
        instr.v1.transLocMemToLocMS.locGSAId = gsaId;
        TransLocMemToLocMSExecutor executor(0, 0, 0, instr, nullptr);
        EXPECT_NO_THROW(executor.Parser());
        EXPECT_NO_THROW(executor.Describe());
    }
}

// Test: TransLocMemToLocMSExecutor Describe contains expected keywords
TEST_F(TransLocMemToLocMSExecutorTest, DescribeContent)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    instr.v1.transLocMemToLocMS.locGSAId = 10;
    instr.v1.transLocMemToLocMS.locMSId = 100;

    TransLocMemToLocMSExecutor executor(0, 0, 0, instr, nullptr);
    executor.Parser();
    std::string desc = executor.Describe();
    EXPECT_FALSE(desc.empty());
    EXPECT_NE(desc.find("Trans LocMem"), std::string::npos);
}

// Test: TransLocMemToLocMSExecutor inheritance check
TEST_F(TransLocMemToLocMSExecutorTest, InheritanceCheck)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    TransLocMemToLocMSExecutor executor(0, 0, 0, instr, nullptr);
    CcuExecutorBase* base = &executor;
    EXPECT_NE(base, nullptr);
}

TEST_F(TransLocMemToLocMSExecutorTest, ProcessWithNullAddressFails)
{
    auto& mgr = CcuResourceManager::GetInstance();
    mgr.Init(0, 2, RunnerCcuVersion::CCU_V1, {});
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    instr.v1.transLocMemToLocMS.waitCKEMask = 0;
    CcuSimulator sim(0, 0, 0, 1, 1, RunnerCcuVersion::CCU_V1);
    TransLocMemToLocMSExecutor executor(0, 0, 0, instr, &sim);
    executor.Parser();
    EXPECT_NO_THROW(executor.Process(mgr));
}

TEST_F(TransLocMemToLocMSExecutorTest, ProcessWithZeroValues)
{
    auto& mgr = CcuResourceManager::GetInstance();
    mgr.Init(0, 2, RunnerCcuVersion::CCU_V1, {});
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    instr.v1.transLocMemToLocMS.locGSAId = 0;
    instr.v1.transLocMemToLocMS.locMSId = 0;
    instr.v1.transLocMemToLocMS.lengthEn = 0;
    instr.v1.transLocMemToLocMS.setCKEId = 0;
    instr.v1.transLocMemToLocMS.setCKEMask = 0;
    CcuSimulator sim(0, 0, 0, 1, 1, RunnerCcuVersion::CCU_V1);
    TransLocMemToLocMSExecutor executor(0, 0, 0, instr, &sim);
    executor.Parser();
    EXPECT_NO_THROW(executor.Process(mgr));
}

TEST_F(TransLocMemToLocMSExecutorTest, RunWithCkeNotSatisfied)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    instr.v1.transLocMemToLocMS.waitCKEId = 0;
    instr.v1.transLocMemToLocMS.waitCKEMask = 0x0001;
    instr.v1.transLocMemToLocMS.clearType = 0;
    CcuSimulator sim(0, 0, 0, 1, 1, RunnerCcuVersion::CCU_V1);
    TransLocMemToLocMSExecutor executor(0, 0, 0, instr, &sim);
    executor.Parser();
    executor.Run();
}
