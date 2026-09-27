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
#include "trans_loc_mem_to_loc_mem_executor.h"

using namespace hcomm::CcuRep;

class TransLocMemToLocMemExecutorTest : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

// Test: TransLocMemToLocMemExecutor struct size check
TEST_F(TransLocMemToLocMemExecutorTest, StructSize) { EXPECT_GT(sizeof(TransLocMemToLocMemExecutor), 0); }

// Test: TransLocMemToLocMemExecutor default constructor
TEST_F(TransLocMemToLocMemExecutorTest, DefaultConstructor)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    TransLocMemToLocMemExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Describe());
}

// Test: TransLocMemToLocMemExecutor parameterized constructor
TEST_F(TransLocMemToLocMemExecutorTest, ParameterizedConstructor)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    TransLocMemToLocMemExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Describe());
}

// Test: TransLocMemToLocMemExecutor Parser with zero values
TEST_F(TransLocMemToLocMemExecutorTest, ParserZeroValues)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    TransLocMemToLocMemExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Parser());
    EXPECT_NO_THROW(executor.Describe());
}

// Test: TransLocMemToLocMemExecutor Parser with max values
TEST_F(TransLocMemToLocMemExecutorTest, ParserMaxValues)
{
    CcuInstr instr;
    memset(&instr, 0xFF, sizeof(instr));

    TransLocMemToLocMemExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Parser());
    EXPECT_NO_THROW(executor.Describe());
}

// Test: TransLocMemToLocMemExecutor Parser with specific parameters
TEST_F(TransLocMemToLocMemExecutorTest, ParserSpecificParameters)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    instr.v1.transLocMemToLocMem.dstGSAId = 100;
    instr.v1.transLocMemToLocMem.dstXnId = 50;
    instr.v1.transLocMemToLocMem.srcGSAId = 10;
    instr.v1.transLocMemToLocMem.srcXnId = 5;
    instr.v1.transLocMemToLocMem.lengthXnId = 3;
    instr.v1.transLocMemToLocMem.lengthEn = 1;

    TransLocMemToLocMemExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Parser());
    std::string desc = executor.Describe();
    EXPECT_FALSE(desc.empty());
}

// Test: TransLocMemToLocMemExecutor with different source and destination GSA IDs
TEST_F(TransLocMemToLocMemExecutorTest, DifferentGsaIds)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    uint16_t gsaIds[] = {0, 1, 100, SimCcuV1::CCU_RESOURCE_GSA_NUM - 1};

    for (auto srcGsaId : gsaIds) {
        for (auto dstGsaId : gsaIds) {
            instr.v1.transLocMemToLocMem.srcGSAId = srcGsaId;
            instr.v1.transLocMemToLocMem.dstGSAId = dstGsaId;
            TransLocMemToLocMemExecutor executor(0, 0, 0, instr, nullptr);
            EXPECT_NO_THROW(executor.Parser());
            EXPECT_NO_THROW(executor.Describe());
        }
    }
}

// Test: TransLocMemToLocMemExecutor Describe contains expected keywords
TEST_F(TransLocMemToLocMemExecutorTest, DescribeContent)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    instr.v1.transLocMemToLocMem.srcGSAId = 10;
    instr.v1.transLocMemToLocMem.dstGSAId = 100;

    TransLocMemToLocMemExecutor executor(0, 0, 0, instr, nullptr);
    executor.Parser();
    std::string desc = executor.Describe();
    EXPECT_FALSE(desc.empty());
    EXPECT_NE(desc.find("Trans LocMem"), std::string::npos);
}

// Test: TransLocMemToLocMemExecutor inheritance check
TEST_F(TransLocMemToLocMemExecutorTest, InheritanceCheck)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    TransLocMemToLocMemExecutor executor(0, 0, 0, instr, nullptr);
    CcuExecutorBase* base = &executor;
    EXPECT_NE(base, nullptr);
}

TEST_F(TransLocMemToLocMemExecutorTest, ProcessWithNullAddress)
{
    auto& mgr = CcuResourceManager::GetInstance();
    mgr.Init(0, 2, RunnerCcuVersion::CCU_V1, {});
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    instr.v1.transLocMemToLocMem.waitCKEMask = 0;
    CcuSimulator sim(0, 0, 0, 1, 1, RunnerCcuVersion::CCU_V1);
    TransLocMemToLocMemExecutor executor(0, 0, 0, instr, &sim);
    executor.Parser();
    EXPECT_NO_THROW(executor.Process(mgr));
}

TEST_F(TransLocMemToLocMemExecutorTest, ProcessWithZeroValues)
{
    auto& mgr = CcuResourceManager::GetInstance();
    mgr.Init(0, 2, RunnerCcuVersion::CCU_V1, {});
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    instr.v1.transLocMemToLocMem.srcGSAId = 0;
    instr.v1.transLocMemToLocMem.dstGSAId = 0;
    instr.v1.transLocMemToLocMem.lengthEn = 0;
    instr.v1.transLocMemToLocMem.setCKEId = 0;
    instr.v1.transLocMemToLocMem.setCKEMask = 0;
    CcuSimulator sim(0, 0, 0, 1, 1, RunnerCcuVersion::CCU_V1);
    TransLocMemToLocMemExecutor executor(0, 0, 0, instr, &sim);
    executor.Parser();
    EXPECT_NO_THROW(executor.Process(mgr));
}

TEST_F(TransLocMemToLocMemExecutorTest, RunWithCkeNotSatisfied)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    instr.v1.transLocMemToLocMem.waitCKEId = 0;
    instr.v1.transLocMemToLocMem.waitCKEMask = 0x0001;
    instr.v1.transLocMemToLocMem.clearType = 0;
    CcuSimulator sim(0, 0, 0, 1, 1, RunnerCcuVersion::CCU_V1);
    TransLocMemToLocMemExecutor executor(0, 0, 0, instr, &sim);
    executor.Parser();
    executor.Run();
}
