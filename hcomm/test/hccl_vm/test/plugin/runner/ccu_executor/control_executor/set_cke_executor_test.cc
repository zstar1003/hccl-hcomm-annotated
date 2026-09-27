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
#include "set_cke_executor.h"

using namespace hcomm::CcuRep;

class SetCkeExecutorTest : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

// Test: SetCkeExecutor struct size check
TEST_F(SetCkeExecutorTest, StructSize) { EXPECT_GT(sizeof(SetCkeExecutor), 0); }

// Test: SetCkeExecutor default constructor
TEST_F(SetCkeExecutorTest, DefaultConstructor)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    SetCkeExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Describe());
}

// Test: SetCkeExecutor parameterized constructor
TEST_F(SetCkeExecutorTest, ParameterizedConstructor)
{
    int streamId = 0;
    int rankId = 0;
    int dieId = 0;
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    SetCkeExecutor executor(streamId, rankId, dieId, instr, nullptr);
    EXPECT_NO_THROW(executor.Describe());
}

// Test: SetCkeExecutor Parser with zero values
TEST_F(SetCkeExecutorTest, ParserZeroValues)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    SetCkeExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Parser());
    EXPECT_NO_THROW(executor.Describe());
}

// Test: SetCkeExecutor Parser with max values
TEST_F(SetCkeExecutorTest, ParserMaxValues)
{
    CcuInstr instr;
    memset(&instr, 0xFF, sizeof(instr));

    SetCkeExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Parser());
    EXPECT_NO_THROW(executor.Describe());
}

// Test: SetCkeExecutor Parser with specific CKE parameters
TEST_F(SetCkeExecutorTest, ParserCkeParameters)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    instr.v1.setCKE.clearType = 1;
    instr.v1.setCKE.setCKEId = 100;
    instr.v1.setCKE.setCKEMask = 0xFF00;
    instr.v1.setCKE.waitCKEId = 50;
    instr.v1.setCKE.waitCKEMask = 0x00FF;

    SetCkeExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Parser());
    std::string desc = executor.Describe();
    EXPECT_FALSE(desc.empty());
}

// Test: SetCkeExecutor with different clearType values
TEST_F(SetCkeExecutorTest, DifferentClearTypeValues)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    for (uint16_t clearType = 0; clearType <= 2; clearType++) {
        instr.v1.setCKE.clearType = clearType;
        SetCkeExecutor executor(0, 0, 0, instr, nullptr);
        EXPECT_NO_THROW(executor.Parser());
        EXPECT_NO_THROW(executor.Describe());
    }
}

// Test: SetCkeExecutor with different mask patterns
TEST_F(SetCkeExecutorTest, DifferentMaskPatterns)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    uint16_t masks[] = {0x0000, 0x0001, 0x00FF, 0xFF00, 0xFFFF, 0x5555, 0xAAAA};

    for (auto mask : masks) {
        instr.v1.setCKE.setCKEMask = mask;
        instr.v1.setCKE.waitCKEMask = mask;
        SetCkeExecutor executor(0, 0, 0, instr, nullptr);
        EXPECT_NO_THROW(executor.Parser());
        EXPECT_NO_THROW(executor.Describe());
    }
}

// Test: SetCkeExecutor Describe contains expected keywords
TEST_F(SetCkeExecutorTest, DescribeContent)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    instr.v1.setCKE.setCKEId = 10;
    instr.v1.setCKE.setCKEMask = 0xFF;
    instr.v1.setCKE.waitCKEId = 5;
    instr.v1.setCKE.waitCKEMask = 0xF0;

    SetCkeExecutor executor(0, 0, 0, instr, nullptr);
    executor.Parser();
    std::string desc = executor.Describe();
    EXPECT_FALSE(desc.empty());
    EXPECT_NE(desc.find("Set CKE"), std::string::npos);
}

// Test: SetCkeExecutor inheritance check
TEST_F(SetCkeExecutorTest, InheritanceCheck)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    SetCkeExecutor executor(0, 0, 0, instr, nullptr);
    CcuExecutorBase* base = &executor;
    EXPECT_NE(base, nullptr);
}

// Test: SetCkeExecutor with various CKE IDs
TEST_F(SetCkeExecutorTest, VariousCkeIds)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    uint16_t ckeIds[] = {0, 1, 100, SimCcuV1::CCU_RESOURCE_CKE_NUM - 1, 0xFFFF};

    for (auto ckeId : ckeIds) {
        instr.v1.setCKE.setCKEId = ckeId;
        instr.v1.setCKE.waitCKEId = ckeId;
        SetCkeExecutor executor(0, 0, 0, instr, nullptr);
        EXPECT_NO_THROW(executor.Parser());
        EXPECT_NO_THROW(executor.Describe());
    }
}
