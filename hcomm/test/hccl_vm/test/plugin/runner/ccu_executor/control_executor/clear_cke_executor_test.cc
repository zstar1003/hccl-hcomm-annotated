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
#include "clear_cke_executor.h"

using namespace hcomm::CcuRep;

class ClearCkeExecutorTest : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

// Test: ClearCkeExecutor struct size check
TEST_F(ClearCkeExecutorTest, StructSize) { EXPECT_GT(sizeof(ClearCkeExecutor), 0); }

// Test: ClearCkeExecutor default constructor
TEST_F(ClearCkeExecutorTest, DefaultConstructor)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    ClearCkeExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Describe());
}

// Test: ClearCkeExecutor parameterized constructor with valid parameters
TEST_F(ClearCkeExecutorTest, ParameterizedConstructor)
{
    int streamId = 0;
    int rankId = 0;
    int dieId = 0;
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    ClearCkeExecutor executor(streamId, rankId, dieId, instr, nullptr);
    EXPECT_NO_THROW(executor.Describe());
}

// Test: ClearCkeExecutor Parser with zero values
TEST_F(ClearCkeExecutorTest, ParserZeroValues)
{
    int streamId = 0;
    int rankId = 0;
    int dieId = 0;
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    ClearCkeExecutor executor(streamId, rankId, dieId, instr, nullptr);
    EXPECT_NO_THROW(executor.Parser());
    EXPECT_NO_THROW(executor.Describe());
}

// Test: ClearCkeExecutor Parser with max values
TEST_F(ClearCkeExecutorTest, ParserMaxValues)
{
    int streamId = 0;
    int rankId = 0;
    int dieId = 0;
    CcuInstr instr;
    memset(&instr, 0xFF, sizeof(instr));

    ClearCkeExecutor executor(streamId, rankId, dieId, instr, nullptr);
    EXPECT_NO_THROW(executor.Parser());
    EXPECT_NO_THROW(executor.Describe());
}

// Test: ClearCkeExecutor Parser with boundary values
TEST_F(ClearCkeExecutorTest, ParserBoundaryValues)
{
    int streamId = 0;
    int rankId = 0;
    int dieId = 0;
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    // Set specific boundary values
    instr.v1.clearCKE.clearType = 1;
    instr.v1.clearCKE.clearCKEId = 0xFFFF;
    instr.v1.clearCKE.clearMask = 0xFFFF;
    instr.v1.clearCKE.waitCKEId = 0xFFFF;
    instr.v1.clearCKE.waitCKEMask = 0xFFFF;

    ClearCkeExecutor executor(streamId, rankId, dieId, instr, nullptr);
    EXPECT_NO_THROW(executor.Parser());
    std::string desc = executor.Describe();
    EXPECT_FALSE(desc.empty());
}

// Test: ClearCkeExecutor with different rank and die IDs
TEST_F(ClearCkeExecutorTest, DifferentRankDieIds)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    for (int rankId = 0; rankId < 8; rankId++) {
        for (int dieId = 0; dieId < 4; dieId++) {
            ClearCkeExecutor executor(0, rankId, dieId, instr, nullptr);
            EXPECT_NO_THROW(executor.Parser());
            EXPECT_NO_THROW(executor.Describe());
        }
    }
}

// Test: ClearCkeExecutor Describe returns non-empty string
TEST_F(ClearCkeExecutorTest, DescribeNonEmpty)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    instr.v1.clearCKE.clearType = 1;
    instr.v1.clearCKE.clearCKEId = 100;
    instr.v1.clearCKE.clearMask = 0x00FF;
    instr.v1.clearCKE.waitCKEId = 50;
    instr.v1.clearCKE.waitCKEMask = 0xFF00;

    ClearCkeExecutor executor(0, 0, 0, instr, nullptr);
    executor.Parser();
    std::string desc = executor.Describe();
    EXPECT_FALSE(desc.empty());
    EXPECT_NE(desc.find("Clear CKE"), std::string::npos);
}

// Test: ClearCkeExecutor inheritance from CcuExecutorBase
TEST_F(ClearCkeExecutorTest, InheritanceCheck)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    ClearCkeExecutor executor(0, 0, 0, instr, nullptr);
    CcuExecutorBase* base = &executor;
    EXPECT_NE(base, nullptr);
}
