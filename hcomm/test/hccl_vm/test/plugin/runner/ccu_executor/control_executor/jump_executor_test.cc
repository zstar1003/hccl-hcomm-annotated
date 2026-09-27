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
#include "jump_executor.h"

using namespace hcomm::CcuRep;

class JumpExecutorTest : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

// Test: JumpExecutor struct size check
TEST_F(JumpExecutorTest, StructSize) { EXPECT_GT(sizeof(JumpExecutor), 0); }

// Test: JumpExecutor default constructor
TEST_F(JumpExecutorTest, DefaultConstructor)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    JumpExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Describe());
}

// Test: JumpExecutor parameterized constructor
TEST_F(JumpExecutorTest, ParameterizedConstructor)
{
    int streamId = 0;
    int rankId = 0;
    int dieId = 0;
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    JumpExecutor executor(streamId, rankId, dieId, instr, nullptr);
    EXPECT_NO_THROW(executor.Describe());
}

// Test: JumpExecutor Parser with zero values
TEST_F(JumpExecutorTest, ParserZeroValues)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    JumpExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Parser());
    EXPECT_NO_THROW(executor.Describe());
}

// Test: JumpExecutor Parser with max values
TEST_F(JumpExecutorTest, ParserMaxValues)
{
    CcuInstr instr;
    memset(&instr, 0xFF, sizeof(instr));

    JumpExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Parser());
    EXPECT_NO_THROW(executor.Describe());
}

// Test: JumpExecutor Parser with boundary values
TEST_F(JumpExecutorTest, ParserBoundaryValues)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    instr.v1.jmp.dstInstrXnId = 0xFFFF;
    instr.v1.jmp.conditionXnId = 0xFFFF;
    instr.v1.jmp.expectData = 0xFFFFFFFFFFFFFFFFULL;

    JumpExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Parser());
    std::string desc = executor.Describe();
    EXPECT_FALSE(desc.empty());
}

// Test: JumpExecutor with different expectData values
TEST_F(JumpExecutorTest, DifferentExpectDataValues)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    uint64_t testValues[] = {0, 1, 0x7FFFFFFFFFFFFFFFLL, 0xFFFFFFFFFFFFFFFFULL, 1000, 0x12345678ABCDEF00ULL};

    for (auto val : testValues) {
        instr.v1.jmp.expectData = val;
        JumpExecutor executor(0, 0, 0, instr, nullptr);
        EXPECT_NO_THROW(executor.Parser());
        EXPECT_NO_THROW(executor.Describe());
    }
}

// Test: JumpExecutor Describe contains expected keywords
TEST_F(JumpExecutorTest, DescribeContent)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    instr.v1.jmp.dstInstrXnId = 100;
    instr.v1.jmp.conditionXnId = 50;
    instr.v1.jmp.expectData = 12345;

    JumpExecutor executor(0, 0, 0, instr, nullptr);
    executor.Parser();
    std::string desc = executor.Describe();
    EXPECT_FALSE(desc.empty());
    EXPECT_NE(desc.find("Jump"), std::string::npos);
}

// Test: JumpExecutor with different stream IDs
TEST_F(JumpExecutorTest, DifferentStreamIds)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    for (int streamId = 0; streamId < 16; streamId++) {
        JumpExecutor executor(streamId, 0, 0, instr, nullptr);
        EXPECT_NO_THROW(executor.Parser());
        EXPECT_NO_THROW(executor.Describe());
    }
}

// Test: JumpExecutor inheritance check
TEST_F(JumpExecutorTest, InheritanceCheck)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    JumpExecutor executor(0, 0, 0, instr, nullptr);
    CcuExecutorBase* base = &executor;
    EXPECT_NE(base, nullptr);
}

// Test: JumpExecutor multiple parse calls
TEST_F(JumpExecutorTest, MultipleParseCalls)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    instr.v1.jmp.dstInstrXnId = 100;

    JumpExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Parser());
    EXPECT_NO_THROW(executor.Parser());
    EXPECT_NO_THROW(executor.Parser());
}
