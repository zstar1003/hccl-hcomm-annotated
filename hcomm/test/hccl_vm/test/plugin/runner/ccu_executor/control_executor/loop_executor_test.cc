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
#include "loop_executor.h"

using namespace hcomm::CcuRep;

class LoopExecutorTest : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

// Test: LoopExecutor struct size check
TEST_F(LoopExecutorTest, StructSize) { EXPECT_GT(sizeof(LoopExecutor), 0); }

// Test: LoopExecutor default constructor
TEST_F(LoopExecutorTest, DefaultConstructor)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    LoopExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Describe());
}

// Test: LoopExecutor parameterized constructor
TEST_F(LoopExecutorTest, ParameterizedConstructor)
{
    int streamId = 0;
    int rankId = 0;
    int dieId = 0;
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    LoopExecutor executor(streamId, rankId, dieId, instr, nullptr);
    EXPECT_NO_THROW(executor.Describe());
}

// Test: LoopExecutor Parser with zero values
TEST_F(LoopExecutorTest, ParserZeroValues)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    LoopExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Parser());
    EXPECT_NO_THROW(executor.Describe());
}

// Test: LoopExecutor Parser with max values
TEST_F(LoopExecutorTest, ParserMaxValues)
{
    CcuInstr instr;
    memset(&instr, 0xFF, sizeof(instr));

    LoopExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Parser());
    EXPECT_NO_THROW(executor.Describe());
}

// Test: LoopExecutor Parser with specific loop parameters
TEST_F(LoopExecutorTest, ParserLoopParameters)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    instr.v1.loop.startInstrId = 10;
    instr.v1.loop.endInstrId = 100;
    instr.v1.loop.xnId = 5;

    LoopExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Parser());
    std::string desc = executor.Describe();
    EXPECT_FALSE(desc.empty());
    EXPECT_NE(desc.find("Loop"), std::string::npos);
}

// Test: LoopExecutor with different instruction ranges
TEST_F(LoopExecutorTest, DifferentInstructionRanges)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    struct Range {
        uint16_t start;
        uint16_t end;
    };
    Range ranges[] = {{0, 10}, {10, 100}, {0, 0xFFFF}, {1000, 2000}};

    for (const auto& range : ranges) {
        instr.v1.loop.startInstrId = range.start;
        instr.v1.loop.endInstrId = range.end;

        LoopExecutor executor(0, 0, 0, instr, nullptr);
        EXPECT_NO_THROW(executor.Parser());
        EXPECT_NO_THROW(executor.Describe());
    }
}

// Test: LoopExecutor with different xn IDs
TEST_F(LoopExecutorTest, DifferentXnIds)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    uint16_t xnIds[] = {0, 1, 100, 0x7FFF, 0xFFFF};

    for (auto xnId : xnIds) {
        instr.v1.loop.xnId = xnId;
        LoopExecutor executor(0, 0, 0, instr, nullptr);
        EXPECT_NO_THROW(executor.Parser());
        EXPECT_NO_THROW(executor.Describe());
    }
}

// Test: LoopExecutor Describe contains expected keywords
TEST_F(LoopExecutorTest, DescribeContent)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    instr.v1.loop.startInstrId = 5;
    instr.v1.loop.endInstrId = 50;
    instr.v1.loop.xnId = 10;

    LoopExecutor executor(0, 0, 0, instr, nullptr);
    executor.Parser();
    std::string desc = executor.Describe();
    EXPECT_FALSE(desc.empty());
    EXPECT_NE(desc.find("startInstrId"), std::string::npos);
    EXPECT_NE(desc.find("endInstrId"), std::string::npos);
}

// Test: LoopExecutor inheritance check
TEST_F(LoopExecutorTest, InheritanceCheck)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    LoopExecutor executor(0, 0, 0, instr, nullptr);
    CcuExecutorBase* base = &executor;
    EXPECT_NE(base, nullptr);
}

// Test: LoopExecutor with various rank and die combinations
TEST_F(LoopExecutorTest, VariousRankDieCombinations)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    instr.v1.loop.startInstrId = 0;
    instr.v1.loop.endInstrId = 10;
    instr.v1.loop.xnId = 0;

    for (int rank = 0; rank < 4; rank++) {
        for (int die = 0; die < 2; die++) {
            LoopExecutor executor(0, rank, die, instr, nullptr);
            EXPECT_NO_THROW(executor.Parser());
            EXPECT_NO_THROW(executor.Describe());
        }
    }
}
