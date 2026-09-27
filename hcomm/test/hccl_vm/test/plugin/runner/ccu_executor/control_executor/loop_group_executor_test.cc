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
#include "loop_group_executor.h"

using namespace hcomm::CcuRep;

class LoopGroupExecutorTest : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

// Test: LoopGroupExecutor struct size check
TEST_F(LoopGroupExecutorTest, StructSize) { EXPECT_GT(sizeof(LoopGroupExecutor), 0); }

// Test: LoopGroupExecutor default constructor
TEST_F(LoopGroupExecutorTest, DefaultConstructor)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    LoopGroupExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Describe());
}

// Test: LoopGroupExecutor parameterized constructor
TEST_F(LoopGroupExecutorTest, ParameterizedConstructor)
{
    int streamId = 0;
    int rankId = 0;
    int dieId = 0;
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    LoopGroupExecutor executor(streamId, rankId, dieId, instr, nullptr);
    EXPECT_NO_THROW(executor.Describe());
}

// Test: LoopGroupExecutor Parser with zero values
TEST_F(LoopGroupExecutorTest, ParserZeroValues)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    LoopGroupExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Parser());
    EXPECT_NO_THROW(executor.Describe());
}

// Test: LoopGroupExecutor Parser with max values
TEST_F(LoopGroupExecutorTest, ParserMaxValues)
{
    CcuInstr instr;
    memset(&instr, 0xFF, sizeof(instr));

    LoopGroupExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Parser());
    EXPECT_NO_THROW(executor.Describe());
}

// Test: LoopGroupExecutor Parser with specific parameters
TEST_F(LoopGroupExecutorTest, ParserSpecificParameters)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    instr.v1.loopGroup.startLoopInstrId = 10;
    instr.v1.loopGroup.xnId = 5;
    instr.v1.loopGroup.xmId = 6;
    instr.v1.loopGroup.highPerfModeEn = 1;

    LoopGroupExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Parser());
    std::string desc = executor.Describe();
    EXPECT_FALSE(desc.empty());
}

// Test: LoopGroupExecutor with different highPerfModeEn values
TEST_F(LoopGroupExecutorTest, DifferentHighPerfModeValues)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    for (uint16_t mode = 0; mode <= 1; mode++) {
        instr.v1.loopGroup.highPerfModeEn = mode;
        LoopGroupExecutor executor(0, 0, 0, instr, nullptr);
        EXPECT_NO_THROW(executor.Parser());
        EXPECT_NO_THROW(executor.Describe());
    }
}

// Test: LoopGroupExecutor Describe contains expected keywords
TEST_F(LoopGroupExecutorTest, DescribeContent)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    instr.v1.loopGroup.startLoopInstrId = 5;
    instr.v1.loopGroup.xnId = 10;
    instr.v1.loopGroup.xmId = 15;
    instr.v1.loopGroup.highPerfModeEn = 1;

    LoopGroupExecutor executor(0, 0, 0, instr, nullptr);
    executor.Parser();
    std::string desc = executor.Describe();
    EXPECT_FALSE(desc.empty());
    EXPECT_NE(desc.find("LoopGroup"), std::string::npos);
}

// Test: LoopGroupExecutor inheritance check
TEST_F(LoopGroupExecutorTest, InheritanceCheck)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    LoopGroupExecutor executor(0, 0, 0, instr, nullptr);
    CcuExecutorBase* base = &executor;
    EXPECT_NE(base, nullptr);
}

// Test: LoopGroupExecutor with various xnId and xmId combinations
TEST_F(LoopGroupExecutorTest, VariousXnXmCombinations)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    uint16_t testIds[] = {0, 1, 100, 0x7FFF, 0xFFFF};

    for (auto xnId : testIds) {
        for (auto xmId : testIds) {
            instr.v1.loopGroup.xnId = xnId;
            instr.v1.loopGroup.xmId = xmId;
            LoopGroupExecutor executor(0, 0, 0, instr, nullptr);
            EXPECT_NO_THROW(executor.Parser());
            EXPECT_NO_THROW(executor.Describe());
        }
    }
}
