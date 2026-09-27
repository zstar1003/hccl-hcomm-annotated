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
#include "reduce_max_executor.h"

using namespace hcomm::CcuRep;

class ReduceMaxExecutorTest : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

// Test: ReduceMaxExecutor struct size check
TEST_F(ReduceMaxExecutorTest, StructSize) { EXPECT_GT(sizeof(ReduceMaxExecutor), 0); }

// Test: ReduceMaxExecutor default constructor
TEST_F(ReduceMaxExecutorTest, DefaultConstructor)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    ReduceMaxExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Describe());
}

// Test: ReduceMaxExecutor parameterized constructor
TEST_F(ReduceMaxExecutorTest, ParameterizedConstructor)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    ReduceMaxExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Describe());
}

// Test: ReduceMaxExecutor Parser with zero values
TEST_F(ReduceMaxExecutorTest, ParserZeroValues)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    ReduceMaxExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Parser());
    EXPECT_NO_THROW(executor.Describe());
}

// Test: ReduceMaxExecutor Parser with max values
TEST_F(ReduceMaxExecutorTest, ParserMaxValues)
{
    CcuInstr instr;
    memset(&instr, 0xFF, sizeof(instr));

    ReduceMaxExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Parser());
    EXPECT_NO_THROW(executor.Describe());
}

// Test: ReduceMaxExecutor Parser with specific parameters
TEST_F(ReduceMaxExecutorTest, ParserSpecificParameters)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    instr.v1.max.count = 2;
    instr.v1.max.dataType = 0; // INT16
    instr.v1.max.clearType = 1;
    instr.v1.max.setCKEId = 10;
    instr.v1.max.setCKEMask = 0xFF;
    instr.v1.max.waitCKEId = 5;
    instr.v1.max.waitCKEMask = 0xF0;

    ReduceMaxExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Parser());
    std::string desc = executor.Describe();
    EXPECT_FALSE(desc.empty());
}

// Test: ReduceMaxExecutor with different data types
TEST_F(ReduceMaxExecutorTest, DifferentDataTypes)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    for (uint16_t dataType = 0; dataType <= 8; dataType++) {
        instr.v1.max.dataType = dataType;
        ReduceMaxExecutor executor(0, 0, 0, instr, nullptr);
        EXPECT_NO_THROW(executor.Parser());
        EXPECT_NO_THROW(executor.Describe());
    }
}

// Test: ReduceMaxExecutor with different count values
TEST_F(ReduceMaxExecutorTest, DifferentCountValues)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    uint16_t counts[] = {0, 1, 2, 5, 10, 0xFF};

    for (auto count : counts) {
        instr.v1.max.count = count;
        ReduceMaxExecutor executor(0, 0, 0, instr, nullptr);
        EXPECT_NO_THROW(executor.Parser());
        EXPECT_NO_THROW(executor.Describe());
    }
}

// Test: ReduceMaxExecutor Describe contains expected keywords
TEST_F(ReduceMaxExecutorTest, DescribeContent)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    instr.v1.max.count = 2;
    instr.v1.max.dataType = 0;

    ReduceMaxExecutor executor(0, 0, 0, instr, nullptr);
    executor.Parser();
    std::string desc = executor.Describe();
    EXPECT_FALSE(desc.empty());
    EXPECT_NE(desc.find("Max"), std::string::npos);
}

// Test: ReduceMaxExecutor inheritance check
TEST_F(ReduceMaxExecutorTest, InheritanceCheck)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    ReduceMaxExecutor executor(0, 0, 0, instr, nullptr);
    CcuExecutorBase* base = &executor;
    EXPECT_NE(base, nullptr);
}

// Test: ReduceMaxExecutor with various MS IDs
TEST_F(ReduceMaxExecutorTest, VariousMsIds)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    for (int i = 0; i < 10; i++) {
        instr.v1.max.msId[i] = i * 100;
    }

    ReduceMaxExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Parser());
    EXPECT_NO_THROW(executor.Describe());
}
