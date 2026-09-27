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
#include "reduce_add_executor.h"

using namespace hcomm::CcuRep;

class ReduceAddExecutorTest : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

// Test: ReduceAddExecutor struct size check
TEST_F(ReduceAddExecutorTest, StructSize) { EXPECT_GT(sizeof(ReduceAddExecutor), 0); }

// Test: ReduceAddExecutor default constructor
TEST_F(ReduceAddExecutorTest, DefaultConstructor)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    ReduceAddExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Describe());
}

// Test: ReduceAddExecutor parameterized constructor
TEST_F(ReduceAddExecutorTest, ParameterizedConstructor)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    ReduceAddExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Describe());
}

// Test: ReduceAddExecutor Parser with zero values
TEST_F(ReduceAddExecutorTest, ParserZeroValues)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    ReduceAddExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Parser());
    EXPECT_NO_THROW(executor.Describe());
}

// Test: ReduceAddExecutor Parser with max values
TEST_F(ReduceAddExecutorTest, ParserMaxValues)
{
    CcuInstr instr;
    memset(&instr, 0xFF, sizeof(instr));

    ReduceAddExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Parser());
    EXPECT_NO_THROW(executor.Describe());
}

// Test: ReduceAddExecutor Parser with specific parameters
TEST_F(ReduceAddExecutorTest, ParserSpecificParameters)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    instr.v1.add.count = 2;
    instr.v1.add.castEn = 1;
    instr.v1.add.dataType = 0; // INT16
    instr.v1.add.clearType = 1;
    instr.v1.add.setCKEId = 10;
    instr.v1.add.setCKEMask = 0xFF;
    instr.v1.add.waitCKEId = 5;
    instr.v1.add.waitCKEMask = 0xF0;

    ReduceAddExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Parser());
    std::string desc = executor.Describe();
    EXPECT_FALSE(desc.empty());
}

// Test: ReduceAddExecutor with different data types
TEST_F(ReduceAddExecutorTest, DifferentDataTypes)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    for (uint16_t dataType = 0; dataType <= 6; dataType++) {
        instr.v1.add.dataType = dataType;
        ReduceAddExecutor executor(0, 0, 0, instr, nullptr);
        EXPECT_NO_THROW(executor.Parser());
        EXPECT_NO_THROW(executor.Describe());
    }
}

// Test: ReduceAddExecutor with different count values
TEST_F(ReduceAddExecutorTest, DifferentCountValues)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    uint16_t counts[] = {0, 1, 2, 5, 10, 0xFF};

    for (auto count : counts) {
        instr.v1.add.count = count;
        ReduceAddExecutor executor(0, 0, 0, instr, nullptr);
        EXPECT_NO_THROW(executor.Parser());
        EXPECT_NO_THROW(executor.Describe());
    }
}

// Test: ReduceAddExecutor with different castEn values
TEST_F(ReduceAddExecutorTest, DifferentCastEnValues)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    for (uint16_t castEn = 0; castEn <= 1; castEn++) {
        instr.v1.add.castEn = castEn;
        ReduceAddExecutor executor(0, 0, 0, instr, nullptr);
        EXPECT_NO_THROW(executor.Parser());
        EXPECT_NO_THROW(executor.Describe());
    }
}

// Test: ReduceAddExecutor Describe contains expected keywords
TEST_F(ReduceAddExecutorTest, DescribeContent)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    instr.v1.add.count = 2;
    instr.v1.add.dataType = 0;

    ReduceAddExecutor executor(0, 0, 0, instr, nullptr);
    executor.Parser();
    std::string desc = executor.Describe();
    EXPECT_FALSE(desc.empty());
    EXPECT_NE(desc.find("Add"), std::string::npos);
}

// Test: ReduceAddExecutor inheritance check
TEST_F(ReduceAddExecutorTest, InheritanceCheck)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    ReduceAddExecutor executor(0, 0, 0, instr, nullptr);
    CcuExecutorBase* base = &executor;
    EXPECT_NE(base, nullptr);
}

// Test: ReduceAddExecutor with various MS IDs
TEST_F(ReduceAddExecutorTest, VariousMsIds)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    for (int i = 0; i < 10; i++) {
        instr.v1.add.msId[i] = i * 100;
    }

    ReduceAddExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Parser());
    EXPECT_NO_THROW(executor.Describe());
}
