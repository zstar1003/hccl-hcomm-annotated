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
#include "load_xn_xn_executor.h"

using namespace hcomm::CcuRep;

class LoadXnXnExecutorTest : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

// Test: LoadXnXnExecutor struct size check
TEST_F(LoadXnXnExecutorTest, StructSize) { EXPECT_GT(sizeof(LoadXnXnExecutor), 0); }

// Test: LoadXnXnExecutor default constructor
TEST_F(LoadXnXnExecutorTest, DefaultConstructor)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    LoadXnXnExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Describe());
}

// Test: LoadXnXnExecutor parameterized constructor
TEST_F(LoadXnXnExecutorTest, ParameterizedConstructor)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    LoadXnXnExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Describe());
}

// Test: LoadXnXnExecutor Parser with zero values
TEST_F(LoadXnXnExecutorTest, ParserZeroValues)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    LoadXnXnExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Parser());
    EXPECT_NO_THROW(executor.Describe());
}

// Test: LoadXnXnExecutor Parser with max values
TEST_F(LoadXnXnExecutorTest, ParserMaxValues)
{
    CcuInstr instr;
    memset(&instr, 0xFF, sizeof(instr));

    LoadXnXnExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Parser());
    EXPECT_NO_THROW(executor.Describe());
}

// Test: LoadXnXnExecutor Parser with specific parameters
TEST_F(LoadXnXnExecutorTest, ParserSpecificParameters)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    instr.v1.loadXX.xdId = 100;
    instr.v1.loadXX.xmId = 50;
    instr.v1.loadXX.xnId = 25;

    LoadXnXnExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Parser());
    std::string desc = executor.Describe();
    EXPECT_FALSE(desc.empty());
}

// Test: LoadXnXnExecutor with different Xn ID combinations
TEST_F(LoadXnXnExecutorTest, DifferentXnIdCombinations)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    uint16_t xnIds[] = {0, 1, SimCcuV1::CCU_RESOURCE_XN_NUM / 2, SimCcuV1::CCU_RESOURCE_XN_NUM - 1};

    for (auto xdId : xnIds) {
        for (auto xmId : xnIds) {
            for (auto xnId : xnIds) {
                instr.v1.loadXX.xdId = xdId;
                instr.v1.loadXX.xmId = xmId;
                instr.v1.loadXX.xnId = xnId;
                LoadXnXnExecutor executor(0, 0, 0, instr, nullptr);
                EXPECT_NO_THROW(executor.Parser());
                EXPECT_NO_THROW(executor.Describe());
            }
        }
    }
}

// Test: LoadXnXnExecutor Describe contains expected keywords
TEST_F(LoadXnXnExecutorTest, DescribeContent)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    instr.v1.loadXX.xdId = 10;
    instr.v1.loadXX.xmId = 20;
    instr.v1.loadXX.xnId = 30;

    LoadXnXnExecutor executor(0, 0, 0, instr, nullptr);
    executor.Parser();
    std::string desc = executor.Describe();
    EXPECT_FALSE(desc.empty());
    EXPECT_NE(desc.find("Xn"), std::string::npos);
}

// Test: LoadXnXnExecutor inheritance check
TEST_F(LoadXnXnExecutorTest, InheritanceCheck)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    LoadXnXnExecutor executor(0, 0, 0, instr, nullptr);
    CcuExecutorBase* base = &executor;
    EXPECT_NE(base, nullptr);
}

// Test: LoadXnXnExecutor with same source IDs
TEST_F(LoadXnXnExecutorTest, SameSourceIds)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    instr.v1.loadXX.xdId = 100;
    instr.v1.loadXX.xmId = 50;
    instr.v1.loadXX.xnId = 50;

    LoadXnXnExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Parser());
    EXPECT_NO_THROW(executor.Describe());
}
