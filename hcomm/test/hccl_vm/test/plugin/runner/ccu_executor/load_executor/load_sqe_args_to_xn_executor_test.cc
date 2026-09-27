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
#include "load_sqe_args_to_xn_executor.h"

using namespace hcomm::CcuRep;

class LoadSqeArgsToXnExecutorTest : public testing::Test {
protected:
    void SetUp() override
    {
        CcuResourceManager::GetInstance().Init(0, 1, RunnerCcuVersion::CCU_V1, std::vector<uint64_t>{});
    }
    void TearDown() override {}
};

// Test: LoadSqeArgsToXnExecutor struct size check
TEST_F(LoadSqeArgsToXnExecutorTest, StructSize) { EXPECT_GT(sizeof(LoadSqeArgsToXnExecutor), 0); }

// Test: LoadSqeArgsToXnExecutor default constructor
TEST_F(LoadSqeArgsToXnExecutorTest, DefaultConstructor)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    LoadSqeArgsToXnExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Describe());
}

// Test: LoadSqeArgsToXnExecutor parameterized constructor
TEST_F(LoadSqeArgsToXnExecutorTest, ParameterizedConstructor)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    LoadSqeArgsToXnExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Describe());
}

// Test: LoadSqeArgsToXnExecutor Parser with zero values
TEST_F(LoadSqeArgsToXnExecutorTest, ParserZeroValues)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    LoadSqeArgsToXnExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Parser());
    EXPECT_NO_THROW(executor.Describe());
}

// Test: LoadSqeArgsToXnExecutor Parser with max values
TEST_F(LoadSqeArgsToXnExecutorTest, ParserMaxValues)
{
    CcuInstr instr;
    memset(&instr, 0xFF, sizeof(instr));

    LoadSqeArgsToXnExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Parser());
    EXPECT_NO_THROW(executor.Describe());
}

// Test: LoadSqeArgsToXnExecutor Parser with specific parameters
TEST_F(LoadSqeArgsToXnExecutorTest, ParserSpecificParameters)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    instr.v1.loadSqeArgsToXn.xnId = 100;
    instr.v1.loadSqeArgsToXn.sqeArgsId = 50;

    LoadSqeArgsToXnExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Parser());
    std::string desc = executor.Describe();
    EXPECT_FALSE(desc.empty());
}

// Test: LoadSqeArgsToXnExecutor with different Xn and SQE Arg IDs
TEST_F(LoadSqeArgsToXnExecutorTest, DifferentIdCombinations)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    uint16_t ids[] = {0, 1, 100, 0x7FFF, 0xFFFF};

    for (auto xnId : ids) {
        for (auto sqeArgId : ids) {
            instr.v1.loadSqeArgsToXn.xnId = xnId;
            instr.v1.loadSqeArgsToXn.sqeArgsId = sqeArgId;
            LoadSqeArgsToXnExecutor executor(0, 0, 0, instr, nullptr);
            EXPECT_NO_THROW(executor.Parser());
            EXPECT_NO_THROW(executor.Describe());
        }
    }
}

// Test: LoadSqeArgsToXnExecutor Describe contains expected keywords
TEST_F(LoadSqeArgsToXnExecutorTest, DescribeContent)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    instr.v1.loadSqeArgsToXn.xnId = 10;
    instr.v1.loadSqeArgsToXn.sqeArgsId = 5;

    LoadSqeArgsToXnExecutor executor(0, 0, 0, instr, nullptr);
    executor.Parser();
    std::string desc = executor.Describe();
    EXPECT_FALSE(desc.empty());
    EXPECT_NE(desc.find("SqeArg"), std::string::npos);
    EXPECT_NE(desc.find("Xn"), std::string::npos);
}

// Test: LoadSqeArgsToXnExecutor inheritance check
TEST_F(LoadSqeArgsToXnExecutorTest, InheritanceCheck)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    LoadSqeArgsToXnExecutor executor(0, 0, 0, instr, nullptr);
    CcuExecutorBase* base = &executor;
    EXPECT_NE(base, nullptr);
}
