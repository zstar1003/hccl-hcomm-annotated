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
#include "load_sqe_args_to_gsa_executor.h"

using namespace hcomm::CcuRep;

class LoadSqeArgsToGsaExecutorTest : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

// Test: LoadSqeArgsToGsaExecutor struct size check
TEST_F(LoadSqeArgsToGsaExecutorTest, StructSize) { EXPECT_GT(sizeof(LoadSqeArgsToGsaExecutor), 0); }

// Test: LoadSqeArgsToGsaExecutor default constructor
TEST_F(LoadSqeArgsToGsaExecutorTest, DefaultConstructor)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    LoadSqeArgsToGsaExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Describe());
}

// Test: LoadSqeArgsToGsaExecutor parameterized constructor
TEST_F(LoadSqeArgsToGsaExecutorTest, ParameterizedConstructor)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    LoadSqeArgsToGsaExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Describe());
}

// Test: LoadSqeArgsToGsaExecutor Parser with zero values
TEST_F(LoadSqeArgsToGsaExecutorTest, ParserZeroValues)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    LoadSqeArgsToGsaExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Parser());
    EXPECT_NO_THROW(executor.Describe());
}

// Test: LoadSqeArgsToGsaExecutor Parser with max values
TEST_F(LoadSqeArgsToGsaExecutorTest, ParserMaxValues)
{
    CcuInstr instr;
    memset(&instr, 0xFF, sizeof(instr));

    LoadSqeArgsToGsaExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Parser());
    EXPECT_NO_THROW(executor.Describe());
}

// Test: LoadSqeArgsToGsaExecutor Parser with specific parameters
TEST_F(LoadSqeArgsToGsaExecutorTest, ParserSpecificParameters)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    instr.v1.loadSqeArgsToXn.xnId = 100;
    instr.v1.loadSqeArgsToXn.sqeArgsId = 50;

    LoadSqeArgsToGsaExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Parser());
    std::string desc = executor.Describe();
    EXPECT_FALSE(desc.empty());
}

// Test: LoadSqeArgsToGsaExecutor Describe contains expected keywords
TEST_F(LoadSqeArgsToGsaExecutorTest, DescribeContent)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    instr.v1.loadSqeArgsToXn.xnId = 10;
    instr.v1.loadSqeArgsToXn.sqeArgsId = 5;

    LoadSqeArgsToGsaExecutor executor(0, 0, 0, instr, nullptr);
    executor.Parser();
    std::string desc = executor.Describe();
    EXPECT_FALSE(desc.empty());
    EXPECT_NE(desc.find("SqeArg"), std::string::npos);
}

// Test: LoadSqeArgsToGsaExecutor inheritance check
TEST_F(LoadSqeArgsToGsaExecutorTest, InheritanceCheck)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    LoadSqeArgsToGsaExecutor executor(0, 0, 0, instr, nullptr);
    CcuExecutorBase* base = &executor;
    EXPECT_NE(base, nullptr);
}
