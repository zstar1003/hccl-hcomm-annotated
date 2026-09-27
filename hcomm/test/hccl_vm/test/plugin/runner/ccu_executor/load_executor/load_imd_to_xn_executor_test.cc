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
#include "load_imd_to_xn_executor.h"

using namespace hcomm::CcuRep;

class LoadImdToXnExecutorTest : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

// Test: LoadImdToXnExecutor struct size check
TEST_F(LoadImdToXnExecutorTest, StructSize) { EXPECT_GT(sizeof(LoadImdToXnExecutor), 0); }

// Test: LoadImdToXnExecutor default constructor
TEST_F(LoadImdToXnExecutorTest, DefaultConstructor)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    LoadImdToXnExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Describe());
}

// Test: LoadImdToXnExecutor parameterized constructor
TEST_F(LoadImdToXnExecutorTest, ParameterizedConstructor)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    LoadImdToXnExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Describe());
}

// Test: LoadImdToXnExecutor Parser with zero values
TEST_F(LoadImdToXnExecutorTest, ParserZeroValues)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    LoadImdToXnExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Parser());
    EXPECT_NO_THROW(executor.Describe());
}

// Test: LoadImdToXnExecutor Parser with max values
TEST_F(LoadImdToXnExecutorTest, ParserMaxValues)
{
    CcuInstr instr;
    memset(&instr, 0xFF, sizeof(instr));

    LoadImdToXnExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Parser());
    EXPECT_NO_THROW(executor.Describe());
}

// Test: LoadImdToXnExecutor Parser with specific parameters
TEST_F(LoadImdToXnExecutorTest, ParserSpecificParameters)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    instr.v1.loadImdToXn.xnId = 100;
    instr.v1.loadImdToXn.immediate = 0x12345678ABCDEF00ULL;

    LoadImdToXnExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Parser());
    std::string desc = executor.Describe();
    EXPECT_FALSE(desc.empty());
}

// Test: LoadImdToXnExecutor with different immediate values
TEST_F(LoadImdToXnExecutorTest, DifferentImmediateValues)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    uint64_t immediates[] = {0, 1, 0x7FFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xDEADBEEFCAFEBABEULL};

    for (auto imm : immediates) {
        instr.v1.loadImdToXn.xnId = 10;
        instr.v1.loadImdToXn.immediate = imm;
        LoadImdToXnExecutor executor(0, 0, 0, instr, nullptr);
        EXPECT_NO_THROW(executor.Parser());
        EXPECT_NO_THROW(executor.Describe());
    }
}

// Test: LoadImdToXnExecutor with different Xn IDs
TEST_F(LoadImdToXnExecutorTest, DifferentXnIds)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    uint16_t xnIds[] = {0, 1, SimCcuV1::CCU_RESOURCE_XN_NUM / 2, SimCcuV1::CCU_RESOURCE_XN_NUM - 1, 0xFFFF};

    for (auto xnId : xnIds) {
        instr.v1.loadImdToXn.xnId = xnId;
        LoadImdToXnExecutor executor(0, 0, 0, instr, nullptr);
        EXPECT_NO_THROW(executor.Parser());
        EXPECT_NO_THROW(executor.Describe());
    }
}

// Test: LoadImdToXnExecutor Describe contains expected keywords
TEST_F(LoadImdToXnExecutorTest, DescribeContent)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    instr.v1.loadImdToXn.xnId = 10;
    instr.v1.loadImdToXn.immediate = 12345;

    LoadImdToXnExecutor executor(0, 0, 0, instr, nullptr);
    executor.Parser();
    std::string desc = executor.Describe();
    EXPECT_FALSE(desc.empty());
    EXPECT_NE(desc.find("immediate"), std::string::npos);
    EXPECT_NE(desc.find("Xn"), std::string::npos);
}

// Test: LoadImdToXnExecutor inheritance check
TEST_F(LoadImdToXnExecutorTest, InheritanceCheck)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    LoadImdToXnExecutor executor(0, 0, 0, instr, nullptr);
    CcuExecutorBase* base = &executor;
    EXPECT_NE(base, nullptr);
}
