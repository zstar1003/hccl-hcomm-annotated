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
#include "reduce_min_executor.h"

using namespace hcomm::CcuRep;

class ReduceMinExecutorTest : public testing::Test {
protected:
    void SetUp() override
    {
        auto& mgr = CcuResourceManager::GetInstance();
        mgr.Init(0, 4, RunnerCcuVersion::CCU_V1, {});
    }
    void TearDown() override {}
};

// Test: ReduceMinExecutor struct size check
TEST_F(ReduceMinExecutorTest, StructSize) { EXPECT_GT(sizeof(ReduceMinExecutor), 0); }

// Test: ReduceMinExecutor default constructor
TEST_F(ReduceMinExecutorTest, DefaultConstructor)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    ReduceMinExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Describe());
}

// Test: ReduceMinExecutor parameterized constructor
TEST_F(ReduceMinExecutorTest, ParameterizedConstructor)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    ReduceMinExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Describe());
}

// Test: ReduceMinExecutor Parser with zero values
TEST_F(ReduceMinExecutorTest, ParserZeroValues)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    ReduceMinExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Parser());
    EXPECT_NO_THROW(executor.Describe());
}

// Test: ReduceMinExecutor Parser with max values
TEST_F(ReduceMinExecutorTest, ParserMaxValues)
{
    CcuInstr instr;
    memset(&instr, 0xFF, sizeof(instr));

    ReduceMinExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Parser());
    EXPECT_NO_THROW(executor.Describe());
}

// Test: ReduceMinExecutor Parser with specific parameters
TEST_F(ReduceMinExecutorTest, ParserSpecificParameters)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    instr.v1.min.count = 2;
    instr.v1.min.dataType = 0; // INT16
    instr.v1.min.clearType = 1;
    instr.v1.min.setCKEId = 10;
    instr.v1.min.setCKEMask = 0xFF;
    instr.v1.min.waitCKEId = 5;
    instr.v1.min.waitCKEMask = 0xF0;

    ReduceMinExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Parser());
    std::string desc = executor.Describe();
    EXPECT_FALSE(desc.empty());
}

// Test: ReduceMinExecutor with different data types
TEST_F(ReduceMinExecutorTest, DifferentDataTypes)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    for (uint16_t dataType = 0; dataType <= 8; dataType++) {
        instr.v1.min.dataType = dataType;
        ReduceMinExecutor executor(0, 0, 0, instr, nullptr);
        EXPECT_NO_THROW(executor.Parser());
        EXPECT_NO_THROW(executor.Describe());
    }
}

// Test: ReduceMinExecutor with different count values
TEST_F(ReduceMinExecutorTest, DifferentCountValues)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    uint16_t counts[] = {0, 1, 2, 5, 10, 0xFF};

    for (auto count : counts) {
        instr.v1.min.count = count;
        ReduceMinExecutor executor(0, 0, 0, instr, nullptr);
        EXPECT_NO_THROW(executor.Parser());
        EXPECT_NO_THROW(executor.Describe());
    }
}

// Test: ReduceMinExecutor Describe contains expected keywords
TEST_F(ReduceMinExecutorTest, DescribeContent)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    instr.v1.min.count = 2;
    instr.v1.min.dataType = 0;

    ReduceMinExecutor executor(0, 0, 0, instr, nullptr);
    executor.Parser();
    std::string desc = executor.Describe();
    EXPECT_FALSE(desc.empty());
    EXPECT_NE(desc.find("Min"), std::string::npos);
}

// Test: ReduceMinExecutor inheritance check
TEST_F(ReduceMinExecutorTest, InheritanceCheck)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    ReduceMinExecutor executor(0, 0, 0, instr, nullptr);
    CcuExecutorBase* base = &executor;
    EXPECT_NE(base, nullptr);
}

// Test: ReduceMinExecutor with various MS IDs
TEST_F(ReduceMinExecutorTest, VariousMsIds)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    for (int i = 0; i < 10; i++) {
        instr.v1.min.msId[i] = i * 100;
    }

    ReduceMinExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Parser());
    EXPECT_NO_THROW(executor.Describe());
}

// Test: ReduceMinExecutor with boundary data types
TEST_F(ReduceMinExecutorTest, BoundaryDataTypes)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    uint16_t reservedTypes[] = {3, 5, 7, 9};

    for (auto dataType : reservedTypes) {
        instr.v1.min.dataType = dataType;
        ReduceMinExecutor executor(0, 0, 0, instr, nullptr);
        EXPECT_NO_THROW(executor.Parser());
        EXPECT_NO_THROW(executor.Describe());
    }
}

// Test: ReduceMinExecutor Process with reserved data types (early return paths)
TEST_F(ReduceMinExecutorTest, Process_ReservedDataType1_ReturnsEarly)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    instr.v1.min.count = 1;
    instr.v1.min.dataType = static_cast<uint16_t>(ReduceMaxMinDataType::MAX_MIN_RESERVED1);

    ReduceMinExecutor executor(0, 0, 0, instr, nullptr);
    ASSERT_NO_THROW(executor.Parser());
    CcuResourceManager& ccuResMgr = CcuResourceManager::GetInstance();
    EXPECT_NO_THROW(executor.Process(ccuResMgr));
}

TEST_F(ReduceMinExecutorTest, Process_ReservedDataType2_ReturnsEarly)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    instr.v1.min.count = 1;
    instr.v1.min.dataType = static_cast<uint16_t>(ReduceMaxMinDataType::MAX_MIN_RESERVED2);

    ReduceMinExecutor executor(0, 0, 0, instr, nullptr);
    ASSERT_NO_THROW(executor.Parser());
    CcuResourceManager& ccuResMgr = CcuResourceManager::GetInstance();
    EXPECT_NO_THROW(executor.Process(ccuResMgr));
}

TEST_F(ReduceMinExecutorTest, Process_ReservedDataType3_ReturnsEarly)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    instr.v1.min.count = 1;
    instr.v1.min.dataType = static_cast<uint16_t>(ReduceMaxMinDataType::MAX_MIN_RESERVED3);

    ReduceMinExecutor executor(0, 0, 0, instr, nullptr);
    ASSERT_NO_THROW(executor.Parser());
    CcuResourceManager& ccuResMgr = CcuResourceManager::GetInstance();
    EXPECT_NO_THROW(executor.Process(ccuResMgr));
}

TEST_F(ReduceMinExecutorTest, Process_ReservedDataType4_ReturnsEarly)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    instr.v1.min.count = 1;
    instr.v1.min.dataType = static_cast<uint16_t>(ReduceMaxMinDataType::MAX_MIN_RESERVED4);

    ReduceMinExecutor executor(0, 0, 0, instr, nullptr);
    ASSERT_NO_THROW(executor.Parser());
    CcuResourceManager& ccuResMgr = CcuResourceManager::GetInstance();
    EXPECT_NO_THROW(executor.Process(ccuResMgr));
}

TEST_F(ReduceMinExecutorTest, Process_DataTypeGreaterThanReserved4_ReturnsEarly)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    instr.v1.min.count = 1;
    instr.v1.min.dataType = static_cast<uint16_t>(ReduceMaxMinDataType::MAX_MIN_RESERVED4) + 1;

    ReduceMinExecutor executor(0, 0, 0, instr, nullptr);
    ASSERT_NO_THROW(executor.Parser());
    CcuResourceManager& ccuResMgr = CcuResourceManager::GetInstance();
    EXPECT_NO_THROW(executor.Process(ccuResMgr));
}

// Test: ReduceMinExecutor Describe contains ReduceMin info
TEST_F(ReduceMinExecutorTest, Describe_ContainsReduceMinInfo)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    instr.v1.min.count = 3;
    instr.v1.min.dataType = static_cast<uint16_t>(ReduceMaxMinDataType::MAX_MIN_INT16);
    instr.v1.min.setCKEId = 5;
    instr.v1.min.setCKEMask = 0xAABB;
    instr.v1.min.waitCKEId = 1;
    instr.v1.min.waitCKEMask = 0xCCDD;

    ReduceMinExecutor executor(0, 0, 0, instr, nullptr);
    executor.Parser();
    std::string desc = executor.Describe();
    EXPECT_NE(desc.find("Min"), std::string::npos);
    EXPECT_NE(desc.find("Wait CKE"), std::string::npos);
    EXPECT_NE(desc.find("Set CKE"), std::string::npos);
}
