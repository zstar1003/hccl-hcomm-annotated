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
#include "reduce_max_executor.h"
#include "reduce_min_executor.h"

using namespace hcomm::CcuRep;

// ReduceAddExecutor test fixture
class ReduceAddExecutorTest : public testing::Test {
protected:
    void SetUp() override
    {
        auto& mgr = CcuResourceManager::GetInstance();
        mgr.Init(0, 4, RunnerCcuVersion::CCU_V1, {});
    }
    void TearDown() override {}
};

// ReduceMaxExecutor test fixture
class ReduceMaxExecutorTest : public testing::Test {
protected:
    void SetUp() override
    {
        auto& mgr = CcuResourceManager::GetInstance();
        mgr.Init(0, 4, RunnerCcuVersion::CCU_V1, {});
    }
    void TearDown() override {}
};

// ReduceMinExecutor test fixture
class ReduceMinExecutorTest : public testing::Test {
protected:
    void SetUp() override
    {
        auto& mgr = CcuResourceManager::GetInstance();
        mgr.Init(0, 4, RunnerCcuVersion::CCU_V1, {});
    }
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

TEST_F(ReduceMaxExecutorTest, Parser_SpecificHighBitCount)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    instr.v1.max.count = 5;
    instr.v1.max.dataType = static_cast<uint16_t>(ReduceMaxMinDataType::MAX_MIN_INT32);
    instr.v1.max.clearType = 1;
    instr.v1.max.setCKEId = 7;
    instr.v1.max.setCKEMask = 0x7FFF;
    instr.v1.max.waitCKEId = 3;
    instr.v1.max.waitCKEMask = 0xFFFF;

    ReduceMaxExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Parser());
    std::string desc = executor.Describe();
    EXPECT_FALSE(desc.empty());
}

TEST_F(ReduceMaxExecutorTest, Describe_ContainsReduceMaxInfo)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    instr.v1.max.count = 3;
    instr.v1.max.dataType = static_cast<uint16_t>(ReduceMaxMinDataType::MAX_MIN_INT16);
    instr.v1.max.setCKEId = 5;
    instr.v1.max.setCKEMask = 0xAABB;
    instr.v1.max.waitCKEId = 1;
    instr.v1.max.waitCKEMask = 0xCCDD;

    ReduceMaxExecutor executor(0, 0, 0, instr, nullptr);
    executor.Parser();
    std::string desc = executor.Describe();
    EXPECT_NE(desc.find("Max"), std::string::npos);
    EXPECT_NE(desc.find("Wait CKE"), std::string::npos);
    EXPECT_NE(desc.find("Set CKE"), std::string::npos);
}

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

TEST_F(ReduceAddExecutorTest, Process_UnsupportedDataType_ReturnsEarly)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    instr.v1.add.count = 1;
    instr.v1.add.dataType = 9;
    ReduceAddExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Parser());
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

TEST_F(ReduceMinExecutorTest, Process_SupportedDataType_INT16)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    instr.v1.min.count = 0;
    instr.v1.min.dataType = static_cast<uint16_t>(ReduceMaxMinDataType::MAX_MIN_INT16);
    instr.v1.min.msId[0] = 0;
    instr.v1.min.msId[1] = 1;

    auto simulator = std::make_unique<CcuSimulator>(0, 0, 0, 10, 10, RunnerCcuVersion::CCU_V1);
    ReduceMinExecutor executor(0, 0, 0, instr, simulator.get());
    ASSERT_NO_THROW(executor.Parser());
    CcuResourceManager& ccuResMgr = CcuResourceManager::GetInstance();
    EXPECT_NO_THROW(executor.Process(ccuResMgr));
}

TEST_F(ReduceMinExecutorTest, Process_SupportedDataType_INT32)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    instr.v1.min.count = 0;
    instr.v1.min.dataType = static_cast<uint16_t>(ReduceMaxMinDataType::MAX_MIN_INT32);
    instr.v1.min.msId[0] = 0;
    instr.v1.min.msId[1] = 1;

    auto simulator = std::make_unique<CcuSimulator>(0, 0, 0, 10, 10, RunnerCcuVersion::CCU_V1);
    ReduceMinExecutor executor(0, 0, 0, instr, simulator.get());
    ASSERT_NO_THROW(executor.Parser());
    CcuResourceManager& ccuResMgr = CcuResourceManager::GetInstance();
    EXPECT_NO_THROW(executor.Process(ccuResMgr));
}

TEST_F(ReduceMinExecutorTest, Process_SupportedDataType_UINT8)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    instr.v1.min.count = 0;
    instr.v1.min.dataType = static_cast<uint16_t>(ReduceMaxMinDataType::MAX_MIN_UINT8);
    instr.v1.min.msId[0] = 0;
    instr.v1.min.msId[1] = 1;

    auto simulator = std::make_unique<CcuSimulator>(0, 0, 0, 10, 10, RunnerCcuVersion::CCU_V1);
    ReduceMinExecutor executor(0, 0, 0, instr, simulator.get());
    ASSERT_NO_THROW(executor.Parser());
    CcuResourceManager& ccuResMgr = CcuResourceManager::GetInstance();
    EXPECT_NO_THROW(executor.Process(ccuResMgr));
}

TEST_F(ReduceMinExecutorTest, Process_SupportedDataType_INT8)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    instr.v1.min.count = 0;
    instr.v1.min.dataType = static_cast<uint16_t>(ReduceMaxMinDataType::MAX_MIN_INT8);
    instr.v1.min.msId[0] = 0;
    instr.v1.min.msId[1] = 1;

    auto simulator = std::make_unique<CcuSimulator>(0, 0, 0, 10, 10, RunnerCcuVersion::CCU_V1);
    ReduceMinExecutor executor(0, 0, 0, instr, simulator.get());
    ASSERT_NO_THROW(executor.Parser());
    CcuResourceManager& ccuResMgr = CcuResourceManager::GetInstance();
    EXPECT_NO_THROW(executor.Process(ccuResMgr));
}

TEST_F(ReduceMinExecutorTest, Process_SupportedDataType_FP32)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    instr.v1.min.count = 0;
    instr.v1.min.dataType = static_cast<uint16_t>(ReduceMaxMinDataType::MAX_MIN_FP32);
    instr.v1.min.msId[0] = 0;
    instr.v1.min.msId[1] = 1;

    auto simulator = std::make_unique<CcuSimulator>(0, 0, 0, 10, 10, RunnerCcuVersion::CCU_V1);
    ReduceMinExecutor executor(0, 0, 0, instr, simulator.get());
    ASSERT_NO_THROW(executor.Parser());
    CcuResourceManager& ccuResMgr = CcuResourceManager::GetInstance();
    EXPECT_NO_THROW(executor.Process(ccuResMgr));
}

TEST_F(ReduceMinExecutorTest, Process_SupportedDataType_FP16)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    instr.v1.min.count = 0;
    instr.v1.min.dataType = static_cast<uint16_t>(ReduceMaxMinDataType::MAX_MIN_FP16);
    instr.v1.min.msId[0] = 0;
    instr.v1.min.msId[1] = 1;

    auto simulator = std::make_unique<CcuSimulator>(0, 0, 0, 10, 10, RunnerCcuVersion::CCU_V1);
    ReduceMinExecutor executor(0, 0, 0, instr, simulator.get());
    ASSERT_NO_THROW(executor.Parser());
    CcuResourceManager& ccuResMgr = CcuResourceManager::GetInstance();
    EXPECT_NO_THROW(executor.Process(ccuResMgr));
}

TEST_F(ReduceMinExecutorTest, Process_LoopState_Normal)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    instr.v1.min.count = 0;
    instr.v1.min.dataType = static_cast<uint16_t>(ReduceMaxMinDataType::MAX_MIN_INT16);
    instr.v1.min.msId[0] = 0;
    instr.v1.min.msId[1] = 1;

    auto simulator = std::make_unique<CcuSimulator>(0, 0, 0, 10, 10, RunnerCcuVersion::CCU_V1);
    simulator->SetExecState(CcuExecState::EXEC_LOOP_INSTR);
    LoopGroupInfo loopGroupInfo;
    loopGroupInfo.msOffset_ = 3;
    simulator->InitLoopGroupInfo(loopGroupInfo);

    ReduceMinExecutor executor(0, 0, 0, instr, simulator.get());
    ASSERT_NO_THROW(executor.Parser());
    CcuResourceManager& ccuResMgr = CcuResourceManager::GetInstance();
    EXPECT_NO_THROW(executor.Process(ccuResMgr));
}

TEST_F(ReduceMinExecutorTest, Run_Normal)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    instr.v1.min.count = 0;
    instr.v1.min.dataType = static_cast<uint16_t>(ReduceMaxMinDataType::MAX_MIN_INT16);
    instr.v1.min.clearType = 0;
    instr.v1.min.waitCKEId = 0;
    instr.v1.min.waitCKEMask = 0xFFFF;
    instr.v1.min.msId[0] = 0;
    instr.v1.min.msId[1] = 1;

    auto simulator = std::make_unique<CcuSimulator>(0, 0, 0, 10, 10, RunnerCcuVersion::CCU_V1);
    ReduceMinExecutor executor(0, 0, 0, instr, simulator.get());
    ASSERT_NO_THROW(executor.Parser());

    CcuResourceManager& ccuResMgr = CcuResourceManager::GetInstance();
    ccuResMgr.UpdateCkeValue(0, 0, 0, 0xFFFF);

    EXPECT_NO_THROW(executor.Run());
}

// ReduceMaxExecutor Process tests
TEST_F(ReduceMaxExecutorTest, Process_ReservedDataType_ReturnsEarly)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    instr.v1.max.count = 1;
    instr.v1.max.dataType = static_cast<uint16_t>(ReduceMaxMinDataType::MAX_MIN_RESERVED4);

    ReduceMaxExecutor executor(0, 0, 0, instr, nullptr);
    ASSERT_NO_THROW(executor.Parser());
    CcuResourceManager& ccuResMgr = CcuResourceManager::GetInstance();
    EXPECT_NO_THROW(executor.Process(ccuResMgr));
}

TEST_F(ReduceMaxExecutorTest, Process_SupportedDataType_INT16)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    instr.v1.max.count = 0;
    instr.v1.max.dataType = static_cast<uint16_t>(ReduceMaxMinDataType::MAX_MIN_INT16);
    instr.v1.max.msId[0] = 0;
    instr.v1.max.msId[1] = 1;

    auto simulator = std::make_unique<CcuSimulator>(0, 0, 0, 10, 10, RunnerCcuVersion::CCU_V1);
    ReduceMaxExecutor executor(0, 0, 0, instr, simulator.get());
    ASSERT_NO_THROW(executor.Parser());
    CcuResourceManager& ccuResMgr = CcuResourceManager::GetInstance();
    EXPECT_NO_THROW(executor.Process(ccuResMgr));
}

TEST_F(ReduceMaxExecutorTest, Process_SupportedDataType_INT32)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    instr.v1.max.count = 0;
    instr.v1.max.dataType = static_cast<uint16_t>(ReduceMaxMinDataType::MAX_MIN_INT32);
    instr.v1.max.msId[0] = 0;
    instr.v1.max.msId[1] = 1;

    auto simulator = std::make_unique<CcuSimulator>(0, 0, 0, 10, 10, RunnerCcuVersion::CCU_V1);
    ReduceMaxExecutor executor(0, 0, 0, instr, simulator.get());
    ASSERT_NO_THROW(executor.Parser());
    CcuResourceManager& ccuResMgr = CcuResourceManager::GetInstance();
    EXPECT_NO_THROW(executor.Process(ccuResMgr));
}

TEST_F(ReduceMaxExecutorTest, Process_SupportedDataType_UINT8)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    instr.v1.max.count = 0;
    instr.v1.max.dataType = static_cast<uint16_t>(ReduceMaxMinDataType::MAX_MIN_UINT8);
    instr.v1.max.msId[0] = 0;
    instr.v1.max.msId[1] = 1;

    auto simulator = std::make_unique<CcuSimulator>(0, 0, 0, 10, 10, RunnerCcuVersion::CCU_V1);
    ReduceMaxExecutor executor(0, 0, 0, instr, simulator.get());
    ASSERT_NO_THROW(executor.Parser());
    CcuResourceManager& ccuResMgr = CcuResourceManager::GetInstance();
    EXPECT_NO_THROW(executor.Process(ccuResMgr));
}

TEST_F(ReduceMaxExecutorTest, Process_SupportedDataType_FP32)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    instr.v1.max.count = 0;
    instr.v1.max.dataType = static_cast<uint16_t>(ReduceMaxMinDataType::MAX_MIN_FP32);
    instr.v1.max.msId[0] = 0;
    instr.v1.max.msId[1] = 1;

    auto simulator = std::make_unique<CcuSimulator>(0, 0, 0, 10, 10, RunnerCcuVersion::CCU_V1);
    ReduceMaxExecutor executor(0, 0, 0, instr, simulator.get());
    ASSERT_NO_THROW(executor.Parser());
    CcuResourceManager& ccuResMgr = CcuResourceManager::GetInstance();
    EXPECT_NO_THROW(executor.Process(ccuResMgr));
}

TEST_F(ReduceMaxExecutorTest, Process_SupportedDataType_INT8)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    instr.v1.max.count = 0;
    instr.v1.max.dataType = static_cast<uint16_t>(ReduceMaxMinDataType::MAX_MIN_INT8);
    instr.v1.max.msId[0] = 0;
    instr.v1.max.msId[1] = 1;

    auto simulator = std::make_unique<CcuSimulator>(0, 0, 0, 10, 10, RunnerCcuVersion::CCU_V1);
    ReduceMaxExecutor executor(0, 0, 0, instr, simulator.get());
    ASSERT_NO_THROW(executor.Parser());
    CcuResourceManager& ccuResMgr = CcuResourceManager::GetInstance();
    EXPECT_NO_THROW(executor.Process(ccuResMgr));
}

TEST_F(ReduceMaxExecutorTest, Process_SupportedDataType_FP16)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    instr.v1.max.count = 0;
    instr.v1.max.dataType = static_cast<uint16_t>(ReduceMaxMinDataType::MAX_MIN_FP16);
    instr.v1.max.msId[0] = 0;
    instr.v1.max.msId[1] = 1;

    auto simulator = std::make_unique<CcuSimulator>(0, 0, 0, 10, 10, RunnerCcuVersion::CCU_V1);
    ReduceMaxExecutor executor(0, 0, 0, instr, simulator.get());
    ASSERT_NO_THROW(executor.Parser());
    CcuResourceManager& ccuResMgr = CcuResourceManager::GetInstance();
    EXPECT_NO_THROW(executor.Process(ccuResMgr));
}

TEST_F(ReduceMaxExecutorTest, Process_LoopState_Normal)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    instr.v1.max.count = 0;
    instr.v1.max.dataType = static_cast<uint16_t>(ReduceMaxMinDataType::MAX_MIN_INT16);
    instr.v1.max.msId[0] = 0;
    instr.v1.max.msId[1] = 1;

    auto simulator = std::make_unique<CcuSimulator>(0, 0, 0, 10, 10, RunnerCcuVersion::CCU_V1);
    simulator->SetExecState(CcuExecState::EXEC_LOOP_INSTR);
    LoopGroupInfo loopGroupInfo;
    loopGroupInfo.msOffset_ = 3;
    simulator->InitLoopGroupInfo(loopGroupInfo);

    ReduceMaxExecutor executor(0, 0, 0, instr, simulator.get());
    ASSERT_NO_THROW(executor.Parser());
    CcuResourceManager& ccuResMgr = CcuResourceManager::GetInstance();
    EXPECT_NO_THROW(executor.Process(ccuResMgr));
}

TEST_F(ReduceMaxExecutorTest, Run_Normal)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    instr.v1.max.count = 0;
    instr.v1.max.dataType = static_cast<uint16_t>(ReduceMaxMinDataType::MAX_MIN_INT16);
    instr.v1.max.clearType = 0;
    instr.v1.max.waitCKEId = 0;
    instr.v1.max.waitCKEMask = 0xFFFF;
    instr.v1.max.msId[0] = 0;
    instr.v1.max.msId[1] = 1;

    auto simulator = std::make_unique<CcuSimulator>(0, 0, 0, 10, 10, RunnerCcuVersion::CCU_V1);
    ReduceMaxExecutor executor(0, 0, 0, instr, simulator.get());
    ASSERT_NO_THROW(executor.Parser());

    CcuResourceManager& ccuResMgr = CcuResourceManager::GetInstance();
    ccuResMgr.UpdateCkeValue(0, 0, 0, 0xFFFF);

    EXPECT_NO_THROW(executor.Run());
}

// ReduceAddExecutor Process tests
TEST_F(ReduceAddExecutorTest, Process_SupportedDataType_INT16)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    instr.v1.add.count = 0;
    instr.v1.add.dataType = static_cast<uint16_t>(ReduceAddDataType::ADD_INT16);
    instr.v1.add.msId[0] = 0;
    instr.v1.add.msId[1] = 1;

    auto simulator = std::make_unique<CcuSimulator>(0, 0, 0, 10, 10, RunnerCcuVersion::CCU_V1);
    ReduceAddExecutor executor(0, 0, 0, instr, simulator.get());
    ASSERT_NO_THROW(executor.Parser());
    CcuResourceManager& ccuResMgr = CcuResourceManager::GetInstance();
    EXPECT_NO_THROW(executor.Process(ccuResMgr));
}

TEST_F(ReduceAddExecutorTest, Process_SupportedDataType_INT32)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    instr.v1.add.count = 0;
    instr.v1.add.dataType = static_cast<uint16_t>(ReduceAddDataType::ADD_INT32);
    instr.v1.add.msId[0] = 0;
    instr.v1.add.msId[1] = 1;

    auto simulator = std::make_unique<CcuSimulator>(0, 0, 0, 10, 10, RunnerCcuVersion::CCU_V1);
    ReduceAddExecutor executor(0, 0, 0, instr, simulator.get());
    ASSERT_NO_THROW(executor.Parser());
    CcuResourceManager& ccuResMgr = CcuResourceManager::GetInstance();
    EXPECT_NO_THROW(executor.Process(ccuResMgr));
}

TEST_F(ReduceAddExecutorTest, Process_SupportedDataType_UINT8)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    instr.v1.add.count = 0;
    instr.v1.add.dataType = static_cast<uint16_t>(ReduceAddDataType::ADD_UINT8);
    instr.v1.add.msId[0] = 0;
    instr.v1.add.msId[1] = 1;

    auto simulator = std::make_unique<CcuSimulator>(0, 0, 0, 10, 10, RunnerCcuVersion::CCU_V1);
    ReduceAddExecutor executor(0, 0, 0, instr, simulator.get());
    ASSERT_NO_THROW(executor.Parser());
    CcuResourceManager& ccuResMgr = CcuResourceManager::GetInstance();
    EXPECT_NO_THROW(executor.Process(ccuResMgr));
}

TEST_F(ReduceAddExecutorTest, Process_SupportedDataType_INT8)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    instr.v1.add.count = 0;
    instr.v1.add.dataType = static_cast<uint16_t>(ReduceAddDataType::ADD_INT8);
    instr.v1.add.msId[0] = 0;
    instr.v1.add.msId[1] = 1;

    auto simulator = std::make_unique<CcuSimulator>(0, 0, 0, 10, 10, RunnerCcuVersion::CCU_V1);
    ReduceAddExecutor executor(0, 0, 0, instr, simulator.get());
    ASSERT_NO_THROW(executor.Parser());
    CcuResourceManager& ccuResMgr = CcuResourceManager::GetInstance();
    EXPECT_NO_THROW(executor.Process(ccuResMgr));
}

TEST_F(ReduceAddExecutorTest, Process_SupportedDataType_FP32)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    instr.v1.add.count = 0;
    instr.v1.add.dataType = static_cast<uint16_t>(ReduceAddDataType::ADD_FP32);
    instr.v1.add.msId[0] = 0;
    instr.v1.add.msId[1] = 1;

    auto simulator = std::make_unique<CcuSimulator>(0, 0, 0, 10, 10, RunnerCcuVersion::CCU_V1);
    ReduceAddExecutor executor(0, 0, 0, instr, simulator.get());
    ASSERT_NO_THROW(executor.Parser());
    CcuResourceManager& ccuResMgr = CcuResourceManager::GetInstance();
    EXPECT_NO_THROW(executor.Process(ccuResMgr));
}

TEST_F(ReduceAddExecutorTest, Process_SupportedDataType_FP16)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    instr.v1.add.count = 0;
    instr.v1.add.dataType = static_cast<uint16_t>(ReduceAddDataType::ADD_FP16);
    instr.v1.add.msId[0] = 0;
    instr.v1.add.msId[1] = 1;

    auto simulator = std::make_unique<CcuSimulator>(0, 0, 0, 10, 10, RunnerCcuVersion::CCU_V1);
    ReduceAddExecutor executor(0, 0, 0, instr, simulator.get());
    ASSERT_NO_THROW(executor.Parser());
    CcuResourceManager& ccuResMgr = CcuResourceManager::GetInstance();
    EXPECT_NO_THROW(executor.Process(ccuResMgr));
}

TEST_F(ReduceAddExecutorTest, Process_LoopState_Normal)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    instr.v1.add.count = 0;
    instr.v1.add.dataType = static_cast<uint16_t>(ReduceAddDataType::ADD_INT16);
    instr.v1.add.msId[0] = 0;
    instr.v1.add.msId[1] = 1;

    auto simulator = std::make_unique<CcuSimulator>(0, 0, 0, 10, 10, RunnerCcuVersion::CCU_V1);
    simulator->SetExecState(CcuExecState::EXEC_LOOP_INSTR);
    LoopGroupInfo loopGroupInfo;
    loopGroupInfo.msOffset_ = 3;
    simulator->InitLoopGroupInfo(loopGroupInfo);

    ReduceAddExecutor executor(0, 0, 0, instr, simulator.get());
    ASSERT_NO_THROW(executor.Parser());
    CcuResourceManager& ccuResMgr = CcuResourceManager::GetInstance();
    EXPECT_NO_THROW(executor.Process(ccuResMgr));
}

TEST_F(ReduceAddExecutorTest, Run_Normal)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    instr.v1.add.count = 0;
    instr.v1.add.dataType = static_cast<uint16_t>(ReduceAddDataType::ADD_INT16);
    instr.v1.add.clearType = 0;
    instr.v1.add.waitCKEId = 0;
    instr.v1.add.waitCKEMask = 0xFFFF;
    instr.v1.add.msId[0] = 0;
    instr.v1.add.msId[1] = 1;

    auto simulator = std::make_unique<CcuSimulator>(0, 0, 0, 10, 10, RunnerCcuVersion::CCU_V1);
    ReduceAddExecutor executor(0, 0, 0, instr, simulator.get());
    ASSERT_NO_THROW(executor.Parser());

    CcuResourceManager& ccuResMgr = CcuResourceManager::GetInstance();
    ccuResMgr.UpdateCkeValue(0, 0, 0, 0xFFFF);

    EXPECT_NO_THROW(executor.Run());
}
