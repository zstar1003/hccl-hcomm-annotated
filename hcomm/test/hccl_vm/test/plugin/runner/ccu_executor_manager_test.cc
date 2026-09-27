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
#include <memory>

#include "ccu_executor_manager.h"
#include "ccu_microcode_common_v1.h"
#include "ccu_simulator.h"

using namespace hcomm::CcuRep;

// Mock executor for testing factory
class TestExecutor : public CcuExecutorBase {
public:
    TestExecutor(int streamId, int rankId, int dieId, const CcuInstr& instr, CcuSimulator* ccuSimulator)
        : CcuExecutorBase(streamId, rankId, dieId, instr, ccuSimulator)
    {}

    void Parser() override {}
    void Run() override {}
    std::string Describe() override { return "TestExecutor"; }
};

class CcuExecutorManagerTest : public testing::Test {
protected:
    void SetUp() override { memset(&instr_, 0, sizeof(instr_)); }

    void TearDown() override {}

    CcuInstr instr_;
};

// Test: Singleton instance
TEST_F(CcuExecutorManagerTest, SingletonInstance)
{
    CcuExecutorCreateFuncMgr& instance1 = CcuExecutorCreateFuncMgr::Instance();
    CcuExecutorCreateFuncMgr& instance2 = CcuExecutorCreateFuncMgr::Instance();

    EXPECT_EQ(&instance1, &instance2);
}

// Test: RegFunc and GetFunc
TEST_F(CcuExecutorManagerTest, RegFuncAndGetFunc)
{
    uint16_t testType = 0x1234;

    auto createFunc = [](int streamId, int rankId, int dieId, const CcuInstr& instr, CcuSimulator* ccuSimulator) {
        return std::make_unique<TestExecutor>(streamId, rankId, dieId, instr, ccuSimulator);
    };

    CcuExecutorCreateFuncMgr::Instance().RegFunc(RunnerCcuVersion::CCU_V1, testType, createFunc);

    auto retrievedFunc = CcuExecutorCreateFuncMgr::Instance().GetFunc(RunnerCcuVersion::CCU_V1, testType);
    EXPECT_NE(retrievedFunc, nullptr);
}

// Test: GetFunc for non-existent type
TEST_F(CcuExecutorManagerTest, GetFuncNonExistent)
{
    uint16_t nonExistentType = 0xFFFF;

    auto retrievedFunc = CcuExecutorCreateFuncMgr::Instance().GetFunc(RunnerCcuVersion::CCU_V1, nonExistentType);
    EXPECT_EQ(retrievedFunc, nullptr);
}

// Test: CcuExecutorFactory MakeCcuExecutorInstance with registered type
TEST_F(CcuExecutorManagerTest, FactoryMakeInstanceRegistered)
{
    uint16_t testType = 0x5678;

    auto createFunc = [](int streamId, int rankId, int dieId, const CcuInstr& instr, CcuSimulator* ccuSimulator) {
        return std::make_unique<TestExecutor>(streamId, rankId, dieId, instr, ccuSimulator);
    };

    CcuExecutorCreateFuncMgr::Instance().RegFunc(RunnerCcuVersion::CCU_V1, testType, createFunc);

    auto executor
        = CcuExecutorFactory::MakeCcuExecutorInstance(RunnerCcuVersion::CCU_V1, testType, 0, 0, 0, instr_, nullptr);
    EXPECT_NE(executor, nullptr);
    EXPECT_EQ(executor->Describe(), "TestExecutor");
}

// Test: CcuExecutorFactory MakeCcuExecutorInstance with non-registered type
TEST_F(CcuExecutorManagerTest, FactoryMakeInstanceNotRegistered)
{
    uint16_t nonExistentType = 0xEEEE;

    auto executor = CcuExecutorFactory::MakeCcuExecutorInstance(
        RunnerCcuVersion::CCU_V1, nonExistentType, 0, 0, 0, instr_, nullptr);
    EXPECT_EQ(executor, nullptr);
}

// Test: CcuExecutorCreateFuncRegister constructor
TEST_F(CcuExecutorManagerTest, CreateFuncRegister)
{
    uint16_t type = SimCcuV1::LOAD_TYPE;
    uint16_t code = SimCcuV1::LOADSQEARGSTOGSA_CODE;

    CcuExecutorCreateFuncRegister reg(
        RunnerCcuVersion::CCU_V1, type, code,
        [](int streamId, int rankId, int dieId, const CcuInstr& instr, CcuSimulator* ccuSimulator) {
            auto executor = std::make_unique<TestExecutor>(streamId, rankId, dieId, instr, ccuSimulator);
            executor->SetVersion(RunnerCcuVersion::CCU_V1);
            return executor;
        });

    // Verify registration was successful
    CcuInstrHeader header = InstrHeader(type, code);
    auto retrievedFunc = CcuExecutorCreateFuncMgr::Instance().GetFunc(RunnerCcuVersion::CCU_V1, header.header);
    EXPECT_NE(retrievedFunc, nullptr);
}

// Test: Register multiple functions
TEST_F(CcuExecutorManagerTest, RegisterMultipleFunctions)
{
    auto createFunc1 = [](int streamId, int rankId, int dieId, const CcuInstr& instr, CcuSimulator* ccuSimulator) {
        return std::make_unique<TestExecutor>(streamId, rankId, dieId, instr, ccuSimulator);
    };

    CcuExecutorCreateFuncMgr::Instance().RegFunc(RunnerCcuVersion::CCU_V1, 0x1000, createFunc1);
    CcuExecutorCreateFuncMgr::Instance().RegFunc(RunnerCcuVersion::CCU_V1, 0x2000, createFunc1);
    CcuExecutorCreateFuncMgr::Instance().RegFunc(RunnerCcuVersion::CCU_V1, 0x3000, createFunc1);

    EXPECT_NE(CcuExecutorCreateFuncMgr::Instance().GetFunc(RunnerCcuVersion::CCU_V1, 0x1000), nullptr);
    EXPECT_NE(CcuExecutorCreateFuncMgr::Instance().GetFunc(RunnerCcuVersion::CCU_V1, 0x2000), nullptr);
    EXPECT_NE(CcuExecutorCreateFuncMgr::Instance().GetFunc(RunnerCcuVersion::CCU_V1, 0x3000), nullptr);
}

// Test: Overwrite existing registration
TEST_F(CcuExecutorManagerTest, OverwriteRegistration)
{
    uint16_t testType = 0x9999;

    auto createFunc1 = [](int streamId, int rankId, int dieId, const CcuInstr& instr, CcuSimulator* ccuSimulator) {
        return std::make_unique<TestExecutor>(streamId, rankId, dieId, instr, ccuSimulator);
    };

    CcuExecutorCreateFuncMgr::Instance().RegFunc(RunnerCcuVersion::CCU_V1, testType, createFunc1);
    EXPECT_NE(CcuExecutorCreateFuncMgr::Instance().GetFunc(RunnerCcuVersion::CCU_V1, testType), nullptr);

    // Overwrite with new function
    CcuExecutorCreateFuncMgr::Instance().RegFunc(RunnerCcuVersion::CCU_V1, testType, createFunc1);
    EXPECT_NE(CcuExecutorCreateFuncMgr::Instance().GetFunc(RunnerCcuVersion::CCU_V1, testType), nullptr);
}

// Test: Factory creates executor with correct parameters
TEST_F(CcuExecutorManagerTest, FactoryCorrectParameters)
{
    uint16_t testType = 0xAAAA;
    int testStreamId = 5;
    int testRankId = 3;
    int testDieId = 1;

    auto createFunc = [](int streamId, int rankId, int dieId, const CcuInstr& instr, CcuSimulator* ccuSimulator) {
        auto executor = std::make_unique<TestExecutor>(streamId, rankId, dieId, instr, ccuSimulator);
        return executor;
    };

    CcuExecutorCreateFuncMgr::Instance().RegFunc(RunnerCcuVersion::CCU_V1, testType, createFunc);

    auto executor = CcuExecutorFactory::MakeCcuExecutorInstance(
        RunnerCcuVersion::CCU_V1, testType, testStreamId, testRankId, testDieId, instr_, nullptr);
    EXPECT_NE(executor, nullptr);
    EXPECT_EQ(executor->streamId_, testStreamId);
    EXPECT_EQ(executor->rankId_, testRankId);
    EXPECT_EQ(executor->dieId_, testDieId);
}

// Test: InstrHeader function
TEST_F(CcuExecutorManagerTest, InstrHeaderFunction)
{
    uint16_t type = 0x5;
    uint16_t code = 0x123;

    CcuInstrHeader header = InstrHeader(type, code);
    EXPECT_EQ(header.type, type);
    EXPECT_EQ(header.code, code);
}

// Test: InstrHeader bit fields
TEST_F(CcuExecutorManagerTest, InstrHeaderBitFields)
{
    CcuInstrHeader header = {};
    header.type = 0xF;   // 4 bits max
    header.code = 0x7FF; // 11 bits max

    EXPECT_EQ(header.type, 0xF);
    EXPECT_EQ(header.code, 0x7FF);
}

// Test: CcuInstrHeader union
TEST_F(CcuExecutorManagerTest, CcuInstrHeaderUnion)
{
    CcuInstrHeader header = {};
    header.type = 0x1;
    header.code = 0x2;

    uint16_t headerValue = header.header;
    EXPECT_NE(headerValue, 0);

    CcuInstrHeader header2 = {};
    header2.header = headerValue;
    EXPECT_EQ(header2.type, 0x1);
    EXPECT_EQ(header2.code, 0x2);
}

// Test: CreateFunc with simulator
TEST_F(CcuExecutorManagerTest, CreateFuncWithSimulator)
{
    uint16_t testType = 0xBBBB;
    auto simulator = std::make_unique<CcuSimulator>(0, 0, 0, 10, 10, RunnerCcuVersion::CCU_V1);

    auto createFunc = [](int streamId, int rankId, int dieId, const CcuInstr& instr, CcuSimulator* ccuSimulator) {
        return std::make_unique<TestExecutor>(streamId, rankId, dieId, instr, ccuSimulator);
    };

    CcuExecutorCreateFuncMgr::Instance().RegFunc(RunnerCcuVersion::CCU_V1, testType, createFunc);

    auto executor = CcuExecutorFactory::MakeCcuExecutorInstance(
        RunnerCcuVersion::CCU_V1, testType, 0, 0, 0, instr_, simulator.get());
    EXPECT_NE(executor, nullptr);
    EXPECT_EQ(executor->ccuSimulator_, simulator.get());
}

// Test: Multiple factory calls
TEST_F(CcuExecutorManagerTest, MultipleFactoryCalls)
{
    uint16_t testType = 0xCCCC;

    auto createFunc = [](int streamId, int rankId, int dieId, const CcuInstr& instr, CcuSimulator* ccuSimulator) {
        return std::make_unique<TestExecutor>(streamId, rankId, dieId, instr, ccuSimulator);
    };

    CcuExecutorCreateFuncMgr::Instance().RegFunc(RunnerCcuVersion::CCU_V1, testType, createFunc);

    auto executor1
        = CcuExecutorFactory::MakeCcuExecutorInstance(RunnerCcuVersion::CCU_V1, testType, 0, 0, 0, instr_, nullptr);
    auto executor2
        = CcuExecutorFactory::MakeCcuExecutorInstance(RunnerCcuVersion::CCU_V1, testType, 1, 1, 1, instr_, nullptr);
    auto executor3
        = CcuExecutorFactory::MakeCcuExecutorInstance(RunnerCcuVersion::CCU_V1, testType, 2, 2, 2, instr_, nullptr);

    EXPECT_NE(executor1, nullptr);
    EXPECT_NE(executor2, nullptr);
    EXPECT_NE(executor3, nullptr);
}

// Test: Delete copy constructor
TEST_F(CcuExecutorManagerTest, DeleteCopyConstructor)
{
    EXPECT_FALSE(std::is_copy_constructible<CcuExecutorCreateFuncMgr>::value);
}

// Test: Delete copy assignment
TEST_F(CcuExecutorManagerTest, DeleteCopyAssignment)
{
    EXPECT_FALSE(std::is_copy_assignable<CcuExecutorCreateFuncMgr>::value);
}

// Test: Register with different type/code combinations
TEST_F(CcuExecutorManagerTest, DifferentTypeCodeCombinations)
{
    struct TypeCode {
        uint16_t type;
        uint16_t code;
    };
    TypeCode combinations[] = {
        {SimCcuV1::LOAD_TYPE, SimCcuV1::LOADSQEARGSTOGSA_CODE},
        {SimCcuV1::LOAD_TYPE, SimCcuV1::LOADSQEARGSTOXN_CODE},
        {SimCcuV1::CTRL_TYPE, SimCcuV1::LOOP_CODE},
        {SimCcuV1::CTRL_TYPE, SimCcuV1::LOOPGROUP_CODE},
        {SimCcuV1::TRANS_TYPE, SimCcuV1::TRANSLOCMEMTOLOCMS_CODE},
        {SimCcuV1::REDUCE_TYPE, SimCcuV1::ADD_CODE},
    };

    auto createFunc = [](int streamId, int rankId, int dieId, const CcuInstr& instr, CcuSimulator* ccuSimulator) {
        return std::make_unique<TestExecutor>(streamId, rankId, dieId, instr, ccuSimulator);
    };

    for (const auto& tc : combinations) {
        CcuInstrHeader header = InstrHeader(tc.type, tc.code);
        CcuExecutorCreateFuncMgr::Instance().RegFunc(RunnerCcuVersion::CCU_V1, header.header, createFunc);
        EXPECT_NE(CcuExecutorCreateFuncMgr::Instance().GetFunc(RunnerCcuVersion::CCU_V1, header.header), nullptr);
    }
}
