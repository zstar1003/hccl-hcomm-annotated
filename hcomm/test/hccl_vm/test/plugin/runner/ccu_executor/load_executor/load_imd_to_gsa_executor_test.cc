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
#include "ccu_simulator.h"
#include "load_imd_to_gsa_executor.h"

#define private public
#undef private

using namespace hcomm::CcuRep;

class LoadImdToGSAExecutorTest : public testing::Test {
protected:
    void SetUp() override
    {
        auto& mgr = CcuResourceManager::GetInstance();
        mgr.Init(0, 4, RunnerCcuVersion::CCU_V1, {});
    }
    void TearDown() override {}
};

TEST_F(LoadImdToGSAExecutorTest, StructSize) { EXPECT_GT(sizeof(LoadImdToGSAExecutor), 0); }

TEST_F(LoadImdToGSAExecutorTest, DefaultConstructor)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    LoadImdToGSAExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Describe());
}

TEST_F(LoadImdToGSAExecutorTest, ParameterizedConstructor)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    LoadImdToGSAExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Describe());
}

TEST_F(LoadImdToGSAExecutorTest, ParserZeroValues)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    LoadImdToGSAExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Parser());
    EXPECT_NO_THROW(executor.Describe());
}

TEST_F(LoadImdToGSAExecutorTest, ParserMaxValues)
{
    CcuInstr instr;
    memset(&instr, 0xFF, sizeof(instr));
    LoadImdToGSAExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Parser());
    EXPECT_NO_THROW(executor.Describe());
}

TEST_F(LoadImdToGSAExecutorTest, ParserSpecificParameters)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    instr.v1.loadImdToGSA.gsaId = 100;
    instr.v1.loadImdToGSA.immediate = 0x12345678;
    LoadImdToGSAExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Parser());
    std::string desc = executor.Describe();
    EXPECT_FALSE(desc.empty());
}

TEST_F(LoadImdToGSAExecutorTest, DifferentGsaIdCombinations)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    uint16_t gsaIds[] = {0, 1, SimCcuV1::CCU_RESOURCE_GSA_NUM / 2, SimCcuV1::CCU_RESOURCE_GSA_NUM - 1};
    for (auto gsaId : gsaIds) {
        instr.v1.loadImdToGSA.gsaId = gsaId;
        instr.v1.loadImdToGSA.immediate = gsaId * 10;
        LoadImdToGSAExecutor executor(0, 0, 0, instr, nullptr);
        EXPECT_NO_THROW(executor.Parser());
        EXPECT_NO_THROW(executor.Describe());
    }
}

TEST_F(LoadImdToGSAExecutorTest, DescribeContent)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    instr.v1.loadImdToGSA.gsaId = 10;
    instr.v1.loadImdToGSA.immediate = 0xABCD;
    LoadImdToGSAExecutor executor(0, 0, 0, instr, nullptr);
    executor.Parser();
    std::string desc = executor.Describe();
    EXPECT_FALSE(desc.empty());
    EXPECT_NE(desc.find("GSA"), std::string::npos);
}

TEST_F(LoadImdToGSAExecutorTest, InheritanceCheck)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    LoadImdToGSAExecutor executor(0, 0, 0, instr, nullptr);
    CcuExecutorBase* base = &executor;
    EXPECT_NE(base, nullptr);
}

TEST_F(LoadImdToGSAExecutorTest, Run_BasicImmediate)
{
    auto& mgr = CcuResourceManager::GetInstance();
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    instr.v1.loadImdToGSA.gsaId = 5;
    instr.v1.loadImdToGSA.immediate = 12345;
    LoadImdToGSAExecutor executor(0, 0, 0, instr, nullptr);
    executor.Parser();
    executor.Run();
    uint64_t result = mgr.GetGsaValue(0, 0, 5);
    EXPECT_EQ(result, 12345);
}

TEST_F(LoadImdToGSAExecutorTest, Run_ZeroImmediate)
{
    auto& mgr = CcuResourceManager::GetInstance();
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    instr.v1.loadImdToGSA.gsaId = 10;
    instr.v1.loadImdToGSA.immediate = 0;
    LoadImdToGSAExecutor executor(0, 0, 0, instr, nullptr);
    executor.Parser();
    executor.Run();
    uint64_t result = mgr.GetGsaValue(0, 0, 10);
    EXPECT_EQ(result, 0);
}

TEST_F(LoadImdToGSAExecutorTest, Run_LargeImmediate)
{
    auto& mgr = CcuResourceManager::GetInstance();
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    instr.v1.loadImdToGSA.gsaId = 0;
    instr.v1.loadImdToGSA.immediate = 0xFFFFFFFFFFFFFFFFULL;
    LoadImdToGSAExecutor executor(0, 0, 0, instr, nullptr);
    executor.Parser();
    executor.Run();
    uint64_t result = mgr.GetGsaValue(0, 0, 0);
    EXPECT_EQ(result, 0xFFFFFFFFFFFFFFFFULL);
}
