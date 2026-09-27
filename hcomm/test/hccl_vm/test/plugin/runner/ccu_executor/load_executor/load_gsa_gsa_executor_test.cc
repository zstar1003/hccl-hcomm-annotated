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
#include "load_gsa_gsa_executor.h"
#define private public
#undef private

using namespace hcomm::CcuRep;

class LoadGsaGsaExecutorTest : public testing::Test {
protected:
    void SetUp() override
    {
        auto& mgr = CcuResourceManager::GetInstance();
        mgr.Init(0, 4, RunnerCcuVersion::CCU_V1, {});
    }
    void TearDown() override {}
};

TEST_F(LoadGsaGsaExecutorTest, StructSize) { EXPECT_GT(sizeof(LoadGsaGsaExecutor), 0); }

TEST_F(LoadGsaGsaExecutorTest, DefaultConstructor)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    LoadGsaGsaExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Describe());
}

TEST_F(LoadGsaGsaExecutorTest, ParameterizedConstructor)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    LoadGsaGsaExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Describe());
}

TEST_F(LoadGsaGsaExecutorTest, ParserZeroValues)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    LoadGsaGsaExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Parser());
    EXPECT_NO_THROW(executor.Describe());
}

TEST_F(LoadGsaGsaExecutorTest, ParserMaxValues)
{
    CcuInstr instr;
    memset(&instr, 0xFF, sizeof(instr));
    LoadGsaGsaExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Parser());
    EXPECT_NO_THROW(executor.Describe());
}

TEST_F(LoadGsaGsaExecutorTest, ParserGsaParameters)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    instr.v1.loadGSAGSA.gsAdId = 100;
    instr.v1.loadGSAGSA.gsAmId = 50;
    instr.v1.loadGSAGSA.gsAnId = 25;
    LoadGsaGsaExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Parser());
    std::string desc = executor.Describe();
    EXPECT_FALSE(desc.empty());
}

TEST_F(LoadGsaGsaExecutorTest, DifferentGsaIdCombinations)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    uint16_t gsaIds[] = {0, 1, SimCcuV1::CCU_RESOURCE_GSA_NUM / 2, SimCcuV1::CCU_RESOURCE_GSA_NUM - 1};
    for (auto gsAdId : gsaIds) {
        for (auto gsAmId : gsaIds) {
            for (auto gsAnId : gsaIds) {
                instr.v1.loadGSAGSA.gsAdId = gsAdId;
                instr.v1.loadGSAGSA.gsAmId = gsAmId;
                instr.v1.loadGSAGSA.gsAnId = gsAnId;
                LoadGsaGsaExecutor executor(0, 0, 0, instr, nullptr);
                EXPECT_NO_THROW(executor.Parser());
                EXPECT_NO_THROW(executor.Describe());
            }
        }
    }
}

TEST_F(LoadGsaGsaExecutorTest, DescribeContent)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    instr.v1.loadGSAGSA.gsAdId = 10;
    instr.v1.loadGSAGSA.gsAmId = 20;
    instr.v1.loadGSAGSA.gsAnId = 30;
    LoadGsaGsaExecutor executor(0, 0, 0, instr, nullptr);
    executor.Parser();
    std::string desc = executor.Describe();
    EXPECT_FALSE(desc.empty());
    EXPECT_NE(desc.find("GSA"), std::string::npos);
}

TEST_F(LoadGsaGsaExecutorTest, InheritanceCheck)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    LoadGsaGsaExecutor executor(0, 0, 0, instr, nullptr);
    CcuExecutorBase* base = &executor;
    EXPECT_NE(base, nullptr);
}

TEST_F(LoadGsaGsaExecutorTest, SameSourceAndDestination)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    instr.v1.loadGSAGSA.gsAdId = 100;
    instr.v1.loadGSAGSA.gsAmId = 100;
    instr.v1.loadGSAGSA.gsAnId = 100;
    LoadGsaGsaExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Parser());
    EXPECT_NO_THROW(executor.Describe());
}

TEST_F(LoadGsaGsaExecutorTest, Run_BasicAddition)
{
    auto& mgr = CcuResourceManager::GetInstance();
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    instr.v1.loadGSAGSA.gsAdId = 5;
    instr.v1.loadGSAGSA.gsAmId = 10;
    instr.v1.loadGSAGSA.gsAnId = 20;
    mgr.UpdateGsaValue(0, 0, 10, 100);
    mgr.UpdateGsaValue(0, 0, 20, 200);
    LoadGsaGsaExecutor executor(0, 0, 0, instr, nullptr);
    executor.Parser();
    executor.Run();
    uint64_t result = mgr.GetGsaValue(0, 0, 5);
    EXPECT_EQ(result, 300);
}

TEST_F(LoadGsaGsaExecutorTest, Run_ZeroValues)
{
    auto& mgr = CcuResourceManager::GetInstance();
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    instr.v1.loadGSAGSA.gsAdId = 1;
    instr.v1.loadGSAGSA.gsAmId = 2;
    instr.v1.loadGSAGSA.gsAnId = 3;
    mgr.UpdateGsaValue(0, 0, 2, 0);
    mgr.UpdateGsaValue(0, 0, 3, 0);
    LoadGsaGsaExecutor executor(0, 0, 0, instr, nullptr);
    executor.Parser();
    executor.Run();
    uint64_t result = mgr.GetGsaValue(0, 0, 1);
    EXPECT_EQ(result, 0);
}

TEST_F(LoadGsaGsaExecutorTest, Run_LargeValues)
{
    auto& mgr = CcuResourceManager::GetInstance();
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    instr.v1.loadGSAGSA.gsAdId = 0;
    instr.v1.loadGSAGSA.gsAmId = 1;
    instr.v1.loadGSAGSA.gsAnId = 2;
    mgr.UpdateGsaValue(0, 0, 1, 0xFFFFFFFFFFFFFFFFULL);
    mgr.UpdateGsaValue(0, 0, 2, 1);
    LoadGsaGsaExecutor executor(0, 0, 0, instr, nullptr);
    executor.Parser();
    executor.Run();
    uint64_t result = mgr.GetGsaValue(0, 0, 0);
    EXPECT_EQ(result, 0);
}
