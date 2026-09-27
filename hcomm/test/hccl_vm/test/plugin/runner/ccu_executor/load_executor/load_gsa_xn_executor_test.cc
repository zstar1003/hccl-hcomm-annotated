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
#include "load_gsa_xn_executor.h"

#define private public
#undef private

using namespace hcomm::CcuRep;

class LoadGsaXnExecutorTest : public testing::Test {
protected:
    void SetUp() override
    {
        auto& mgr = CcuResourceManager::GetInstance();
        mgr.Init(0, 4, RunnerCcuVersion::CCU_V1, {});
    }
    void TearDown() override {}
};

TEST_F(LoadGsaXnExecutorTest, StructSize) { EXPECT_GT(sizeof(LoadGsaXnExecutor), 0); }

TEST_F(LoadGsaXnExecutorTest, DefaultConstructor)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    LoadGsaXnExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Describe());
}

TEST_F(LoadGsaXnExecutorTest, ParameterizedConstructor)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    LoadGsaXnExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Describe());
}

TEST_F(LoadGsaXnExecutorTest, ParserZeroValues)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    LoadGsaXnExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Parser());
    EXPECT_NO_THROW(executor.Describe());
}

TEST_F(LoadGsaXnExecutorTest, ParserMaxValues)
{
    CcuInstr instr;
    memset(&instr, 0xFF, sizeof(instr));
    LoadGsaXnExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Parser());
    EXPECT_NO_THROW(executor.Describe());
}

TEST_F(LoadGsaXnExecutorTest, ParserSpecificParameters)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    instr.v1.loadGSAXn.gsAdId = 100;
    instr.v1.loadGSAXn.gsAmId = 50;
    instr.v1.loadGSAXn.xnId = 25;
    LoadGsaXnExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Parser());
    std::string desc = executor.Describe();
    EXPECT_FALSE(desc.empty());
}

TEST_F(LoadGsaXnExecutorTest, DifferentIdCombinations)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    uint16_t ids[] = {0, 1, 100, 0x7FFF, 0xFFFF};
    for (auto gsAdId : ids) {
        for (auto gsAmId : ids) {
            for (auto xnId : ids) {
                instr.v1.loadGSAXn.gsAdId = gsAdId;
                instr.v1.loadGSAXn.gsAmId = gsAmId;
                instr.v1.loadGSAXn.xnId = xnId;
                LoadGsaXnExecutor executor(0, 0, 0, instr, nullptr);
                EXPECT_NO_THROW(executor.Parser());
                EXPECT_NO_THROW(executor.Describe());
            }
        }
    }
}

TEST_F(LoadGsaXnExecutorTest, DescribeContent)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    instr.v1.loadGSAXn.gsAdId = 10;
    instr.v1.loadGSAXn.gsAmId = 20;
    instr.v1.loadGSAXn.xnId = 30;
    LoadGsaXnExecutor executor(0, 0, 0, instr, nullptr);
    executor.Parser();
    std::string desc = executor.Describe();
    EXPECT_FALSE(desc.empty());
    EXPECT_NE(desc.find("GSA"), std::string::npos);
    EXPECT_NE(desc.find("Xn"), std::string::npos);
}

TEST_F(LoadGsaXnExecutorTest, InheritanceCheck)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    LoadGsaXnExecutor executor(0, 0, 0, instr, nullptr);
    CcuExecutorBase* base = &executor;
    EXPECT_NE(base, nullptr);
}

TEST_F(LoadGsaXnExecutorTest, Run_BasicAddition)
{
    auto& mgr = CcuResourceManager::GetInstance();
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    instr.v1.loadGSAXn.gsAdId = 5;
    instr.v1.loadGSAXn.gsAmId = 10;
    instr.v1.loadGSAXn.xnId = 20;
    mgr.UpdateGsaValue(0, 0, 10, 100);
    mgr.UpdateXnValue(0, 0, 20, 200);
    LoadGsaXnExecutor executor(0, 0, 0, instr, nullptr);
    executor.Parser();
    executor.Run();
    uint64_t result = mgr.GetGsaValue(0, 0, 5);
    EXPECT_EQ(result, 300);
}

TEST_F(LoadGsaXnExecutorTest, Run_ZeroValues)
{
    auto& mgr = CcuResourceManager::GetInstance();
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    instr.v1.loadGSAXn.gsAdId = 1;
    instr.v1.loadGSAXn.gsAmId = 2;
    instr.v1.loadGSAXn.xnId = 3;
    mgr.UpdateGsaValue(0, 0, 2, 0);
    mgr.UpdateXnValue(0, 0, 3, 0);
    LoadGsaXnExecutor executor(0, 0, 0, instr, nullptr);
    executor.Parser();
    executor.Run();
    uint64_t result = mgr.GetGsaValue(0, 0, 1);
    EXPECT_EQ(result, 0);
}

TEST_F(LoadGsaXnExecutorTest, Run_LargeValues)
{
    auto& mgr = CcuResourceManager::GetInstance();
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    instr.v1.loadGSAXn.gsAdId = 0;
    instr.v1.loadGSAXn.gsAmId = 1;
    instr.v1.loadGSAXn.xnId = 2;
    mgr.UpdateGsaValue(0, 0, 1, 0xFFFFFFFFFFFFFFFFULL);
    mgr.UpdateXnValue(0, 0, 2, 1);
    LoadGsaXnExecutor executor(0, 0, 0, instr, nullptr);
    executor.Parser();
    executor.Run();
    uint64_t result = mgr.GetGsaValue(0, 0, 0);
    EXPECT_EQ(result, 0);
}
