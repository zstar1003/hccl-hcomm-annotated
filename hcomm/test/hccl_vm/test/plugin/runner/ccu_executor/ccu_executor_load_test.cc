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
#include "load_gsa_xn_executor.h"
#include "load_imd_to_gsa_executor.h"
#include "load_imd_to_xn_executor.h"
#include "load_sqe_args_to_gsa_executor.h"
#include "load_sqe_args_to_xn_executor.h"
#include "load_xn_xn_executor.h"

#define private public
#undef private
using namespace hcomm::CcuRep;

// =============================================================================
// LoadGsaGsaExecutor tests
// =============================================================================
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

// =============================================================================
// LoadGsaXnExecutor tests
// =============================================================================
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

// =============================================================================
// LoadImdToGSAExecutor tests
// =============================================================================
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

// =============================================================================
// LoadImdToXnExecutor tests
// =============================================================================
class LoadImdToXnExecutorTest : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(LoadImdToXnExecutorTest, StructSize) { EXPECT_GT(sizeof(LoadImdToXnExecutor), 0); }

TEST_F(LoadImdToXnExecutorTest, DefaultConstructor)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    LoadImdToXnExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Describe());
}

TEST_F(LoadImdToXnExecutorTest, ParameterizedConstructor)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    LoadImdToXnExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Describe());
}

TEST_F(LoadImdToXnExecutorTest, ParserZeroValues)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    LoadImdToXnExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Parser());
    EXPECT_NO_THROW(executor.Describe());
}

TEST_F(LoadImdToXnExecutorTest, ParserMaxValues)
{
    CcuInstr instr;
    memset(&instr, 0xFF, sizeof(instr));
    LoadImdToXnExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Parser());
    EXPECT_NO_THROW(executor.Describe());
}

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

TEST_F(LoadImdToXnExecutorTest, InheritanceCheck)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    LoadImdToXnExecutor executor(0, 0, 0, instr, nullptr);
    CcuExecutorBase* base = &executor;
    EXPECT_NE(base, nullptr);
}

// =============================================================================
// LoadSqeArgsToGsaExecutor tests
// =============================================================================
class LoadSqeArgsToGsaExecutorTest : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(LoadSqeArgsToGsaExecutorTest, StructSize) { EXPECT_GT(sizeof(LoadSqeArgsToGsaExecutor), 0); }

TEST_F(LoadSqeArgsToGsaExecutorTest, DefaultConstructor)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    LoadSqeArgsToGsaExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Describe());
}

TEST_F(LoadSqeArgsToGsaExecutorTest, ParameterizedConstructor)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    LoadSqeArgsToGsaExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Describe());
}

TEST_F(LoadSqeArgsToGsaExecutorTest, ParserZeroValues)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    LoadSqeArgsToGsaExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Parser());
    EXPECT_NO_THROW(executor.Describe());
}

TEST_F(LoadSqeArgsToGsaExecutorTest, ParserMaxValues)
{
    CcuInstr instr;
    memset(&instr, 0xFF, sizeof(instr));
    LoadSqeArgsToGsaExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Parser());
    EXPECT_NO_THROW(executor.Describe());
}

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

TEST_F(LoadSqeArgsToGsaExecutorTest, InheritanceCheck)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    LoadSqeArgsToGsaExecutor executor(0, 0, 0, instr, nullptr);
    CcuExecutorBase* base = &executor;
    EXPECT_NE(base, nullptr);
}

// =============================================================================
// LoadSqeArgsToXnExecutor tests
// =============================================================================
class LoadSqeArgsToXnExecutorTest : public testing::Test {
protected:
    void SetUp() override
    {
        CcuResourceManager::GetInstance().Init(0, 1, RunnerCcuVersion::CCU_V1, std::vector<uint64_t>{});
    }
    void TearDown() override {}
};

TEST_F(LoadSqeArgsToXnExecutorTest, StructSize) { EXPECT_GT(sizeof(LoadSqeArgsToXnExecutor), 0); }

TEST_F(LoadSqeArgsToXnExecutorTest, DefaultConstructor)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    LoadSqeArgsToXnExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Describe());
}

TEST_F(LoadSqeArgsToXnExecutorTest, ParameterizedConstructor)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    LoadSqeArgsToXnExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Describe());
}

TEST_F(LoadSqeArgsToXnExecutorTest, ParserZeroValues)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    LoadSqeArgsToXnExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Parser());
    EXPECT_NO_THROW(executor.Describe());
}

TEST_F(LoadSqeArgsToXnExecutorTest, ParserMaxValues)
{
    CcuInstr instr;
    memset(&instr, 0xFF, sizeof(instr));
    LoadSqeArgsToXnExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Parser());
    EXPECT_NO_THROW(executor.Describe());
}

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

TEST_F(LoadSqeArgsToXnExecutorTest, InheritanceCheck)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    LoadSqeArgsToXnExecutor executor(0, 0, 0, instr, nullptr);
    CcuExecutorBase* base = &executor;
    EXPECT_NE(base, nullptr);
}

// =============================================================================
// LoadXnXnExecutor tests
// =============================================================================
class LoadXnXnExecutorTest : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(LoadXnXnExecutorTest, StructSize) { EXPECT_GT(sizeof(LoadXnXnExecutor), 0); }

TEST_F(LoadXnXnExecutorTest, DefaultConstructor)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    LoadXnXnExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Describe());
}

TEST_F(LoadXnXnExecutorTest, ParameterizedConstructor)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    LoadXnXnExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Describe());
}

TEST_F(LoadXnXnExecutorTest, ParserZeroValues)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    LoadXnXnExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Parser());
    EXPECT_NO_THROW(executor.Describe());
}

TEST_F(LoadXnXnExecutorTest, ParserMaxValues)
{
    CcuInstr instr;
    memset(&instr, 0xFF, sizeof(instr));
    LoadXnXnExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Parser());
    EXPECT_NO_THROW(executor.Describe());
}

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

TEST_F(LoadXnXnExecutorTest, InheritanceCheck)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    LoadXnXnExecutor executor(0, 0, 0, instr, nullptr);
    CcuExecutorBase* base = &executor;
    EXPECT_NE(base, nullptr);
}

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
