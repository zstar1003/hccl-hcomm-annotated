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
#include "ccu_resource_common.h"
#include "ccu_resource_manager.h"
#include "ccu_simulator.h"
#include "sync_cke_executor.h"
#include "sync_gsa_executor.h"
#include "sync_xn_executor.h"
#include "trans_loc_mem_to_loc_mem_executor.h"
#include "trans_loc_mem_to_loc_ms_executor.h"
#include "trans_loc_mem_to_rmt_mem_executor.h"
#include "trans_loc_ms_to_loc_mem_executor.h"
#include "trans_loc_ms_to_loc_ms_executor.h"
#include "trans_loc_ms_to_rmt_mem_executor.h"
#include "trans_loc_ms_to_rmt_ms_executor.h"
#include "trans_rmt_mem_to_loc_mem_executor.h"
#include "trans_rmt_mem_to_loc_ms_executor.h"
#include "trans_rmt_ms_to_loc_mem_executor.h"
#include "trans_rmt_ms_to_loc_ms_executor.h"

using namespace hcomm::CcuRep;

// ==================== SyncCkeExecutor Tests ====================

class SyncCkeExecutorTest : public testing::Test {
protected:
    void SetUp() override
    {
        auto& mgr = CcuResourceManager::GetInstance();
        mgr.Init(0, 2, RunnerCcuVersion::CCU_V1, {});
        mgr.Init(1, 2, RunnerCcuVersion::CCU_V1, {});
        RankChannelInfo chInfo;
        CcuInfo rmtInfo{1, 0};
        chInfo[0][0] = rmtInfo;
        mgr.InitChannelInfo(0, chInfo);
    }
};

TEST_F(SyncCkeExecutorTest, StructSize) { EXPECT_GT(sizeof(SyncCkeExecutor), 0); }

TEST_F(SyncCkeExecutorTest, DefaultConstructor)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    SyncCkeExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Describe());
}

TEST_F(SyncCkeExecutorTest, ParserZeroValues)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    SyncCkeExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Parser());
    EXPECT_NO_THROW(executor.Describe());
}

TEST_F(SyncCkeExecutorTest, ParserMaxValues)
{
    CcuInstr instr;
    memset(&instr, 0xFF, sizeof(instr));
    SyncCkeExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Parser());
    EXPECT_NO_THROW(executor.Describe());
}

TEST_F(SyncCkeExecutorTest, ParserSpecificParameters)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    instr.v1.syncCKE.rmtCKEId = 10;
    instr.v1.syncCKE.locCKEId = 20;
    instr.v1.syncCKE.locCKEMask = 0xFF;
    instr.v1.syncCKE.channelId = 0;
    instr.v1.syncCKE.setCKEId = 5;
    instr.v1.syncCKE.setCKEMask = 0x0F;
    instr.v1.syncCKE.waitCKEId = 3;
    instr.v1.syncCKE.waitCKEMask = 0x01;
    SyncCkeExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Parser());
    std::string desc = executor.Describe();
    EXPECT_FALSE(desc.empty());
}

TEST_F(SyncCkeExecutorTest, DescribeContent)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    instr.v1.syncCKE.rmtCKEId = 10;
    instr.v1.syncCKE.locCKEId = 20;
    instr.v1.syncCKE.channelId = 5;
    SyncCkeExecutor executor(0, 0, 0, instr, nullptr);
    executor.Parser();
    std::string desc = executor.Describe();
    EXPECT_FALSE(desc.empty());
    EXPECT_NE(desc.find("Sync LocCKE"), std::string::npos);
}

TEST_F(SyncCkeExecutorTest, InheritanceCheck)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    SyncCkeExecutor executor(0, 0, 0, instr, nullptr);
    CcuExecutorBase* base = &executor;
    EXPECT_NE(base, nullptr);
}

TEST_F(SyncCkeExecutorTest, ProcessWithValidChannel)
{
    auto& mgr = CcuResourceManager::GetInstance();
    mgr.UpdateCkeValue(0, 0, 0, 0x00FF);
    mgr.UpdateCkeValue(1, 0, 0, 0x0000);
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    instr.v1.syncCKE.channelId = 0;
    instr.v1.syncCKE.locCKEId = 0;
    instr.v1.syncCKE.rmtCKEId = 0;
    instr.v1.syncCKE.locCKEMask = 0xFFFF;
    instr.v1.syncCKE.setCKEId = 0;
    instr.v1.syncCKE.setCKEMask = 0;
    CcuSimulator sim(0, 0, 0, 1, 1, RunnerCcuVersion::CCU_V1);
    SyncCkeExecutor executor(0, 0, 0, instr, &sim);
    executor.Parser();
    EXPECT_NO_THROW(executor.Process(mgr));
}

TEST_F(SyncCkeExecutorTest, RunWithCkeNotSatisfied)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    instr.v1.syncCKE.channelId = 0;
    instr.v1.syncCKE.waitCKEId = 0;
    instr.v1.syncCKE.waitCKEMask = 0x0001;
    instr.v1.syncCKE.clearType = 0;
    CcuSimulator sim(0, 0, 0, 1, 1, RunnerCcuVersion::CCU_V1);
    SyncCkeExecutor executor(0, 0, 0, instr, &sim);
    executor.Parser();
    EXPECT_NO_THROW(executor.Run());
}

// ==================== SyncGsaExecutor Tests ====================

class SyncGsaExecutorTest : public testing::Test {
protected:
    void SetUp() override
    {
        auto& mgr = CcuResourceManager::GetInstance();
        mgr.Init(0, 2, RunnerCcuVersion::CCU_V1, {});
        mgr.Init(1, 2, RunnerCcuVersion::CCU_V1, {});
        RankChannelInfo chInfo;
        CcuInfo rmtInfo{1, 0};
        chInfo[0][0] = rmtInfo;
        mgr.InitChannelInfo(0, chInfo);
    }
};

TEST_F(SyncGsaExecutorTest, StructSize) { EXPECT_GT(sizeof(SyncGsaExecutor), 0); }

TEST_F(SyncGsaExecutorTest, DefaultConstructor)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    SyncGsaExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Describe());
}

TEST_F(SyncGsaExecutorTest, ParserZeroValues)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    SyncGsaExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Parser());
    EXPECT_NO_THROW(executor.Describe());
}

TEST_F(SyncGsaExecutorTest, ParserMaxValues)
{
    CcuInstr instr;
    memset(&instr, 0xFF, sizeof(instr));
    SyncGsaExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Parser());
    EXPECT_NO_THROW(executor.Describe());
}

TEST_F(SyncGsaExecutorTest, ParserSpecificParameters)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    instr.v1.syncGSA.rmtGSAId = 10;
    instr.v1.syncGSA.locGSAId = 20;
    instr.v1.syncGSA.channelId = 0;
    instr.v1.syncGSA.setRmtCKEId = 5;
    instr.v1.syncGSA.setRmtCKEMask = 0x0F;
    instr.v1.syncGSA.setCKEId = 3;
    instr.v1.syncGSA.setCKEMask = 0x01;
    instr.v1.syncGSA.waitCKEId = 7;
    instr.v1.syncGSA.waitCKEMask = 0x02;
    SyncGsaExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Parser());
    std::string desc = executor.Describe();
    EXPECT_FALSE(desc.empty());
}

TEST_F(SyncGsaExecutorTest, DescribeContent)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    instr.v1.syncGSA.rmtGSAId = 10;
    instr.v1.syncGSA.locGSAId = 20;
    instr.v1.syncGSA.channelId = 5;
    SyncGsaExecutor executor(0, 0, 0, instr, nullptr);
    executor.Parser();
    std::string desc = executor.Describe();
    EXPECT_FALSE(desc.empty());
    EXPECT_NE(desc.find("Sync locGSAId"), std::string::npos);
}

TEST_F(SyncGsaExecutorTest, InheritanceCheck)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    SyncGsaExecutor executor(0, 0, 0, instr, nullptr);
    CcuExecutorBase* base = &executor;
    EXPECT_NE(base, nullptr);
}

TEST_F(SyncGsaExecutorTest, ProcessWithValidChannel)
{
    auto& mgr = CcuResourceManager::GetInstance();
    mgr.UpdateGsaValue(0, 0, 0, 0xDEADBEEF);
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    instr.v1.syncGSA.channelId = 0;
    instr.v1.syncGSA.locGSAId = 0;
    instr.v1.syncGSA.rmtGSAId = 0;
    instr.v1.syncGSA.setRmtCKEId = 0;
    instr.v1.syncGSA.setRmtCKEMask = 0;
    instr.v1.syncGSA.setCKEId = 0;
    instr.v1.syncGSA.setCKEMask = 0;
    CcuSimulator sim(0, 0, 0, 1, 1, RunnerCcuVersion::CCU_V1);
    SyncGsaExecutor executor(0, 0, 0, instr, &sim);
    executor.Parser();
    EXPECT_NO_THROW(executor.Process(mgr));
}

TEST_F(SyncGsaExecutorTest, RunWithCkeNotSatisfied)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    instr.v1.syncGSA.channelId = 0;
    instr.v1.syncGSA.waitCKEId = 0;
    instr.v1.syncGSA.waitCKEMask = 0x0001;
    instr.v1.syncGSA.clearType = 0;
    CcuSimulator sim(0, 0, 0, 1, 1, RunnerCcuVersion::CCU_V1);
    SyncGsaExecutor executor(0, 0, 0, instr, &sim);
    executor.Parser();
    EXPECT_NO_THROW(executor.Run());
}

// ==================== SyncXnExecutor Tests ====================

class SyncXnExecutorTest : public testing::Test {
protected:
    void SetUp() override
    {
        auto& mgr = CcuResourceManager::GetInstance();
        mgr.Init(0, 2, RunnerCcuVersion::CCU_V1, {});
        mgr.Init(1, 2, RunnerCcuVersion::CCU_V1, {});
        RankChannelInfo chInfo;
        CcuInfo rmtInfo{1, 0};
        chInfo[0][0] = rmtInfo;
        mgr.InitChannelInfo(0, chInfo);
    }
    void TearDown() override {}
};

// Test: SyncXnExecutor struct size check
TEST_F(SyncXnExecutorTest, StructSize) { EXPECT_GT(sizeof(SyncXnExecutor), 0); }

// Test: SyncXnExecutor default constructor
TEST_F(SyncXnExecutorTest, DefaultConstructor)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    SyncXnExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Describe());
}

// Test: SyncXnExecutor parameterized constructor
TEST_F(SyncXnExecutorTest, ParameterizedConstructor)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    SyncXnExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Describe());
}

// Test: SyncXnExecutor Parser with zero values
TEST_F(SyncXnExecutorTest, ParserZeroValues)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    SyncXnExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Parser());
    EXPECT_NO_THROW(executor.Describe());
}

// Test: SyncXnExecutor Parser with max values
TEST_F(SyncXnExecutorTest, ParserMaxValues)
{
    CcuInstr instr;
    memset(&instr, 0xFF, sizeof(instr));

    SyncXnExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Parser());
    EXPECT_NO_THROW(executor.Describe());
}

// Test: SyncXnExecutor Parser with specific Xn parameters
TEST_F(SyncXnExecutorTest, ParserXnParameters)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    instr.v1.syncXn.rmtXnId = 100;
    instr.v1.syncXn.locXnId = 200;
    instr.v1.syncXn.channelId = 5;
    instr.v1.syncXn.setRmtCKEId = 10;
    instr.v1.syncXn.setRmtCKEMask = 0xFF;

    SyncXnExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Parser());
    std::string desc = executor.Describe();
    EXPECT_FALSE(desc.empty());
}

// Test: SyncXnExecutor with different Xn IDs
TEST_F(SyncXnExecutorTest, DifferentXnIds)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    uint16_t xnIds[] = {0, 1, SimCcuV1::CCU_RESOURCE_XN_NUM / 2, SimCcuV1::CCU_RESOURCE_XN_NUM - 1, 0xFFFF};

    for (auto xnId : xnIds) {
        instr.v1.syncXn.locXnId = xnId;
        instr.v1.syncXn.rmtXnId = xnId;
        SyncXnExecutor executor(0, 0, 0, instr, nullptr);
        EXPECT_NO_THROW(executor.Parser());
        EXPECT_NO_THROW(executor.Describe());
    }
}

// Test: SyncXnExecutor Describe contains expected keywords
TEST_F(SyncXnExecutorTest, DescribeContent)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    instr.v1.syncXn.rmtXnId = 10;
    instr.v1.syncXn.locXnId = 20;
    instr.v1.syncXn.channelId = 5;

    SyncXnExecutor executor(0, 0, 0, instr, nullptr);
    executor.Parser();
    std::string desc = executor.Describe();
    EXPECT_FALSE(desc.empty());
    EXPECT_NE(desc.find("Sync locXnId"), std::string::npos);
}

// Test: SyncXnExecutor inheritance check
TEST_F(SyncXnExecutorTest, InheritanceCheck)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    SyncXnExecutor executor(0, 0, 0, instr, nullptr);
    CcuExecutorBase* base = &executor;
    EXPECT_NE(base, nullptr);
}

TEST_F(SyncXnExecutorTest, ProcessWithValidChannel)
{
    auto& mgr = CcuResourceManager::GetInstance();
    mgr.UpdateXnValue(0, 0, 0, 0x2000);
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    instr.v1.syncXn.channelId = 0;
    instr.v1.syncXn.locXnId = 0;
    instr.v1.syncXn.rmtXnId = 0;
    instr.v1.syncXn.setCKEId = 0;
    instr.v1.syncXn.setCKEMask = 0;
    instr.v1.syncXn.setRmtCKEId = 0;
    instr.v1.syncXn.setRmtCKEMask = 0;
    CcuSimulator sim(0, 0, 0, 1, 1, RunnerCcuVersion::CCU_V1);
    SyncXnExecutor executor(0, 0, 0, instr, &sim);
    executor.Parser();
    EXPECT_NO_THROW(executor.Process(mgr));
}

TEST_F(SyncXnExecutorTest, RunWithCkeNotSatisfied)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    instr.v1.syncXn.channelId = 0;
    instr.v1.syncXn.waitCKEId = 0;
    instr.v1.syncXn.waitCKEMask = 0x0001;
    instr.v1.syncXn.clearType = 0;
    CcuSimulator sim(0, 0, 0, 1, 1, RunnerCcuVersion::CCU_V1);
    SyncXnExecutor executor(0, 0, 0, instr, &sim);
    executor.Parser();
    EXPECT_NO_THROW(executor.Run());
}

// ==================== TransLocMemToLocMemExecutor Tests ====================

class TransLocMemToLocMemExecutorTest : public testing::Test {
protected:
    void SetUp() override
    {
        auto& mgr = CcuResourceManager::GetInstance();
        mgr.Init(0, 2, RunnerCcuVersion::CCU_V1, {});
    }
    void TearDown() override {}
};

// Test: TransLocMemToLocMemExecutor struct size check
TEST_F(TransLocMemToLocMemExecutorTest, StructSize) { EXPECT_GT(sizeof(TransLocMemToLocMemExecutor), 0); }

// Test: TransLocMemToLocMemExecutor default constructor
TEST_F(TransLocMemToLocMemExecutorTest, DefaultConstructor)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    TransLocMemToLocMemExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Describe());
}

// Test: TransLocMemToLocMemExecutor parameterized constructor
TEST_F(TransLocMemToLocMemExecutorTest, ParameterizedConstructor)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    TransLocMemToLocMemExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Describe());
}

// Test: TransLocMemToLocMemExecutor Parser with zero values
TEST_F(TransLocMemToLocMemExecutorTest, ParserZeroValues)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    TransLocMemToLocMemExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Parser());
    EXPECT_NO_THROW(executor.Describe());
}

// Test: TransLocMemToLocMemExecutor Parser with max values
TEST_F(TransLocMemToLocMemExecutorTest, ParserMaxValues)
{
    CcuInstr instr;
    memset(&instr, 0xFF, sizeof(instr));

    TransLocMemToLocMemExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Parser());
    EXPECT_NO_THROW(executor.Describe());
}

// Test: TransLocMemToLocMemExecutor Parser with specific parameters
TEST_F(TransLocMemToLocMemExecutorTest, ParserSpecificParameters)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    instr.v1.transLocMemToLocMem.dstGSAId = 100;
    instr.v1.transLocMemToLocMem.dstXnId = 50;
    instr.v1.transLocMemToLocMem.srcGSAId = 10;
    instr.v1.transLocMemToLocMem.srcXnId = 5;
    instr.v1.transLocMemToLocMem.lengthXnId = 3;
    instr.v1.transLocMemToLocMem.lengthEn = 1;

    TransLocMemToLocMemExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Parser());
    std::string desc = executor.Describe();
    EXPECT_FALSE(desc.empty());
}

// Test: TransLocMemToLocMemExecutor with different source and destination GSA IDs
TEST_F(TransLocMemToLocMemExecutorTest, DifferentGsaIds)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    uint16_t gsaIds[] = {0, 1, 100, SimCcuV1::CCU_RESOURCE_GSA_NUM - 1};

    for (auto srcGsaId : gsaIds) {
        for (auto dstGsaId : gsaIds) {
            instr.v1.transLocMemToLocMem.srcGSAId = srcGsaId;
            instr.v1.transLocMemToLocMem.dstGSAId = dstGsaId;
            TransLocMemToLocMemExecutor executor(0, 0, 0, instr, nullptr);
            EXPECT_NO_THROW(executor.Parser());
            EXPECT_NO_THROW(executor.Describe());
        }
    }
}

// Test: TransLocMemToLocMemExecutor Describe contains expected keywords
TEST_F(TransLocMemToLocMemExecutorTest, DescribeContent)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    instr.v1.transLocMemToLocMem.srcGSAId = 10;
    instr.v1.transLocMemToLocMem.dstGSAId = 100;

    TransLocMemToLocMemExecutor executor(0, 0, 0, instr, nullptr);
    executor.Parser();
    std::string desc = executor.Describe();
    EXPECT_FALSE(desc.empty());
    EXPECT_NE(desc.find("Trans LocMem"), std::string::npos);
}

// Test: TransLocMemToLocMemExecutor inheritance check
TEST_F(TransLocMemToLocMemExecutorTest, InheritanceCheck)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    TransLocMemToLocMemExecutor executor(0, 0, 0, instr, nullptr);
    CcuExecutorBase* base = &executor;
    EXPECT_NE(base, nullptr);
}

// ==================== TransLocMemToLocMSExecutor Tests ====================

class TransLocMemToLocMSExecutorTest : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

// Test: TransLocMemToLocMSExecutor struct size check
TEST_F(TransLocMemToLocMSExecutorTest, StructSize) { EXPECT_GT(sizeof(TransLocMemToLocMSExecutor), 0); }

// Test: TransLocMemToLocMSExecutor default constructor
TEST_F(TransLocMemToLocMSExecutorTest, DefaultConstructor)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    TransLocMemToLocMSExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Describe());
}

// Test: TransLocMemToLocMSExecutor parameterized constructor
TEST_F(TransLocMemToLocMSExecutorTest, ParameterizedConstructor)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    TransLocMemToLocMSExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Describe());
}

// Test: TransLocMemToLocMSExecutor Parser with zero values
TEST_F(TransLocMemToLocMSExecutorTest, ParserZeroValues)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    TransLocMemToLocMSExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Parser());
    EXPECT_NO_THROW(executor.Describe());
}

// Test: TransLocMemToLocMSExecutor Parser with max values
TEST_F(TransLocMemToLocMSExecutorTest, ParserMaxValues)
{
    CcuInstr instr;
    memset(&instr, 0xFF, sizeof(instr));

    TransLocMemToLocMSExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Parser());
    EXPECT_NO_THROW(executor.Describe());
}

// Test: TransLocMemToLocMSExecutor Parser with specific parameters
TEST_F(TransLocMemToLocMSExecutorTest, ParserSpecificParameters)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    instr.v1.transLocMemToLocMS.locGSAId = 100;
    instr.v1.transLocMemToLocMS.locXnId = 50;
    instr.v1.transLocMemToLocMS.locMSId = 0x8000 | 10;
    instr.v1.transLocMemToLocMS.lengthXnId = 5;
    instr.v1.transLocMemToLocMS.lengthEn = 0;

    TransLocMemToLocMSExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Parser());
    std::string desc = executor.Describe();
    EXPECT_FALSE(desc.empty());
}

// Test: TransLocMemToLocMSExecutor with different GSA IDs
TEST_F(TransLocMemToLocMSExecutorTest, DifferentGsaIds)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    uint16_t gsaIds[] = {0, 1, SimCcuV1::CCU_RESOURCE_GSA_NUM / 2, SimCcuV1::CCU_RESOURCE_GSA_NUM - 1};

    for (auto gsaId : gsaIds) {
        instr.v1.transLocMemToLocMS.locGSAId = gsaId;
        TransLocMemToLocMSExecutor executor(0, 0, 0, instr, nullptr);
        EXPECT_NO_THROW(executor.Parser());
        EXPECT_NO_THROW(executor.Describe());
    }
}

// Test: TransLocMemToLocMSExecutor Describe contains expected keywords
TEST_F(TransLocMemToLocMSExecutorTest, DescribeContent)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    instr.v1.transLocMemToLocMS.locGSAId = 10;
    instr.v1.transLocMemToLocMS.locMSId = 100;

    TransLocMemToLocMSExecutor executor(0, 0, 0, instr, nullptr);
    executor.Parser();
    std::string desc = executor.Describe();
    EXPECT_FALSE(desc.empty());
    EXPECT_NE(desc.find("Trans LocMem"), std::string::npos);
}

// Test: TransLocMemToLocMSExecutor inheritance check
TEST_F(TransLocMemToLocMSExecutorTest, InheritanceCheck)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    TransLocMemToLocMSExecutor executor(0, 0, 0, instr, nullptr);
    CcuExecutorBase* base = &executor;
    EXPECT_NE(base, nullptr);
}

// ==================== TransLocMemToRmtMemExecutor Tests ====================

class TransLocMemToRmtMemExecutorTest : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

// Test: TransLocMemToRmtMemExecutor struct size check
TEST_F(TransLocMemToRmtMemExecutorTest, StructSize) { EXPECT_GT(sizeof(TransLocMemToRmtMemExecutor), 0); }

// Test: TransLocMemToRmtMemExecutor default constructor
TEST_F(TransLocMemToRmtMemExecutorTest, DefaultConstructor)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    TransLocMemToRmtMemExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Describe());
}

// Test: TransLocMemToRmtMemExecutor parameterized constructor
TEST_F(TransLocMemToRmtMemExecutorTest, ParameterizedConstructor)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    TransLocMemToRmtMemExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Describe());
}

// Test: TransLocMemToRmtMemExecutor Parser with zero values
TEST_F(TransLocMemToRmtMemExecutorTest, ParserZeroValues)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    TransLocMemToRmtMemExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Parser());
    EXPECT_NO_THROW(executor.Describe());
}

// Test: TransLocMemToRmtMemExecutor Parser with max values
TEST_F(TransLocMemToRmtMemExecutorTest, ParserMaxValues)
{
    CcuInstr instr;
    memset(&instr, 0xFF, sizeof(instr));

    TransLocMemToRmtMemExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Parser());
    EXPECT_NO_THROW(executor.Describe());
}

// Test: TransLocMemToRmtMemExecutor Parser with specific parameters
TEST_F(TransLocMemToRmtMemExecutorTest, ParserSpecificParameters)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    instr.v1.transLocMemToRmtMem.rmtGSAId = 100;
    instr.v1.transLocMemToRmtMem.rmtXnId = 50;
    instr.v1.transLocMemToRmtMem.locGSAId = 10;
    instr.v1.transLocMemToRmtMem.locXnId = 5;
    instr.v1.transLocMemToRmtMem.lengthXnId = 3;
    instr.v1.transLocMemToRmtMem.channelId = 1;

    TransLocMemToRmtMemExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Parser());
    std::string desc = executor.Describe();
    EXPECT_FALSE(desc.empty());
}

// Test: TransLocMemToRmtMemExecutor inheritance check
TEST_F(TransLocMemToRmtMemExecutorTest, InheritanceCheck)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    TransLocMemToRmtMemExecutor executor(0, 0, 0, instr, nullptr);
    CcuExecutorBase* base = &executor;
    EXPECT_NE(base, nullptr);
}

// ==================== TransLocMSToLocMemExecutor Tests ====================

class TransLocMSToLocMemExecutorTest : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

// Test: TransLocMSToLocMemExecutor struct size check
TEST_F(TransLocMSToLocMemExecutorTest, StructSize) { EXPECT_GT(sizeof(TransLocMSToLocMemExecutor), 0); }

// Test: TransLocMSToLocMemExecutor default constructor
TEST_F(TransLocMSToLocMemExecutorTest, DefaultConstructor)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    TransLocMSToLocMemExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Describe());
}

// Test: TransLocMSToLocMemExecutor parameterized constructor
TEST_F(TransLocMSToLocMemExecutorTest, ParameterizedConstructor)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    TransLocMSToLocMemExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Describe());
}

// Test: TransLocMSToLocMemExecutor Parser with zero values
TEST_F(TransLocMSToLocMemExecutorTest, ParserZeroValues)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    TransLocMSToLocMemExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Parser());
    EXPECT_NO_THROW(executor.Describe());
}

// Test: TransLocMSToLocMemExecutor Parser with max values
TEST_F(TransLocMSToLocMemExecutorTest, ParserMaxValues)
{
    CcuInstr instr;
    memset(&instr, 0xFF, sizeof(instr));

    TransLocMSToLocMemExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Parser());
    EXPECT_NO_THROW(executor.Describe());
}

// Test: TransLocMSToLocMemExecutor Parser with specific parameters
TEST_F(TransLocMSToLocMemExecutorTest, ParserSpecificParameters)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    instr.v1.transLocMSToLocMem.locGSAId = 100;
    instr.v1.transLocMSToLocMem.locXnId = 50;
    instr.v1.transLocMSToLocMem.locMSId = 0x8000 | 10;
    instr.v1.transLocMSToLocMem.lengthXnId = 5;
    instr.v1.transLocMSToLocMem.channelId = 0;
    instr.v1.transLocMSToLocMem.lengthEn = 1;

    TransLocMSToLocMemExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Parser());
    std::string desc = executor.Describe();
    EXPECT_FALSE(desc.empty());
}

// Test: TransLocMSToLocMemExecutor with different lengthEn values
TEST_F(TransLocMSToLocMemExecutorTest, DifferentLengthEnValues)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    for (uint16_t lenEn = 0; lenEn <= 1; lenEn++) {
        instr.v1.transLocMSToLocMem.lengthEn = lenEn;
        TransLocMSToLocMemExecutor executor(0, 0, 0, instr, nullptr);
        EXPECT_NO_THROW(executor.Parser());
        EXPECT_NO_THROW(executor.Describe());
    }
}

// Test: TransLocMSToLocMemExecutor with different MS IDs
TEST_F(TransLocMSToLocMemExecutorTest, DifferentMsIds)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    uint16_t msIds[] = {0, 1, 100, SimCcuV1::CCU_RESOURCE_MS_NUM - 1, 0x7FFF, 0xFFFF};

    for (auto msId : msIds) {
        instr.v1.transLocMSToLocMem.locMSId = msId;
        TransLocMSToLocMemExecutor executor(0, 0, 0, instr, nullptr);
        EXPECT_NO_THROW(executor.Parser());
        EXPECT_NO_THROW(executor.Describe());
    }
}

// Test: TransLocMSToLocMemExecutor Describe contains expected keywords
TEST_F(TransLocMSToLocMemExecutorTest, DescribeContent)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    instr.v1.transLocMSToLocMem.locGSAId = 10;
    instr.v1.transLocMSToLocMem.locMSId = 100;

    TransLocMSToLocMemExecutor executor(0, 0, 0, instr, nullptr);
    executor.Parser();
    std::string desc = executor.Describe();
    EXPECT_FALSE(desc.empty());
    EXPECT_NE(desc.find("LocMSToLocMem"), std::string::npos);
}

// Test: TransLocMSToLocMemExecutor inheritance check
TEST_F(TransLocMSToLocMemExecutorTest, InheritanceCheck)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    TransLocMSToLocMemExecutor executor(0, 0, 0, instr, nullptr);
    CcuExecutorBase* base = &executor;
    EXPECT_NE(base, nullptr);
}

// ==================== TransLocMSToLocMSExecutor Tests ====================

class TransLocMSToLocMSExecutorTest : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

// Test: TransLocMSToLocMSExecutor struct size check
TEST_F(TransLocMSToLocMSExecutorTest, StructSize) { EXPECT_GT(sizeof(TransLocMSToLocMSExecutor), 0); }

// Test: TransLocMSToLocMSExecutor default constructor
TEST_F(TransLocMSToLocMSExecutorTest, DefaultConstructor)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    TransLocMSToLocMSExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Describe());
}

// Test: TransLocMSToLocMSExecutor parameterized constructor
TEST_F(TransLocMSToLocMSExecutorTest, ParameterizedConstructor)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    TransLocMSToLocMSExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Describe());
}

// Test: TransLocMSToLocMSExecutor Parser with zero values
TEST_F(TransLocMSToLocMSExecutorTest, ParserZeroValues)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    TransLocMSToLocMSExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Parser());
    EXPECT_NO_THROW(executor.Describe());
}

// Test: TransLocMSToLocMSExecutor Parser with max values
TEST_F(TransLocMSToLocMSExecutorTest, ParserMaxValues)
{
    CcuInstr instr;
    memset(&instr, 0xFF, sizeof(instr));

    TransLocMSToLocMSExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Parser());
    EXPECT_NO_THROW(executor.Describe());
}

// Test: TransLocMSToLocMSExecutor Parser with specific parameters
TEST_F(TransLocMSToLocMSExecutorTest, ParserSpecificParameters)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    instr.v1.transLocMSToLocMS.dstMSId = 100;
    instr.v1.transLocMSToLocMS.srcMSId = 50;
    instr.v1.transLocMSToLocMS.lengthXnId = 5;
    instr.v1.transLocMSToLocMS.lengthEn = 1;

    TransLocMSToLocMSExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Parser());
    std::string desc = executor.Describe();
    EXPECT_FALSE(desc.empty());
}

// Test: TransLocMSToLocMSExecutor with different source and destination MS IDs
TEST_F(TransLocMSToLocMSExecutorTest, DifferentMsIds)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    uint16_t msIds[] = {0, 1, 100, SimCcuV1::CCU_RESOURCE_MS_NUM - 1};

    for (auto srcMsId : msIds) {
        for (auto dstMsId : msIds) {
            instr.v1.transLocMSToLocMS.srcMSId = srcMsId;
            instr.v1.transLocMSToLocMS.dstMSId = dstMsId;
            TransLocMSToLocMSExecutor executor(0, 0, 0, instr, nullptr);
            EXPECT_NO_THROW(executor.Parser());
            EXPECT_NO_THROW(executor.Describe());
        }
    }
}

// Test: TransLocMSToLocMSExecutor Describe contains expected keywords
TEST_F(TransLocMSToLocMSExecutorTest, DescribeContent)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    instr.v1.transLocMSToLocMS.srcMSId = 10;
    instr.v1.transLocMSToLocMS.dstMSId = 100;

    TransLocMSToLocMSExecutor executor(0, 0, 0, instr, nullptr);
    executor.Parser();
    std::string desc = executor.Describe();
    EXPECT_FALSE(desc.empty());
    EXPECT_NE(desc.find("Trans LocMS"), std::string::npos);
}

// Test: TransLocMSToLocMSExecutor inheritance check
TEST_F(TransLocMSToLocMSExecutorTest, InheritanceCheck)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    TransLocMSToLocMSExecutor executor(0, 0, 0, instr, nullptr);
    CcuExecutorBase* base = &executor;
    EXPECT_NE(base, nullptr);
}

// ==================== TransLocMSToRmtMemExecutor Tests ====================

class TransLocMSToRmtMemExecutorTest : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

// Test: TransLocMSToRmtMemExecutor struct size check
TEST_F(TransLocMSToRmtMemExecutorTest, StructSize) { EXPECT_GT(sizeof(TransLocMSToRmtMemExecutor), 0); }

// Test: TransLocMSToRmtMemExecutor default constructor
TEST_F(TransLocMSToRmtMemExecutorTest, DefaultConstructor)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    TransLocMSToRmtMemExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Describe());
}

// Test: TransLocMSToRmtMemExecutor parameterized constructor
TEST_F(TransLocMSToRmtMemExecutorTest, ParameterizedConstructor)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    TransLocMSToRmtMemExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Describe());
}

// Test: TransLocMSToRmtMemExecutor Parser with zero values
TEST_F(TransLocMSToRmtMemExecutorTest, ParserZeroValues)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    TransLocMSToRmtMemExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Parser());
    EXPECT_NO_THROW(executor.Describe());
}

// Test: TransLocMSToRmtMemExecutor Parser with max values
TEST_F(TransLocMSToRmtMemExecutorTest, ParserMaxValues)
{
    CcuInstr instr;
    memset(&instr, 0xFF, sizeof(instr));

    TransLocMSToRmtMemExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Parser());
    EXPECT_NO_THROW(executor.Describe());
}

// Test: TransLocMSToRmtMemExecutor Parser with specific parameters
TEST_F(TransLocMSToRmtMemExecutorTest, ParserSpecificParameters)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    instr.v1.transLocMSToRmtMem.rmtGSAId = 100;
    instr.v1.transLocMSToRmtMem.rmtXnId = 50;
    instr.v1.transLocMSToRmtMem.locMSId = 10;
    instr.v1.transLocMSToRmtMem.lengthXnId = 5;
    instr.v1.transLocMSToRmtMem.channelId = 1;

    TransLocMSToRmtMemExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Parser());
    std::string desc = executor.Describe();
    EXPECT_FALSE(desc.empty());
}

// Test: TransLocMSToRmtMemExecutor Describe contains expected keywords
TEST_F(TransLocMSToRmtMemExecutorTest, DescribeContent)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    instr.v1.transLocMSToRmtMem.locMSId = 10;
    instr.v1.transLocMSToRmtMem.rmtGSAId = 100;

    TransLocMSToRmtMemExecutor executor(0, 0, 0, instr, nullptr);
    executor.Parser();
    std::string desc = executor.Describe();
    EXPECT_FALSE(desc.empty());
    EXPECT_NE(desc.find("Trans LocMS"), std::string::npos);
}

// Test: TransLocMSToRmtMemExecutor inheritance check
TEST_F(TransLocMSToRmtMemExecutorTest, InheritanceCheck)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    TransLocMSToRmtMemExecutor executor(0, 0, 0, instr, nullptr);
    CcuExecutorBase* base = &executor;
    EXPECT_NE(base, nullptr);
}

// ==================== TransLocMSToRmtMSExecutor Tests ====================

class TransLocMSToRmtMSExecutorTest : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

// Test: TransLocMSToRmtMSExecutor struct size check
TEST_F(TransLocMSToRmtMSExecutorTest, StructSize) { EXPECT_GT(sizeof(TransLocMSToRmtMSExecutor), 0); }

// Test: TransLocMSToRmtMSExecutor default constructor
TEST_F(TransLocMSToRmtMSExecutorTest, DefaultConstructor)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    TransLocMSToRmtMSExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Describe());
}

// Test: TransLocMSToRmtMSExecutor parameterized constructor
TEST_F(TransLocMSToRmtMSExecutorTest, ParameterizedConstructor)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    TransLocMSToRmtMSExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Describe());
}

// Test: TransLocMSToRmtMSExecutor Parser with zero values
TEST_F(TransLocMSToRmtMSExecutorTest, ParserZeroValues)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    TransLocMSToRmtMSExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Parser());
    EXPECT_NO_THROW(executor.Describe());
}

// Test: TransLocMSToRmtMSExecutor Parser with max values
TEST_F(TransLocMSToRmtMSExecutorTest, ParserMaxValues)
{
    CcuInstr instr;
    memset(&instr, 0xFF, sizeof(instr));

    TransLocMSToRmtMSExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Parser());
    EXPECT_NO_THROW(executor.Describe());
}

// Test: TransLocMSToRmtMSExecutor Parser with specific parameters
TEST_F(TransLocMSToRmtMSExecutorTest, ParserSpecificParameters)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    instr.v1.transLocMSToRmtMS.rmtMSId = 100;
    instr.v1.transLocMSToRmtMS.locMSId = 50;
    instr.v1.transLocMSToRmtMS.lengthXnId = 5;
    instr.v1.transLocMSToRmtMS.channelId = 10;
    instr.v1.transLocMSToRmtMS.setRmtCKEId = 20;
    instr.v1.transLocMSToRmtMS.setRmtCKEMask = 0xFF;

    TransLocMSToRmtMSExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Parser());
    std::string desc = executor.Describe();
    EXPECT_FALSE(desc.empty());
}

// Test: TransLocMSToRmtMSExecutor with different channel IDs
TEST_F(TransLocMSToRmtMSExecutorTest, DifferentChannelIds)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    for (uint16_t ch = 0; ch < 16; ch++) {
        instr.v1.transLocMSToRmtMS.channelId = ch;
        TransLocMSToRmtMSExecutor executor(0, 0, 0, instr, nullptr);
        EXPECT_NO_THROW(executor.Parser());
        EXPECT_NO_THROW(executor.Describe());
    }
}

// Test: TransLocMSToRmtMSExecutor Describe contains expected keywords
TEST_F(TransLocMSToRmtMSExecutorTest, DescribeContent)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    instr.v1.transLocMSToRmtMS.locMSId = 10;
    instr.v1.transLocMSToRmtMS.rmtMSId = 100;
    instr.v1.transLocMSToRmtMS.channelId = 5;

    TransLocMSToRmtMSExecutor executor(0, 0, 0, instr, nullptr);
    executor.Parser();
    std::string desc = executor.Describe();
    EXPECT_FALSE(desc.empty());
    EXPECT_NE(desc.find("Trans LocMS"), std::string::npos);
}

// Test: TransLocMSToRmtMSExecutor inheritance check
TEST_F(TransLocMSToRmtMSExecutorTest, InheritanceCheck)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    TransLocMSToRmtMSExecutor executor(0, 0, 0, instr, nullptr);
    CcuExecutorBase* base = &executor;
    EXPECT_NE(base, nullptr);
}

// ==================== TransRmtMSToLocMSExecutor Tests ====================

class TransRmtMSToLocMSExecutorTest : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(TransRmtMSToLocMSExecutorTest, StructSize) { EXPECT_GT(sizeof(TransRmtMSToLocMSExecutor), 0); }

TEST_F(TransRmtMSToLocMSExecutorTest, DefaultConstructor)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    TransRmtMSToLocMSExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Describe());
}

TEST_F(TransRmtMSToLocMSExecutorTest, ParameterizedConstructor)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    TransRmtMSToLocMSExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Describe());
}

TEST_F(TransRmtMSToLocMSExecutorTest, ParserZeroValues)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    TransRmtMSToLocMSExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Parser());
    EXPECT_NO_THROW(executor.Describe());
}

TEST_F(TransRmtMSToLocMSExecutorTest, ParserMaxValues)
{
    CcuInstr instr;
    memset(&instr, 0xFF, sizeof(instr));
    TransRmtMSToLocMSExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Parser());
    EXPECT_NO_THROW(executor.Describe());
}

TEST_F(TransRmtMSToLocMSExecutorTest, InheritanceCheck)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    TransRmtMSToLocMSExecutor executor(0, 0, 0, instr, nullptr);
    CcuExecutorBase* base = &executor;
    EXPECT_NE(base, nullptr);
}

// ==================== TransRmtMSToLocMemExecutor Tests ====================

class TransRmtMSToLocMemExecutorTest : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(TransRmtMSToLocMemExecutorTest, StructSize) { EXPECT_GT(sizeof(TransRmtMSToLocMemExecutor), 0); }

TEST_F(TransRmtMSToLocMemExecutorTest, DefaultConstructor)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    TransRmtMSToLocMemExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Describe());
}

TEST_F(TransRmtMSToLocMemExecutorTest, ParameterizedConstructor)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    TransRmtMSToLocMemExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Describe());
}

TEST_F(TransRmtMSToLocMemExecutorTest, ParserZeroValues)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    TransRmtMSToLocMemExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Parser());
    EXPECT_NO_THROW(executor.Describe());
}

TEST_F(TransRmtMSToLocMemExecutorTest, ParserMaxValues)
{
    CcuInstr instr;
    memset(&instr, 0xFF, sizeof(instr));
    TransRmtMSToLocMemExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Parser());
    EXPECT_NO_THROW(executor.Describe());
}

TEST_F(TransRmtMSToLocMemExecutorTest, InheritanceCheck)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    TransRmtMSToLocMemExecutor executor(0, 0, 0, instr, nullptr);
    CcuExecutorBase* base = &executor;
    EXPECT_NE(base, nullptr);
}

// ==================== TransRmtMemToLocMSExecutor Tests ====================

class TransRmtMemToLocMSExecutorTest : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(TransRmtMemToLocMSExecutorTest, StructSize) { EXPECT_GT(sizeof(TransRmtMemToLocMSExecutor), 0); }

TEST_F(TransRmtMemToLocMSExecutorTest, DefaultConstructor)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    TransRmtMemToLocMSExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Describe());
}

TEST_F(TransRmtMemToLocMSExecutorTest, ParameterizedConstructor)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    TransRmtMemToLocMSExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Describe());
}

TEST_F(TransRmtMemToLocMSExecutorTest, ParserZeroValues)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    TransRmtMemToLocMSExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Parser());
    EXPECT_NO_THROW(executor.Describe());
}

TEST_F(TransRmtMemToLocMSExecutorTest, ParserMaxValues)
{
    CcuInstr instr;
    memset(&instr, 0xFF, sizeof(instr));
    TransRmtMemToLocMSExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Parser());
    EXPECT_NO_THROW(executor.Describe());
}

TEST_F(TransRmtMemToLocMSExecutorTest, InheritanceCheck)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    TransRmtMemToLocMSExecutor executor(0, 0, 0, instr, nullptr);
    CcuExecutorBase* base = &executor;
    EXPECT_NE(base, nullptr);
}

// ==================== TransRmtMemToLocMemExecutor Tests ====================

class TransRmtMemToLocMemExecutorTest : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(TransRmtMemToLocMemExecutorTest, StructSize) { EXPECT_GT(sizeof(TransRmtMemToLocMemExecutor), 0); }

TEST_F(TransRmtMemToLocMemExecutorTest, DefaultConstructor)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    TransRmtMemToLocMemExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Describe());
}

TEST_F(TransRmtMemToLocMemExecutorTest, ParameterizedConstructor)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    TransRmtMemToLocMemExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Describe());
}

TEST_F(TransRmtMemToLocMemExecutorTest, ParserZeroValues)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    TransRmtMemToLocMemExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Parser());
    EXPECT_NO_THROW(executor.Describe());
}

TEST_F(TransRmtMemToLocMemExecutorTest, ParserMaxValues)
{
    CcuInstr instr;
    memset(&instr, 0xFF, sizeof(instr));
    TransRmtMemToLocMemExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Parser());
    EXPECT_NO_THROW(executor.Describe());
}

TEST_F(TransRmtMemToLocMemExecutorTest, InheritanceCheck)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    TransRmtMemToLocMemExecutor executor(0, 0, 0, instr, nullptr);
    CcuExecutorBase* base = &executor;
    EXPECT_NE(base, nullptr);
}

// ==================== Process() & Run() Tests ====================

TEST_F(TransLocMemToLocMemExecutorTest, ProcessWithZeroValues)
{
    auto& mgr = CcuResourceManager::GetInstance();
    mgr.Init(0, 2, RunnerCcuVersion::CCU_V1, {});
    mgr.UpdateGsaValue(0, 0, 0, 0x1000);
    mgr.UpdateGsaValue(0, 0, 1, 0x2000);
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    instr.v1.transLocMemToLocMem.srcGSAId = 0;
    instr.v1.transLocMemToLocMem.dstGSAId = 1;
    instr.v1.transLocMemToLocMem.lengthEn = 0;
    instr.v1.transLocMemToLocMem.setCKEId = 0;
    instr.v1.transLocMemToLocMem.setCKEMask = 0;
    CcuSimulator sim(0, 0, 0, 1, 1, RunnerCcuVersion::CCU_V1);
    TransLocMemToLocMemExecutor executor(0, 0, 0, instr, &sim);
    executor.Parser();
    EXPECT_NO_THROW(executor.Process(mgr));
}

TEST_F(TransLocMemToLocMemExecutorTest, RunWithCkeNotSatisfied)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    instr.v1.transLocMemToLocMem.waitCKEId = 0;
    instr.v1.transLocMemToLocMem.waitCKEMask = 0x0001;
    instr.v1.transLocMemToLocMem.clearType = 0;
    CcuSimulator sim(0, 0, 0, 1, 1, RunnerCcuVersion::CCU_V1);
    TransLocMemToLocMemExecutor executor(0, 0, 0, instr, &sim);
    executor.Parser();
    executor.Run();
}

TEST_F(TransLocMemToLocMSExecutorTest, ProcessWithZeroValues)
{
    auto& mgr = CcuResourceManager::GetInstance();
    mgr.Init(0, 2, RunnerCcuVersion::CCU_V1, {});
    mgr.UpdateGsaValue(0, 0, 0, 0x1000);
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    instr.v1.transLocMemToLocMS.locGSAId = 0;
    instr.v1.transLocMemToLocMS.locMSId = 0;
    instr.v1.transLocMemToLocMS.lengthEn = 0;
    instr.v1.transLocMemToLocMS.setCKEId = 0;
    instr.v1.transLocMemToLocMS.setCKEMask = 0;
    CcuSimulator sim(0, 0, 0, 1, 1, RunnerCcuVersion::CCU_V1);
    TransLocMemToLocMSExecutor executor(0, 0, 0, instr, &sim);
    executor.Parser();
    EXPECT_NO_THROW(executor.Process(mgr));
}

TEST_F(TransLocMemToLocMSExecutorTest, RunWithCkeNotSatisfied)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    instr.v1.transLocMemToLocMS.waitCKEId = 0;
    instr.v1.transLocMemToLocMS.waitCKEMask = 0x0001;
    instr.v1.transLocMemToLocMS.clearType = 0;
    CcuSimulator sim(0, 0, 0, 1, 1, RunnerCcuVersion::CCU_V1);
    TransLocMemToLocMSExecutor executor(0, 0, 0, instr, &sim);
    executor.Parser();
    executor.Run();
}

TEST_F(TransLocMemToRmtMemExecutorTest, ProcessWithZeroValues)
{
    auto& mgr = CcuResourceManager::GetInstance();
    mgr.Init(0, 2, RunnerCcuVersion::CCU_V1, {});
    RankChannelInfo chInfo;
    CcuInfo rmtInfo{1, 0};
    chInfo[0][0] = rmtInfo;
    mgr.InitChannelInfo(0, chInfo);
    mgr.UpdateGsaValue(0, 0, 0, 0x1000);
    mgr.UpdateGsaValue(0, 0, 1, 0x2000);
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    instr.v1.transLocMemToRmtMem.rmtGSAId = 1;
    instr.v1.transLocMemToRmtMem.locGSAId = 0;
    instr.v1.transLocMemToRmtMem.lengthEn = 0;
    instr.v1.transLocMemToRmtMem.setCKEId = 0;
    instr.v1.transLocMemToRmtMem.setCKEMask = 0;
    CcuSimulator sim(0, 0, 0, 1, 1, RunnerCcuVersion::CCU_V1);
    TransLocMemToRmtMemExecutor executor(0, 0, 0, instr, &sim);
    executor.Parser();
    EXPECT_NO_THROW(executor.Process(mgr));
}

TEST_F(TransLocMemToRmtMemExecutorTest, RunWithCkeNotSatisfied)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    instr.v1.transLocMemToRmtMem.waitCKEId = 0;
    instr.v1.transLocMemToRmtMem.waitCKEMask = 0x0001;
    instr.v1.transLocMemToRmtMem.clearType = 0;
    CcuSimulator sim(0, 0, 0, 1, 1, RunnerCcuVersion::CCU_V1);
    TransLocMemToRmtMemExecutor executor(0, 0, 0, instr, &sim);
    executor.Parser();
    executor.Run();
}

TEST_F(TransLocMSToLocMemExecutorTest, ProcessWithZeroValues)
{
    auto& mgr = CcuResourceManager::GetInstance();
    mgr.Init(0, 2, RunnerCcuVersion::CCU_V1, {});
    mgr.UpdateGsaValue(0, 0, 0, 0x1000);
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    instr.v1.transLocMSToLocMem.locGSAId = 0;
    instr.v1.transLocMSToLocMem.locMSId = 0;
    instr.v1.transLocMSToLocMem.lengthEn = 0;
    instr.v1.transLocMSToLocMem.setCKEId = 0;
    instr.v1.transLocMSToLocMem.setCKEMask = 0;
    CcuSimulator sim(0, 0, 0, 1, 1, RunnerCcuVersion::CCU_V1);
    TransLocMSToLocMemExecutor executor(0, 0, 0, instr, &sim);
    executor.Parser();
    EXPECT_NO_THROW(executor.Process(mgr));
}

TEST_F(TransLocMSToLocMemExecutorTest, RunWithCkeNotSatisfied)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    instr.v1.transLocMSToLocMem.waitCKEId = 0;
    instr.v1.transLocMSToLocMem.waitCKEMask = 0x0001;
    instr.v1.transLocMSToLocMem.clearType = 0;
    CcuSimulator sim(0, 0, 0, 1, 1, RunnerCcuVersion::CCU_V1);
    TransLocMSToLocMemExecutor executor(0, 0, 0, instr, &sim);
    executor.Parser();
    executor.Run();
}

TEST_F(TransLocMSToLocMSExecutorTest, ProcessWithZeroValues)
{
    auto& mgr = CcuResourceManager::GetInstance();
    mgr.Init(0, 2, RunnerCcuVersion::CCU_V1, {});
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    instr.v1.transLocMSToLocMS.srcMSId = 0;
    instr.v1.transLocMSToLocMS.dstMSId = 0;
    instr.v1.transLocMSToLocMS.lengthEn = 0;
    instr.v1.transLocMSToLocMS.setCKEId = 0;
    instr.v1.transLocMSToLocMS.setCKEMask = 0;
    CcuSimulator sim(0, 0, 0, 1, 1, RunnerCcuVersion::CCU_V1);
    TransLocMSToLocMSExecutor executor(0, 0, 0, instr, &sim);
    executor.Parser();
    EXPECT_NO_THROW(executor.Process(mgr));
}

TEST_F(TransLocMSToLocMSExecutorTest, RunWithCkeNotSatisfied)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    instr.v1.transLocMSToLocMS.waitCKEId = 0;
    instr.v1.transLocMSToLocMS.waitCKEMask = 0x0001;
    instr.v1.transLocMSToLocMS.clearType = 0;
    CcuSimulator sim(0, 0, 0, 1, 1, RunnerCcuVersion::CCU_V1);
    TransLocMSToLocMSExecutor executor(0, 0, 0, instr, &sim);
    executor.Parser();
    executor.Run();
}

TEST_F(TransLocMSToRmtMemExecutorTest, ProcessWithZeroValues)
{
    auto& mgr = CcuResourceManager::GetInstance();
    mgr.Init(0, 2, RunnerCcuVersion::CCU_V1, {});
    mgr.UpdateGsaValue(0, 0, 0, 0x1000);
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    instr.v1.transLocMSToRmtMem.rmtGSAId = 0;
    instr.v1.transLocMSToRmtMem.locMSId = 0;
    instr.v1.transLocMSToRmtMem.lengthEn = 0;
    instr.v1.transLocMSToRmtMem.setCKEId = 0;
    instr.v1.transLocMSToRmtMem.setCKEMask = 0;
    CcuSimulator sim(0, 0, 0, 1, 1, RunnerCcuVersion::CCU_V1);
    TransLocMSToRmtMemExecutor executor(0, 0, 0, instr, &sim);
    executor.Parser();
    EXPECT_NO_THROW(executor.Process(mgr));
}

TEST_F(TransLocMSToRmtMemExecutorTest, RunWithCkeNotSatisfied)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    instr.v1.transLocMSToRmtMem.waitCKEId = 0;
    instr.v1.transLocMSToRmtMem.waitCKEMask = 0x0001;
    instr.v1.transLocMSToRmtMem.clearType = 0;
    CcuSimulator sim(0, 0, 0, 1, 1, RunnerCcuVersion::CCU_V1);
    TransLocMSToRmtMemExecutor executor(0, 0, 0, instr, &sim);
    executor.Parser();
    executor.Run();
}

TEST_F(TransLocMSToRmtMSExecutorTest, ProcessWithInvalidChannelId)
{
    auto& mgr = CcuResourceManager::GetInstance();
    mgr.Init(0, 2, RunnerCcuVersion::CCU_V1, {});
    RankChannelInfo chInfo;
    CcuInfo rmtInfo{1, 0};
    chInfo[0][0] = rmtInfo;
    mgr.InitChannelInfo(0, chInfo);
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    instr.v1.transLocMSToRmtMS.channelId = SimCcuV1::MAX_CCU_CHANNEL_NUM;
    instr.v1.transLocMSToRmtMS.locMSId = 0;
    instr.v1.transLocMSToRmtMS.rmtMSId = 0;
    instr.v1.transLocMSToRmtMS.lengthEn = 0;
    instr.v1.transLocMSToRmtMS.setCKEId = 0;
    instr.v1.transLocMSToRmtMS.setCKEMask = 0;
    instr.v1.transLocMSToRmtMS.setRmtCKEId = 0;
    instr.v1.transLocMSToRmtMS.setRmtCKEMask = 0;
    CcuSimulator sim(0, 0, 0, 1, 1, RunnerCcuVersion::CCU_V1);
    TransLocMSToRmtMSExecutor executor(0, 0, 0, instr, &sim);
    executor.Parser();
    EXPECT_NO_THROW(executor.Process(mgr));
}

TEST_F(TransLocMSToRmtMSExecutorTest, RunWithCkeNotSatisfied)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    instr.v1.transLocMSToRmtMS.waitCKEId = 0;
    instr.v1.transLocMSToRmtMS.waitCKEMask = 0x0001;
    instr.v1.transLocMSToRmtMS.clearType = 0;
    CcuSimulator sim(0, 0, 0, 1, 1, RunnerCcuVersion::CCU_V1);
    TransLocMSToRmtMSExecutor executor(0, 0, 0, instr, &sim);
    executor.Parser();
    executor.Run();
}

TEST_F(TransRmtMemToLocMemExecutorTest, RunWithCkeNotSatisfied)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    instr.v1.transRmtMemToLocMem.waitCKEId = 0;
    instr.v1.transRmtMemToLocMem.waitCKEMask = 0x0001;
    instr.v1.transRmtMemToLocMem.clearType = 0;
    CcuSimulator sim(0, 0, 0, 1, 1, RunnerCcuVersion::CCU_V1);
    TransRmtMemToLocMemExecutor executor(0, 0, 0, instr, &sim);
    executor.Parser();
    executor.Run();
}

TEST_F(TransRmtMemToLocMSExecutorTest, ProcessWithZeroValues)
{
    auto& mgr = CcuResourceManager::GetInstance();
    mgr.Init(0, 2, RunnerCcuVersion::CCU_V1, {});
    mgr.UpdateGsaValue(0, 0, 0, 0x1000);
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    instr.v1.transRmtMemToLocMS.rmtGSAId = 0;
    instr.v1.transRmtMemToLocMS.locMSId = 0;
    instr.v1.transRmtMemToLocMS.lengthEn = 0;
    instr.v1.transRmtMemToLocMS.setCKEId = 0;
    instr.v1.transRmtMemToLocMS.setCKEMask = 0;
    CcuSimulator sim(0, 0, 0, 1, 1, RunnerCcuVersion::CCU_V1);
    TransRmtMemToLocMSExecutor executor(0, 0, 0, instr, &sim);
    executor.Parser();
    EXPECT_NO_THROW(executor.Process(mgr));
}

TEST_F(TransRmtMemToLocMSExecutorTest, RunWithCkeNotSatisfied)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    instr.v1.transRmtMemToLocMS.waitCKEId = 0;
    instr.v1.transRmtMemToLocMS.waitCKEMask = 0x0001;
    instr.v1.transRmtMemToLocMS.clearType = 0;
    CcuSimulator sim(0, 0, 0, 1, 1, RunnerCcuVersion::CCU_V1);
    TransRmtMemToLocMSExecutor executor(0, 0, 0, instr, &sim);
    executor.Parser();
    executor.Run();
}

TEST_F(TransRmtMSToLocMemExecutorTest, ProcessWithInvalidChannelId)
{
    auto& mgr = CcuResourceManager::GetInstance();
    mgr.Init(0, 2, RunnerCcuVersion::CCU_V1, {});
    RankChannelInfo chInfo;
    CcuInfo rmtInfo{1, 0};
    chInfo[0][0] = rmtInfo;
    mgr.InitChannelInfo(0, chInfo);
    mgr.UpdateGsaValue(0, 0, 0, 0x1000);
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    instr.v1.transRmtMSToLocMem.channelId = SimCcuV1::MAX_CCU_CHANNEL_NUM;
    instr.v1.transRmtMSToLocMem.locGSAId = 0;
    instr.v1.transRmtMSToLocMem.rmtMSId = 0;
    instr.v1.transRmtMSToLocMem.lengthEn = 0;
    instr.v1.transRmtMSToLocMem.setCKEId = 0;
    instr.v1.transRmtMSToLocMem.setCKEMask = 0;
    CcuSimulator sim(0, 0, 0, 1, 1, RunnerCcuVersion::CCU_V1);
    TransRmtMSToLocMemExecutor executor(0, 0, 0, instr, &sim);
    executor.Parser();
    EXPECT_NO_THROW(executor.Process(mgr));
}

TEST_F(TransRmtMSToLocMemExecutorTest, RunWithCkeNotSatisfied)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    instr.v1.transRmtMSToLocMem.waitCKEId = 0;
    instr.v1.transRmtMSToLocMem.waitCKEMask = 0x0001;
    instr.v1.transRmtMSToLocMem.clearType = 0;
    CcuSimulator sim(0, 0, 0, 1, 1, RunnerCcuVersion::CCU_V1);
    TransRmtMSToLocMemExecutor executor(0, 0, 0, instr, &sim);
    executor.Parser();
    executor.Run();
}

TEST_F(TransRmtMSToLocMSExecutorTest, ProcessWithInvalidChannelId)
{
    auto& mgr = CcuResourceManager::GetInstance();
    mgr.Init(0, 2, RunnerCcuVersion::CCU_V1, {});
    RankChannelInfo chInfo;
    CcuInfo rmtInfo{1, 0};
    chInfo[0][0] = rmtInfo;
    mgr.InitChannelInfo(0, chInfo);
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    instr.v1.transRmtMSToLocMS.channelId = SimCcuV1::MAX_CCU_CHANNEL_NUM;
    instr.v1.transRmtMSToLocMS.locMSId = 0;
    instr.v1.transRmtMSToLocMS.rmtMSId = 0;
    instr.v1.transRmtMSToLocMS.lengthEn = 0;
    instr.v1.transRmtMSToLocMS.setCKEId = 0;
    instr.v1.transRmtMSToLocMS.setCKEMask = 0;
    CcuSimulator sim(0, 0, 0, 1, 1, RunnerCcuVersion::CCU_V1);
    TransRmtMSToLocMSExecutor executor(0, 0, 0, instr, &sim);
    executor.Parser();
    EXPECT_NO_THROW(executor.Process(mgr));
}

TEST_F(TransRmtMSToLocMSExecutorTest, RunWithCkeNotSatisfied)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    instr.v1.transRmtMSToLocMS.waitCKEId = 0;
    instr.v1.transRmtMSToLocMS.waitCKEMask = 0x0001;
    instr.v1.transRmtMSToLocMS.clearType = 0;
    CcuSimulator sim(0, 0, 0, 1, 1, RunnerCcuVersion::CCU_V1);
    TransRmtMSToLocMSExecutor executor(0, 0, 0, instr, &sim);
    executor.Parser();
    executor.Run();
}
