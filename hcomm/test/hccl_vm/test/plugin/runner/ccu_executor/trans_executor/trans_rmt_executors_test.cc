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
#include "ccu_resource_manager.h"
#include "ccu_simulator.h"
#include "trans_rmt_mem_to_loc_mem_executor.h"
#include "trans_rmt_mem_to_loc_ms_executor.h"
#include "trans_rmt_ms_to_loc_mem_executor.h"
#include "trans_rmt_ms_to_loc_ms_executor.h"

using namespace hcomm::CcuRep;

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

TEST_F(TransRmtMSToLocMSExecutorTest, ProcessWithDieIdMismatch)
{
    auto& mgr = CcuResourceManager::GetInstance();
    mgr.Init(0, 2, RunnerCcuVersion::CCU_V1, {});
    mgr.Init(1, 2, RunnerCcuVersion::CCU_V1, {});
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    instr.v1.transRmtMSToLocMS.channelId = 0;
    instr.v1.transRmtMSToLocMS.rmtMSId = 0x8000;
    instr.v1.transRmtMSToLocMS.locMSId = 0;
    instr.v1.transRmtMSToLocMS.waitCKEMask = 0;
    CcuSimulator sim(0, 0, 0, 1, 1, RunnerCcuVersion::CCU_V1);
    TransRmtMSToLocMSExecutor executor(0, 0, 0, instr, &sim);
    executor.Parser();
    EXPECT_NO_THROW(executor.Process(mgr));
}

TEST_F(TransRmtMSToLocMSExecutorTest, ProcessWithValidChannel)
{
    auto& mgr = CcuResourceManager::GetInstance();
    mgr.Init(0, 2, RunnerCcuVersion::CCU_V1, {});
    mgr.Init(1, 2, RunnerCcuVersion::CCU_V1, {});
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    instr.v1.transRmtMSToLocMS.channelId = 0;
    instr.v1.transRmtMSToLocMS.rmtMSId = 0;
    instr.v1.transRmtMSToLocMS.locMSId = 0;
    instr.v1.transRmtMSToLocMS.lengthEn = 1;
    instr.v1.transRmtMSToLocMS.lengthXnId = 0;
    instr.v1.transRmtMSToLocMS.waitCKEMask = 0;
    CcuSimulator sim(0, 0, 0, 1, 1, RunnerCcuVersion::CCU_V1);
    TransRmtMSToLocMSExecutor executor(0, 0, 0, instr, &sim);
    executor.Parser();
    EXPECT_NO_THROW(executor.Process(mgr));
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

TEST_F(TransRmtMSToLocMemExecutorTest, ProcessWithDieIdMismatch)
{
    auto& mgr = CcuResourceManager::GetInstance();
    mgr.Init(0, 2, RunnerCcuVersion::CCU_V1, {});
    mgr.Init(1, 2, RunnerCcuVersion::CCU_V1, {});
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    instr.v1.transRmtMSToLocMem.channelId = 0;
    instr.v1.transRmtMSToLocMem.rmtMSId = 0x8000;
    instr.v1.transRmtMSToLocMem.waitCKEMask = 0;
    CcuSimulator sim(0, 0, 0, 1, 1, RunnerCcuVersion::CCU_V1);
    TransRmtMSToLocMemExecutor executor(0, 0, 0, instr, &sim);
    executor.Parser();
    EXPECT_NO_THROW(executor.Process(mgr));
}

TEST_F(TransRmtMSToLocMemExecutorTest, ProcessWithNullAddressFails)
{
    auto& mgr = CcuResourceManager::GetInstance();
    mgr.Init(0, 2, RunnerCcuVersion::CCU_V1, {});
    mgr.Init(1, 2, RunnerCcuVersion::CCU_V1, {});
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    instr.v1.transRmtMSToLocMem.channelId = 0;
    instr.v1.transRmtMSToLocMem.rmtMSId = 0;
    instr.v1.transRmtMSToLocMem.waitCKEMask = 0;
    CcuSimulator sim(0, 0, 0, 1, 1, RunnerCcuVersion::CCU_V1);
    TransRmtMSToLocMemExecutor executor(0, 0, 0, instr, &sim);
    executor.Parser();
    EXPECT_NO_THROW(executor.Process(mgr));
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

TEST_F(TransRmtMemToLocMSExecutorTest, ProcessWithNullAddressFails)
{
    auto& mgr = CcuResourceManager::GetInstance();
    mgr.Init(0, 2, RunnerCcuVersion::CCU_V1, {});
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    instr.v1.transRmtMemToLocMS.waitCKEMask = 0;
    CcuSimulator sim(0, 0, 0, 1, 1, RunnerCcuVersion::CCU_V1);
    TransRmtMemToLocMSExecutor executor(0, 0, 0, instr, &sim);
    executor.Parser();
    EXPECT_NO_THROW(executor.Process(mgr));
}

TEST_F(TransRmtMemToLocMSExecutorTest, ProcessWithLengthEnEnabled)
{
    auto& mgr = CcuResourceManager::GetInstance();
    mgr.Init(0, 2, RunnerCcuVersion::CCU_V1, {});
    mgr.UpdateXnValue(0, 0, 0, 4096);
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    instr.v1.transRmtMemToLocMS.lengthEn = 1;
    instr.v1.transRmtMemToLocMS.lengthXnId = 0;
    instr.v1.transRmtMemToLocMS.waitCKEMask = 0;
    CcuSimulator sim(0, 0, 0, 1, 1, RunnerCcuVersion::CCU_V1);
    TransRmtMemToLocMSExecutor executor(0, 0, 0, instr, &sim);
    executor.Parser();
    EXPECT_NO_THROW(executor.Process(mgr));
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

TEST_F(TransRmtMemToLocMemExecutorTest, ProcessWithZeroValues)
{
    auto& mgr = CcuResourceManager::GetInstance();
    mgr.Init(0, 2, RunnerCcuVersion::CCU_V1, {});
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    instr.v1.transRmtMemToLocMem.rmtGSAId = 0;
    instr.v1.transRmtMemToLocMem.locGSAId = 0;
    instr.v1.transRmtMemToLocMem.lengthEn = 0;
    instr.v1.transRmtMemToLocMem.setCKEId = 0;
    instr.v1.transRmtMemToLocMem.setCKEMask = 0;
    CcuSimulator sim(0, 0, 0, 1, 1, RunnerCcuVersion::CCU_V1);
    TransRmtMemToLocMemExecutor executor(0, 0, 0, instr, &sim);
    executor.Parser();
    EXPECT_NO_THROW(executor.Process(mgr));
}

TEST_F(TransRmtMemToLocMSExecutorTest, ProcessWithZeroValues)
{
    auto& mgr = CcuResourceManager::GetInstance();
    mgr.Init(0, 2, RunnerCcuVersion::CCU_V1, {});
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

TEST_F(TransRmtMSToLocMemExecutorTest, ProcessWithInvalidChannelId)
{
    auto& mgr = CcuResourceManager::GetInstance();
    mgr.Init(0, 2, RunnerCcuVersion::CCU_V1, {});
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

TEST_F(TransRmtMSToLocMSExecutorTest, ProcessWithInvalidChannelId)
{
    auto& mgr = CcuResourceManager::GetInstance();
    mgr.Init(0, 2, RunnerCcuVersion::CCU_V1, {});
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

TEST_F(TransRmtMemToLocMemExecutorTest, ProcessWithNullAddress)
{
    auto& mgr = CcuResourceManager::GetInstance();
    mgr.Init(0, 2, RunnerCcuVersion::CCU_V1, {});
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    instr.v1.transRmtMemToLocMem.waitCKEMask = 0;
    CcuSimulator sim(0, 0, 0, 1, 1, RunnerCcuVersion::CCU_V1);
    TransRmtMemToLocMemExecutor executor(0, 0, 0, instr, &sim);
    executor.Parser();
    EXPECT_NO_THROW(executor.Process(mgr));
}

TEST_F(TransRmtMemToLocMemExecutorTest, ProcessWithLengthEnEnabled)
{
    auto& mgr = CcuResourceManager::GetInstance();
    mgr.Init(0, 2, RunnerCcuVersion::CCU_V1, {});
    mgr.UpdateXnValue(0, 0, 0, 4096);
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    instr.v1.transRmtMemToLocMem.lengthEn = 1;
    instr.v1.transRmtMemToLocMem.lengthXnId = 0;
    instr.v1.transRmtMemToLocMem.waitCKEMask = 0;
    CcuSimulator sim(0, 0, 0, 1, 1, RunnerCcuVersion::CCU_V1);
    TransRmtMemToLocMemExecutor executor(0, 0, 0, instr, &sim);
    executor.Parser();
    EXPECT_NO_THROW(executor.Process(mgr));
}
