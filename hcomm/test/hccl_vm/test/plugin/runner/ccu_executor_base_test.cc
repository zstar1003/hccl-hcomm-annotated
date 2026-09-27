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
#include <thread>

#define private public
#include "ccu_executor_base.h"
#include "ccu_microcode_common_v1.h"
#include "ccu_resource_common.h"
#include "ccu_resource_manager.h"

#include "ccu_simulator.h"
#include "ccu_simulator_base.h"

#undef private

using namespace hcomm::CcuRep;

class MockCcuExecutor : public CcuExecutorBase {
public:
    MockCcuExecutor(int streamId, int rankId, int dieId, const CcuInstr& instr, CcuSimulator* ccuSimulator)
        : CcuExecutorBase(streamId, rankId, dieId, instr, ccuSimulator)
    {}

    bool processCalled{false};
    CcuResourceManager* processResMgr{nullptr};

    void Parser() override {}
    void Run() override {}
    std::string Describe() override { return "MockExecutor"; }
    void Process(CcuResourceManager& ccuResMgr) override
    {
        processCalled = true;
        processResMgr = &ccuResMgr;
    }
};

class CcuExecutorBaseTest : public testing::Test {
protected:
    void SetUp() override
    {
        auto& mgr = CcuResourceManager::GetInstance();
        mgr.Init(0, 4, RunnerCcuVersion::CCU_V1, {});

        CcuInstr instr;
        memset(&instr, 0, sizeof(CcuInstr));
        instr.v1.add.count = 0;
        for (uint16_t i = 0; i < CCU_REDUCE_MAX_MS; i++) {
            instr.v1.add.msId[i] = 0;
        }

        simulator_ = std::make_unique<CcuSimulator>(0, 0, 0, 10, 10, RunnerCcuVersion::CCU_V1);
        executor_ = std::make_unique<MockCcuExecutor>(0, 0, 0, instr, simulator_.get());
    }

    std::unique_ptr<CcuSimulator> simulator_;
    std::unique_ptr<MockCcuExecutor> executor_;
};

TEST_F(CcuExecutorBaseTest, SetCkeSignal)
{
    auto& mgr = CcuResourceManager::GetInstance();
    mgr.UpdateCkeValue(0, 0, 0, 0x0000);
    executor_->SetCkeSignal(mgr, 0, 0x00FF);
    uint16_t val = mgr.GetCkeValue(0, 0, 0);
    EXPECT_EQ(val, 0x00FF);
}

TEST_F(CcuExecutorBaseTest, SetCkeSignal_WithExistingValue)
{
    auto& mgr = CcuResourceManager::GetInstance();
    mgr.UpdateCkeValue(0, 0, 0, 0x00F0);
    executor_->SetCkeSignal(mgr, 0, 0x000F);
    uint16_t val = mgr.GetCkeValue(0, 0, 0);
    EXPECT_EQ(val, 0x00FF);
}

TEST_F(CcuExecutorBaseTest, SetRmtCKESignal)
{
    auto& mgr = CcuResourceManager::GetInstance();
    mgr.Init(1, 4, RunnerCcuVersion::CCU_V1, {});
    mgr.UpdateCkeValue(1, 0, 0, 0x0000);
    executor_->SetRmtCKESignal(mgr, 1, 0, 0, 0x00FF);
    uint16_t val = mgr.GetCkeValue(1, 0, 0);
    EXPECT_EQ(val, 0x00FF);
}

TEST_F(CcuExecutorBaseTest, ClearCkeSignal_NormalState)
{
    auto& mgr = CcuResourceManager::GetInstance();
    mgr.UpdateCkeValue(0, 0, 0, 0xFFFF);
    executor_->ClearCkeSignal(mgr, 0, 0x00FF);
    uint16_t val = mgr.GetCkeValue(0, 0, 0);
    EXPECT_EQ(val, 0xFF00);
}

TEST_F(CcuExecutorBaseTest, ClearCkeSignal_LoopState)
{
    auto& mgr = CcuResourceManager::GetInstance();
    simulator_->SetExecState(CcuExecState::EXEC_LOOP_INSTR);
    LoopGroupInfo loopGroupInfo;
    loopGroupInfo.ckeOffset_ = 5;
    loopGroupInfo.loopStatus_.loopExtendIndex = 1;
    simulator_->InitLoopGroupInfo(loopGroupInfo);

    mgr.UpdateCkeValue(0, 0, 5, 0xFFFF);
    executor_->ClearCkeSignal(mgr, 0, 0x00FF);
    uint16_t val = mgr.GetCkeValue(0, 0, 5);
    EXPECT_EQ(val, 0xFF00);
}

TEST_F(CcuExecutorBaseTest, ParseMSList)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(CcuInstr));
    instr.v1.add.count = 1;
    instr.v1.add.msId[0] = 0x8001;
    instr.v1.add.msId[1] = 0x8002;
    instr.v1.add.msId[2] = 0x8003;

    auto sim = std::make_unique<CcuSimulator>(0, 0, 0, 10, 10, RunnerCcuVersion::CCU_V1);
    auto exec = std::make_unique<MockCcuExecutor>(0, 0, 0, instr, sim.get());
    std::string result = exec->ParseMSList();
    EXPECT_FALSE(result.empty());
    EXPECT_NE(result.find("MS["), std::string::npos);
}

TEST_F(CcuExecutorBaseTest, ParseMSList_ZeroCount)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(CcuInstr));
    instr.v1.add.count = 0;
    instr.v1.add.msId[0] = 0x8001;
    instr.v1.add.msId[1] = 0x8002;

    auto sim = std::make_unique<CcuSimulator>(0, 0, 0, 10, 10, RunnerCcuVersion::CCU_V1);
    auto exec = std::make_unique<MockCcuExecutor>(0, 0, 0, instr, sim.get());
    std::string result = exec->ParseMSList();
    EXPECT_FALSE(result.empty());
}

TEST_F(CcuExecutorBaseTest, WaitCkeProcess_MaskMatched)
{
    auto& mgr = CcuResourceManager::GetInstance();
    mgr.UpdateCkeValue(0, 0, 0, 0xFFFF);
    executor_->WaitCkeProcess(0, 0xFFFF, 0, "test_instr");
    EXPECT_TRUE(executor_->processCalled);
    EXPECT_FALSE(simulator_->waitCKE_);
}

TEST_F(CcuExecutorBaseTest, WaitCkeProcess_MaskNotMatched)
{
    auto& mgr = CcuResourceManager::GetInstance();
    mgr.UpdateCkeValue(0, 0, 0, 0x0000);
    executor_->WaitCkeProcess(0, 0xFFFF, 0, "test_instr");
    EXPECT_FALSE(executor_->processCalled);
    EXPECT_TRUE(simulator_->waitCKE_);
}

TEST_F(CcuExecutorBaseTest, WaitCkeProcess_ZeroMask)
{
    auto& mgr = CcuResourceManager::GetInstance();
    executor_->WaitCkeProcess(0, 0, 0, "test_instr");
    EXPECT_TRUE(executor_->processCalled);
    EXPECT_FALSE(simulator_->waitCKE_);
}

TEST_F(CcuExecutorBaseTest, WaitCkeProcess_WithClearType)
{
    auto& mgr = CcuResourceManager::GetInstance();
    mgr.UpdateCkeValue(0, 0, 0, 0xFFFF);
    executor_->WaitCkeProcess(0, 0xFFFF, 1, "test_instr");
    EXPECT_TRUE(executor_->processCalled);
    uint16_t val = mgr.GetCkeValue(0, 0, 0);
    EXPECT_EQ(val, 0x0000);
}

TEST_F(CcuExecutorBaseTest, WaitCkeProcess_LoopState)
{
    auto& mgr = CcuResourceManager::GetInstance();
    simulator_->SetExecState(CcuExecState::EXEC_LOOP_INSTR);
    LoopGroupInfo loopGroupInfo;
    loopGroupInfo.ckeOffset_ = 3;
    loopGroupInfo.loopStatus_.loopExtendIndex = 1;
    simulator_->InitLoopGroupInfo(loopGroupInfo);

    mgr.UpdateCkeValue(0, 0, 3, 0xFFFF);
    executor_->WaitCkeProcess(0, 0xFFFF, 0, "test_instr_loop");
    EXPECT_TRUE(executor_->processCalled);
}

TEST_F(CcuExecutorBaseTest, Constructor)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(CcuInstr));
    auto sim = std::make_unique<CcuSimulator>(0, 0, 0, 10, 10, RunnerCcuVersion::CCU_V1);
    auto exec = std::make_unique<MockCcuExecutor>(1, 0, 0, instr, sim.get());
    EXPECT_EQ(exec->rankId_, 0);
    EXPECT_EQ(exec->dieId_, 0);
    EXPECT_EQ(exec->streamId_, 1);
    EXPECT_EQ(exec->ccuSimulator_, sim.get());
}

TEST_F(CcuExecutorBaseTest, Describe)
{
    std::string desc = executor_->Describe();
    EXPECT_EQ(desc, "MockExecutor");
}

TEST_F(CcuExecutorBaseTest, Process_Default)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(CcuInstr));
    auto sim = std::make_unique<CcuSimulator>(0, 0, 0, 10, 10, RunnerCcuVersion::CCU_V1);
    auto exec = std::make_unique<MockCcuExecutor>(0, 0, 0, instr, sim.get());
    auto& mgr = CcuResourceManager::GetInstance();
    EXPECT_NO_THROW(exec->Process(mgr));
}

TEST_F(CcuExecutorBaseTest, WaitCkeProcess_MaskNotMatched_DebugLog)
{
    auto& mgr = CcuResourceManager::GetInstance();
    mgr.UpdateCkeValue(0, 0, 0, 0x0000);
    executor_->WaitCkeProcess(0, 0xFFFF, 0, "test_instr");
    EXPECT_TRUE(simulator_->waitCKE_);
    std::this_thread::sleep_for(std::chrono::milliseconds(1100));
    CcuInstr instr;
    memset(&instr, 0, sizeof(CcuInstr));
    auto sim2 = std::make_unique<CcuSimulator>(0, 0, 0, 10, 10, RunnerCcuVersion::CCU_V1);
    auto exec2 = std::make_unique<MockCcuExecutor>(0, 0, 0, instr, sim2.get());
    mgr.UpdateCkeValue(0, 0, 0, 0x0000);
    exec2->WaitCkeProcess(0, 0xFFFF, 0, "test_instr_debug");
    EXPECT_TRUE(sim2->waitCKE_);
}

TEST_F(CcuExecutorBaseTest, StaticBlockingCountMap)
{
    CcuExecutorBase::s_blockingCountMap_["instr_a"] = 100;
    CcuExecutorBase::s_blockingCountMap_["instr_b"] = 200;
    EXPECT_EQ(CcuExecutorBase::s_blockingCountMap_["instr_a"], 100);
    EXPECT_EQ(CcuExecutorBase::s_blockingCountMap_["instr_b"], 200);
    EXPECT_EQ(CcuExecutorBase::s_blockingCountMap_.size(), 2);
    CcuExecutorBase::s_blockingCountMap_.clear();
    EXPECT_TRUE(CcuExecutorBase::s_blockingCountMap_.empty());
}

TEST_F(CcuExecutorBaseTest, SetCkeSignal_ZeroMask)
{
    auto& mgr = CcuResourceManager::GetInstance();
    mgr.UpdateCkeValue(0, 0, 0, 0xABCD);
    executor_->SetCkeSignal(mgr, 0, 0x0000);
    uint16_t val = mgr.GetCkeValue(0, 0, 0);
    EXPECT_EQ(val, 0xABCD);
}

TEST_F(CcuExecutorBaseTest, SetCkeSignal_FullMask)
{
    auto& mgr = CcuResourceManager::GetInstance();
    mgr.UpdateCkeValue(0, 0, 0, 0x0000);
    executor_->SetCkeSignal(mgr, 0, 0xFFFF);
    uint16_t val = mgr.GetCkeValue(0, 0, 0);
    EXPECT_EQ(val, 0xFFFF);
}

TEST_F(CcuExecutorBaseTest, ClearCkeSignal_ZeroMask_NoChange)
{
    auto& mgr = CcuResourceManager::GetInstance();
    mgr.UpdateCkeValue(0, 0, 0, 0xABCD);
    executor_->ClearCkeSignal(mgr, 0, 0x0000);
    uint16_t val = mgr.GetCkeValue(0, 0, 0);
    EXPECT_EQ(val, 0xABCD);
}

TEST_F(CcuExecutorBaseTest, ClearCkeSignal_FullClear)
{
    auto& mgr = CcuResourceManager::GetInstance();
    mgr.UpdateCkeValue(0, 0, 0, 0xFFFF);
    executor_->ClearCkeSignal(mgr, 0, 0xFFFF);
    uint16_t val = mgr.GetCkeValue(0, 0, 0);
    EXPECT_EQ(val, 0x0000);
}

TEST_F(CcuExecutorBaseTest, ClearCkeSignal_DifferentCkeId)
{
    auto& mgr = CcuResourceManager::GetInstance();
    mgr.UpdateCkeValue(0, 0, 3, 0xFFFF);
    executor_->ClearCkeSignal(mgr, 3, 0x00FF);
    uint16_t val = mgr.GetCkeValue(0, 0, 3);
    EXPECT_EQ(val, 0xFF00);
}

TEST_F(CcuExecutorBaseTest, SetRmtCKESignal_DifferentDie)
{
    auto& mgr = CcuResourceManager::GetInstance();
    mgr.UpdateCkeValue(0, 1, 0, 0x0000);
    executor_->SetRmtCKESignal(mgr, 0, 1, 0, 0x00FF);
    uint16_t val = mgr.GetCkeValue(0, 1, 0);
    EXPECT_EQ(val, 0x00FF);
}

TEST_F(CcuExecutorBaseTest, WaitCkeProcess_LoopState_WithClearType)
{
    auto& mgr = CcuResourceManager::GetInstance();
    simulator_->SetExecState(CcuExecState::EXEC_LOOP_INSTR);
    LoopGroupInfo loopGroupInfo;
    loopGroupInfo.ckeOffset_ = 2;
    loopGroupInfo.loopStatus_.loopExtendIndex = 1;
    simulator_->InitLoopGroupInfo(loopGroupInfo);

    mgr.UpdateCkeValue(0, 0, 2, 0xFFFF);
    mgr.UpdateCkeValue(0, 0, 4, 0xFFFF);
    executor_->WaitCkeProcess(0, 0xFFFF, 1, "test_instr_loop_clear");
    EXPECT_TRUE(executor_->processCalled);
    uint16_t val = mgr.GetCkeValue(0, 0, 4);
    EXPECT_EQ(val, 0x0000);
}

TEST_F(CcuExecutorBaseTest, WaitCkeProcess_MaskMatched_NoClear)
{
    auto& mgr = CcuResourceManager::GetInstance();
    mgr.UpdateCkeValue(0, 0, 0, 0xFFFF);
    executor_->WaitCkeProcess(0, 0xFFFF, 0, "test_instr_no_clear");
    EXPECT_TRUE(executor_->processCalled);
    EXPECT_FALSE(simulator_->waitCKE_);
    uint16_t val = mgr.GetCkeValue(0, 0, 0);
    EXPECT_EQ(val, 0xFFFF);
}

TEST_F(CcuExecutorBaseTest, ParseMSList_ExactFormat)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(CcuInstr));
    instr.v1.add.count = 0;
    instr.v1.add.msId[0] = 0x8001;
    instr.v1.add.msId[1] = 0x0002;

    auto sim = std::make_unique<CcuSimulator>(0, 0, 0, 10, 10, RunnerCcuVersion::CCU_V1);
    auto exec = std::make_unique<MockCcuExecutor>(0, 0, 0, instr, sim.get());
    std::string result = exec->ParseMSList();
    EXPECT_EQ(result, "MS[1:1, 0:2]");
}

TEST_F(CcuExecutorBaseTest, ParseMSList_WithLargerCount)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(CcuInstr));
    instr.v1.add.count = 2;
    instr.v1.add.msId[0] = 0x0000;
    instr.v1.add.msId[1] = 0x8001;
    instr.v1.add.msId[2] = 0x8002;
    instr.v1.add.msId[3] = 0x8003;

    auto sim = std::make_unique<CcuSimulator>(0, 0, 0, 10, 10, RunnerCcuVersion::CCU_V1);
    auto exec = std::make_unique<MockCcuExecutor>(0, 0, 0, instr, sim.get());
    std::string result = exec->ParseMSList();
    EXPECT_EQ(result, "MS[0:0, 1:1, 1:2, 1:3]");
}
