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

#define private public
#include "ccu_microcode_common_v1.h"
#include "ccu_resource_manager.h"
#include "ccu_simulator.h"
#include "ccu_simulator_base.h"

#undef private

class CcuSimulatorTest : public testing::Test {
protected:
    void SetUp() override { simulator_ = std::make_unique<CcuSimulator>(0, 0, 0, 10, 10, RunnerCcuVersion::CCU_V1); }

    std::unique_ptr<CcuSimulator> simulator_;
};

TEST_F(CcuSimulatorTest, DefaultConstructor)
{
    CcuSimulator sim;
    EXPECT_EQ(sim.GetState(), CcuExecState::EXEC_NORMAL_INSTR);
    EXPECT_EQ(sim.GetCurInstrId(), 0);
}

TEST_F(CcuSimulatorTest, ParamConstructor)
{
    CcuSimulator sim(1, 2, 5, 15, 10, RunnerCcuVersion::CCU_V1);
    EXPECT_EQ(sim.GetState(), CcuExecState::EXEC_NORMAL_INSTR);
    EXPECT_EQ(sim.GetCurInstrId(), 5);
}

TEST_F(CcuSimulatorTest, SetExecState)
{
    simulator_->SetExecState(CcuExecState::EXEC_LOOP_INSTR);
    EXPECT_EQ(simulator_->GetState(), CcuExecState::EXEC_LOOP_INSTR);
}

TEST_F(CcuSimulatorTest, SetWaitCKEFlag)
{
    simulator_->SetWaitCKEFlag(true);
    simulator_->SetWaitCKEFlag(false);
}

TEST_F(CcuSimulatorTest, InitLoopGroupInfo_WithParams)
{
    uint16_t startLoopId = 5;
    uint64_t offsetCfg = 0x12345;
    uint64_t repeatCfg = 0x123456789ABCULL;

    simulator_->InitLoopGroupInfo(startLoopId, offsetCfg, repeatCfg);
    EXPECT_EQ(simulator_->GetState(), CcuExecState::EXEC_LOOPGROUP_INSTR);
}

TEST_F(CcuSimulatorTest, InitLoopInfo) { simulator_->InitLoopInfo(2, 8, 5, 100); }

TEST_F(CcuSimulatorTest, GetLoopGsaAddrOffset_NotInLoopState)
{
    uint64_t offset = simulator_->GetLoopGsaAddrOffset();
    EXPECT_EQ(offset, 0);
}

TEST_F(CcuSimulatorTest, GetLoopMsOffset_NotInLoopState)
{
    uint16_t offset = simulator_->GetLoopMsOffset();
    EXPECT_EQ(offset, 0);
}

TEST_F(CcuSimulatorTest, GetLoopCKEOffset_NotInLoopState)
{
    uint16_t offset = simulator_->GetLoopCKEOffset();
    EXPECT_EQ(offset, 0);
}

TEST_F(CcuSimulatorTest, GetLoopGsaAddrOffset_InLoopState)
{
    simulator_->SetExecState(CcuExecState::EXEC_LOOP_INSTR);
    LoopGroupInfo loopGroupInfo;
    loopGroupInfo.gsaOffset_ = 100;
    loopGroupInfo.loopStatus_.loopExtendIndex = 2;
    loopGroupInfo.loopStatus_.loopGsaIterStep = 50;
    loopGroupInfo.loopStatus_.curLoopRound = 3;
    simulator_->InitLoopGroupInfo(loopGroupInfo);

    uint64_t offset = simulator_->GetLoopGsaAddrOffset();
    EXPECT_EQ(offset, 200 + 150);
}

TEST_F(CcuSimulatorTest, GetLoopMsOffset_InLoopState)
{
    simulator_->SetExecState(CcuExecState::EXEC_LOOP_INSTR);
    LoopGroupInfo loopGroupInfo;
    loopGroupInfo.msOffset_ = 10;
    loopGroupInfo.loopStatus_.loopExtendIndex = 3;
    simulator_->InitLoopGroupInfo(loopGroupInfo);

    uint16_t offset = simulator_->GetLoopMsOffset();
    EXPECT_EQ(offset, 30);
}

TEST_F(CcuSimulatorTest, GetLoopCKEOffset_InLoopState)
{
    simulator_->SetExecState(CcuExecState::EXEC_LOOP_INSTR);
    LoopGroupInfo loopGroupInfo;
    loopGroupInfo.ckeOffset_ = 5;
    loopGroupInfo.loopStatus_.loopExtendIndex = 4;
    simulator_->InitLoopGroupInfo(loopGroupInfo);

    uint16_t offset = simulator_->GetLoopCKEOffset();
    EXPECT_EQ(offset, 20);
}

TEST_F(CcuSimulatorTest, InitJumpStatus)
{
    simulator_->InitJumpStatus(7);
    EXPECT_EQ(simulator_->GetState(), CcuExecState::EXEC_JUMP_INSTR);
}

TEST_F(CcuSimulatorTest, Init_NotFinished) { simulator_->SetExecState(CcuExecState::EXEC_SUCCESS); }

TEST_F(CcuSimulatorTest, Init_Normal)
{
    simulator_->finished_ = true;
    simulator_->Init(3, 8, 5, RunnerCcuVersion::CCU_V1);
    EXPECT_EQ(simulator_->GetState(), CcuExecState::EXEC_NORMAL_INSTR);
}

TEST_F(CcuSimulatorTest, Init_AlreadyFinished)
{
    simulator_->finished_ = true;
    simulator_->Init(0, 5, 5, RunnerCcuVersion::CCU_V1);
}

TEST_F(CcuSimulatorTest, UpdateLoopStatus_WaitCKE)
{
    simulator_->SetWaitCKEFlag(true);
    bool result = simulator_->UpdateLoopStatus();
    EXPECT_EQ(result, false);
}

TEST_F(CcuSimulatorTest, UpdateLoopStatus_LoopGroupState)
{
    simulator_->SetExecState(CcuExecState::EXEC_LOOPGROUP_INSTR);
    bool result = simulator_->UpdateLoopStatus();
    EXPECT_EQ(result, true);
}

TEST_F(CcuSimulatorTest, UpdateLoopStatus_NormalState)
{
    simulator_->finished_ = true;
    simulator_->Init(0, 5, 5, RunnerCcuVersion::CCU_V1);
    simulator_->SetExecState(CcuExecState::EXEC_NORMAL_INSTR);
    bool result = simulator_->UpdateLoopStatus();
    EXPECT_EQ(result, true);
    EXPECT_EQ(simulator_->GetCurInstrId(), 1);
}

TEST_F(CcuSimulatorTest, UpdateLoopStatus_NormalState_ReachEnd)
{
    simulator_->finished_ = true;
    simulator_->Init(0, 5, 5, RunnerCcuVersion::CCU_V1);
    simulator_->SetExecState(CcuExecState::EXEC_NORMAL_INSTR);
    while (simulator_->GetCurInstrId() < 5) {
        simulator_->UpdateLoopStatus();
    }
    EXPECT_EQ(simulator_->GetState(), CcuExecState::EXEC_SUCCESS);
}

TEST_F(CcuSimulatorTest, UpdateLoopStatus_JumpState)
{
    simulator_->InitJumpStatus(10);
    simulator_->UpdateLoopStatus();
    EXPECT_EQ(simulator_->GetState(), CcuExecState::EXEC_NORMAL_INSTR);
}

TEST_F(CcuSimulatorTest, InitLoopGroupInfo_WithStruct)
{
    LoopGroupInfo loopGroupInfo;
    loopGroupInfo.startLoopId_ = 5;
    loopGroupInfo.loopNum_ = 3;
    loopGroupInfo.loopOffset_ = 2;
    loopGroupInfo.loopExtendNum_ = 1;
    loopGroupInfo.ckeOffset_ = 10;
    loopGroupInfo.msOffset_ = 20;
    loopGroupInfo.gsaOffset_ = 100;

    simulator_->InitLoopGroupInfo(loopGroupInfo);
    EXPECT_EQ(simulator_->GetState(), CcuExecState::EXEC_NORMAL_INSTR);
}

TEST_F(CcuSimulatorTest, GetState)
{
    EXPECT_EQ(simulator_->GetState(), CcuExecState::EXEC_NORMAL_INSTR);

    simulator_->SetExecState(CcuExecState::EXEC_LOOPGROUP_INSTR);
    EXPECT_EQ(simulator_->GetState(), CcuExecState::EXEC_LOOPGROUP_INSTR);

    simulator_->SetExecState(CcuExecState::EXEC_LOOP_INSTR);
    EXPECT_EQ(simulator_->GetState(), CcuExecState::EXEC_LOOP_INSTR);

    simulator_->SetExecState(CcuExecState::EXEC_JUMP_INSTR);
    EXPECT_EQ(simulator_->GetState(), CcuExecState::EXEC_JUMP_INSTR);

    simulator_->SetExecState(CcuExecState::EXEC_SUCCESS);
    EXPECT_EQ(simulator_->GetState(), CcuExecState::EXEC_SUCCESS);

    simulator_->SetExecState(CcuExecState::EXEC_FAIL);
    EXPECT_EQ(simulator_->GetState(), CcuExecState::EXEC_FAIL);
}

TEST_F(CcuSimulatorTest, GetCurInstrId) { EXPECT_EQ(simulator_->GetCurInstrId(), 0); }

TEST_F(CcuSimulatorTest, UpdateLoopStatus_FailState)
{
    simulator_->SetExecState(CcuExecState::EXEC_FAIL);
    bool result = simulator_->UpdateLoopStatus();
    EXPECT_EQ(result, false);
}

TEST_F(CcuSimulatorTest, Init_NotFinished_ReturnsEarly)
{
    simulator_->finished_ = true;
    simulator_->Init(0, 5, 5, RunnerCcuVersion::CCU_V1);
    EXPECT_EQ(simulator_->finished_, false);
    uint16_t prevStart = simulator_->startInstrId_;
    simulator_->Init(2, 8, 6, RunnerCcuVersion::CCU_V1);
    EXPECT_EQ(simulator_->startInstrId_, prevStart);
}

TEST_F(CcuSimulatorTest, UpdateLoopStatus_LoopState)
{
    simulator_->SetExecState(CcuExecState::EXEC_LOOP_INSTR);
    simulator_->finished_ = true;
    simulator_->Init(0, 5, 5, RunnerCcuVersion::CCU_V1);
    simulator_->SetExecState(CcuExecState::EXEC_LOOP_INSTR);
    bool result = simulator_->UpdateLoopStatus();
    EXPECT_EQ(result, true);
    EXPECT_EQ(simulator_->GetCurInstrId(), 1);
}

TEST_F(CcuSimulatorTest, ExecuteInstr_InvalidCurInstrId)
{
    auto& mgr = CcuResourceManager::GetInstance();
    mgr.Init(0, 4, RunnerCcuVersion::CCU_V1, {});
    CcuInstrData instrData;
    instrData.instrCnt = 0;
    instrData.instrData.resize(1);
    memset(&instrData.instrData[0], 0, sizeof(hcomm::CcuRep::CcuInstr));
    mgr.InitInstrInfo(0, 0, instrData);

    bool result = simulator_->ExecuteInstr(10);
    EXPECT_EQ(result, false);
    EXPECT_EQ(simulator_->GetState(), CcuExecState::EXEC_FAIL);
}

TEST_F(CcuSimulatorTest, ExecuteInstr_NullExecutor)
{
    auto& mgr = CcuResourceManager::GetInstance();
    mgr.Init(0, 4, RunnerCcuVersion::CCU_V1, {});
    CcuInstrData instrData;
    instrData.instrCnt = 1;
    instrData.instrData.resize(1);
    memset(&instrData.instrData[0], 0, sizeof(hcomm::CcuRep::CcuInstr));
    instrData.instrData[0].header.header = 0xFFFF;
    mgr.InitInstrInfo(0, 0, instrData);

    auto sim = std::make_unique<CcuSimulator>(0, 0, 0, 2, 2, RunnerCcuVersion::CCU_V1);
    bool result = sim->ExecuteInstr(0);
    EXPECT_EQ(result, true);
}

TEST_F(CcuSimulatorTest, ExecuteLoopGroup_LoopNumZero)
{
    LoopGroupInfo loopGroupInfo;
    loopGroupInfo.loopNum_ = 0;
    simulator_->InitLoopGroupInfo(loopGroupInfo);
    bool result = simulator_->ExecuteLoopGroup();
    EXPECT_EQ(result, true);
}

TEST_F(CcuSimulatorTest, ExecuteLoopGroup_LoopNumTooLarge)
{
    LoopGroupInfo loopGroupInfo;
    loopGroupInfo.loopNum_ = 3;
    simulator_->InitLoopGroupInfo(loopGroupInfo);
    bool result = simulator_->ExecuteLoopGroup();
    EXPECT_EQ(result, false);
}

TEST_F(CcuSimulatorTest, ExecuteLoopGroup_SingleLoop)
{
    auto& mgr = CcuResourceManager::GetInstance();
    mgr.Init(0, 4, RunnerCcuVersion::CCU_V1, {});
    CcuInstrData instrData;
    instrData.instrCnt = 2;
    instrData.instrData.resize(2);
    memset(&instrData.instrData[0], 0, sizeof(hcomm::CcuRep::CcuInstr));
    instrData.instrData[0].header.header = 0xFFFF;
    instrData.instrData[1].header.header = 0xFFFF;
    mgr.InitInstrInfo(0, 0, instrData);

    LoopGroupInfo loopGroupInfo;
    loopGroupInfo.startLoopId_ = 0;
    loopGroupInfo.loopNum_ = 1;
    loopGroupInfo.loopExtendNum_ = 0;
    simulator_->InitLoopGroupInfo(loopGroupInfo);
    bool result = simulator_->ExecuteLoopGroup();
    EXPECT_EQ(result, true);
}

TEST_F(CcuSimulatorTest, Execute_Success)
{
    auto& mgr = CcuResourceManager::GetInstance();
    mgr.Init(0, 4, RunnerCcuVersion::CCU_V1, {});
    CcuInstrData instrData;
    instrData.instrCnt = 1;
    instrData.instrData.resize(1);
    memset(&instrData.instrData[0], 0, sizeof(hcomm::CcuRep::CcuInstr));
    instrData.instrData[0].header.header = 0xFFFF;
    mgr.InitInstrInfo(0, 0, instrData);

    auto sim = std::make_unique<CcuSimulator>(0, 0, 0, 1, 1, RunnerCcuVersion::CCU_V1);
    bool result = sim->Execute();
    EXPECT_EQ(result, true);
}

TEST_F(CcuSimulatorTest, Execute_Fail)
{
    auto& mgr = CcuResourceManager::GetInstance();
    mgr.Init(0, 4, RunnerCcuVersion::CCU_V1, {});
    CcuInstrData instrData;
    instrData.instrCnt = 1;
    instrData.instrData.resize(1);
    memset(&instrData.instrData[0], 0, sizeof(hcomm::CcuRep::CcuInstr));
    mgr.InitInstrInfo(0, 0, instrData);

    auto sim = std::make_unique<CcuSimulator>(0, 0, 0, 1, 1, RunnerCcuVersion::CCU_V1);
    sim->endInstrId_ = 0;
    bool result = sim->Execute();
    EXPECT_EQ(result, true);
}

TEST_F(CcuSimulatorTest, ExecuteLoop_ZeroExecCount)
{
    LoopGroupInfo loopGroupInfo;
    loopGroupInfo.loopStatus_.loopStartInstrId = 0;
    loopGroupInfo.loopStatus_.loopEndInstrId = 0;
    loopGroupInfo.loopStatus_.loopExecCount = 0;
    simulator_->InitLoopGroupInfo(loopGroupInfo);
    bool result = simulator_->ExecuteLoop();
    EXPECT_EQ(result, true);
}
