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
#include <gtest/gtest.h>

#include "ccu_simulator_base.h"

class SimulatorBaseTest : public testing::Test {
protected:
};

TEST_F(SimulatorBaseTest, LoopStatusInfo_DefaultValues)
{
    LoopStatusInfo info;
    EXPECT_EQ(info.loopCurInstrId, 0);
    EXPECT_EQ(info.loopStartInstrId, 0);
    EXPECT_EQ(info.loopEndInstrId, 0);
    EXPECT_EQ(info.loopExecCount, 0);
    EXPECT_EQ(info.curLoopRound, 0);
    EXPECT_EQ(info.loopGsaIterStep, 0);
    EXPECT_EQ(info.loopExtendIndex, 0);
}

TEST_F(SimulatorBaseTest, LoopStatusInfo_SetValues)
{
    LoopStatusInfo info;
    info.loopCurInstrId = 5;
    info.loopStartInstrId = 2;
    info.loopEndInstrId = 10;
    info.loopExecCount = 3;
    info.curLoopRound = 2;
    info.loopGsaIterStep = 100;
    info.loopExtendIndex = 4;

    EXPECT_EQ(info.loopCurInstrId, 5);
    EXPECT_EQ(info.loopStartInstrId, 2);
    EXPECT_EQ(info.loopEndInstrId, 10);
    EXPECT_EQ(info.loopExecCount, 3);
    EXPECT_EQ(info.curLoopRound, 2);
    EXPECT_EQ(info.loopGsaIterStep, 100);
    EXPECT_EQ(info.loopExtendIndex, 4);
}

TEST_F(SimulatorBaseTest, LoopGroupInfo_DefaultValues)
{
    LoopGroupInfo info;
    EXPECT_EQ(info.startLoopId_, 0);
    EXPECT_EQ(info.loopNum_, 0);
    EXPECT_EQ(info.loopOffset_, 0);
    EXPECT_EQ(info.loopExtendNum_, 0);
    EXPECT_EQ(info.gsaOffset_, 0);
    EXPECT_EQ(info.msOffset_, 0);
    EXPECT_EQ(info.ckeOffset_, 0);
}

TEST_F(SimulatorBaseTest, LoopGroupInfo_SetValues)
{
    LoopGroupInfo info;
    info.startLoopId_ = 5;
    info.loopNum_ = 3;
    info.loopOffset_ = 2;
    info.loopExtendNum_ = 1;
    info.gsaOffset_ = 0x1000;
    info.msOffset_ = 100;
    info.ckeOffset_ = 50;

    EXPECT_EQ(info.startLoopId_, 5);
    EXPECT_EQ(info.loopNum_, 3);
    EXPECT_EQ(info.loopOffset_, 2);
    EXPECT_EQ(info.loopExtendNum_, 1);
    EXPECT_EQ(info.gsaOffset_, 0x1000);
    EXPECT_EQ(info.msOffset_, 100);
    EXPECT_EQ(info.ckeOffset_, 50);
}

TEST_F(SimulatorBaseTest, ExecuteStatusInfo_DefaultValues)
{
    ExecuteStatusInfo info;
    EXPECT_EQ(info.state, CcuExecState::EXEC_NORMAL_INSTR);
    EXPECT_EQ(info.curInstrId, 0);
    EXPECT_EQ(info.startInstrId, 0);
    EXPECT_EQ(info.endInstrId, 0);
    EXPECT_EQ(info.jumpInstrId, 0);
    EXPECT_EQ(info.instrType, 0);
    EXPECT_EQ(info.waitCKE, false);
    EXPECT_EQ(info.initialized, false);
}

TEST_F(SimulatorBaseTest, ExecuteStatusInfo_SetValues)
{
    ExecuteStatusInfo info;
    info.state = CcuExecState::EXEC_LOOP_INSTR;
    info.curInstrId = 10;
    info.startInstrId = 5;
    info.endInstrId = 20;
    info.jumpInstrId = 15;
    info.instrType = 1;
    info.waitCKE = true;
    info.initialized = true;

    EXPECT_EQ(info.state, CcuExecState::EXEC_LOOP_INSTR);
    EXPECT_EQ(info.curInstrId, 10);
    EXPECT_EQ(info.startInstrId, 5);
    EXPECT_EQ(info.endInstrId, 20);
    EXPECT_EQ(info.jumpInstrId, 15);
    EXPECT_EQ(info.instrType, 1);
    EXPECT_EQ(info.waitCKE, true);
    EXPECT_EQ(info.initialized, true);
}

TEST_F(SimulatorBaseTest, CcuExecState_AllValues)
{
    EXPECT_EQ(static_cast<int>(CcuExecState::EXEC_NORMAL_INSTR), 0);
    EXPECT_EQ(static_cast<int>(CcuExecState::EXEC_LOOPGROUP_INSTR), 1);
    EXPECT_EQ(static_cast<int>(CcuExecState::EXEC_LOOP_INSTR), 2);
    EXPECT_EQ(static_cast<int>(CcuExecState::EXEC_JUMP_INSTR), 3);
    EXPECT_EQ(static_cast<int>(CcuExecState::EXEC_SUCCESS), 4);
    EXPECT_EQ(static_cast<int>(CcuExecState::EXEC_FAIL), 5);
}

TEST_F(SimulatorBaseTest, ReduceAddDataType_AllValues)
{
    EXPECT_EQ(static_cast<int>(ReduceAddDataType::ADD_FP32), 0);
    EXPECT_EQ(static_cast<int>(ReduceAddDataType::ADD_FP16), 1);
    EXPECT_EQ(static_cast<int>(ReduceAddDataType::ADD_BF16), 2);
    EXPECT_EQ(static_cast<int>(ReduceAddDataType::ADD_HIF8), 3);
    EXPECT_EQ(static_cast<int>(ReduceAddDataType::ADD_FP8_E4M3), 4);
    EXPECT_EQ(static_cast<int>(ReduceAddDataType::ADD_FP8_E5M2), 5);
    EXPECT_EQ(static_cast<int>(ReduceAddDataType::ADD_INT8), 6);
    EXPECT_EQ(static_cast<int>(ReduceAddDataType::ADD_UINT8), 7);
    EXPECT_EQ(static_cast<int>(ReduceAddDataType::ADD_INT16), 8);
    EXPECT_EQ(static_cast<int>(ReduceAddDataType::ADD_INT32), 9);
    EXPECT_EQ(static_cast<int>(ReduceAddDataType::ADD_RESERVED), 10);
}

TEST_F(SimulatorBaseTest, ReduceMaxMinDataType_AllValues)
{
    EXPECT_EQ(static_cast<int>(ReduceMaxMinDataType::MAX_MIN_FP32), 0);
    EXPECT_EQ(static_cast<int>(ReduceMaxMinDataType::MAX_MIN_FP16), 1);
    EXPECT_EQ(static_cast<int>(ReduceMaxMinDataType::MAX_MIN_BF16), 2);
    EXPECT_EQ(static_cast<int>(ReduceMaxMinDataType::MAX_MIN_RESERVED1), 3);
    EXPECT_EQ(static_cast<int>(ReduceMaxMinDataType::MAX_MIN_RESERVED2), 4);
    EXPECT_EQ(static_cast<int>(ReduceMaxMinDataType::MAX_MIN_RESERVED3), 5);
    EXPECT_EQ(static_cast<int>(ReduceMaxMinDataType::MAX_MIN_INT8), 6);
    EXPECT_EQ(static_cast<int>(ReduceMaxMinDataType::MAX_MIN_UINT8), 7);
    EXPECT_EQ(static_cast<int>(ReduceMaxMinDataType::MAX_MIN_INT16), 8);
    EXPECT_EQ(static_cast<int>(ReduceMaxMinDataType::MAX_MIN_INT32), 9);
    EXPECT_EQ(static_cast<int>(ReduceMaxMinDataType::MAX_MIN_RESERVED4), 10);
}

TEST_F(SimulatorBaseTest, LoopGroupInfo_WithLoopStatus)
{
    LoopGroupInfo info;
    info.startLoopId_ = 10;
    info.loopNum_ = 5;
    info.loopOffset_ = 3;
    info.loopExtendNum_ = 2;
    info.gsaOffset_ = 0x2000;
    info.msOffset_ = 200;
    info.ckeOffset_ = 100;

    info.loopStatus_.loopCurInstrId = 12;
    info.loopStatus_.loopStartInstrId = 10;
    info.loopStatus_.loopEndInstrId = 15;
    info.loopStatus_.loopExecCount = 5;
    info.loopStatus_.curLoopRound = 2;
    info.loopStatus_.loopGsaIterStep = 256;
    info.loopStatus_.loopExtendIndex = 1;

    EXPECT_EQ(info.loopStatus_.loopStartInstrId, 10);
    EXPECT_EQ(info.loopStatus_.loopCurInstrId, 12);
    EXPECT_EQ(info.loopStatus_.loopEndInstrId, 15);
    EXPECT_EQ(info.loopStatus_.loopExecCount, 5);
    EXPECT_EQ(info.loopStatus_.curLoopRound, 2);
    EXPECT_EQ(info.loopStatus_.loopGsaIterStep, 256);
    EXPECT_EQ(info.loopStatus_.loopExtendIndex, 1);
}

TEST_F(SimulatorBaseTest, ExecuteStatusInfo_WithLoopGroupState)
{
    ExecuteStatusInfo info;
    info.state = CcuExecState::EXEC_LOOPGROUP_INSTR;
    info.curInstrId = 5;
    info.startInstrId = 0;
    info.endInstrId = 100;
    info.jumpInstrId = 50;
    info.instrType = 2;
    info.waitCKE = true;
    info.initialized = true;

    info.loopGroupState.startLoopId_ = 10;
    info.loopGroupState.loopNum_ = 4;
    info.loopGroupState.loopOffset_ = 2;
    info.loopGroupState.loopExtendNum_ = 1;
    info.loopGroupState.gsaOffset_ = 0x1000;
    info.loopGroupState.msOffset_ = 50;
    info.loopGroupState.ckeOffset_ = 25;

    EXPECT_EQ(info.loopGroupState.startLoopId_, 10);
    EXPECT_EQ(info.loopGroupState.loopNum_, 4);
    EXPECT_EQ(info.loopGroupState.loopOffset_, 2);
    EXPECT_EQ(info.loopGroupState.loopExtendNum_, 1);
}
