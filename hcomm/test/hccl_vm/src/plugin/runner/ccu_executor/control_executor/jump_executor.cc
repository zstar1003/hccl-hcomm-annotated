/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "jump_executor.h"

#include <cstdint>

#include "ccu_executor_manager.h"
#include "ccu_string_util.h"

using namespace std;
using namespace hcomm::CcuRep;
constexpr uint32_t JUMP_INSTR_MAX = 0x10000;
// 注册JumpExecutor create Func
REG_CCU_EXECUTOR_CREATE_FUNC(SimCcuV1::CTRL_TYPE, SimCcuV1::JMP_CODE, JumpExecutor);
REG_CCU_EXECUTOR_CREATE_FUNC_V2(SimCcuV2::CTRL_TYPE, SimCcuV2::JMP_CODE, JumpExecutor);

void JumpExecutor::Parser()
{
    if (version_ == RunnerCcuVersion::CCU_V1) {
        dstInstrXnId_ = instr_.v1.jmp.dstInstrXnId;
        conditionXnId_ = instr_.v1.jmp.conditionXnId;
        expectData_ = instr_.v1.jmp.expectData;
#ifdef BUILD_A6_CCU_INSTR
    } else if (version_ == RunnerCcuVersion::CCU_V2) {
        relTarInstrXnId_ = instr_.v2.jmp.relTarInstrXnId;
        conditionXnId_ = instr_.v2.jmp.conditionXnId;
        expectedXnId_ = instr_.v2.jmp.expectedXnId;
        conditionType_ = instr_.v2.jmp.conditionType;
        jumpMode_ = instr_.v2.jmp.jumpMode;
#endif
    } else {
        HCCL_VM_ERROR("Invalid ccu version:{}", RunnerCcuVersionToString(version_));
        ccuSimulator_->SetExecState(CcuExecState::EXEC_FAIL);
        return;
    }
}

void JumpExecutor::RunV1()
{
    auto& ccuResMgr = CcuResourceManager::GetInstance();
    uint64_t instrId = ccuResMgr.GetXnValue(rankId_, dieId_, dstInstrXnId_);
    uint64_t value = ccuResMgr.GetXnValue(rankId_, dieId_, conditionXnId_);

    if (value != expectData_) {
        ccuSimulator_->InitJumpStatus(instrId);
    }
    return;
}

void JumpExecutor::RunV2()
{
    auto& ccuResMgr = CcuResourceManager::GetInstance();
    uint16_t expectedXnId = GetXnId(expectedXnId_);
    uint16_t conditionXnId = GetXnId(conditionXnId_);
    uint16_t relTarInstrXnId = GetXnId(relTarInstrXnId_);

    uint64_t expectedValue = ccuResMgr.GetXnValue(rankId_, dieId_, expectedXnId);
    uint64_t conditionValue = ccuResMgr.GetXnValue(rankId_, dieId_, conditionXnId);
    uint64_t relTarInstrValue = ccuResMgr.GetXnValue(rankId_, dieId_, relTarInstrXnId);
    uint64_t nextInsIdx = 0;
    auto curInstrId = ccuSimulator_->GetCurInstrId();
    auto updateNextInsIdx = [this, curInstrId, relTarInstrValue, &nextInsIdx, &ccuResMgr](bool isJump) -> void {
        if (isJump) {
            if (jumpMode_ == 0) {
                // 相对跳转
                nextInsIdx = curInstrId + relTarInstrValue;
                if (nextInsIdx >= JUMP_INSTR_MAX) {
                    nextInsIdx -= JUMP_INSTR_MAX;
                }
            } else if (jumpMode_ == 1) {
                // 绝对跳转
                nextInsIdx = relTarInstrValue;
            }
            ccuSimulator_->InitJumpStatus(nextInsIdx);
        }
    };

    switch (conditionType_) {
        case 0: // 等于
            updateNextInsIdx(conditionValue == expectedValue);
            HCCL_VM_INFO(
                "When conditionX{}{} equal to expectData{}{}, Jump Mode{},Jump instruction offset{}{}", conditionXnId,
                conditionValue, expectedXnId, expectedValue, jumpMode_, relTarInstrXnId, relTarInstrValue);
            break;
        case 1: // 不等于
            updateNextInsIdx(conditionValue != expectedValue);
            HCCL_VM_INFO(
                "When conditionX{}{} not equal to expectData{}{}, Jump Mode{}, Jump instruction offset{}{}",
                conditionXnId, conditionValue, expectedXnId, expectedValue, jumpMode_, relTarInstrXnId,
                relTarInstrValue);
            break;
        case 2: // 大于
            updateNextInsIdx(conditionValue > expectedValue);
            HCCL_VM_INFO(
                "When conditionX{}{} greater than expectData{}{}, Jump Mode{}, Jump instruction offset{}{}",
                conditionXnId, conditionValue, expectedXnId, expectedValue, jumpMode_, relTarInstrXnId,
                relTarInstrValue);
            break;
        case 3: // 大于/等于
            updateNextInsIdx(conditionValue >= expectedValue);
            HCCL_VM_INFO(
                "When conditionX{}{} greater than or equal to expectData{}{}, Jump Mode{}, Jump instruction offset{}{}",
                conditionXnId, conditionValue, expectedXnId, expectedValue, jumpMode_, relTarInstrXnId,
                relTarInstrValue);
            break;
        case 4: // 小于
            updateNextInsIdx(conditionValue < expectedValue);
            HCCL_VM_INFO(
                "When conditionX{}{} less than expectData{}{}, Jump Mode{}, Jump instruction offset{}{}", conditionXnId,
                conditionValue, expectedXnId, expectedValue, jumpMode_, relTarInstrXnId, relTarInstrValue);
            break;
        case 5: // 小于/等于
            updateNextInsIdx(conditionValue <= expectedValue);
            HCCL_VM_INFO(
                "When conditionX{}{} less than or equal to expectData{}{}, Jump Mode{}, Jump instruction offset{}{}",
                conditionXnId, conditionValue, expectedXnId, expectedValue, jumpMode_, relTarInstrXnId,
                relTarInstrValue);
            break;
        default: // 无条件跳转
            updateNextInsIdx(true);
            HCCL_VM_INFO(
                "When conditionX{}{} is true, Jump Mode{}, Jump instruction offset{}{}", conditionXnId, conditionValue,
                jumpMode_, relTarInstrXnId, relTarInstrValue);
            break;
    }
}

void JumpExecutor::Run()
{
    if (version_ == RunnerCcuVersion::CCU_V1) {
        RunV1();
    } else if (version_ == RunnerCcuVersion::CCU_V2) {
        RunV2();
    } else {
        HCCL_VM_ERROR("Invalid ccu version:{}", RunnerCcuVersionToString(version_));
        ccuSimulator_->SetExecState(CcuExecState::EXEC_FAIL);
        return;
    }
}

std::string JumpExecutor::Describe()
{
    return HcclSim::StringFormat(
        "[Simulation Execute] When conditionXn[%u] not equal to expectData[%lu], Jump To InstrIdXn[%u]\n",
        conditionXnId_, expectData_, dstInstrXnId_);
}

CcuTrace::CcuInstrTraceDetail JumpExecutor::CollectTraceDetail()
{
    CcuTrace::CcuInstrTraceDetail detail;
    detail.typeName = "Jump";
    auto& ccuResMgr = CcuResourceManager::GetInstance();
    if (version_ == RunnerCcuVersion::CCU_V1) {
        detail.args["targetInstrId"] = std::to_string(ccuResMgr.GetXnValue(rankId_, dieId_, dstInstrXnId_));
        detail.args["conditionValue"] = std::to_string(ccuResMgr.GetXnValue(rankId_, dieId_, conditionXnId_));
    } else {
        detail.args["expectedValue"] = std::to_string(ccuResMgr.GetXnValue(rankId_, dieId_, GetXnId(expectedXnId_)));
        detail.args["conditionValue"] = std::to_string(ccuResMgr.GetXnValue(rankId_, dieId_, GetXnId(conditionXnId_)));
        detail.args["relTarInstrValue"]
            = std::to_string(ccuResMgr.GetXnValue(rankId_, dieId_, GetXnId(relTarInstrXnId_)));
    }
    return detail;
}
