/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "wait_executor.h"

#include <cstdint>

#include "ccu_executor_manager.h"
#include "ccu_string_util.h"

using namespace std;
using namespace hcomm::CcuRep;

// 注册NopExecutor create Func
REG_CCU_EXECUTOR_CREATE_FUNC_V2(SimCcuV2::CTRL_TYPE, SimCcuV2::WAIT_CODE, WaitExecutor);

void WaitExecutor::Parser()
{
    ValidateVersionExclusive(RunnerCcuVersion::CCU_V2, "WaitExecutor");
    expectedXnId_ = instr_.v2.wait.expectedXnId;
    conditionXnId_ = instr_.v2.wait.conditionXnId;
    conditionType_ = instr_.v2.wait.conditionType;
}

void WaitExecutor::Run()
{
    auto& ccuResMgr = CcuResourceManager::GetInstance();
    uint16_t expectedXnId = GetXnId(expectedXnId_);
    uint16_t conditionXnId = GetXnId(conditionXnId_);
    uint64_t expectedValue = ccuResMgr.GetXnValue(rankId_, dieId_, expectedXnId);
    uint64_t conditionValue = ccuResMgr.GetXnValue(rankId_, dieId_, conditionXnId);
    uint64_t nextInsIdx = 0;
    auto curInstrId = ccuSimulator_->GetCurInstrId();
    auto updateNextInsIdx = [curInstrId, &nextInsIdx](bool notWait) -> void {
        if (notWait) {
            nextInsIdx = curInstrId + 1;
        } else {
            nextInsIdx = curInstrId;
        }
    };

    switch (conditionType_) {
        case 0: // 等于
            updateNextInsIdx(conditionValue == expectedValue);
            HCCL_VM_INFO(
                "When conditionXn{}{} equal to expectData{}{}, WaitFlag{} ", conditionXnId, conditionValue,
                expectedXnId, expectedValue, conditionValue == expectedValue);
            break;
        case 1: // 不等于
            updateNextInsIdx(conditionValue != expectedValue);
            HCCL_VM_INFO(
                "When conditionXn{}{} not equal to expectData{}{}, WaitFlag{} ", conditionXnId, conditionValue,
                expectedXnId, expectedValue, conditionValue != expectedValue);
            break;
        case 2: // 大于
            updateNextInsIdx(conditionValue > expectedValue);
            HCCL_VM_INFO(
                "When conditionXn{}{} greater than expectData{}{}, WaitFlag{} ", conditionXnId, conditionValue,
                expectedXnId, expectedValue, conditionValue > expectedValue);
            break;
        case 3: // 大于/等于
            updateNextInsIdx(conditionValue >= expectedValue);
            HCCL_VM_INFO(
                "When conditionXn{}{} greater than or equal to expectData{}{}, WaitFlag{} ", conditionXnId,
                conditionValue, expectedXnId, expectedValue, conditionValue >= expectedValue);
            break;
        case 4: // 小于
            updateNextInsIdx(conditionValue < expectedValue);
            HCCL_VM_INFO(
                "When conditionXn{}{} less than expectData{}{}, WaitFlag{} ", conditionXnId, conditionValue,
                expectedXnId, expectedValue, conditionValue < expectedValue);
            break;
        case 5: // 小于/等于
            updateNextInsIdx(conditionValue <= expectedValue);
            HCCL_VM_INFO(
                "When conditionXn{}{} less than or equal to expectData{}{}, WaitFlag{} ", conditionXnId, conditionValue,
                expectedXnId, expectedValue, conditionValue <= expectedValue);
            break;
        default: // 参考等于
            updateNextInsIdx(conditionValue == expectedValue);
            HCCL_VM_INFO(
                "When conditionXn{}{} equal to expectData{}{}, WaitFlag{} ", conditionXnId, conditionValue,
                expectedXnId, expectedValue, conditionValue == expectedValue);
            break;
    }
    ccuSimulator_->InitJumpStatus(nextInsIdx - ccuSimulator_->GetStartInstrId());
}

std::string WaitExecutor::Describe() { return HcclSim::StringFormat("[WaitExecutor]\n"); }

CcuTrace::CcuInstrTraceDetail WaitExecutor::CollectTraceDetail()
{
    CcuTrace::CcuInstrTraceDetail detail;
    detail.typeName = "Wait";
    auto& ccuResMgr = CcuResourceManager::GetInstance();
    detail.args["expectedValue"] = std::to_string(ccuResMgr.GetXnValue(rankId_, dieId_, GetXnId(expectedXnId_)));
    detail.args["conditionValue"] = std::to_string(ccuResMgr.GetXnValue(rankId_, dieId_, GetXnId(conditionXnId_)));
    detail.args["conditionType"] = std::to_string(conditionType_);
    return detail;
}
