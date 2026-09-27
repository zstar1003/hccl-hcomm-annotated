/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "loop_executor.h"

#include <cstdint>

#include "ccu_executor_manager.h"
#include "ccu_string_util.h"

using namespace std;
using namespace hcomm::CcuRep;

// 注册SyncXnExecutor create Func
REG_CCU_EXECUTOR_CREATE_FUNC(SimCcuV1::CTRL_TYPE, SimCcuV1::LOOP_CODE, LoopExecutor);
REG_CCU_EXECUTOR_CREATE_FUNC_V2(SimCcuV2::CTRL_TYPE, SimCcuV2::LOOP_CODE, LoopExecutor);

void LoopExecutor::Parser()
{
    if (version_ == RunnerCcuVersion::CCU_V1) {
        startInstrId_ = instr_.v1.loop.startInstrId;
        endInstrId_ = instr_.v1.loop.endInstrId;
        xnId_ = instr_.v1.loop.xnId;
    } else if (version_ == RunnerCcuVersion::CCU_V2) {
        startInstrId_ = instr_.v2.loop.startInstrId;
        endInstrId_ = instr_.v2.loop.endInstrId;
        xnId_ = instr_.v2.loop.xnId;
        xmId_ = instr_.v2.loop.xmId;
        xpId_ = instr_.v2.loop.xpId;
    } else {
        HCCL_VM_ERROR("Invalid ccu version:{}", RunnerCcuVersionToString(version_));
        ccuSimulator_->SetExecState(CcuExecState::EXEC_FAIL);
        return;
    }
}

void LoopExecutor::RunV1()
{
    uint64_t xnValue = CcuResourceManager::GetInstance().GetXnValue(rankId_, dieId_, xnId_);
    uint16_t loopCnt = xnValue & 0x1FFF; // 0x1FFF: 取低13位
    // [搬运任务]用到的GSA地址的偏移步长
    uint32_t gsaAddrStep = (xnValue >> 13) & 0xFFFFFFFF; // 13: 右移13位 0xFFFFFFFF: 取低32位

    ccuSimulator_->InitLoopInfo(startInstrId_, endInstrId_, loopCnt, gsaAddrStep);
    ccuSimulator_->ExecuteLoop();
}

void LoopExecutor::RunV2()
{
    uint16_t xnId = GetXnId(xnId_);
    uint16_t xmId = GetXnId(xmId_);
    uint16_t xpId = GetXnId(xpId_);
    auto& ccuResMgr = CcuResourceManager::GetInstance();
    uint64_t xnValue = ccuResMgr.GetXnValue(rankId_, dieId_, xnId);
    uint64_t xmValue = ccuResMgr.GetXnValue(rankId_, dieId_, xmId);
    uint64_t xpValue = ccuResMgr.GetXnValue(rankId_, dieId_, xpId);

    ccuSimulator_->InitLoopInfoV2(startInstrId_, endInstrId_, xnValue, xmValue, xpValue);
    ccuSimulator_->ExecuteLoop();
}

void LoopExecutor::Run()
{
    if (version_ == RunnerCcuVersion::CCU_V1) {
        RunV1();
        return;
    } else if (version_ == RunnerCcuVersion::CCU_V2) {
        RunV2();
        return;
    } else {
        HCCL_VM_ERROR("Invalid ccu version:{}", RunnerCcuVersionToString(version_));
        ccuSimulator_->SetExecState(CcuExecState::EXEC_FAIL);
        return;
    }
}

std::string LoopExecutor::Describe()
{
    return HcclSim::StringFormat(
        "[Simulation Execute] Loop From startInstrId[%u] to endInstrId[%u] with loopXn[%u]\n", startInstrId_,
        endInstrId_, xnId_);
}

CcuTrace::CcuInstrTraceDetail LoopExecutor::CollectTraceDetail()
{
    CcuTrace::CcuInstrTraceDetail detail;
    detail.typeName = "Loop";
    auto& ccuResMgr = CcuResourceManager::GetInstance();
    if (version_ == RunnerCcuVersion::CCU_V1) {
        uint64_t xnValue = ccuResMgr.GetXnValue(rankId_, dieId_, xnId_);
        detail.args["xnValue"] = std::to_string(xnValue);
        detail.args["loopCnt"] = std::to_string(xnValue & 0x1FFF);
        detail.args["gsaAddrStep"] = std::to_string((xnValue >> 13) & 0xFFFFFFFF);
    } else {
        uint16_t xnId = GetXnId(xnId_);
        uint16_t xmId = GetXnId(xmId_);
        uint16_t xpId = GetXnId(xpId_);
        detail.args["xnValue"] = std::to_string(ccuResMgr.GetXnValue(rankId_, dieId_, xnId));
        detail.args["xmValue"] = std::to_string(ccuResMgr.GetXnValue(rankId_, dieId_, xmId));
        detail.args["xpValue"] = std::to_string(ccuResMgr.GetXnValue(rankId_, dieId_, xpId));
    }
    return detail;
}
