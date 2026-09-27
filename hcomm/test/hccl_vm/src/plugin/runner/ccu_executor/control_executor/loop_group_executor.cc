/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "loop_group_executor.h"

#include <cstdint>

#include "ccu_executor_manager.h"
#include "ccu_string_util.h"

using namespace std;
using namespace hcomm::CcuRep;

// 注册SyncXnExecutor create Func
REG_CCU_EXECUTOR_CREATE_FUNC(SimCcuV1::CTRL_TYPE, SimCcuV1::LOOPGROUP_CODE, LoopGroupExecutor);
REG_CCU_EXECUTOR_CREATE_FUNC_V2(SimCcuV2::CTRL_TYPE, SimCcuV2::LOOPGROUP_CODE, LoopGroupExecutor);

void LoopGroupExecutor::Parser()
{
    if (version_ == RunnerCcuVersion::CCU_V1) {
        startLoopInstrId_ = instr_.v1.loopGroup.startLoopInstrId;
        xnId_ = instr_.v1.loopGroup.xnId;
        xmId_ = instr_.v1.loopGroup.xmId;
        highPerfModeEn_ = instr_.v1.loopGroup.highPerfModeEn;
    } else if (version_ == RunnerCcuVersion::CCU_V2) {
        startLoopInstrId_ = instr_.v2.loopGroup.startLoopInstrId;
        xnId_ = instr_.v2.loopGroup.xnId & 0x7FFF;
        xmId_ = instr_.v2.loopGroup.xmId & 0x7FFF;
        xpId_ = instr_.v2.loopGroup.xpId & 0x7FFF;
    } else {
        HCCL_VM_ERROR("Invalid ccu version:{}", RunnerCcuVersionToString(version_));
        ccuSimulator_->SetExecState(CcuExecState::EXEC_FAIL);
        return;
    }
}

void LoopGroupExecutor::RunV1()
{
    auto& ccuResMgr = CcuResourceManager::GetInstance();
    uint64_t offsetCfg = ccuResMgr.GetXnValue(rankId_, dieId_, xmId_);
    uint64_t repeatCfg = ccuResMgr.GetXnValue(rankId_, dieId_, xnId_);
    ccuSimulator_->InitLoopGroupInfo(startLoopInstrId_, offsetCfg, repeatCfg);

    // 执行loop指令
    ccuSimulator_->ExecuteLoopGroup();
    ccuSimulator_->SetExecState(CcuExecState::EXEC_NORMAL_INSTR);
}

void LoopGroupExecutor::RunV2()
{
    auto& ccuResMgr = CcuResourceManager::GetInstance();
    uint64_t xnValue = ccuResMgr.GetXnValue(rankId_, dieId_, xnId_);
    uint64_t xmValue = ccuResMgr.GetXnValue(rankId_, dieId_, xmId_);
    uint64_t xpValue = ccuResMgr.GetXnValue(rankId_, dieId_, xpId_);

    ccuSimulator_->InitLoopGroupInfoV2(startLoopInstrId_, xnValue, xmValue, xpValue);

    // 执行loop指令
    ccuSimulator_->ExecuteLoopGroup();
    ccuSimulator_->SetExecState(CcuExecState::EXEC_NORMAL_INSTR);
}

void LoopGroupExecutor::Run()
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

std::string LoopGroupExecutor::Describe()
{
    return HcclSim::StringFormat(
        "[Simulation Execute] LoopGroup From startLoopInstrId[%u] with loopGroupXn[%u], "
        "offsetXn[%u] and highPerfModeEn[%u]\n",
        startLoopInstrId_, xnId_, xmId_, highPerfModeEn_);
}

CcuTrace::CcuInstrTraceDetail LoopGroupExecutor::CollectTraceDetail()
{
    CcuTrace::CcuInstrTraceDetail detail;
    detail.typeName = "LoopGroup";
    auto& ccuResMgr = CcuResourceManager::GetInstance();
    if (version_ == RunnerCcuVersion::CCU_V1) {
        detail.args["offsetCfg"] = std::to_string(ccuResMgr.GetXnValue(rankId_, dieId_, xmId_));
        detail.args["repeatCfg"] = std::to_string(ccuResMgr.GetXnValue(rankId_, dieId_, xnId_));
    } else {
        detail.args["xnValue"] = std::to_string(ccuResMgr.GetXnValue(rankId_, dieId_, GetXnId(xnId_)));
        detail.args["xmValue"] = std::to_string(ccuResMgr.GetXnValue(rankId_, dieId_, GetXnId(xmId_)));
        detail.args["xpValue"] = std::to_string(ccuResMgr.GetXnValue(rankId_, dieId_, GetXnId(xpId_)));
    }
    return detail;
}
