/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "sync_xn_executor.h"

#include "ccu_executor_manager.h"
#include "ccu_microcode_common_v1.h"
#include "sim_log.h"
#include "ccu_string_util.h"

using namespace std;
using namespace hcomm::CcuRep;

// 注册SyncXnExecutor create Func
REG_CCU_EXECUTOR_CREATE_FUNC(SimCcuV1::TRANS_TYPE, SimCcuV1::SYNCXN_CODE, SyncXnExecutor);

void SyncXnExecutor::Parser()
{
    ValidateVersionExclusive(RunnerCcuVersion::CCU_V1, "SyncXnExecutor");
    rmtXnId_ = instr_.v1.syncXn.rmtXnId;
    locXnId_ = instr_.v1.syncXn.locXnId;
    channelId_ = instr_.v1.syncXn.channelId;
    setRmtCKEId_ = instr_.v1.syncXn.setRmtCKEId;
    setRmtCKEMask_ = instr_.v1.syncXn.setRmtCKEMask;
    clearType_ = instr_.v1.syncXn.clearType;
    setCKEId_ = instr_.v1.syncXn.setCKEId;
    setCKEMask_ = instr_.v1.syncXn.setCKEMask;
    waitCKEId_ = instr_.v1.syncXn.waitCKEId;
    waitCKEMask_ = instr_.v1.syncXn.waitCKEMask;
}

void SyncXnExecutor::Process(CcuResourceManager& ccuResMgr)
{
    // 根据channel id获取remote rank id
    auto rmtCcu = ccuResMgr.GetRmtCcu(rankId_, dieId_, channelId_);
    // 将本端的xn内容写到目的端的xn中
    auto locXnValue = ccuResMgr.GetXnValue(rankId_, dieId_, locXnId_);
    HCCL_VM_INFO(
        "sync xn, rankId{}, dieId{}, xnId{}, channelId{}, locXnValue: {}", rmtCcu.first, rmtCcu.second, rmtXnId_,
        channelId_, locXnValue);
    ccuResMgr.UpdateXnValue(rmtCcu.first, rmtCcu.second, rmtXnId_, locXnValue);

    // 设置目的端的cke
    SetRmtCKESignal(ccuResMgr, rmtCcu.first, rmtCcu.second, setRmtCKEId_, setRmtCKEMask_);

    // 设置本端的cke
    SetCkeSignal(ccuResMgr, setCKEId_, setCKEMask_);
}

void SyncXnExecutor::Run() { WaitCkeProcess(waitCKEId_, waitCKEMask_, clearType_, "SyncXn"); }

std::string SyncXnExecutor::Describe()
{
    return HcclSim::StringFormat(
        "[Simulation Execute] Wait CKE[%u:%04x], Sync locXnId[%u] To rmtXnId[%u] Use "
        "Channel[%u], Set rmtCKE[%u:%04x], Set "
        "CKE[%u:%04x], clearType[%u]\n",
        waitCKEId_, waitCKEMask_, locXnId_, rmtXnId_, channelId_, setRmtCKEId_, setRmtCKEMask_, setCKEId_, setCKEMask_,
        clearType_);
}

CcuTrace::CcuInstrTraceDetail SyncXnExecutor::CollectTraceDetail()
{
    CcuTrace::CcuInstrTraceDetail detail;
    detail.typeName = "SyncXn";
    return detail;
}
