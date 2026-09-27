/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "sync_cke_executor.h"

#include <cstdint>

#include "ccu_executor_manager.h"
#include "ccu_microcode_common_v1.h"
#include "sim_log.h"
#include "ccu_string_util.h"

using namespace std;
using namespace hcomm::CcuRep;

// 注册SyncCkeExecutor create Func
REG_CCU_EXECUTOR_CREATE_FUNC(SimCcuV1::TRANS_TYPE, SimCcuV1::SYNCCKE_CODE, SyncCkeExecutor);

void SyncCkeExecutor::Parser()
{
    ValidateVersionExclusive(RunnerCcuVersion::CCU_V1, "SyncCkeExecutor");
    rmtCKEId_ = instr_.v1.syncCKE.rmtCKEId;
    locCKEId_ = instr_.v1.syncCKE.locCKEId;
    locCKEMask_ = instr_.v1.syncCKE.locCKEMask;
    channelId_ = instr_.v1.syncCKE.channelId;
    clearType_ = instr_.v1.syncCKE.clearType;
    setCKEId_ = instr_.v1.syncCKE.setCKEId;
    setCKEMask_ = instr_.v1.syncCKE.setCKEMask;
    waitCKEId_ = instr_.v1.syncCKE.waitCKEId;
    waitCKEMask_ = instr_.v1.syncCKE.waitCKEMask;
}

void SyncCkeExecutor::Process(CcuResourceManager& ccuResMgr)
{
    // 根据channel id获取remote rank id
    auto rmtCcu = ccuResMgr.GetRmtCcu(rankId_, dieId_, channelId_);
    // 将本端的CKE内容写到目的端的CKE中
    auto locCKE = ccuResMgr.GetCkeValue(rankId_, dieId_, locCKEId_);
    uint16_t rmtCke = ccuResMgr.GetCkeValue(rmtCcu.first, rmtCcu.second, rmtCKEId_);
    uint16_t newRmtCke = rmtCke | (locCKE & locCKEMask_);
    ccuResMgr.UpdateCkeValue(rmtCcu.first, rmtCcu.second, rmtCKEId_, newRmtCke);

    HCCL_VM_DEBUG(
        "locCcu[{}:{}], channelId=[{}], "
        "rmtDevice[{}:{}], locCKE=[{}:{:04x}], "
        "value=[{}], rmtCKE=[{}:{:04x}], value=[{} --> {}].",
        rankId_, dieId_, channelId_, rmtCcu.first, rmtCcu.second, locCKEId_, locCKEMask_, locCKE, rmtCKEId_,
        locCKEMask_, rmtCke, newRmtCke);

    // 设置本端的cke
    SetCkeSignal(ccuResMgr, setCKEId_, setCKEMask_);
}

void SyncCkeExecutor::Run() { WaitCkeProcess(waitCKEId_, waitCKEMask_, clearType_, "SyncCKE"); }

std::string SyncCkeExecutor::Describe()
{
    return HcclSim::StringFormat(
        "[Simulation Execute] Wait CKE[%u:%04x], Sync LocCKE[%u:%04x] To rmtCKE[%u:%04x] Use Channel[%u], Set "
        "CKE[%u:%04x], clearType[%u]\n",
        waitCKEId_, waitCKEMask_, locCKEId_, locCKEMask_, rmtCKEId_, locCKEMask_, channelId_, setCKEId_, setCKEMask_,
        clearType_);
}

CcuTrace::CcuInstrTraceDetail SyncCkeExecutor::CollectTraceDetail()
{
    CcuTrace::CcuInstrTraceDetail detail;
    detail.typeName = "SyncCke";
    return detail;
}
