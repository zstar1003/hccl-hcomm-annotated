/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "sync_gsa_executor.h"

#include "ccu_executor_manager.h"
#include "ccu_microcode_common_v1.h"
#include "ccu_string_util.h"

using namespace std;
using namespace hcomm::CcuRep;

// 注册SyncGsaExecutor create Func
REG_CCU_EXECUTOR_CREATE_FUNC(SimCcuV1::TRANS_TYPE, SimCcuV1::SYNCGSA_CODE, SyncGsaExecutor);

void SyncGsaExecutor::Parser()
{
    ValidateVersionExclusive(RunnerCcuVersion::CCU_V1, "SyncGsaExecutor");
    rmtGSAId_ = instr_.v1.syncGSA.rmtGSAId;
    locGSAId_ = instr_.v1.syncGSA.locGSAId;
    channelId_ = instr_.v1.syncGSA.channelId;
    setRmtCKEId_ = instr_.v1.syncGSA.setRmtCKEId;
    setRmtCKEMask_ = instr_.v1.syncGSA.setRmtCKEMask;
    clearType_ = instr_.v1.syncGSA.clearType;
    setCKEId_ = instr_.v1.syncGSA.setCKEId;
    setCKEMask_ = instr_.v1.syncGSA.setCKEMask;
    waitCKEId_ = instr_.v1.syncGSA.waitCKEId;
    waitCKEMask_ = instr_.v1.syncGSA.waitCKEMask;
}

void SyncGsaExecutor::Process(CcuResourceManager& ccuResMgr)
{
    // 根据channel id获取remote rank id
    auto rmtCcu = ccuResMgr.GetRmtCcu(rankId_, dieId_, channelId_);
    // 获取地址
    auto srcAdrr = ccuResMgr.GetGsaValue(rankId_, dieId_, locGSAId_);
    // 将本端的地址写到对端的gsa上
    ccuResMgr.UpdateGsaValue(rmtCcu.first, rmtCcu.second, rmtGSAId_, srcAdrr);

    // 设置目的端的cke
    SetRmtCKESignal(ccuResMgr, rmtCcu.first, rmtCcu.second, setRmtCKEId_, setRmtCKEMask_);

    // 设置本端的cke
    SetCkeSignal(ccuResMgr, setCKEId_, setCKEMask_);
}

void SyncGsaExecutor::Run() { WaitCkeProcess(waitCKEId_, waitCKEMask_, clearType_, "SyncGsa"); }

std::string SyncGsaExecutor::Describe()
{
    return HcclSim::StringFormat(
        "[Simulation Execute] Wait CKE[%u:%04x], Sync locGSAId[%u] To rmtGSAId[%u] Use "
        "Channel[%u], Set rmtCKE[%u:%04x], Set "
        "CKE[%u:%04x], clearType[%u]\n",
        waitCKEId_, waitCKEMask_, locGSAId_, rmtGSAId_, channelId_, setRmtCKEId_, setRmtCKEMask_, setCKEId_,
        setCKEMask_, clearType_);
}

CcuTrace::CcuInstrTraceDetail SyncGsaExecutor::CollectTraceDetail()
{
    CcuTrace::CcuInstrTraceDetail detail;
    detail.typeName = "SyncGsa";
    return detail;
}
