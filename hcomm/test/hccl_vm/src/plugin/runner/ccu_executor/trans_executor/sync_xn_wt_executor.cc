/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "sync_xn_wt_executor.h"

#include <cstdint>

#include "ccu_executor_manager.h"
#include "ccu_microcode_common_v2.h"
#include "sim_log.h"
#include "ccu_string_util.h"

using namespace std;
using namespace hcomm::CcuRep;

REG_CCU_EXECUTOR_CREATE_FUNC_V2(SimCcuV2::TRANS_TYPE, SimCcuV2::SYNCXNWTX_CODE, SyncXnWtExecutor);

void SyncXnWtExecutor::Parser()
{
    ValidateVersionExclusive(RunnerCcuVersion::CCU_V2, "SyncXnWtExecutor");
    xdId_ = instr_.v2.syncWtX.xdId;
    xdtId_ = instr_.v2.syncWtX.xdtId;
    xsId_ = instr_.v2.syncWtX.xsId;
    xcId_ = instr_.v2.syncWtX.xcId;
    xnId_ = instr_.v2.syncWtX.xnId;
    xntId_ = instr_.v2.syncWtX.xntId;
    value_ = instr_.v2.syncWtX.value;
    notifyValid_ = instr_.v2.syncWtX.notifyValid;
    parMode_ = instr_.v2.syncWtX.parMode;
    setCKEId_ = instr_.v2.syncWtX.setCKEId;
    setCKEMask_ = instr_.v2.syncWtX.setCKEMask;
}

void SyncXnWtExecutor::Run()
{
    uint16_t xdId = GetXnId(xdId_);
    auto& ccuResMgr = CcuResourceManager::GetInstance();
    uint64_t xdValue = ccuResMgr.GetXnValue(rankId_, dieId_, xdId);
    uint16_t xcId = GetXnId(xcId_);
    uint64_t channlId = ccuResMgr.GetXnValue(rankId_, dieId_, xcId);
    auto rmtCcu = ccuResMgr.GetRmtCcu(rankId_, dieId_, channlId);
    int rmtRankId = rmtCcu.first;
    int rmtDieId = rmtCcu.second;
    uint64_t xsValue = xsId_;
    if (parMode_ == 1) {
        uint16_t xsId = GetXnId(xsId_);
        xsValue = ccuResMgr.GetXnValue(rankId_, dieId_, xsId);
    }
    // 将Xs内容写到Xd中
    uint16_t xdIdRmt{0};
    CcuComponerntType ccuType{CcuComponerntType::UNKNOWN};
    if (!ccuResMgr.GetXnAndTypeIdByAddr(dieId_, xdValue, ccuType, xdIdRmt)) {
        HCCL_VM_ERROR("GetXnAndTypeIdByAddr failed. xdId:[{}]", xdId);
        ccuSimulator_->SetExecState(CcuExecState::EXEC_FAIL);
        return;
    }

    if (ccuType == CcuComponerntType::XN_A6) {
        ccuResMgr.UpdateXnValue(rmtRankId, rmtDieId, xdIdRmt, xsValue);
    } else if (ccuType == CcuComponerntType::CKE_A6) {
        SetRmtCKESignal(ccuResMgr, rmtRankId, rmtDieId, xdIdRmt, xsValue);
    } else {
        HCCL_VM_ERROR("TransformSyncXnWtxInstr is not supported. xdId:[{}]", xdId);
        ccuSimulator_->SetExecState(CcuExecState::EXEC_FAIL);
        return;
    }
    // 写notify信息
    if (notifyValid_ == 1) {
        uint16_t xnId = GetXnId(xnId_);
        uint64_t xnAddr = ccuResMgr.GetXnValue(rankId_, dieId_, xnId);
        uint16_t cekIdRmtId = 0;
        if (!ccuResMgr.GetXnIdByAddr(dieId_, CcuComponerntType::CKE_A6, xnAddr, cekIdRmtId)) {
            ccuSimulator_->SetExecState(CcuExecState::EXEC_FAIL);
            return;
        }
        SetRmtCKESignal(ccuResMgr, rmtRankId, rmtDieId, cekIdRmtId, value_);
    }
}

std::string SyncXnWtExecutor::Describe()
{
    return HcclSim::StringFormat(
        "[SyncXnWtExecutor] xdId[%u] xdtId[%u] xsId[%u] xcId[%u] "
        "xnId[%u] xntId[%u] value[0x%08x] notifyValid[%u] parMode[%u] "
        "setCKEId[%u] setCKEMask[0x%04x]\n",
        xdId_, xdtId_, xsId_, xcId_, xnId_, xntId_, value_, notifyValid_, parMode_, setCKEId_, setCKEMask_);
}

CcuTrace::CcuInstrTraceDetail SyncXnWtExecutor::CollectTraceDetail()
{
    CcuTrace::CcuInstrTraceDetail detail;
    detail.typeName = "SyncXnWt";
    return detail;
}
