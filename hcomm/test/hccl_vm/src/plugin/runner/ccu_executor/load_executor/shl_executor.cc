/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "shl_executor.h"

#include <cstdint>

#include "ccu_executor_manager.h"
#include "ccu_string_util.h"

using namespace std;
using namespace hcomm::CcuRep;

REG_CCU_EXECUTOR_CREATE_FUNC_V2(SimCcuV2::LOAD_TYPE, SimCcuV2::SHL_CODE, ShlExecutor);

void ShlExecutor::Parser()
{
    ValidateVersionExclusive(RunnerCcuVersion::CCU_V2, "ShlExecutor");
    xdId_ = instr_.v2.operate.xdId;
    xnId_ = instr_.v2.operate.xnId;
    xmId_ = instr_.v2.operate.xmId;
    shiftType_ = instr_.v2.operate.shiftType;
    ckeId_ = instr_.v2.operate.setCKEId;
    ckeMask_ = instr_.v2.operate.setCKEMask;
}

void ShlExecutor::Run()
{
    uint16_t xnId = GetXnId(xnId_);
    uint16_t xmId = GetXnId(xmId_);
    uint16_t xdId = GetXnId(xdId_);
    uint64_t xdValue = 0;
    auto& ccuResMgr = CcuResourceManager::GetInstance();
    uint64_t xnValue = ccuResMgr.GetXnValue(rankId_, dieId_, xnId);
    uint64_t xmValue = ccuResMgr.GetXnValue(rankId_, dieId_, xmId);

    switch (shiftType_) {
        case 0:
            xdValue = xnValue << xmValue;
            HCCL_VM_INFO("Shl logical Xn{}{} << Xm{}{} to Xd{}{}", xnId, xnValue, xmId, xmValue, xdId_, xdValue);
            break;
        case 1:
            xdValue = static_cast<int64_t>(xnValue) << xmValue;
            HCCL_VM_INFO("Shl arithmetic Xn{}{} << Xm{}{} to Xd{}{}", xnId, xnValue, xmId, xmValue, xdId_, xdValue);
            break;
        case 2:
            xdValue = (xnValue << (xmValue & 0x3F)) | (xnValue >> (64 - (xmValue & 0x3F)));
            HCCL_VM_INFO("Shl rotate Xn{}{} <<< Xm{}{} to Xd{}{}", xnId, xnValue, xmId, xmValue, xdId_, xdValue);
            break;
        default:
            HCCL_VM_ERROR("Unsupported shift type: [{}]", shiftType_);
            return;
    }
    ccuResMgr.UpdateXnValue(rankId_, dieId_, xdId, xdValue);

    uint16_t ckeId = UpdateCkeId(ckeId_);
    SetCkeSignal(ccuResMgr, ckeId, ckeMask_);
}

std::string ShlExecutor::Describe()
{
    return HcclSim::StringFormat(
        "[ShlExecutor] xdId:[%u],xnId_[%u],xmId[%u],shiftType[%u]\n", xdId_, xnId_, xmId_, shiftType_);
}

CcuTrace::CcuInstrTraceDetail ShlExecutor::CollectTraceDetail()
{
    CcuTrace::CcuInstrTraceDetail detail;
    detail.typeName = "Shl";
    auto& ccuResMgr = CcuResourceManager::GetInstance();
    detail.args["xnValue"] = std::to_string(ccuResMgr.GetXnValue(rankId_, dieId_, xnId_));
    detail.args["xmValue"] = std::to_string(ccuResMgr.GetXnValue(rankId_, dieId_, xmId_));
    return detail;
}
