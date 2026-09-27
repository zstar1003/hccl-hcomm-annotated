/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "add_executor.h"

#include <cstdint>

#include "ccu_executor_manager.h"
#include "ccu_string_util.h"

using namespace std;
using namespace hcomm::CcuRep;

REG_CCU_EXECUTOR_CREATE_FUNC_V2(SimCcuV2::LOAD_TYPE, SimCcuV2::ADD_CODE, AddExecutor);

void AddExecutor::Parser()
{
    ValidateVersionExclusive(RunnerCcuVersion::CCU_V2, "AddExecutor");
    xdId_ = instr_.v2.operate.xdId;
    xnId_ = instr_.v2.operate.xnId;
    xmId_ = instr_.v2.operate.xmId;
    parMode_ = instr_.v2.operate.parMode;
    ckeId_ = instr_.v2.operate.setCKEId;
    ckeMask_ = instr_.v2.operate.setCKEMask;
}

void AddExecutor::Run()
{
    uint16_t xnId = GetXnId(xnId_);
    uint16_t xmId = GetXnId(xmId_);
    uint16_t xdId = GetXnId(xdId_);
    uint64_t xdValue = 0;
    auto& ccuResMgr = CcuResourceManager::GetInstance();
    uint64_t xnValue = ccuResMgr.GetXnValue(rankId_, dieId_, xnId);
    if (parMode_ == 0) {
        xdValue = xnValue + xmId_;
        HCCL_VM_INFO("Add Xn{}{} + immed{} to Xd{}{}", xnId_, xnValue, xmId_, xdId_, xdValue);
    } else {
        uint64_t xmValue = ccuResMgr.GetXnValue(rankId_, dieId_, xmId);
        xdValue = xnValue + xmValue;
        HCCL_VM_INFO("Add Xn{} {}+ Xm{}{} to Xd{}{}", xnId, xnValue, xmId, xmValue, xdId_, xdValue);
    }
    ccuResMgr.UpdateXnValue(rankId_, dieId_, xdId, xdValue);

    uint16_t ckeId = UpdateCkeId(ckeId_);
    SetCkeSignal(ccuResMgr, ckeId, ckeMask_);
}

std::string AddExecutor::Describe()
{
    return HcclSim::StringFormat(
        "[AddExecutor] xdId:[%u],xnId_[%u],xmId[%u],parMode[%u]\n", xdId_, xnId_, xmId_, parMode_);
}

CcuTrace::CcuInstrTraceDetail AddExecutor::CollectTraceDetail()
{
    CcuTrace::CcuInstrTraceDetail detail;
    detail.typeName = "Add";
    auto& ccuResMgr = CcuResourceManager::GetInstance();
    detail.args["xnValue"] = std::to_string(ccuResMgr.GetXnValue(rankId_, dieId_, xnId_));
    detail.args["xmValue"] = std::to_string(ccuResMgr.GetXnValue(rankId_, dieId_, xmId_));
    return detail;
}
