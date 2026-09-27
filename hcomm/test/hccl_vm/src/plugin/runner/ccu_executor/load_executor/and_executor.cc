/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "and_executor.h"

#include <cstdint>

#include "ccu_executor_manager.h"
#include "ccu_string_util.h"

using namespace std;
using namespace hcomm::CcuRep;

REG_CCU_EXECUTOR_CREATE_FUNC_V2(SimCcuV2::LOAD_TYPE, SimCcuV2::AND_CODE, AndExecutor);

void AndExecutor::Parser()
{
    ValidateVersionExclusive(RunnerCcuVersion::CCU_V2, "AndExecutor");
    xdId_ = instr_.v2.operate.xdId;
    xnId_ = instr_.v2.operate.xnId;
    xmId_ = instr_.v2.operate.xmId;
    ckeId_ = instr_.v2.operate.setCKEId;
    ckeMask_ = instr_.v2.operate.setCKEMask;
}

void AndExecutor::Run()
{
    uint16_t xnId = GetXnId(xnId_);
    uint16_t xmId = GetXnId(xmId_);
    uint16_t xdId = GetXnId(xdId_);
    auto& ccuResMgr = CcuResourceManager::GetInstance();
    uint64_t xnValue = ccuResMgr.GetXnValue(rankId_, dieId_, xnId);
    uint64_t xmValue = ccuResMgr.GetXnValue(rankId_, dieId_, xmId);
    uint64_t xdValue = xnValue & xmValue;
    HCCL_VM_INFO("And Xn{}{} & Xm{}{} to Xd{}{}", xnId, xnValue, xmId, xmValue, xdId_, xdValue);
    ccuResMgr.UpdateXnValue(rankId_, dieId_, xdId, xdValue);

    uint16_t ckeId = UpdateCkeId(ckeId_);
    SetCkeSignal(ccuResMgr, ckeId, ckeMask_);
}

std::string AndExecutor::Describe()
{
    return HcclSim::StringFormat("[AndExecutor] xdId:[%u],xnId_[%u],xmId[%u]\n", xdId_, xnId_, xmId_);
}

CcuTrace::CcuInstrTraceDetail AndExecutor::CollectTraceDetail()
{
    CcuTrace::CcuInstrTraceDetail detail;
    detail.typeName = "And";
    auto& ccuResMgr = CcuResourceManager::GetInstance();
    detail.args["xnValue"] = std::to_string(ccuResMgr.GetXnValue(rankId_, dieId_, xnId_));
    detail.args["xmValue"] = std::to_string(ccuResMgr.GetXnValue(rankId_, dieId_, xmId_));
    return detail;
}
