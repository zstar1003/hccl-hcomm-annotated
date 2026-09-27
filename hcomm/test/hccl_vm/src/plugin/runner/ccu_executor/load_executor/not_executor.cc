/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "not_executor.h"

#include <cstdint>

#include "ccu_executor_manager.h"
#include "ccu_string_util.h"

using namespace std;
using namespace hcomm::CcuRep;

REG_CCU_EXECUTOR_CREATE_FUNC_V2(SimCcuV2::LOAD_TYPE, SimCcuV2::NOT_CODE, NotExecutor);

void NotExecutor::Parser()
{
    ValidateVersionExclusive(RunnerCcuVersion::CCU_V2, "NotExecutor");
    xdId_ = instr_.v2.operate.xdId;
    xnId_ = instr_.v2.operate.xnId;
    ckeId_ = instr_.v2.operate.setCKEId;
    ckeMask_ = instr_.v2.operate.setCKEMask;
}

void NotExecutor::Run()
{
    uint16_t xnId = GetXnId(xnId_);
    uint16_t xdId = GetXnId(xdId_);
    auto& ccuResMgr = CcuResourceManager::GetInstance();
    uint64_t xnValue = ccuResMgr.GetXnValue(rankId_, dieId_, xnId);
    uint64_t xdValue = ~xnValue;
    HCCL_VM_INFO("Not Xn{}{} | to Xd{}{}", xnId, xnValue, xdId_, xdValue);
    ccuResMgr.UpdateXnValue(rankId_, dieId_, xdId, xdValue);

    uint16_t ckeId = UpdateCkeId(ckeId_);
    SetCkeSignal(ccuResMgr, ckeId, ckeMask_);
}

std::string NotExecutor::Describe()
{
    return HcclSim::StringFormat(
        "[NotExecutor] xdId:[%u], xnId:[%u], ckeId:[%u], ckeMask:[0x%04x]\n", xdId_, xnId_, ckeId_, ckeMask_);
}

CcuTrace::CcuInstrTraceDetail NotExecutor::CollectTraceDetail()
{
    CcuTrace::CcuInstrTraceDetail detail;
    detail.typeName = "Not";
    auto& ccuResMgr = CcuResourceManager::GetInstance();
    detail.args["xnValue"] = std::to_string(ccuResMgr.GetXnValue(rankId_, dieId_, xnId_));
    return detail;
}
