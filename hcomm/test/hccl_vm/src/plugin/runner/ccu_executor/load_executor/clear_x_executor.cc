/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "clear_x_executor.h"

#include <cstdint>

#include "ccu_executor_manager.h"
#include "ccu_string_util.h"

using namespace std;
using namespace hcomm::CcuRep;
constexpr uint16_t MAX_LOADX_STOREX_ID_NUM = 16383;
// 注册ClearXExecutor create Func
REG_CCU_EXECUTOR_CREATE_FUNC_V2(SimCcuV2::LOAD_TYPE, SimCcuV2::CLEARX_CODE, ClearXExecutor);

void ClearXExecutor::Parser()
{
    ValidateVersionExclusive(RunnerCcuVersion::CCU_V2, "ClearXExecutor");
#ifdef BUILD_A6_CCU_INSTR
    xnId_ = instr_.v2.clearX.xnId;
    xmId_ = instr_.v2.clearX.xmId;
    xnIdMode_ = instr_.v2.clearX.xnIdMode;
    xmIdMode_ = instr_.v2.clearX.xmIdMode;
    ckeId_ = instr_.v2.clearX.setCKEId;
    ckeMask_ = instr_.v2.clearX.setCKEMask;
#endif
}

void ClearXExecutor::Run()
{
    uint16_t xnId = (xnIdMode_ == 0) ? xnId_ : UpdateXnId(xnId_);
    uint16_t xmId = (xmIdMode_ == 0) ? xmId_ : UpdateXnId(xmId_);
    xmId = (xnId <= xmId) ? xmId : MAX_LOADX_STOREX_ID_NUM;

    auto& ccuResMgr = CcuResourceManager::GetInstance();
    for (uint16_t i = xnId; i <= xmId; i++) {
        ccuResMgr.UpdateXnValue(rankId_, dieId_, i, 0);
    }
    HCCL_VM_INFO("ClearX from Xn[{}] to Xm[{}]", xnId, xmId);

    uint16_t ckeId = UpdateCkeId(ckeId_);
    SetCkeSignal(ccuResMgr, ckeId, ckeMask_);
}

std::string ClearXExecutor::Describe()
{
    return HcclSim::StringFormat(
        "[ClearXExecutor] xnId:[%u],xmId[%u],xnIdMode[%u],xmIdMode[%u]\n", xnId_, xmId_, xnIdMode_, xmIdMode_);
}

CcuTrace::CcuInstrTraceDetail ClearXExecutor::CollectTraceDetail()
{
    CcuTrace::CcuInstrTraceDetail detail;
    detail.typeName = "ClearX";
    return detail;
}
