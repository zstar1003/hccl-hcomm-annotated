/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "store_x_executor.h"

#include <cstdint>

#include "ccu_executor_manager.h"
#include "ccu_string_util.h"

using namespace std;
using namespace hcomm::CcuRep;

REG_CCU_EXECUTOR_CREATE_FUNC_V2(SimCcuV2::LOAD_TYPE, SimCcuV2::STOREX_CODE, StoreXExecutor);

void StoreXExecutor::Parser()
{
    ValidateVersionExclusive(RunnerCcuVersion::CCU_V2, "StoreXExecutor");
    xdId_ = instr_.v2.loadStoreX.xdId;
    xsId_ = instr_.v2.loadStoreX.xsId;
    xsoId_ = instr_.v2.loadStoreX.xso;
    xdoId_ = instr_.v2.loadStoreX.xdo;
    oMode_ = instr_.v2.loadStoreX.oMode;
    ckeId_ = instr_.v2.loadStoreX.setCKEId;
    ckeMask_ = instr_.v2.loadStoreX.setCKEMask;
}

void StoreXExecutor::Run()
{
    uint16_t xsId = GetXnId(xsId_);
    uint16_t xdId = GetXnId(xdId_);
    auto& ccuResMgr = CcuResourceManager::GetInstance();
    uint64_t xsValue = ccuResMgr.GetXnValue(rankId_, dieId_, xsId);
    uint64_t xdValue = ccuResMgr.GetXnValue(rankId_, dieId_, xdId);

    uint16_t newId = 0;
    uint64_t newValue = 0;

    if (oMode_ == 0) {
        newId = static_cast<uint16_t>(xdValue) + xdoId_;
        newValue = xsValue + static_cast<uint64_t>(xsoId_);
    } else if (oMode_ == 1) {
        uint16_t xsDoId = GetXnId(xdoId_);
        uint64_t xsDoValue = ccuResMgr.GetXnValue(rankId_, dieId_, xsDoId);
        newId = static_cast<uint16_t>(xdValue + xsDoValue);

        uint16_t xsSoId = GetXnId(xsoId_);
        uint64_t xsSoValue = ccuResMgr.GetXnValue(rankId_, dieId_, xsSoId);
        newValue = xsValue + xsSoValue;
    } else {
        HCCL_VM_ERROR("StoreXExecutor unsupport mode[{}]", oMode_);
        return;
    }

    ccuResMgr.UpdateXnValue(rankId_, dieId_, newId, newValue);
    HCCL_VM_INFO("StoreX *X(Xd[{}]+{}) = Xs[{}]+Xso[{}]", xdId, xdoId_, xsId, xsoId_);

    uint16_t ckeId = UpdateCkeId(ckeId_);
    SetCkeSignal(ccuResMgr, ckeId, ckeMask_);
}

std::string StoreXExecutor::Describe()
{
    return HcclSim::StringFormat(
        "[StoreXExecutor] xdId:[%u],xsId[%u],xsoId[%u],xdoId[%u],oMode[%u]\n", xdId_, xsId_, xsoId_, xdoId_, oMode_);
}

CcuTrace::CcuInstrTraceDetail StoreXExecutor::CollectTraceDetail()
{
    CcuTrace::CcuInstrTraceDetail detail;
    detail.typeName = "StoreX";
    auto& ccuResMgr = CcuResourceManager::GetInstance();
    detail.args["xsValue"] = std::to_string(ccuResMgr.GetXnValue(rankId_, dieId_, xsId_));
    detail.args["xdValue"] = std::to_string(ccuResMgr.GetXnValue(rankId_, dieId_, xdId_));
    return detail;
}
