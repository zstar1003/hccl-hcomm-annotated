/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "store_executor.h"

#include <cstdint>

#include "ccu_executor_manager.h"
#include "sim_log.h"
#include "ccu_string_util.h"

using namespace std;
using namespace hcomm::CcuRep;

REG_CCU_EXECUTOR_CREATE_FUNC_V2(SimCcuV2::LOAD_TYPE, SimCcuV2::STORE_CODE, StoreExecutor);

void StoreExecutor::Parser()
{
    ValidateVersionExclusive(RunnerCcuVersion::CCU_V2, "StoreExecutor");
    srcType_ = instr_.v2.store.srcType;
    xdId_ = instr_.v2.store.xdId;
    xsId_ = instr_.v2.store.xsId;
    xlId_ = instr_.v2.store.xlId;
    ckeId_ = instr_.v2.store.setCKEId;
    ckeMask_ = instr_.v2.store.setCKEMask;
}

void StoreExecutor::Run()
{
    if (srcType_ != 0) {
        HCCL_VM_ERROR("TransformStoreInstr ERROR,srcType={}", srcType_);
        ccuSimulator_->SetExecState(CcuExecState::EXEC_FAIL);
        return;
    }
    uint16_t xdId = GetXnId(xdId_);
    uint16_t xsId = GetXnId(xsId_);
    uint16_t xlId = GetXnId(xlId_);
    auto& ccuResMgr = CcuResourceManager::GetInstance();
    uint64_t xdValue = ccuResMgr.GetXnValue(rankId_, dieId_, xdId);
    uint64_t xsValue = ccuResMgr.GetXnValue(rankId_, dieId_, xsId);
    uint64_t length = ccuResMgr.GetXnValue(rankId_, dieId_, xlId);
    if ((length & 0x7) != 0) {
        HCCL_VM_ERROR("TransformStoreInstr ERROR,length = {}", length);
        ccuSimulator_->SetExecState(CcuExecState::EXEC_FAIL);
        return;
    }
    if (!ccuResMgr.TransXnToMem(rankId_, dieId_, xsValue, xdValue, length)) {
        ccuSimulator_->SetExecState(CcuExecState::EXEC_FAIL);
        return;
    }
    uint16_t ckeId = UpdateCkeId(ckeId_);
    SetCkeSignal(ccuResMgr, ckeId, ckeMask_);
}

std::string StoreExecutor::Describe()
{
    return HcclSim::StringFormat(
        "[StoreExecutor] xdId:[%u], xsId:[%u], xlId:[%u], ckeId:[%u], ckeMask:[0x%04x]\n", xdId_, xsId_, xlId_, ckeId_,
        ckeMask_);
}

CcuTrace::CcuInstrTraceDetail StoreExecutor::CollectTraceDetail()
{
    CcuTrace::CcuInstrTraceDetail detail;
    detail.typeName = "Store";
    auto& ccuResMgr = CcuResourceManager::GetInstance();
    detail.args["xdValue"] = std::to_string(ccuResMgr.GetXnValue(rankId_, dieId_, xdId_));
    detail.args["xsValue"] = std::to_string(ccuResMgr.GetXnValue(rankId_, dieId_, xsId_));
    detail.args["length"] = std::to_string(ccuResMgr.GetXnValue(rankId_, dieId_, xlId_));
    return detail;
}
