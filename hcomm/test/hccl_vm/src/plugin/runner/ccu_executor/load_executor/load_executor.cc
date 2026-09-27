/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "load_executor.h"

#include <cstdint>

#include "ccu_executor_manager.h"
#include "ccu_string_util.h"

using namespace std;
using namespace hcomm::CcuRep;

REG_CCU_EXECUTOR_CREATE_FUNC_V2(SimCcuV2::LOAD_TYPE, SimCcuV2::LOAD_CODE, LoadExecutor);

void LoadExecutor::Parser()
{
    ValidateVersionExclusive(RunnerCcuVersion::CCU_V2, "LoadExecutor");
    dstType_ = instr_.v2.load.dstType;
    xdId_ = instr_.v2.load.xdId;
    xsId_ = instr_.v2.load.xsId;
    xlId_ = instr_.v2.load.xlId;
    ckeId_ = instr_.v2.load.setCKEId;
    ckeMask_ = instr_.v2.load.setCKEMask;
}

void LoadExecutor::Run()
{
    if (dstType_ != 0) {
        // 目前只支持Xn寄存器
        HCCL_VM_ERROR("TransformLoadInstr ERROR,dstType={}", dstType_);
        ccuSimulator_->SetExecState(CcuExecState::EXEC_FAIL);
        return;
    }
    uint16_t xdId = GetXnId(xdId_);
    uint16_t xsId = GetXnId(xsId_);
    uint16_t xlId = GetXnId(xlId_);
    auto& ccuResMgr = CcuResourceManager::GetInstance();
    uint64_t xdValue = ccuResMgr.GetXnValue(rankId_, dieId_, xdId);
    uint64_t xsStartAddr = ccuResMgr.GetXnValue(rankId_, dieId_, xsId);
    if ((xsStartAddr & 0x7) != 0) {
        HCCL_VM_ERROR("TransformLoadInstr ERROR,xsStartAddr = {}", xsStartAddr);
        ccuSimulator_->SetExecState(CcuExecState::EXEC_FAIL);
        return;
    }
    uint64_t length = ccuResMgr.GetXnValue(rankId_, dieId_, xlId);
    if (!ccuResMgr.TransMemToXn(rankId_, dieId_, xdValue, xsStartAddr, length)) {
        ccuSimulator_->SetExecState(CcuExecState::EXEC_FAIL);
        return;
    }
    uint16_t ckeId = UpdateCkeId(ckeId_);
    SetCkeSignal(ccuResMgr, ckeId, ckeMask_);
}

std::string LoadExecutor::Describe()
{
    return HcclSim::StringFormat(
        "[LoadExecutor] dstType:[%u], xdId:[%u], xsId:[%u], xlId:[%u], ckeId:[%u], ckeMask:[0x%04x]\n", dstType_, xdId_,
        xsId_, xlId_, ckeId_, ckeMask_);
}

CcuTrace::CcuInstrTraceDetail LoadExecutor::CollectTraceDetail()
{
    CcuTrace::CcuInstrTraceDetail detail;
    detail.typeName = "Load";
    auto& ccuResMgr = CcuResourceManager::GetInstance();
    detail.args["xdValue"] = std::to_string(ccuResMgr.GetXnValue(rankId_, dieId_, xdId_));
    detail.args["xsStartAddr"] = std::to_string(ccuResMgr.GetXnValue(rankId_, dieId_, xsId_));
    detail.args["length"] = std::to_string(ccuResMgr.GetXnValue(rankId_, dieId_, xlId_));
    return detail;
}
