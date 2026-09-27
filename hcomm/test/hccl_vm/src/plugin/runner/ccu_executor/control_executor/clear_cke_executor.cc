/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "clear_cke_executor.h"

#include "ccu_executor_manager.h"
#include "ccu_string_util.h"

using namespace std;
using namespace hcomm::CcuRep;

// 注册ClearCkeExecutor create Func
REG_CCU_EXECUTOR_CREATE_FUNC(SimCcuV1::CTRL_TYPE, SimCcuV1::CLEARCKE_CODE, ClearCkeExecutor);
REG_CCU_EXECUTOR_CREATE_FUNC_V2(SimCcuV2::CTRL_TYPE, SimCcuV2::CLEARCKE_CODE, ClearCkeExecutor);

void ClearCkeExecutor::Parser()
{
    if (version_ == RunnerCcuVersion::CCU_V1) {
        clearType_ = instr_.v1.clearCKE.clearType;
        clearCKEId_ = instr_.v1.clearCKE.clearCKEId;
        clearMask_ = instr_.v1.clearCKE.clearMask;
        waitCKEId_ = instr_.v1.clearCKE.waitCKEId;
        waitCKEMask_ = instr_.v1.clearCKE.waitCKEMask;
    } else if (version_ == RunnerCcuVersion::CCU_V2) {
        clearType_ = instr_.v2.clearCKE.clearType;
        clearCKEId_ = instr_.v2.clearCKE.clearCKEId;
        clearMask_ = instr_.v2.clearCKE.clearMask;
        waitCKEId_ = instr_.v2.clearCKE.waitCKEId;
        waitCKEMask_ = instr_.v2.clearCKE.waitCKEMask;
    } else {
        HCCL_VM_ERROR("Invalid ccu version:{}", RunnerCcuVersionToString(version_));
        ccuSimulator_->SetExecState(CcuExecState::EXEC_FAIL);
        return;
    }
}

void ClearCkeExecutor::Process(CcuResourceManager& ccuResMgr) { ClearCkeSignal(ccuResMgr, clearCKEId_, clearMask_); }

void ClearCkeExecutor::Run() { WaitCkeProcess(waitCKEId_, waitCKEMask_, clearType_, "ClearCKE"); }

std::string ClearCkeExecutor::Describe()
{
    return HcclSim::StringFormat(
        "[Simulation Execute] Wait CKE[%u:%04x], Clear CKE[%u:%04x], clearType[%u]\n", waitCKEId_, waitCKEMask_,
        clearCKEId_, clearMask_, clearType_);
}

CcuTrace::CcuInstrTraceDetail ClearCkeExecutor::CollectTraceDetail()
{
    CcuTrace::CcuInstrTraceDetail detail;
    detail.typeName = "ClearCke";
    return detail;
}
