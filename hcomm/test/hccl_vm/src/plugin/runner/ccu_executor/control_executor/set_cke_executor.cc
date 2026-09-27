/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "set_cke_executor.h"

#include "ccu_executor_manager.h"
#include "ccu_string_util.h"

using namespace std;
using namespace hcomm::CcuRep;

// 注册SetCkeExecutor create Func
REG_CCU_EXECUTOR_CREATE_FUNC(SimCcuV1::CTRL_TYPE, SimCcuV1::SETCKE_CODE, SetCkeExecutor);
REG_CCU_EXECUTOR_CREATE_FUNC_V2(SimCcuV2::CTRL_TYPE, SimCcuV2::SETCKBIT_CODE, SetCkeExecutor);

void SetCkeExecutor::Parser()
{
    if (version_ == RunnerCcuVersion::CCU_V1) {
        clearType_ = instr_.v1.setCKE.clearType;
        setCKEId_ = instr_.v1.setCKE.setCKEId;
        setCKEMask_ = instr_.v1.setCKE.setCKEMask;
        waitCKEId_ = instr_.v1.setCKE.waitCKEId;
        waitCKEMask_ = instr_.v1.setCKE.waitCKEMask;
    } else if (version_ == RunnerCcuVersion::CCU_V2) {
        clearType_ = instr_.v2.setCKE.clearType;
        setCKEId_ = instr_.v2.setCKE.setCKEId;
        setCKEMask_ = instr_.v2.setCKE.setCKEMask;
        waitCKEId_ = instr_.v2.setCKE.waitCKEId;
        waitCKEMask_ = instr_.v2.setCKE.waitCKEMask;
    } else {
        HCCL_VM_ERROR("Invalid ccu version:{}", RunnerCcuVersionToString(version_));
        ccuSimulator_->SetExecState(CcuExecState::EXEC_FAIL);
        return;
    }
}

void SetCkeExecutor::Process(CcuResourceManager& ccuResMgr)
{
    // 设置本端的cke
    SetCkeSignal(ccuResMgr, setCKEId_, setCKEMask_);
}

void SetCkeExecutor::Run() { WaitCkeProcess(waitCKEId_, waitCKEMask_, clearType_, "SetCKE"); }

std::string SetCkeExecutor::Describe()
{
    return HcclSim::StringFormat(
        "[Simulation Execute] Wait CKE[%u:%04x], Set CKE[%u:%04x], clearType[%u]\n", waitCKEId_, waitCKEMask_,
        setCKEId_, setCKEMask_, clearType_);
}

CcuTrace::CcuInstrTraceDetail SetCkeExecutor::CollectTraceDetail()
{
    CcuTrace::CcuInstrTraceDetail detail;
    detail.typeName = "SetCke";
    return detail;
}
