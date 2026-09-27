/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "sync_at_executor.h"

#include "ccu_executor_manager.h"
#include "ccu_microcode_common_v2.h"
#include "sim_log.h"
#include "ccu_string_util.h"

using namespace std;
using namespace hcomm::CcuRep;

REG_CCU_EXECUTOR_CREATE_FUNC_V2(SimCcuV2::TRANS_TYPE, SimCcuV2::SYNCATX_CODE, SyncAtExecutor);

void SyncAtExecutor::Parser()
{
    ValidateVersionExclusive(RunnerCcuVersion::CCU_V2, "SyncAtExecutor");
    xdId_ = instr_.v2.syncAtX.xdId;
    xdtId_ = instr_.v2.syncAtX.xdtId;
    xsId_ = instr_.v2.syncAtX.xsId;
    xcId_ = instr_.v2.syncAtX.xcId;
    parMode_ = instr_.v2.syncAtX.parMode;
    setCKEId_ = instr_.v2.syncAtX.setCKEId;
    setCKEMask_ = instr_.v2.syncAtX.setCKEMask;
}

void SyncAtExecutor::Run()
{
    HCCL_VM_ERROR("SyncAtExecutor unsupport run");
    ccuSimulator_->SetExecState(CcuExecState::EXEC_FAIL);
    return;
}

std::string SyncAtExecutor::Describe()
{
    return HcclSim::StringFormat(
        "[SyncAtExecutor] xdId[%u] xdtId[%u] xsId[%u] xcId[%u] "
        "parMode[%u] setCKEId[%u] setCKEMask[0x%04x]\n",
        xdId_, xdtId_, xsId_, xcId_, parMode_, setCKEId_, setCKEMask_);
}

CcuTrace::CcuInstrTraceDetail SyncAtExecutor::CollectTraceDetail()
{
    CcuTrace::CcuInstrTraceDetail detail;
    detail.typeName = "SyncAt";
    return detail;
}
