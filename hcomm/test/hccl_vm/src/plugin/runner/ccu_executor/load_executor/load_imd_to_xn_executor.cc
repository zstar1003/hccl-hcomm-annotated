/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "load_imd_to_xn_executor.h"

#include <cstdint>

#include "ccu_executor_manager.h"
#include "sim_log.h"
#include "ccu_string_util.h"

using namespace std;
using namespace hcomm::CcuRep;

// 注册LoadImdToXnExecutor create Func
REG_CCU_EXECUTOR_CREATE_FUNC(SimCcuV1::LOAD_TYPE, SimCcuV1::LOADIMDTOXN_CODE, LoadImdToXnExecutor);
REG_CCU_EXECUTOR_CREATE_FUNC_V2(SimCcuV2::LOAD_TYPE, SimCcuV2::LOADIMDTOX_CODE, LoadImdToXnExecutor);
void LoadImdToXnExecutor::Parser()
{
    if (version_ == RunnerCcuVersion::CCU_V1) {
        xnId_ = instr_.v1.loadImdToXn.xnId;
        immediate_ = instr_.v1.loadImdToXn.immediate;
    } else if (version_ == RunnerCcuVersion::CCU_V2) {
        xnId_ = instr_.v2.loadImdToX.xnId;
        immediate_ = instr_.v2.loadImdToX.immediate;
        ckeId_ = instr_.v2.loadImdToX.setCKEId;
        ckeMask_ = instr_.v2.loadImdToX.setCKEMask;
    } else {
        HCCL_VM_ERROR("Invalid ccu version:{}", RunnerCcuVersionToString(version_));
        ccuSimulator_->SetExecState(CcuExecState::EXEC_FAIL);
        return;
    }
}

void LoadImdToXnExecutor::Run()
{
    auto& ccuResMgr = CcuResourceManager::GetInstance();
    ccuResMgr.UpdateXnValue(rankId_, dieId_, xnId_, immediate_);
    if (version_ == RunnerCcuVersion::CCU_V2) {
        uint16_t ckeId = UpdateCkeId(ckeId_);
        SetCkeSignal(ccuResMgr, ckeId, ckeMask_);
    }
    HCCL_VM_DEBUG("Load immediate: locCcu[{}:{}], XnId=[{}], immediate=[{}]", rankId_, dieId_, xnId_, immediate_);
}

std::string LoadImdToXnExecutor::Describe()
{
    return HcclSim::StringFormat(
        "[Simulation Execute] locCcu[%d:%d], Load immediate[%lu] to Xn[%u]\n", rankId_, dieId_, immediate_, xnId_);
}

CcuTrace::CcuInstrTraceDetail LoadImdToXnExecutor::CollectTraceDetail()
{
    CcuTrace::CcuInstrTraceDetail detail;
    detail.typeName = "LoadImdToXn";
    return detail;
}
