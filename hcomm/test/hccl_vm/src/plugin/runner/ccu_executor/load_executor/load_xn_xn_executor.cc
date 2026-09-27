/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "load_xn_xn_executor.h"

#include <cstdint>

#include "ccu_executor_manager.h"
#include "ccu_microcode_common_v1.h"
#include "sim_log.h"
#include "ccu_string_util.h"

using namespace std;
using namespace hcomm::CcuRep;

// 注册LoadGSAGSAExecutor create Func
REG_CCU_EXECUTOR_CREATE_FUNC(SimCcuV1::LOAD_TYPE, SimCcuV1::LOADXX_CODE, LoadXnXnExecutor);

void LoadXnXnExecutor::Parser()
{
    ValidateVersionExclusive(RunnerCcuVersion::CCU_V1, "LoadXnXnExecutor");
    xdId_ = instr_.v1.loadXX.xdId;
    xmId_ = instr_.v1.loadXX.xmId;
    xnId_ = instr_.v1.loadXX.xnId;
}

void LoadXnXnExecutor::Run()
{
    auto& ccuResMgr = CcuResourceManager::GetInstance();
    uint64_t xn1 = ccuResMgr.GetXnValue(rankId_, dieId_, xmId_);
    uint64_t xn2 = ccuResMgr.GetXnValue(rankId_, dieId_, xnId_);
    uint64_t val = xn1 + xn2;
    ccuResMgr.UpdateXnValue(rankId_, dieId_, xdId_, val);

    HCCL_VM_DEBUG("Load Xn[{}] + Xn[{}] to Xn[{}]", xmId_, xnId_, xdId_);
}

std::string LoadXnXnExecutor::Describe()
{
    return HcclSim::StringFormat("[Simulation Execute] Load Xn[%u] + Xn[%u] to Xn[%u]\n", xmId_, xnId_, xdId_);
}

CcuTrace::CcuInstrTraceDetail LoadXnXnExecutor::CollectTraceDetail()
{
    CcuTrace::CcuInstrTraceDetail detail;
    detail.typeName = "LoadXnXn";
    auto& ccuResMgr = CcuResourceManager::GetInstance();
    detail.args["xn1"] = std::to_string(ccuResMgr.GetXnValue(rankId_, dieId_, xmId_));
    detail.args["xn2"] = std::to_string(ccuResMgr.GetXnValue(rankId_, dieId_, xnId_));
    return detail;
}
