/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "load_gsa_xn_executor.h"

#include <cstdint>

#include "ccu_executor_manager.h"
#include "ccu_microcode_common_v1.h"
#include "ccu_string_util.h"

using namespace std;
using namespace hcomm::CcuRep;

// 注册LoadGSAXnExecutor create Func
REG_CCU_EXECUTOR_CREATE_FUNC(SimCcuV1::LOAD_TYPE, SimCcuV1::LOADGSAXN_CODE, LoadGsaXnExecutor);

void LoadGsaXnExecutor::Parser()
{
    ValidateVersionExclusive(RunnerCcuVersion::CCU_V1, "LoadGsaXn");
    gsAdId_ = instr_.v1.loadGSAXn.gsAdId;
    gsAmId_ = instr_.v1.loadGSAXn.gsAmId;
    xnId_ = instr_.v1.loadGSAXn.xnId;
}

void LoadGsaXnExecutor::Run()
{
    auto& ccuResMgr = CcuResourceManager::GetInstance();
    uint64_t gsaVal = ccuResMgr.GetGsaValue(rankId_, dieId_, gsAmId_);
    uint64_t xnVal = ccuResMgr.GetXnValue(rankId_, dieId_, xnId_);
    uint64_t val = gsaVal + xnVal;
    ccuResMgr.UpdateGsaValue(rankId_, dieId_, gsAdId_, val);
}

std::string LoadGsaXnExecutor::Describe()
{
    return HcclSim::StringFormat("[Simulation Execute] Load GSA[%u] + Xn[%u] to GSA[%u]\n", gsAmId_, xnId_, gsAdId_);
}

CcuTrace::CcuInstrTraceDetail LoadGsaXnExecutor::CollectTraceDetail()
{
    CcuTrace::CcuInstrTraceDetail detail;
    detail.typeName = "LoadGsaXn";
    auto& ccuResMgr = CcuResourceManager::GetInstance();
    detail.args["gsaVal"] = std::to_string(ccuResMgr.GetGsaValue(rankId_, dieId_, gsAmId_));
    detail.args["xnVal"] = std::to_string(ccuResMgr.GetXnValue(rankId_, dieId_, xnId_));
    return detail;
}
