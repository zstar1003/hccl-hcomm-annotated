/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "load_gsa_gsa_executor.h"

#include <cstdint>

#include "ccu_executor_manager.h"
#include "ccu_microcode_common_v1.h"
#include "ccu_string_util.h"

using namespace std;
using namespace hcomm::CcuRep;

// 注册LoadGSAGSAExecutor create Func
REG_CCU_EXECUTOR_CREATE_FUNC(SimCcuV1::LOAD_TYPE, SimCcuV1::LOADGSAGSA_CODE, LoadGsaGsaExecutor);

void LoadGsaGsaExecutor::Parser()
{
    ValidateVersionExclusive(RunnerCcuVersion::CCU_V1, "LoadGsaGsa");
    gsAdId_ = instr_.v1.loadGSAGSA.gsAdId;
    gsAmId_ = instr_.v1.loadGSAGSA.gsAmId;
    gsAnId_ = instr_.v1.loadGSAGSA.gsAnId;
}

void LoadGsaGsaExecutor::Run()
{
    auto& ccuResMgr = CcuResourceManager::GetInstance();
    uint64_t gsa1 = ccuResMgr.GetGsaValue(rankId_, dieId_, gsAmId_);
    uint64_t gsa2 = ccuResMgr.GetGsaValue(rankId_, dieId_, gsAnId_);
    uint64_t val = gsa1 + gsa2;
    ccuResMgr.UpdateGsaValue(rankId_, dieId_, gsAdId_, val);
}

std::string LoadGsaGsaExecutor::Describe()
{
    return HcclSim::StringFormat("[Simulation Execute] Load GSA[%u] + GSA[%u] to GSA[%u]\n", gsAmId_, gsAnId_, gsAdId_);
}

CcuTrace::CcuInstrTraceDetail LoadGsaGsaExecutor::CollectTraceDetail()
{
    CcuTrace::CcuInstrTraceDetail detail;
    detail.typeName = "LoadGsaGsa";
    auto& ccuResMgr = CcuResourceManager::GetInstance();
    detail.args["gsa1"] = std::to_string(ccuResMgr.GetGsaValue(rankId_, dieId_, gsAmId_));
    detail.args["gsa2"] = std::to_string(ccuResMgr.GetGsaValue(rankId_, dieId_, gsAnId_));
    return detail;
}
