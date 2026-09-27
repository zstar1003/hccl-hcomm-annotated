/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "load_sqe_args_to_gsa_executor.h"

#include <cstdint>

#include "ccu_executor_manager.h"
#include "ccu_microcode_common_v1.h"
#include "sim_log.h"
#include "ccu_string_util.h"

using namespace std;
using namespace hcomm::CcuRep;

// 注册LoadSqeArgsToGsaExecutor create Func
REG_CCU_EXECUTOR_CREATE_FUNC(SimCcuV1::LOAD_TYPE, SimCcuV1::LOADSQEARGSTOGSA_CODE, LoadSqeArgsToGsaExecutor);

void LoadSqeArgsToGsaExecutor::Parser()
{
    ValidateVersionExclusive(RunnerCcuVersion::CCU_V1, "LoadSqeArgsToGsa");
    gsaId_ = instr_.v1.loadSqeArgsToXn.xnId;
    sqeArgId_ = instr_.v1.loadSqeArgsToXn.sqeArgsId;
}

void LoadSqeArgsToGsaExecutor::Run()
{
    // 加载sqe参数至xn寄存器，只需更新对应dev的ccu资源映射表即可
    auto& ccuResMgr = CcuResourceManager::GetInstance();
    uint64_t sqeArgValue = ccuResMgr.GetSqeArgValue(rankId_, dieId_, sqeArgId_);

    ccuResMgr.UpdateXnValue(rankId_, dieId_, gsaId_, sqeArgValue);
    HCCL_VM_DEBUG("Load SqeArg: locCcu[{}:{}], GSAId=[{}], sqeArgValue=[{}]", rankId_, dieId_, gsaId_, sqeArgValue);
}

std::string LoadSqeArgsToGsaExecutor::Describe()
{
    return HcclSim::StringFormat("[Simulation Execute] Load SqeArg[%u] to GSA[%u]\n", sqeArgId_, gsaId_);
}

CcuTrace::CcuInstrTraceDetail LoadSqeArgsToGsaExecutor::CollectTraceDetail()
{
    CcuTrace::CcuInstrTraceDetail detail;
    detail.typeName = "LoadSqeArgsToGsa";
    auto& ccuResMgr = CcuResourceManager::GetInstance();
    detail.args["sqeArgValue"] = std::to_string(ccuResMgr.GetSqeArgValue(rankId_, dieId_, sqeArgId_));
    return detail;
}
