/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "load_sqe_args_to_xn_executor.h"

#include <cstdint>

#include "ccu_executor_manager.h"
#include "ccu_microcode_common_v1.h"
#include "sim_log.h"
#include "ccu_string_util.h"

using namespace std;
using namespace hcomm::CcuRep;

// 注册LoadSqeArgsToXnExecutor create Func
REG_CCU_EXECUTOR_CREATE_FUNC_V1(SimCcuV1::LOAD_TYPE, SimCcuV1::LOADSQEARGSTOXN_CODE, LoadSqeArgsToXnExecutor);
REG_CCU_EXECUTOR_CREATE_FUNC_V2(SimCcuV2::LOAD_TYPE, SimCcuV2::LOADSQEARGSTOXN_CODE, LoadSqeArgsToXnExecutor);

void LoadSqeArgsToXnExecutor::Parser()
{
    if (version_ == RunnerCcuVersion::CCU_V1) {
        xnId_ = instr_.v1.loadSqeArgsToXn.xnId;
        sqeArgId_ = instr_.v1.loadSqeArgsToXn.sqeArgsId;
    } else if (version_ == RunnerCcuVersion::CCU_V2) {
        sqeArgId_ = instr_.v2.loadSqeArgsToX.sqeArgsId;
        xnId_ = instr_.v2.loadSqeArgsToX.xnId;
        ckeId_ = instr_.v2.loadSqeArgsToX.setCKEId;
        ckeMask_ = instr_.v2.loadSqeArgsToX.setCKEMask;
    } else {
        HCCL_VM_ERROR("Unsupported CCU version: {}", RunnerCcuVersionToString(version_));
    }
}

void LoadSqeArgsToXnExecutor::Run()
{
    // 加载sqe参数至xn寄存器，只需更新对应dev的ccu资源映射表即可
    auto& ccuResMgr = CcuResourceManager::GetInstance();
    uint64_t sqeArgValue = ccuResMgr.GetSqeArgValue(rankId_, dieId_, sqeArgId_);

    HCCL_VM_DEBUG(
        "Load arg: locCcu[{}:{}], XnId=[{}], argId=[{}], value=[{}]", rankId_, dieId_, xnId_, sqeArgId_, sqeArgValue);
    ccuResMgr.UpdateXnValue(rankId_, dieId_, xnId_, sqeArgValue);
    if (version_ == RunnerCcuVersion::CCU_V2) {
        uint16_t ckeId = UpdateCkeId(ckeId_);
        SetCkeSignal(ccuResMgr, ckeId, ckeMask_);
    }
}

std::string LoadSqeArgsToXnExecutor::Describe()
{
    // Describe() 仅包含静态信息（指令 ID、目标寄存器），不含运行时 sqeArgValue
    return HcclSim::StringFormat(
        "[Simulation Execute] locCcu[%d:%d], Load SqeArg[%u] to Xn[%u]\n", rankId_, dieId_, sqeArgId_, xnId_);
}

CcuTrace::CcuInstrTraceDetail LoadSqeArgsToXnExecutor::CollectTraceDetail()
{
    CcuTrace::CcuInstrTraceDetail detail;
    detail.typeName = "LoadSqeArgsToXn";

    // 运行时动态参数：sqeArgValue 随 SQE 任务不同可能不同
    auto& ccuResMgr = CcuResourceManager::GetInstance();
    uint64_t sqeArgValue = ccuResMgr.GetSqeArgValue(rankId_, dieId_, sqeArgId_);

    detail.args["sqeArgValue"] = std::to_string(sqeArgValue);
    if (version_ == RunnerCcuVersion::CCU_V2) {
        detail.args["setCKEId"] = std::to_string(ckeId_);
        detail.args["setCKEMask"] = std::to_string(ckeMask_);
    }
    return detail;
}
