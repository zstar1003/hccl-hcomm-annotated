/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "fence_executor.h"

#include "ccu_executor_manager.h"
#include "ccu_string_util.h"

using namespace std;
using namespace hcomm::CcuRep;

// 注册NopExecutor create Func
REG_CCU_EXECUTOR_CREATE_FUNC_V2(SimCcuV2::CTRL_TYPE, SimCcuV2::FENCE_CODE, FenceExecutor);

void FenceExecutor::Parser() { ValidateVersionExclusive(RunnerCcuVersion::CCU_V2, "FenceExecutor"); }

void FenceExecutor::Process(CcuResourceManager& ccuResMgr) { (void)ccuResMgr; }

void FenceExecutor::Run()
{
    HCCL_VM_ERROR("Unsupported ");
    ccuSimulator_->SetExecState(CcuExecState::EXEC_FAIL);
}

std::string FenceExecutor::Describe()
{
    return HcclSim::StringFormat("[FenceExecutor] V2 Fence instruction, no additional parameters\n");
}

CcuTrace::CcuInstrTraceDetail FenceExecutor::CollectTraceDetail()
{
    CcuTrace::CcuInstrTraceDetail detail;
    detail.typeName = "Fence";
    return detail;
}
