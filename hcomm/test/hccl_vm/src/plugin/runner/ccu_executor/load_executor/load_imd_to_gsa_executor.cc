/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "load_imd_to_gsa_executor.h"

#include "ccu_executor_manager.h"
#include "ccu_microcode_common_v1.h"
#include "ccu_string_util.h"

using namespace std;
using namespace hcomm::CcuRep;

// 注册LoadImdToGSAExecutor create Func
REG_CCU_EXECUTOR_CREATE_FUNC(SimCcuV1::LOAD_TYPE, SimCcuV1::LOADIMDTOGSA_CODE, LoadImdToGSAExecutor);

void LoadImdToGSAExecutor::Parser()
{
    gsaId_ = instr_.v1.loadImdToGSA.gsaId;
    immediate_ = instr_.v1.loadImdToGSA.immediate;
}

void LoadImdToGSAExecutor::Run()
{
    auto& ccuResMgr = CcuResourceManager::GetInstance();
    ccuResMgr.UpdateGsaValue(rankId_, dieId_, gsaId_, immediate_);
}

std::string LoadImdToGSAExecutor::Describe()
{
    return HcclSim::StringFormat("[Simulation Execute] Load immediate[%lu] to GSA[%u]\n", immediate_, gsaId_);
}

CcuTrace::CcuInstrTraceDetail LoadImdToGSAExecutor::CollectTraceDetail()
{
    CcuTrace::CcuInstrTraceDetail detail;
    detail.typeName = "LoadImdToGSA";
    return detail;
}
