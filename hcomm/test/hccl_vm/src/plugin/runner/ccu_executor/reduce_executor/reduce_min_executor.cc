/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "reduce_min_executor.h"

#include <cstdint>
#include <cstring>

#include "ccu_executor_manager.h"
#include "ccu_microcode_common_v1.h"
#include "ccu_fp16.h"
#include "sim_log.h"
#include "ccu_reduce_operator.h"
#include "ccu_string_util.h"

using namespace std;
using namespace hcomm::CcuRep;

// 注册ReduceMinExecutor create Func
REG_CCU_EXECUTOR_CREATE_FUNC(SimCcuV1::REDUCE_TYPE, SimCcuV1::MIN_CODE, ReduceMinExecutor);
REG_CCU_EXECUTOR_CREATE_FUNC_V2(SimCcuV2::REDUCE_TYPE, SimCcuV2::REDUCEMIN_CODE, ReduceMinExecutor);

void ReduceMinExecutor::Parser()
{
    if (version_ == RunnerCcuVersion::CCU_V1) {
        count_ = instr_.v1.min.count;
        dataType_ = instr_.v1.min.dataType;
        clearType_ = instr_.v1.min.clearType;
        setCKEId_ = instr_.v1.min.setCKEId;
        setCKEMask_ = instr_.v1.min.setCKEMask;
        waitCKEId_ = instr_.v1.min.waitCKEId;
        waitCKEMask_ = instr_.v1.min.waitCKEMask;
        (void)memcpy(msId_, instr_.v1.min.msId, sizeof(uint16_t) * CCU_REDUCE_MAX_MS);
    } else if (version_ == RunnerCcuVersion::CCU_V2) {
        count_ = instr_.v2.reduce.count;
        dataType_ = instr_.v2.reduce.dataType;
        setCKEId_ = instr_.v2.reduce.setCKEId;
        setCKEMask_ = instr_.v2.reduce.setCKEMask;

        (void)memcpy(msId_, instr_.v2.reduce.msId, sizeof(uint16_t) * CCU_REDUCE_MAX_MS);
    } else {
        HCCL_VM_ERROR("Invalid ccu version:{}", RunnerCcuVersionToString(version_));
        ccuSimulator_->SetExecState(CcuExecState::EXEC_FAIL);
        return;
    }
}

// Reduce Min操作
void ReduceMinExecutor::Process(CcuResourceManager& ccuResMgr)
{
    HCCL_VM_DEBUG("Reduce Min info, locCcu[{}:{}], count:[{}], dataType:[{}]", rankId_, dieId_, count_, dataType_);
    if (dataType_ >= ReduceMaxMinDataType::MAX_MIN_RESERVED4 || dataType_ == ReduceMaxMinDataType::MAX_MIN_RESERVED1
        || dataType_ == ReduceMaxMinDataType::MAX_MIN_RESERVED2
        || dataType_ == ReduceMaxMinDataType::MAX_MIN_RESERVED3) {
        return;
    }
    for (uint32_t i = 0; i < CCU_REDUCE_MAX_MS; i++) {
        HCCL_VM_TRACE("msId_[{}]:dieId[{}], msId[{}]", i, msId_[i] >> 15, msId_[i] & 0x7FFF);
        msId_[i] = msId_[i] & 0x7FFF;
    }
    // 1. 判断是否在Loop循环内
    if (ccuSimulator_->GetState() == CcuExecState::EXEC_LOOP_INSTR) {
        uint16_t ckeOffset = ccuSimulator_->GetLoopCKEOffset();
        setCKEId_ += ckeOffset;
        auto msOffset = ccuSimulator_->GetLoopMsOffset();
        for (uint32_t i = 0; i < hcomm::CcuRep::CCU_REDUCE_MAX_MS; i++) {
            msId_[i] += msOffset;
        }
        HCCL_VM_DEBUG("locCcu[{}:{}], ms offset=[{}]", rankId_, dieId_, msOffset);
    }
    // 2. reduce操作
    ReduceMaxMinDataType type = static_cast<ReduceMaxMinDataType>(dataType_);
    auto res = reduceMinFuncMap.find(type);
    if (res != reduceMinFuncMap.end()) {
        res->second(rankId_, dieId_, msId_, count_);
    }
    // 3. 设置本端的cke
    SetCkeSignal(ccuResMgr, setCKEId_, setCKEMask_);
}

void ReduceMinExecutor::RunV1() { WaitCkeProcess(waitCKEId_, waitCKEMask_, clearType_, "ReduceMin"); }

void ReduceMinExecutor::RunV2()
{
    HCCL_VM_DEBUG("Reduce Min info, locCcu[{}:{}], count=[{}], dataType=[{}]", rankId_, dieId_, count_, dataType_);
    auto& ccuResMgr = CcuResourceManager::GetInstance();
    for (uint32_t i = 0; i < CCU_REDUCE_MAX_MS; i++) {
        HCCL_VM_TRACE("msId_[{}]:dieId[{}], msId[{}]", i, msId_[i] >> 15, msId_[i] & 0x7FFF);
        msId_[i] = UpdateMSId(msId_[i] & 0x7FFF);
    }
    if (dataType_ >= ReduceMaxMinDataType::MAX_MIN_RESERVED4 || dataType_ == ReduceMaxMinDataType::MAX_MIN_RESERVED1
        || dataType_ == ReduceMaxMinDataType::MAX_MIN_RESERVED2
        || dataType_ == ReduceMaxMinDataType::MAX_MIN_RESERVED3) {
        ccuSimulator_->SetExecState(CcuExecState::EXEC_FAIL);
        return;
    }
    // 2. reduce操作
    ReduceMaxMinDataType type = static_cast<ReduceMaxMinDataType>(dataType_);
    auto res = reduceMinFuncMap.find(type);
    if (res != reduceMinFuncMap.end()) {
        res->second(rankId_, dieId_, msId_, count_);
    }
    // 3.设置本端的cke
    uint16_t ckeId = UpdateCkeId(setCKEId_);
    SetCkeSignal(ccuResMgr, ckeId, setCKEMask_);
}

void ReduceMinExecutor::Run()
{
    if (version_ == RunnerCcuVersion::CCU_V1) {
        RunV1();
    } else if (version_ == RunnerCcuVersion::CCU_V2) {
        RunV2();
    } else {
        HCCL_VM_ERROR("Invalid ccu version:{}", RunnerCcuVersionToString(version_));
        ccuSimulator_->SetExecState(CcuExecState::EXEC_FAIL);
        return;
    }
}

std::string ReduceMinExecutor::Describe()
{
    return HcclSim::StringFormat(
        "[Simulation Execute] Wait CKE[%u:%04x], Min %s with Count[%u], DataType[%u] and "
        "CastEn[%u], Set CKE[%u:%04x], clearType[%u]\n",
        waitCKEId_, waitCKEMask_, ParseMSList().c_str(), count_, dataType_, setCKEId_, setCKEMask_, clearType_);
}

CcuTrace::CcuInstrTraceDetail ReduceMinExecutor::CollectTraceDetail()
{
    CcuTrace::CcuInstrTraceDetail detail;
    detail.typeName = "ReduceMin";
    detail.args["reduceOp"] = "Min";
    return detail;
}
