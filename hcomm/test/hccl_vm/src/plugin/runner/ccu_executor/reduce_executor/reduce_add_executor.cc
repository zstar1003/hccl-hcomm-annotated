/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "reduce_add_executor.h"

#include <cstdint>
#include <cstring>
#include <map>

#include "ccu_executor_manager.h"
#include "ccu_microcode_common_v1.h"
#include "sim_log.h"
#include "ccu_reduce_operator.h"
#include "ccu_simulator_base.h"
#include "ccu_string_util.h"

using namespace std;
using namespace hcomm::CcuRep;

// 注册ReduceAddExecutor create Func
REG_CCU_EXECUTOR_CREATE_FUNC(SimCcuV1::REDUCE_TYPE, SimCcuV1::ADD_CODE, ReduceAddExecutor);
REG_CCU_EXECUTOR_CREATE_FUNC_V2(SimCcuV2::REDUCE_TYPE, SimCcuV2::REDUCEADD_CODE, ReduceAddExecutor);

void ReduceAddExecutor::Parser()
{
    if (version_ == RunnerCcuVersion::CCU_V1) {
        count_ = instr_.v1.add.count;
        castEn_ = instr_.v1.add.castEn;
        dataType_ = instr_.v1.add.dataType;
        clearType_ = instr_.v1.add.clearType;
        setCKEId_ = instr_.v1.add.setCKEId;
        setCKEMask_ = instr_.v1.add.setCKEMask;
        waitCKEId_ = instr_.v1.add.waitCKEId;
        waitCKEMask_ = instr_.v1.add.waitCKEMask;
        (void)memcpy(msId_, instr_.v1.add.msId, sizeof(uint16_t) * CCU_REDUCE_MAX_MS);
    } else if (version_ == RunnerCcuVersion::CCU_V2) {
        count_ = instr_.v2.reduce.count;
        castEn_ = instr_.v2.reduce.castEn;
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

// Reduce Add操作
void ReduceAddExecutor::Process(CcuResourceManager& ccuResMgr)
{
    HCCL_VM_DEBUG(
        "Reduce Add info, locCcu[{}:{}], count=[{}], castEn=[{}], dataType=[{}]", rankId_, dieId_, count_, castEn_,
        dataType_);
    for (uint32_t i = 0; i < CCU_REDUCE_MAX_MS; i++) {
        HCCL_VM_TRACE("msId_[{}]:dieId[{}], msId[{}]", i, msId_[i] >> 15, msId_[i] & 0x7FFF);
        msId_[i] = msId_[i] & 0x7FFF;
    }
    if (dataType_ >= ReduceAddDataType::ADD_RESERVED) {
        return;
    }
    // 1.判断是否在Loop循环内
    if (ccuSimulator_->GetState() == CcuExecState::EXEC_LOOP_INSTR) {
        uint16_t ckeOffset = ccuSimulator_->GetLoopCKEOffset();
        setCKEId_ += ckeOffset;
        auto msOffset = ccuSimulator_->GetLoopMsOffset();
        for (uint32_t i = 0; i < hcomm::CcuRep::CCU_REDUCE_MAX_MS; i++) {
            msId_[i] += msOffset;
        }
        HCCL_VM_DEBUG("locCcu[{}:{}], ms offset=[{}], ckeOffset[{}]", rankId_, dieId_, msOffset, ckeOffset);
    }
    // 2. reduce操作
    ReduceAddDataType type = static_cast<ReduceAddDataType>(dataType_);
    HCCL_VM_INFO("ReduceAddExecutor::Process type {} count {} castEn {}", static_cast<uint16_t>(type), count_, castEn_);
    auto res = reduceAddFuncMap.find(type);
    if (res != reduceAddFuncMap.end()) {
        res->second(rankId_, dieId_, msId_, count_, castEn_);
    }
    // 3.设置本端的cke
    SetCkeSignal(ccuResMgr, setCKEId_, setCKEMask_);
}

void ReduceAddExecutor::RunV2()
{
    HCCL_VM_DEBUG(
        "Reduce Add info, locCcu[{}:{}], count=[{}], castEn=[{}], dataType=[{}]", rankId_, dieId_, count_, castEn_,
        dataType_);
    auto& ccuResMgr = CcuResourceManager::GetInstance();
    for (uint32_t i = 0; i < CCU_REDUCE_MAX_MS; i++) {
        HCCL_VM_TRACE("msId_[{}]:dieId[{}], msId[{}]", i, msId_[i] >> 15, msId_[i] & 0x7FFF);
        msId_[i] = UpdateMSId(msId_[i] & 0x7FFF);
    }
    if (dataType_ >= ReduceAddDataType::ADD_RESERVED) {
        ccuSimulator_->SetExecState(CcuExecState::EXEC_FAIL);
        return;
    }
    // 2. reduce操作
    ReduceAddDataType type = static_cast<ReduceAddDataType>(dataType_);
    auto res = reduceAddFuncMap.find(type);
    if (res != reduceAddFuncMap.end()) {
        res->second(rankId_, dieId_, msId_, count_, castEn_);
    }
    // 3.设置本端的cke
    uint16_t ckeId = UpdateCkeId(setCKEId_);
    SetCkeSignal(ccuResMgr, ckeId, setCKEMask_);
}

void ReduceAddExecutor::RunV1() { WaitCkeProcess(waitCKEId_, waitCKEMask_, clearType_, "ReduceAdd"); }

void ReduceAddExecutor::Run()
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

std::string ReduceAddExecutor::Describe()
{
    return HcclSim::StringFormat(
        "[Simulation Execute] Wait CKE[%u:%04x], Add %s with Count[%u], DataType[%u] and "
        "CastEn[%u], Set CKE[%u:%04x], clearType[%u]\n",
        waitCKEId_, waitCKEMask_, ParseMSList().c_str(), count_, dataType_, castEn_, setCKEId_, setCKEMask_,
        clearType_);
}

CcuTrace::CcuInstrTraceDetail ReduceAddExecutor::CollectTraceDetail()
{
    CcuTrace::CcuInstrTraceDetail detail;
    detail.typeName = "ReduceAdd";
    detail.args["reduceOp"] = "Add";
    return detail;
}
