/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_executor_base.h"

#include <cstdint>

#include "ccu_resource_common.h"
#include "ccu_resource_manager.h"
#include "sim_log.h"

std::map<std::string, uint32_t> CcuExecutorBase::s_blockingCountMap_;

std::string CcuExecutorBase::ParseMSList()
{
    // 待实现，检查sqe类型
    uint16_t msId[hcomm::CcuRep::CCU_REDUCE_MAX_MS];
    uint16_t count = instr_.v1.add.count;
    for (uint16_t index = 0; index < hcomm::CcuRep::CCU_REDUCE_MAX_MS; index++) {
        msId[index] = instr_.v1.add.msId[index];
    }

    std::string res = "MS[";
    for (uint16_t i = 0; i < count + 2; i++) { // 循环范围 0~count + 2
        if (i == count + 1) {
            res += std::to_string(msId[i] / 0x8000) + ":" + std::to_string(msId[i] % 0x8000) + "]";
        } else {
            res += std::to_string(msId[i] / 0x8000) + ":" + std::to_string(msId[i] % 0x8000) + ", ";
        }
    }
    return res;
}

void CcuExecutorBase::SetCkeSignal(CcuResourceManager& ccuResMgr, uint16_t setCKEId, uint16_t setCKEMask)
{
    uint16_t setCKE = ccuResMgr.GetCkeValue(rankId_, dieId_, setCKEId);
    uint16_t newSetCKE = (setCKE & (~setCKEMask)) | setCKEMask;
    ccuResMgr.UpdateCkeValue(rankId_, dieId_, setCKEId, newSetCKE);
    HCCL_VM_DEBUG(
        "success, locCcu[{}:{}], SetCKE[{}:{:04x}], value[{} --> {}]", rankId_, dieId_, setCKEId, setCKEMask, setCKE,
        newSetCKE);
}

void CcuExecutorBase::SetRmtCKESignal(
    CcuResourceManager& ccuResMgr, int rmtRank, int rmtDie, uint16_t setRmtCKEId, uint16_t setRmtCKEMask)
{
    uint16_t rmtCKE = ccuResMgr.GetCkeValue(rmtRank, rmtDie, setRmtCKEId);
    uint16_t newRmtCKE = (rmtCKE & (~setRmtCKEMask)) | setRmtCKEMask;
    ccuResMgr.UpdateCkeValue(rmtRank, rmtDie, setRmtCKEId, newRmtCKE);
    HCCL_VM_DEBUG(
        "success, ccu[{}:{} --> {}:{}], SetRmtCKE[{}:{:04x}], value[{} --> {}]", rankId_, dieId_, rmtRank, rmtDie,
        setRmtCKEId, setRmtCKEMask, rmtCKE, newRmtCKE);
}

void CcuExecutorBase::ClearCkeSignal(CcuResourceManager& ccuResMgr, uint16_t clearCKEId, uint16_t clearMask)
{
    // 判断是否在Loop循环内GSA地址需偏移
    if (ccuSimulator_->GetState() == CcuExecState::EXEC_LOOP_INSTR) {
        auto ckeOffset = ccuSimulator_->GetLoopCKEOffset();
        clearCKEId += ckeOffset;
        HCCL_VM_DEBUG("locCcu[{}:{}], Get cke id offset = [{}]", rankId_, dieId_, ckeOffset);
    }
    uint16_t setCKE = ccuResMgr.GetCkeValue(rankId_, dieId_, clearCKEId);
    uint16_t newSetCKE = setCKE & (~clearMask);
    ccuResMgr.UpdateCkeValue(rankId_, dieId_, clearCKEId, newSetCKE);
    HCCL_VM_DEBUG(
        "success, locCcu[{}:{}], ClearCKE[{}:{:04x}], value[{} --> {}]", rankId_, dieId_, clearCKEId, clearMask, setCKE,
        newSetCKE);
}

uint16_t CcuExecutorBase::UpdateXnId(uint16_t xnIdField)
{
    if (ccuSimulator_->GetState() == CcuExecState::EXEC_LOOP_INSTR) {
        auto xnIdOffset = ccuSimulator_->GetLoopXnIdOffset();
        HCCL_VM_DEBUG(", locCcu[{}:{}], Get cke id offset = [{:x}]", rankId_, dieId_, xnIdOffset);
        return xnIdField + xnIdOffset;
    }
    return xnIdField;
}

uint16_t CcuExecutorBase::UpdateCkeId(uint16_t ckeId)
{
    if (ccuSimulator_->GetState() == CcuExecState::EXEC_LOOP_INSTR) {
        auto ckeOffset = ccuSimulator_->GetLoopCKEOffset();
        HCCL_VM_DEBUG(", locCcu[{}:{}], Get cke id offset = [{:x}]", rankId_, dieId_, ckeOffset);
        return ckeId + ckeOffset;
    }
    return ckeId;
}

uint16_t CcuExecutorBase::UpdateMSId(uint16_t msId)
{
    if (ccuSimulator_->GetState() == CcuExecState::EXEC_LOOP_INSTR) {
        auto ckeOffset = ccuSimulator_->GetLoopMsOffset();
        HCCL_VM_DEBUG(", locCcu[{}:{}], Get cke id offset = [{:x}]", rankId_, dieId_, ckeOffset);
        return msId + ckeOffset;
    }
    return msId;
}

uint16_t CcuExecutorBase::GetXnId(uint16_t xnIdField)
{
    uint16_t xnIdMode = xnIdField & 0x8000;
    uint16_t xnIdOri = xnIdField & 0x7FFF;
    return (xnIdMode == 0) ? xnIdOri : UpdateXnId(xnIdOri);
}

uint64_t CcuExecutorBase::UpdateAddressWithoutStride(uint64_t addr)
{
    return addr + ccuSimulator_->GetCurLoopCnt() * ccuSimulator_->GetLoopIterStepGSA();
}

uint64_t CcuExecutorBase::UpdateAddress(uint64_t addr, uint16_t addrExpandCoef)
{
    return addr + ((ccuSimulator_->GetLoopExtendNum() * ccuSimulator_->GetGSAOffset()) << addrExpandCoef)
           + ((ccuSimulator_->GetCurLoopCnt() * ccuSimulator_->GetLoopIterStepGSA()) << addrExpandCoef);
}

void CcuExecutorBase::WaitCkeProcess(
    uint16_t waitCKEId, uint16_t waitCKEMask, uint16_t clearType, const std::string& instrName)
{
    auto currentTime = std::chrono::steady_clock::now();
    static auto oldTime = currentTime;
    auto& ccuResMgr = CcuResourceManager::GetInstance();
    // 判断是否在Loop循环内GSA地址需偏移
    auto newWaitCKEId = waitCKEId;
    if (ccuSimulator_->GetState() == CcuExecState::EXEC_LOOP_INSTR) {
        auto ckeOffset = ccuSimulator_->GetLoopCKEOffset();
        newWaitCKEId += ckeOffset;
        HCCL_VM_DEBUG("instr[{}], locCcu[{}:{}], Get cke id offset = [{:x}]", instrName, rankId_, dieId_, ckeOffset);
    }
    auto waitCKE = ccuResMgr.GetCkeValue(rankId_, dieId_, newWaitCKEId);
    if (waitCKEMask != 0) {
        if ((waitCKE & waitCKEMask) == waitCKEMask) {
            // 指令具体操作
            Process(ccuResMgr);
        } else {
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(currentTime - oldTime);
            if (duration.count() >= 1000) {
                HCCL_VM_DEBUG(
                    "instr[{}], ccuId=[{}:{}], waitCKE[{}:{:04x}], expect:[{:04x}], actual:[{:04x}], waitCKE:[{:04x}]",
                    instrName, rankId_, dieId_, newWaitCKEId, waitCKEMask, waitCKEMask, (waitCKE & waitCKEMask),
                    waitCKE);
                oldTime = currentTime;
            }
            ccuSimulator_->SetWaitCKEFlag(true);
            return;
        }
    } else {
        // 指令具体操作
        Process(ccuResMgr);
    }
    ccuSimulator_->SetWaitCKEFlag(false);
    // 执行完成后，清除本端cke
    if (clearType == 1) {
        ClearCkeSignal(ccuResMgr, waitCKEId, waitCKEMask);
    }
}
