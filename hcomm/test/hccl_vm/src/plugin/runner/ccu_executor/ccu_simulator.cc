/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_simulator.h"

#include <cstdint>

#include "ccu_executor_manager.h"
#include "ccu_resource_manager.h"
#include "ccu_trace_collector.h"
#include "sim_log.h"

using namespace std;
using namespace hcomm::CcuRep;

void CcuSimulator::InitLoopGroupInfo(uint16_t startLoopId, uint64_t offsetCfg, uint64_t repeatCfg)
{
    state_ = CcuExecState::EXEC_LOOPGROUP_INSTR;
    loopGroupInfo_.startLoopId_ = startLoopId;
    loopGroupInfo_.loopNum_ = (repeatCfg >> 41) & 0x7F;       // 0x7F: 取loop指令个数，即[47:41]位
    loopGroupInfo_.loopOffset_ = (repeatCfg >> 48) & 0x7F;    // 0x7F: 取loop偏移，即[54:48]位
    loopGroupInfo_.loopExtendNum_ = (repeatCfg >> 55) & 0x7F; // 0x7F: 取loop展开次数，即[61:55]位
    HCCL_VM_DEBUG(
        "locCcu[{}:{}], curInstrId_=[{}], loopNum=[{}], loopOffset=[{}], loopExtendNum=[{}]", rankId_, dieId_,
        curInstrId_, loopGroupInfo_.loopNum_, loopGroupInfo_.loopOffset_, loopGroupInfo_.loopExtendNum_);

    loopGroupInfo_.ckeOffset_ = offsetCfg & 0x3FF;              // 0x3FF: 取低[9:0]位
    loopGroupInfo_.msOffset_ = (offsetCfg >> 10) & 0x7FF;       // 0x7FF: 取低[20:10]位
    loopGroupInfo_.gsaOffset_ = (offsetCfg >> 21) & 0xFFFFFFFF; // 0xFFFFFFFF: 取低[52:21]位
    HCCL_VM_DEBUG(
        "locCcu[{}:{}], ckeOffset=[{}], msOffset=[{}], gsaOffset=[{}]", rankId_, dieId_, loopGroupInfo_.ckeOffset_,
        loopGroupInfo_.msOffset_, loopGroupInfo_.gsaOffset_);
}

void CcuSimulator::InitLoopGroupInfoV2(uint16_t startLoopId, uint64_t xnValue, uint64_t xmValue, uint64_t xpValue)
{
    state_ = CcuExecState::EXEC_LOOPGROUP_INSTR;
    loopGroupInfo_.startLoopId_ = startLoopId;
    loopGroupInfo_.loopNum_ = xnValue & 0x7F;                // 0x7F: 取loop指令个数，即[9:0]位
    loopGroupInfo_.loopOffset_ = (xnValue >> 10) & 0x1FF;    // 0x1FF: 取loop偏移，即[18:10]位
    loopGroupInfo_.loopExtendNum_ = (xnValue >> 19) & 0x1FF; // 0x7F: 取loop展开次数，即[27:19]位
    HCCL_VM_DEBUG(
        "locCcu[{}:{}], curInstrId_=[{}], loopNum=[{}], loopOffset=[{}], loopExtendNum=[{}]", rankId_, dieId_,
        curInstrId_, loopGroupInfo_.loopNum_, loopGroupInfo_.loopOffset_, loopGroupInfo_.loopExtendNum_);

    loopGroupInfo_.ckeOffset_ = xmValue & 0x3FF;              // 0x3FF: 取低[9:0]位
    loopGroupInfo_.msOffset_ = (xmValue >> 10) & 0x7FF;       // 0x7FF: 取低[20:10]位
    loopGroupInfo_.gsaOffset_ = (xmValue >> 21) & 0xFFFFFFFF; // 0xFFFFFFFF: 取低[52:21]位
    loopGroupInfo_.xnIdOffset = xpValue & 0xFFFFFFFF;
    HCCL_VM_DEBUG(
        "locCcu[{}:{}], ckeOffset=[{}], msOffset=[{}], gsaOffset=[{}], xnIdOffset=[{}]", rankId_, dieId_,
        loopGroupInfo_.ckeOffset_, loopGroupInfo_.msOffset_, loopGroupInfo_.gsaOffset_, loopGroupInfo_.xnIdOffset);
}

void CcuSimulator::InitLoopInfo(uint16_t startInstrId, uint16_t endInstrId, uint16_t execCount, uint32_t addrStep)
{
    loopGroupInfo_.loopStatus_.loopStartInstrId = startInstrId;
    loopGroupInfo_.loopStatus_.loopCurInstrId = startInstrId;
    loopGroupInfo_.loopStatus_.loopEndInstrId = endInstrId;
    loopGroupInfo_.loopStatus_.loopExecCount = execCount;
    loopGroupInfo_.loopStatus_.curLoopRound = 0;
    loopGroupInfo_.loopStatus_.loopGsaIterStep = addrStep;

    HCCL_VM_DEBUG(
        "locCcu[{}:{}], loopStartInstrId=[{}], loopCurInstrId=[{}], "
        "loopEndInstrId=[{}], loopExecCount=[{}], loopGsaIterStep=[{}]",
        rankId_, dieId_, startInstrId, startInstrId, endInstrId, execCount, addrStep);
}

void CcuSimulator::InitLoopInfoV2(
    uint16_t startInstrId, uint16_t endInstrId, uint64_t xnValue, uint64_t xmValue, uint64_t xpValue)
{
    loopGroupInfo_.loopStatus_.loopStartInstrId = startInstrId;
    loopGroupInfo_.loopStatus_.loopCurInstrId = startInstrId;
    loopGroupInfo_.loopStatus_.loopEndInstrId = endInstrId;
    loopGroupInfo_.loopStatus_.loopExecCount = (xmValue & 0x1FFFF); // 循环次数
    loopGroupInfo_.loopStatus_.curLoopRound = 0;
    loopGroupInfo_.loopStatus_.loopGsaIterStep = (xnValue & 0xFFFFFFFF);
    loopGroupInfo_.loopStatus_.loopCtxId = (xpValue & 0x1FF);

    HCCL_VM_DEBUG(
        "locCcu[{}:{}], loopStartInstrId=[{}], loopCurInstrId=[{}], "
        "loopEndInstrId=[{}], xnValue=[{}], xpValue=[{}], xmValue=[{}]",
        rankId_, dieId_, startInstrId, startInstrId, endInstrId, xnValue, xpValue, xmValue);
}

void CcuSimulator::InitLoopGroupInfo(const LoopGroupInfo& loopGroupInfo) { loopGroupInfo_ = loopGroupInfo; }

uint64_t CcuSimulator::GetLoopGsaAddrOffset()
{
    if (state_ != CcuExecState::EXEC_LOOP_INSTR) {
        return 0;
    }
    uint64_t addrOffset = loopGroupInfo_.gsaOffset_ * loopGroupInfo_.loopStatus_.loopExtendIndex
                          + loopGroupInfo_.loopStatus_.loopGsaIterStep * loopGroupInfo_.loopStatus_.curLoopRound;
    return addrOffset;
}

uint16_t CcuSimulator::GetLoopMsOffset()
{
    if (state_ != CcuExecState::EXEC_LOOP_INSTR) {
        return 0;
    }
    return loopGroupInfo_.msOffset_ * loopGroupInfo_.loopStatus_.loopExtendIndex;
}

uint16_t CcuSimulator::GetLoopCKEOffset()
{
    if (state_ != CcuExecState::EXEC_LOOP_INSTR) {
        return 0;
    }
    return loopGroupInfo_.ckeOffset_ * loopGroupInfo_.loopStatus_.loopExtendIndex;
}

uint16_t CcuSimulator::GetLoopXnIdOffset()
{
    if (state_ != CcuExecState::EXEC_LOOP_INSTR) {
        return 0;
    }
    return loopGroupInfo_.xnIdOffset * loopGroupInfo_.loopStatus_.loopExtendIndex;
}

uint16_t CcuSimulator::GetCurLoopCnt()
{
    if (state_ != CcuExecState::EXEC_LOOP_INSTR) {
        return 0;
    }
    return loopGroupInfo_.loopStatus_.curLoopRound;
}

uint32_t CcuSimulator::GetLoopIterStepGSA()
{
    if (state_ != CcuExecState::EXEC_LOOP_INSTR) {
        return 0;
    }
    return loopGroupInfo_.loopStatus_.loopGsaIterStep;
}

uint16_t CcuSimulator::GetLoopExtendNum()
{
    if (state_ != CcuExecState::EXEC_LOOP_INSTR) {
        return 0;
    }
    return loopGroupInfo_.loopStatus_.loopExtendIndex;
}

uint32_t CcuSimulator::GetGSAOffset()
{
    if (state_ != CcuExecState::EXEC_LOOP_INSTR) {
        return 0;
    }
    return loopGroupInfo_.gsaOffset_;
}

void CcuSimulator::SetExecState(CcuExecState state) { state_ = state; }

bool CcuSimulator::UpdateLoopStatus()
{
    if (waitCKE_) {
        // 如果是等待CKE的状态，那么就不更新状态，下次执行的时候还是等待CKE
        return false;
    }
    if (state_ == CcuExecState::EXEC_FAIL) {
        return false;
    }
    if (state_ == CcuExecState::EXEC_LOOPGROUP_INSTR) {
        return true;
    }
    if (state_ == CcuExecState::EXEC_NORMAL_INSTR || state_ == CcuExecState::EXEC_LOOP_INSTR) {
        curInstrId_++;
        if (curInstrId_ == endInstrId_) {
            state_ = CcuExecState::EXEC_SUCCESS;
            return true;
        }
    } else if (state_ == CcuExecState::EXEC_JUMP_INSTR) {
        curInstrId_ = jumpInstrId_;
        state_ = CcuExecState::EXEC_NORMAL_INSTR;
    }
    return true;
}

void CcuSimulator::InitJumpStatus(uint16_t jumpInstrId)
{
    state_ = CcuExecState::EXEC_JUMP_INSTR;
    jumpInstrId_ = jumpInstrId;
}

// 参数超过13个时，分为多个SQE任务下发：新SQE任务执行前，须初始化模拟器参数。
void CcuSimulator::Init(uint16_t startInstrId, uint16_t endInstrId, uint16_t instrCnt, RunnerCcuVersion version)
{
    if (finished_ == false) {
        return;
    }
    finished_ = false;
    startInstrId_ = startInstrId;
    endInstrId_ = endInstrId;
    instrCnt_ = instrCnt;
    curInstrId_ = startInstrId;
    version_ = version;
    state_ = CcuExecState::EXEC_NORMAL_INSTR;
    waitCKE_ = false;
    jumpInstrId_ = 0;
    instrType_ = 0;
    loopGroupInfo_ = {};
}

CcuExecState CcuSimulator::GetState() { return state_; }

void CcuSimulator::SetWaitCKEFlag(bool needCKE) { waitCKE_ = needCKE; }

bool CcuSimulator::ExecuteInstr(uint16_t curInstrId)
{
    auto& ccuResMgr = CcuResourceManager::GetInstance();
    auto instrData = ccuResMgr.GetInstrData(rankId_, dieId_);
    auto& traceCollector = CcuTrace::CcuTraceCollector::GetInstance();

    HCCL_VM_DEBUG(
        "locCcu[{}:{}], current instr[{}], state=[{}]", rankId_, dieId_, curInstrId, static_cast<int>(state_));
    if (curInstrId >= endInstrId_) {
        state_ = CcuExecState::EXEC_FAIL;
        HCCL_VM_ERROR(
            "locCcu[{}:{}], Invalid curInstrId_[{}], endInstrId[{}]", rankId_, dieId_, curInstrId, endInstrId_);
        return false;
    }
    auto executor = CcuExecutorFactory::MakeCcuExecutorInstance(
        version_, instrData[curInstrId].header.header, 0, rankId_, dieId_, instrData[curInstrId], this);
    if (executor == nullptr) {
        UpdateLoopStatus();
        HCCL_VM_DEBUG("locCcu[{}:{}], curInstrId_[{}] executor is null.", rankId_, dieId_, curInstrId);
        return true; // 让还未添加的Executor先跑下去，看下是否能从start->loop->end
    }

    // === Trace: 设置当前执行 CCU ===
    traceCollector.BeginGlobalStep();
    traceCollector.SetCurrentExecutingCcu(rankId_, dieId_);

    // === Trace: 资源快照（执行前） ===
    std::vector<uint64_t> xnBefore, xnAfter;
    std::vector<uint64_t> gsaBefore, gsaAfter;
    std::vector<uint16_t> ckeBefore, ckeAfter;
    bool captureDelta = traceCollector.IsEnabled();
    if (captureDelta) {
        xnBefore.resize(SimCcuV1::CCU_RESOURCE_XN_NUM);
        gsaBefore.resize(SimCcuV1::CCU_RESOURCE_GSA_NUM);
        ckeBefore.resize(SimCcuV1::CCU_RESOURCE_CKE_NUM);
        for (uint16_t i = 0; i < SimCcuV1::CCU_RESOURCE_XN_NUM; i++) {
            xnBefore[i] = ccuResMgr.GetXnValue(rankId_, dieId_, i);
        }
        for (uint16_t i = 0; i < SimCcuV1::CCU_RESOURCE_GSA_NUM; i++) {
            gsaBefore[i] = ccuResMgr.GetGsaValue(rankId_, dieId_, i);
        }
        for (uint16_t i = 0; i < SimCcuV1::CCU_RESOURCE_CKE_NUM; i++) {
            ckeBefore[i] = ccuResMgr.GetCkeValue(rankId_, dieId_, i);
        }
    }

    executor->Parser();
    executor->Run();

    // === Trace: CKE Wait 自旋检测与合并 ===
    if (waitCKE_) {
        // CKE 条件不满足，记录一次自旋，不产生 trace entry
        // waitCKEId/waitCKEMask 由 WaitCkeProcess 设置，此处取不到精确值，传 0
        traceCollector.RecordWaitSpin(rankId_, dieId_, curInstrId, 0, 0, 0);
        if (UpdateLoopStatus() == false) {
            return false;
        }
        return true;
    }

    // === Trace: CKE 通过（或本条指令不涉及 CKE 等待），采集并记录 trace entry ===
    if (traceCollector.IsEnabled()) {
        CcuTrace::CcuTraceEntry entry;

        // 全局定位信息
        auto globalCtx = traceCollector.GetCurrentGlobalContext();
        entry.globalSeqId = globalCtx.globalSeqId;
        entry.execRound = globalCtx.execRound;
        entry.sqeTaskId = globalCtx.currentSqeTaskId;

        // CCU 归属 + 指令索引（三元组索引指令空间）
        entry.rankId = rankId_;
        entry.dieId = dieId_;
        entry.instrId = curInstrId;

        // 指令类别（仅用于统计聚合）
        uint16_t instrType = instrData[curInstrId].header.header & 0x3; // 低 2 位为类型
        switch (instrType) {
            case 0:
                entry.category = CcuTrace::CcuInstrCategory::LOAD;
                break;
            case 1:
                entry.category = CcuTrace::CcuInstrCategory::CONTROL;
                break;
            case 2:
                entry.category = CcuTrace::CcuInstrCategory::TRANS;
                break;
            case 3:
                entry.category = CcuTrace::CcuInstrCategory::REDUCE;
                break;
            default:
                entry.category = CcuTrace::CcuInstrCategory::UNKNOWN;
                break;
        }

        // 执行状态
        entry.execState = state_;

        // 执行上下文（Loop 偏移信息）
        entry.context.inLoop = (state_ == CcuExecState::EXEC_LOOP_INSTR);
        entry.context.loopRound = GetCurLoopCnt();
        entry.context.loopExtendIndex = GetLoopExtendNum();
        entry.context.gsaOffset = GetGSAOffset();
        entry.context.iterStepGSA = GetLoopIterStepGSA();
        entry.context.loopExtendNum = GetLoopExtendNum();
        entry.context.curLoopCnt = GetCurLoopCnt();
        entry.context.gsaAddrOffset = GetLoopGsaAddrOffset();
        entry.context.msOffset = GetLoopMsOffset();
        entry.context.ckeOffset = GetLoopCKEOffset();
        entry.context.xnIdOffset = GetLoopXnIdOffset();

        // 跨 CCU 变更（由 ResourceManager 拦截填充）
        entry.crossCcuChanges = traceCollector.ConsumeCrossCcuChanges(rankId_, dieId_);

        // CKE Wait 自旋信息（合并之前的自旋）
        entry.waitInfo = traceCollector.FinalizeWaitInfo(rankId_, dieId_, curInstrId, 0);

        // 指令专属细节
        entry.detail = executor->CollectTraceDetail();

        // === Trace: 资源变化（执行后快照 + diff） ===
        if (captureDelta) {
            xnAfter.resize(SimCcuV1::CCU_RESOURCE_XN_NUM);
            gsaAfter.resize(SimCcuV1::CCU_RESOURCE_GSA_NUM);
            ckeAfter.resize(SimCcuV1::CCU_RESOURCE_CKE_NUM);
            for (uint16_t i = 0; i < SimCcuV1::CCU_RESOURCE_XN_NUM; i++) {
                xnAfter[i] = ccuResMgr.GetXnValue(rankId_, dieId_, i);
                if (xnAfter[i] != xnBefore[i]) {
                    CcuTrace::CcuXnChange change;
                    change.id = i;
                    change.valueBefore = xnBefore[i];
                    change.valueAfter = xnAfter[i];
                    entry.resourceDelta.xnChanges.push_back(change);
                }
            }
            for (uint16_t i = 0; i < SimCcuV1::CCU_RESOURCE_GSA_NUM; i++) {
                gsaAfter[i] = ccuResMgr.GetGsaValue(rankId_, dieId_, i);
                if (gsaAfter[i] != gsaBefore[i]) {
                    CcuTrace::CcuGsaChange change;
                    change.id = i;
                    change.valueBefore = gsaBefore[i];
                    change.valueAfter = gsaAfter[i];
                    entry.resourceDelta.gsaChanges.push_back(change);
                }
            }
            for (uint16_t i = 0; i < SimCcuV1::CCU_RESOURCE_CKE_NUM; i++) {
                ckeAfter[i] = ccuResMgr.GetCkeValue(rankId_, dieId_, i);
                if (ckeAfter[i] != ckeBefore[i]) {
                    CcuTrace::CcuCkeChange change;
                    change.id = i;
                    change.valueBefore = ckeBefore[i];
                    change.valueAfter = ckeAfter[i];
                    entry.resourceDelta.ckeChanges.push_back(change);
                }
            }
        }

        traceCollector.RecordEntry(entry);
    }

    if (UpdateLoopStatus() == false) {
        return false;
    }

    HCCL_VM_DEBUG("locCcu[{}:{}], end instr[{}], {}", rankId_, dieId_, curInstrId, executor->Describe());
    return true;
}

// loop循环内指令执行，后续改为多线程并行执行
bool CcuSimulator::ExecuteLoop()
{
    auto instrCnt = loopGroupInfo_.loopStatus_.loopEndInstrId - loopGroupInfo_.loopStatus_.loopStartInstrId + 1;
    for (uint32_t i = 0; i < loopGroupInfo_.loopStatus_.loopExecCount; i++) {
        HCCL_VM_DEBUG(
            "locCcu[{}:{}], loopStartInstrId=[{}], loopCurInstrId=[{}], instrCnt=[{}], loopRound=[{}].", rankId_,
            dieId_, loopGroupInfo_.loopStatus_.loopStartInstrId, curInstrId_, instrCnt, i);
        auto simulator = std::make_unique<CcuSimulator>(
            rankId_, dieId_, loopGroupInfo_.loopStatus_.loopStartInstrId, loopGroupInfo_.loopStatus_.loopEndInstrId + 1,
            instrCnt, version_);
        loopGroupInfo_.loopStatus_.curLoopRound = i; // 设置当前迭代次数
        simulator->InitLoopGroupInfo(loopGroupInfo_);
        simulator->SetExecState(CcuExecState::EXEC_LOOP_INSTR);
        simulator->Execute();
    }
    return true;
}

// loopGroup内所有展开的loop执行，后续改为多线程并行执行
bool CcuSimulator::ExecuteLoopGroup()
{
    HCCL_VM_DEBUG(
        "locCcu[{}:{}], lopNum=[{}], loopExtendNum=[{}], loopOffset=[{}].", rankId_, dieId_, loopGroupInfo_.loopNum_,
        loopGroupInfo_.loopExtendNum_, loopGroupInfo_.loopOffset_);
    if (loopGroupInfo_.loopNum_ == 0) {
        return true;
    }
    if (loopGroupInfo_.loopNum_ > 2) { // 2: 目前最多支持两个loop模板
        return false;
    }

    // 原始loop模板 + 展开loop的执行
    for (uint32_t idx = 0; idx <= loopGroupInfo_.loopExtendNum_; idx++) {
        // idx = 0时，原始loop模板：从0开始执行loop，展开index = 0；
        // idx ≠ 0时，展开loop指令：从loopOffset开始执行，展开index = idx。
        int loopStartIdx = (idx == 0 ? 0 : loopGroupInfo_.loopOffset_);
        for (uint32_t i = loopStartIdx; i < loopGroupInfo_.loopNum_; i++) {
            loopGroupInfo_.loopStatus_.loopExtendIndex = idx;
            ExecuteInstr(loopGroupInfo_.startLoopId_ + i);
        }
    }
    return true;
}

bool CcuSimulator::Execute()
{
    while (curInstrId_ != endInstrId_) {
        if (ExecuteInstr(curInstrId_)) {
            continue;
        }
        return false;
    }
    if (curInstrId_ == endInstrId_) {
        finished_ = true;
    }
    // CcuResourceManager::GetInstance().DumpCcuXnResouceInfo();
    return true;
}

uint16_t CcuSimulator::GetCurInstrId() { return curInstrId_; }
