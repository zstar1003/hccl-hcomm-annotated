/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_trace_collector.h"

#include <chrono>
#include <sstream>

#include "ccu_trace_serializer.h"
#include "sim_log.h"

namespace CcuTrace {

CcuTraceCollector& CcuTraceCollector::GetInstance()
{
    static CcuTraceCollector instance;
    return instance;
}

void CcuTraceCollector::SetEnabled(bool enabled)
{
    m_enabled = enabled;
    if (enabled) {
        HCCL_VM_INFO("Trace collection enabled");
    } else {
        HCCL_VM_INFO("Trace collection disabled");
    }
}

bool CcuTraceCollector::IsEnabled() const { return m_enabled; }

void CcuTraceCollector::StartRun(const CcuRunMetadata& metadata)
{
    if (!m_enabled) {
        return;
    }
    std::lock_guard<std::mutex> lock(m_mutex);
    m_traceRun.runMetadata = metadata;
    m_globalSeqId = 0;
    m_execRound = 0;
    m_currentSqeTaskId = 0;
    m_nextSqeTaskId = 0;
    m_waitSpinStates.clear();
    m_crossCcuBuffer.clear();
    m_registeredCcus.clear();
    m_traceRun.globalEntries.clear();
    m_traceRun.nonCcuEntries.clear();
    m_traceRun.ccuRegistry.clear();
    m_traceRun.sqeTaskRegistry.clear();
    m_traceRun.instrSpaces.clear();
    m_traceRun.channelSpaces.clear();
    m_traceRun.ccuFinalSnapshots.clear();
    m_traceRun.runSummary = CcuRunSummary{};
    HCCL_VM_INFO("StartRun: rankSize={}, diePerRank={}", metadata.rankSize, metadata.diePerRank);
}

void CcuTraceCollector::EndRun()
{
    if (!m_enabled) {
        return;
    }
    std::lock_guard<std::mutex> lock(m_mutex);
    m_traceRun.runMetadata.totalExecRounds = m_execRound.load();
    m_traceRun.runMetadata.totalSqeTaskCount = m_nextSqeTaskId.load();
    HCCL_VM_INFO(
        "EndRun: totalEntries={}, totalRounds={}, totalSqeTasks={}", m_traceRun.globalEntries.size(),
        m_execRound.load(), m_nextSqeTaskId.load());
}

void CcuTraceCollector::RegisterCcuIdentity(const CcuIdentity& identity)
{
    if (!m_enabled) {
        return;
    }
    std::lock_guard<std::mutex> lock(m_mutex);
    m_traceRun.ccuRegistry.push_back(identity);
}

void CcuTraceCollector::RegisterInstrSpace(const CcuInstrSpace& instrSpace)
{
    if (!m_enabled) {
        return;
    }
    std::lock_guard<std::mutex> lock(m_mutex);
    m_traceRun.instrSpaces.push_back(instrSpace);
}

void CcuTraceCollector::RegisterChannelSpace(const CcuChannelSpace& channelSpace)
{
    if (!m_enabled) {
        return;
    }
    std::lock_guard<std::mutex> lock(m_mutex);
    m_traceRun.channelSpaces.push_back(channelSpace);
}

// ===== 全局上下文管理 =====

void CcuTraceCollector::BeginRound(uint32_t execRound)
{
    if (!m_enabled) {
        return;
    }
    m_execRound = execRound;
}

void CcuTraceCollector::BeginGlobalStep()
{
    if (!m_enabled) {
        return;
    }
    m_globalSeqId++;
}

uint32_t CcuTraceCollector::RegisterSqeTask(
    int32_t rankId, uint16_t dieId, uint8_t missionId, uint16_t instStartId, uint16_t instCnt, uint32_t key,
    const std::vector<uint64_t>& args, uint64_t simulatorPtr)
{
    if (!m_enabled) {
        return 0;
    }
    std::lock_guard<std::mutex> lock(m_mutex);
    uint32_t sqeTaskId = m_nextSqeTaskId++;
    CcuSqeTask task;
    task.sqeTaskId = sqeTaskId;
    task.rankId = rankId;
    task.dieId = dieId;
    task.missionId = missionId;
    task.instStartId = instStartId;
    task.instCnt = instCnt;
    task.key = key;
    task.args = args;
    task.simulatorPtr = simulatorPtr;
    task.firstExecRound = m_execRound.load();
    m_traceRun.sqeTaskRegistry.push_back(task);
    return sqeTaskId;
}

void CcuTraceCollector::SetCurrentSqeTaskId(uint32_t sqeTaskId)
{
    if (!m_enabled) {
        return;
    }
    m_currentSqeTaskId = sqeTaskId;
}

CcuGlobalContext CcuTraceCollector::GetCurrentGlobalContext() const
{
    CcuGlobalContext ctx;
    ctx.globalSeqId = m_globalSeqId.load();
    ctx.execRound = m_execRound.load();
    ctx.currentSqeTaskId = m_currentSqeTaskId.load();
    return ctx;
}

// ===== 执行 CCU 标识 =====

void CcuTraceCollector::SetCurrentExecutingCcu(int32_t rankId, uint16_t dieId)
{
    if (!m_enabled) {
        return;
    }
    m_execRank = rankId;
    m_execDie = dieId;
}

std::pair<int32_t, uint16_t> CcuTraceCollector::GetCurrentExecutingCcu() const { return {m_execRank, m_execDie}; }

// ===== CKE Wait 自旋检测与合并 =====

void CcuTraceCollector::RecordWaitSpin(
    int32_t rankId, uint16_t dieId, uint16_t instrId, uint16_t waitCKEId, uint16_t waitCKEMask, uint16_t ckeValue)
{
    if (!m_enabled) {
        return;
    }
    std::lock_guard<std::mutex> lock(m_mutex);
    std::string key = MakeWaitKey(rankId, dieId, instrId);
    auto& spinState = m_waitSpinStates[key];
    if (!spinState.active) {
        spinState.active = true;
        spinState.retryCount = 1;
        spinState.waitCKEId = waitCKEId;
        spinState.waitCKEMask = waitCKEMask;
        spinState.ckeValueOnFirstCheck = ckeValue;
        spinState.firstCheckTimeNs = GetCurrentTimeNs();
    } else {
        spinState.retryCount++;
    }
}

CcuWaitInfo
CcuTraceCollector::FinalizeWaitInfo(int32_t rankId, uint16_t dieId, uint16_t instrId, uint16_t ckeValueOnPass)
{
    CcuWaitInfo info;
    if (!m_enabled) {
        return info;
    }
    std::lock_guard<std::mutex> lock(m_mutex);
    std::string key = MakeWaitKey(rankId, dieId, instrId);
    auto it = m_waitSpinStates.find(key);
    if (it != m_waitSpinStates.end() && it->second.active) {
        info.hadWait = true;
        info.waitRetryCount = it->second.retryCount;
        info.waitCKEId = it->second.waitCKEId;
        info.waitCKEMask = it->second.waitCKEMask;
        info.ckeValueOnFirstCheck = it->second.ckeValueOnFirstCheck;
        info.ckeValueOnPass = ckeValueOnPass;
        info.waitDurationNs = GetCurrentTimeNs() - it->second.firstCheckTimeNs;
        m_waitSpinStates.erase(it);

        // 更新摘要统计
        m_traceRun.runSummary.totalCkeWaitSpins += info.waitRetryCount;
        m_traceRun.runSummary.totalHoldEvents++;
    }
    return info;
}

// ===== 跨 CCU 变更拦截 =====

void CcuTraceCollector::RecordCrossCcuCkeChange(
    int32_t rankId, uint16_t dieId, uint16_t ckeId, uint16_t oldValue, uint16_t newValue, int32_t execRank,
    uint16_t execDie)
{
    if (!m_enabled) {
        return;
    }
    std::lock_guard<std::mutex> lock(m_mutex);
    std::string key = MakeCcuKey(execRank, execDie);
    auto& changes = m_crossCcuBuffer[key];
    changes.hasCrossCcuChange = true;
    CcuRemoteCkeChange change;
    change.remoteRankId = rankId;
    change.remoteDieId = dieId;
    change.ckeId = ckeId;
    change.valueBefore = oldValue;
    change.valueAfter = newValue;
    changes.remoteCkeChanges.push_back(change);
}

void CcuTraceCollector::RecordCrossCcuMsChange(
    int32_t rankId, uint16_t dieId, uint16_t msId, uint64_t offset, uint32_t length,
    const std::vector<uint8_t>& dataAfter, int32_t execRank, uint16_t execDie)
{
    if (!m_enabled) {
        return;
    }
    std::lock_guard<std::mutex> lock(m_mutex);
    std::string key = MakeCcuKey(execRank, execDie);
    auto& changes = m_crossCcuBuffer[key];
    changes.hasCrossCcuChange = true;
    CcuRemoteMsChange change;
    change.remoteRankId = rankId;
    change.remoteDieId = dieId;
    change.msId = msId;
    change.offset = offset;
    change.length = length;
    change.dataAfter = dataAfter;
    changes.remoteMsChanges.push_back(change);
}

CcuCrossCcuChanges CcuTraceCollector::ConsumeCrossCcuChanges(int32_t rankId, uint16_t dieId)
{
    CcuCrossCcuChanges result;
    if (!m_enabled) {
        return result;
    }
    std::lock_guard<std::mutex> lock(m_mutex);
    std::string key = MakeCcuKey(rankId, dieId);
    auto it = m_crossCcuBuffer.find(key);
    if (it != m_crossCcuBuffer.end()) {
        result = it->second;
        m_crossCcuBuffer.erase(it);
    }
    return result;
}

// ===== 静态 CCU 注册跟踪 =====

bool CcuTraceCollector::TryRegisterCcuStatic(int32_t rankId, uint16_t dieId)
{
    if (!m_enabled) {
        return false;
    }
    std::lock_guard<std::mutex> lock(m_mutex);
    auto key = std::make_pair(rankId, dieId);
    return m_registeredCcus.insert(key).second; // 返回 true 表示首次插入
}

// ===== 记录 trace entry =====

void CcuTraceCollector::RecordEntry(const CcuTraceEntry& entry)
{
    if (!m_enabled) {
        return;
    }
    std::lock_guard<std::mutex> lock(m_mutex);
    m_traceRun.globalEntries.push_back(entry);

    // 更新摘要统计
    auto& summary = m_traceRun.runSummary;
    summary.totalInstrExecuted++;
    if (entry.execState == EXEC_FAIL) {
        summary.totalFailedInstr++;
    }
    std::string catStr = CcuInstrCategoryToString(entry.category);
    summary.instrCountByCategory[catStr]++;
    std::string ccuKey = MakeCcuKey(entry.rankId, entry.dieId);
    summary.instrCountByCcu[ccuKey]++;
}

void CcuTraceCollector::RecordNonCcuEntry(const CcuTraceNonCcuEntry& entry)
{
    if (!m_enabled) {
        return;
    }
    std::lock_guard<std::mutex> lock(m_mutex);
    m_traceRun.nonCcuEntries.push_back(entry);
}

void CcuTraceCollector::CaptureInstrSpace(
    int32_t rankId, uint16_t dieId, uint32_t instrCnt, const std::vector<CcuInstrSpaceEntry>& entries)
{
    if (!m_enabled) {
        return;
    }
    std::lock_guard<std::mutex> lock(m_mutex);
    CcuInstrSpace space;
    space.rankId = rankId;
    space.dieId = dieId;
    space.instructions = entries;
    m_traceRun.instrSpaces.push_back(std::move(space));
    HCCL_VM_INFO("CaptureInstrSpace: rankId={}, dieId={}, instrCount={}", rankId, dieId, instrCnt);
}

void CcuTraceCollector::SetFinalSnapshot(int32_t rankId, uint16_t dieId, const CcuResourceSnapshot& snapshot)
{
    if (!m_enabled) {
        return;
    }
    std::lock_guard<std::mutex> lock(m_mutex);
    std::string key = MakeCcuKey(rankId, dieId);
    m_traceRun.ccuFinalSnapshots[key] = snapshot;
}

CcuTraceRun CcuTraceCollector::GetTraceRun() const
{
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_traceRun;
}

// ===== 输出路径管理 =====

void CcuTraceCollector::SetOutputPath(const std::string& outputPath)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    m_outputPath = outputPath;
}

const std::string& CcuTraceCollector::GetOutputPath() const { return m_outputPath; }

// ===== 增量落盘 =====

void CcuTraceCollector::IncrementalDump()
{
    if (!m_enabled || m_outputPath.empty()) {
        return;
    }
    // GetTraceRun 内部含锁保护，不要在 m_mutex 持有期间调用
    auto traceRun = GetTraceRun();
    CcuTraceSerializer::DumpToFile(traceRun, m_outputPath);
}

void CcuTraceCollector::Reset()
{
    std::lock_guard<std::mutex> lock(m_mutex);
    m_traceRun = CcuTraceRun{};
    m_globalSeqId = 0;
    m_execRound = 0;
    m_currentSqeTaskId = 0;
    m_nextSqeTaskId = 0;
    m_waitSpinStates.clear();
    m_crossCcuBuffer.clear();
    m_execRank = -1;
    m_execDie = 0;
    m_outputPath.clear();
}

// ===== 私有辅助 =====

std::string CcuTraceCollector::MakeCcuKey(int32_t rankId, uint16_t dieId) const
{
    return std::to_string(rankId) + "_" + std::to_string(dieId);
}

std::string CcuTraceCollector::MakeWaitKey(int32_t rankId, uint16_t dieId, uint16_t instrId) const
{
    return std::to_string(rankId) + "_" + std::to_string(dieId) + "_" + std::to_string(instrId);
}

uint64_t CcuTraceCollector::GetCurrentTimeNs()
{
    auto now = std::chrono::steady_clock::now();
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count());
}

} // namespace CcuTrace
