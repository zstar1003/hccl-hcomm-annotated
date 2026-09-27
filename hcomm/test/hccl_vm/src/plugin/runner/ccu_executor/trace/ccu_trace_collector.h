/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_SIM_CCU_TRACE_COLLECTOR_H
#define HCCL_SIM_CCU_TRACE_COLLECTOR_H

#include <atomic>
#include <cstdint>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <vector>

#include "ccu_trace_types.h"

class CcuExecutorBase;
class CcuSimulator;

namespace CcuTrace {

// CKE Wait 自旋状态（每条指令可能多次 waitCKE）
struct WaitSpinState {
    bool active{false};
    uint32_t retryCount{0};
    uint16_t waitCKEId{0};
    uint16_t waitCKEMask{0};
    uint16_t ckeValueOnFirstCheck{0};
    uint64_t firstCheckTimeNs{0};
};

class CcuTraceCollector {
public:
    static CcuTraceCollector& GetInstance();

    CcuTraceCollector(const CcuTraceCollector&) = delete;
    CcuTraceCollector& operator=(const CcuTraceCollector&) = delete;

    // ===== 启用/禁用 =====
    void SetEnabled(bool enabled);
    bool IsEnabled() const;

    // ===== 会话管理 =====
    void StartRun(const CcuRunMetadata& metadata);
    void EndRun();
    void RegisterCcuIdentity(const CcuIdentity& identity);

    // ===== 静态配置注册（per CCU，初始化后不变） =====
    void RegisterInstrSpace(const CcuInstrSpace& instrSpace);
    void RegisterChannelSpace(const CcuChannelSpace& channelSpace);

    // ===== 全局上下文管理（由 SqeuentialExecutor 层调用） =====
    void BeginRound(uint32_t execRound);
    void BeginGlobalStep();
    uint32_t RegisterSqeTask(
        int32_t rankId, uint16_t dieId, uint8_t missionId, uint16_t instStartId, uint16_t instCnt, uint32_t key,
        const std::vector<uint64_t>& args, uint64_t simulatorPtr);
    void SetCurrentSqeTaskId(uint32_t sqeTaskId);
    CcuGlobalContext GetCurrentGlobalContext() const;

    // ===== 执行 CCU 标识（由 CcuSimulator 设置） =====
    void SetCurrentExecutingCcu(int32_t rankId, uint16_t dieId);
    std::pair<int32_t, uint16_t> GetCurrentExecutingCcu() const;

    // ===== CKE Wait 自旋检测与合并 =====
    // waitCKE 不满足时调用，记录一次自旋
    void RecordWaitSpin(
        int32_t rankId, uint16_t dieId, uint16_t instrId, uint16_t waitCKEId, uint16_t waitCKEMask, uint16_t ckeValue);
    // CKE 通过后调用，获取并重置自旋信息
    CcuWaitInfo FinalizeWaitInfo(int32_t rankId, uint16_t dieId, uint16_t instrId, uint16_t ckeValueOnPass);

    // ===== 跨 CCU 变更拦截（由 CcuResourceManager 调用） =====
    void RecordCrossCcuCkeChange(
        int32_t rankId, uint16_t dieId, uint16_t ckeId, uint16_t oldValue, uint16_t newValue, int32_t execRank,
        uint16_t execDie);
    void RecordCrossCcuMsChange(
        int32_t rankId, uint16_t dieId, uint16_t msId, uint64_t offset, uint32_t length,
        const std::vector<uint8_t>& dataAfter, int32_t execRank, uint16_t execDie);
    CcuCrossCcuChanges ConsumeCrossCcuChanges(int32_t rankId, uint16_t dieId);

    // ===== 记录 trace entry =====
    void RecordEntry(const CcuTraceEntry& entry);

    // ===== 静态 CCU 注册跟踪（随 StartRun 自动重置）=====
    // 返回 true 表示首次注册（需要执行注册逻辑），false 表示已注册过
    bool TryRegisterCcuStatic(int32_t rankId, uint16_t dieId);

    // ===== 记录非 CCU 任务 =====
    void RecordNonCcuEntry(const CcuTraceNonCcuEntry& entry);

    // ===== 指令空间采集（per CCU，纯静态） =====
    // 在 trace 开始前调用，传入预计算的 Describe() 结果
    void CaptureInstrSpace(
        int32_t rankId, uint16_t dieId, uint32_t instrCnt, const std::vector<CcuInstrSpaceEntry>& entries);

    // ===== 设置最终快照 =====
    void SetFinalSnapshot(int32_t rankId, uint16_t dieId, const CcuResourceSnapshot& snapshot);

    // ===== 获取完整 trace 数据 =====
    CcuTraceRun GetTraceRun() const;

    // ===== 输出路径管理（用于增量落盘和崩溃 dump） =====
    void SetOutputPath(const std::string& outputPath);
    const std::string& GetOutputPath() const;

    // ===== 增量落盘：将当前已采集的 trace 数据序列化写入磁盘 =====
    // 由 SqeuentialExecutor 定期调用，确保崩溃时磁盘上有最新数据
    void IncrementalDump();

    // ===== 重置 =====
    void Reset();

private:
    CcuTraceCollector() = default;
    ~CcuTraceCollector() = default;

    std::string MakeCcuKey(int32_t rankId, uint16_t dieId) const;
    std::string MakeWaitKey(int32_t rankId, uint16_t dieId, uint16_t instrId) const;

    static uint64_t GetCurrentTimeNs();

private:
    bool m_enabled{false};
    mutable std::mutex m_mutex;

    // Trace 数据
    CcuTraceRun m_traceRun;

    // 全局上下文
    std::atomic<uint32_t> m_globalSeqId{0};
    std::atomic<uint32_t> m_execRound{0};
    std::atomic<uint32_t> m_currentSqeTaskId{0};
    std::atomic<uint32_t> m_nextSqeTaskId{0};

    // 当前执行的 CCU（线程局部，单线程模型下简单用成员变量）
    int32_t m_execRank{-1};
    uint16_t m_execDie{0};

    // CKE Wait 自旋状态: key = "rankId_dieId_instrId"
    std::map<std::string, WaitSpinState> m_waitSpinStates;

    // 跨 CCU 变更缓冲: key = "rankId_dieId"（当前执行 CCU 的 key）
    std::map<std::string, CcuCrossCcuChanges> m_crossCcuBuffer;

    // 已注册的 CCU 静态信息（随 StartRun 重置）
    std::set<std::pair<int32_t, uint16_t>> m_registeredCcus;

    // 输出文件路径（用于增量落盘和崩溃 dump）
    std::string m_outputPath;
};

} // namespace CcuTrace

#endif // HCCL_SIM_CCU_TRACE_COLLECTOR_H
