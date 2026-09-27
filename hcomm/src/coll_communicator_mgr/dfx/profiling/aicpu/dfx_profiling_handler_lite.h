/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_DFXPROFILING_HANDLER_LITE_H
#define HCCL_DFXPROFILING_HANDLER_LITE_H

#include <vector>
#include <unordered_map>
#include "hccl/hccl_types.h"
#include "prof_common.h"
#include "aprof_pub.h"
#include "dfx_circular_queue.h"
#include "res_pub.h"

extern "C" {
__attribute__((weak)) int32_t
MsprofReportBatchAdditionalInfo(uint32_t nonPersistantFlag, const VOID_PTR data, uint32_t length);
__attribute__((weak)) int32_t
AdprofReportBatchAdditionalInfo(uint32_t nonPersistantFlag, const void* data, uint32_t length);
__attribute__((weak)) int32_t AdprofReportAdditionalInfo(uint32_t nonPersistantFlag, const void* data, uint32_t length);
__attribute__((weak)) int32_t
MsprofReportAdditionalInfo(uint32_t nonPersistantFlag, const VOID_PTR data, uint32_t length);
__attribute__((weak)) int32_t AdprofCheckFeatureIsOn(uint64_t feature);
__attribute__((weak)) int32_t MsprofRegisterCallback(uint32_t moduleId, ProfCommandHandle handle);
__attribute__((weak)) uint64_t AdprofGetHashId(const char* hashInfo, size_t length);
__attribute__((weak)) uint64_t MsprofStr2Id(const char* hashInfo, size_t length);
};

namespace Hccl {
enum DfxMainStreamTaskType : uint8_t { HEAD = 0, TAIL = 1 };

enum DfxProfilingLevel : uint8_t { L0 = 0, L1 = 1 };

struct DfxFlagTaskInfo {
    uint32_t taskId;
    DfxMainStreamTaskType type;
};

struct DfxCommContext {
    const std::unordered_map<u64, u32>* channelRemoteRankIdMap{nullptr};
    u64 groupName{DFX_INVALID_U64};
    u32 localRank{INVALID_U32};
    u32 rankSize{0};
};

class DfxProfilingHandlerLite {
public:
    ~DfxProfilingHandlerLite();
    DfxProfilingHandlerLite(const DfxProfilingHandlerLite& that) = delete;
    DfxProfilingHandlerLite& operator=(const DfxProfilingHandlerLite& that) = delete;
    static DfxProfilingHandlerLite& GetInstance();
    HcclResult Init();
    void ReportHcclOpInfo(const DfxDfxOpInfo& opInfo, const DfxCommContext& ctx) const;
    void ReportMainStreamTask(const DfxFlagTaskInfo& flagTaskInfo) const;
    uint64_t GetCachedAlgTypeHashId() const;
    void UpdateProfSwitch();
    void SetProL0On(bool val);
    void SetProL1On(bool val);
    inline bool GetProfL0State() const { return enableHcclL0_; }
    inline bool GetProfL1State() const { return enableHcclL1_; }
    uint64_t GetProfHashId(const char* name, uint32_t len) const;

    void ReportStreamTaskDetails(TaskInfoCircularQueue& taskQueue, const DfxCommContext& ctx) const;
    void ReportStreamTaskDetailsLog(TaskInfoCircularQueue& taskQueue) const;

private:
    explicit DfxProfilingHandlerLite();
    void BindProfilingHandles();
    void InitHashCaches();
    void ReportAdditionInfo(const MsprofAdditionalInfo& reporterData) const;

    bool IsProfOn(uint64_t feature) const;
    bool IsProfSwitchOn(DfxProfilingLevel level);
    bool FillBatchReporterData(
        uint32_t batchId, const MsprofAicpuHcclTaskInfo* taskInfos, MsprofAdditionalInfo& addInfo) const;
    bool ReportBatchAddInfo(
        uint32_t batchId, const MsprofAicpuHcclTaskInfo* taskInfos, MsprofAdditionalInfo* addInfoVec,
        uint32_t& addInfoIndx, uint32_t maxBatchNum, bool isLastBatch) const;

    void GetTaskDetailInfosFromDfxTaskInfo(
        const DfxTaskInfo* it, MsprofAicpuHcclTaskInfo& taskDetailsInfos, const DfxCommContext& ctx) const;

    void FillReduceInlineDetail(const DfxTaskInfo* it, MsprofAicpuHcclTaskInfo& taskDetailsInfos) const;
    void FillSdmaRdmaDetail(const DfxTaskInfo* it, MsprofAicpuHcclTaskInfo& taskDetailsInfos) const;
    void FillUbDmaDetail(const DfxTaskInfo* it, MsprofAicpuHcclTaskInfo& taskDetailsInfos) const;
    void FillNotifyDetail(const DfxTaskInfo* it, MsprofAicpuHcclTaskInfo& taskDetailsInfos) const;
    void FillDefaultDetail(const DfxTaskInfo* it, MsprofAicpuHcclTaskInfo& taskDetailsInfos) const;
    void FillCclTagAndRemoteRank(
        const DfxTaskInfo* it, MsprofAicpuHcclTaskInfo& taskDetailsInfos, const DfxCommContext& ctx) const;
    void FillCommonTailFields(const DfxTaskInfo* it, MsprofAicpuHcclTaskInfo& taskDetailsInfos) const;

private:
    static DfxProfilingHandlerLite instance_;
    bool enableHcclL0_{false};
    bool enableHcclL1_{false};
    bool initializedFlag_{false};
    std::unordered_map<uint32_t, uint64_t> taskTypeHashCache_;
    std::unordered_map<u8, uint64_t> opTypeHashCache_;
    std::unordered_map<u8, uint64_t> algTypeHashCache_;
    uint64_t cachedAlgTypeHashId_{0};
    using ReportAdditionalInfoHandle = int32_t (*)(uint32_t, const void*, uint32_t);
    ReportAdditionalInfoHandle reportAdditionalInfo_{nullptr};
    using ReportBatchAdditionalInfoHandle = int32_t (*)(uint32_t, const void*, uint32_t);
    ReportBatchAdditionalInfoHandle reportBatchAdditionalInfo_{nullptr};
    using GetProfHashIdHandle = uint64_t (*)(const char*, size_t);
    GetProfHashIdHandle getProfHashId_{nullptr};
};

} // namespace Hccl

#endif // HCCL_DFXPROFILING_HANDLER_LITE_H
