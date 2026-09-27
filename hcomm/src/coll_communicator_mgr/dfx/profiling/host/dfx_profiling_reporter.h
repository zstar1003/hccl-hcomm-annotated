/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_DFX_PROFILING_REPORTER_H
#define HCCL_DFX_PROFILING_REPORTER_H
#include <mutex>
#include "mirror_task_manager.h"
#include "dfx_profiling_handler.h"
#include "queue.h"

namespace Hccl {
class DfxProfilingReporter {
public:
    DfxProfilingReporter(MirrorTaskManager* mirrorTaskMgr, DfxProfilingHandler* profilingHandler);
    virtual ~DfxProfilingReporter();
    HcclResult Init();
    void ReportOp(uint64_t beginTime, bool cachedReq, bool opbased) const;
    void ReportAllTasks(bool cachedReq);
    void SetCurrDfxOpInfo(std::shared_ptr<DfxOpInfo> dfxOpInfo) const;
    void UpdateProfStat();
    void CallReportMc2CommInfo(
        const Stream& kfcStream, const Stream& stream, const std::vector<Stream*>& aicpuStreams, const std::string& id,
        RankId myRank, u32 rankSize, RankId rankInParentComm) const;

    void CallReportMc2CommInfo(
        const u32 kfcStreamId, const std::vector<u32>& aicpuStreamsId, const std::string& id, RankId myRank,
        u32 rankSize, RankId rankInParentComm) const;

private:
    void ReportCallBackAllTasks(bool cachedReq = false);
    void ReportAllTasksLog() const;

private:
    MirrorTaskManager* mirrorTaskMgr_{nullptr};
    std::atomic<bool> enableHcclL1_{false};
    using lastPosesMap = std::unordered_map<u32, std::shared_ptr<Queue<std::unique_ptr<TaskInfo>>::Iterator>>;
    mutable lastPosesMap curLastPoses_{};
    DfxProfilingHandler* profilingHandler_{nullptr};
    bool initializedFlag_{false};
    std::vector<TaskInfo*> taskInfoBatch_;
};
} // namespace Hccl

#endif // HCCL_DFX_PROFILING_REPORTER_H
