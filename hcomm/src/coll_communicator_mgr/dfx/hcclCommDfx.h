/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_COMM_DFX_H
#define HCCL_COMM_DFX_H

#include <memory>
#include "mirror_task_manager.h"
#include "hcclCommProfiling.h"
#include "global_mirror_tasks.h"
#include <shared_mutex>
#include "hccl_common.h"
#include <unordered_map>
#include <mutex>
#include "buffer.h"
#include "common.h"
#include "hcclCommOp.h"

namespace hccl {

class HcclCommDfx {
public:
    HcclCommDfx();

    ~HcclCommDfx();

    HcclResult Init(u32 deviceId, const std::string& comTag, u32 myRankId);

    HcclResult AddTaskInfoCallback(u32 streamId, u32 taskId, const Hccl::TaskParam& taskParam, u64 handle);
    HcclResult AddDpuTaskInfoCallback(const Hccl::TaskParam& taskParam, u64 handle);

    Hccl::MirrorTaskManager* GetMirrorTaskManager() const;

    HcclResult SetCurrDfxOpInfo(std::shared_ptr<Hccl::DfxOpInfo> dfxOpInfo);

    HcclResult ReportAllTasks(bool cachedReq);
    HcclResult ReportOp(uint64_t beginTime, bool cachedReq, bool isOpBase);
    void ReportMc2CommInfo(const Mc2CommInfo& mc2CommInfo);
    HcclResult UpdateProfStat();

    static void AddChannelRemoteRankId(const std::string& commTag, u64 handle, u32 remoteRankId);
    static HcclResult GetChannelRemoteRankId(const std::string& commTag, u64 handle, u32& remoteRankId);
    static u32 GetTaskId(u32 streamId);
    std::function<HcclResult(u32, u32, const Hccl::TaskParam&, u64)> GetCallback() const { return setAddTaskCallback_; }
    std::function<HcclResult(const Hccl::TaskParam&, u64)> GetDpuCallback() const { return setAddDpuTaskCallback_; }
    HcclResult ReportKernel(
        uint64_t beginTime, const std::string& commTag, const std::string& kernelName, uint32_t threadId,
        bool cachedReq);
    HcclResult GetOpModeFlags(bool& isOpBase, bool& isCached);
    u64 GetGroupNameHash() const { return groupNameHash_; }

    void SetDpuStreamId(u32 dpuStreamId);
    void SetAicpuTaskIdAndStreamId(u32 taskId, u32 streamId)
    {
        aicpuTaskId_ = taskId;
        aicpuStreamId_ = streamId;
    }

private:
    std::unique_ptr<Hccl::MirrorTaskManager> mirrorTaskManager_{nullptr};
    std::unique_ptr<HcclCommProfiling> profiling_{nullptr};
    static std::unordered_map<std::string, std::unordered_map<u64, u32>> channelRemoteRankId_;
    static std::unordered_map<u32, u32> streamIdToTaskId_;
    static std::shared_mutex baseLock_;
    static std::mutex taskIdMutex_;
    std::string commTag_{};
    u64 groupNameHash_{0};
    u32 deviceId_{0};
    u32 myRankId_{0};
    u32 dpuStreamId_{0};
    u32 aicpuTaskId_{INVALID_UINT};
    u32 aicpuStreamId_{INVALID_UINT};
    std::function<HcclResult(u32, u32, const Hccl::TaskParam&, u64)> setAddTaskCallback_{};
    std::function<HcclResult(const Hccl::TaskParam&, u64)> setAddDpuTaskCallback_{};
    bool initializedFlag_{false};
    void AddTaskInfoCallbackLog(const Hccl::TaskParam& taskParam, const std::unordered_map<u64, u32>& handleMap) const;
};

} // namespace hccl

#endif
