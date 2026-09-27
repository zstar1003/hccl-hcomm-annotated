/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COMM_ENGINE_MANAGER_H
#define COMM_ENGINE_MANAGER_H

#include <unordered_map>
#include <mutex>
#include "hccl/hccl_res.h"
#include "hccl_independent_common.h"
#include "thread_manager.h"
#include "notify_manager.h"

namespace hccl {
class CommEngineResMgr {
public:
    CommEngineResMgr();
    HcclResult Init(
        uint32_t threadNum, uint32_t notifyNumPerThread, const std::string& commId, const aclrtBinHandle binHandle,
        const ManagerCallbacks& callbacks);
    HcclResult HcclThreadAcquireV2(
        CommEngine engine, uint32_t threadNum, ThreadType type, const ThreadConfig* config, ThreadHandle* threads,
        std::vector<uint32_t>& threadId);
    HcclResult HcclThreadAcquire(
        CommEngine engine, uint32_t threadNum, ThreadType type, const ThreadConfig* config, ThreadHandle* threads,
        std::vector<uint32_t>& threadId);
    HcclResult
    HcclThreadAcquireWithStream(CommEngine engine, rtStream_t stream, uint32_t notifyNum, ThreadHandle* thread);
    HcclResult HcclGetNotifyNumInThread(ThreadHandle thread, CommEngine engine, uint32_t* notifyNum);
    HcclResult HcclAllocNotify(
        CommEngine commEngine, ::NotifyType notifyType, uint32_t notifyNum, NotifyHandle** notifyHandleList);
    HcclResult HcommFreeNotify(uint32_t notifyNum, NotifyHandle* notifyHandleList);
    HcclResult HcclThreadExportToCommEngine(
        uint32_t threadNum, const ThreadHandle* threads, CommEngine dstCommEngine, ThreadHandle* exportedThreads);
    HcclResult HcclThreadResGetInfo(ThreadHandle thread, ThreadResType resType, uint32_t infoLen, void** info);
    HcclResult
    HcclDedicatedThreadAcquire(HcclDedicatedThreadType useType, uint32_t notifyNumPerThread, ThreadHandle* thread);
    HcclResult RegisterOrderLaunchThread(ThreadHandle thread);
    // N秒快恢 Resume：重置 CommEngine 平面上已触发但未消费的 local notify 状态
    HcclResult ResetCommLocalNotifies();

private:
    std::unique_ptr<ThreadMgr> threadMgr_;
    std::unique_ptr<NotifyManager> notifyMgr_;
    std::mutex mtx_;
};
} // namespace hccl
#endif
