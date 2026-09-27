/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef THREAD_MANAGER_H
#define THREAD_MANAGER_H
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <memory>
#include <mutex>
#include "hccl/hccl_res.h"
#include "hccl_independent_common.h"
#include "aicpu_ts_thread.h"
#include "cpu_ts_thread.h"
#include "log.h"
#include "manager_common.h"

namespace hccl {

class ThreadMgr {
public:
    ThreadMgr(
        uint32_t threadNum, uint32_t notifyNumPerThread, std::string commId, aclrtBinHandle binHandle,
        const ManagerCallbacks& callbacks);
    ~ThreadMgr();
    HcclResult HcclThreadAcquire(
        CommEngine engine, uint32_t threadNum, ThreadType type, const ThreadConfig* config, ThreadHandle* threads,
        std::vector<uint32_t>& threadId);
    HcclResult HcclThreadAcquireV2(
        CommEngine engine, uint32_t threadNum, ThreadType type, const ThreadConfig* config, ThreadHandle* threads,
        std::vector<uint32_t>& threadId);
    HcclResult
    HcclThreadAcquireWithStream(CommEngine engine, rtStream_t stream, uint32_t notifyNum, ThreadHandle* thread);
    HcclResult HcclGetNotifyNumInThread(ThreadHandle thread, uint32_t* notifyNum);
    HcclResult HcclThreadExportToCommEngine(
        uint32_t threadNum, const ThreadHandle* threads, CommEngine dstCommEngine, ThreadHandle* exportedThreads);
    HcclResult HcclThreadResGetInfo(ThreadHandle thread, ThreadResType resType, uint32_t infoLen, void** info);
    HcclResult
    HcclDedicatedThreadAcquire(HcclDedicatedThreadType useType, uint32_t notifyNumPerThread, ThreadHandle* thread);
    HcclResult RegisterOrderLaunchThread(ThreadHandle thread);
    HcclResult ResetThreadLocalNotifies();
    u32 GetThreadNum() const { return threadNum_; }
    u32 GetNotifyNumPerThread() const { return notifyNumPerThread_; }

private:
    uint64_t GetMaxNotifyTotal();
    HcclResult CheckNotifyNum(CommEngine engine, uint32_t threadNum, uint32_t notifyNumPerThread);
    HcclResult CheckThreadNum(CommEngine engine, uint32_t threadNum, uint32_t notifyNumPerThread);
    HcclResult SupplementNotify(CommEngine engine, std::vector<std::shared_ptr<Thread>>& needSupplementThread);
    HcclResult SupplementNotify(CommEngine engine, uint32_t threadNum, ThreadType type, const ThreadConfig* config);
    HcclResult SupplementThread(
        CommEngine engine, std::vector<std::shared_ptr<Thread>>& newThreads,
        std::unique_ptr<ThreadHandle[]>& hostHandle);
    HcclResult
    SupplementThread(CommEngine engine, uint32_t supplementThreadNum, ThreadType type, const ThreadConfig* config);
    HcclResult
    ThreadExportToCommEngineCpu(uint32_t threadNum, const ThreadHandle* threads, ThreadHandle* exportedThreads);
    HcclResult ThreadExportToCommEngineAicpu(
        uint32_t threadNum, const ThreadHandle* threads, CommEngine dstCommEngine, ThreadHandle* exportedThreads);
    HcclResult GetExportedThread(
        const ThreadHandle threadHandle, CommEngine commEngine, Thread*& exportedThread,
        std::shared_ptr<Thread>& threadOut);
    HcclResult ExportHostThreadsToAicpu(
        std::vector<std::shared_ptr<Thread>>& hostThreads, const std::vector<u32>& index, const ThreadHandle* threads,
        CommEngine dstCommEngine, ThreadHandle* exportedThreads);
    HcclResult ExportOrderLaunchThreadsToAicpu(
        std::vector<std::shared_ptr<Thread>>& orderLaunchHostThreads, const std::vector<u32>& orderLaunchIndex,
        const ThreadHandle* threads, CommEngine dstCommEngine, ThreadHandle* exportedThreads);
    HcclResult CreateAndInitThreads(
        CommEngine engine, StreamType streamType, NotifyLoadType notifyLoadType, uint32_t threadNum,
        const ThreadConfig* config, std::vector<std::shared_ptr<Thread>>& newThreads);
    HcclResult AssignThreadHandles(
        CommEngine engine, std::vector<std::shared_ptr<Thread>>& newThreads, ThreadHandle* threads,
        std::unique_ptr<ThreadHandle[]>& hostHandle);
    HcclResult StoreThreadsAndBuildHandleMap(
        CommEngine engine, std::vector<std::shared_ptr<Thread>>& newThreads,
        std::unique_ptr<ThreadHandle[]>& hostHandle);
    HcclResult
    HcclUnfoldThreadAcquire(HcclDedicatedThreadType useType, uint32_t notifyNumPerThread, ThreadHandle* thread);
    HcclResult
    HcclDeviceOrderThreadCreate(HcclDedicatedThreadType useType, uint32_t notifyNumPerThread, ThreadHandle* thread);
    HcclResult ResetLocalNotify(LocalNotify* notify, uint32_t notifyIdx, uint64_t threadHandle);
    HcclResult ResetNotifiesInThread(Thread* thread);
    HcclResult ResetThreadPoolLocalNotifies();        // 普通线程池 threads_
    HcclResult ResetMainThreadLocalNotifies();        // 主线程 mainThread_
    HcclResult ResetDedicatedThreadLocalNotifies();   // 专用线程 dedicatedThreadMap_
    HcclResult ResetOrderLaunchThreadLocalNotifies(); // 保序流线程 orderLaunchThreads_

    u32 threadNum_ = 0;
    u32 notifyNumPerThread_ = 0;
    std::string commId_;
    aclrtBinHandle binHandle_;

    u64 usedNotifyNum_ = 0;
    std::mutex threadMutex_;
    std::vector<std::shared_ptr<Thread>> threads_;

    std::mutex mainThreadMutex_;
    std::map<rtStream_t, std::shared_ptr<Thread>> mainThread_;

    std::mutex engineToThreadMutex_;
    std::map<std::pair<CommEngine, ThreadType>, std::vector<std::shared_ptr<Thread>>> engineToThreadsMap_;

    std::mutex threadMapMutex_;
    std::unordered_map<ThreadHandle, ThreadHandle>
        threadHandleOthersToCpu_; // 其他引擎上的ThreadHandle与CPU_TS上的ThreadHandle的映射
    std::unordered_map<ThreadHandle, ThreadHandle> hostToDeviceThreadHandle_;
    ManagerCallbacks callbacks_;

    std::mutex threadhandleToThreadMutex_;
    std::unordered_map<ThreadHandle, std::shared_ptr<Thread>> threadMap_;

    std::mutex dedicatedThreadMutex_;
    std::unordered_map<HcclDedicatedThreadType, ThreadHandle> dedicatedThreadMap_;

    std::unordered_set<ThreadHandle>
        orderLaunchThreads_; // 保序流（非所有权，资源由 OrderLaunchThreadMgr 管理，同 comm 操作串行无需加锁）
};
} // namespace hccl
#endif
