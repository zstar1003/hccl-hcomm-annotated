/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TOPOINFO_EXCHANGE_DISPATCHER_H
#define TOPOINFO_EXCHANGE_DISPATCHER_H

#include <map>
#include <atomic>
#include <vector>
#include <mutex>
#include <thread>
#include <climits>
#include <condition_variable>
#include <queue>

#include "types.h"
#include "socket.h"
#include "ip_address.h"
#include "rank_info_detect_service.h"

namespace Hccl {

class RankInfoDispatcher {
public:
    struct SendState {
        u32 rankId;
        u64 header;
        size_t headerLen = sizeof(u64); // the header need to send
        size_t headerSended = 0;        // the header have sent length
        size_t bodyLen = 0;             // the whole data length
        size_t bodySended = 0;          // the data have sent
        void* data;                     // data pointer

        bool Send(std::shared_ptr<Socket> socket);
        bool SendHeader(std::shared_ptr<Socket> socket);
        bool SendBody(std::shared_ptr<Socket> socket);
        bool SendHelper(std::shared_ptr<Socket> socket, void* buf, size_t dataLen, size_t& sendedLen);
        bool IsOk() { return bodyLen != 0 && headerSended == headerLen && bodySended == bodyLen; }
    };

    struct FdContext {
        std::shared_ptr<Socket> socket;
        SendState txState;
    };

    using WorkerTask = std::function<void(void)>;

public:
    static constexpr u32 DEFAULT_THREAD_NUM = 1;
    static constexpr u32 MAX_THREAD_NUM = 4;
    static constexpr s32 INVALID_EPOLL_EVENT_FD = -1;
    static constexpr s32 EPOLL_TIMEOUT_MS = 100;    // 100ms
    static constexpr s32 LAST_EPOLL_TIMEOUT_MS = 5; // 5ms
    static constexpr s32 RANK_CAPACITY_PER_THREAD = 512;

    explicit RankInfoDispatcher(RankInfoDetectService* rankInfoDetectServer, u32 threadNum = DEFAULT_THREAD_NUM)
        : rankInfoDetectServer_(rankInfoDetectServer),
          threadNum_(threadNum)
    {}
    ~RankInfoDispatcher();

    void BroadcastRankTable(
        const std::unordered_map<std::string, std::shared_ptr<Socket>>& connectSockets,
        const RankTableInfo& clusterInfo, const std::string& failedAgentIdList, u32 step);

private:
    void InitWorkerThread();
    void WorkerWait(int workId);
    void WakeWoker();
    void RunWorkerThread(int workId);
    bool GetTask(WorkerTask& workTask);
    void PrepareResource(
        const std::unordered_map<std::string, std::shared_ptr<Socket>> connectSockets, const RankTableInfo& clusterInfo,
        const std::string& failedAgentIdList, u32 step);
    void SendOnce();
    void ProcessOneSendEvent(int epollFd, FdHandle& fdHandle);
    void ProcessSend();
    void CleanResource();
    void CloseEpollFd();

    RankInfoDetectService* rankInfoDetectServer_;
    u32 threadNum_{DEFAULT_THREAD_NUM};
    u32 rankNum_{0};
    std::vector<std::thread> workerThreads_;
    std::queue<WorkerTask> taskQueue_;
    std::mutex taskQueueMutex_;

    std::unordered_map<FdHandle, FdContext> fdHandleToFdContextMap_;
    std::mutex fdHandleMapMutex_;
    s32 epollFds_ = INVALID_EPOLL_EVENT_FD;
    std::atomic<u32> sendDoneCount_{0};

    std::vector<char> rankTableMsg_;

    std::mutex wakeMutex_;
    std::atomic<bool> epollCreate_{false};
    std::atomic<bool> ready_{false};
    std::atomic<bool> stop_{false};
    std::condition_variable wakeManager_;
};
} // namespace Hccl

#endif /* TOPOINFO_EXCHANGE_DISPATCHER_H */
