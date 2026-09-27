/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
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

#include "adapter_hccp_common.h"
#include "externalinput_pub.h"
#include "hccl_socket.h"
#include "hccl_common.h"
#include "topoinfo_exchange_server.h"
#include "../json_utils.h"

namespace hccl {
class TopoInfoExchangeDispatcher {
    // avoid the struct name pollution hccl namespace, so use the struct in class
public:
    struct SendState {
        u32 rankId;
        u32 header;
        u32 identify = UINT_MAX;          // 默认UINT_MAX时，不发送identify
        size_t headerLen = sizeof(u32);   // the header need to send
        size_t headerSended = 0;          // the header have sent length
        size_t bodyLen = 0;               // the whole data length
        size_t bodySended = 0;            // the data have sent
        size_t identifyLen = sizeof(u32); // the identify need to send (MasterInfo mode)
        size_t identifySended = 0;        // the identify have sent
        void* data;                       // data pointer
        bool firstSendFlag_ = true;

        HcclResult Send(std::shared_ptr<HcclSocket> socket);
        HcclResult SendHeader(std::shared_ptr<HcclSocket> socket);
        HcclResult SendBody(std::shared_ptr<HcclSocket> socket);
        HcclResult SendIdentify(std::shared_ptr<HcclSocket> socket);
        HcclResult SendHelper(std::shared_ptr<HcclSocket> socket, char* buf, size_t dataLen, size_t& sendedLen);
        bool IsOk() const { return bodyLen != 0 && headerSended == headerLen && bodySended == bodyLen; }
    };

    struct FdContext {
        std::shared_ptr<HcclSocket> socket;
        SendState txState;
    };

    using WorkerTask = std::function<HcclResult(void)>;

public:
    static constexpr u32 DEFAULT_THREAD_NUM = 1;
    static constexpr u32 MAX_THREAD_NUM = 4;
    static constexpr s32 INVALID_EPOLL_EVENT_FD = -1;
    static constexpr s32 EPOLL_TIMEOUT_MS = 100;    // 100ms
    static constexpr s32 LAST_EPOLL_TIMEOUT_MS = 5; // 5ms
    static constexpr s32 RANK_CAPACITY_PER_THREAD = 512;

    explicit TopoInfoExchangeDispatcher(
        TopoInfoExchangeServer* topoInfoExchangeServer, u32 threadNum = DEFAULT_THREAD_NUM)
        : topoInfoExchangeServer_(topoInfoExchangeServer),
          threadNum_(threadNum)
    {}
    ~TopoInfoExchangeDispatcher();

    HcclResult BroadcastRankTable(
        const std::map<std::string, std::shared_ptr<HcclSocket>> connectSockets, const RankTable_t& clusterInfo,
        const std::string& failedAgentIdList);
    HcclResult BroadcastGroupLeaderInfo(
        const std::map<std::string, std::shared_ptr<HcclSocket>> connectSockets, const GroupLeader_t& leaderInfo);

private:
    void InitWorkerThread();
    void WorkerWait(int workId);
    void WakeWoker();
    void RunWorkerThread(int workId);
    bool GetTask(WorkerTask& workTask);
    HcclResult PrepareResource(
        const std::map<std::string, std::shared_ptr<HcclSocket>> connectSockets, const RankTable_t& clusterInfo,
        const std::string& failedAgentIdList);
    HcclResult PrepareLeaderResource(
        const std::map<std::string, std::shared_ptr<HcclSocket>> connectSockets, const GroupLeader_t& leaderInfo);
    HcclResult SendOnce();
    HcclResult ProcessOneSendEvent(s32 epollFd, FdHandle& fdHandle);
    HcclResult ProcessSend();
    void CleanResource();
    HcclResult CloseEpollFd();

    TopoInfoExchangeServer* topoInfoExchangeServer_;
    u32 threadNum_ = 1;
    u32 rankNum_ = 0;
    std::vector<std::thread> workerThreads_;
    std::queue<WorkerTask> taskQueue_;
    std::mutex taskQueueMutex_;

    std::unordered_map<FdHandle, FdContext> fdHandleToFdContextMap_;
    std::mutex fdHandleMapMutex_;
    s32 epollFds_ = INVALID_EPOLL_EVENT_FD;
    std::atomic<u32> sendDoneCount_{0};

    std::string rankTableJson_;

    std::mutex wakeMutex_;
    std::atomic<bool> ready_{false};
    std::atomic<bool> stop_{false};
    std::condition_variable wakeManager_;
};
} // namespace hccl

#endif /* TOPOINFO_EXCHANGE_DISPATCHER_H */
