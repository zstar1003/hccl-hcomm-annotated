/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <sys/socket.h>
#include <sys/epoll.h>
#include <unistd.h>
#include <ctime>
#include "sal_pub.h"
#include "topoinfo_exchange_dispatcher.h"

namespace hccl {
TopoInfoExchangeDispatcher::~TopoInfoExchangeDispatcher() { CleanResource(); }

HcclResult TopoInfoExchangeDispatcher::BroadcastRankTable(
    const std::map<std::string, std::shared_ptr<HcclSocket>> connectSockets, const RankTable_t& clusterInfo,
    const std::string& failedAgentIdList)
{
    CHK_RET(PrepareResource(connectSockets, clusterInfo, failedAgentIdList));
    CHK_RET(ProcessSend());
    HCCL_INFO(
        "cluster topo exchange worker broadcast topoinfo success, rankNum[%d], "
        "threadNum[%u]",
        rankNum_, threadNum_);
    return HCCL_SUCCESS;
}

HcclResult TopoInfoExchangeDispatcher::BroadcastGroupLeaderInfo(
    const std::map<std::string, std::shared_ptr<HcclSocket>> connectSockets, const GroupLeader_t& leaderInfo)
{
    CHK_RET(PrepareLeaderResource(connectSockets, leaderInfo));
    CHK_RET(ProcessSend());
    HCCL_INFO(
        "cluster topo exchange worker broadcast GroupLeaderInfo success, rankNum[%d], "
        "threadNum[%u]",
        rankNum_, threadNum_);
    return HCCL_SUCCESS;
}

void TopoInfoExchangeDispatcher::InitWorkerThread()
{
    threadNum_ = std::max(1, std::min(int(rankNum_ / RANK_CAPACITY_PER_THREAD), int(MAX_THREAD_NUM)));
    HCCL_INFO(
        "[TopoInfoExchangeDispatcher][InitWorkerThread]calculate threadNum[%u], rankNum[%d]", threadNum_, rankNum_);
    for (u32 i = 0; i < threadNum_; ++i) {
        auto th = std::thread(&TopoInfoExchangeDispatcher::RunWorkerThread, this, i);
        workerThreads_.emplace_back(std::move(th));
        HCCL_DEBUG("[TopoInfoExchangeDispatcher][InitWorkerThread]create thread[%u]", i);
    }
}

void TopoInfoExchangeDispatcher::WorkerWait(int workId)
{
    HCCL_DEBUG("[TopoInfoExchangeDispatcher][WorkerWait]start wait! workId[%d]", workId);
    std::unique_lock<std::mutex> lck(wakeMutex_);
    while (!ready_ && !stop_) {
        wakeManager_.wait(lck);
    }
    HCCL_DEBUG("[TopoInfoExchangeDispatcher][WorkerWait]finish wait! workId[%d]", workId);
}

bool TopoInfoExchangeDispatcher::GetTask(WorkerTask& workTask)
{
    auto& taskQueue = taskQueue_;
    std::unique_lock<std::mutex> lckForGetTask(taskQueueMutex_);
    if (taskQueue.empty()) {
        ready_ = false;
        return false;
    }
    workTask = taskQueue.front();
    taskQueue.pop();
    return true;
}

void TopoInfoExchangeDispatcher::RunWorkerThread(int workId)
{
    // 给当前线程添加名字
    SetThreadName("Hccl_RunWorker");

    while (!stop_) {
        WorkerWait(workId);
        while (true) {
            WorkerTask task;
            if (GetTask(task)) {
                task();
            } else {
                break;
            }
        }
    }
    HCCL_DEBUG("[TopoInfoExchangeDispatcher][RunWorkerThread]finish thread! workId[%d]", workId);
}

HcclResult TopoInfoExchangeDispatcher::PrepareResource(
    const std::map<std::string, std::shared_ptr<HcclSocket>> connectSockets, const RankTable_t& clusterInfo,
    const std::string& failedAgentIdList)
{
    rankNum_ = connectSockets.size();
    InitWorkerThread();

    HcclResult ret = hrtRaCreateEventHandle(epollFds_);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR(
            "[TopoInfoExchangeDispatcher][PrepareEpollResource]hrtRaCreateEventHandle create"
            " epollFds_ failed, ret[%d]",
            ret);
        return HCCL_E_TCP_TRANSFER;
    }

    nlohmann::json basicJson;
    CHK_RET(topoInfoExchangeServer_->TopoInfoExchangeBase::Struct2Json(clusterInfo, basicJson));
    basicJson[PROP_STEP] = topoInfoExchangeServer_->TopoInfoExchangeBase::currentStep_;
    if (!failedAgentIdList.empty()) {
        basicJson["fault_info"] = "[" + failedAgentIdList + "]";
        basicJson["fault_type"] = static_cast<int>(TopoDetectResult::TOPO_CONNECT_FAILED);
    }
    rankTableJson_ = basicJson.dump();

    u32 socketIndex = 0; // socket已经经过rankid（or serverip +deviceid排序）
    for (auto it : connectSockets) {
        FdContext fdcontext;
        fdcontext.socket = it.second;
        if (topoInfoExchangeServer_->TopoInfoExchangeBase::isByMasterInfo_) { // masterInfo场景下无法获取rankid
            fdcontext.txState.identify = socketIndex;
        }
        fdcontext.txState.bodyLen = rankTableJson_.length();
        fdcontext.txState.data = &rankTableJson_[0];
        fdcontext.txState.rankId = socketIndex;
        socketIndex++;
        HCCL_DEBUG(
            "[TopoInfoExchangeDispatcher][PrepareResource]socketIndex:%u, bodyLen:%u, data:%u", socketIndex,
            fdcontext.txState.bodyLen, fdcontext.txState.data);
        fdHandleToFdContextMap_.emplace(it.second->GetFdHandle(), fdcontext);
    }

    HCCL_DEBUG(
        "[TopoInfoExchangeDispatcher][PrepareEpollResource]fdHandleToFdContextMap_ size[%d]",
        fdHandleToFdContextMap_.size());
    return HCCL_SUCCESS;
}

HcclResult TopoInfoExchangeDispatcher::PrepareLeaderResource(
    const std::map<std::string, std::shared_ptr<HcclSocket>> connectSockets, const GroupLeader_t& leaderInfo)
{
    rankNum_ = connectSockets.size();
    InitWorkerThread();

    HcclResult ret = hrtRaCreateEventHandle(epollFds_);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR(
            "[TopoInfoExchangeDispatcher][PrepareEpollResource]hrtRaCreateEventHandle create"
            " epollFds_ failed, ret[%d]",
            ret);
        return HCCL_E_TCP_TRANSFER;
    }

    nlohmann::json basicJson;
    CHK_RET(topoInfoExchangeServer_->TopoInfoExchangeBase::GrpLeader2Json(leaderInfo, basicJson));
    basicJson[PROP_STEP] = topoInfoExchangeServer_->TopoInfoExchangeBase::currentStep_;
    rankTableJson_ = basicJson.dump();

    u32 socketIndex = 0; // socket已经经过rankid（or serverip +deviceid排序）
    for (auto it : connectSockets) {
        FdContext fdContext;
        fdContext.socket = it.second;
        if (topoInfoExchangeServer_->TopoInfoExchangeBase::isByMasterInfo_) { // masterInfo场景下无法获取rankid
            fdContext.txState.identify = socketIndex;
        }
        fdContext.txState.bodyLen = rankTableJson_.length();
        fdContext.txState.data = &rankTableJson_[0];
        fdContext.txState.rankId = socketIndex;
        socketIndex++;
        HCCL_DEBUG(
            "[TopoInfoExchangeDispatcher][PrepareLeaderResource]socketIndex:%u, bodyLen:%u, data:%u", socketIndex,
            fdContext.txState.bodyLen, fdContext.txState.data);
        fdHandleToFdContextMap_.emplace(it.second->GetFdHandle(), fdContext);
    }

    HCCL_DEBUG(
        "[TopoInfoExchangeDispatcher][PrepareEpollResource]fdHandleToFdContextMap_ size[%d]",
        fdHandleToFdContextMap_.size());
    return HCCL_SUCCESS;
}

void TopoInfoExchangeDispatcher::WakeWoker()
{
    std::unique_lock<std::mutex> lck(wakeMutex_);
    ready_ = true;
    wakeManager_.notify_all();
}

void TopoInfoExchangeDispatcher::CleanResource()
{
    // 主线程广播结束，结束从线程（不确定是否存在出于wait状态的线程，统一全部唤醒）
    stop_ = true;
    WakeWoker();
    HCCL_INFO("[TopoInfoExchangeDispatcher][PrepareEpollResource]wake all workers.");
    for (auto& th : workerThreads_) {
        if (th.joinable()) {
            th.join();
        }
    }
    fdHandleToFdContextMap_.clear();
    workerThreads_.clear();
}

HcclResult TopoInfoExchangeDispatcher::CloseEpollFd()
{
    HcclResult ret = hrtRaDestroyEventHandle(epollFds_);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR(
            "[TopoInfoExchangeDispatcher][CloseEpollFd]DestroyEventHandle destroy "
            "epollFds_ failed, ret[%d]",
            ret);
        return HCCL_E_TCP_TRANSFER;
    }
    return HCCL_SUCCESS;
}

HcclResult TopoInfoExchangeDispatcher::ProcessOneSendEvent([[maybe_unused]] s32 epollFd, FdHandle& fdHandle)
{
    std::unique_lock<std::mutex> lckForMap(fdHandleMapMutex_);
    if (fdHandleToFdContextMap_.find(fdHandle) == fdHandleToFdContextMap_.end()) {
        HCCL_ERROR("[TopoInfoExchangeDispatcher][ProcessOneSendEvent]no fdhandle[%p]", fdHandle);
        stop_ = true;
        return HCCL_E_INTERNAL;
    }
    auto ctx = &(fdHandleToFdContextMap_.at(fdHandle));
    if (ctx->txState.Send(ctx->socket) != 0) {
        HCCL_ERROR(
            "[TopoInfoExchangeDispatcher][ProcessOneSendEvent]send data to rank[%u] failed.", ctx->txState.rankId);
        stop_ = true;
        return HCCL_E_INTERNAL;
    }

    int ctlType = EPOLL_CTL_DEL;
    if (ctx->txState.IsOk()) {
        sendDoneCount_++;
    } else {
        ctlType = EPOLL_CTL_MOD;
    }
    // EPOLLOUT_LET_ONESHOT -> EPOLLOUT | EPOLLET | EPOLLONESHOT, 防止多个线程同时操作同一个fd（fd重复触发）
    HcclResult ret = hrtRaCtlEventHandle(epollFds_, fdHandle, ctlType, HcclEpollEvent::HCCL_EPOLLOUT_LET_ONESHOT);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR(
            "[TopoInfoExchangeDispatcher][ProcessOneSendEvent]epoll_ctl delete/modify "
            "failed, ctlType[%d]",
            ctlType);
        stop_ = true;
        return HCCL_E_INTERNAL;
    }
    return HCCL_SUCCESS;
}

HcclResult TopoInfoExchangeDispatcher::SendOnce()
{
    HcclResult ret;
    for (auto& it : fdHandleToFdContextMap_) {
        auto fdCtx = &(it.second);
        if (fdCtx->txState.Send(fdCtx->socket) != 0) {
            HCCL_ERROR("[TopoInfoExchangeDispatcher][SendOnce]Send data to rank[%u] failed.", fdCtx->txState.rankId);
            stop_ = true;
            return HCCL_E_INTERNAL;
        }

        // 数据未发送完成，添加epoll事件
        if (!fdCtx->txState.IsOk()) {
            // EPOLLOUT_LET_ONESHOT -> EPOLLOUT | EPOLLET | EPOLLONESHOT, 防止多个线程同时操作同一个fd（fd重复触发）
            ret = hrtRaCtlEventHandle(epollFds_, it.first, EPOLL_CTL_ADD, HcclEpollEvent::HCCL_EPOLLOUT_LET_ONESHOT);
            if (ret != HCCL_SUCCESS) {
                HCCL_ERROR("[TopoInfoExchangeDispatcher][SendOnce]epoll_ctl add fd failed.");
                stop_ = true;
                return HCCL_E_INTERNAL;
            }
        } else {
            sendDoneCount_++;
        }
    }
    return HCCL_SUCCESS;
}

HcclResult TopoInfoExchangeDispatcher::ProcessSend()
{
    HcclResult ret = SendOnce(); // 先尝试发送数据
    CHK_RET(ret);
    HCCL_INFO(
        "[TopoInfoExchangeDispatcher][ProcessSend]sendOnce success, start epoll_wait."
        " sendDoneCount[%d], rankNum[%d].",
        sendDoneCount_.load(), rankNum_);
    const int sendEvsCount = 20; // epoll_wait 缓冲区大小（单次触发的事件个数）
    std::vector<SocketEventInfo> eventInfos(sendEvsCount);
    bool lastEpollWaitFlag = false; // 最后一轮epoll_wait标识位
    auto startTime = std::chrono::steady_clock::now();
    auto timeout = std::chrono::seconds(GetExternalInputHcclLinkTimeOut());
    while (sendDoneCount_ != rankNum_) {
        if (stop_) {
            return HCCL_E_INTERNAL;
        }
        if (rankNum_ - sendDoneCount_ < sendEvsCount && !lastEpollWaitFlag) { // 最后一轮epoll_wait
            lastEpollWaitFlag = true;
        }
        if ((std::chrono::steady_clock::now() - startTime) >= timeout) {
            HCCL_ERROR("[TopoInfoExchangeDispatcher][ProcessSend]epoll_wait timeout!");
            return HCCL_E_INTERNAL;
        }
        s32 epollTimeout = lastEpollWaitFlag ? LAST_EPOLL_TIMEOUT_MS : EPOLL_TIMEOUT_MS;
        u32 eventsNum{0};
        ret = hrtRaWaitEventHandle(epollFds_, eventInfos, epollTimeout, sendEvsCount, eventsNum);
        if (eventsNum == 0 && ret == HCCL_SUCCESS && sendDoneCount_ == rankNum_) {
            // 最后一轮epoll_wait结束, 等待超时，epoll池内无事件
            HCCL_WARNING(
                "[TopoInfoExchangeDispatcher][ProcessSend]hrtRaWaitEventHandle is timeout[%d] ms, "
                "eventsNum[%u], ret[%d], sendDoneCount_[%d]",
                epollTimeout, eventsNum, ret, sendDoneCount_.load());
            return HCCL_SUCCESS;
        }
        if (eventsNum <= 0 && ret != HCCL_SUCCESS) {
            HCCL_ERROR(
                "[TopoInfoExchangeDispatcher][ProcessSend]hrtRaWaitEventHandle failed, eventsNum[%u], "
                "ret[%d]",
                eventsNum, ret);
            return HCCL_E_INTERNAL;
        }
        for (u32 i = 0; i < eventsNum; ++i) {
            std::unique_lock<std::mutex> lck(taskQueueMutex_);
            taskQueue_.push(
                std::bind(&TopoInfoExchangeDispatcher::ProcessOneSendEvent, this, epollFds_, eventInfos[i].fdHandle));
            lck.unlock();
        }
        // 唤醒处理
        WakeWoker();
    }
    CHK_RET(CloseEpollFd());
    return HCCL_SUCCESS;
}

HcclResult TopoInfoExchangeDispatcher::SendState::Send(std::shared_ptr<HcclSocket> socket)
{
    if (headerSended != headerLen) {
        header = bodyLen;
        CHK_RET(SendHeader(socket));
    }

    if ((headerSended == headerLen) && (bodyLen != bodySended)) {
        CHK_RET(SendBody(socket));
    }

    if ((headerSended == headerLen) && (bodyLen == bodySended)
        && (identify != UINT_MAX && identifyLen != identifySended)) {
        CHK_RET(SendIdentify(socket));
    }

    return HCCL_SUCCESS;
}

HcclResult TopoInfoExchangeDispatcher::SendState::SendHeader(std::shared_ptr<HcclSocket> socket)
{
    return SendHelper(socket, reinterpret_cast<char*>(&header), headerLen, headerSended);
}

HcclResult TopoInfoExchangeDispatcher::SendState::SendBody(std::shared_ptr<HcclSocket> socket)
{
    return SendHelper(socket, reinterpret_cast<char*>(data), bodyLen, bodySended);
}

HcclResult TopoInfoExchangeDispatcher::SendState::SendIdentify(std::shared_ptr<HcclSocket> socket)
{
    return SendHelper(socket, reinterpret_cast<char*>(&identify), identifyLen, identifySended);
}

HcclResult TopoInfoExchangeDispatcher::SendState::SendHelper(
    std::shared_ptr<HcclSocket> socket, char* buf, size_t dataLen, size_t& sendedLen)
{
    CHK_SMART_PTR_NULL(socket);
    CHK_PTR_NULL(buf);
    u64 needSend = dataLen - sendedLen;
    u64 sentSize = 0;
    HcclResult ret = socket->ISend(buf + sendedLen, needSend, sentSize);
    if (ret == HCCL_E_NETWORK) {
        HCCL_ERROR("[TopoInfoExchangeDispatcher][SendState][SendHelper]SendHelper fail error[%d].", ret);
        return HCCL_E_TCP_TRANSFER;
    }
    if (ret != HCCL_SUCCESS && ret != HCCL_E_AGAIN) {
        HCCL_ERROR("[TopoInfoExchangeDispatcher][SendState][SendHelper]socket send fail error[%d].", ret);
        return HCCL_E_INTERNAL;
    }
    if (ret == HCCL_SUCCESS) {
        sendedLen += sentSize;
    }
    return HCCL_SUCCESS;
}

} // namespace hccl
