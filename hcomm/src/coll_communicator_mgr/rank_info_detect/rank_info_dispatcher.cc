/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "rank_info_dispatcher.h"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <sys/socket.h>
#include <sys/epoll.h>
#include <unistd.h>
#include <ctime>
#include "sal.h"
#include "hccp.h"
#include "env_config/env_config_v2.h"
#include "hccp_common.h"
#include "network_api_exception.h"
#include "adapter_error_manager_pub.h"
#include "hccl_log_keywords.h"

namespace Hccl {

RankInfoDispatcher::~RankInfoDispatcher()
{
    DECTOR_TRY_CATCH("RankInfoDispatcher", CleanResource());
    DECTOR_TRY_CATCH("RankInfoDispatcher", CloseEpollFd());
}

void RankInfoDispatcher::BroadcastRankTable(
    const std::unordered_map<std::string, std::shared_ptr<Socket>>& connectSockets, const RankTableInfo& clusterInfo,
    const std::string& failedAgentIdList, u32 step)
{
    PrepareResource(connectSockets, clusterInfo, failedAgentIdList, step);
    ProcessSend();
    HCCL_INFO(
        "[RankInfoDispatcher::%s] broadcast topoinfo success, rankNum[%u], threadNum[%u]", __func__, rankNum_,
        threadNum_);
}

void RankInfoDispatcher::InitWorkerThread()
{
    threadNum_ = std::max(1, std::min(s32(rankNum_ / RANK_CAPACITY_PER_THREAD), s32(MAX_THREAD_NUM)));
    for (u32 i = 0; i < threadNum_; ++i) {
        auto th = std::thread(&RankInfoDispatcher::RunWorkerThread, this, i);
        workerThreads_.emplace_back(std::move(th));
    }
    HCCL_INFO("[RankInfoDispatcher::%s]calculate threadNum[%u], rankNum[%u]", __func__, threadNum_, rankNum_);
}

void RankInfoDispatcher::WorkerWait(s32 workId)
{
    HCCL_DEBUG("[RankInfoDispatcher::%s]start wait! workId[%d]", __func__, workId);
    std::unique_lock<std::mutex> lck(wakeMutex_);
    while (!ready_ && !stop_) {
        wakeManager_.wait(lck);
    }
    HCCL_DEBUG("[RankInfoDispatcher::%s]finish wait! workId[%d]", __func__, workId);
}

bool RankInfoDispatcher::GetTask(WorkerTask& workTask)
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

void RankInfoDispatcher::RunWorkerThread(s32 workId)
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
    HCCL_DEBUG("[RankInfoDispatcher::%s]finish thread! workId[%d]", __func__, workId);
}

void RankInfoDispatcher::PrepareResource(
    const std::unordered_map<std::string, std::shared_ptr<Socket>> connectSockets, const RankTableInfo& clusterInfo,
    const std::string& failedAgentIdList, u32 step)
{
    rankNum_ = connectSockets.size();
    InitWorkerThread();

    s32 res = RaCreateEventHandle(&epollFds_);
    CHK_PRT_THROW(
        res != 0, HCCL_ERROR("[RankInfoDispatcher::%s] create epoll event failed, res[%d].", __func__, res),
        NetworkApiException, "create epoll event error.");
    epollCreate_ = true;

    BinaryStream binaryStream;
    clusterInfo.GetBinStream(true, binaryStream);
    binaryStream << step;
    binaryStream << failedAgentIdList;

    binaryStream.Dump(rankTableMsg_);

    for (auto& it : connectSockets) {
        FdContext fdcontext;
        fdcontext.socket = it.second;
        fdcontext.txState.bodyLen = rankTableMsg_.size();
        fdcontext.txState.data = rankTableMsg_.data();
        CHK_RET_THROW(
            InvalidParamsException,
            StringFormat("[RankInfoDispatcher::%s] ranid[%s] strToULong fail.", __func__, it.first.c_str()),
            SalStrToULong(it.first, HCCL_BASE_DECIMAL, fdcontext.txState.rankId));
        HCCL_DEBUG(
            "[RankInfoDispatcher::%s]rankId:%u, bodyLen:%u", __func__, fdcontext.txState.rankId,
            fdcontext.txState.bodyLen);
        fdHandleToFdContextMap_.emplace(it.second->GetFdHandle(), fdcontext);
    }

    HCCL_INFO("[RankInfoDispatcher::%s]fdHandleToFdContextMap_ size[%d]", __func__, fdHandleToFdContextMap_.size());
}

void RankInfoDispatcher::WakeWoker()
{
    std::unique_lock<std::mutex> lck(wakeMutex_);
    ready_ = true;
    wakeManager_.notify_all();
}

void RankInfoDispatcher::CleanResource()
{
    // 主线程广播结束，结束从线程（不确定是否存在出于wait状态的线程，统一全部唤醒）
    stop_ = true;
    WakeWoker();
    HCCL_INFO("[RankInfoDispatcher::%s]wake all workers.", __func__);
    for (auto& th : workerThreads_) {
        if (th.joinable()) {
            th.join();
        }
    }
    fdHandleToFdContextMap_.clear();
    workerThreads_.clear();
}

void RankInfoDispatcher::ProcessOneSendEvent([[maybe_unused]] s32 epollFd, FdHandle& fdHanlde)
{
    std::unique_lock<std::mutex> lckForMap(fdHandleMapMutex_);
    bool noFdHandle = (fdHandleToFdContextMap_.find(fdHanlde) == fdHandleToFdContextMap_.end());
    if (noFdHandle) {
        stop_ = true;
    }
    CHK_PRT_RET(noFdHandle, HCCL_ERROR("[RankInfoDispatcher::%s]no fdhandle[%p]", __func__, fdHanlde), );
    auto ctx = &(fdHandleToFdContextMap_.at(fdHanlde));
    bool sendFailed = !ctx->txState.Send(ctx->socket);
    if (sendFailed) {
        stop_ = true;
    }
    CHK_PRT_RET(
        sendFailed,
        HCCL_ERROR("[RankInfoDispatcher::%s]send data to rank[%u] failed.", __func__, ctx->txState.rankId), );

    s32 ctlType = EPOLL_CTL_DEL;
    if (ctx->txState.IsOk()) {
        sendDoneCount_++;
    } else {
        ctlType = EPOLL_CTL_MOD;
    }
    // EPOLLOUT_LET_ONESHOT -> EPOLLOUT | EPOLLET | EPOLLONESHOT, 防止多个线程同时操作同一个fd（fd重复触发）
    s32 ret = RaCtlEventHandle(epollFds_, fdHanlde, ctlType, RaEpollEvent::RA_EPOLLOUT_LET_ONESHOT);
    bool ctlFailed = (ret != 0);
    if (ctlFailed) {
        stop_ = true;
    }
    CHK_PRT_RET(ctlFailed, HCCL_ERROR("[RankInfoDispatcher::%s]epoll_ctl failed, ctlType[%d]", __func__, ctlType), );
}

void RankInfoDispatcher::SendOnce()
{
    for (auto& it : fdHandleToFdContextMap_) {
        auto fdCtx = &(it.second);
        CHK_PRT_THROW(
            !fdCtx->txState.Send(fdCtx->socket),
            HCCL_ERROR("[RankInfoDispatcher::%s]Send data to rank[%u] failed.", __func__, fdCtx->txState.rankId),
            InvalidParamsException, "send data error.");

        // 数据未发送完成，添加epoll事件
        if (!fdCtx->txState.IsOk()) {
            // EPOLLOUT_LET_ONESHOT -> EPOLLOUT | EPOLLET | EPOLLONESHOT, 防止多个线程同时操作同一个fd（fd重复触发）
            s32 ret = RaCtlEventHandle(epollFds_, it.first, EPOLL_CTL_ADD, RaEpollEvent::RA_EPOLLOUT_LET_ONESHOT);
            CHK_PRT_THROW(
                ret != 0, HCCL_ERROR("[RankInfoDispatcher::%s]epoll_ctl add fd failed.", __func__),
                InvalidParamsException, "send data error.");
        } else {
            sendDoneCount_++;
        }
    }
}

void RankInfoDispatcher::ProcessSend()
{
    SendOnce(); // 先尝试发送数据
    HCCL_INFO(
        "[RankInfoDispatcher::%s]sendOnce success, start epoll_wait. sendDoneCount[%d], rankNum[%u].", __func__,
        sendDoneCount_.load(), rankNum_);
    const s32 sendEvsCount = 20; // epoll_wait 缓冲区大小（单次触发的事件个数）
    std::vector<SocketEventInfo> eventInfos(sendEvsCount);
    bool lastEpollWaitFlag = false; // 最后一轮epoll_wait标识位
    auto timeout = std::chrono::seconds(EnvConfig::GetInstance().GetSocketConfig().GetLinkTimeOut());
    auto startTime = std::chrono::steady_clock::now();
    HcclResult ret;
    while (sendDoneCount_ != rankNum_) {
        CHK_PRT_THROW(
            stop_, HCCL_ERROR("[RankInfoDispatcher::%s] process stop.", __func__), InvalidParamsException,
            "process stop.");

        if (rankNum_ - sendDoneCount_ < sendEvsCount && !lastEpollWaitFlag) { // 最后一轮epoll_wait
            lastEpollWaitFlag = true;
        }

        // 循环超时
        if ((std::chrono::steady_clock::now() - startTime) >= timeout) {
            RPT_INPUT_ERR(
                true, "EI0015", std::vector<std::string>({"error_reason"}),
                std::vector<std::string>({StringFormat(
                    "Receiving message from the root node timed out "
                    "Timeout was set to %lld seconds. Expected to send to %u nodes, completed %u nodes.",
                    static_cast<long long>(timeout.count()), rankNum_, sendDoneCount_.load())}));
            HCCL_ERROR(
                "[%s][%s] errNo[0x%016llx] topo exchange server get socket timeout, timeout[%lld s], "
                "expected[%u] nodes, completed[%u] nodes",
                LOG_KEYWORDS_INIT_GROUP.c_str(), LOG_KEYWORDS_RANKTABLE_DETECT.c_str(),
                HCOM_ERROR_CODE(HcclResult::HCCL_E_TIMEOUT), static_cast<long long>(timeout.count()), rankNum_,
                sendDoneCount_.load());
            THROW<TimeoutException>("epoll_wait timeout");
        }

        // 等待epoll事件
        s32 epollTimeout = lastEpollWaitFlag ? LAST_EPOLL_TIMEOUT_MS : EPOLL_TIMEOUT_MS;
        u32 eventsNum{0};
        ret = HrtRaWaitEventHandle(epollFds_, eventInfos, epollTimeout, sendEvsCount, eventsNum);

        // 最后一轮epoll_wait结束, 等待超时，epoll池内无事件
        CHK_PRT_RET(
            (eventsNum == 0 && ret == HCCL_SUCCESS && sendDoneCount_ == rankNum_),
            HCCL_WARNING(
                "[RankInfoDispatcher::%s]hrtRaWaitEventHandle is timeout[%d] ms, eventsNum[%u], "
                "sendDoneCount_[%d]",
                __func__, epollTimeout, eventsNum, sendDoneCount_.load()), );

        // epoll wait事件失败
        // 可能出现ret==HCCL_SUCCESS但eventsNum==0的情况，属于正常情况，不报错退出，需要继续循环
        CHK_PRT_THROW(
            ret != HCCL_SUCCESS,
            HCCL_ERROR(
                "[RankInfoDispatcher::%s] HrtRaWaitEventHandle failed ret[%d], eventsNum[%u].", __func__, ret,
                eventsNum),
            InvalidParamsException, "epoll_wait fail");
        for (u32 i = 0; i < eventsNum; ++i) {
            std::unique_lock<std::mutex> lck(taskQueueMutex_);
            taskQueue_.push(std::bind(
                &RankInfoDispatcher::ProcessOneSendEvent, this, epollFds_, static_cast<void*>(eventInfos[i].fdHandle)));
            lck.unlock();
        }
        // 唤醒处理
        WakeWoker();
    }

    CloseEpollFd();
    HCCL_INFO(
        "[RankInfoDispatcher::%s]ProcessSend success, sendDoneCount[%d], rankNum[%d].", __func__, sendDoneCount_.load(),
        rankNum_);
}

void RankInfoDispatcher::CloseEpollFd()
{
    if (epollCreate_) {
        s32 ret = RaDestroyEventHandle(&epollFds_);
        CHK_PRT_THROW(
            ret != 0, HCCL_ERROR("[RankInfoDispatcher::%s] destroy epoll event failed, res[%d].", __func__, ret),
            NetworkApiException, "destroy epoll event error.");
        epollCreate_ = false;
    }
}

bool RankInfoDispatcher::SendState::Send(std::shared_ptr<Socket> socket)
{
    if (headerSended != headerLen) {
        header = bodyLen;
        CHK_PRT_RET(!SendHeader(socket), HCCL_ERROR("SendHeader error"), false);
    }

    if ((headerSended == headerLen) && (bodyLen != bodySended)) {
        CHK_PRT_RET(!SendBody(socket), HCCL_ERROR("SendBody error"), false);
    }

    return true;
}

bool RankInfoDispatcher::SendState::SendHeader(std::shared_ptr<Socket> socket)
{
    return SendHelper(socket, &header, headerLen, headerSended);
}

bool RankInfoDispatcher::SendState::SendBody(std::shared_ptr<Socket> socket)
{
    return SendHelper(socket, data, bodyLen, bodySended);
}

bool RankInfoDispatcher::SendState::SendHelper(
    std::shared_ptr<Socket> socket, void* buf, size_t dataLen, size_t& sendedLen)
{
    u64 needSend = dataLen - sendedLen;
    u64 sentSize = 0;
    u8* sendData = static_cast<u8*>(buf) + sendedLen;
    CHK_PRT_RET(!socket->ISend(sendData, needSend, sentSize), HCCL_ERROR("ISend fail"), false);
    sendedLen += sentSize;
    return true;
}

} // namespace Hccl
