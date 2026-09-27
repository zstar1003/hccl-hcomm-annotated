/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "flush_manager.h"
#include "env_config/env_config_v2.h"

namespace Hccl {
FlushManager::FlushManager() {}

FlushManager& FlushManager::GetInstance()
{
    static FlushManager flushManager;
    return flushManager;
}

FlushManager::~FlushManager() { DestroyAll(); }

HcclResult FlushManager::initFlushHandle(IpAddress ip, u32 devPhyId)
{
    HCCL_INFO("[initFlushHandle]FlushHandle init start.");
    if (flushHandleMap_.find(ip) != flushHandleMap_.end()) {
        HCCL_INFO("[initFlushHandle]FlushHandle already exists");
        return HCCL_SUCCESS;
    }

    auto flushHandlePtr = std::make_shared<FlushHandle>();
    HcclResult ret = flushHandlePtr->Init(ip, devPhyId);
    if (ret != HCCL_SUCCESS) {
        HCCL_INFO("[initFlushHandle]FlushHandle init fail.");
        return ret;
    }

    flushHandleMap_.insert({ip, flushHandlePtr});
    HCCL_INFO("[initFlushHandle]FlushHandle init success.");
    return HCCL_SUCCESS;
}

HcclResult FlushManager::DestroyAll()
{
    if (flushHandleMap_.empty()) {
        HCCL_DEBUG("flushHandleMap_ is empty");
        return HCCL_SUCCESS;
    }
    for (auto item : flushHandleMap_) {
        auto flushHandlePtr = item.second;
        HcclResult ret = flushHandlePtr->Destroy();
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR("[DestroyAll]Failed to destroy flush resources. Error: %d", ret);
            return ret;
        }
    }
    flushHandleMap_.clear();
    HCCL_INFO("[DestroyAll]FlushHandle destroy success.");
    return HCCL_SUCCESS;
}

HcclResult FlushManager::Flush()
{
    std::lock_guard<std::mutex> lock(mutex_);
    HCCL_INFO("[Flush] Start: Entering Flush function.");
    if (flushHandleMap_.empty()) {
        HCCL_INFO("[Flush] No FLUSH is needed to be executed.");
        return HCCL_SUCCESS;
    }

    for (auto item : flushHandleMap_) {
        auto flushHandlePtr = item.second;

        ibv_qp* loopbackqp0 = static_cast<ibv_qp*>(flushHandlePtr->loopBackQpParam.ibvQp0);
        CHK_PTR_NULL(loopbackqp0);
        ibv_cq* cq = loopbackqp0->send_cq;
        CHK_PTR_NULL(cq);
        HCCL_DEBUG("[Flush] Successfully retrieved QP and CQ handles: qp=%p, cq=%p", loopbackqp0, cq);

        // 接口数据设置
        ibv_send_wr swr{};
        ibv_sge sg_list{};
        swr.sg_list = &sg_list;
        HcclResult paramsRet = FlushParamPrepare(flushHandlePtr, &swr);
        if (paramsRet != HCCL_SUCCESS) {
            HCCL_INFO("[Flush] Set work request failed.");
            return paramsRet;
        }
        HCCL_DEBUG("[FlushParamPrepare] Posting RDMA_READ operation... ");

        // 执行读和轮训操作
        HcclResult loopQpRet
            = ExecuteRdmaRead(loopbackqp0, cq, swr, Hccl::EnvConfig::GetInstance().GetRtsConfig().GetExecTimeOut());
        if (loopQpRet != HCCL_SUCCESS) {
            HCCL_INFO("[Flush] RDMA_READ operation failed.");
            return loopQpRet;
        }
    }
    HCCL_INFO("[Flush] Successfully completed: RDMA_READ operation finished.");
    return HCCL_SUCCESS;
}

HcclResult FlushManager::FlushParamPrepare(std::shared_ptr<FlushHandle> flushHandlePtr, ibv_send_wr* swr) const
{
    CHK_PTR_NULL(swr);
    swr->wr_id = 0;
    CHK_PTR_NULL(swr->sg_list);
    swr->sg_list->addr = reinterpret_cast<uint64_t>(flushHandlePtr->loopBackQpMrLocalInfo.addr);
    swr->sg_list->length = flushHandlePtr->loopBackQpMrLocalInfo.size;
    swr->sg_list->lkey = flushHandlePtr->loopBackQpMrLocalInfo.lkey;
    swr->next = nullptr;
    swr->num_sge = 1;
    swr->opcode = (flushHandlePtr->GetFlushOpcodeSupport()) ? ROCE_WR_FLUSH : IBV_WR_RDMA_READ;
    swr->send_flags = IBV_SEND_SIGNALED;
    swr->wr.rdma.remote_addr = reinterpret_cast<uint64_t>(flushHandlePtr->loopBackQpMrRemoteInfo.addr);
    swr->wr.rdma.rkey = flushHandlePtr->loopBackQpMrRemoteInfo.rkey;
    return HCCL_SUCCESS;
}

HcclResult FlushManager::ExecuteRdmaRead(ibv_qp* loopbackqp0, ibv_cq* cq, ibv_send_wr& swr, int timeoutSec) const
{
    ibv_send_wr* send_wr = nullptr;
    int ret = FlushPostSend(loopbackqp0, &swr, &send_wr);
    if (ret != 0) {
        HCCL_ERROR("[ExecuteRdmaRead] ibv_post_send failed: %s", strerror(errno));
        return HCCL_E_NETWORK;
    }

    HCCL_DEBUG("[ExecuteRdmaRead] RDMA_READ posted successfully. Starting polling for completion...");
    ibv_wc wc{};
    struct timespec start;
    struct timespec current;
    clock_gettime(CLOCK_MONOTONIC, &start);
    while (true) {
        // 计算已流逝时间（秒）
        clock_gettime(CLOCK_MONOTONIC, &current);
        int elapsedSec = (current.tv_sec - start.tv_sec) + (current.tv_nsec - start.tv_nsec) / 1000000000;

        // 超时判断
        if (elapsedSec >= timeoutSec) {
            HCCL_ERROR(
                "[ExecuteRdmaRead] Failed: Wait for completion queue timeout (elapsed=%d s, max=%d s)", elapsedSec,
                timeoutSec);
            return HCCL_E_TIMEOUT;
        }

        // 轮询 CQ
        int numCqes = FlushPollCq(cq, 1, &wc);
        if (numCqes < 0) {
            HCCL_ERROR("[ExecuteRdmaRead] ibv_poll_cq returned error: %s", strerror(errno));
            return HCCL_E_NETWORK;
        }
        // 成功收到完成事件
        if (numCqes > 0) {
            if (wc.status == IBV_WC_SUCCESS) {
                HCCL_DEBUG(
                    "[ExecuteRdmaRead] RDMA_READ completed successfully. "
                    "wr_id=%llu, status=%d",
                    wc.wr_id, wc.status);
                return HCCL_SUCCESS;
            } else {
                HCCL_ERROR("[ExecuteRdmaRead] RDMA_READ operation failed: status=%d, wr_id=%llu", wc.status, wc.wr_id);
                return HCCL_E_NETWORK;
            }
        }
    }
}

} // namespace Hccl
