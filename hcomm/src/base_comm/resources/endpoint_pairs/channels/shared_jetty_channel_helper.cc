/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "shared_jetty_channel_helper.h"
#include "log.h"
#include "adapter_rts_common.h"
#include <chrono>
#include <thread>

namespace hcomm {

// 轮询临时 connection 状态机到 EXCHANGEABLE（jetty 已创建），或超时失败
static HcclResult WaitForJettyCreated(Hccl::DevUbConnection& conn, uint32_t timeoutMs)
{
    auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeoutMs);
    while (true) {
        Hccl::RmaConnStatus st = conn.GetStatus();
        if (st == Hccl::RmaConnStatus::EXCHANGEABLE || st == Hccl::RmaConnStatus::READY) {
            return HCCL_SUCCESS;
        }
        if (st == Hccl::RmaConnStatus::CONN_INVALID) {
            HCCL_ERROR("[%s] temp connection became CONN_INVALID.", __func__);
            return HCCL_E_INTERNAL;
        }
        if (st == Hccl::RmaConnStatus::CLOSE) {
            HCCL_ERROR("[%s] temp connection CLOSED.", __func__);
            return HCCL_E_INTERNAL;
        }
        if (std::chrono::steady_clock::now() >= deadline) {
            HCCL_ERROR("[%s] wait jetty create timeout[%ums].", __func__, timeoutMs);
            return HCCL_E_TIMEOUT;
        }
        constexpr uint32_t kRetrySleepMs = 2;
        std::this_thread::sleep_for(std::chrono::milliseconds(kRetrySleepMs));
    }
}

// 分配 device 内存并清零：PI/CI 必须初值为 0，否则生产者索引非 0 会导致首条 WQE 越界
static HcclResult AllocAndZeroQueueIndex(void** ptr, uint64_t size, const char* name)
{
    HcclResult allocRet = hrtMalloc(ptr, size);
    if (allocRet != HCCL_SUCCESS || *ptr == nullptr) {
        HCCL_ERROR("[%s] hrtMalloc %s failed, ret[%d].", __func__, name, allocRet);
        return HCCL_E_MEMORY;
    }
    aclError memsetRet = aclrtMemset(*ptr, size, 0, size);
    if (memsetRet != ACL_SUCCESS) {
        HCCL_ERROR("[%s] aclrtMemset %s failed, ret[%d].", __func__, name, memsetRet);
        (void)hrtFree(*ptr);
        *ptr = nullptr;
        return HCCL_E_MEMORY;
    }
    return HCCL_SUCCESS;
}

// 首次创建共享 jetty：用临时 connection 走完整 jetty 创建流程，分配共享 PI/CI device 内存
static HcclResult
ProvideSharedJettyCtx(const TempConnFactory& tempConnFactory, uint64_t sharedQueueIndexMemSize, JettyContext::Ctx& ctx)
{
    std::unique_ptr<Hccl::DevUbConnection> tempConn = tempConnFactory();
    CHK_SMART_PTR_NULL(tempConn);
    // 16s 覆盖 TP 建链(~8s) + jetty 异步创建(~8s) 的串行最坏时长, 与 jettyTimeOut hwValue=16(=8s) 配套
    constexpr uint32_t kJettyCreateTimeOutMs = 16000;
    CHK_RET(WaitForJettyCreated(*tempConn, kJettyCreateTimeOutMs));

    // 通过适配层提取 jetty 字段（含 sqDepth），不直接调 legacy GetJettyInfo
    CHK_RET(ExtractJettyInfoFromConn(tempConn.get(), ctx));

    // 分配共享 PI/CI device 内存并清零，供同 endpoint 下后续 channel 复用。
    // 失败时释放已分配的指针，避免 device 内存泄漏。
    auto cleanup = [&ctx]() {
        if (ctx.sqPiPtr != nullptr) {
            (void)hrtFree(ctx.sqPiPtr);
            ctx.sqPiPtr = nullptr;
        }
        if (ctx.sqCiPtr != nullptr) {
            (void)hrtFree(ctx.sqCiPtr);
            ctx.sqCiPtr = nullptr;
        }
        if (ctx.cqPiPtr != nullptr) {
            (void)hrtFree(ctx.cqPiPtr);
            ctx.cqPiPtr = nullptr;
        }
        if (ctx.cqCiPtr != nullptr) {
            (void)hrtFree(ctx.cqCiPtr);
            ctx.cqCiPtr = nullptr;
        }
    };
    struct QueueIndexEntry {
        void** ptr;
        const char* name;
    };
    QueueIndexEntry entries[] = {
        {&ctx.sqPiPtr, "sqPiPtr"},
        {&ctx.sqCiPtr, "sqCiPtr"},
        {&ctx.cqPiPtr, "cqPiPtr"},
        {&ctx.cqCiPtr, "cqCiPtr"},
    };
    for (const auto& entry : entries) {
        HcclResult allocRet = AllocAndZeroQueueIndex(entry.ptr, sharedQueueIndexMemSize, entry.name);
        if (allocRet != HCCL_SUCCESS) {
            cleanup();
            return allocRet;
        }
    }
    ctx.queueIndexMemSize = sharedQueueIndexMemSize;
    // 所有可能失败的操作完成后，再分离 jetty 所有权，阻止临时 connection 析构销毁 jetty。
    // 若提前 DetachJetty，后续 PI/CI 分配失败时 jetty 会因所有权已分离而泄漏。
    // DetachJetty 自身失败时也需调用 cleanup() 释放已分配的 PI/CI device 内存，避免泄漏。
    HcclResult transferRet = DetachConnJetty(tempConn.get());
    if (transferRet != HCCL_SUCCESS) {
        cleanup();
        return transferRet;
    }
    // 临时 connection 析构：ReleaseResource 跳过 DestroyJetty，ReleaseTp 释放 TP
    return HCCL_SUCCESS;
}

HcclResult AcquireSharedJettyForChannel(
    Endpoint* endpoint, Hccl::DevUbConnection* connection, const TempConnFactory& tempConnFactory,
    JettyContext::Ctx& outCtx)
{
    if (endpoint == nullptr || connection == nullptr) {
        return HCCL_E_PARA;
    }

    // 共享 jetty 下每个 channel 单 conn，PI/CI 每段大小 = connNum(1) * sizeof(void*)。
    // 同 endpoint 下多 channel 共用这块 PI/CI device 内存，避免各 channel 各自分配指向同一 SQ
    // 导致生产者索引无法协调前进、WQE 互相覆盖、doorbell 不前进、notify 超时。
    constexpr uint64_t sharedQueueIndexMemSize = sizeof(void*);
    auto provideCtx = [&tempConnFactory, sharedQueueIndexMemSize](JettyContext::Ctx& ctx) -> HcclResult {
        return ProvideSharedJettyCtx(tempConnFactory, sharedQueueIndexMemSize, ctx);
    };

    JettyContext::Ctx ctx{};
    JettyContext* jettyCtx = dynamic_cast<JettyContext*>(endpoint->GetCommQueueContext());
    CHK_PTR_NULL(jettyCtx);
    HcclResult ret = jettyCtx->Acquire(provideCtx, ctx);
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[%s] Acquire shared jetty failed, ret[%d].", __func__, ret), ret);
    outCtx = ctx;

    // 命中复用或首次创建完成：通过适配层填充 jetty 字段到主 connection
    // releaseCb: connection 销毁时通知 Endpoint 减引用计数
    auto releaseCb = [](void* tag) {
        Endpoint* ep = static_cast<Endpoint*>(tag);
        if (ep != nullptr) {
            JettyContext* jc = dynamic_cast<JettyContext*>(ep->GetCommQueueContext());
            if (jc != nullptr) {
                (void)jc->Release();
            }
        }
    };
    HcclResult injectRet = SetSharedJettyFieldsToConn(connection, ctx, endpoint, std::move(releaseCb));
    if (injectRet != HCCL_SUCCESS) {
        HCCL_ERROR("[%s] SetSharedJettyFields failed, ret[%d], rollback refCount.", __func__, injectRet);
        (void)jettyCtx->Release();
        return injectRet;
    }
    HCCL_INFO(
        "[%s] shared jetty acquired and fields set, handle[0x%llx], sqPi[%p] sqCi[%p] cqPi[%p] cqCi[%p].", __func__,
        static_cast<unsigned long long>(ctx.handle), ctx.sqPiPtr, ctx.sqCiPtr, ctx.cqPiPtr, ctx.cqCiPtr);
    return HCCL_SUCCESS;
}

} // namespace hcomm
