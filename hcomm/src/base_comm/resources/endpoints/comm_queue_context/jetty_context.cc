/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "jetty_context.h"
#include "log.h"
#include "adapter_rts_common.h"

namespace hcomm {
namespace {

    constexpr uint32_t JETTY_CREATE_TIMEOUT_SECONDS = 32; // 覆盖 provideCtx 内部 16s jetty 创建 + 余量
    bool IsSameRemoteJetty(const JettyContext::SharedRemoteJettyCtx& ctx, const uint8_t* remoteQpKey, uint32_t keySize)
    {
        return ctx.remoteQpKey.size() == keySize && std::memcmp(ctx.remoteQpKey.data(), remoteQpKey, keySize) == 0;
    }
} // namespace

JettyContext::~JettyContext()
{
    std::lock_guard<std::mutex> lk(mtx_);
    if (inner_.valid && inner_.handle != 0) {
        if (inner_.refCount == 0) {
            // 与 DevUbConnection::ReleaseResource 的 ctxValid 防御一致：进程退出/DeInit 阶段
            // RdmaHandleManager 可能已销毁，句柄可能已失效，用失效句柄 unimport/destroy 会崩溃。
            const bool rdmaValid = (inner_.rdmaHandle != nullptr)
                                   && Hccl::RdmaHandleManager::GetInstance().IsHandleValid(
                                       static_cast<Hccl::RdmaHandle>(inner_.rdmaHandle));
            if (!rdmaValid) {
                HCCL_WARNING(
                    "[JettyContext][~JettyContext] shared jetty still valid on destroy but rdmaHandle "
                    "invalid, skip unimport/destroy to avoid using invalid handle, handle[%llu].",
                    static_cast<unsigned long long>(inner_.handle));
                return;
            }
            HCCL_WARNING(
                "[JettyContext][~JettyContext] shared jetty still valid on destroy, handle[%llu], "
                "force destroy.",
                static_cast<unsigned long long>(inner_.handle));
            UnimportSharedRemoteJettys(inner_);
            DestroyJettyResources(inner_);
        } else {
            HCCL_WARNING(
                "[JettyContext][~JettyContext] shared jetty still in use, refCount[%u], handle[%llu], "
                "skip destroy to avoid use-after-free.",
                inner_.refCount, static_cast<unsigned long long>(inner_.handle));
        }
    }
}

HcclResult JettyContext::Acquire(const std::function<HcclResult(Ctx&)>& provideCtx, Ctx& outCtx)
{
    // 第一段（持锁）：检查是否已创建或正在创建。已创建则 refCount++ 返回；未创建则标记 creating。
    // 超时上限 32s（覆盖 provideCtx 内部 16s jetty 创建 + 余量），避免创建线程异常崩溃后其他线程永久阻塞。
    {
        std::unique_lock<std::mutex> lk(mtx_);
        if (!cv_.wait_for(lk, std::chrono::seconds(JETTY_CREATE_TIMEOUT_SECONDS), [this] {
                return inner_.valid || !inner_.creating;
            })) {
            HCCL_ERROR("[JettyContext][Acquire] wait for shared jetty creation timeout[32s].");
            return HCCL_E_TIMEOUT;
        }
        if (inner_.valid) {
            CHK_RET(InnerToCtx(inner_, outCtx));
            inner_.refCount++;
            HCCL_INFO(
                "[JettyContext][Acquire] reuse shared jetty, handle[%llu], refCount[%u]",
                static_cast<unsigned long long>(outCtx.handle), inner_.refCount);
            return HCCL_SUCCESS;
        }
        // 抢占创建权
        inner_.creating = true;
    }

    // 第二段（无锁）：执行首次创建回调（含网络建链 I/O，可能耗时数秒）。
    // 创建期间不持锁，其他线程的 Acquire 在 cv_ 上等待，Release 不被阻塞。
    Ctx createdCtx;
    HcclResult createRet = provideCtx(createdCtx);
    if (createRet != HCCL_SUCCESS) {
        std::lock_guard<std::mutex> lk(mtx_);
        inner_.creating = false;
        cv_.notify_all();
        HCCL_ERROR("[JettyContext][Acquire] provideCtx failed, ret[%d].", createRet);
        return createRet;
    }

    // 第三段（持锁）：写入缓存，清除 creating 标记，设置 refCount=1。
    {
        std::lock_guard<std::mutex> lk(mtx_);
        inner_.handle = createdCtx.handle;
        inner_.handlePtr = createdCtx.handlePtr;
        inner_.jettyId = createdCtx.jettyId;
        inner_.sqBuffVa = createdCtx.sqBuffVa;
        inner_.dbAddr = createdCtx.dbAddr;
        inner_.keySize = createdCtx.keySize;
        inner_.sqDepth = createdCtx.sqDepth;
        inner_.sqPiPtr = createdCtx.sqPiPtr;
        inner_.sqCiPtr = createdCtx.sqCiPtr;
        inner_.cqPiPtr = createdCtx.cqPiPtr;
        inner_.cqCiPtr = createdCtx.cqCiPtr;
        inner_.queueIndexMemSize = createdCtx.queueIndexMemSize;
        inner_.rdmaHandle = createdCtx.rdmaHandle;
        inner_.jfcHandle = createdCtx.jfcHandle;
        inner_.cqInfo = createdCtx.cqInfo;
        inner_.localPsn = createdCtx.localPsn;
        if (createdCtx.keySize > 0 && createdCtx.keySize <= Hccl::HRT_UB_QP_KEY_MAX_LEN) {
            errno_t cpyRet
                = memcpy_s(inner_.localQpKey, Hccl::HRT_UB_QP_KEY_MAX_LEN, createdCtx.localQpKey, createdCtx.keySize);
            if (cpyRet != EOK) {
                // memcpy_s 失败：inner_ 已写入部分字段（handle/PI·CI 指针等）但 valid 仍为 false，
                // 需销毁已写入的 device 资源避免泄漏，再清空 inner_ 让其他线程可重新创建。
                DestroyJettyResources(inner_);
                inner_ = Inner{};
                inner_.creating = false;
                cv_.notify_all();
                HCCL_ERROR("[JettyContext][Acquire] memcpy_s localQpKey failed, ret[%d].", cpyRet);
                return HCCL_E_INTERNAL;
            }
        }
        inner_.valid = true;
        inner_.creating = false;
        inner_.refCount = 1;
        HcclResult innerToCtxRet = InnerToCtx(inner_, outCtx);
        if (innerToCtxRet != HCCL_SUCCESS) {
            DestroyJettyResources(inner_);
            inner_ = Inner{};
            inner_.creating = false;
            cv_.notify_all();
            HCCL_ERROR("[JettyContext][Acquire] InnerToCtx failed, ret[%d].", innerToCtxRet);
            return innerToCtxRet;
        }
        cv_.notify_all();
    }
    HCCL_INFO(
        "[JettyContext][Acquire] created shared jetty, handle[%llu]", static_cast<unsigned long long>(outCtx.handle));
    return HCCL_SUCCESS;
}

HcclResult JettyContext::Release()
{
    std::lock_guard<std::mutex> lk(mtx_);
    if (!inner_.valid) {
        HCCL_WARNING("[JettyContext][Release] shared jetty already invalid, skip release.");
        return HCCL_SUCCESS;
    }
    if (inner_.refCount == 0) {
        HCCL_WARNING("[JettyContext][Release] refCount already 0, skip release.");
        return HCCL_SUCCESS;
    }
    inner_.refCount--;
    HCCL_INFO(
        "[JettyContext][Release] release shared jetty, handle[%llu], refCount[%u]",
        static_cast<unsigned long long>(inner_.handle), inner_.refCount);
    if (inner_.refCount == 0) {
        UnimportSharedRemoteJettys(inner_);
        DestroyJettyResources(inner_);
        inner_ = Inner{};
    }
    return HCCL_SUCCESS;
}

HcclResult JettyContext::AcquireSharedRemoteJetty(
    const uint8_t* remoteQpKey, uint32_t keySize, bool& needImport, uint64_t& handle, void*& handlePtr, uint32_t& tpn)
{
    CHK_PTR_NULL(remoteQpKey);
    CHK_PRT_RET(
        keySize == 0 || keySize > Hccl::HRT_UB_QP_KEY_MAX_LEN,
        HCCL_ERROR("[%s] invalid keySize[%u].", __func__, keySize), HCCL_E_PARA);

    std::lock_guard<std::mutex> lk(mtx_);
    CHK_PRT_RET(!inner_.valid, HCCL_ERROR("[%s] shared local jetty is invalid.", __func__), HCCL_E_INTERNAL);

    needImport = false;
    handle = 0;
    handlePtr = nullptr;
    tpn = 0;
    for (const auto& remoteCtx : inner_.remoteJettys) {
        if (!IsSameRemoteJetty(remoteCtx, remoteQpKey, keySize)) {
            continue;
        }
        if (remoteCtx.ready) {
            handle = remoteCtx.handle;
            handlePtr = remoteCtx.handlePtr;
            tpn = remoteCtx.tpn;
            HCCL_INFO(
                "[%s] reuse shared remote jetty, handle[%llu], tpn[%u].", __func__,
                static_cast<unsigned long long>(handle), tpn);
        }
        return HCCL_SUCCESS;
    }

    SharedRemoteJettyCtx remoteCtx;
    remoteCtx.remoteQpKey.assign(remoteQpKey, remoteQpKey + keySize);
    inner_.remoteJettys.push_back(std::move(remoteCtx));
    needImport = true;
    HCCL_INFO("[%s] reserve shared remote jetty import.", __func__);
    return HCCL_SUCCESS;
}

HcclResult JettyContext::PublishSharedRemoteJetty(
    const uint8_t* remoteQpKey, uint32_t keySize, uint64_t handle, void* handlePtr, uint32_t tpn)
{
    CHK_PTR_NULL(remoteQpKey);
    CHK_PTR_NULL(handlePtr);
    CHK_PRT_RET(
        keySize == 0 || keySize > Hccl::HRT_UB_QP_KEY_MAX_LEN || handle == 0,
        HCCL_ERROR(
            "[%s] invalid params, keySize[%u], handle[%llu].", __func__, keySize,
            static_cast<unsigned long long>(handle)),
        HCCL_E_PARA);

    std::lock_guard<std::mutex> lk(mtx_);
    for (auto& remoteCtx : inner_.remoteJettys) {
        if (!IsSameRemoteJetty(remoteCtx, remoteQpKey, keySize)) {
            continue;
        }
        remoteCtx.handle = handle;
        remoteCtx.handlePtr = handlePtr;
        remoteCtx.tpn = tpn;
        remoteCtx.ready = true;
        HCCL_INFO(
            "[%s] publish shared remote jetty, handle[%llu], tpn[%u].", __func__,
            static_cast<unsigned long long>(handle), tpn);
        return HCCL_SUCCESS;
    }
    HCCL_ERROR("[%s] shared remote jetty reservation not found.", __func__);
    return HCCL_E_NOT_FOUND;
}

void JettyContext::UnimportSharedRemoteJettys(Inner& inner) const
{
    if (inner.rdmaHandle == nullptr) {
        return;
    }
    for (auto& remoteCtx : inner.remoteJettys) {
        if (!remoteCtx.ready || remoteCtx.handle == 0) {
            continue;
        }
        Hccl::HrtRaUbUnimportJetty(
            static_cast<Hccl::RdmaHandle>(inner.rdmaHandle), static_cast<Hccl::TargetJettyHandle>(remoteCtx.handle));
        HCCL_INFO(
            "[JettyContext][%s] unimport shared remote jetty, handle[%llu].", __func__,
            static_cast<unsigned long long>(remoteCtx.handle));
        remoteCtx.handle = 0;
        remoteCtx.handlePtr = nullptr;
        remoteCtx.ready = false;
    }
}

void JettyContext::DestroyJettyResources(Inner& inner) const
{
    if (inner.handle != 0) {
        Hccl::HrtRaUbDestroyJetty(inner.handle);
        HCCL_INFO(
            "[JettyContext][%s] destroyed shared jetty, handle[%llu].", __func__,
            static_cast<unsigned long long>(inner.handle));
    }
    if (inner.jfcHandle != 0 && inner.rdmaHandle != nullptr) {
        Hccl::HrtRaUbDestroyJfc(static_cast<Hccl::RdmaHandle>(inner.rdmaHandle), inner.jfcHandle);
        HCCL_INFO(
            "[JettyContext][%s] destroyed shared jfc, jfcHandle[%llu].", __func__,
            static_cast<unsigned long long>(inner.jfcHandle));
    }
    if (inner.sqPiPtr != nullptr) {
        (void)hrtFree(inner.sqPiPtr);
        inner.sqPiPtr = nullptr;
    }
    if (inner.sqCiPtr != nullptr) {
        (void)hrtFree(inner.sqCiPtr);
        inner.sqCiPtr = nullptr;
    }
    if (inner.cqPiPtr != nullptr) {
        (void)hrtFree(inner.cqPiPtr);
        inner.cqPiPtr = nullptr;
    }
    if (inner.cqCiPtr != nullptr) {
        (void)hrtFree(inner.cqCiPtr);
        inner.cqCiPtr = nullptr;
    }
    // 防御性清空：销毁完成后句柄类字段不再有效，避免后续误用悬挂句柄
    // （Release 路径随后会 inner_ = Inner{}，此处清空主要保护 memcpy_s 失败分支等提前销毁场景）
    inner.handle = 0;
    inner.jfcHandle = 0;
    inner.rdmaHandle = nullptr;
}

HcclResult JettyContext::InnerToCtx(const Inner& inner, Ctx& outCtx) const
{
    Ctx ctx{};
    ctx.handle = inner.handle;
    ctx.handlePtr = inner.handlePtr;
    ctx.jettyId = inner.jettyId;
    ctx.sqBuffVa = inner.sqBuffVa;
    ctx.dbAddr = inner.dbAddr;
    ctx.keySize = inner.keySize;
    ctx.sqDepth = inner.sqDepth;
    ctx.sqPiPtr = inner.sqPiPtr;
    ctx.sqCiPtr = inner.sqCiPtr;
    ctx.cqPiPtr = inner.cqPiPtr;
    ctx.cqCiPtr = inner.cqCiPtr;
    ctx.queueIndexMemSize = inner.queueIndexMemSize;
    ctx.rdmaHandle = inner.rdmaHandle;
    ctx.jfcHandle = inner.jfcHandle;
    ctx.cqInfo = inner.cqInfo;
    ctx.localPsn = inner.localPsn;
    if (inner.keySize > 0 && inner.keySize <= Hccl::HRT_UB_QP_KEY_MAX_LEN) {
        CHK_SAFETY_FUNC_RET(memcpy_s(ctx.localQpKey, Hccl::HRT_UB_QP_KEY_MAX_LEN, inner.localQpKey, inner.keySize));
    }
    outCtx = ctx;
    return HCCL_SUCCESS;
}

} // namespace hcomm
