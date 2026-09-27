/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "endpoint_ctx_mgr.h"
#include "log.h"
#include "rdma_handle_manager.h"

namespace hcomm {

HcclResult EndpointCtxMgr::Acquire(const EndpointCtxKey& key, bool isUboeIp, std::shared_ptr<EndpointCtx>& endpointCtx)
{
    if (key.protocol == COMM_PROTOCOL_RESERVED || key.locType == ENDPOINT_LOC_TYPE_RESERVED) {
        HCCL_ERROR(
            "[%s] invalid key, protocol[%d] locType[%d], devPhyId[%u]", __func__, static_cast<int>(key.protocol),
            static_cast<int>(key.locType), key.devPhyId);
        return HCCL_E_PARA;
    }
    auto& rdmaHandleMgr = Hccl::RdmaHandleManager::GetInstance();
    Hccl::IpAddress queryIp = key.ip;
    if (isUboeIp) {
        rdmaHandleMgr.UboeIpv4ToEid(key.ip, queryIp, key.devPhyId);
    }
    EndpointCtxKey queryKey{key.devPhyId, key.protocol, key.locType, queryIp};

    std::lock_guard<std::mutex> lock(mtx_);

    // 缓存命中：设备 reset 时 RdmaHandleManager::DeInit 清句柄表但不联动本表，
    // 需校验句柄有效性，失效则擦除条目并走下方创建分支重建，避免向外派发已销毁句柄
    if (FindValidCachedLocked(rdmaHandleMgr, queryKey, endpointCtx)) {
        return HCCL_SUCCESS;
    }

    void* ctxHandle = nullptr;
    CHK_RET(AcquireHandleLocked(rdmaHandleMgr, queryKey, ctxHandle));

    auto ctx = std::make_shared<EndpointCtx>(EndpointCtx{ctxHandle, queryKey});
    endpointCtxMap_.emplace(queryKey, ctx);
    HCCL_INFO(
        "[%s] created, devPhyId[%u] protocol[%d] ctxHandle[%p]", __func__, queryKey.devPhyId,
        static_cast<int>(queryKey.protocol), ctxHandle);
    endpointCtx = std::move(ctx);
    return HCCL_SUCCESS;
}

// 缓存命中校验，调用方须持有 mtx_
bool EndpointCtxMgr::FindValidCachedLocked(
    Hccl::RdmaHandleManager& rdmaHandleMgr, const EndpointCtxKey& key, std::shared_ptr<EndpointCtx>& endpointCtx)
{
    auto it = endpointCtxMap_.find(key);
    if (it == endpointCtxMap_.end()) {
        return false;
    }
    void* cachedCtxHandle = it->second->ctxHandle;
    if (rdmaHandleMgr.IsHandleValid(static_cast<Hccl::RdmaHandle>(cachedCtxHandle))) {
        HCCL_INFO(
            "[%s] hit cache, devPhyId[%u] protocol[%d] ctxHandle[%p]", __func__, key.devPhyId,
            static_cast<int>(key.protocol), cachedCtxHandle);
        endpointCtx = it->second;
        return true;
    }
    HCCL_WARNING(
        "[%s] cached ctxHandle[%p] invalid (maybe after device reset), drop and rebuild, devPhyId[%u]", __func__,
        cachedCtxHandle, key.devPhyId);
    endpointCtxMap_.erase(key);
    return false;
}

// 按 key.locType 分发底层句柄查询，调用方须持有 mtx_
HcclResult EndpointCtxMgr::AcquireHandleLocked(
    Hccl::RdmaHandleManager& rdmaHandleMgr, const EndpointCtxKey& key, void*& ctxHandle) const
{
    // key.protocol 为业务协议（CommProtocol），底层 GetByAddr 需要 LinkProtoType：按业务协议映射
    // （ROCE→RDMA；UB 系协议→UB。迁移前 5 子类直调时的映射与此一致）
    Hccl::LinkProtoType linkProto = Hccl::LinkProtoType::UB;
    if (key.protocol == COMM_PROTOCOL_ROCE) {
        linkProto = Hccl::LinkProtoType::RDMA;
    }
    if (key.locType == ENDPOINT_LOC_TYPE_HOST) {
        Hccl::IpAddress ip = key.ip; // GetByAddr takes non-const ref
        EXCEPTION_CATCH(
            ctxHandle = static_cast<void*>(
                rdmaHandleMgr.GetByAddr(key.devPhyId, linkProto, ip, Hccl::PortDeploymentType::HOST_NET)),
            return HCCL_E_INTERNAL);
    } else if (key.locType == ENDPOINT_LOC_TYPE_DEVICE) {
        EXCEPTION_CATCH(
            ctxHandle = static_cast<void*>(rdmaHandleMgr.GetByIp(key.devPhyId, key.ip)), return HCCL_E_INTERNAL);
    } else {
        HCCL_ERROR("[%s] unexpected locType[%d], devPhyId[%u]", __func__, static_cast<int>(key.locType), key.devPhyId);
        return HCCL_E_PARA;
    }
    if (ctxHandle == nullptr) {
        HCCL_ERROR(
            "[%s] GetByAddr/GetByIp returned null, locType[%d], devPhyId[%u]", __func__, static_cast<int>(key.locType),
            key.devPhyId);
        return HCCL_E_PTR;
    }
    return HCCL_SUCCESS;
}

void EndpointCtxMgr::Release(const EndpointCtxKey& key)
{
    std::shared_ptr<EndpointCtx> victim{nullptr};
    {
        std::lock_guard<std::mutex> lock(mtx_);
        auto it = endpointCtxMap_.find(key);
        if (it == endpointCtxMap_.end()) {
            return;
        }
        // use_count()==1 表示只剩 map 引用（调用方须先 reset 自身引用），移出后锁外析构
        if (it->second.use_count() <= 1) {
            victim = std::move(it->second);
            endpointCtxMap_.erase(it);
        }
    }
    // victim 置空触发 ~EndpointCtx，锁外析构，与 Remove/DeInit 的锁外析构纪律一致
}

void EndpointCtxMgr::DeInit()
{
    std::unordered_map<EndpointCtxKey, std::shared_ptr<EndpointCtx>, EndpointCtxKeyHash> ctxs;
    {
        std::lock_guard<std::mutex> lock(mtx_);
        if (!endpointCtxMap_.empty()) {
            HCCL_WARNING(
                "[%s] %zu endpointCtx(s) still in map during DeInit, force releasing", __func__,
                endpointCtxMap_.size());
            ctxs = std::move(endpointCtxMap_);
        }
    }
    // 锁外清空条目（~EndpointCtx 为平凡析构，此处保持与 EndpointMgr::DeInit 一致的锁外析构纪律）
    ctxs.clear();
}

} // namespace hcomm
