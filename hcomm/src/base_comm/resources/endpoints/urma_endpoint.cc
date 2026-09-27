/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "urma_endpoint.h"
#include <algorithm>
#include "log.h"
#include "hccl/hccl_res.h"
#include "ub_reged_mem_mgr.h"
#include "proc_reged_mem_mgr_cache.h"
#include "adapter_rts_common.h"
#include "server_socket_manager.h"
#include "rdma_handle_manager.h"
#include "ra_rs_comm.h"
#include "mgr/endpoint_ctx_mgr.h"
#include "hcomm_res_mgr.h"

namespace hcomm {

UrmaEndpoint::UrmaEndpoint(const EndpointDesc& endpointDesc) : Endpoint(endpointDesc) {}

UrmaEndpoint::~UrmaEndpoint() noexcept
{
    (void)ReleaseCache();
    (void)ReleaseEndpointCtx();
}

bool UrmaEndpoint::IsCtxHandleValid() const
{
    if (ctxHandle_ == nullptr) {
        return false;
    }
    return Hccl::RdmaHandleManager::GetInstance().IsHandleValid(static_cast<Hccl::RdmaHandle>(ctxHandle_));
}

CommQueueContext* UrmaEndpoint::GetCommQueueContext()
{
    std::call_once(jettyContextOnce_, [this] {
        jettyContext_ = std::make_unique<JettyContext>();
    });
    return jettyContext_.get();
}

HcclResult UrmaEndpoint::ReleaseEndpointCtx()
{
    if (endpointCtx_ == nullptr) {
        HCCL_WARNING("[UrmaEndpoint][%s] endpointCtx_ is null, nothing to release", __func__);
        return HCCL_E_PTR;
    }
    // 先放掉自身引用，再触发 EndpointCtxMgr 移除
    EndpointCtxKey key = endpointCtx_->key;
    endpointCtx_.reset();
    HcommResMgr::GetInstance().GetDeviceResMgr(key.devPhyId).GetEndpointCtxMgr().Release(key);
    return HCCL_SUCCESS;
}

HcclResult
UrmaEndpoint::AttachCache(const MemMgrCacheKey& key, const std::function<std::shared_ptr<RegedMemMgr>()>& creator)
{
    cacheKey_ = key;
    cacheKeepAlive_ = ProcRegedMemMgrCache::GetHolder();
    // cache key 含 protocol 唯一决定具体 RegedMemMgr 类型，static_pointer_cast 转换安全
    regedMemMgr_ = std::static_pointer_cast<UbRegedMemMgr>(cacheKeepAlive_->GetOrCreate(cacheKey_, creator));
    if (regedMemMgr_ == nullptr) {
        HCCL_ERROR("[UrmaEndpoint][%s] regedMemMgr_ is null", __func__);
        CHK_RET(ReleaseCache());
        return HCCL_E_INTERNAL;
    }
    return HCCL_SUCCESS;
}

HcclResult UrmaEndpoint::ReleaseCache()
{
    if (cacheKeepAlive_ == nullptr) {
        HCCL_WARNING("[UrmaEndpoint][%s] cacheKeepAlive_ is null, nothing to release", __func__);
        return HCCL_E_PTR;
    }
    cacheKeepAlive_->Release(cacheKey_);
    cacheKeepAlive_.reset();
    return HCCL_SUCCESS;
}

HcclResult UrmaEndpoint::Init()
{
    HCCL_INFO("[%s] localEndpoint protocol[%d]", __func__, endpointDesc_.protocol);

    if (endpointDesc_.loc.locType != ENDPOINT_LOC_TYPE_DEVICE) {
        HCCL_ERROR(
            "[UrmaEndpoint][%s] endpointDesc.loc.locType[%d] only support ENDPOINT_LOC_TYPE_DEVICE", __func__,
            endpointDesc_.loc.locType);
        return HCCL_E_PARA;
    }

    Hccl::IpAddress ipAddr{};
    HcclResult ret = CommAddrToIpAddress(endpointDesc_.commAddr, ipAddr);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("call CommAddrToIpAddress failed");
        return ret;
    }

    u32 devPhyId;
    s32 deviceLogicId;
    ret = hrtGetDevice(&deviceLogicId);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("call hrtGetDevice failed, deviceLogicId[%d]", deviceLogicId);
        return ret;
    }
    Hccl::HccpHdcManager::GetInstance().Init(deviceLogicId);
    ret = hrtGetDevicePhyIdByIndex(static_cast<uint32_t>(deviceLogicId), devPhyId);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("call hrtGetDevicePhyIdByIndex failed, deviceLogicId[%d], devPhyId[%u]", deviceLogicId, devPhyId);
        return ret;
    }

    if (endpointDesc_.loc.device.devPhyId != devPhyId) {
        HCCL_WARNING(
            "[UrmaEndpoint][%s] endpointDesc.loc.device.devPhyId[%u] incorrect", __func__,
            endpointDesc_.loc.device.devPhyId);
        // 当前endpointDesc.loc.device.devPhyId不准，暂时由查询的devPhyId赋值
        endpointDesc_.loc.device.devPhyId = devPhyId;
    }

    EndpointCtxKey ctxKey{devPhyId, COMM_PROTOCOL_UB_CTP, endpointDesc_.loc.locType, ipAddr};
    CHK_RET(HcommResMgr::GetInstance()
                .GetDeviceResMgr(ctxKey.devPhyId)
                .GetEndpointCtxMgr()
                .Acquire(ctxKey, false, endpointCtx_));
    CHK_PTR_NULL(endpointCtx_);
    ctxHandle_ = endpointCtx_->ctxHandle;
    HCCL_INFO(
        "%s success, devPhyId[%u], ipAddr[%s], ctxHandle[%p]", __func__, devPhyId, ipAddr.Describe().c_str(),
        ctxHandle_);

    MemMgrCacheKey key{devPhyId, COMM_PROTOCOL_UB_CTP, ipAddr, LocTypeToPortType(endpointDesc_.loc.locType)};
    auto createMgr = [this]() {
        return std::make_shared<UbRegedMemMgr>(ctxHandle_);
    };
    ret = AttachCache(key, createMgr);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[UrmaEndpoint][%s] AttachCache failed, ret[%d]", __func__, ret);
        CHK_RET(ReleaseEndpointCtx());
        return ret;
    }

    // ServerSocketContext：以修正后的 devPhyId/locType 构造，commAddr 供方法内转换 IpAddress 使用
    serverSocketContext_.emplace(
        Hccl::ConnectProtoType::UB, endpointDesc_.loc.device.devPhyId, endpointDesc_.loc.locType,
        endpointDesc_.commAddr);

    // ccu模式专用的资源分配器
    ccuChannelCtxPool_.reset(new (std::nothrow) CcuChannelCtxPool(deviceLogicId));
    CHK_PTR_NULL(ccuChannelCtxPool_);

    return HcclResult::HCCL_SUCCESS;
}

CcuChannelCtxPool* UrmaEndpoint::GetCcuChannelCtxPool() { return ccuChannelCtxPool_.get(); }

HcclResult UrmaEndpoint::GetAsyncEvents(uint32_t devPhyId, struct AsyncEvent events[], uint32_t& num)
{
    uint32_t interfaceVersion{0};

    int ret;
    // 对RaCtxGetAsyncEvents接口的版本检验
    ret = RaGetInterfaceVersion(devPhyId, RA_RS_CTX_GET_ASYNC_EVENTS, &interfaceVersion);
    if (ret != 0) {
        HCCL_ERROR(
            "[%s] devPhyId[%u] epHandle[%p] RaGetInterfaceVersion failed, ret[%d]", __func__, devPhyId, this, ret);
        return HCCL_E_INTERNAL;
    }

    if (interfaceVersion <= 1) {
        HCCL_ERROR(
            "[%s] devPhyId[%u] epHandle[%p] version[%u] not support", __func__, devPhyId, this, interfaceVersion);
        return HCCL_E_NOT_SUPPORT;
    }

    if (!IsCtxHandleValid()) {
        HCCL_ERROR(
            "[%s] devPhyId[%u] ctxHandle_[%p] has been invalidated, "
            "RdmaHandleManager may have DeInit this device",
            __func__, devPhyId, ctxHandle_);
        return HCCL_E_INTERNAL;
    }

    ret = RaCtxGetAsyncEvents(ctxHandle_, events, &num);
    if (ret != 0) {
        HCCL_ERROR(
            "[%s] devPhyId[%u] epHandle[%p] RaCtxGetAsyncEvents failed, ctxHandle[%p] ret[%d]", __func__, devPhyId,
            this, static_cast<void*>(ctxHandle_), ret);
        return HCCL_E_INTERNAL;
    }
    return HCCL_SUCCESS;
}

} // namespace hcomm
