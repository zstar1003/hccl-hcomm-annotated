/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "uboe_endpoint.h"
#include "log.h"
#include "hccl/hccl_res.h"
#include "ub_reged_mem_mgr.h"
#include "proc_reged_mem_mgr_cache.h"
#include "adapter_rts_common.h"
#include "rdma_handle_manager.h"
#include "mgr/endpoint_ctx_mgr.h"
#include "hcomm_res_mgr.h"

namespace hcomm {

UboeEndpoint::UboeEndpoint(const EndpointDesc& endpointDesc) : Endpoint(endpointDesc) {}

UboeEndpoint::~UboeEndpoint() noexcept
{
    (void)ReleaseCache();
    (void)ReleaseEndpointCtx();
}

bool UboeEndpoint::IsCtxHandleValid() const
{
    if (ctxHandle_ == nullptr) {
        return false;
    }
    return Hccl::RdmaHandleManager::GetInstance().IsHandleValid(static_cast<Hccl::RdmaHandle>(ctxHandle_));
}

HcclResult UboeEndpoint::ReleaseEndpointCtx()
{
    if (endpointCtx_ == nullptr) {
        HCCL_WARNING("[UboeEndpoint][%s] endpointCtx_ is null, nothing to release", __func__);
        return HCCL_E_PTR;
    }
    // 先放掉自身引用，再触发 EndpointCtxMgr 移除
    EndpointCtxKey key = endpointCtx_->key;
    endpointCtx_.reset();
    HcommResMgr::GetInstance().GetDeviceResMgr(key.devPhyId).GetEndpointCtxMgr().Release(key);
    return HCCL_SUCCESS;
}

HcclResult
UboeEndpoint::AttachCache(const MemMgrCacheKey& key, const std::function<std::shared_ptr<RegedMemMgr>()>& creator)
{
    cacheKey_ = key;
    cacheKeepAlive_ = ProcRegedMemMgrCache::GetHolder();
    // cache key 含 protocol 唯一决定具体 RegedMemMgr 类型，static_pointer_cast 转换安全
    regedMemMgr_ = std::static_pointer_cast<UbRegedMemMgr>(cacheKeepAlive_->GetOrCreate(cacheKey_, creator));
    if (regedMemMgr_ == nullptr) {
        HCCL_ERROR("[UboeEndpoint][%s] regedMemMgr_ is null", __func__);
        CHK_RET(ReleaseCache());
        return HCCL_E_INTERNAL;
    }
    return HCCL_SUCCESS;
}

HcclResult UboeEndpoint::ReleaseCache()
{
    if (cacheKeepAlive_ == nullptr) {
        HCCL_WARNING("[UboeEndpoint][%s] cacheKeepAlive_ is null, nothing to release", __func__);
        return HCCL_E_PTR;
    }
    cacheKeepAlive_->Release(cacheKey_);
    cacheKeepAlive_.reset();
    return HCCL_SUCCESS;
}

HcclResult UboeEndpoint::Init()
{
    HCCL_INFO("[%s] localEndpoint protocol[%d]", __func__, endpointDesc_.protocol);

    Hccl::IpAddress ipAddr{};
    CHK_RET(CommAddrToIpAddress(endpointDesc_.commAddr, ipAddr));

    s32 deviceLogicId;
    u32 devPhyId;
    CHK_RET(hrtGetDevice(&deviceLogicId));
    CHK_RET(hrtGetDevicePhyIdByIndex(deviceLogicId, devPhyId));
    endpointDesc_.loc.device.devPhyId = devPhyId;

    Hccl::HccpHdcManager::GetInstance().Init(deviceLogicId);
    EndpointCtxKey ctxKey{devPhyId, COMM_PROTOCOL_UB_CTP, endpointDesc_.loc.locType, ipAddr};
    CHK_RET(HcommResMgr::GetInstance()
                .GetDeviceResMgr(ctxKey.devPhyId)
                .GetEndpointCtxMgr()
                .Acquire(ctxKey, true, endpointCtx_));
    CHK_PTR_NULL(endpointCtx_);
    ctxHandle_ = endpointCtx_->ctxHandle;
    HCCL_INFO(
        "%s success, devPhyId[%u], ipAddr[%s], ctxHandle[%p]", __func__, devPhyId, ipAddr.Describe().c_str(),
        ctxHandle_);

    MemMgrCacheKey key{devPhyId, COMM_PROTOCOL_UB_CTP, ipAddr, LocTypeToPortType(endpointDesc_.loc.locType)};
    auto createMgr = [this]() {
        return std::make_shared<UbRegedMemMgr>(ctxHandle_);
    };
    auto ret = AttachCache(key, createMgr);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[UboeEndpoint][%s] AttachCache failed, ret[%d]", __func__, ret);
        CHK_RET(ReleaseEndpointCtx());
        return ret;
    }

    return HcclResult::HCCL_SUCCESS;
}

} // namespace hcomm
