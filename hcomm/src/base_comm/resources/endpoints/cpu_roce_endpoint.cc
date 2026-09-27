/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hccl_mem_defs.h"
#include "cpu_roce_endpoint.h"
#include "hccl/hccl_res.h"
#include "log.h"
#include "roce_reged_mem_mgr.h"
#include "proc_reged_mem_mgr_cache.h"
#include "host_socket_handle_manager.h"
#include "adapter_rts_common.h"
#include "hccp_peer_manager.h"
#include "server_socket_manager.h"
#include "rdma_handle_manager.h"
#include "mgr/endpoint_ctx_mgr.h"
#include "hcomm_res_mgr.h"
#include "hccp.h"

using Hccl::HcclException;
using std::exception;
using std::string;

namespace hcomm {
CpuRoceEndpoint::CpuRoceEndpoint(const EndpointDesc& endpointDesc) : Endpoint(endpointDesc) {}

CpuRoceEndpoint::~CpuRoceEndpoint() noexcept
{
    (void)ReleaseCache();
    (void)ReleaseEndpointCtx();
}

bool CpuRoceEndpoint::IsCtxHandleValid() const
{
    if (ctxHandle_ == nullptr) {
        return false;
    }
    return Hccl::RdmaHandleManager::GetInstance().IsHandleValid(static_cast<Hccl::RdmaHandle>(ctxHandle_));
}

HcclResult CpuRoceEndpoint::ReleaseEndpointCtx()
{
    if (endpointCtx_ == nullptr) {
        HCCL_WARNING("[CpuRoceEndpoint][%s] endpointCtx_ is null, nothing to release", __func__);
        return HCCL_E_PTR;
    }
    // 先放掉自身引用，当use_count 只剩 map 持有的 1 份时再触发 EndpointCtxMgr 移除
    EndpointCtxKey key = endpointCtx_->key;
    endpointCtx_.reset();
    HcommResMgr::GetInstance().GetDeviceResMgr(key.devPhyId).GetEndpointCtxMgr().Release(key);
    return HCCL_SUCCESS;
}

HcclResult
CpuRoceEndpoint::AttachCache(const MemMgrCacheKey& key, const std::function<std::shared_ptr<RegedMemMgr>()>& creator)
{
    cacheKey_ = key;
    cacheKeepAlive_ = ProcRegedMemMgrCache::GetHolder();
    // cache key 含 protocol 唯一决定具体 RegedMemMgr 类型，static_pointer_cast 转换安全
    regedMemMgr_ = std::static_pointer_cast<RoceRegedMemMgr>(cacheKeepAlive_->GetOrCreate(cacheKey_, creator));
    if (regedMemMgr_ == nullptr) {
        HCCL_ERROR("[CpuRoceEndpoint][%s] regedMemMgr_ is null", __func__);
        CHK_RET(ReleaseCache());
        return HCCL_E_INTERNAL;
    }
    return HCCL_SUCCESS;
}

HcclResult CpuRoceEndpoint::ReleaseCache()
{
    if (cacheKeepAlive_ == nullptr) {
        HCCL_WARNING("[CpuRoceEndpoint][%s] cacheKeepAlive_ is null, nothing to release", __func__);
        return HCCL_E_PTR;
    }
    cacheKeepAlive_->Release(cacheKey_);
    cacheKeepAlive_.reset();
    return HCCL_SUCCESS;
}

HcclResult CpuRoceEndpoint::Init()
{
    HCCL_INFO("[%s] localEndpoint protocol[%d]", __func__, endpointDesc_.protocol);

    if (endpointDesc_.loc.locType != ENDPOINT_LOC_TYPE_HOST) {
        HCCL_INFO("[CpuRoceEndpoint][%s] CpuRoceEndpoint not support device", __func__);
        return HCCL_E_NOT_SUPPORT;
    }
    Hccl::IpAddress ipAddr{};
    CHK_RET(CommAddrToIpAddress(endpointDesc_.commAddr, ipAddr));
    s32 devId = 0;
    CHK_RET(hrtGetDevice(&devId));
    EXCEPTION_CATCH(Hccl::HccpPeerManager::GetInstance().Init(devId), return HCCL_E_INTERNAL);
    u32 devPhyId = 0;
    CHK_RET(hrtGetDevicePhyIdByIndex(devId, devPhyId));
    // RdmaHandleManager 调用收口到 EndpointCtxMgr（per-device，经 HcommBaseResMgr 访问）
    EndpointCtxKey ctxKey{devPhyId, COMM_PROTOCOL_ROCE, endpointDesc_.loc.locType, ipAddr};
    CHK_RET(HcommResMgr::GetInstance()
                .GetDeviceResMgr(ctxKey.devPhyId)
                .GetEndpointCtxMgr()
                .Acquire(ctxKey, false, endpointCtx_));
    CHK_PTR_NULL(endpointCtx_);
    ctxHandle_ = endpointCtx_->ctxHandle;
    HCCL_INFO(
        "CpuRoceEndpoint::%s success, devPhyId[%u], ipAddr[%s], ctxHandle[%p]", __func__, devPhyId,
        ipAddr.Describe().c_str(), ctxHandle_);

    // useAllocMemBase：见 RoceRegedMemMgr 类注（本质 D2N=UB；当前以 lbMax>0 代理，待 HCCP 接口）
    Capabilities caps{};
    HcclResult capRet = GetCapabilities(caps);
    if (capRet != HCCL_SUCCESS) {
        HCCL_ERROR("[CpuRoceEndpoint][%s] GetCapabilities failed, ret[%d]", __func__, capRet);
        CHK_RET(ReleaseEndpointCtx());
        return capRet;
    }
    const bool useAllocMemBase = (caps.lbMax > 0);
    HCCL_INFO("[CpuRoceEndpoint::Init] lbMax[%d] useAllocMemBase[%d]", caps.lbMax, static_cast<int>(useAllocMemBase));

    MemMgrCacheKey key{devPhyId, COMM_PROTOCOL_ROCE, ipAddr, LocTypeToPortType(endpointDesc_.loc.locType)};
    auto createMgr = [this, useAllocMemBase]() {
        return std::make_shared<RoceRegedMemMgr>(ctxHandle_, useAllocMemBase);
    };
    auto ret = AttachCache(key, createMgr);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[CpuRoceEndpoint][%s] AttachCache failed, ret[%d]", __func__, ret);
        CHK_RET(ReleaseEndpointCtx());
        return ret;
    }

    // ServerSocketContext：Init 成功路径构造，commAddr 供方法内转换 IpAddress 使用
    serverSocketContext_.emplace(Hccl::ConnectProtoType::RDMA, endpointDesc_.commAddr);
    return HCCL_SUCCESS;
}

HcclResult CpuRoceEndpoint::GetCapabilities(Capabilities& caps)
{
    HCCL_INFO("[CpuRoceEndpoint::%s] START.", __func__);
    static constexpr uint64_t RDMA_MAX_WR_LENGTH = 1ULL * 1024 * 1024 * 1024; // 单次RDMA操作最大长度1GB
    if (!isCapabilitiesAvailable_) {
        // Init 在创建 mem mgr 前就会调用；用 ctxHandle_，不依赖 regedMemMgr_。
        // 待 HCCP 提供查询设备支持的最大发送消息的接口后，查询设备实际值。
        CHK_PTR_NULL(ctxHandle_);
        capabilities_.maxMsgSize = RDMA_MAX_WR_LENGTH;
        capabilities_.lbMax = 0;
        uint32_t ret = RaGetLbMax(ctxHandle_, &(capabilities_.lbMax));
        HCCL_DEBUG("[CpuRoceEndpoint::GetCapabilities] lbMax = %d.", capabilities_.lbMax);
        CHK_PRT_RET(
            ret != 0,
            HCCL_ERROR(
                "[CpuRoceEndpoint::GetCapabilities][GetLbMax]errNo[0x%016llx] RaGetLbMax fail. "
                "return[%u], params: rdmaHandle[%p], lbMax[%d]",
                HCCL_ERROR_CODE(HCCL_E_NETWORK), ret, ctxHandle_, capabilities_.lbMax),
            HCCL_E_NETWORK);
        isCapabilitiesAvailable_ = true;
    }
    caps = capabilities_;
    HCCL_INFO("[CpuRoceEndpoint::%s] SUCCESS.", __func__);
    return HCCL_SUCCESS;
}
} // namespace hcomm
