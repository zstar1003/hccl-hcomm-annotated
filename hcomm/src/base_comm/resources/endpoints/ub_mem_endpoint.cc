/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ub_mem_endpoint.h"
#include "log.h"
#include "hccl/hccl_res.h"
#include "adapter_rts_common.h"
#include "server_socket_mgr.h"
#include "ub_mem_reged_mem_mgr.h"
#include "proc_reged_mem_mgr_cache.h"
#include "hccl_mem_defs.h"

namespace hcomm {
UbMemEndpoint::UbMemEndpoint(const EndpointDesc& endpointDesc) : Endpoint(endpointDesc) {}

UbMemEndpoint::~UbMemEndpoint() noexcept { (void)ReleaseCache(); }

HcclResult
UbMemEndpoint::AttachCache(const MemMgrCacheKey& key, const std::function<std::shared_ptr<RegedMemMgr>()>& creator)
{
    cacheKey_ = key;
    cacheKeepAlive_ = ProcRegedMemMgrCache::GetHolder();
    // cache key 含 protocol 唯一决定具体 RegedMemMgr 类型，static_pointer_cast 转换安全
    regedMemMgr_ = std::static_pointer_cast<UbMemRegedMemMgr>(cacheKeepAlive_->GetOrCreate(cacheKey_, creator));
    if (regedMemMgr_ == nullptr) {
        HCCL_ERROR("[UbMemEndpoint][%s] regedMemMgr_ is null", __func__);
        CHK_RET(ReleaseCache());
        return HCCL_E_INTERNAL;
    }
    return HCCL_SUCCESS;
}

HcclResult UbMemEndpoint::ReleaseCache()
{
    if (cacheKeepAlive_ == nullptr) {
        HCCL_WARNING("[UbMemEndpoint][%s] cacheKeepAlive_ is null, nothing to release", __func__);
        return HCCL_E_PTR;
    }
    cacheKeepAlive_->Release(cacheKey_);
    cacheKeepAlive_.reset();
    return HCCL_SUCCESS;
}

HcclResult UbMemEndpoint::Init()
{
    Hccl::IpAddress ipAddr{};
    CHK_RET(CommAddrToIpAddress(endpointDesc_.commAddr, ipAddr));

    s32 devId = 0;
    CHK_RET(hrtGetDevice(&devId));
    u32 devPhyId = 0;
    CHK_RET(hrtGetDevicePhyIdByIndex(devId, devPhyId));

    MemMgrCacheKey key{devPhyId, COMM_PROTOCOL_UB_MEM, ipAddr, LocTypeToPortType(endpointDesc_.loc.locType)};
    auto createMgr = []() {
        return std::make_shared<UbMemRegedMemMgr>();
    };
    CHK_RET(AttachCache(key, createMgr));

    return HcclResult::HCCL_SUCCESS;
}
} // namespace hcomm
