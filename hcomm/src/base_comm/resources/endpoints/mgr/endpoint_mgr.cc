/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "endpoint_mgr.h"
#include "log.h"

namespace hcomm {

void EndpointMgr::Add(EndpointHandle handle, std::unique_ptr<Endpoint> endpoint)
{
    std::unique_ptr<Endpoint> replaced{nullptr};
    {
        std::lock_guard<std::mutex> lock(mtx_);
        auto it = endpointMap_.find(handle);
        if (it != endpointMap_.end()) {
            replaced = std::move(it->second); // 旧 Endpoint 在锁外析构（其析构会回调各设备 EndpointCtxMgr 重新拿锁）
            it->second = std::move(endpoint);
            return;
        }
        endpointMap_.emplace(handle, std::move(endpoint));
    }
}

bool EndpointMgr::Remove(EndpointHandle handle)
{
    std::unique_ptr<Endpoint> victim{nullptr};
    {
        std::lock_guard<std::mutex> lock(mtx_);
        auto it = endpointMap_.find(handle);
        if (it == endpointMap_.end()) {
            return false;
        }
        victim = std::move(it->second); // 锁外析构，避免与子类析构里的 ReleaseEndpointCtx 递归锁死锁
        endpointMap_.erase(it);
    }
    return true;
}

bool EndpointMgr::Update(EndpointHandle handle, std::unique_ptr<Endpoint> newEndpoint)
{
    std::unique_ptr<Endpoint> victim{nullptr};
    {
        std::lock_guard<std::mutex> lock(mtx_);
        auto it = endpointMap_.find(handle);
        if (it == endpointMap_.end()) {
            return false;
        }
        victim = std::move(it->second); // 锁外析构，避免与子类析构里的 ReleaseEndpointCtx 递归锁死锁
        it->second = std::move(newEndpoint);
    }
    return true;
}

Endpoint* EndpointMgr::Get(EndpointHandle handle)
{
    std::lock_guard<std::mutex> lock(mtx_);
    auto it = endpointMap_.find(handle);
    if (it != endpointMap_.end()) {
        return it->second.get();
    }
    return nullptr;
}

void EndpointMgr::DeInit()
{
    std::unordered_map<EndpointHandle, std::unique_ptr<Endpoint>> endpoints;
    {
        std::lock_guard<std::mutex> lock(mtx_);
        if (!endpointMap_.empty()) {
            HCCL_WARNING(
                "[%s] %zu endpoint(s) still in map during DeInit, force releasing", __func__, endpointMap_.size());
            endpoints = std::move(endpointMap_);
        }
    }
    // 锁外触发 ~Endpoint 析构链（子类析构会回调各设备 EndpointCtxMgr::Release 重新拿锁，锁内析构会死锁）
    // 残留 EndpointCtx 由 ~HcommResMgr() 遍历各设备 EndpointCtxMgr::DeInit() 兜底
    endpoints.clear();
}

} // namespace hcomm
