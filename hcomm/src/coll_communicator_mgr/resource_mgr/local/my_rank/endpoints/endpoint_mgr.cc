/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "endpoint_mgr.h"
#include <algorithm>
#include "hcomm_c_adpt.h"

namespace hccl {

TaggedMemMap::~TaggedMemMap()
{
    if (handle_ == nullptr) {
        return;
    }
    for (const auto& kv : tagToHandle_) {
        HcommResult ret = HcommMemUnreg(handle_, kv.second);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR(
                "[TaggedMemMap::~TaggedMemMap] HcommMemUnreg failed, handle[%p] tag[%s] ret[%d]", handle_,
                kv.first.c_str(), ret);
        }
    }
}

MemHandle TaggedMemMap::FindHandle(const std::string& tag) const
{
    auto it = tagToHandle_.find(tag);
    return it != tagToHandle_.end() ? it->second : nullptr;
}

bool TaggedMemMap::HasTag(const std::string& tag) const { return tagToHandle_.find(tag) != tagToHandle_.end(); }

void TaggedMemMap::EmplaceHandle(const std::string& tag, MemHandle handle) { tagToHandle_.emplace(tag, handle); }

MemHandle TaggedMemMap::RemoveTag(const std::string& tag)
{
    auto it = tagToHandle_.find(tag);
    if (it == tagToHandle_.end()) {
        return nullptr;
    }
    MemHandle handle = it->second;
    tagToHandle_.erase(it);
    return handle;
}

EndpointMgr::~EndpointMgr()
{
    endpointTagMemMap_.clear();
    for (const auto& kv : endpointMap_) {
        const EndpointHandle& endpointHandle = kv.second;
        (void)HcommEndpointDestroy(endpointHandle);
    }
    // 销毁共享 jetty 场景按 tag 创建的独立 Endpoint
    for (const auto& kv : taggedEndpointMap_) {
        (void)HcommEndpointDestroy(kv.second);
    }
    taggedEndpointMap_.clear();
}

HcclResult EndpointMgr::Get(EndpointDesc epDesc, EndpointHandle& handle)
{
    std::lock_guard<std::mutex> lock(mutex_);
    auto iterPtr = endpointMap_.find(epDesc);
    if (iterPtr != endpointMap_.end()) {
        handle = iterPtr->second;
        return HCCL_SUCCESS;
    }
    HCCL_INFO("[EndpointMgr::Get] create Endpoint");
    CHK_RET(static_cast<HcclResult>(HcommEndpointCreate(&epDesc, &handle)));

    endpointMap_.emplace(epDesc, handle);
    return HCCL_SUCCESS;
}

HcclResult EndpointMgr::GetWithTag(EndpointDesc epDesc, const std::string& sharedQueueTag, EndpointHandle& handle)
{
    // tag 为空：退化为默认 Get，兼容非共享路径或无 tag 场景
    if (sharedQueueTag.empty()) {
        return Get(epDesc, handle);
    }

    EndpointDescTagKey key{epDesc, sharedQueueTag};

    // 快路径：持锁查缓存，命中直接返回
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto iter = taggedEndpointMap_.find(key);
        if (iter != taggedEndpointMap_.end()) {
            handle = iter->second;
            return HCCL_SUCCESS;
        }
    }

    // 慢路径：持锁创建 + 二次检查。
    // 不采用"无锁创建+失败销毁"乐观模式：HcommEndpointCreate 涉及 device context 分配等重操作，
    // 高并发同 key 多线程重复创建+销毁的代价高于锁内串行等待；且 create/destroy 非严格幂等时可能残留状态。
    std::lock_guard<std::mutex> lock(mutex_);
    // 二次检查：另一线程可能已在快路径后、本线程拿锁前完成创建
    auto iter = taggedEndpointMap_.find(key);
    if (iter != taggedEndpointMap_.end()) {
        handle = iter->second;
        return HCCL_SUCCESS;
    }
    // 锁内创建：同一 key 不会有并发的重复创建
    CHK_RET(static_cast<HcclResult>(HcommEndpointCreate(&epDesc, &handle)));
    taggedEndpointMap_.emplace(std::move(key), handle);
    HCCL_INFO("[EndpointMgr::GetWithTag] create tagged Endpoint, tag[%s], handle[%p].", sharedQueueTag.c_str(), handle);
    return HCCL_SUCCESS;
}

HcclResult EndpointMgr::RegisterMemory(
    EndpointHandle epHandle, const std::vector<std::string>& memTag, const std::vector<HcclMem>& memVec,
    uint64_t commMemsVersion)
{
    std::lock_guard<std::mutex> lock(mutex_);
    auto& taggedMap = endpointTagMemMap_.try_emplace(epHandle, epHandle).first->second;

    // 版本一致，CommMems 无变更，跳过注册
    if (taggedMap.GetVersion() == commMemsVersion) {
        HCCL_INFO(
            "[%s]commMemsVersion[%llu] unchanged, skip registration, epHandle[%p]", __FUNCTION__, commMemsVersion,
            epHandle);
        return HCCL_SUCCESS;
    }
    CHK_PRT_RET(
        memTag.size() < memVec.size(),
        HCCL_ERROR("[%s] memTag.size()[%zu] < memVec.size()[%zu]", __FUNCTION__, memTag.size(), memVec.size()),
        HCCL_E_PARA);

    size_t index = 0;
    for (const auto& mem : memVec) {
        const std::string& tag = memTag[index];
        index++;
        // 检查tag是否已注册，避免重复注册
        if (taggedMap.HasTag(tag)) {
            HCCL_INFO("[%s]tag already registered, reuse existing handle, tag=%s", __FUNCTION__, tag.c_str());
            continue;
        }
        MemHandle memHandle = nullptr;
        CommMem commMem{static_cast<CommMemType>(mem.type), mem.addr, mem.size};
        HcclResult ret = static_cast<HcclResult>(HcommMemReg(epHandle, tag.c_str(), &commMem, &memHandle));
        if (ret != HCCL_SUCCESS && ret != HCCL_E_AGAIN) {
            HCCL_ERROR("[%s]call trace: hcclRet -> %d", __FUNCTION__, ret);
            return ret;
        }
        CHK_PTR_NULL(memHandle);
        taggedMap.EmplaceHandle(tag, memHandle); // 记录到tag映射，后续相同tag直接命中
        if (ret == HCCL_E_AGAIN) {
            HCCL_WARNING("This mem has already been registered, addr=%p, size=%llu", mem.addr, mem.size);
        }
    }

    taggedMap.SetVersion(commMemsVersion);
    return HCCL_SUCCESS;
}

HcclResult EndpointMgr::GetMemHandlesByTags(
    EndpointHandle epHandle, const std::vector<std::string>& memTags, std::vector<MemHandle>& memHandleVec)
{
    std::lock_guard<std::mutex> lock(mutex_);
    memHandleVec.clear();
    auto it = endpointTagMemMap_.find(epHandle);
    if (it == endpointTagMemMap_.end()) {
        HCCL_ERROR("[%s] epHandle[%p] not found in endpointTagMemMap_", __FUNCTION__, epHandle);
        return HCCL_E_MEMORY;
    }
    const auto& taggedMap = it->second;
    for (const auto& tag : memTags) {
        MemHandle handle = taggedMap.FindHandle(tag);
        if (handle == nullptr) {
            HCCL_ERROR(
                "[%s] tag[%s] not found in endpoint[%p], registration may have been skipped", __FUNCTION__, tag.c_str(),
                epHandle);
            return HCCL_E_NOT_FOUND;
        }
        memHandleVec.push_back(handle);
    }
    return HCCL_SUCCESS;
}

HcclResult EndpointMgr::UnregMemByTag(const std::string& tag)
{
    std::lock_guard<std::mutex> lock(mutex_);
    HcclResult lastErr = HCCL_SUCCESS;
    for (auto& kv : endpointTagMemMap_) {
        MemHandle handle = kv.second.FindHandle(tag);
        if (handle == nullptr) {
            continue;
        }
        HcommResult ret = HcommMemUnreg(kv.first, handle);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR(
                "[%s] HcommMemUnreg failed, epHandle[%p] tag[%s] ret[%d]", __FUNCTION__, kv.first, tag.c_str(), ret);
            lastErr = static_cast<HcclResult>(ret);
            continue;
        }
        kv.second.RemoveTag(tag);
    }
    return lastErr;
}

bool EndpointMgr::IsDescExist(EndpointDesc epDesc) { return endpointMap_.find(epDesc) != endpointMap_.end(); }

} // namespace hccl
