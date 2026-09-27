/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aicpu_task_cache.h"

#include <cstring>

#include "aicpu_task_utils.h"
#include "log.h"

namespace hcomm {

AicpuTaskCache::AicpuTaskCache(const uint64_t maxCacheBytes) : cacheBytes_(0), maxCacheBytes_(maxCacheBytes)
{
    HCCL_RUN_INFO(
        "[AicpuTaskCache][AicpuTaskCache] create aicpu task cache at 0x%016llx: maxCacheBytes[%llu]", this,
        maxCacheBytes_);
}

AicpuTaskCache::~AicpuTaskCache()
{
    const uint64_t entryCnt = cacheHashMap_.size();
    for (CacheHashMap::const_iterator constIter = cacheHashMap_.cbegin(); constIter != cacheHashMap_.cend();
         ++constIter) {
        AicpuTaskCacheEntry* entryPtr = constIter->second;
        if (UNLIKELY(entryPtr == nullptr)) {
            HCCL_ERROR("[AicpuTaskCache][~AicpuTaskCache] entryPtr is nullptr");
            continue;
        }

        delete entryPtr;
        entryPtr = nullptr;
    }

    HCCL_RUN_INFO(
        "[AicpuTaskCache][~AicpuTaskCache] release aicpu task cache at 0x%016llx: "
        "entryCnt[%zu], cacheBytes_[%llu]",
        this, entryCnt, cacheBytes_);
}

HcclResult AicpuTaskCache::FindEntry(const char* cacheTag, AicpuTaskCacheEntry** entryPtrPtr) const
{
    CHK_PTRPTR_NULL(entryPtrPtr);

    bool reportCacheHit = false;
    {
        std::shared_lock<std::shared_timed_mutex> lock(cacheMtx_); // 读锁
        CacheHashMap::const_iterator constIter = cacheHashMap_.find(cacheTag);
        if (constIter != cacheHashMap_.cend()) {
            *entryPtrPtr = constIter->second;
            CHK_PTR_NULL(*entryPtrPtr);
            reportCacheHit = !cacheHitRunInfoPrinted_;
            HCCL_INFO("[AicpuTaskCache][FindEntry] find cache entry for cacheTag[%s]", cacheTag);
        } else {
            *entryPtrPtr = nullptr;
            HCCL_INFO("[AicpuTaskCache][FindEntry] not find cache entry for cacheTag[%s]", cacheTag);
        }
    }

    if (UNLIKELY(reportCacheHit)) {
        std::lock_guard<std::shared_timed_mutex> lock(cacheMtx_); // 写锁后二次确认
        if (!cacheHitRunInfoPrinted_) {
            cacheHitRunInfoPrinted_ = true;
            HCCL_RUN_INFO(
                "[AicpuTaskCacheManager][ReportCacheHitOnce] aicpu task cache hit for the first time in "
                "current process, cacheTag[%s]",
                cacheTag);
        }
    }

    return HCCL_SUCCESS;
}

HcclResult AicpuTaskCache::AddEntry(const char* cacheTag, AicpuTaskCacheEntry** entryPtrPtr)
{
    std::lock_guard<std::shared_timed_mutex> lock(cacheMtx_); // 写锁

    CHK_PTRPTR_NULL(entryPtrPtr);

    // 检查cache容量
    if (cacheBytes_ >= maxCacheBytes_) {
        HCCL_INFO(
            "[AicpuTaskCache][AddEntry] cacheBytes[%llu] >= maxCacheBytes[%llu] -> aicpu task cache is full "
            "not add cache entry for cacheTag[%s]",
            cacheBytes_, maxCacheBytes_, cacheTag);
        *entryPtrPtr = nullptr;
        if (!cacheFullRunInfoPrinted_) {
            cacheFullRunInfoPrinted_ = true;
            HCCL_RUN_INFO(
                "[AicpuTaskCacheManager][ReportCacheFullOnce] aicpu task cache is full for the first time "
                "in current process, cacheTag[%s]",
                cacheTag);
        }
        return HCCL_SUCCESS;
    }

    // 打印cache容量
    HCCL_INFO(
        "[AicpuTaskCache][AddEntry] cacheBytes[%llu] (maxCacheBytes[%llu]) -> aicpu task cache not full", cacheBytes_,
        maxCacheBytes_);

    CacheHashMap::iterator iter = cacheHashMap_.find(cacheTag);
    CHK_PRT_RET(
        iter != cacheHashMap_.end(),
        HCCL_ERROR("[AicpuTaskCache][AddEntry] cache entry for cacheTag[%s] exists", cacheTag), HCCL_E_INTERNAL);

    // 初始化new cache entry
    AicpuTaskCacheEntry* newCacheEntryPtr = (new (std::nothrow) AicpuTaskCacheEntry());
    CHK_PTR_NULL(newCacheEntryPtr);

    // 添加到cache
    std::pair<CacheHashMap::iterator, bool> insertResult = cacheHashMap_.emplace(cacheTag, newCacheEntryPtr);
    if (UNLIKELY(!(insertResult.second))) {
        HCCL_ERROR("[AicpuTaskCache][AddEntry] fail to insert a new cache entry for cacheTag[%s]", cacheTag);

        delete newCacheEntryPtr;
        newCacheEntryPtr = nullptr;

        return HCCL_E_INTERNAL;
    } else {
        HCCL_INFO(
            "[AicpuTaskCache][AddEntry] add a new cache entry for cacheTag[%s] cacheHashMap_.size[%zu]", cacheTag,
            cacheHashMap_.size());

        iter = insertResult.first;
        *entryPtrPtr = iter->second;
        CHK_PTR_NULL(*entryPtrPtr);
    }

    return HCCL_SUCCESS;
}

HcclResult AicpuTaskCache::IncCacheBytes(const char* cacheTag, const uint64_t entryBytes)
{
    std::lock_guard<std::shared_timed_mutex> lock(cacheMtx_); // 写锁

    cacheBytes_ += entryBytes;
    cacheBytes_ += strlen(cacheTag);
    cacheBytes_ += sizeof(AicpuTaskCacheEntry*);

    HCCL_INFO(
        "[AicpuTaskCache][IncCacheBytes] add entryBytes[%llu] + cacheTag[%zu] + AicpuTaskCacheEntry*[%llu] -> "
        "cacheBytes[%llu] (maxCacheBytes[%llu])",
        entryBytes, strlen(cacheTag), sizeof(AicpuTaskCacheEntry*), cacheBytes_, maxCacheBytes_);

    return HCCL_SUCCESS;
}

HcclResult AicpuTaskCache::ClearEntry(const char* cacheTag)
{
    std::lock_guard<std::shared_timed_mutex> lock(cacheMtx_); // 写锁

    CacheHashMap::iterator iter = cacheHashMap_.find(cacheTag);
    if (iter != cacheHashMap_.end()) {
        HCCL_INFO("[AicpuTaskCache][ClearEntry] clear cache entry for cacheTag[%s]", cacheTag);

        // 获取entry bytes
        AicpuTaskCacheEntry* entryPtr = iter->second;
        CHK_PTR_NULL(entryPtr);
        const uint64_t entryBytes = entryPtr->GetEntryBytes();

        // 计算clear bytes
        const uint64_t clearBytes = entryBytes + strlen(cacheTag) + sizeof(AicpuTaskCacheEntry*);
        // 更新cache bytes
        if (cacheBytes_ > clearBytes) {
            cacheBytes_ -= clearBytes;
        } else {
            cacheBytes_ = 0;
        }

        HCCL_INFO(
            "[AicpuTaskCache][ClearEntry] dec entryBytes[%llu] + cacheTag[%zu] + AicpuTaskCacheEntry*[%llu] -> "
            "cacheBytes[%llu] (maxCacheBytes[%llu])",
            entryBytes, strlen(cacheTag), sizeof(AicpuTaskCacheEntry*), cacheBytes_, maxCacheBytes_);

        // 释放cache entry
        delete entryPtr;
        entryPtr = nullptr;
        iter->second = nullptr;

        // 释放cacheTag-entryPtr mapping
        cacheHashMap_.erase(iter);
    } else {
        HCCL_INFO("[AicpuTaskCache][ClearEntry] not find cache entry for cacheTag[%s]", cacheTag);
    }

    return HCCL_SUCCESS;
}

} // namespace hcomm
