/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCOMM_AICPU_TASK_CACHE_H
#define HCOMM_AICPU_TASK_CACHE_H

#include <shared_mutex>
#include <unordered_map>

#include "aicpu_task_cache_entry.h"

namespace hcomm {

class AicpuTaskCache {
public:
    constexpr static uint64_t AICPU_TASK_CACHE_CAPACITY = 10 * 1024 * 1024; // 10 MiB by default

    explicit AicpuTaskCache(const uint64_t maxCacheBytes = AICPU_TASK_CACHE_CAPACITY);
    ~AicpuTaskCache();

    HcclResult FindEntry(const char* cacheTag, AicpuTaskCacheEntry** entryPtrPtr) const;

    // Cache miss
    HcclResult AddEntry(const char* cacheTag, AicpuTaskCacheEntry** entryPtrPtr); // 如果cache满了, 则设置nullptr
    HcclResult IncCacheBytes(const char* cacheTag, const uint64_t entryBytes);

    HcclResult ClearEntry(const char* cacheTag);

private:
    using CacheHashMap = std::unordered_map<std::string, AicpuTaskCacheEntry*>;

    mutable std::shared_timed_mutex cacheMtx_; // 多个通信域共享的超时读写锁

    CacheHashMap cacheHashMap_;
    uint64_t cacheBytes_ = 0;
    const uint64_t maxCacheBytes_;
    mutable bool cacheHitRunInfoPrinted_ = false;
    bool cacheFullRunInfoPrinted_ = false;
};

} // namespace hcomm

#endif // HCOMM_AICPU_TASK_CACHE_H
