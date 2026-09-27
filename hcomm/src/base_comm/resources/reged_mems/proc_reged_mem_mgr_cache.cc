/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "proc_reged_mem_mgr_cache.h"
#include "log.h"

namespace hcomm {

std::shared_ptr<ProcRegedMemMgrCache> ProcRegedMemMgrCache::GetHolder()
{
    // 构造 private，make_shared 无法调用，只能 new。static 保证只 new 一次。
    static std::shared_ptr<ProcRegedMemMgrCache> holder(new ProcRegedMemMgrCache());
    return holder;
}

std::shared_ptr<RegedMemMgr>
ProcRegedMemMgrCache::GetOrCreate(const MemMgrCacheKey& key, std::function<std::shared_ptr<RegedMemMgr>()> creator)
{
    std::lock_guard<std::mutex> lock(mtx_);
    auto it = cacheMap_.find(key);
    if (it != cacheMap_.end()) {
        it->second.refCount++;
        HCCL_INFO(
            "[ProcRegedMemMgrCache][GetOrCreate] cache hit, refCount[%llu]",
            static_cast<unsigned long long>(it->second.refCount));
        return it->second.mgrPtr;
    }

    auto mgrPtr = creator();
    MemMgrEntry entry{mgrPtr, 1};
    cacheMap_.emplace(key, entry);
    HCCL_INFO("[ProcRegedMemMgrCache][GetOrCreate] cache miss, new instance inserted");
    return mgrPtr;
}

void ProcRegedMemMgrCache::Release(const MemMgrCacheKey& key)
{
    std::lock_guard<std::mutex> lock(mtx_);
    auto it = cacheMap_.find(key);
    if (it == cacheMap_.end()) {
        HCCL_WARNING("[ProcRegedMemMgrCache][Release] key not found, skip release");
        return;
    }
    if (it->second.refCount > 0) {
        it->second.refCount--;
    }
    if (it->second.refCount == 0) {
        cacheMap_.erase(it);
        HCCL_INFO("[ProcRegedMemMgrCache][Release] refCount zero, entry erased");
    } else {
        HCCL_INFO(
            "[ProcRegedMemMgrCache][Release] refCount remain [%llu]",
            static_cast<unsigned long long>(it->second.refCount));
    }
}

} // namespace hcomm
