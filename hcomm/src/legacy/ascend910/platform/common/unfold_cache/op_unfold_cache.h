/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __OP_UNFOLD_CACHE_H__
#define __OP_UNFOLD_CACHE_H__

#include <unordered_map>

#include "aicpu_hccl_sqcqv2.h"
#include "op_unfold_key.h"
#include "op_unfold_cache_entry.h"
#include "rt_external_stars_define.h"

namespace hccl {

// 算子展开的动态缓存 (每个通信域单独维护一个动态缓存; 针对单算子模式的buffer copy和zero copy两种场景)
// 注意: 目前不考虑variable类型算子，也不考虑BatchSendRecv算子
// 注意: A3下scratch memory一定在HCCL buffer中, 所以目前不考虑scratch memory的更新
class OpUnfoldCache {
public:
    explicit OpUnfoldCache();
    ~OpUnfoldCache();

    bool IsCacheFull() const;

    HcclResult FindEntry(const OpUnfoldKey& key, OpUnfoldCacheEntry** entryPtrPtr)
        const; // 查看是否存在key对应的cache entry (如果不存在, *entryPtrPtr会被置为空)
    HcclResult AddEntry(
        const OpUnfoldKey& key, const std::vector<OpUnfoldMemRange>& userInputMemRanges,
        const std::vector<OpUnfoldMemRange>& userOutputMemRanges,
        OpUnfoldCacheEntry** entryPtrPtr);         // 插入新的cache entry
    HcclResult ClearEntry(const OpUnfoldKey& key); // 如果key存在对应的cache entry, 清理entry

    HcclResult ClearEntryForAlltoallv(); // 清理与alltoallv类算子相关的cache entry

    // 只会在DEBUG_LEVEL下打印SQE内容 (通过比较打印算子正常展开的SQE与缓存的SQE, 判断刷新后的SQE是否正确)
    static HcclResult DumpSqeContent(const uint8_t* sqePtr, const uint8_t sqeType);

private:
    using CacheHashMap = std::unordered_map<OpUnfoldKey, OpUnfoldCacheEntry*>;

    // 只会在DEBUG_LEVEL下打印SQE header的内容
    static HcclResult DumpSqeHeader(const rtStarsSqeHeader_t& sqeHeader);
    static HcclResult DumpSqeHeader(const rtStarsSqeHeaderV2_t& sqeHeader);

    CacheHashMap cacheHashMap_; // key-entry mapping
};

} // namespace hccl

#endif // __OP_UNFOLD_CACHE_H__
