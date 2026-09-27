/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCOMM_AICPU_TASK_CACHE_MANAGER_H
#define HCOMM_AICPU_TASK_CACHE_MANAGER_H

#include <cstdint>
#include <string>
#include <vector>

#include "aicpu_task_cache.h"

namespace hcomm {

class AicpuTaskCacheManager {
public:
    // 注意: 由于存在全局线程变量, 这里不适用单例模式

    // cache miss且cache容量未满
    static bool NeedCacheTask();

    static HcclResult AddWqeArray(
        Hccl::UbConnLite* ubConnLitePtr, Hccl::UbTransportLiteImpl* ubTransportLiteImplPtr,
        const std::vector<Hccl::WqeTask>& wqeTasks, const uint32_t streamId, const uint32_t dbSqeIdx,
        const bool isReportTask, const Hccl::DbSqeProfInfo& dbSqeProfInfo);

    static HcclResult AddSqeArray(
        Hccl::RtsqA5* rtsqPtr, hccl::AicpuTsThread* aicpuTsThreadPtr, const uint64_t sqeCount, const uint8_t* sqeArray,
        const uint32_t streamId);

    // 多个通信域共享的全局变量 (仅声明)
    static AicpuTaskCache aicpuTaskCache;

    // 单个通信域aicpu kfc线程独立使用的全局线程变量 (仅声明)
    // Case 1: isHit == false && cacheEntryPtr == nullptr -> aicpu task cache未命中且容量已满, 不进行cache
    // Case 2: isHit == false && cacheEntryPtr != nullptr -> aicpu task cache未命中, 创建新的cache entry
    // Case 3: isHit == true && cacheEntryPtr != nullptr -> aicpu task cache命中, 使用已有cache entry刷新下发task
    static thread_local std::string cacheTag;
    static thread_local bool isHit;
    static thread_local AicpuTaskCacheEntry* cacheEntryPtr;
};

} // namespace hcomm

#endif
