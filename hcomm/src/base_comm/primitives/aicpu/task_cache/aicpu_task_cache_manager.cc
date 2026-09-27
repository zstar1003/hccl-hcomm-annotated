/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aicpu_task_cache_manager.h"

#include "log.h"

namespace hcomm {

// 全局变量定义
AicpuTaskCache AicpuTaskCacheManager::aicpuTaskCache;

// 全局线程变量定义
thread_local std::string AicpuTaskCacheManager::cacheTag = "";
thread_local bool AicpuTaskCacheManager::isHit = false;
thread_local AicpuTaskCacheEntry* AicpuTaskCacheManager::cacheEntryPtr = nullptr;

bool AicpuTaskCacheManager::NeedCacheTask()
{
    // cache miss且cache容量未满
    return !isHit && cacheEntryPtr != nullptr;
}

HcclResult AicpuTaskCacheManager::AddWqeArray(
    Hccl::UbConnLite* ubConnLitePtr, Hccl::UbTransportLiteImpl* ubTransportLiteImplPtr,
    const std::vector<Hccl::WqeTask>& wqeTasks, const uint32_t streamId, const uint32_t dbSqeIdx,
    const bool isReportTask, const Hccl::DbSqeProfInfo& dbSqeProfInfo)
{
    // 注意: 只有需要cache task时, UbTransportLiteImpl才会调用本函数
    CHK_PRT_RET(
        NeedCacheTask() == false,
        HCCL_ERROR(
            "[AicpuTaskCacheManager][AddWqeArray] should not invoke if isHit[%d] cacheEntryPtr[0x%016llx]", isHit,
            cacheEntryPtr),
        HCCL_E_INTERNAL);

    CHK_PTR_NULL(ubConnLitePtr);

    CHK_RET(cacheEntryPtr->AddWqeArray(
        ubConnLitePtr, ubTransportLiteImplPtr, wqeTasks, streamId, dbSqeIdx, isReportTask, dbSqeProfInfo));

    return HCCL_SUCCESS;
}

HcclResult AicpuTaskCacheManager::AddSqeArray(
    Hccl::RtsqA5* rtsqPtr, hccl::AicpuTsThread* aicpuTsThreadPtr, const uint64_t sqeCount, const uint8_t* sqeArray,
    const uint32_t streamId)
{
    // 注意: 只有需要cache task时, RtsqA5才会调用本函数
    CHK_PRT_RET(
        NeedCacheTask() == false,
        HCCL_ERROR(
            "[AicpuTaskCacheManager][AddSqeArray] should not invoke if isHit[%d] cacheEntryPtr[0x%016llx]", isHit,
            cacheEntryPtr),
        HCCL_E_INTERNAL);

    CHK_PTR_NULL(rtsqPtr);
    CHK_PTR_NULL(aicpuTsThreadPtr);

    CHK_RET(cacheEntryPtr->AddSqeArray(rtsqPtr, aicpuTsThreadPtr, sqeCount, sqeArray, streamId));

    return HCCL_SUCCESS;
}

} // namespace hcomm
