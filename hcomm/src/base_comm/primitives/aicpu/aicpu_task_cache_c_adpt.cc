/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "cast_utils.h"
#include "hcomm_primitives.h"

#include <cstring>

#include "dtype_common.h"
#include "aicpu_task_cache_manager.h"
#include "log.h"

using namespace hcomm;

HcommResult HcommAicpuTsTaskCacheLookup(const char* tag, bool* isHit)
{
    DevType deviceType;
    CHK_RET(hrtGetDeviceType(deviceType));
    if (deviceType != DevType::DEV_TYPE_950) {
        HCCL_ERROR("[%s] deviceType[%d] is not supported", __func__, deviceType);
        return HCCL_E_NOT_SUPPORT;
    }

    CHK_PTR_NULL(tag);
    CHK_PTR_NULL(isHit);

    // 记录当前tag, 后续校验tag一致性
    AicpuTaskCacheManager::cacheTag = tag;

    // 注意: 相同tag的算子一定不会被多个aicpu kernel threads同时展开,
    // 否则后续threads可能命中第一个thread插入的不完整的cache entry 例如: tag中有commId时,
    // 相同tag的算子一定属于同一个通信域, 必定按序展开 因此, FindEntry和AddEntry不需要统一成单个接口, 即FindEntry后,
    // 当前tag的缓存状态不会被其他threads改变, 可以直接AddEntry

    // 查询tag对应的cache entry
    AicpuTaskCacheManager::cacheEntryPtr = nullptr;
    CHK_RET(AicpuTaskCacheManager::aicpuTaskCache.FindEntry(tag, &AicpuTaskCacheManager::cacheEntryPtr));

    // 判断并记录是否为cache hit
    AicpuTaskCacheManager::isHit = (AicpuTaskCacheManager::cacheEntryPtr != nullptr);
    *isHit = AicpuTaskCacheManager::isHit;

    // 如果是cache miss, 尝试添加cache entry
    // 注意: 如果aicpu task cache容量已满, 不会添加新的cache entry, AicpuTaskCacheManager::cacheEntryPtr被设置为nullptr
    if (!(*isHit)) {
        CHK_RET(AicpuTaskCacheManager::aicpuTaskCache.AddEntry(tag, &AicpuTaskCacheManager::cacheEntryPtr));
    }

    return HCCL_SUCCESS;
}

HcommResult HcommAicpuTsTaskCacheStart(const char* tag, void** addrs, uint64_t* sizes, uint64_t count)
{
    DevType deviceType;
    CHK_RET(hrtGetDeviceType(deviceType));
    if (deviceType != DevType::DEV_TYPE_950) {
        HCCL_ERROR("[HcommAicpuTsTaskCacheStart] deviceType[%d] is not supported", deviceType);
        return HCCL_E_NOT_SUPPORT;
    }

    CHK_PTR_NULL(tag);
    CHK_PTR_NULL(addrs);
    CHK_PTR_NULL(sizes);

    // 校验tag一致性 (即Submit对应的tag一定与Lookup对应的tag一致)
    CHK_PRT_RET(
        strcmp(AicpuTaskCacheManager::cacheTag.c_str(), tag) != 0,
        HCCL_ERROR(
            "[HcommAicpuTsTaskCacheStart] submit's tag[%s] != lookup's tag[%s]", tag,
            AicpuTaskCacheManager::cacheTag.c_str()),
        HCCL_E_PARA);

    // 一定是cache miss
    CHK_PRT_RET(
        AicpuTaskCacheManager::isHit, HCCL_ERROR("[HcommAicpuTsTaskCacheStart] cache hit, but should be miss"),
        HCCL_E_INTERNAL);

    // Aicpu task cache容量未满
    if (AicpuTaskCacheManager::cacheEntryPtr != nullptr) {
        // 保存地址信息到cache entry
        CHK_RET(
            AicpuTaskCacheManager::cacheEntryPtr->InitCacheEntry(ReinterpretAs<const uint64_t*>(addrs), sizes, count));
    }

    return HCCL_SUCCESS;
}

HcommResult HcommAicpuTsTaskCacheEnd(const char* tag)
{
    DevType deviceType;
    CHK_RET(hrtGetDeviceType(deviceType));
    if (deviceType != DevType::DEV_TYPE_950) {
        HCCL_ERROR("[HcommAicpuTsTaskCacheEnd] deviceType[%d] is not supported", deviceType);
        return HCCL_E_NOT_SUPPORT;
    }

    CHK_PTR_NULL(tag);

    // 校验tag一致性 (即Submit对应的tag一定与Lookup对应的tag一致)
    CHK_PRT_RET(
        strcmp(AicpuTaskCacheManager::cacheTag.c_str(), tag) != 0,
        HCCL_ERROR(
            "[HcommAicpuTsTaskCacheEnd] submit's tag[%s] != lookup's tag[%s]", tag,
            AicpuTaskCacheManager::cacheTag.c_str()),
        HCCL_E_PARA);

    // 一定是cache miss
    CHK_PRT_RET(
        AicpuTaskCacheManager::isHit, HCCL_ERROR("[HcommAicpuTsTaskCacheEnd] cache hit, but should be miss"),
        HCCL_E_INTERNAL);

    HcclResult ret = HCCL_SUCCESS;
    do {
        // Aicpu task cache容量未满
        if (AicpuTaskCacheManager::cacheEntryPtr != nullptr) {
            // 提交cache entry, 更新cache entry内部信息
            ret = AicpuTaskCacheManager::cacheEntryPtr->SubmitCacheEntry();
            CHK_PRT_BREAK(
                ret != HCCL_SUCCESS, HCCL_ERROR("[HcommAicpuTsTaskCacheEnd] SubmitCacheEntry error,ret[%d]", ret),
                (void)0);

            // 更新cache空间消耗
            const uint64_t entryBytes = AicpuTaskCacheManager::cacheEntryPtr->GetEntryBytes();
            ret = AicpuTaskCacheManager::aicpuTaskCache.IncCacheBytes(
                AicpuTaskCacheManager::cacheTag.c_str(), entryBytes);
            CHK_PRT_BREAK(
                ret != HCCL_SUCCESS, HCCL_ERROR("[HcommAicpuTsTaskCacheEnd] IncCacheBytes error, ret[%d]", ret),
                (void)0);
        }
    } while (0);

    // 重置cache上下文
    AicpuTaskCacheManager::cacheTag.clear();
    AicpuTaskCacheManager::isHit = false;
    AicpuTaskCacheManager::cacheEntryPtr = nullptr;

    return ret;
}

HcommResult HcommAicpuTsTaskCacheExecute(const char* tag, void** addrs, uint64_t* sizes, uint64_t count)
{
    DevType deviceType;
    CHK_RET(hrtGetDeviceType(deviceType));
    if (deviceType != DevType::DEV_TYPE_950) {
        HCCL_ERROR("[HcommAicpuTsTaskCacheExecute] deviceType[%d] is not supported", deviceType);
        return HCCL_E_NOT_SUPPORT;
    }

    CHK_PTR_NULL(tag);
    CHK_PTR_NULL(addrs);
    CHK_PTR_NULL(sizes);

    // 校验tag一致性 (即Submit对应的tag一定与Lookup对应的tag一致)
    CHK_PRT_RET(
        strcmp(AicpuTaskCacheManager::cacheTag.c_str(), tag) != 0,
        HCCL_ERROR(
            "[HcommAicpuTsTaskCacheExecute] submit's tag[%s] != lookup's tag[%s]", tag,
            AicpuTaskCacheManager::cacheTag.c_str()),
        HCCL_E_PARA);

    // 一定是cache hit
    CHK_PRT_RET(
        !AicpuTaskCacheManager::isHit, HCCL_ERROR("[HcommAicpuTsTaskCacheExecute] cache miss, but should be hit"),
        HCCL_E_INTERNAL);

    // cache hit一定存在对应的cache entry
    CHK_PTR_NULL(AicpuTaskCacheManager::cacheEntryPtr);

    // 刷新并下发task
    HcclResult ret
        = AicpuTaskCacheManager::cacheEntryPtr->RefreshAndLaunch(ReinterpretAs<const uint64_t*>(addrs), sizes, count);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[HcommAicpuTsTaskCacheExecute] RefreshAndLaunch error, ret[%d]", ret);
    }

    // 重置cache上下文
    AicpuTaskCacheManager::cacheTag.clear();
    AicpuTaskCacheManager::isHit = false;
    AicpuTaskCacheManager::cacheEntryPtr = nullptr;

    return ret;
}

HcommResult HcommAicpuTsTaskCacheClear(const char* tag)
{
    DevType deviceType;
    CHK_RET(hrtGetDeviceType(deviceType));
    if (deviceType != DevType::DEV_TYPE_950) {
        HCCL_ERROR("[%s] deviceType[%d] is not supported", __func__, deviceType);
        return HCCL_E_NOT_SUPPORT;
    }

    CHK_PTR_NULL(tag);

    // 清除tag对应的cache entry (if any)
    CHK_RET(AicpuTaskCacheManager::aicpuTaskCache.ClearEntry(tag));
    if (strcmp(AicpuTaskCacheManager::cacheTag.c_str(), tag) == 0) {
        // 重置cache上下文
        AicpuTaskCacheManager::cacheTag.clear();
        AicpuTaskCacheManager::isHit = false;
        AicpuTaskCacheManager::cacheEntryPtr = nullptr;
    }

    return HCCL_SUCCESS;
}
