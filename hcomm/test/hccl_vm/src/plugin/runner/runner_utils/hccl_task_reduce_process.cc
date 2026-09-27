/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hccl_task_reduce_process.h"
#include <cstdint>
#include "ccu_fp16.h"

// 本地任务定义
namespace VirtualRunTime {
// 对不同类型的做Sum的Reduce操作
template <typename T>
void MemReduceSumTemplate(void* dst_addr, const void* src_addr, uint64_t length)
{
    T* dst = reinterpret_cast<T*>(dst_addr);
    const T* src = reinterpret_cast<const T*>(src_addr);
    auto count = length / sizeof(T);
    for (auto index = 0; index < count; ++index) {
        dst[index] += src[index];
    }
}

// 对不同类型的做MIN的Reduce操作
template <typename T>
void MemReduceMinTemplate(void* dst_addr, const void* src_addr, uint64_t length)
{
    T* dst = reinterpret_cast<T*>(dst_addr);
    const T* src = reinterpret_cast<const T*>(src_addr);
    auto count = length / sizeof(T);
    for (auto index = 1; index < count; ++index) {
        if (src[index] < dst[index]) {
            dst[index] = src[index];
        }
    }
}

// 对不同类型的做MAX的Reduce操作
template <typename T>
void MemReduceMaxTemplate(void* dst_addr, const void* src_addr, uint64_t length)
{
    T* dst = reinterpret_cast<T*>(dst_addr);
    const T* src = reinterpret_cast<const T*>(src_addr);
    auto count = length / sizeof(T);
    for (auto index = 1; index < count; ++index) {
        if (src[index] > dst[index]) {
            dst[index] = src[index];
        }
    }
}

void MemReduceSum(void* src, void* dst, uint32_t length, HcclDataType dataType)
{
    switch (dataType) {
        case HcclDataType::HCCL_DATA_TYPE_INT8:
            return MemReduceSumTemplate<int8_t>(dst, src, length);
        case HcclDataType::HCCL_DATA_TYPE_INT16:
            return MemReduceSumTemplate<int16_t>(dst, src, length);
        case HcclDataType::HCCL_DATA_TYPE_INT32:
            return MemReduceSumTemplate<int32_t>(dst, src, length);
        case HcclDataType::HCCL_DATA_TYPE_INT64:
            return MemReduceSumTemplate<int64_t>(dst, src, length);
        case HcclDataType::HCCL_DATA_TYPE_UINT8:
            return MemReduceSumTemplate<uint8_t>(dst, src, length);
        case HcclDataType::HCCL_DATA_TYPE_UINT16:
            return MemReduceSumTemplate<uint16_t>(dst, src, length);
        case HcclDataType::HCCL_DATA_TYPE_UINT32:
            return MemReduceSumTemplate<uint32_t>(dst, src, length);
        case HcclDataType::HCCL_DATA_TYPE_UINT64:
            return MemReduceSumTemplate<uint64_t>(dst, src, length);
        case HcclDataType::HCCL_DATA_TYPE_FP16:
            return MemReduceSumTemplate<FP16>(dst, src, length);
        case HcclDataType::HCCL_DATA_TYPE_FP32:
            return MemReduceSumTemplate<float>(dst, src, length);
        default: {
            return;
        }
    }
}

void MemReduceMin(void* src, void* dst, uint32_t length, HcclDataType dataType)
{
    switch (dataType) {
        case HcclDataType::HCCL_DATA_TYPE_INT8:
            return MemReduceMinTemplate<int8_t>(dst, src, length);
        case HcclDataType::HCCL_DATA_TYPE_INT16:
            return MemReduceMinTemplate<int16_t>(dst, src, length);
        case HcclDataType::HCCL_DATA_TYPE_INT32:
            return MemReduceMinTemplate<int32_t>(dst, src, length);
        case HcclDataType::HCCL_DATA_TYPE_INT64:
            return MemReduceMinTemplate<int64_t>(dst, src, length);
        case HcclDataType::HCCL_DATA_TYPE_UINT8:
            return MemReduceMinTemplate<uint8_t>(dst, src, length);
        case HcclDataType::HCCL_DATA_TYPE_UINT16:
            return MemReduceMinTemplate<uint16_t>(dst, src, length);
        case HcclDataType::HCCL_DATA_TYPE_UINT32:
            return MemReduceMinTemplate<uint32_t>(dst, src, length);
        case HcclDataType::HCCL_DATA_TYPE_UINT64:
            return MemReduceMinTemplate<uint64_t>(dst, src, length);
        case HcclDataType::HCCL_DATA_TYPE_FP16:
            return MemReduceMinTemplate<FP16>(dst, src, length);
        case HcclDataType::HCCL_DATA_TYPE_FP32:
            return MemReduceMinTemplate<float>(dst, src, length);
        default: {
            return;
        }
    }
}

void MemReduceMax(void* src, void* dst, uint32_t length, HcclDataType dataType)
{
    switch (dataType) {
        case HcclDataType::HCCL_DATA_TYPE_INT8:
            return MemReduceMaxTemplate<int8_t>(dst, src, length);
        case HcclDataType::HCCL_DATA_TYPE_INT16:
            return MemReduceMaxTemplate<int16_t>(dst, src, length);
        case HcclDataType::HCCL_DATA_TYPE_INT32:
            return MemReduceMaxTemplate<int32_t>(dst, src, length);
        case HcclDataType::HCCL_DATA_TYPE_INT64:
            return MemReduceMaxTemplate<int64_t>(dst, src, length);
        case HcclDataType::HCCL_DATA_TYPE_UINT8:
            return MemReduceMaxTemplate<uint8_t>(dst, src, length);
        case HcclDataType::HCCL_DATA_TYPE_UINT16:
            return MemReduceMaxTemplate<uint16_t>(dst, src, length);
        case HcclDataType::HCCL_DATA_TYPE_UINT32:
            return MemReduceMaxTemplate<uint32_t>(dst, src, length);
        case HcclDataType::HCCL_DATA_TYPE_UINT64:
            return MemReduceMaxTemplate<uint64_t>(dst, src, length);
        case HcclDataType::HCCL_DATA_TYPE_FP16:
            return MemReduceMaxTemplate<FP16>(dst, src, length);
        case HcclDataType::HCCL_DATA_TYPE_FP32:
            return MemReduceMaxTemplate<float>(dst, src, length);
        default: {
            return;
        }
    }
}
} // namespace VirtualRunTime
