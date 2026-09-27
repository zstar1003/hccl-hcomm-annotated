/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <map>
#include "ccu_resource_common.h"
#include "ccu_microcode_common_v1.h"
#include "sim_log.h"
#include "ccu_simulator_base.h"
#include "ccu_fp16.h"

using namespace std;

template <typename T>
void ReduceAdd(const void* srcBuf, void* dstBuf, uint64_t length)
{
    // 1. 类型转换
    auto srcPtr = static_cast<const T*>(srcBuf);
    auto dstPtr = static_cast<T*>(dstBuf);

    // 2. Add
    for (uint64_t i = 0; i < (length / sizeof(T)); ++i) {
        dstPtr[i] += srcPtr[i];
    }
}

template <typename T>
void ReduceMax(const void* srcBuf, void* dstBuf, uint64_t length)
{
    // 1. 类型转换
    auto srcPtr = static_cast<const T*>(srcBuf);
    auto dstPtr = static_cast<T*>(dstBuf);

    // 2. Max
    for (uint64_t i = 0; i < (length / sizeof(T)); ++i) {
        dstPtr[i] = max(dstPtr[i], srcPtr[i]);
    }
}

template <typename T>
void ReduceMin(const void* srcBuf, void* dstBuf, uint64_t length)
{
    // 1. 类型转换
    auto srcPtr = static_cast<const T*>(srcBuf);
    auto dstPtr = static_cast<T*>(dstBuf);

    // 2. Min
    for (uint64_t i = 0; i < (length / sizeof(T)); ++i) {
        dstPtr[i] = min(dstPtr[i], srcPtr[i]);
    }
}

const std::map<TransMemReduceDataTypeV1, std::function<void(const void*, void*, uint64_t)>> transmemReduceAddFuncMapV1
    = {{TransMemReduceDataTypeV1::INT8_V1,
        [](const void* srcBuf, void* dstBuf, uint64_t length) {
            ReduceAdd<int8_t>(srcBuf, dstBuf, length);
        }},
       {TransMemReduceDataTypeV1::INT16_V1,
        [](const void* srcBuf, void* dstBuf, uint64_t length) {
            ReduceAdd<int16_t>(srcBuf, dstBuf, length);
        }},
       {TransMemReduceDataTypeV1::INT32_V1,
        [](const void* srcBuf, void* dstBuf, uint64_t length) {
            ReduceAdd<int32_t>(srcBuf, dstBuf, length);
        }},
       {TransMemReduceDataTypeV1::UINT8_V1,
        [](const void* srcBuf, void* dstBuf, uint64_t length) {
            ReduceAdd<uint8_t>(srcBuf, dstBuf, length);
        }},
       {TransMemReduceDataTypeV1::UINT16_V1,
        [](const void* srcBuf, void* dstBuf, uint64_t length) {
            ReduceAdd<uint16_t>(srcBuf, dstBuf, length);
        }},
       {TransMemReduceDataTypeV1::UINT32_V1,
        [](const void* srcBuf, void* dstBuf, uint64_t length) {
            ReduceAdd<uint32_t>(srcBuf, dstBuf, length);
        }},
       {TransMemReduceDataTypeV1::FP16_NORMAL_V1,
        [](const void* srcBuf, void* dstBuf, uint64_t length) {
            ReduceAdd<FP16>(srcBuf, dstBuf, length);
        }},
       {TransMemReduceDataTypeV1::FP32_V1, [](const void* srcBuf, void* dstBuf, uint64_t length) {
            ReduceAdd<float>(srcBuf, dstBuf, length);
        }}};

const std::map<TransMemReduceDataTypeV1, std::function<void(const void*, void*, uint64_t)>> transmemReduceMaxFuncMapV1
    = {{TransMemReduceDataTypeV1::INT8_V1,
        [](const void* srcBuf, void* dstBuf, uint64_t length) {
            ReduceMax<int8_t>(srcBuf, dstBuf, length);
        }},
       {TransMemReduceDataTypeV1::INT16_V1,
        [](const void* srcBuf, void* dstBuf, uint64_t length) {
            ReduceMax<int16_t>(srcBuf, dstBuf, length);
        }},
       {TransMemReduceDataTypeV1::INT32_V1,
        [](const void* srcBuf, void* dstBuf, uint64_t length) {
            ReduceMax<int32_t>(srcBuf, dstBuf, length);
        }},
       {TransMemReduceDataTypeV1::UINT8_V1,
        [](const void* srcBuf, void* dstBuf, uint64_t length) {
            ReduceMax<uint8_t>(srcBuf, dstBuf, length);
        }},
       {TransMemReduceDataTypeV1::UINT16_V1,
        [](const void* srcBuf, void* dstBuf, uint64_t length) {
            ReduceMax<uint16_t>(srcBuf, dstBuf, length);
        }},
       {TransMemReduceDataTypeV1::UINT32_V1,
        [](const void* srcBuf, void* dstBuf, uint64_t length) {
            ReduceMax<uint32_t>(srcBuf, dstBuf, length);
        }},
       {TransMemReduceDataTypeV1::FP16_NORMAL_V1,
        [](const void* srcBuf, void* dstBuf, uint64_t length) {
            ReduceMax<FP16>(srcBuf, dstBuf, length);
        }},
       {TransMemReduceDataTypeV1::FP32_V1, [](const void* srcBuf, void* dstBuf, uint64_t length) {
            ReduceMax<float>(srcBuf, dstBuf, length);
        }}};

const std::map<TransMemReduceDataTypeV1, std::function<void(const void*, void*, uint64_t)>> transmemReduceMinFuncMapV1
    = {{TransMemReduceDataTypeV1::INT8_V1,
        [](const void* srcBuf, void* dstBuf, uint64_t length) {
            ReduceMin<int8_t>(srcBuf, dstBuf, length);
        }},
       {TransMemReduceDataTypeV1::INT16_V1,
        [](const void* srcBuf, void* dstBuf, uint64_t length) {
            ReduceMin<int16_t>(srcBuf, dstBuf, length);
        }},
       {TransMemReduceDataTypeV1::INT32_V1,
        [](const void* srcBuf, void* dstBuf, uint64_t length) {
            ReduceMin<int32_t>(srcBuf, dstBuf, length);
        }},
       {TransMemReduceDataTypeV1::UINT8_V1,
        [](const void* srcBuf, void* dstBuf, uint64_t length) {
            ReduceMin<uint8_t>(srcBuf, dstBuf, length);
        }},
       {TransMemReduceDataTypeV1::UINT16_V1,
        [](const void* srcBuf, void* dstBuf, uint64_t length) {
            ReduceMin<uint16_t>(srcBuf, dstBuf, length);
        }},
       {TransMemReduceDataTypeV1::UINT32_V1,
        [](const void* srcBuf, void* dstBuf, uint64_t length) {
            ReduceMin<uint32_t>(srcBuf, dstBuf, length);
        }},
       {TransMemReduceDataTypeV1::FP16_NORMAL_V1,
        [](const void* srcBuf, void* dstBuf, uint64_t length) {
            ReduceMin<FP16>(srcBuf, dstBuf, length);
        }},
       {TransMemReduceDataTypeV1::FP32_V1, [](const void* srcBuf, void* dstBuf, uint64_t length) {
            ReduceMin<float>(srcBuf, dstBuf, length);
        }}};

const std::map<TransMemReduceDataTypeV2, std::function<void(const void*, void*, uint64_t)>> transmemReduceAddFuncMapV2
    = {{TransMemReduceDataTypeV2::INT8_V2,
        [](const void* srcBuf, void* dstBuf, uint64_t length) {
            ReduceAdd<int8_t>(srcBuf, dstBuf, length);
        }},
       {TransMemReduceDataTypeV2::INT16_V2,
        [](const void* srcBuf, void* dstBuf, uint64_t length) {
            ReduceAdd<int16_t>(srcBuf, dstBuf, length);
        }},
       {TransMemReduceDataTypeV2::INT32_V2,
        [](const void* srcBuf, void* dstBuf, uint64_t length) {
            ReduceAdd<int32_t>(srcBuf, dstBuf, length);
        }},
       {TransMemReduceDataTypeV2::UINT32_V2,
        [](const void* srcBuf, void* dstBuf, uint64_t length) {
            ReduceAdd<uint32_t>(srcBuf, dstBuf, length);
        }},
       {TransMemReduceDataTypeV2::FP16_NORMAL_V2,
        [](const void* srcBuf, void* dstBuf, uint64_t length) {
            ReduceAdd<FP16>(srcBuf, dstBuf, length);
        }},
       {TransMemReduceDataTypeV2::FP32_V2, [](const void* srcBuf, void* dstBuf, uint64_t length) {
            ReduceAdd<float>(srcBuf, dstBuf, length);
        }}};

const std::map<TransMemReduceDataTypeV2, std::function<void(const void*, void*, uint64_t)>> transmemReduceMaxFuncMapV2
    = {{TransMemReduceDataTypeV2::INT8_V2,
        [](const void* srcBuf, void* dstBuf, uint64_t length) {
            ReduceMax<int8_t>(srcBuf, dstBuf, length);
        }},
       {TransMemReduceDataTypeV2::INT16_V2,
        [](const void* srcBuf, void* dstBuf, uint64_t length) {
            ReduceMax<int16_t>(srcBuf, dstBuf, length);
        }},
       {TransMemReduceDataTypeV2::INT32_V2,
        [](const void* srcBuf, void* dstBuf, uint64_t length) {
            ReduceMax<int32_t>(srcBuf, dstBuf, length);
        }},
       {TransMemReduceDataTypeV2::UINT32_V2,
        [](const void* srcBuf, void* dstBuf, uint64_t length) {
            ReduceMax<uint32_t>(srcBuf, dstBuf, length);
        }},
       {TransMemReduceDataTypeV2::FP16_NORMAL_V2,
        [](const void* srcBuf, void* dstBuf, uint64_t length) {
            ReduceMax<FP16>(srcBuf, dstBuf, length);
        }},
       {TransMemReduceDataTypeV2::FP32_V2, [](const void* srcBuf, void* dstBuf, uint64_t length) {
            ReduceMax<float>(srcBuf, dstBuf, length);
        }}};

const std::map<TransMemReduceDataTypeV2, std::function<void(const void*, void*, uint64_t)>> transmemReduceMinFuncMapV2
    = {{TransMemReduceDataTypeV2::INT8_V2,
        [](const void* srcBuf, void* dstBuf, uint64_t length) {
            ReduceMin<int8_t>(srcBuf, dstBuf, length);
        }},
       {TransMemReduceDataTypeV2::INT16_V2,
        [](const void* srcBuf, void* dstBuf, uint64_t length) {
            ReduceMin<int16_t>(srcBuf, dstBuf, length);
        }},
       {TransMemReduceDataTypeV2::INT32_V2,
        [](const void* srcBuf, void* dstBuf, uint64_t length) {
            ReduceMin<int32_t>(srcBuf, dstBuf, length);
        }},
       {TransMemReduceDataTypeV2::UINT32_V2,
        [](const void* srcBuf, void* dstBuf, uint64_t length) {
            ReduceMin<uint32_t>(srcBuf, dstBuf, length);
        }},
       {TransMemReduceDataTypeV2::FP16_NORMAL_V2,
        [](const void* srcBuf, void* dstBuf, uint64_t length) {
            ReduceMin<FP16>(srcBuf, dstBuf, length);
        }},
       {TransMemReduceDataTypeV2::FP32_V2, [](const void* srcBuf, void* dstBuf, uint64_t length) {
            ReduceMin<float>(srcBuf, dstBuf, length);
        }}};

bool ReduceAddProcess(const void* srcBuf, void* dstBuf, uint64_t length, uint16_t dataType, RunnerCcuVersion version)
{
    HCCL_VM_INFO("ReduceAddProcess dataType {} length {} srcBuf {} dstBuf {}", dataType, length, srcBuf, dstBuf);

    if (version == RunnerCcuVersion::CCU_V1) {
        TransMemReduceDataTypeV1 type = static_cast<TransMemReduceDataTypeV1>(dataType);
        auto res = transmemReduceAddFuncMapV1.find(type);
        if (res != transmemReduceAddFuncMapV1.end()) {
            res->second(srcBuf, dstBuf, length);
            return true;
        }
    } else if (version == RunnerCcuVersion::CCU_V2) {
        TransMemReduceDataTypeV2 type = static_cast<TransMemReduceDataTypeV2>(dataType);
        auto res = transmemReduceAddFuncMapV2.find(type);
        if (res != transmemReduceAddFuncMapV2.end()) {
            res->second(srcBuf, dstBuf, length);
            return true;
        }
    }
    HCCL_VM_ERROR("ReduceAddProcess dataType {} not support", dataType);
    return false;
}

bool ReduceMaxProcess(const void* srcBuf, void* dstBuf, uint64_t length, uint16_t dataType, RunnerCcuVersion version)
{
    HCCL_VM_INFO("ReduceMaxProcess dataType {} length {} srcBuf {} dstBuf {}", dataType, length, srcBuf, dstBuf);

    if (version == RunnerCcuVersion::CCU_V1) {
        TransMemReduceDataTypeV1 type = static_cast<TransMemReduceDataTypeV1>(dataType);
        auto res = transmemReduceMaxFuncMapV1.find(type);
        if (res != transmemReduceMaxFuncMapV1.end()) {
            res->second(srcBuf, dstBuf, length);
            return true;
        }
    } else if (version == RunnerCcuVersion::CCU_V2) {
        TransMemReduceDataTypeV2 type = static_cast<TransMemReduceDataTypeV2>(dataType);
        auto res = transmemReduceMaxFuncMapV2.find(type);
        if (res != transmemReduceMaxFuncMapV2.end()) {
            res->second(srcBuf, dstBuf, length);
            return true;
        }
    }
    HCCL_VM_ERROR("ReduceMaxProcess dataType {} not support", dataType);
    return false;
}

bool ReduceMinProcess(const void* srcBuf, void* dstBuf, uint64_t length, uint16_t dataType, RunnerCcuVersion version)
{
    HCCL_VM_INFO("ReduceMinProcess dataType {} length {} srcBuf {} dstBuf {}", dataType, length, srcBuf, dstBuf);

    if (version == RunnerCcuVersion::CCU_V1) {
        TransMemReduceDataTypeV1 type = static_cast<TransMemReduceDataTypeV1>(dataType);
        auto res = transmemReduceMinFuncMapV1.find(type);
        if (res != transmemReduceMinFuncMapV1.end()) {
            res->second(srcBuf, dstBuf, length);
            return true;
        }
    } else if (version == RunnerCcuVersion::CCU_V2) {
        TransMemReduceDataTypeV2 type = static_cast<TransMemReduceDataTypeV2>(dataType);
        auto res = transmemReduceMinFuncMapV2.find(type);
        if (res != transmemReduceMinFuncMapV2.end()) {
            res->second(srcBuf, dstBuf, length);
            return true;
        }
    }
    HCCL_VM_ERROR("ReduceMinProcess dataType {} not support", dataType);
    return false;
}

bool ReduceProcess(
    const void* srcBuf, void* dstBuf, uint64_t length, uint16_t dataType, uint32_t reduceOp, RunnerCcuVersion version)
{
    switch (reduceOp) {
        case 8: // MAX
            return ReduceMaxProcess(srcBuf, dstBuf, length, dataType, version);
        case 9: // MIN
            return ReduceMinProcess(srcBuf, dstBuf, length, dataType, version);
        case 10: // SUM
            return ReduceAddProcess(srcBuf, dstBuf, length, dataType, version);
        case 11: // EQUAL
            HCCL_VM_ERROR("Reduce reduceOp {} not support", reduceOp);
            return false;
        default:
            HCCL_VM_ERROR("Reduce reduceOp {} not support", reduceOp);
            return false;
    }
}
