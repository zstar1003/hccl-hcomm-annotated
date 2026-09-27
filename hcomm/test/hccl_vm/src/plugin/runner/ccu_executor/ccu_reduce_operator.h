/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_SIM_CCU_REDUCE_OPERATOR_H
#define HCCL_SIM_CCU_REDUCE_OPERATOR_H

#include <cstdint>
#include <functional>
#include <map>

#include "ccu_resource_manager.h"
#include "ccu_fp16.h"

bool ReduceProcess(
    const void* srcBuf, void* dstBuf, uint64_t length, uint16_t dataType, uint32_t reduceOp, RunnerCcuVersion version);

// FP32/INT16/INT32/UINT8类型
template <typename T>
void ReduceAdd(int rankId, int dieId, uint16_t msId[], uint16_t count)
{
    auto& ccuResMgr = CcuResourceManager::GetInstance();
    auto msValuePtr0 = reinterpret_cast<T*>(ccuResMgr.GetMsAddr(rankId, dieId, msId[0]));
    for (uint16_t i = 1; i < count + 2; i++) {
        auto msValuePtr = reinterpret_cast<T*>(ccuResMgr.GetMsAddr(rankId, dieId, msId[i]));
        for (uint16_t j = 0; j < HcclSim::BYTE_NUM_4K / sizeof(T); j++) {
            *(msValuePtr0 + j) += *(msValuePtr + j);
        }
    }
}

template <typename T>
void ReduceAddFp16(int rankId, int dieId, uint16_t msId[], uint16_t count, uint16_t castEn)
{
    auto& ccuResMgr = CcuResourceManager::GetInstance();
    auto msValuePtr0 = reinterpret_cast<T*>(ccuResMgr.GetMsAddr(rankId, dieId, msId[0]));
    for (uint16_t i = 1; i < count + 2; i++) {
        auto msValuePtr = reinterpret_cast<T*>(ccuResMgr.GetMsAddr(rankId, dieId, msId[i]));
        for (uint16_t j = 0; j < HcclSim::BYTE_NUM_4K / sizeof(T); j++) {
            *(msValuePtr0 + j) += *(msValuePtr + j);
        }
    }
}

const std::map<ReduceAddDataType, std::function<void(int, int, uint16_t[], uint16_t, uint16_t)>> reduceAddFuncMap
    = {{ReduceAddDataType::ADD_INT16,
        [](int rankId, int dieId, uint16_t msId[], uint16_t count, uint16_t castEn) {
            ReduceAdd<int16_t>(rankId, dieId, msId, count);
        }},
       {ReduceAddDataType::ADD_INT32,
        [](int rankId, int dieId, uint16_t msId[], uint16_t count, uint16_t castEn) {
            ReduceAdd<int32_t>(rankId, dieId, msId, count);
        }},
       {ReduceAddDataType::ADD_UINT8,
        [](int rankId, int dieId, uint16_t msId[], uint16_t count, uint16_t castEn) {
            ReduceAdd<uint8_t>(rankId, dieId, msId, count);
        }},
       {ReduceAddDataType::ADD_INT8,
        [](int rankId, int dieId, uint16_t msId[], uint16_t count, uint16_t castEn) {
            ReduceAdd<int8_t>(rankId, dieId, msId, count);
        }},
       {ReduceAddDataType::ADD_FP32,
        [](int rankId, int dieId, uint16_t msId[], uint16_t count, uint16_t castEn) {
            ReduceAdd<float>(rankId, dieId, msId, count);
        }},
       {ReduceAddDataType::ADD_FP16, [](int rankId, int dieId, uint16_t msId[], uint16_t count, uint16_t castEn) {
            ReduceAddFp16<FP16>(rankId, dieId, msId, count, castEn);
        }}};

// INT16/INT32/UINT8/INT8类型
template <typename T>
void ReduceMax(int rankId, int dieId, uint16_t msId[], uint16_t count)
{
    auto& ccuResMgr = CcuResourceManager::GetInstance();
    auto msValuePtr0 = reinterpret_cast<T*>(ccuResMgr.GetMsAddr(rankId, dieId, msId[0]));
    for (uint16_t i = 1; i < count + 2; i++) {
        auto msValuePtr = reinterpret_cast<T*>(ccuResMgr.GetMsAddr(rankId, dieId, msId[i]));
        for (uint16_t j = 0; j < HcclSim::BYTE_NUM_4K / sizeof(T); j++) {
            T ms0Value = *(msValuePtr0 + j);
            T msxValue = *(msValuePtr + j);
            *(msValuePtr0 + j) = (ms0Value > msxValue) ? ms0Value : msxValue;
        }
    }
}

// FP32类型
template <typename T>
void ReduceMaxFp32(int rankId, int dieId, uint16_t msId[], uint16_t count)
{
    float epsilon = 1e-5;
    auto& ccuResMgr = CcuResourceManager::GetInstance();
    auto msValuePtr0 = reinterpret_cast<T*>(ccuResMgr.GetMsAddr(rankId, dieId, msId[0]));
    for (uint16_t i = 1; i < count + 2; i++) {
        auto msValuePtr = reinterpret_cast<T*>(ccuResMgr.GetMsAddr(rankId, dieId, msId[i]));
        for (uint16_t j = 0; j < HcclSim::BYTE_NUM_4K / sizeof(T); j++) {
            T ms0Value = *(msValuePtr0 + j);
            T msxValue = *(msValuePtr + j);
            *(msValuePtr0 + j) = (fabs(ms0Value - msxValue) >= epsilon) ? ms0Value : msxValue;
        }
    }
}

// FP16类型
template <typename T>
void ReduceMaxFp16(int rankId, int dieId, uint16_t msId[], uint16_t count)
{
    float epsilon = 1e-5;
    auto& ccuResMgr = CcuResourceManager::GetInstance();
    auto msValuePtr0 = reinterpret_cast<T*>(ccuResMgr.GetMsAddr(rankId, dieId, msId[0]));
    for (uint16_t i = 1; i < count + 2; i++) {
        auto msValuePtr = reinterpret_cast<T*>(ccuResMgr.GetMsAddr(rankId, dieId, msId[i]));
        for (uint16_t j = 0; j < HcclSim::BYTE_NUM_4K / sizeof(T); j++) {
            T ms0Value = *(msValuePtr0 + j);
            T msxValue = *(msValuePtr + j);
            *(msValuePtr0 + j) = (fabs(ms0Value.to_float() - msxValue.to_float()) >= epsilon) ? ms0Value : msxValue;
        }
    }
}

const std::map<ReduceMaxMinDataType, std::function<void(int, int, uint16_t[], uint16_t)>> reduceMaxFuncMap
    = {{ReduceMaxMinDataType::MAX_MIN_INT16,
        [](int rankId, int dieId, uint16_t msId[], uint16_t count) {
            ReduceMax<int16_t>(rankId, dieId, msId, count);
        }},
       {ReduceMaxMinDataType::MAX_MIN_INT32,
        [](int rankId, int dieId, uint16_t msId[], uint16_t count) {
            ReduceMax<int32_t>(rankId, dieId, msId, count);
        }},
       {ReduceMaxMinDataType::MAX_MIN_UINT8,
        [](int rankId, int dieId, uint16_t msId[], uint16_t count) {
            ReduceMax<uint8_t>(rankId, dieId, msId, count);
        }},
       {ReduceMaxMinDataType::MAX_MIN_FP32,
        [](int rankId, int dieId, uint16_t msId[], uint16_t count) {
            ReduceMaxFp32<float>(rankId, dieId, msId, count);
        }},
       {ReduceMaxMinDataType::MAX_MIN_INT8,
        [](int rankId, int dieId, uint16_t msId[], uint16_t count) {
            ReduceMax<int8_t>(rankId, dieId, msId, count);
        }},
       {ReduceMaxMinDataType::MAX_MIN_FP16, [](int rankId, int dieId, uint16_t msId[], uint16_t count) {
            ReduceMaxFp16<FP16>(rankId, dieId, msId, count);
        }}};

// INT16/INT32/UINT8/INT8类型
template <typename T>
void ReduceMin(int rankId, int dieId, uint16_t msId[], uint16_t count)
{
    auto& ccuResMgr = CcuResourceManager::GetInstance();
    auto msValuePtr0 = reinterpret_cast<T*>(ccuResMgr.GetMsAddr(rankId, dieId, msId[0]));
    for (uint16_t i = 1; i < count + 2; i++) {
        auto msValuePtr = reinterpret_cast<T*>(ccuResMgr.GetMsAddr(rankId, dieId, msId[i]));
        for (uint16_t j = 0; j < HcclSim::BYTE_NUM_4K / sizeof(T); j++) {
            T ms0Value = *(msValuePtr0 + j);
            T msxValue = *(msValuePtr + j);
            *(msValuePtr0 + j) = (ms0Value < msxValue) ? ms0Value : msxValue;
        }
    }
}

// FP32类型
template <typename T>
void ReduceMinFp32(int rankId, int dieId, uint16_t msId[], uint16_t count)
{
    float epsilon = 1e-5;
    auto& ccuResMgr = CcuResourceManager::GetInstance();
    auto msValuePtr0 = reinterpret_cast<T*>(ccuResMgr.GetMsAddr(rankId, dieId, msId[0]));
    for (uint16_t i = 1; i < count + 2; i++) {
        auto msValuePtr = reinterpret_cast<T*>(ccuResMgr.GetMsAddr(rankId, dieId, msId[i]));
        for (uint16_t j = 0; j < HcclSim::BYTE_NUM_4K / sizeof(T); j++) {
            T ms0Value = *(msValuePtr0 + j);
            T msxValue = *(msValuePtr + j);
            *(msValuePtr0 + j) = (fabs(ms0Value - msxValue) <= epsilon) ? ms0Value : msxValue;
        }
    }
}

// FP16类型
template <typename T>
void ReduceMinFp16(int rankId, int dieId, uint16_t msId[], uint16_t count)
{
    float epsilon = 1e-5;
    auto& ccuResMgr = CcuResourceManager::GetInstance();
    auto msValuePtr0 = reinterpret_cast<T*>(ccuResMgr.GetMsAddr(rankId, dieId, msId[0]));
    for (uint16_t i = 1; i < count + 2; i++) {
        auto msValuePtr = reinterpret_cast<T*>(ccuResMgr.GetMsAddr(rankId, dieId, msId[i]));
        for (uint16_t j = 0; j < HcclSim::BYTE_NUM_4K / sizeof(T); j++) {
            T ms0Value = *(msValuePtr0 + j);
            T msxValue = *(msValuePtr + j);
            *(msValuePtr0 + j) = (fabs(ms0Value.to_float() - msxValue.to_float()) <= epsilon) ? ms0Value : msxValue;
        }
    }
}

const std::map<ReduceMaxMinDataType, std::function<void(int, int, uint16_t[], uint16_t)>> reduceMinFuncMap
    = {{ReduceMaxMinDataType::MAX_MIN_INT16,
        [](int rankId, int dieId, uint16_t msId[], uint16_t count) {
            ReduceMin<int16_t>(rankId, dieId, msId, count);
        }},
       {ReduceMaxMinDataType::MAX_MIN_INT32,
        [](int rankId, int dieId, uint16_t msId[], uint16_t count) {
            ReduceMin<int32_t>(rankId, dieId, msId, count);
        }},
       {ReduceMaxMinDataType::MAX_MIN_UINT8,
        [](int rankId, int dieId, uint16_t msId[], uint16_t count) {
            ReduceMin<uint8_t>(rankId, dieId, msId, count);
        }},
       {ReduceMaxMinDataType::MAX_MIN_FP32,
        [](int rankId, int dieId, uint16_t msId[], uint16_t count) {
            ReduceMinFp32<float>(rankId, dieId, msId, count);
        }},
       {ReduceMaxMinDataType::MAX_MIN_INT8,
        [](int rankId, int dieId, uint16_t msId[], uint16_t count) {
            ReduceMin<int8_t>(rankId, dieId, msId, count);
        }},
       {ReduceMaxMinDataType::MAX_MIN_FP16, [](int rankId, int dieId, uint16_t msId[], uint16_t count) {
            ReduceMinFp16<FP16>(rankId, dieId, msId, count);
        }}};

#endif // HCCL_SIM_CCU_REDUCE_OPERATOR_H
