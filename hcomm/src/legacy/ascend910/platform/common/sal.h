/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCOMM_HCCL_SRC_SAL_H
#define HCOMM_HCCL_SRC_SAL_H

#include <nlohmann/json.hpp>
#include "sal_pub.h"
#include "log.h"

#if HCOMM_T_DESC("日志处理适配", true)

constexpr u32 TIME_FROM_1900 = 1900;

/* 日志信息中时间戳长度 */
constexpr s32 LOG_TIME_STAMP_SIZE = 27;

#endif

#if HCOMM_T_DESC("json处理函数", true)
HcclResult SalParseInformation(nlohmann::json& parseInformation, const std::string& information);
HcclResult SalGetJsonProperty(const nlohmann::json& obj, const std::string& propName, std::string& propValue);
#endif

u32 SalStrLen(const char* s, u32 maxLen = INT_MAX);
HcclResult SalStrToDouble(const std::string str, double& val);
template <typename MapType>
void ExceptionTransportInfoRetriever(const u32 localRank, const MapType& rankTransportMap)
{
    for (const auto& it : rankTransportMap) {
        if (it.second == nullptr) {
            continue;
        }
        std::string linkTag;
        (it.second)->GetLinkTag(linkTag);
        HCCL_ERROR(
            "[ExceptionTransportInfo]localRank[%u], remoteRank[%u], linkTag[%s], linkState[%d]", localRank, it.first,
            linkTag.c_str(), (it.second)->GetState());
    }
}

inline s32 FloatHighSpeedMove(float* dest, float* src, size_t len)
{
    if (UNLIKELY(dest == nullptr || src == nullptr)) {
        HCCL_ERROR("[FloatHighSpeedMove]dest[%p], src[%p]", dest, src);
        return EINVAL;
    }
    s32 ret = 0;
    if (len * sizeof(float) > MEMCPY_THRESHOLD) {
        ret = memmove_s(dest, len * sizeof(float), src, len * sizeof(float));
    } else {
        while ((len--) != 0) {
            *(dest++) = *(src++);
        }
    }
    return ret;
}

inline s32 S64HighSpeedMove(s64* dest, s64* src, size_t len)
{
    if (UNLIKELY(dest == nullptr || src == nullptr)) {
        HCCL_ERROR("[S64HighSpeedMove]dest[%p], src[%p]", dest, src);
        return EINVAL;
    }
    s32 ret = 0;
    if (len * sizeof(s64) > MEMCPY_THRESHOLD) {
        ret = memmove_s(dest, len * sizeof(s64), src, len * sizeof(s64));
    } else {
        while ((len--) != 0) {
            *(dest++) = *(src++);
        }
    }
    return ret;
}

// src和dest在调用处保证非空
inline void FloatHighValueSum(float* src, float* dest, size_t len)
{
    while ((len--) != 0) {
        *dest++ += *src++;
    }
    return;
}

inline s32 FloatMemClear(float* dst, size_t count)
{
    if (UNLIKELY(dst == nullptr)) {
        HCCL_ERROR("[FloatMemClear]dst ptr is nullptr");
        return EINVAL;
    }
    s32 ret = 0;
    if (count * sizeof(float) > MEMCPY_THRESHOLD) {
        ret = memset_s(dst, count * sizeof(float), 0, count * sizeof(float));
    } else {
        while (count--) {
            *dst++ = 0.0;
        }
    }
    return ret;
}

#endif // HCOMM_HCCL_SRC_SAL_H
