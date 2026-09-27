/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "acl/acl_rt.h"
#include "rt_external.h"
#include "securec.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum tagRtClearStep {
    RT_STREAM_STOP = 0,
    RT_STREAM_CLEAR,
} rtClearStep_t;

RTS_API rtError_t rtStreamClear(rtStream_t stm, rtClearStep_t step);

RTS_API rtError_t rtGetServerIDBySDID(uint32_t sdid, uint32_t* srvId);

#ifdef __cplusplus
}
#endif

rtError_t rtStreamClear(rtStream_t stm, rtClearStep_t step) { return 0; }

rtError_t rtGetServerIDBySDID(uint32_t sdid, uint32_t* srvId)
{
    *srvId = 0;
    return 0;
}

rtError_t rtMemcpyEx(void* dst, uint64_t destMax, const void* src, uint64_t cnt, rtMemcpyKind_t kind)
{
    if (dst == nullptr || src == nullptr || cnt == 0 || cnt > destMax) {
        return RT_ERROR_NONE;
    }
    (void)memcpy_s(dst, destMax, src, cnt);
    return RT_ERROR_NONE;
}

RTS_API rtError_t rtUbDevQueryInfo(rtUbDevQueryCmd cmd, void* devInfo) { return 0; }

rtError_t rtCCULaunch(rtCcuTaskInfo_t* taskInfo, rtStream_t const stm) { return 0; }

rtError_t rtReleaseDevResAddress(rtDevResInfo* const resInfo) { return RT_ERROR_NONE; }

rtError_t rtGetDevResAddress(rtDevResInfo* const resInfo, rtDevResAddrInfo* const addrInfo)
{
    if (resInfo == nullptr || addrInfo == nullptr || addrInfo->resAddress == nullptr || addrInfo->len == nullptr) {
        return static_cast<rtError_t>(0xFFFFFFFF);
    }
    // 合成一个非0映射地址，按resId偏移，便于UT校验
    constexpr uint64_t stubResBaseAddr = 0x100000000ULL;
    constexpr uint64_t stubResPerSize = 0x1000ULL;
    *addrInfo->resAddress = stubResBaseAddr + static_cast<uint64_t>(resInfo->resId) * stubResPerSize;
    *addrInfo->len = static_cast<uint32_t>(stubResPerSize);
    return RT_ERROR_NONE;
}
