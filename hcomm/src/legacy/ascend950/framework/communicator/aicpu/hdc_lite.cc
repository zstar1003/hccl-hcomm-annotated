/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hdc_lite.h"
#include <chrono>
#include "log.h"

namespace Hccl {
constexpr u32 HCCL_HDC_CONTROL_WORDS = 2;
constexpr u32 HCCL_HDC_HEAD_POS = 2;
constexpr u32 HCCL_HDC_TAIL_POS = 1;

inline u32* HcclHdcGetControlWordAddr(void* base, u64 size, u32 pos)
{
    return reinterpret_cast<u32*>(reinterpret_cast<u8*>((base)) + size - pos * sizeof(pos));
}

HcclResult HDCommunicateLite::Init(const struct HDCommunicateParams& params)
{
    CHK_PRT_RET(
        (params.devMemSize == 0), HCCL_ERROR("[HDCommunicateLite][InitDevice]Invalid devMemSize=%u", params.devMemSize),
        HCCL_E_PARA);
    void* deviceAddr = reinterpret_cast<void*>(params.deviceAddr);
    CHK_PTR_NULL(deviceAddr);
    readCacheAddr = reinterpret_cast<void*>(params.readCacheAddr);
    CHK_PTR_NULL(readCacheAddr);
    devMem = std::make_unique<Buffer>(reinterpret_cast<uintptr_t>(deviceAddr), params.devMemSize);
    buffLen = params.buffLen;
    flag = params.flag;

    headCntAddr
        = HcclHdcGetControlWordAddr(reinterpret_cast<void*>(devMem->GetAddr()), devMem->GetSize(), HCCL_HDC_HEAD_POS);
    tailCntAddr
        = HcclHdcGetControlWordAddr(reinterpret_cast<void*>(devMem->GetAddr()), devMem->GetSize(), HCCL_HDC_TAIL_POS);

    HCCL_INFO(
        "[HDCommunicateLite][Init] buffLen=%u, flag=%u, readCacheAddr=%p, headCntAddr=%p, tailCntAddr=%p", buffLen,
        flag, readCacheAddr, headCntAddr, tailCntAddr);
    return HCCL_SUCCESS;
}

HcclResult HDCommunicateLite::Put(u32 offset, u32 length, u8* value)
{
    if (length == 0) {
        return HCCL_SUCCESS;
    }
    CHK_PTR_NULL(value);
    if ((flag == HCCLV2_HDC_TYPE_H2D)) {
        HCCL_ERROR("[HDCommunicateLite][Put]Invalid usage, flag=%u", flag);
        return HCCL_E_PARA;
    }
    CHK_PRT_RET(
        (static_cast<u64>(offset) + length > buffLen),
        HCCL_ERROR("[HDCommunicateLite][Put]Invalid length, offset=%u, length=%u", offset, length), HCCL_E_PARA);
    std::unique_lock<std::mutex> lock(shmLock);
    return Write(offset, length, value);
}

HcclResult HDCommunicateLite::Get(u32 offset, u32 length, u8* value)
{
    if (length == 0) {
        return HCCL_SUCCESS;
    }
    CHK_PTR_NULL(value);
    CHK_PRT_RET(
        (static_cast<u64>(offset) + length > buffLen),
        HCCL_ERROR("[HDCommunicateLite][Get]Invalid length, offset=%u, length=%u, buffLen=%u", offset, length, buffLen),
        HCCL_E_PARA);
    std::unique_lock<std::mutex> lock(shmLock);
    return Read(offset, length, value);
}

#pragma GCC push_options
#pragma GCC optimize("O0")
HcclResult HDCommunicateLite::Write(u32 offset, u32 length, u8* value)
{
    if (length == 0) {
        return HCCL_SUCCESS;
    }
    CHK_PTR_NULL(value);
    u32 head = *headCntAddr;
    head++;
    *headCntAddr = head;

    auto ret = memcpy_s(
        reinterpret_cast<u8*>(devMem->GetAddr()) + offset,
        devMem->GetSize() - HCCL_HDC_CONTROL_WORDS * sizeof(u32) - offset, value, length);
    CHK_PRT_RET(ret != EOK, HCCL_ERROR("[HDCommunicateLite][Write]memcpy_s failed, return[%d].", ret), HCCL_E_INTERNAL);

    u32 tail = *tailCntAddr;
    tail++;
    *tailCntAddr = tail;

    return HCCL_SUCCESS;
}

HcclResult HDCommunicateLite::Read(u32 offset, u32 length, u8* value)
{
    if (length == 0) {
        return HCCL_SUCCESS;
    }
    CHK_PTR_NULL(value);
    u32* cachedTailCntAddr = HcclHdcGetControlWordAddr(readCacheAddr, devMem->GetSize(), HCCL_HDC_TAIL_POS);
    volatile u32 cachedTailCnt = *cachedTailCntAddr;
    volatile u32 tailCnt = 0;
    tailCnt = *tailCntAddr;

    if (cachedTailCnt != tailCnt) {
        // 默认HDC超时时间为10s
        CHK_RET(UpdateCache(10));
    }
    auto ret = memcpy_s(value, length, static_cast<u8*>(readCacheAddr) + offset, length);
    CHK_PRT_RET(ret != EOK, HCCL_ERROR("[HDCommunicateLite][Read]memcpy_s failed, return[%d].", ret), HCCL_E_INTERNAL);
    return HCCL_SUCCESS;
}

HcclResult HDCommunicateLite::UpdateCache(u32 timeoutSec)
{
    void* srcBaseAddr = reinterpret_cast<void*>(devMem->GetAddr());
    u32* srcHeadCntAddr = HcclHdcGetControlWordAddr(srcBaseAddr, devMem->GetSize(), HCCL_HDC_HEAD_POS);
    u32* srcTailCntAddr = HcclHdcGetControlWordAddr(srcBaseAddr, devMem->GetSize(), HCCL_HDC_TAIL_POS);
    u32* cachedHeadCntAddr = HcclHdcGetControlWordAddr(readCacheAddr, devMem->GetSize(), HCCL_HDC_HEAD_POS);
    u32* cachedTailCntAddr = HcclHdcGetControlWordAddr(readCacheAddr, devMem->GetSize(), HCCL_HDC_TAIL_POS);

    s32 ret = 0;
    auto startTime = std::chrono::steady_clock::now();
    auto timeout = std::chrono::seconds(timeoutSec);
    while (1) {
        // step1: cache尾计数
        ret = memcpy_s(cachedTailCntAddr, sizeof(u32), srcTailCntAddr, sizeof(u32));
        CHK_PRT_RET(
            ret != EOK, HCCL_ERROR("[HDCommunicateLite][UpdateCache]memcpy_s failed, return[%d].", ret),
            HCCL_E_INTERNAL);

        // step2: cache数据
        ret = memcpy_s(
            readCacheAddr, devMem->GetSize() - HCCL_HDC_CONTROL_WORDS * sizeof(u32), srcBaseAddr,
            devMem->GetSize() - HCCL_HDC_CONTROL_WORDS * sizeof(u32));
        CHK_PRT_RET(
            ret != EOK, HCCL_ERROR("[HDCommunicateLite][UpdateCache]memcpy_s failed, return[%d].", ret),
            HCCL_E_INTERNAL);

        // step3：cache头计数
        ret = memcpy_s(cachedHeadCntAddr, sizeof(u32), srcHeadCntAddr, sizeof(u32));
        CHK_PRT_RET(
            ret != EOK, HCCL_ERROR("[HDCommunicateLite][UpdateCache]memcpy_s failed, return[%d].", ret),
            HCCL_E_INTERNAL);

        volatile u32 cachedHeadCnt = *cachedHeadCntAddr;
        volatile u32 cachedTailCnt = *cachedTailCntAddr;

        if (cachedHeadCnt == cachedTailCnt) {
            break;
        }
        CHK_PRT_RET(
            ((std::chrono::steady_clock::now() - startTime) >= timeout),
            HCCL_WARNING("[HDCommunicateLite][UpdateCache]get remote data timeout[%u s].", timeoutSec), HCCL_E_AGAIN);
    }
    return HCCL_SUCCESS;
}
#pragma GCC pop_options

} // namespace Hccl
