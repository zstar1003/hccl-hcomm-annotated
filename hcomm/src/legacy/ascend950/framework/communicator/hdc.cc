/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hdc.h"
#include <chrono>
#include "log.h"
#include "ascend_hal.h"
#include "orion_adapter_rts.h"

namespace Hccl {

HDCommunicate::HDCommunicate(u32 deviceLogicId, u32 flag, u32 buffLen)
    : deviceLogicId(deviceLogicId),
      flag(flag),
      buffLen(buffLen)
{}

HDCommunicate::~HDCommunicate()
{
    HCCL_INFO("[~HDCommunicate]start hdc destroy");
    if ((devMem->GetAddr() != 0) && supportDevMemReg) {
        (void)halHostUnregister(reinterpret_cast<void*>(devMem->GetAddr()), deviceLogicId);
    }
}

// 在device中申请共享内存，其数据格式如下所示:
//     +---------------------+
//     |                     |
//     |      content        |
//     |                     |
//     +---------------------+
//     |    head_cnt[u32]    |
//     +---------------------+
//     |    tail_cnt[u32]    |
//     +---------------------+
// 发送方更新content前，需要将head_cnt加1,更新数据后需要将tail_cnt加1;
// 接收方在读取数据前判断共享内存中的tail_cnt和本地cache中的tailcnt是否一致，如不一致则需要更新本地cache;
//      更新本地cache时，需要确保cache中head_cnt和tail_cnt一致，否则舍弃本次数据，继续更新cache直至一致;

constexpr u32 HCCL_SHM_ALIGN = 4096;
constexpr u32 HCCL_HDC_CONTROL_WORDS = 2;
constexpr u32 HCCL_HDC_HEAD_POS = 2;
constexpr u32 HCCL_HDC_TAIL_POS = 1;

inline u32* HcclHdcGetControlWordAddr(void* base, u64 size, u32 pos)
{
    return reinterpret_cast<u32*>(reinterpret_cast<u8*>((base)) + size - pos * sizeof(pos));
}

HcclResult HDCommunicate::Init()
{
    CHK_RET(VerifyDeviceMemoryRegisterSupport());

    CHK_RET(AllocShm());
    CHK_RET(AllocReadCache());

    headCntAddr
        = HcclHdcGetControlWordAddr(reinterpret_cast<void*>(hostMem->GetAddr()), hostMem->GetSize(), HCCL_HDC_HEAD_POS);
    tailCntAddr
        = HcclHdcGetControlWordAddr(reinterpret_cast<void*>(hostMem->GetAddr()), hostMem->GetSize(), HCCL_HDC_TAIL_POS);

    devHeadCntAddr
        = HcclHdcGetControlWordAddr(reinterpret_cast<void*>(devMem->GetAddr()), devMem->GetSize(), HCCL_HDC_HEAD_POS);
    devTailCntAddr
        = HcclHdcGetControlWordAddr(reinterpret_cast<void*>(devMem->GetAddr()), devMem->GetSize(), HCCL_HDC_TAIL_POS);
    HCCL_INFO(
        "[HDCommunicate][Init] buffLen=%u, flag=%u, readCacheAddr=%p, devHeadCntAddr=%p, devTailCntAddr=%p", buffLen,
        flag, readCacheAddr, devHeadCntAddr, devTailCntAddr);
    return HCCL_SUCCESS;
}

HcclResult HDCommunicate::VerifyDeviceMemoryRegisterSupport()
{
    supportDevMemReg = false;
    size_t outputLen = 0;
    struct supportFeaturePara input = {};
    struct supportFeaturePara output = {};
    s32 userDevId = HrtGetDevice();
    s32 logicDevId = 0;
    // 调用驱动接口前需将userDevId转换为logicDevId
    CHK_RET(HrtGetLogicDevIdByUserDevId(userDevId, logicDevId));
    deviceLogicId = static_cast<u32>(logicDevId);
    input.support_feature = CTRL_SUPPORT_PCIE_BAR_MEM_MASK;
    input.devid = static_cast<unsigned int>(logicDevId);
    halMemCtl(CTRL_TYPE_SUPPORT_FEATURE, &input, sizeof(struct supportFeaturePara), &output, &outputLen);

    if ((output.support_feature & CTRL_SUPPORT_PCIE_BAR_MEM_MASK) != 0) {
        supportDevMemReg = true;
    }
    HCCL_INFO("[HDCommunicate]supportDevMemReg[%d]", supportDevMemReg);
    return HCCL_SUCCESS;
}

struct HDCommunicateParams HDCommunicate::GetCommunicateParams() const
{
    struct HDCommunicateParams params;
    params.hostAddr = reinterpret_cast<u64>(reinterpret_cast<void*>(hostMem->GetAddr()));
    params.deviceAddr = reinterpret_cast<u64>(reinterpret_cast<void*>(devMem->GetAddr()));
    params.readCacheAddr = reinterpret_cast<u64>(readCacheAddr);
    params.devMemSize = devMem->GetSize();
    params.buffLen = buffLen;
    params.flag = flag;
    return params;
}
// 为了按照调用顺序执行，防止编译器优化导致产生异常行为
#pragma GCC push_options
#pragma GCC optimize("O0")
HcclResult HDCommunicate::Put(u32 offset, u32 length, u8* value)
{
    if (length == 0) {
        return HCCL_SUCCESS;
    }
    CHK_PTR_NULL(value);

    if (flag == HCCLV2_HDC_TYPE_D2H) {
        HCCL_ERROR("[HDCommunicate][Put]Invalid usage, flag=%u", flag);
        return HCCL_E_PARA;
    }
    CHK_PRT_RET(
        (offset + length > buffLen),
        HCCL_ERROR("[HDCommunicate][Put]Invalid length, offset=%u, length=%u", offset, length), HCCL_E_PARA);
    std::lock_guard<std::mutex> lock(shmLock);
    return Write(offset, length, value);
}

HcclResult HDCommunicate::Get(u32 offset, u32 length, u8* value)
{
    if (length == 0) {
        return HCCL_SUCCESS;
    }
    CHK_PTR_NULL(value);
    CHK_PRT_RET(
        (offset + length > buffLen),
        HCCL_ERROR("[HDCommunicate][Get]Invalid length, offset=%u, length=%u, buffLen=%u", offset, length, buffLen),
        HCCL_E_PARA);
    std::lock_guard<std::mutex> lock(shmLock);
    return Read(offset, length, value);
}

HcclResult HrtDrvMemCpy(void* dst, uint64_t destMax, const void* src, uint64_t count)
{
    // 参数有效性检查
    CHK_PTR_NULL(dst);
    CHK_PTR_NULL(src);

    uint64_t dstAddr = reinterpret_cast<uintptr_t>(dst);
    uint64_t srcAddr = reinterpret_cast<uintptr_t>(const_cast<void*>(src));
    drvError_t ret = drvMemcpy(dstAddr, destMax, srcAddr, count);
    CHK_PRT_RET(
        ret != DRV_ERROR_NONE,
        HCCL_ERROR(
            "errNo[0x%016llx] hrtDrvMemCpy fail,"
            "return[%d].",
            HCCL_ERROR_CODE(HCCL_E_DRV), ret),
        HCCL_E_DRV);

    return HCCL_SUCCESS;
}

HcclResult HDCommunicate::Write(u32 offset, u32 length, u8* value)
{
    if (length == 0) {
        return HCCL_SUCCESS;
    }
    CHK_PTR_NULL(value);

    u32 head = *headCntAddr;
    head++;
    *headCntAddr = head;
    if (!supportDevMemReg) {
        CHK_RET(HrtDrvMemCpy(devHeadCntAddr, sizeof(u32), headCntAddr, sizeof(u32)));
    }

    auto ret = memcpy_s(
        reinterpret_cast<u8*>(hostMem->GetAddr()) + offset, hostMem->GetSize() - HCCL_HDC_CONTROL_WORDS * sizeof(u32),
        value, length);
    CHK_PRT_RET(ret != EOK, HCCL_ERROR("[HDCommunicate][Write]memcpy_s failed, return[%d].", ret), HCCL_E_INTERNAL);

    if (!supportDevMemReg) {
        CHK_RET(HrtDrvMemCpy(
            reinterpret_cast<u8*>(devMem->GetAddr()) + offset,
            hostMem->GetSize() - HCCL_HDC_CONTROL_WORDS * sizeof(u32), value, length));
    }

    u32 tail = *tailCntAddr;
    tail++;
    *tailCntAddr = tail;
    if (!supportDevMemReg) {
        CHK_RET(HrtDrvMemCpy(devTailCntAddr, sizeof(u32), tailCntAddr, sizeof(u32)));
    }
    return HCCL_SUCCESS;
}

HcclResult HDCommunicate::Read(u32 offset, u32 length, u8* value)
{
    if (length == 0) {
        return HCCL_SUCCESS;
    }
    CHK_PTR_NULL(value);
    u32* cachedTailCntAddr = HcclHdcGetControlWordAddr(readCacheAddr, devMem->GetSize(), HCCL_HDC_TAIL_POS);
    volatile u32 cachedTailCnt = *cachedTailCntAddr;
    volatile u32 tailCnt = 0;
    if (!supportDevMemReg) {
        u32 tempTailCnt = 0;
        u32* devSrcTailCntAddr = HcclHdcGetControlWordAddr(
            reinterpret_cast<void*>(devMem->GetAddr()), devMem->GetSize(), HCCL_HDC_TAIL_POS);
        CHK_RET(HrtDrvMemCpy(&tempTailCnt, sizeof(u32), devSrcTailCntAddr, sizeof(u32)));
        tailCnt = tempTailCnt;
    } else {
        tailCnt = *tailCntAddr;
    }
    if (cachedTailCnt != tailCnt) {
        // 默认HDC超时时间为10s
        CHK_RET(UpdateCache(10));
    }
    auto ret = memcpy_s(value, length, static_cast<u8*>(readCacheAddr) + offset, length);
    CHK_PRT_RET(ret != EOK, HCCL_ERROR("[HDCommunicate][Read]memcpy_s failed, return[%d].", ret), HCCL_E_INTERNAL);
    return HCCL_SUCCESS;
}

HcclResult HDCommunicate::UpdateCache(u32 timeoutSec)
{
    void* srcBaseAddr = reinterpret_cast<void*>(hostMem->GetAddr());
    u32* srcHeadCntAddr = HcclHdcGetControlWordAddr(srcBaseAddr, devMem->GetSize(), HCCL_HDC_HEAD_POS);
    u32* srcTailCntAddr = HcclHdcGetControlWordAddr(srcBaseAddr, devMem->GetSize(), HCCL_HDC_TAIL_POS);
    u32* devSrcHeadCntAddr
        = HcclHdcGetControlWordAddr(reinterpret_cast<void*>(devMem->GetAddr()), devMem->GetSize(), HCCL_HDC_HEAD_POS);
    u32* devSrcTailCntAddr
        = HcclHdcGetControlWordAddr(reinterpret_cast<void*>(devMem->GetAddr()), devMem->GetSize(), HCCL_HDC_TAIL_POS);
    u32* cachedHeadCntAddr = HcclHdcGetControlWordAddr(readCacheAddr, devMem->GetSize(), HCCL_HDC_HEAD_POS);
    u32* cachedTailCntAddr = HcclHdcGetControlWordAddr(readCacheAddr, devMem->GetSize(), HCCL_HDC_TAIL_POS);

    s32 ret = 0;
    auto startTime = std::chrono::steady_clock::now();
    auto timeout = std::chrono::seconds(timeoutSec);
    while (1) {
        if (!supportDevMemReg) {
            // step1: cache尾计数
            CHK_RET(HrtDrvMemCpy(cachedTailCntAddr, sizeof(u32), devSrcTailCntAddr, sizeof(u32)));

            // step2: cache数据
            CHK_RET(HrtDrvMemCpy(
                readCacheAddr, devMem->GetSize() - HCCL_HDC_CONTROL_WORDS * sizeof(u32),
                reinterpret_cast<void*>(devMem->GetAddr()), devMem->GetSize() - HCCL_HDC_CONTROL_WORDS * sizeof(u32)));

            // step3：cache头计数
            CHK_RET(HrtDrvMemCpy(cachedHeadCntAddr, sizeof(u32), devSrcHeadCntAddr, sizeof(u32)));
        } else {
            // step1: cache尾计数
            ret = memcpy_s(cachedTailCntAddr, sizeof(u32), srcTailCntAddr, sizeof(u32));
            CHK_PRT_RET(
                ret != EOK, HCCL_ERROR("[HDCommunicate][UpdateCache]memcpy_s failed, return[%d].", ret),
                HCCL_E_INTERNAL);

            // step2: cache数据
            ret = memcpy_s(
                readCacheAddr, devMem->GetSize() - HCCL_HDC_CONTROL_WORDS * sizeof(u32), srcBaseAddr,
                devMem->GetSize() - HCCL_HDC_CONTROL_WORDS * sizeof(u32));
            CHK_PRT_RET(
                ret != EOK, HCCL_ERROR("[HDCommunicate][UpdateCache]memcpy_s failed, return[%d].", ret),
                HCCL_E_INTERNAL);

            // step3：cache头计数
            ret = memcpy_s(cachedHeadCntAddr, sizeof(u32), srcHeadCntAddr, sizeof(u32));
            CHK_PRT_RET(
                ret != EOK, HCCL_ERROR("[HDCommunicate][UpdateCache]memcpy_s failed, return[%d].", ret),
                HCCL_E_INTERNAL);
        }

        volatile u32 cachedHeadCnt = *cachedHeadCntAddr;
        volatile u32 cachedTailCnt = *cachedTailCntAddr;

        if (cachedHeadCnt == cachedTailCnt) {
            break;
        }
        CHK_PRT_RET(
            ((std::chrono::steady_clock::now() - startTime) >= timeout),
            HCCL_WARNING("[HDCommunicate][UpdateCache]get remote data timeout[%u s].", timeoutSec), HCCL_E_AGAIN);
    }
    return HCCL_SUCCESS;
}
#pragma GCC pop_options

HcclResult HDCommunicate::AllocShm()
{
    // 共享内存size需要按照4K(4*1024=4096)对齐
    size_t size
        = (buffLen + HCCL_HDC_CONTROL_WORDS * sizeof(u32) + HCCL_SHM_ALIGN - 1) / HCCL_SHM_ALIGN * HCCL_SHM_ALIGN;
    devMem = std::make_unique<DevBuffer>(size);
    HrtMemset(reinterpret_cast<void*>(devMem->GetAddr()), devMem->GetSize(), devMem->GetSize());

    if (supportDevMemReg) {
        void* hostAddr = nullptr;
        halHostRegister(
            reinterpret_cast<void*>(devMem->GetAddr()), devMem->GetSize(), DEV_SVM_MAP_HOST, deviceLogicId, &hostAddr);

        hostMem = std::make_unique<HostBuffer>(reinterpret_cast<uintptr_t>(hostAddr), devMem->GetSize());
    } else {
        hostMem = std::make_unique<HostBuffer>(devMem->GetSize());
    }

    auto ret = memset_s(reinterpret_cast<void*>(hostMem->GetAddr()), hostMem->GetSize(), 0, hostMem->GetSize());
    CHK_PRT_RET(ret != EOK, HCCL_ERROR("[HDCommunicate][AllocShm]memset_s failed, return[%d].", ret), HCCL_E_INTERNAL);

    return HCCL_SUCCESS;
}

HcclResult HDCommunicate::AllocReadCache()
{
    if (flag == HCCLV2_HDC_TYPE_D2H) {
        hostCache = std::make_unique<HostBuffer>(hostMem->GetSize());
        auto ret
            = memset_s(reinterpret_cast<void*>(hostCache->GetAddr()), hostCache->GetSize(), 0, hostCache->GetSize());
        CHK_PRT_RET(
            ret != EOK, HCCL_ERROR("[HDCommunicate][AllocReadCache]memset_s failed, return[%d].", ret),
            HCCL_E_INTERNAL);
        readCacheAddr = reinterpret_cast<void*>(hostCache->GetAddr());
    } else {
        devCache = std::make_unique<DevBuffer>(devMem->GetSize());
        HrtMemset(reinterpret_cast<void*>(devCache->GetAddr()), devCache->GetSize(), devCache->GetSize());
        readCacheAddr = reinterpret_cast<void*>(devCache->GetAddr());
    }
    return HCCL_SUCCESS;
}
} // namespace Hccl
