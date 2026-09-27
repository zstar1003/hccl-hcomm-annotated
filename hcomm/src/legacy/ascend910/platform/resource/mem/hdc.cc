/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hdc_pub.h"
#include "log.h"
#include "ascend_hal.h"
#include "adapter_hal.h"
#include "adapter_rts.h"
#include <atomic>
#include <chrono>

namespace hccl {
HDCommunicate::HDCommunicate(u32 deviceLogicId, u32 flag, u32 buffLen)
    : deviceLogicId_(deviceLogicId),
      flag_(flag),
      buffLen_(buffLen)
{}
HDCommunicate::HDCommunicate() : deviceLogicId_(INVALID_UINT), flag_(0), buffLen_(0) {}

HDCommunicate::~HDCommunicate()
{
    if (isHost_ && devMem_.ptr() && supportDevMemReg_) {
        (void)hrtHalHostUnregister(devMem_.ptr(), deviceLogicId_);
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

#define HCCL_SHM_ALIGN 4096
#define HCCL_HDC_CONTROL_WORDS 2
#define HCCL_HDC_HEAD_POS 2
#define HCCL_HDC_TAIL_POS 1

inline u32* HcclHdcGetControlWordAddr(void* base, u64 size, u32 pos)
{
    return reinterpret_cast<u32*>(reinterpret_cast<u8*>((base)) + size - pos * sizeof(pos));
}

HcclResult HDCommunicate::InitHost()
{
    CHK_RET(VerifyDeviceMemoryRegisterSupport());

    CHK_RET(AllocShm(deviceLogicId_, devMem_, hostMem_));
    CHK_RET(AllocReadCache(flag_, readCacheAddr_));

    headCntAddr_ = HcclHdcGetControlWordAddr(hostMem_.ptr(), hostMem_.size(), HCCL_HDC_HEAD_POS);
    tailCntAddr_ = HcclHdcGetControlWordAddr(hostMem_.ptr(), hostMem_.size(), HCCL_HDC_TAIL_POS);

    devHeadCntAddr_ = HcclHdcGetControlWordAddr(devMem_.ptr(), devMem_.size(), HCCL_HDC_HEAD_POS);
    devTailCntAddr_ = HcclHdcGetControlWordAddr(devMem_.ptr(), devMem_.size(), HCCL_HDC_TAIL_POS);
    return HCCL_SUCCESS;
}

HcclResult HDCommunicate::VerifyDeviceMemoryRegisterSupport()
{
    supportDevMemReg_ = false;
    size_t outputLen = 0;
    struct supportFeaturePara input = {};
    struct supportFeaturePara output = {};
    s32 deviceId = 0;
    CHK_RET(hrtGetDevice(&deviceId));
    s32 logicDevId = 0;
    // 调用驱动接口前需将userDevId转换为logicDevId
    CHK_RET(hrtGetLogicDevIdByUserDevId(deviceId, logicDevId));
    deviceLogicId_ = static_cast<u32>(logicDevId);
    input.support_feature = CTRL_SUPPORT_PCIE_BAR_MEM_MASK;
    input.devid = static_cast<unsigned int>(logicDevId);
    CHK_RET(hrtHalMemCtl(CTRL_TYPE_SUPPORT_FEATURE, &input, sizeof(struct supportFeaturePara), &output, &outputLen));

    if ((output.support_feature & CTRL_SUPPORT_PCIE_BAR_MEM_MASK) != 0) {
        supportDevMemReg_ = true;
    }
    HCCL_INFO("[HDCommunicate]supportDevMemReg_ %d deviceId %d", supportDevMemReg_, input.devid);
    return HCCL_SUCCESS;
}

struct HDCommunicateParams HDCommunicate::GetCommunicateParams()
{
    struct HDCommunicateParams params;
    params.hostAddr = reinterpret_cast<u64>(hostMem_.ptr());
    params.deviceAddr = reinterpret_cast<u64>(devMem_.ptr());
    params.readCacheAddr = reinterpret_cast<u64>(readCacheAddr_);
    params.devMemSize = devMem_.size();
    params.buffLen = buffLen_;
    params.flag = flag_;
    HCCL_DEBUG(
        "[HDCommunicate][GetCommunicateParams] hostAddr %p deviceAddr %p readCacheAddr %p devMemSize %u "
        "buffLen %u flag %u",
        params.hostAddr, params.deviceAddr, params.readCacheAddr, devMem_.size(), buffLen_, flag_);
    return params;
}

HcclResult HDCommunicate::InitDevice(const struct HDCommunicateParams& params)
{
    CHK_PRT_RET(
        (params.devMemSize == 0), HCCL_ERROR("[HDCommunicate][InitDevice]Invalid devMemSize=%u", params.devMemSize),
        HCCL_E_PARA);
    void* deviceAddr = reinterpret_cast<void*>(params.deviceAddr);
    CHK_PTR_NULL(deviceAddr);
    readCacheAddr_ = reinterpret_cast<void*>(params.readCacheAddr);
    CHK_PTR_NULL(readCacheAddr_);
    devMem_ = DeviceMem::create(deviceAddr, params.devMemSize);
    buffLen_ = params.buffLen;
    flag_ = params.flag;

    headCntAddr_ = HcclHdcGetControlWordAddr(devMem_.ptr(), devMem_.size(), HCCL_HDC_HEAD_POS);
    tailCntAddr_ = HcclHdcGetControlWordAddr(devMem_.ptr(), devMem_.size(), HCCL_HDC_TAIL_POS);
    isHost_ = false;

    HCCL_DEBUG(
        "[debug HDCommunicate][InitDevice] buffLen_=%u, flag_=%u, readCacheAddr_=%p, headCntAddr_=%p, "
        "tailCntAddr_=%p, deviceAddr %p",
        buffLen_, flag_, readCacheAddr_, headCntAddr_, tailCntAddr_, devMem_.ptr());
    return HCCL_SUCCESS;
}

HcclResult HDCommunicate::Put(u32 offset, u32 length, u8* value)
{
    if (length == 0) {
        return HCCL_SUCCESS;
    }
    CHK_PTR_NULL(value);
    if (((flag_ == HCCL_HDC_TYPE_D2H) && isHost_) || ((flag_ == HCCL_HDC_TYPE_H2D) && !isHost_)) {
        HCCL_ERROR("[HDCommunicate][Put]Invalid usage, flag=%u, isHost=%d", flag_, isHost_);
        return HCCL_E_PARA;
    }
    CHK_PRT_RET(
        (offset + length > buffLen_),
        HCCL_ERROR("[HDCommunicate][Put]Invalid length, offset=%u, length=%u", offset, length), HCCL_E_PARA);
    std::unique_lock<std::shared_mutex> lock(lock_);
    HcclResult ret = Write(offset, length, value);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[HDCommunicate][Put]Write failed, offset=%u, length=%u", offset, length);
        return ret;
    }
    return HCCL_SUCCESS;
}

HcclResult HDCommunicate::Get(u32 offset, u32 length, u8* value)
{
    if (length == 0) {
        return HCCL_SUCCESS;
    }
    CHK_PTR_NULL(value);
    CHK_PRT_RET(
        (offset + length > buffLen_),
        HCCL_ERROR("[HDCommunicate][Get]Invalid length, offset=%u, length=%u, bufferLen=%u", offset, length, buffLen_),
        HCCL_E_PARA);
    std::shared_lock<std::shared_mutex> lock(lock_);
    HcclResult ret = Read(offset, length, value);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[HDCommunicate][Get]Read failed, offset=%u, length=%u", offset, length);
        return ret;
    }
    return HCCL_SUCCESS;
}

#pragma GCC push_options
#pragma GCC optimize("O0")
HcclResult HDCommunicate::Write(u32 offset, u32 length, u8* value)
{
    if (length == 0) {
        return HCCL_SUCCESS;
    }
    CHK_PTR_NULL(value);
    u32 head = *headCntAddr_;
    head++;
    *headCntAddr_ = head;
    if (isHost_ && !supportDevMemReg_) {
        CHK_RET(hrtDrvMemCpy(devHeadCntAddr_, sizeof(u32), headCntAddr_, sizeof(u32)));
    }

    if (isHost_) {
        auto ret = memcpy_s(
            reinterpret_cast<u8*>(hostMem_.ptr()) + offset, hostMem_.size() - HCCL_HDC_CONTROL_WORDS * sizeof(u32),
            value, length);
        CHK_PRT_RET(ret != EOK, HCCL_ERROR("[HDCommunicate][Write]memcpy_s failed, return[%d].", ret), HCCL_E_INTERNAL);
    } else {
        auto ret = memcpy_s(
            reinterpret_cast<u8*>(devMem_.ptr()) + offset, devMem_.size() - HCCL_HDC_CONTROL_WORDS * sizeof(u32), value,
            length);
        CHK_PRT_RET(ret != EOK, HCCL_ERROR("[HDCommunicate][Write]memcpy_s failed, return[%d].", ret), HCCL_E_INTERNAL);
    }
    if (isHost_ && !supportDevMemReg_) {
        CHK_RET(hrtDrvMemCpy(
            reinterpret_cast<u8*>(devMem_.ptr()) + offset, hostMem_.size() - HCCL_HDC_CONTROL_WORDS * sizeof(u32),
            value, length));
    }

    std::atomic_thread_fence(std::memory_order_seq_cst);

    u32 tail = *tailCntAddr_;
    tail++;
    *tailCntAddr_ = tail;
    if (isHost_ && !supportDevMemReg_) {
        CHK_RET(hrtDrvMemCpy(devTailCntAddr_, sizeof(u32), tailCntAddr_, sizeof(u32)));
    }
    return HCCL_SUCCESS;
}

HcclResult HDCommunicate::Read(u32 offset, u32 length, u8* value)
{
    if (length == 0) {
        return HCCL_SUCCESS;
    }
    CHK_PTR_NULL(value);
    u32* cachedTailCntAddr = HcclHdcGetControlWordAddr(readCacheAddr_, devMem_.size(), HCCL_HDC_TAIL_POS);
    volatile u32 cachedTailCnt = *cachedTailCntAddr;
    volatile u32 tailCnt = 0;
    if (isHost_ && !supportDevMemReg_) {
        u32 tempTailCnt = 0;
        u32* devSrcTailCntAddr = HcclHdcGetControlWordAddr(devMem_.ptr(), devMem_.size(), HCCL_HDC_TAIL_POS);
        CHK_RET(hrtDrvMemCpy(&tempTailCnt, sizeof(u32), devSrcTailCntAddr, sizeof(u32)));
        tailCnt = tempTailCnt;
    } else {
        tailCnt = *tailCntAddr_;
    }
    if (cachedTailCnt != tailCnt) {
        // 默认HDC超时时间为10s
        CHK_RET(UpdateCache(10));
    }
    auto ret = memcpy_s(value, length, static_cast<u8*>(readCacheAddr_) + offset, length);
    CHK_PRT_RET(ret != EOK, HCCL_ERROR("[HDCommunicate][Read]memcpy_s failed, return[%d].", ret), HCCL_E_INTERNAL);
    return HCCL_SUCCESS;
}

HcclResult HDCommunicate::UpdateCache(u32 timeoutSec)
{
    void* srcBaseAddr = isHost_ ? hostMem_.ptr() : devMem_.ptr();
    u32* srcHeadCntAddr = HcclHdcGetControlWordAddr(srcBaseAddr, devMem_.size(), HCCL_HDC_HEAD_POS);
    u32* srcTailCntAddr = HcclHdcGetControlWordAddr(srcBaseAddr, devMem_.size(), HCCL_HDC_TAIL_POS);
    u32* devSrcHeadCntAddr = HcclHdcGetControlWordAddr(devMem_.ptr(), devMem_.size(), HCCL_HDC_HEAD_POS);
    u32* devSrcTailCntAddr = HcclHdcGetControlWordAddr(devMem_.ptr(), devMem_.size(), HCCL_HDC_TAIL_POS);
    u32* cachedHeadCntAddr = HcclHdcGetControlWordAddr(readCacheAddr_, devMem_.size(), HCCL_HDC_HEAD_POS);
    u32* cachedTailCntAddr = HcclHdcGetControlWordAddr(readCacheAddr_, devMem_.size(), HCCL_HDC_TAIL_POS);

    s32 ret = 0;
    auto startTime = std::chrono::steady_clock::now();
    auto timeout = std::chrono::seconds(timeoutSec);
    while (1) {
        if (isHost_ && !supportDevMemReg_) {
            // step1: cache尾计数
            CHK_RET(hrtDrvMemCpy(cachedTailCntAddr, sizeof(u32), devSrcTailCntAddr, sizeof(u32)));

            // step2: cache数据
            CHK_RET(hrtDrvMemCpy(
                readCacheAddr_, devMem_.size() - HCCL_HDC_CONTROL_WORDS * sizeof(u32), devMem_.ptr(),
                devMem_.size() - HCCL_HDC_CONTROL_WORDS * sizeof(u32)));

            // step3：cache头计数
            CHK_RET(hrtDrvMemCpy(cachedHeadCntAddr, sizeof(u32), devSrcHeadCntAddr, sizeof(u32)));
        } else {
            // step1: cache尾计数
            ret = memcpy_s(cachedTailCntAddr, sizeof(u32), srcTailCntAddr, sizeof(u32));
            CHK_PRT_RET(
                ret != EOK, HCCL_ERROR("[HDCommunicate][UpdateCache]memcpy_s failed, return[%d].", ret),
                HCCL_E_INTERNAL);

            // step2: cache数据
            ret = memcpy_s(
                readCacheAddr_, devMem_.size() - HCCL_HDC_CONTROL_WORDS * sizeof(u32), srcBaseAddr,
                devMem_.size() - HCCL_HDC_CONTROL_WORDS * sizeof(u32));
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

HcclResult HDCommunicate::AllocShm(u32 devid, DeviceMem& devShm, HostMem& hostShm)
{
    // 共享内存size需要按照4K(4*1024=4096)对齐
    u32 size = (buffLen_ + HCCL_HDC_CONTROL_WORDS * sizeof(u32) + HCCL_SHM_ALIGN - 1) / HCCL_SHM_ALIGN * HCCL_SHM_ALIGN;
    CHK_RET(DeviceMem::alloc(devShm, size));
    CHK_RET(hrtMemSet(devShm.ptr(), size, size));

    if (supportDevMemReg_) {
        void* hostAddr = nullptr;
        CHK_RET(hrtHalHostRegister(devShm.ptr(), devShm.size(), DEV_SVM_MAP_HOST, devid, hostAddr));

        hostShm = HostMem::create(hostAddr, devShm.size());
    } else {
        hostShm = HostMem::alloc(devShm.size());
    }
    CHK_PTR_NULL(hostShm.ptr());

    auto ret = memset_s(hostShm.ptr(), hostShm.size(), 0, hostShm.size());
    CHK_PRT_RET(ret != EOK, HCCL_ERROR("[HDCommunicate][AllocShm]memset_s failed, return[%d].", ret), HCCL_E_INTERNAL);

    return HCCL_SUCCESS;
}

HcclResult HDCommunicate::AllocReadCache(u32 flag, void*& readCacheAddr)
{
    if (flag == HCCL_HDC_TYPE_D2H) {
        hostCache_ = HostMem::alloc(hostMem_.size());
        CHK_PTR_NULL(hostCache_.ptr());
        auto ret = memset_s(hostCache_.ptr(), hostCache_.size(), 0, hostCache_.size());
        CHK_PRT_RET(
            ret != EOK, HCCL_ERROR("[HDCommunicate][AllocReadCache]memset_s failed, return[%d].", ret),
            HCCL_E_INTERNAL);
        readCacheAddr = hostCache_.ptr();
    } else {
        CHK_RET(DeviceMem::alloc(devCache_, devMem_.size()));
        CHK_RET(hrtMemSet(devCache_.ptr(), devCache_.size(), devCache_.size()));
        readCacheAddr = devCache_.ptr();
    }
    return HCCL_SUCCESS;
}
} // namespace hccl
