/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "zero_copy_address_mgr.h"
#include "adapter_rts_common.h"

namespace hccl {
HcclResult ZeroCopyAddressMgr::InitRingBuffer()
{
    if (ringBuffer_.ptr() != nullptr) {
        return HCCL_SUCCESS;
    }

    CHK_RET(DeviceMem::alloc(ringBuffer_, ZERO_COPY_BUFFER_MAX_MAP_COUNT * sizeof(ZeroCopyRingBufferItem)));
    CHK_RET(hrtMemSet(ringBuffer_.ptr(), ringBuffer_.size(), ringBuffer_.size()));

    CHK_RET(DeviceMem::alloc(ringBufferCtl_, sizeof(u32) + sizeof(u32)));
    CHK_RET(hrtMemSet(ringBufferCtl_.ptr(), ringBufferCtl_.size(), ringBufferCtl_.size()));

    devRingBufBase_ = reinterpret_cast<ZeroCopyRingBufferItem*>(ringBuffer_.ptr());
    devRingHead_ = reinterpret_cast<u32*>(ringBufferCtl_.ptr());
    devRingTail_ = devRingHead_ + 1;

    HCCL_RUN_INFO(
        "[ZeroCopyAddressMgr][InitRingBuffer] ringbuffer[%p] len[%lu] bufferCtl[%p] len[%lu] head[%p] tail[%p]",
        ringBuffer_.ptr(), ringBuffer_.size(), ringBufferCtl_.ptr(), ringBufferCtl_.size(), devRingHead_, devRingTail_);

    HCCL_INFO(
        "[ZeroCopyAddressMgr][InitRingBuffer] ringbuffer[%p] head[%p] tail[%p]", devRingBufBase_, devRingHead_,
        devRingTail_);
    return HCCL_SUCCESS;
}

HcclResult ZeroCopyAddressMgr::PushOne(ZeroCopyRingBufferItem& item)
{
    std::lock_guard<std::mutex> guard(processRingBufferLock_);
    if (!needPushOne) {
        HCCL_DEBUG("[ZeroCopyAddressMgr][PushOne] don't need push");
        return HCCL_SUCCESS;
    }

    // 检测RingBuffer是否已经初始化，没有的话就初始化一下
    CHK_RET(InitRingBuffer());

    u32 head = 0;
    CHK_RET(hrtMemSyncCopy(
        &head, sizeof(head), devRingHead_, sizeof(head), HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_DEVICE_TO_HOST));
    u32 tail = 0;
    CHK_RET(hrtMemSyncCopy(
        &tail, sizeof(tail), devRingTail_, sizeof(tail), HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_DEVICE_TO_HOST));

    u32 updateTail = (tail + 1) % ZERO_COPY_BUFFER_MAX_MAP_COUNT;
    CHK_PRT_RET(
        updateTail == head,
        HCCL_ERROR(
            "[ZeroCopyAddressMgr][PushOne] ring buffer is full head[%u] tail[%u] capacity[%u]", head, tail,
            ZERO_COPY_BUFFER_MAX_MAP_COUNT),
        HCCL_E_INTERNAL);

    HCCL_INFO(
        "[ZeroCopyAddressMgr][PushOne] type[%d] head[%u] tail[%u] updateTail[%u] tailAddr[%p]", item.type, head, tail,
        updateTail, devRingBufBase_ + tail);
    CHK_RET(hrtMemSyncCopy(
        devRingBufBase_ + tail, sizeof(ZeroCopyRingBufferItem), &item, sizeof(ZeroCopyRingBufferItem),
        HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE));
    CHK_RET(hrtMemSyncCopy(
        devRingTail_, sizeof(updateTail), &updateTail, sizeof(updateTail),
        HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE));

    return HCCL_SUCCESS;
}

} // namespace hccl
