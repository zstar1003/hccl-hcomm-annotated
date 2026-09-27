/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hccl_aicpu_transport_interface.h"

#include <sstream>
#include "common/aicpu_hccl_def.h"
#include "common/aicpu_sqe_context.h"
#include "profiling_manager_device.h"
#include "framework/aicpu_hccl_process.h"
#include "utils/hccl_aicpu_utils.h"

extern "C" {
__attribute__((visibility("default"))) uint32_t RunTransportRoceTx(void* args)
{
    CHK_PTR_NULL(args);
    PostSendTaskParam* SRInfo = reinterpret_cast<PostSendTaskParam*>(args);
    // Check Local Flag
    uint32_t lfKey = SRInfo->lfKey;
    uint32_t rfKey = SRInfo->rfKey;
    HcclQpInfoV2 qpInfo = SRInfo->qpInfo;
    hccl::Transport::Buffer localFlagBufforWrite[3];
    hccl::Transport::Buffer localFlagBufforCheck[3];
    hccl::Transport::Buffer remoteFlagBuf[3];
    uint32_t* lFlagAddr = reinterpret_cast<uint32_t*>(SRInfo->localFlagAddr);
    uint32_t* rFlagAddr = reinterpret_cast<uint32_t*>(SRInfo->remoteFlagAddr);
    const uint64_t timeout = SRInfo->timeOut;

    // Init Flag Area
    HcclResult ret = AicpuHcclProcess::InitAsyncFlag(
        lFlagAddr, rFlagAddr, localFlagBufforCheck, localFlagBufforWrite, remoteFlagBuf);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR(
            "[AiCpuKernel][RunTransportRoceTx]InitAsyncFlag Failed, lFlagAddr is [%p], rFlagAddr is [%p], "
            "localFlagBufforCheck is [%p], localFlagBufforWrite is [%p], remoteFlagBuf is [%p].",
            lFlagAddr, rFlagAddr, localFlagBufforCheck, localFlagBufforWrite, remoteFlagBuf);
        return ret;
    }
    // start write flag 1
    ret = HcclAicpuUtils::PostSend(
        lfKey, rfKey, qpInfo, remoteFlagBuf[0], localFlagBufforWrite[0], true); // RDMA WriteFlag
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR(
            "[AiCpuKernel][RunTransportRoceTx]PostSendFlag Failed, lfKey is [%u], rfKey is [%u], qpInfo.qpPtr is [%p]"
            "localFlagBufforWrite.addr is [%llx], localFlagBufforWrite.size is [%u],remoteFlagBuf.addr is [%llx],"
            "remoteFlagBuf.size is [%u]",
            lfKey, rfKey, qpInfo.qpPtr, localFlagBufforWrite[0].addr, localFlagBufforWrite[0].size,
            remoteFlagBuf[0].addr, remoteFlagBuf[0].size);
        return ret;
    }
    // 轮询等待flag 1
    ret = AicpuHcclProcess::WaitAsyncFlag(localFlagBufforCheck, 1, timeout);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[AiCpuKernel][RunTransportRoceTx]WaitFlag Failed lfKey %u rfKey %u.", lfKey, rfKey);
        return ret;
    }
    // 向对端开始写数据
    uint32_t lKey = SRInfo->lKey;
    uint32_t rKey = SRInfo->rKey;
    hccl::Transport::Buffer remoteBuf;
    remoteBuf.addr = reinterpret_cast<void*>(SRInfo->remoteAddr);
    remoteBuf.size = static_cast<uint32_t>(SRInfo->dataSize);
    hccl::Transport::Buffer localBuf;
    localBuf.addr = reinterpret_cast<void*>(SRInfo->localAddr);
    localBuf.size = static_cast<uint32_t>(SRInfo->dataSize);
    ret = HcclAicpuUtils::PostSend(lKey, rKey, qpInfo, remoteBuf, localBuf, true); // RDMA WriteData
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR(
            "[AiCpuKernel][RunTransportRoceTx]PostSendData Failed, lKey is [%u], rKey is [%u],"
            "qpInfo.qpPtr is [%llx],remoteBuf.addr is [%llx], remoteBuf.size is [%u], localBuf.addr is [%llx],"
            "localBuf.size is [%u]",
            lKey, rKey, qpInfo.qpPtr, remoteBuf.addr, remoteBuf.size, localBuf.addr, localBuf.size);
        return ret;
    }

    // 写完数据发个flag 2，告知对端数据已经写过去了
    ret = HcclAicpuUtils::PostSend(
        lfKey, rfKey, qpInfo, remoteFlagBuf[1], localFlagBufforWrite[1], true); // RDMA WriteFlag
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR(
            "[AiCpuKernel][RunTransportRoceTx]PostSendFlag Failed, now Operation is WriteFlag, lfKey is [%u], rfKey is "
            "[%u],"
            "localFlagBufforWrite.addr is [%llx], localFlagBufforWrite.size is [%u],remoteFlagBuf.addr is [%llx],"
            "remoteFlagBuf.size is [%u]",
            lfKey, rfKey, localFlagBufforWrite[1].addr, localFlagBufforWrite[1].size, remoteFlagBuf[1].addr,
            remoteFlagBuf[1].size);
        return ret;
    }
    // 轮询等待flag 2
    ret = AicpuHcclProcess::WaitAsyncFlag(localFlagBufforCheck, 2, timeout);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[AiCpuKernel][RunTransportRoceTx]WaitFlag Failed lfKey %u rfKey %u.", lfKey, rfKey);
        return ret;
    }
    // 再发个flag 3，尾同步
    ret = HcclAicpuUtils::PostSend(
        lfKey, rfKey, qpInfo, remoteFlagBuf[2], localFlagBufforWrite[2], true); // RDMA WriteFlag
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR(
            "[AiCpuKernel][RunTransportRoceTx]PostSendFlag Failed, now Operation is WriteFlag, lfKey is [%u], rfKey is "
            "[%u],"
            "localFlagBufforWrite.addr is [%llx], localFlagBufforWrite.size is [%u],remoteFlagBuf.addr is [%llx],"
            "remoteFlagBuf.size is [%u]",
            lfKey, rfKey, localFlagBufforWrite[2].addr, localFlagBufforWrite[2].size, remoteFlagBuf[2].addr,
            remoteFlagBuf[2].size);
        return ret;
    }
    // 轮询等待flag 3
    ret = AicpuHcclProcess::WaitAsyncFlag(localFlagBufforCheck, 3, timeout);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[AiCpuKernel][RunTransportRoceTx]WaitFlag Failed lfKey %u rfKey %u.", lfKey, rfKey);
        return ret;
    }
    HCCL_INFO("[AiCpuKernel][RunTransportRoceTx]Kernel run success");
    return HCCL_SUCCESS;
}

__attribute__((visibility("default"))) uint32_t RunTransportRoceRx(void* args)
{
    CHK_PTR_NULL(args);
    PostSendTaskParam* SRInfo = reinterpret_cast<PostSendTaskParam*>(args);
    HcclQpInfoV2 qpInfo = SRInfo->qpInfo;
    uint32_t lfKey = SRInfo->lfKey;
    uint32_t rfKey = SRInfo->rfKey;
    hccl::Transport::Buffer localFlagBufforWrite[3];
    hccl::Transport::Buffer localFlagBufforCheck[3];
    hccl::Transport::Buffer remoteFlagBuf[3];

    uint32_t* lFlagAddr = reinterpret_cast<uint32_t*>(SRInfo->localFlagAddr);
    uint32_t* rFlagAddr = reinterpret_cast<uint32_t*>(SRInfo->remoteFlagAddr);
    uint64_t timeout = SRInfo->timeOut;
    // Init Flag Area
    HcclResult ret = AicpuHcclProcess::InitAsyncFlag(
        lFlagAddr, rFlagAddr, localFlagBufforCheck, localFlagBufforWrite, remoteFlagBuf);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR(
            "[AiCpuKernel][RunTransportRoceTx]InitAsyncFlag Failed, lFlagAddr is [%p], rFlagAddr is [%p], "
            "localFlagBufforCheck is [%p], localFlagBufforWrite is [%p], remoteFlagBuf is [%p].",
            lFlagAddr, rFlagAddr, localFlagBufforCheck, localFlagBufforWrite, remoteFlagBuf);
        return ret;
    }
    // start write flag 1
    ret = HcclAicpuUtils::PostSend(
        lfKey, rfKey, qpInfo, remoteFlagBuf[0], localFlagBufforWrite[0], true); // RDMA WriteFlag
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR(
            "[AiCpuKernel][RunTransportRoceRx]PostSendFlag Failed, lfKey is [%u], rfKey is [%u], qpInfo.qpPtr is [%p]"
            "localFlagBufforWrite.addr is [%llx], localFlagBufforWrite.size is [%u],remoteFlagBuf.addr is [%llx],"
            "remoteFlagBuf.size is [%u]",
            lfKey, rfKey, qpInfo.qpPtr, localFlagBufforWrite[0].addr, localFlagBufforWrite[0].size,
            remoteFlagBuf[0].addr, remoteFlagBuf[0].size);
        return ret;
    }
    // 轮询等待flag 1
    ret = AicpuHcclProcess::WaitAsyncFlag(localFlagBufforCheck, 1, timeout);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[AiCpuKernel][RunTransportRoceRx]WaitFlag Failed lfKey %u rfKey %u.", lfKey, rfKey);
        return ret;
    }

    ret = HcclAicpuUtils::PostSend(
        lfKey, rfKey, qpInfo, remoteFlagBuf[1], localFlagBufforWrite[1], true); // RDMA WriteFlag
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR(
            "[AiCpuKernel][RunTransportRoceRx]PostSendFlag Failed, now Operation is WriteFlag, lfKey is [%u], rfKey is "
            "[%u],"
            "localFlagBufforWrite.addr is [%llx], localFlagBufforWrite.size is [%u],remoteFlagBuf.addr is [%llx],"
            "remoteFlagBuf.size is [%u]",
            lfKey, rfKey, localFlagBufforWrite[1].addr, localFlagBufforWrite[1].size, remoteFlagBuf[1].addr,
            remoteFlagBuf[1].size);
        return ret;
    }
    // 轮询等待flag 2
    ret = AicpuHcclProcess::WaitAsyncFlag(localFlagBufforCheck, 2, timeout);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[AiCpuKernel][RunTransportRoceRx]WaitFlag Failed lfKey %u rfKey %u.", lfKey, rfKey);
        return ret;
    }
    // 再发个flag 3，尾同步
    ret = HcclAicpuUtils::PostSend(
        lfKey, rfKey, qpInfo, remoteFlagBuf[2], localFlagBufforWrite[2], true); // RDMA WriteFlag
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR(
            "[AiCpuKernel][RunTransportRoceRx]PostSendFlag Failed, now Operation is WriteFlag, lfKey is [%u], rfKey is "
            "[%u],"
            "localFlagBufforWrite.addr is [%llx], localFlagBufforWrite.size is [%u],remoteFlagBuf.addr is [%llx],"
            "remoteFlagBuf.size is [%u]",
            lfKey, rfKey, localFlagBufforWrite[2].addr, localFlagBufforWrite[2].size, remoteFlagBuf[2].addr,
            remoteFlagBuf[2].size);
        return ret;
    }
    // 轮询等待flag 3
    ret = AicpuHcclProcess::WaitAsyncFlag(localFlagBufforCheck, 3, timeout);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[AiCpuKernel][RunTransportRoceRx]WaitFlag Failed lfKey %u rfKey %u.", lfKey, rfKey);
        return ret;
    }
    HCCL_INFO("[AiCpuKernel][RunTransportRoceRx]Kernel run success");
    return HCCL_SUCCESS;
}
} // extern "C"
