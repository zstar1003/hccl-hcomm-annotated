/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BUILTIN_CHANNEL_OPS_H
#define BUILTIN_CHANNEL_OPS_H

#include "cast_utils.h"
#include "hcomm_nic_plugin.h"
#include "nic_plugin_manager.h"
#include "channel.h"
#include "channel_process.h"
#include "dtype_common.h"
// CreateBuiltinChannel 已调用 channelPtr->Init()，此处 no-op，此接口内置流程暂未调用。
inline int32_t BuiltinChannelInit(void* ctx)
{
    (void)ctx;
    return HCCL_SUCCESS;
}

// channel 生命周期由 g_ChannelMap 的 unique_ptr 管理，此处 no-op，此接口内置流程暂未调用。
inline int32_t BuiltinChannelDestroy(void* ctx)
{
    (void)ctx;
    return HCCL_SUCCESS;
}

inline int32_t BuiltinGetStatus(void* ctx, int32_t* status)
{
    auto* const channelPtr = static_cast<hcomm::Channel*>(ctx);
    CHK_PTR_NULL(channelPtr);
    auto channelStatus = channelPtr->GetStatus();
    switch (channelStatus) {
        case hcomm::ChannelStatus::FAILED:
            *status = hcomm::HCOMM_CHANNEL_STATUS_FAILED;
            break;
        case hcomm::ChannelStatus::SOCKET_TIMEOUT:
            *status = hcomm::HCOMM_CHANNEL_STATUS_TIMEOUT;
            break;
        case hcomm::ChannelStatus::READY:
            *status = hcomm::HCOMM_CHANNEL_STATUS_READY;
            break;
        default:
            *status = hcomm::HCOMM_CHANNEL_STATUS_CONNECTING;
            break;
    }
    return HCCL_SUCCESS;
}

template <typename Op>
inline int32_t BuiltinRwNbiOnThread(void* ctx, ThreadHandle thread, void* dst, const void* src, uint64_t len, Op&& op)
{
    HCCL_INFO(
        "[%s] START. thread[0x%llx], channel[0x%llx], dst[0x%llx], src[0x%llx], len[%llu].", __func__, thread, ctx, dst,
        src, len);

    (void)thread;
    CHK_PTR_NULL(src);
    CHK_PTR_NULL(dst);

    HcclResult ret = HCCL_SUCCESS;
    DevType devType;
    CHK_RET(hrtGetDeviceType(devType));
    if (devType == DevType::DEV_TYPE_950 || devType == DevType::DEV_TYPE_960) {
        auto* const channelPtr = static_cast<hcomm::Channel*>(ctx);
        CHK_PTR_NULL(channelPtr);
        ret = op(channelPtr, dst, src, len);
    } else {
        ret = HCCL_E_NOT_SUPPORT;
    }
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[%s] FAIL. thread[0x%llx], channel[0x%llx], dst[0x%llx], src[0x%llx], len[%llu].", __func__, thread, ctx,
            dst, src, len),
        ret);
    HCCL_INFO("[%s] SUCCESS.", __func__);
    return HCCL_SUCCESS;
}

inline int32_t BuiltinWriteNbiOnThread(void* ctx, ThreadHandle thread, void* dst, const void* src, uint64_t len)
{
    return BuiltinRwNbiOnThread(ctx, thread, dst, src, len, [](hcomm::Channel* ch, void* d, const void* s, uint64_t l) {
        return ch->Write(d, s, l);
    });
}

inline int32_t BuiltinWriteNbi(void* ctx, void* dst, const void* src, uint64_t len)
{
    return BuiltinWriteNbiOnThread(ctx, 0, dst, src, len);
}

inline int32_t BuiltinWriteWithNotifyNbiOnThread(
    void* ctx, ThreadHandle thread, void* dst, const void* src, uint64_t len, uint32_t remoteNotifyIdx)
{
    HCCL_INFO(
        "[%s] START. thread[0x%llx], channel[0x%llx], dst[0x%llx], src[0x%llx], len[%llu], remoteNotifyIdx[%u].",
        __func__, thread, ctx, dst, src, len, remoteNotifyIdx);

    (void)thread;
    CHK_PTR_NULL(src);
    CHK_PTR_NULL(dst);
    HcclResult ret = HCCL_SUCCESS;
    DevType devType;
    CHK_RET(hrtGetDeviceType(devType));
    if (devType == DevType::DEV_TYPE_950 || devType == DevType::DEV_TYPE_960 || thread == 0) {
        auto* const channelPtr = static_cast<hcomm::Channel*>(ctx);
        CHK_PTR_NULL(channelPtr);
        ret = channelPtr->WriteWithNotify(dst, src, len, remoteNotifyIdx);
    } else {
        ret = HCCL_E_NOT_SUPPORT;
    }
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[%s] FAIL. thread[0x%llx], channel[0x%llx], dst[0x%llx], src[0x%llx], len[%llu], remoteNotifyIdx[%u].",
            __func__, thread, ctx, dst, src, len, remoteNotifyIdx),
        ret);
    HCCL_INFO("[%s] SUCCESS.", __func__);
    return HCCL_SUCCESS;
}

inline int32_t BuiltinWriteWithNotifyNbi(void* ctx, void* dst, const void* src, uint64_t len, uint32_t remoteNotifyIdx)
{
    return BuiltinWriteWithNotifyNbiOnThread(ctx, 0, dst, src, len, remoteNotifyIdx);
}

// 供A5/A6接口BuiltinNotifyWait调用（原流程HcommChannelNotifyWait->HcommChannelNotifyWaitOnThread）,A2/A3内置不走该接口
inline int32_t BuiltinNotifyWaitOnThread(void* ctx, ThreadHandle thread, uint32_t localNotifyIdx, uint32_t timeOut)
{
    (void)thread;
    HCCL_INFO(
        "[%s] START. thread[0x%llx], channel[0x%llx], localNotifyIdx[%u], timeOut[%u].", __func__, thread, ctx,
        localNotifyIdx, timeOut);
    auto* const channelPtr = static_cast<hcomm::Channel*>(ctx);
    CHK_PTR_NULL(channelPtr);
    HcclResult ret = channelPtr->NotifyWait(localNotifyIdx, timeOut);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[%s] FAIL. thread[0x%llx], channel[0x%llx], localNotifyIdx[%u], timeOut[%u].", __func__, thread, ctx,
            localNotifyIdx, timeOut),
        ret);
    HCCL_INFO("[%s] SUCCESS.", __func__);
    return HCCL_SUCCESS;
}

inline int32_t BuiltinNotifyWait(void* ctx, uint32_t localNotifyIdx, uint32_t timeOut)
{
    DevType devType;
    CHK_RET(hrtGetDeviceType(devType));
    if (devType != DevType::DEV_TYPE_950 && devType != DevType::DEV_TYPE_960) {
        return HCCL_E_NOT_SUPPORT;
    }
    return BuiltinNotifyWaitOnThread(ctx, 0, localNotifyIdx, timeOut);
}

// 供A5/A6接口BuiltinNotifyRecord调用（原流程HcommChannelNotifyRecord->HcommChannelNotifyRecordOnThread）,A2/A3内置不走该接口
inline int32_t BuiltinNotifyRecordOnThread(void* ctx, ThreadHandle thread, uint32_t remoteNotifyIdx)
{
    (void)thread;
    HCCL_INFO(
        "[%s] START. thread[0x%llx], channel[0x%llx], remoteNotifyIdx[%u].", __func__, thread, ctx, remoteNotifyIdx);
    auto* const channelPtr = static_cast<hcomm::Channel*>(ctx);
    CHK_PTR_NULL(channelPtr);
    HcclResult ret = channelPtr->NotifyRecord(remoteNotifyIdx);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[%s] FAIL. thread[0x%llx], channel[0x%llx], remoteNotifyIdx[%u].", __func__, thread, ctx, remoteNotifyIdx),
        ret);
    HCCL_INFO("[%s] SUCCESS.", __func__);
    return HCCL_SUCCESS;
}

inline int32_t BuiltinNotifyRecord(void* ctx, uint32_t remoteNotifyIdx)
{
    DevType devType;
    CHK_RET(hrtGetDeviceType(devType));
    if (devType != DevType::DEV_TYPE_950 && devType != DevType::DEV_TYPE_960) {
        return HCCL_E_NOT_SUPPORT;
    }
    return BuiltinNotifyRecordOnThread(ctx, 0, remoteNotifyIdx);
}

inline int32_t BuiltinReadNbiOnThread(void* ctx, ThreadHandle thread, void* dst, const void* src, uint64_t len)
{
    return BuiltinRwNbiOnThread(ctx, thread, dst, src, len, [](hcomm::Channel* ch, void* d, const void* s, uint64_t l) {
        return ch->Read(d, s, l);
    });
}

inline int32_t BuiltinReadNbi(void* ctx, void* dst, const void* src, uint64_t len)
{
    return BuiltinReadNbiOnThread(ctx, 0, dst, src, len);
}

inline int32_t BuiltinFenceOnThread(void* ctx, ThreadHandle thread)
{
    HCCL_INFO("[%s] START. thread[0x%llx], channel[0x%llx].", __func__, thread, ctx);

    (void)thread;
    HcclResult ret = HCCL_SUCCESS;
    DevType devType;
    CHK_RET(hrtGetDeviceType(devType));
    if (devType == DevType::DEV_TYPE_950 || devType == DevType::DEV_TYPE_960 || thread == 0) {
        auto* const channelPtr = static_cast<hcomm::Channel*>(ctx);
        CHK_PTR_NULL(channelPtr);
        ret = channelPtr->ChannelFence();
    } else {
        ret = HCCL_E_NOT_SUPPORT;
    }
    CHK_PRT_RET(
        ret != HCCL_SUCCESS, HCCL_ERROR("[%s] FAIL. thread[0x%llx], channel[0x%llx].", __func__, thread, ctx), ret);
    HCCL_INFO("[%s] SUCCESS.", __func__);
    return HCCL_SUCCESS;
}

inline int32_t BuiltinFence(void* ctx) { return BuiltinFenceOnThread(ctx, 0); }

inline int32_t BuiltinDrainOnThread(void* ctx, ThreadHandle thread)
{
    HCCL_INFO("[%s] START. thread[0x%llx], channel[0x%llx].", __func__, thread, ctx);

    HcclResult ret = HCCL_SUCCESS;
    DevType devType;
    CHK_RET(hrtGetDeviceType(devType));
    if (devType == DevType::DEV_TYPE_950 || devType == DevType::DEV_TYPE_960) {
        auto* const channelPtr = ReinterpretAs<hcomm::Channel*>(ctx);
        CHK_PTR_NULL(channelPtr);
        ret = channelPtr->ChannelDrain();
    } else {
        ret = HCCL_E_NOT_SUPPORT;
    }
    CHK_PRT_RET(
        ret != HCCL_SUCCESS, HCCL_ERROR("[%s] FAIL. thread[0x%llx], channel[0x%llx].", __func__, thread, ctx), ret);
    HCCL_INFO("[%s] SUCCESS.", __func__);
    return HCCL_SUCCESS;
}

// 以下接口内置不支持，复用 nic_plugin_manager.cc 中的 DefaultChannel* 默认实现

inline HcommNicChannelOps g_BuiltinChannelOps = {
    {HCOMM_NIC_CHANNEL_OPS_VERSION, HCOMM_NIC_CHANNEL_OPS_MAGIC_WORD, sizeof(HcommNicChannelOps), 0},
    BuiltinChannelInit,                                        // init
    BuiltinChannelDestroy,                                     // destroy
    BuiltinGetStatus,                                          // getStatus
    BuiltinWriteNbi,                                           // writeNbi
    BuiltinWriteNbiOnThread,                                   // writeNbiOnThread
    hcomm::DefaultChannelWriteOnThread,                        // writeOnThread
    BuiltinWriteWithNotifyNbi,                                 // writeWithNotifyNbi
    BuiltinWriteWithNotifyNbiOnThread,                         // writeWithNotifyNbiOnThread
    hcomm::DefaultChannelWriteWithNotifyOnThread,              // writeWithNotifyOnThread
    hcomm::DefaultChannelWriteReduceOnThread,                  // writeReduceOnThread
    hcomm::DefaultChannelWriteReduceWithNotifyOnThread,        // writeReduceWithNotifyOnThread
    BuiltinReadNbi,                                            // readNbi
    BuiltinReadNbiOnThread,                                    // readNbiOnThread
    hcomm::DefaultChannelReadOnThread,                         // readOnThread
    hcomm::DefaultChannelReadReduceOnThread,                   // readReduceOnThread
    BuiltinNotifyRecord,                                       // notifyRecord
    BuiltinNotifyRecordOnThread,                               // notifyRecordOnThread
    BuiltinNotifyWait,                                         // notifyWait
    BuiltinNotifyWaitOnThread,                                 // notifyWaitOnThread
    hcomm::DefaultChannelNotifyWaitOnThreadWithDefaultTimeout, // notifyWaitOnThreadWithDefaultTimeout
    hcomm::DefaultChannelBatchTransferOnThread,                // batchTransferOnThread
    BuiltinFence,                                              // fence
    BuiltinFenceOnThread,                                      // fenceOnThread
    BuiltinDrainOnThread,                                      // drainOnThread
};

#endif // BUILTIN_CHANNEL_OPS_H
