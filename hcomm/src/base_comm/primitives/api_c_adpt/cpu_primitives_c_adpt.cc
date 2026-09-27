/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "cast_utils.h"
#include "hccl_api_data.h"
#include "new/hccl_primitive_local.h"
#include "new/hccl_primitive_remote.h"
#include "thread.h"
#include "dfx_dlprof_function.h"
#include "launch_context.h"
#include "host/host_cpu_roce_channel.h"
#include "hccl_comm_pub.h"
#include "op_base.h"
#include "hcclCommOp.h"
#include "adapter_prof.h"
#include "hccl_diag.h"
#include "exception_handler.h"
#include "task_param.h"
#include "nic_plugin_manager.h"

using namespace hccl;
thread_local LaunchContext g_threadLaunchCtx;

void AddThreadWithTag(ThreadHandle thread) { g_threadLaunchCtx.AddThreadWithTag(thread); }

bool IsSupportReduce(HcommDataType dataType, HcommReduceOp op)
{
    bool checkDataType
        = (dataType == HCOMM_DATA_TYPE_FP32 || dataType == HCOMM_DATA_TYPE_FP16 || dataType == HCOMM_DATA_TYPE_INT8
           || dataType == HCOMM_DATA_TYPE_INT16 || dataType == HCOMM_DATA_TYPE_INT32
           || dataType == HCOMM_DATA_TYPE_BFP16);
    bool checkReduceType = (op == HCOMM_REDUCE_SUM || op == HCOMM_REDUCE_MAX || op == HCOMM_REDUCE_MIN);
    return checkDataType && checkReduceType;
}

int32_t HcommSetNotifyWaitTimeOut(float timeOut)
{
    if (std::isnan(timeOut) || timeOut < 0.0f || timeOut > static_cast<float>(UINT32_MAX)) {
        HCCL_ERROR("[%s] in cpu timeOut[%f s] is invalid.", __func__, timeOut);
        return HCCL_E_PARA;
    }
    uint32_t timeOutInt = static_cast<uint32_t>(timeOut);
    HCCL_INFO("[%s] START in cpu. timeOut[%u s].", __func__, timeOutInt);
    return g_threadLaunchCtx.SetNotifyWaitTimeOut(timeOutInt);
}

int32_t HcommThreadResAcquireTimeOut(float timeOut)
{
    HCCL_ERROR("[%s] timeOut[%f s], not support in cpu.", __func__, timeOut);
    return HCCL_E_NOT_SUPPORT;
}

int32_t HcommLocalCopyOnThread(ThreadHandle thread, void* dst, const void* src, uint64_t len)
{
    HCCL_INFO("[%s] START. thread[0x%llx], dst[0x%llx], src[0x%llx], len[%llu].", __func__, thread, dst, src, len);

    CHK_PTR_NULL(dst);
    CHK_PTR_NULL(src);
    AddThreadWithTag(thread);

    Thread* const threadPtr = ReinterpretAs<Thread*>(thread);
    CHK_PTR_NULL(threadPtr);

    if (threadPtr->IsDeviceA5()) {
        CHK_RET(threadPtr->LocalCopy(dst, src, len));
    } else {
        HcclBuf srcBuf{const_cast<void*>(src), len, nullptr};
        HcclBuf dstBuf{dst, len, nullptr};
        Stream* stream = GetStream(thread);
        CHK_PTR_NULL(stream);

        HcclResult ret = HcclLocalCopy(stream, &dstBuf, &srcBuf);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR(
                "[%s] FAIL. thread[0x%llx], dst[0x%llx], src[0x%llx], len[%llu].", __func__, thread, dst, src, len),
            ret);
    }
    HCCL_INFO("[%s] SUCCESS.", __func__);
    return HCCL_SUCCESS;
}

int32_t HcommLocalReduceOnThread(
    ThreadHandle thread, void* dst, const void* src, uint64_t count, HcommDataType dataType, HcommReduceOp reduceOp)
{
    HCCL_INFO(
        "[%s] START. thread[0x%llx], dst[0x%llx], src[0x%llx], count[%llu], dataType[%d], reduceOp[%d].", __func__,
        thread, dst, src, count, dataType, reduceOp);

    CHK_PTR_NULL(dst);
    CHK_PTR_NULL(src);
    CHK_PRT_RET(
        (IsSupportReduce(dataType, reduceOp) == false),
        HCCL_ERROR(
            "[HcommLocalReduceOnThread] Not support reduce, "
            "dst[%p], src[%p], count[%llu], dataType[%d], reduceOp[%d]",
            dst, src, count, dataType, reduceOp),
        HCCL_E_PARA);
    AddThreadWithTag(thread);

    Thread* const threadPtr = ReinterpretAs<Thread*>(thread);
    CHK_PTR_NULL(threadPtr);

    uint64_t len = count * SIZE_TABLE[dataType];

    if (threadPtr->IsDeviceA5()) {
        CHK_RET(threadPtr->LocalReduce(dst, src, len, dataType, reduceOp));
    } else {
        HcclBuf srcBuf{const_cast<void*>(src), len, nullptr};
        HcclBuf dstBuf{dst, len, nullptr};
        HcclReduceInfo reduceInfo{static_cast<HcclDataType>(dataType), static_cast<HcclReduceOp>(reduceOp)};
        Stream* stream = GetStream(thread);
        CHK_PTR_NULL(stream);

        HcclResult ret = HcclLocalCopyReduce(stream, &dstBuf, &srcBuf, reduceInfo);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR(
                "[%s] FAIL. thread[0x%llx], dst[0x%llx], src[0x%llx], count[%llu], dataType[%d], reduceOp[%d].",
                __func__, thread, dst, src, count, dataType, reduceOp),
            ret);
    }
    HCCL_INFO("[%s] SUCCESS.", __func__);
    return HCCL_SUCCESS;
}

int32_t HcommThreadNotifyRecordOnThread(ThreadHandle thread, ThreadHandle dstThread, uint32_t dstNotifyIdx)
{
    HCCL_INFO(
        "[%s] START. thread[0x%llx], dstThread[0x%llx], dstNotifyIdx[%u].", __func__, thread, dstThread, dstNotifyIdx);

    AddThreadWithTag(thread);

    Thread* const threadPtr = ReinterpretAs<Thread*>(thread);
    CHK_PTR_NULL(threadPtr);

    if (threadPtr->IsDeviceA5()) {
        HcclResult ret = threadPtr->LocalNotifyRecord(dstThread, dstNotifyIdx);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR(
                "[%s] FAIL. thread[0x%llx], dstThread[0x%llx], notifyIdx[%u].", __func__, thread, dstThread,
                dstNotifyIdx),
            ret);
    } else {
        Stream* stream = GetStream(thread);
        CHK_PTR_NULL(stream);

        LocalNotify* notify = GetNotify(dstThread, dstNotifyIdx);
        CHK_PTR_NULL(notify);

        HcclResult ret = HcclLocalNotifyRecord(stream, notify);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR(
                "[%s] FAIL. thread[0x%llx], dstThread[0x%llx], notifyIdx[%u].", __func__, thread, dstThread,
                dstNotifyIdx),
            ret);
    }

    HCCL_INFO("[%s] SUCCESS.", __func__);
    return HCCL_SUCCESS;
}

int32_t HcommThreadNotifyWaitOnThread(ThreadHandle thread, uint32_t notifyIdx, uint32_t timeOut)
{
    HCCL_INFO("[%s] START. thread[0x%llx], notifyIdx[%u], timeOut[%u s].", __func__, thread, notifyIdx, timeOut);

    AddThreadWithTag(thread);

    Thread* const threadPtr = ReinterpretAs<Thread*>(thread);
    CHK_PTR_NULL(threadPtr);

    if (threadPtr->IsDeviceA5()) {
        HcclResult ret = threadPtr->LocalNotifyWait(notifyIdx, timeOut);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR(
                "[%s] FAIL. thread[0x%llx], notifyIdx[%u], timeOut[%u s].", __func__, thread, notifyIdx, timeOut),
            ret);
    } else {
        Stream* stream = GetStream(thread);
        CHK_PTR_NULL(stream);
        LocalNotify* notify = GetNotify(thread, notifyIdx);
        CHK_PTR_NULL(notify);

        HcclResult ret = HcclLocalNotifyWait(stream, notify, timeOut);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR(
                "[%s] FAIL. thread[0x%llx], notifyIdx[%u], timeOut[%u s].", __func__, thread, notifyIdx, timeOut),
            ret);
    }
    HCCL_INFO("[%s] SUCCESS.", __func__);
    return HCCL_SUCCESS;
}

int32_t HcommAclrtNotifyRecordOnThread(ThreadHandle thread, uint64_t dstNotifyId)
{
    HCCL_INFO("[%s] START. thread[0x%llx], dstNotifyId[%u].", __func__, thread, dstNotifyId);

    AddThreadWithTag(thread);

    Thread* const threadPtr = ReinterpretAs<Thread*>(thread);
    CHK_PTR_NULL(threadPtr);

    Stream* stream = GetStream(thread);
    CHK_PTR_NULL(stream);

    HcclResult ret = HcclLocalBareNotifyRecord(stream, dstNotifyId);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS, HCCL_ERROR("[%s] FAIL. thread[0x%llx], dstNotifyId[%u].", __func__, thread, dstNotifyId),
        ret);
    HCCL_INFO("[%s] SUCCESS.", __func__);
    return HCCL_SUCCESS;
}

int32_t HcommAclrtNotifyWaitOnThread(ThreadHandle thread, uint64_t notifyId, uint32_t timeOut)
{
    HCCL_INFO("[%s] START. thread[0x%llx], notifyId[%llu], timeOut[%u s].", __func__, thread, notifyId, timeOut);

    AddThreadWithTag(thread);

    Thread* const threadPtr = ReinterpretAs<Thread*>(thread);
    CHK_PTR_NULL(threadPtr);

    Stream* stream = GetStream(thread);
    CHK_PTR_NULL(stream);

    HcclResult ret = HcclLocalBareNotifyWait(stream, notifyId, timeOut);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR("[%s] FAIL. thread[0x%llx], notifyId[%llu], timeOut[%u s].", __func__, thread, notifyId, timeOut),
        ret);
    HCCL_INFO("[%s] SUCCESS.", __func__);
    return HCCL_SUCCESS;
}

HcclResult CommTaskPrepare(char* key, uint32_t keyLen) // host ffts+使用
{
    std::string keyStr = "temp_key";
    if (key != nullptr && keyLen != 0) {
        keyStr = std::string(key, keyLen);
        HCCL_DEBUG("[CommTaskPrepare]key[%s], keyLen[%u]", key, keyLen);
    } else {
        HCCL_DEBUG("[CommTaskPrepare]disable cache, key[0x%llx], keyLen[%u]", key, keyLen);
    }

    return HcclTaskPrepare(const_cast<char_t*>(keyStr.c_str()), keyStr.length());
}

HcclResult CommTaskLaunch(ThreadHandle* threads, uint32_t threadNum) // host ffts+或aicpu stars使用"
{
    CHK_PTR_NULL(threads);
    CHK_PRT_RET(threadNum < 1, HCCL_ERROR("[CommTaskLaunch]threadNum is less than 1"), HCCL_E_PARA);

    Thread* threadPtr = ReinterpretAs<Thread*>(threads[0]);
    CHK_PTR_NULL(threadPtr);

    std::vector<hccl::Stream> streams;
    for (uint32_t i = 0; i < threadNum; i++) {
        hccl::Stream* stream = GetStream(threads[i]);
        CHK_PTR_NULL(stream);
        streams.push_back(*stream);
    }

    return HcclTaskLaunch(streams.data(), threadNum);
}

HcclResult DispatchAllStreams(const ThreadHandle* threads, uint32_t threadNum)
{
    CHK_PTR_NULL(threads);
    CHK_PRT_RET(threadNum < 1, HCCL_ERROR("[DispatchAllStreams]threadNum is less than 1"), HCCL_E_PARA);

    HCCL_WARNING("[DispatchAllStreams] DispatchAllStreams is only supported on A5 device, skip");
    return HCCL_E_NOT_SUPPORT;
}

int32_t HcommWriteOnThread(ThreadHandle thread, ChannelHandle channel, void* dst, const void* src, uint64_t len)
{
    if (IS_PLUGIN_HANDLE(channel)) {
        auto* ch = CHANNEL_FROM_HANDLE(channel);
        CHK_PTR_NULL(ch);
        return ch->GetNicOps()->writeOnThread(ch->GetNicCtx(), thread, dst, src, len);
    }
    HCCL_INFO(
        "[%s] START. thread[0x%llx], channel[0x%llx], dst[0x%llx], src[0x%llx], len[%llu].", __func__, thread, channel,
        dst, src, len);

    CHK_PTR_NULL(dst);
    CHK_PTR_NULL(src);

    AddThreadWithTag(thread);

    Thread* const threadPtr = ReinterpretAs<Thread*>(thread);
    CHK_PTR_NULL(threadPtr);

    HcclBuf locBuf{const_cast<void*>(src), len, nullptr};
    HcclBuf rmtBuf{dst, len, nullptr};

    Stream* stream = GetStream(thread);
    CHK_PTR_NULL(stream);

    HcclResult ret = HcclRemoteWrite(stream, ReinterpretAs<void*>(channel), &rmtBuf, &locBuf);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[%s] FAIL. thread[0x%llx], channel[0x%llx], dst[0x%llx], src[0x%llx], len[%llu].", __func__, thread,
            channel, dst, src, len),
        ret);
    HCCL_INFO("[%s] SUCCESS.", __func__);
    return HCCL_SUCCESS;
}

int32_t HcommBatchTransferOnThread(
    ThreadHandle thread, ChannelHandle channel, const HcommBatchTransferDesc* transferDescs, uint32_t transferDescNum)
{
    if (IS_PLUGIN_HANDLE(channel)) {
        auto* ch = CHANNEL_FROM_HANDLE(channel);
        CHK_PTR_NULL(ch);
        return ch->GetNicOps()->batchTransferOnThread(ch->GetNicCtx(), thread, transferDescs, transferDescNum);
    }
    HCCL_ERROR(" [HcommBatchTransferOnThread] not support in cpu");
    return HCCL_E_NOT_SUPPORT;
}

int32_t HcommWriteReduceOnThread(
    ThreadHandle thread, ChannelHandle channel, void* dst, const void* src, uint64_t count, HcommDataType dataType,
    HcommReduceOp reduceOp)
{
    if (IS_PLUGIN_HANDLE(channel)) {
        auto* ch = CHANNEL_FROM_HANDLE(channel);
        CHK_PTR_NULL(ch);
        return ch->GetNicOps()->writeReduceOnThread(ch->GetNicCtx(), thread, dst, src, count, dataType, reduceOp);
    }

    HCCL_INFO(
        "[%s] START. thread[0x%llx], channel[0x%llx], dst[0x%llx], src[0x%llx], count[%llu], dataType[%d], "
        "reduceOp[%d].",
        __func__, thread, channel, dst, src, count, dataType, reduceOp);

    CHK_PTR_NULL(dst);
    CHK_PTR_NULL(src);

    CHK_PRT_RET(
        (IsSupportReduce(dataType, reduceOp) == false),
        HCCL_ERROR(
            "[HcommWriteReduceOnThread] Not support reduce, "
            "dst[%p], src[%p], count[%llu], dataType[%d], reduceOp[%d]",
            dst, src, count, dataType, reduceOp),
        HCCL_E_PARA);
    AddThreadWithTag(thread);

    Thread* const threadPtr = ReinterpretAs<Thread*>(thread);
    CHK_PTR_NULL(threadPtr);

    uint64_t len = count * SIZE_TABLE[dataType];

    HcclBuf locBuf{const_cast<void*>(src), len, nullptr};
    HcclBuf rmtBuf{dst, len, nullptr};
    HcclReduceInfo reduceInfo{static_cast<HcclDataType>(dataType), static_cast<HcclReduceOp>(reduceOp)};

    Stream* stream = GetStream(thread);
    CHK_PTR_NULL(stream);

    HcclResult ret = HcclRemoteWriteReduce(stream, ReinterpretAs<void*>(channel), &rmtBuf, &locBuf, reduceInfo);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[%s] FAIL. thread[0x%llx], channel[0x%llx], dst[0x%llx], src[0x%llx], count[%llu], dataType[%d], "
            "reduceOp[%d].",
            __func__, thread, channel, dst, src, count, dataType, reduceOp),
        ret);
    HCCL_INFO("[%s] SUCCESS.", __func__);
    return HCCL_SUCCESS;
}

HcclResult CommWriteReduceWithNotify(
    ThreadHandle thread, ChannelHandle channel, void* dst, const void* src, uint64_t count, HcommDataType dataType,
    HcommReduceOp reduceOp, uint32_t remoteNotifyIdx)
{
    CHK_PTR_NULL(src);
    CHK_PTR_NULL(dst);

    CHK_PRT_RET(
        (IsSupportReduce(dataType, reduceOp) == false),
        HCCL_ERROR(
            "[CommWriteReduceWithNotify] Not support reduce, "
            "dst[%p], src[%p], count[%llu], dataType[%d], reduceOp[%d]",
            dst, src, count, dataType, reduceOp),
        HCCL_E_PARA);
    AddThreadWithTag(thread);
    HcclBuf locBuf{const_cast<void*>(src), count * SIZE_TABLE[dataType], nullptr};
    HcclBuf rmtBuf{dst, count * SIZE_TABLE[dataType], nullptr};
    HcclReduceInfo reduceInfo{static_cast<HcclDataType>(dataType), static_cast<HcclReduceOp>(reduceOp)};

    Stream* stream = GetStream(thread);
    CHK_PTR_NULL(stream);

    return HcclRemoteWriteReduceWithNotify(
        stream, ReinterpretAs<void*>(channel), &rmtBuf, &locBuf, reduceInfo, remoteNotifyIdx);
}

int32_t HcommWriteWithNotifyOnThread(
    ThreadHandle thread, ChannelHandle channel, void* dst, const void* src, uint64_t len, uint32_t remoteNotifyIdx)
{
    if (IS_PLUGIN_HANDLE(channel)) {
        auto* ch = CHANNEL_FROM_HANDLE(channel);
        CHK_PTR_NULL(ch);
        return ch->GetNicOps()->writeWithNotifyOnThread(ch->GetNicCtx(), thread, dst, src, len, remoteNotifyIdx);
    }

    HCCL_INFO(
        "[%s] START. thread[0x%llx], channel[0x%llx], dst[0x%llx], src[0x%llx], len[%llu], remoteNotifyIdx[%u].",
        __func__, thread, channel, dst, src, len, remoteNotifyIdx);

    CHK_PTR_NULL(src);
    CHK_PTR_NULL(dst);

    AddThreadWithTag(thread);

    Thread* const threadPtr = ReinterpretAs<Thread*>(thread);
    CHK_PTR_NULL(threadPtr);

    HcclBuf locBuf{const_cast<void*>(src), len, nullptr};
    HcclBuf rmtBuf{dst, len, nullptr};

    Stream* stream = GetStream(thread);
    CHK_PTR_NULL(stream);

    HcclResult ret
        = HcclRemoteWriteWithNotify(stream, ReinterpretAs<void*>(channel), &rmtBuf, &locBuf, remoteNotifyIdx);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[%s] FAIL. thread[0x%llx], channel[0x%llx], dst[0x%llx], src[0x%llx], len[%llu], remoteNotifyIdx[%u].",
            __func__, thread, channel, dst, src, len, remoteNotifyIdx),
        ret);
    HCCL_INFO("[%s] SUCCESS.", __func__);
    return HCCL_SUCCESS;
}

int32_t HcommWriteReduceWithNotifyOnThread(
    ThreadHandle thread, ChannelHandle channel, void* dst, const void* src, uint64_t count, HcommDataType dataType,
    HcommReduceOp reduceOp, uint32_t remoteNotifyIdx)
{
    if (IS_PLUGIN_HANDLE(channel)) {
        auto* ch = CHANNEL_FROM_HANDLE(channel);
        CHK_PTR_NULL(ch);
        return ch->GetNicOps()->writeReduceWithNotifyOnThread(
            ch->GetNicCtx(), thread, dst, src, count, dataType, reduceOp, remoteNotifyIdx);
    }

    HCCL_INFO(
        "[%s] START. thread[0x%llx], channel[0x%llx], dst[0x%llx], src[0x%llx], count[%llu], dataType[%d], "
        "reduceOp[%d], remoteNotifyIdx[%u].",
        __func__, thread, channel, dst, src, count, dataType, reduceOp, remoteNotifyIdx);

    CHK_PTR_NULL(dst);
    CHK_PTR_NULL(src);

    AddThreadWithTag(thread);

    Thread* const threadPtr = ReinterpretAs<Thread*>(thread);
    CHK_PTR_NULL(threadPtr);

    uint64_t len = count * SIZE_TABLE[dataType];

    HcclResult ret = HCCL_SUCCESS;

    ret = HCCL_E_NOT_SUPPORT;
    (void)len;

    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[%s] FAIL. thread[0x%llx], channel[0x%llx], dst[0x%llx], src[0x%llx], count[%llu], dataType[%d], "
            "reduceOp[%d], remoteNotifyIdx[%u].",
            __func__, thread, channel, dst, src, count, dataType, reduceOp, remoteNotifyIdx),
        ret);
    HCCL_INFO("[%s] SUCCESS.", __func__);
    return HCCL_SUCCESS;
}

int32_t HcommReadOnThread(ThreadHandle thread, ChannelHandle channel, void* dst, const void* src, uint64_t len)
{
    if (IS_PLUGIN_HANDLE(channel)) {
        auto* ch = CHANNEL_FROM_HANDLE(channel);
        CHK_PTR_NULL(ch);
        return ch->GetNicOps()->readOnThread(ch->GetNicCtx(), thread, dst, src, len);
    }

    HCCL_INFO(
        "[%s] START. thread[0x%llx], channel[0x%llx], dst[0x%llx], src[0x%llx], len[%llu].", __func__, thread, channel,
        dst, src, len);

    CHK_PTR_NULL(dst);
    CHK_PTR_NULL(src);

    AddThreadWithTag(thread);

    Thread* const threadPtr = ReinterpretAs<Thread*>(thread);
    CHK_PTR_NULL(threadPtr);

    HcclBuf locBuf{dst, len, nullptr};
    HcclBuf rmtBuf{const_cast<void*>(src), len, nullptr};

    Stream* stream = GetStream(thread);
    CHK_PTR_NULL(stream);

    HcclResult ret = HcclRemoteRead(stream, ReinterpretAs<void*>(channel), &locBuf, &rmtBuf);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[%s] FAIL. thread[0x%llx], channel[0x%llx], dst[0x%llx], src[0x%llx], len[%llu].", __func__, thread,
            channel, dst, src, len),
        ret);
    HCCL_INFO("[%s] SUCCESS.", __func__);
    return HCCL_SUCCESS;
}

int32_t HcommReadReduceOnThread(
    ThreadHandle thread, ChannelHandle channel, void* dst, const void* src, uint64_t count, HcommDataType dataType,
    HcommReduceOp reduceOp)
{
    if (IS_PLUGIN_HANDLE(channel)) {
        auto* ch = CHANNEL_FROM_HANDLE(channel);
        CHK_PTR_NULL(ch);
        return ch->GetNicOps()->readReduceOnThread(ch->GetNicCtx(), thread, dst, src, count, dataType, reduceOp);
    }

    HCCL_INFO(
        "[%s] START. thread[0x%llx], channel[0x%llx], dst[0x%llx], src[0x%llx], count[%llu], dataType[%d], "
        "reduceOp[%d].",
        __func__, thread, channel, dst, src, count, dataType, reduceOp);

    CHK_PTR_NULL(dst);
    CHK_PTR_NULL(src);

    CHK_PRT_RET(
        (IsSupportReduce(dataType, reduceOp) == false),
        HCCL_ERROR(
            "[HcommReadReduceOnThread] Not support reduce, "
            "dst[%p], src[%p], count[%llu], dataType[%d], reduceOp[%d]",
            dst, src, count, dataType, reduceOp),
        HCCL_E_PARA);
    AddThreadWithTag(thread);

    Thread* const threadPtr = ReinterpretAs<Thread*>(thread);
    CHK_PTR_NULL(threadPtr);

    uint64_t len = count * SIZE_TABLE[dataType];

    HcclBuf locBuf{dst, len, nullptr};
    HcclBuf rmtBuf{const_cast<void*>(src), len, nullptr};
    HcclReduceInfo reduceInfo{static_cast<HcclDataType>(dataType), static_cast<HcclReduceOp>(reduceOp)};

    Stream* stream = GetStream(thread);
    CHK_PTR_NULL(stream);

    HcclResult ret = HcclRemoteReadReduce(stream, ReinterpretAs<void*>(channel), &locBuf, &rmtBuf, reduceInfo);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[%s] FAIL. thread[0x%llx], channel[0x%llx], dst[0x%llx], src[0x%llx], count[%llu], dataType[%d], "
            "reduceOp[%d].",
            __func__, thread, channel, dst, src, count, dataType, reduceOp),
        ret);
    HCCL_INFO("[%s] SUCCESS.", __func__);
    return HCCL_SUCCESS;
}

int32_t HcommWriteNbiOnThread(ThreadHandle thread, ChannelHandle channel, void* dst, const void* src, uint64_t len)
{
    auto* ch = CHANNEL_FROM_HANDLE(channel);
    CHK_PTR_NULL(ch);
    return ch->GetNicOps()->writeNbiOnThread(ch->GetNicCtx(), thread, dst, src, len);
}

int32_t HcommWriteNbi(ChannelHandle channel, void* dst, const void* src, uint64_t len)
{
    auto* ch = CHANNEL_FROM_HANDLE(channel);
    CHK_PTR_NULL(ch);
    return ch->GetNicOps()->writeNbi(ch->GetNicCtx(), dst, src, len);
}

int32_t HcommWriteWithNotifyNbiOnThread(
    ThreadHandle thread, ChannelHandle channel, void* dst, const void* src, uint64_t len, uint32_t remoteNotifyIdx)
{
    auto* ch = CHANNEL_FROM_HANDLE(channel);
    CHK_PTR_NULL(ch);
    return ch->GetNicOps()->writeWithNotifyNbiOnThread(ch->GetNicCtx(), thread, dst, src, len, remoteNotifyIdx);
}

int32_t
HcommWriteWithNotifyNbi(ChannelHandle channel, void* dst, const void* src, uint64_t len, uint32_t remoteNotifyIdx)
{
    auto* ch = CHANNEL_FROM_HANDLE(channel);
    CHK_PTR_NULL(ch);
    return ch->GetNicOps()->writeWithNotifyNbi(ch->GetNicCtx(), dst, src, len, remoteNotifyIdx);
}

int32_t HcommReadNbiOnThread(ThreadHandle thread, ChannelHandle channel, void* dst, const void* src, uint64_t len)
{
    auto* ch = CHANNEL_FROM_HANDLE(channel);
    CHK_PTR_NULL(ch);
    return ch->GetNicOps()->readNbiOnThread(ch->GetNicCtx(), thread, dst, src, len);
}

int32_t HcommReadNbi(ChannelHandle channel, void* dst, const void* src, uint64_t len)
{
    auto* ch = CHANNEL_FROM_HANDLE(channel);
    CHK_PTR_NULL(ch);
    return ch->GetNicOps()->readNbi(ch->GetNicCtx(), dst, src, len);
}

int32_t HcommChannelNotifyRecordOnThread(ThreadHandle thread, ChannelHandle channel, uint32_t remoteNotifyIdx)
{
    if (IS_PLUGIN_HANDLE(channel)) {
        auto* ch = CHANNEL_FROM_HANDLE(channel);
        CHK_PTR_NULL(ch);
        return ch->GetNicOps()->notifyRecordOnThread(ch->GetNicCtx(), thread, remoteNotifyIdx);
    }
    HCCL_INFO(
        "[%s] START. thread[0x%llx], channel[0x%llx], remoteNotifyIdx[%u].", __func__, thread, channel,
        remoteNotifyIdx);

    HcclResult ret = HCCL_SUCCESS;
    DevType devType;
    CHK_RET(hrtGetDeviceType(devType));
    if (devType == DevType::DEV_TYPE_950 || devType == DevType::DEV_TYPE_960
        || (thread == 0 && devType == DevType::DEV_TYPE_910B)) {
        auto* const channelPtr = ReinterpretAs<hcomm::Channel*>(channel);
        CHK_PTR_NULL(channelPtr);
        ret = channelPtr->NotifyRecord(remoteNotifyIdx);
    } else { // Non-950 devices use thread-based notify.
        AddThreadWithTag(thread);

        Thread* threadPtr = ReinterpretAs<Thread*>(thread);
        CHK_PTR_NULL(threadPtr);

        Stream* stream = GetStream(thread);
        CHK_PTR_NULL(stream);

        ret = HcclRemoteNotifyRecord(stream, ReinterpretAs<void*>(channel), remoteNotifyIdx);
    }
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[%s] FAIL. thread[0x%llx], channel[0x%llx], remoteNotifyIdx[%u].", __func__, thread, channel,
            remoteNotifyIdx),
        ret);
    HCCL_INFO("[%s] SUCCESS.", __func__);
    return HCCL_SUCCESS;
}

int32_t HcommChannelNotifyRecord(ChannelHandle channel, uint32_t remoteNotifyIdx)
{
    auto* ch = CHANNEL_FROM_HANDLE(channel);
    CHK_PTR_NULL(ch);
    return ch->GetNicOps()->notifyRecord(ch->GetNicCtx(), remoteNotifyIdx);
}

int32_t
HcommChannelNotifyWaitOnThread(ThreadHandle thread, ChannelHandle channel, uint32_t localNotifyIdx, uint32_t timeOut)
{
    if (IS_PLUGIN_HANDLE(channel)) {
        auto* ch = CHANNEL_FROM_HANDLE(channel);
        CHK_PTR_NULL(ch);
        return ch->GetNicOps()->notifyWaitOnThread(ch->GetNicCtx(), thread, localNotifyIdx, timeOut);
    }
    HCCL_INFO(
        "[%s] START. thread[0x%llx], channel[0x%llx], localNotifyIdx[%u], timeOut[%u].", __func__, thread, channel,
        localNotifyIdx, timeOut);

    HcclResult ret = HCCL_SUCCESS;
    DevType devType;
    CHK_RET(hrtGetDeviceType(devType));
    if (devType == DevType::DEV_TYPE_950 || devType == DevType::DEV_TYPE_960
        || (thread == 0 && devType == DevType::DEV_TYPE_910B)) {
        auto* const channelPtr = ReinterpretAs<hcomm::Channel*>(channel);
        CHK_PTR_NULL(channelPtr);
        ret = channelPtr->NotifyWait(localNotifyIdx, timeOut);
    } else { // Non-950 devices use thread-based notify.
        AddThreadWithTag(thread);

        Thread* threadPtr = ReinterpretAs<Thread*>(thread);
        CHK_PTR_NULL(threadPtr);

        Stream* stream = GetStream(thread);
        CHK_PTR_NULL(stream);

        ret = HcclRemoteNotifyWait(stream, ReinterpretAs<void*>(channel), localNotifyIdx, timeOut);
    }
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[%s] FAIL. thread[0x%llx], channel[0x%llx], localNotifyIdx[%u], timeOut[%u]s.", __func__, thread, channel,
            localNotifyIdx, timeOut),
        ret);
    HCCL_INFO("[%s] SUCCESS.", __func__);
    return HCCL_SUCCESS;
}

int32_t HcommChannelNotifyWait(ChannelHandle channel, uint32_t localNotifyIdx, uint32_t timeOut)
{
    auto* ch = CHANNEL_FROM_HANDLE(channel);
    CHK_PTR_NULL(ch);
    return ch->GetNicOps()->notifyWait(ch->GetNicCtx(), localNotifyIdx, timeOut);
}

HcclResult CommFence(ThreadHandle thread, ChannelHandle channel) // 控制前后的任务保序
{
    HCCL_DEBUG("[CommFence] thread[0x%llx], channel[0x%llx].", thread, channel);

    Stream* stream = GetStream(thread);
    CHK_PTR_NULL(stream);

    return HcclRemoteFence(stream, ReinterpretAs<void*>(channel), false);
}

int32_t HcommSetLaunchMode(const char* launchTag, HcommLaunchMode mode)
{
    HCCL_DEBUG("HcommSetLaunchMode launchTag[%s]", launchTag);
    return g_threadLaunchCtx.SetLaunchMode(launchTag, mode);
}

int32_t HcommBatchModeStart(const char* batchTag) { return HcommSetLaunchMode(batchTag, HCOMM_LAUNCH_MODE_BATCH); }

int32_t HcommBatchModeEnd(const char* batchTag) { return HcommSetLaunchMode(batchTag, HCOMM_LAUNCH_MODE_EAGER); }

int32_t
HcommThreadRegisterDfx(ThreadHandle thread, std::function<HcclResult(u32, u32, const Hccl::TaskParam&, u64)> callback)
{
    Thread* threadPtr = ReinterpretAs<Thread*>(thread);
    CHK_PTR_NULL(threadPtr);
    CHK_RET(threadPtr->SetAddTaskInfoCallback(callback));
    return HCCL_SUCCESS;
}

int32_t HcommThreadRegisterCheckExecStatus(ThreadHandle thread, std::function<HcclResult(bool)> callback)
{
    Thread* threadPtr = ReinterpretAs<Thread*>(thread);
    CHK_PTR_NULL(threadPtr);
    CHK_RET(threadPtr->SetCheckExecStatusCallback(callback));
    return HCCL_SUCCESS;
}

int32_t
HcommDpuChannelRegisterDfx(ChannelHandle channel, std::function<HcclResult(const Hccl::TaskParam&, u64)> callback)
{
    auto* const hostCpuRoceChannelPtr = ReinterpretAs<hcomm::HostCpuRoceChannel*>(channel);
    CHK_PTR_NULL(hostCpuRoceChannelPtr);
    CHK_RET(hostCpuRoceChannelPtr->SetDfxCallback(callback));
    return HCCL_SUCCESS;
}

int32_t HcommAcquireComm(const char* commId)
{
    CHK_PTR_NULL(commId);
    std::shared_ptr<hccl::hcclComm> hcclComm;
    HcclGetCommHandle(commId, hcclComm);
    CHK_PRT_RET(hcclComm == nullptr, HCCL_ERROR("%s hcclComm is null, commId[%s]", __func__, commId), HCCL_E_PTR);
    CHK_RET(hcclComm->SetCommDispatcherCtx());
    return HCCL_SUCCESS;
}

int32_t HcommReleaseComm(const char* commId)
{
    CHK_PTR_NULL(commId);
    HCCL_INFO("%s not support, commId[%s], do nothing", __func__, commId);
    return HCCL_SUCCESS;
}

int32_t HcommFenceOnThread(ThreadHandle thread)
{
    HCCL_INFO("[%s] START. thread[0x%llx].", __func__, thread);
    (void)thread;
    HcclResult ret = HcommFlushV2();
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[%s] FAIL. thread[0x%llx].", __func__, thread), ret);
    HCCL_INFO("[%s] SUCCESS.", __func__);
    return HCCL_SUCCESS;
}

int32_t HcommChannelDrainOnThread(ThreadHandle thread, ChannelHandle channel)
{
    HCCL_INFO("[%s] START. thread[0x%llx], channel[0x%llx].", __func__, thread, channel);

    int32_t ret = HCCL_SUCCESS;
    if (IS_PLUGIN_HANDLE(channel)) {
        auto* ch = CHANNEL_FROM_HANDLE(channel);
        CHK_PTR_NULL(ch);
        ret = ch->GetNicOps()->drainOnThread(ch->GetNicCtx(), thread);
    } else {
        DevType devType;
        CHK_RET(hrtGetDeviceType(devType));
        if (devType == DevType::DEV_TYPE_950 || devType == DevType::DEV_TYPE_960) {
            auto* ch = CHANNEL_FROM_HANDLE(channel);
            CHK_PTR_NULL(ch);
            ret = ch->GetNicOps()->drainOnThread(ch->GetNicCtx(), thread);
        } else {
            Thread* const threadPtr = ReinterpretAs<Thread*>(thread);
            CHK_PTR_NULL(threadPtr);
            Stream* stream = GetStream(thread);
            CHK_PTR_NULL(stream);
            ret = HcclRemoteDrain(stream, ReinterpretAs<void*>(channel));
        }
    }
    CHK_PRT_RET(
        ret != HCCL_SUCCESS, HCCL_ERROR("[%s] FAIL. thread[0x%llx], channel[0x%llx].", __func__, thread, channel), ret);

    HCCL_INFO("[%s] SUCCESS.", __func__);
    return HCCL_SUCCESS;
}

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus
int32_t HcommFlush() { return HcommFenceOnThread(0); }
#ifdef __cplusplus
}
#endif // __cplusplus

int32_t HcommChannelFenceOnThread(ThreadHandle thread, ChannelHandle channel)
{
    auto* ch = CHANNEL_FROM_HANDLE(channel);
    CHK_PTR_NULL(ch);
    return ch->GetNicOps()->fenceOnThread(ch->GetNicCtx(), thread);
}

int32_t HcommChannelFence(ChannelHandle channel)
{
    auto* ch = CHANNEL_FROM_HANDLE(channel);
    CHK_PTR_NULL(ch);
    return ch->GetNicOps()->fence(ch->GetNicCtx());
}

HcclResult
HcclDfxRegOpInfo([[maybe_unused]] HcclComm comm, [[maybe_unused]] void* hcclDfxOpInfo) // 兼容性接口，后续删除
{
    HCCL_WARNING("%s not support", __func__);
    return HCCL_SUCCESS;
}

HcclResult HcclDfxRegOpInfoByCommId(char* commId, void* hcclDfxOpInfo)
{
    EXCEPTION_HANDLE_BEGIN
    HcclComm commHandle = nullptr;
    CHK_RET(HcomGetCommHandleByGroup(commId, &commHandle));
    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm*>(commHandle);
    CHK_PRT_RET(hcclComm == nullptr, HCCL_ERROR("%s hcclComm is null, commId[%s]", __func__, commId), HCCL_E_PTR);
    CHK_PRT_RET(hcclDfxOpInfo == nullptr, HCCL_ERROR("[%s] hcclDfxOpInfo is null", __func__), HCCL_E_PTR);
    HcclDfxOpInfo* dfxOpInfo = static_cast<HcclDfxOpInfo*>(hcclDfxOpInfo);
    CHK_PTR_NULL(dfxOpInfo);
    DevType devType;
    CHK_RET(hrtGetDeviceType(devType));
    if (!hcclComm->IsCommunicatorV2() && devType == DevType::DEV_TYPE_910B) {
        return HCCL_SUCCESS;
    }
    if (!hcclComm->IsCommunicatorV2()) {
        HCCL_ERROR("[%s]comm is NOT_SUPPORT", __func__);
        return HCCL_E_NOT_SUPPORT;
    }
    hccl::CollComm* collComm = hcclComm->GetCollComm();
    CHK_PTR_NULL(collComm);

    dfxOpInfo->beginTime = hrtMsprofSysCycleTime();

    // HcclDfxOpInfo转为DfxOpInfo
    auto dfxOpInfoOnce = ConvertToDfxOpInfo(*dfxOpInfo);
    CHK_SMART_PTR_NULL(dfxOpInfoOnce);
    dfxOpInfoOnce->comm_ = static_cast<void*>(collComm);
    dfxOpInfoOnce->isIndop_ = true;
    dfxOpInfoOnce->groupName_ = collComm->GetCommId();
    dfxOpInfoOnce->opIndex_ = collComm->UpdateIndex();
    dfxOpInfoOnce->rankSize_ = collComm->GetRankSize();
    // 单算子模式，暂时覆盖opTag
    dfxOpInfoOnce->op_.opTag = collComm->GetCommId();
    dfxOpInfoOnce->op_.myRank = static_cast<Hccl::RankId>(collComm->GetMyRankId());
    dfxOpInfoOnce->engine = dfxOpInfo->engine;
    HcclCommDfx* hcclCommDfx = collComm->GetHcclCommDfx();
    CHK_PTR_NULL(hcclCommDfx);
    CHK_RET(hcclCommDfx->UpdateProfStat());
    CHK_RET(hcclCommDfx->SetCurrDfxOpInfo(dfxOpInfoOnce));
    bool isOpBase
        = dfxOpInfoOnce->op_.opMode == Hccl::OpMode::OPBASE || dfxOpInfoOnce->op_.opMode == Hccl::OpMode::ACLGRAPH;
    bool isCached
        = dfxOpInfoOnce->op_.opMode == Hccl::OpMode::OFFLOAD || dfxOpInfoOnce->op_.opMode == Hccl::OpMode::ACLGRAPH;
    Hccl::DfxProfilingHandler::GetInstance().SetOpModeFlags(isOpBase, isCached);
    HCCL_INFO(
        "[%s] Register DfxOpInfo success, opMode[%d], isOpBase[%d], isCached[%d], DfxOpInfo: %s", __func__,
        dfxOpInfoOnce->op_.opMode, isOpBase, isCached, dfxOpInfoOnce->Describe().c_str());
    EXCEPTION_HANDLE_END
    return HCCL_SUCCESS;
}

HcclResult HcclProfilingReportOp(HcclComm comm, uint64_t beginTime)
{
    HCCL_INFO("[%s] START, comm[%p].", __func__, comm);
    CHK_PRT_RET(comm == nullptr, HCCL_ERROR("[%s] comm is null", __func__), HCCL_E_PTR);
    auto* hcclComm = static_cast<hccl::hcclComm*>(comm);
    CHK_PTR_NULL(hcclComm);
    DevType devType;
    CHK_RET(hrtGetDeviceType(devType));
    if (devType == DevType::DEV_TYPE_910B && !hcclComm->IsCommunicatorV2()) {
        return HCCL_SUCCESS;
    }
    if (!hcclComm->IsCommunicatorV2()) {
        HCCL_ERROR("[%s] comm is NOT_SUPPORT", __func__);
        return HCCL_E_NOT_SUPPORT;
    }
    hccl::CollComm* collComm = hcclComm->GetCollComm();
    CHK_PTR_NULL(collComm);
    HcclCommDfx* hcclCommDfx = collComm->GetHcclCommDfx();
    CHK_PTR_NULL(hcclCommDfx);
    HCCL_INFO(
        "[%s] Report All Tasks Info, comm[%p], hcclCommDfx[%p] GetMirrorTaskManager[%p].", __func__, comm, hcclCommDfx,
        hcclCommDfx->GetMirrorTaskManager());
    auto* mirrorTaskMgr = hcclCommDfx->GetMirrorTaskManager();
    CHK_PTR_NULL(mirrorTaskMgr);
    if (mirrorTaskMgr->GetCurrDfxOpInfo() == nullptr) {
        HCCL_INFO("[%s] commId[%s] currDfxOpInfo is null, skip report.", __func__, collComm->GetCommId().c_str());
        return HCCL_SUCCESS;
    }
    // 单算子模式暂时默认true
    bool isOpBaseMode = false;
    bool isCached = false;
    CHK_RET(hcclCommDfx->GetOpModeFlags(isOpBaseMode, isCached));
    CHK_RET(hcclCommDfx->ReportAllTasks(isCached));
    CHK_RET(hcclCommDfx->ReportOp(beginTime, isCached, isOpBaseMode));
    HCCL_INFO("[%s] SUCCESS.", __func__);
    return HCCL_SUCCESS;
}

HcclResult HcclReportAicpuKernel(HcclComm comm, uint64_t beginTime, char* kernelName)
{
    HCCL_INFO("[%s] START, comm[%p].", __func__, comm);
    CHK_PRT_RET(comm == nullptr, HCCL_ERROR("[%s] comm is null", __func__), HCCL_E_PTR);
    CHK_PRT_RET(kernelName == nullptr, HCCL_ERROR("[%s] kernelName is null", __func__), HCCL_E_PTR);
    // 填入remoteRankId
    auto hcclComm = static_cast<hccl::hcclComm*>(comm);
    CHK_PTR_NULL(hcclComm);
    if (!hcclComm->IsCommunicatorV2()) {
        return HCCL_SUCCESS;
    }
    hccl::CollComm* collComm = hcclComm->GetCollComm();
    CHK_PTR_NULL(collComm);
    HcclCommDfx* hcclCommDfx = collComm->GetHcclCommDfx();
    CHK_PTR_NULL(hcclCommDfx);

    auto* mirrorTaskMgr = hcclCommDfx->GetMirrorTaskManager();
    CHK_PTR_NULL(mirrorTaskMgr);
    if (mirrorTaskMgr->GetCurrDfxOpInfo() == nullptr) {
        HCCL_INFO("[%s] commId[%s] currDfxOpInfo is null, skip report.", __func__, collComm->GetCommId().c_str());
        return HCCL_SUCCESS;
    }

    std::string kernelNameStr(kernelName);
    uint32_t threadId = SalGetTid();
    bool isOpBaseMode = false;
    bool isCached = false;
    CHK_RET(hcclCommDfx->GetOpModeFlags(isOpBaseMode, isCached));
    CHK_RET(hcclCommDfx->ReportKernel(beginTime, collComm->GetCommId(), kernelNameStr, threadId, isCached));

    Hccl::TaskParam taskParam{};
    taskParam.beginTime = beginTime;
    taskParam.taskType = Hccl::TaskParamType::TASK_AICPU_KERNEL;
    taskParam.endTime = Hccl::DfxDlProfFunction::GetInstance().dlMsprofSysCycleTime();
    uint32_t taskId = INVALID_UINT;
    uint32_t streamId = INVALID_UINT;
    CHK_RET(hrtGetTaskIdAndStreamID(taskId, streamId));
    HCCL_INFO("[%s] taskId[%u], streamId[%u].", __func__, taskId, streamId);
    hcclCommDfx->SetAicpuTaskIdAndStreamId(taskId, streamId);
    CHK_RET(hcclCommDfx->AddTaskInfoCallback(streamId, taskId, taskParam, DFX_INVALID_U64));
    HCCL_INFO("[HcclReportAicpuKernel] HcclReportAicpuKernel success");
    return HCCL_SUCCESS;
}

extern HcclResult HcclReportAivKernel(HcclComm comm, uint64_t beginTime)
{
    HCCL_INFO("[%s] START, comm[%p].", __func__, comm);
    CHK_PRT_RET(comm == nullptr, HCCL_ERROR("[%s] comm is null", __func__), HCCL_E_PTR);
    auto hcclComm = static_cast<hccl::hcclComm*>(comm);
    CHK_PTR_NULL(hcclComm);
    if (!hcclComm->IsCommunicatorV2()) {
        HCCL_ERROR("[%s] comm is not supported", __func__);
        return HCCL_E_NOT_SUPPORT;
    }
    hccl::CollComm* collComm = hcclComm->GetCollComm();
    CHK_PTR_NULL(collComm);
    HcclCommDfx* hcclCommDfx = collComm->GetHcclCommDfx();
    CHK_PTR_NULL(hcclCommDfx);

    Hccl::TaskParam taskParam{};
    taskParam.beginTime = beginTime;
    taskParam.taskType = Hccl::TaskParamType::TASK_AIV;
    taskParam.endTime = Hccl::DfxDlProfFunction::GetInstance().dlMsprofSysCycleTime();
    taskParam.isMaster = true;
    uint32_t taskId = INVALID_UINT;
    uint32_t streamId = INVALID_UINT;
    CHK_RET(hrtGetTaskIdAndStreamID(taskId, streamId));
    CHK_RET(hcclCommDfx->AddTaskInfoCallback(streamId, taskId, taskParam, DFX_INVALID_U64));
    HCCL_INFO("[HcclReportAivKernel] HcclReportAivKernel success");
    return HCCL_SUCCESS;
}

uint64_t HcommGetProfilingSysCycleTime()
{
    DevType devType = DevType::DEV_TYPE_COUNT;
    HcclResult ret = hrtGetDeviceType(devType);
    if (ret != HCCL_SUCCESS) {
        HCCL_WARNING("[%s] hrtGetDeviceType failed, ret[%d], return 0.", __func__, ret);
        return 0;
    }
    if (devType != DevType::DEV_TYPE_950 && devType != DevType::DEV_TYPE_960) {
        return hrtMsprofSysCycleTime();
    }
    return Hccl::DfxDlProfFunction::GetInstance().dlMsprofSysCycleTime();
}

int32_t
HcommChannelNotifyWaitOnThreadWithDefaultTimeout(ThreadHandle thread, ChannelHandle channel, uint32_t localNotifyIdx)
{
    if (IS_PLUGIN_HANDLE(channel)) {
        auto* ch = CHANNEL_FROM_HANDLE(channel);
        CHK_PTR_NULL(ch);
        return ch->GetNicOps()->notifyWaitOnThreadWithDefaultTimeout(ch->GetNicCtx(), thread, localNotifyIdx);
    }
    HCCL_ERROR(
        "[%s] thread[0x%llx], channel[0x%llx], localNotifyIdx[%u] not support in cpu.", __func__, thread, channel,
        localNotifyIdx);
    return HCCL_E_NOT_SUPPORT;
}

int32_t HcommThreadNotifyWaitOnThreadWithDefaultTimeout(ThreadHandle thread, uint32_t notifyIdx)
{
    HCCL_INFO("[%s] data cpu START. thread[0x%llx], notifyIdx[%u].", __func__, thread, notifyIdx);

    uint32_t notifyWaitTimeout;
    g_threadLaunchCtx.GetNotifyWaitTimeOut(notifyWaitTimeout);

    HCCL_DEBUG("[%s] data cpu Using default timeout: %u s", __func__, notifyWaitTimeout);

    int32_t ret = HcommThreadNotifyWaitOnThread(thread, notifyIdx, notifyWaitTimeout);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR(
            "[%s] data cpu HcommThreadNotifyWaitOnThread FAILED. thread[0x%llx], notifyIdx[%u], ret[%d]", __func__,
            thread, notifyIdx, ret);
        return ret;
    }

    HCCL_INFO("[%s] data cpu SUCCESS.", __func__);
    return HCCL_SUCCESS;
}
