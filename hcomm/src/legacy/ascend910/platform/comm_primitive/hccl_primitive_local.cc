/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hccl_primitive_local.h"
#include "log.h"
#include "mem_device_pub.h"
#include "hccl_dispatcher_ctx.h"
#include "dispatcher_task_types.h"
#include "dispatcher_aicpu_pub.h"
#include "local_notify.h"
#include "dispatcher_ctx.h"

HcclResult GetPubDispatcher(hccl::DispatcherPub** dispatcherPtr)
{
    DispatcherCtxPtr ctx = nullptr;
    CHK_RET(AcquireDispatcherCtx(&ctx));
    CHK_PTR_NULL(ctx);
    hccl::DispatcherCtx* ctx_temp = reinterpret_cast<hccl::DispatcherCtx*>(ctx);
    CHK_PTR_NULL(ctx_temp->GetDispatcher());
    *dispatcherPtr = reinterpret_cast<hccl::DispatcherPub*>(ctx_temp->GetDispatcher());
    CHK_PTR_NULL(*dispatcherPtr);
    return HCCL_SUCCESS;
}

HcclResult HcclLocalCopy(StreamHandle streamHandle, HcclBuf* dst, HcclBuf* src)
{
    CHK_PTR_NULL(src);
    CHK_PTR_NULL(dst);
    CHK_PTR_NULL(streamHandle);
    hccl::DeviceMem srcDevMem(src->addr, src->len);
    hccl::DeviceMem dstDevMem(dst->addr, dst->len);
    HCCL_INFO(
        "[hcclLocalCopy] dst addr[%p], size[%llu], src addr[%p], size[%llu]", dst->addr, dst->len, src->addr, src->len);

    hccl::Stream* stream = reinterpret_cast<hccl::Stream*>(streamHandle);

    hccl::DispatcherPub* dispatcherPtr = nullptr;
    CHK_RET(GetPubDispatcher(&dispatcherPtr));
    HCCL_INFO("[%s] dispatcherPtr[%p]", __func__, (void*)dispatcherPtr);
    return dispatcherPtr->MemcpyAsync(dstDevMem, srcDevMem, *stream);
}

HcclResult HcclLocalCopyReduce(StreamHandle streamHandle, HcclBuf* dst, HcclBuf* src, HcclReduceInfo reduceInfo)
{
    CHK_PTR_NULL(src);
    CHK_PTR_NULL(dst);
    CHK_PTR_NULL(streamHandle);

    HCCL_INFO(
        "[HcclLocalCopyReduce] dst ptr[%p], size[%llu], src ptr[%p], size[%llu], datatype[%d], reduceOp[%d]", dst->addr,
        dst->len, src->addr, src->len, static_cast<int>(reduceInfo.dataType), static_cast<int>(reduceInfo.reduceOp));

    hccl::DispatcherPub* dispatcherPtr = nullptr;
    CHK_RET(GetPubDispatcher(&dispatcherPtr));

    hccl::Stream* stream = reinterpret_cast<hccl::Stream*>(streamHandle);

    return dispatcherPtr->InlineReduceAsync(
        src->addr, src->len / SIZE_TABLE[reduceInfo.dataType], reduceInfo.dataType, reduceInfo.reduceOp, *stream,
        dst->addr, INVALID_VALUE_RANKID, hccl::LinkType::LINK_ONCHIP);
}

HcclResult HcclLocalLaunchTaskExtend(aclrtStream& stream, std::vector<aclrtStream>& subStreams)
{
    CHK_PTR_NULL(stream);
    hccl::Stream stream_temp(stream, false);

    hccl::DispatcherPub* dispatcherPtr = nullptr;
    hccl::DispatcherCtx* ctx_temp = reinterpret_cast<hccl::DispatcherCtx*>(GetDispatcherCtx());
    CHK_PTR_NULL(ctx_temp);
    CHK_PTR_NULL(ctx_temp->GetDispatcher());
    dispatcherPtr = reinterpret_cast<hccl::DispatcherPub*>(ctx_temp->GetDispatcher());

    if (ctx_temp->GetLaunchTaskCallback() != nullptr) {
        CHK_RET(ctx_temp->GetLaunchTaskCallback()(dispatcherPtr, stream_temp));
    }

    std::vector<hccl::Stream> subStreams_temp;
    for (auto& s : subStreams) {
        CHK_PTR_NULL(s);
        subStreams_temp.push_back(*(reinterpret_cast<hccl::Stream*>(s)));
    }

    return dispatcherPtr->LaunchTasksEx(stream_temp, subStreams_temp);
}

HcclResult
HcclLocalInitTask(aclrtStream stream, const bool enableCache, const std::string& key, bool useGraphConstructorV2)
{
    CHK_PTR_NULL(stream);

    hccl::DispatcherPub* dispatcherPtr = nullptr;
    hccl::DispatcherCtx* ctx_temp = reinterpret_cast<hccl::DispatcherCtx*>(GetDispatcherCtx());
    CHK_PTR_NULL(ctx_temp);
    CHK_PTR_NULL(ctx_temp->GetDispatcher());
    dispatcherPtr = reinterpret_cast<hccl::DispatcherPub*>(ctx_temp->GetDispatcher());

    HCCL_INFO(
        "InitTask enableCache[%d], key[%s], useGraphConstructorV2[%d]", enableCache, key.c_str(),
        useGraphConstructorV2);

    CHK_RET(dispatcherPtr->ResetGraphCtx(enableCache, key, useGraphConstructorV2));

    hccl::Stream stream_temp(stream, false);

    if (ctx_temp->GetInitTaskCallback() != nullptr) {
        CHK_RET(ctx_temp->GetInitTaskCallback()(dispatcherPtr, stream_temp));
    }
    return HCCL_SUCCESS;
}

HcclResult HcclLocalNotifyRecord(StreamHandle streamHandle, aclrtNotify notify)
{
    CHK_PTR_NULL(streamHandle);
    CHK_PTR_NULL(notify);
    hccl::DispatcherPub* dispatcherPtr = nullptr;
    CHK_RET(GetPubDispatcher(&dispatcherPtr));

    hccl::Stream* stream = reinterpret_cast<hccl::Stream*>(streamHandle);
    hccl::LocalNotify* localNotify = reinterpret_cast<hccl::LocalNotify*>(notify);

    return dispatcherPtr->SignalRecord(
        localNotify->ptr(), *stream, INVALID_VALUE_RANKID, INVALID_U64, INVALID_VALUE_STAGE, true, INVALID_U64,
        localNotify->notifyId_);
}

HcclResult HcclLocalNotifyWait(StreamHandle streamHandle, aclrtNotify notify, const uint32_t timeOut)
{
    CHK_PTR_NULL(streamHandle);
    CHK_PTR_NULL(notify);

    hccl::DispatcherPub* dispatcherPtr = nullptr;
    CHK_RET(GetPubDispatcher(&dispatcherPtr));

    hccl::LocalNotify* localNotify = reinterpret_cast<hccl::LocalNotify*>(notify);
    hccl::Stream* stream = reinterpret_cast<hccl::Stream*>(streamHandle);
    return dispatcherPtr->SignalWait(
        localNotify->ptr(), *stream, INVALID_VALUE_RANKID, INVALID_VALUE_RANKID, INVALID_VALUE_STAGE, true,
        localNotify->notifyId_, timeOut);
}

HcclResult HcclTaskPrepare(char* key, uint32_t keyLen) // host ffts+使用
{
    bool enableCache = false;
    std::string keyStr = "temp_key";
    if (key != nullptr && keyLen != 0) {
        enableCache = true;
        keyStr = std::string(key, keyLen);
        HCCL_DEBUG("[HcclTaskPrepare]key[%s], keyLen[%u]", key, keyLen);
    } else {
        HCCL_DEBUG("[HcclTaskPrepare]disable cache, key[%p], keyLen[%u]", key, keyLen);
    }

    hccl::DispatcherPub* dispatcherPtr = nullptr;
    CHK_RET(GetPubDispatcher(&dispatcherPtr));

    return dispatcherPtr->ResetGraphCtx(enableCache, keyStr, true);
}

HcclResult HcclTaskLaunch(hccl::Stream* streams, uint32_t streamNum) // host ffts+或aicpu stars使用"
{
    CHK_PTR_NULL(streams);
    CHK_PRT_RET(streamNum < 1, HCCL_ERROR("[HcclTaskLaunch]threadNum is less than 1"), HCCL_E_PARA);
    hccl::Stream mainStream = streams[0];
    std::vector<hccl::Stream> subStreams;
    for (uint32_t i = 1; i < streamNum; i++) {
        subStreams.push_back(streams[i]);
    }

    hccl::DispatcherPub* dispatcherPtr = nullptr;
    CHK_RET(GetPubDispatcher(&dispatcherPtr));

    return dispatcherPtr->LaunchTasksEx(mainStream, subStreams);
}

HcclResult HcclLocalBareNotifyRecord(StreamHandle streamHandle, uint64_t dstNotifyId)
{
    CHK_PTR_NULL(streamHandle);
    hccl::DispatcherPub* dispatcherPtr = nullptr;
    CHK_RET(GetPubDispatcher(&dispatcherPtr));

    hccl::Stream* stream = reinterpret_cast<hccl::Stream*>(streamHandle);

    return dispatcherPtr->SignalRecord(*stream, dstNotifyId);
}

HcclResult HcclLocalBareNotifyWait(StreamHandle streamHandle, uint64_t notifyId, uint32_t timeOut)
{
    CHK_PTR_NULL(streamHandle);

    hccl::DispatcherPub* dispatcherPtr = nullptr;
    CHK_RET(GetPubDispatcher(&dispatcherPtr));

    hccl::Stream* stream = reinterpret_cast<hccl::Stream*>(streamHandle);
    HCCL_INFO("%s notifyId[%llu]", __func__, notifyId);
    return dispatcherPtr->SignalWait(*stream, notifyId, timeOut);
}

HcclResult HcclTaskClear(std::string key) // host ffts+使用
{
    hccl::DispatcherPub* dispatcherPtr = nullptr;
    CHK_RET(GetPubDispatcher(&dispatcherPtr));
    return dispatcherPtr->ResetGraphCtx(false, key, true);
}
