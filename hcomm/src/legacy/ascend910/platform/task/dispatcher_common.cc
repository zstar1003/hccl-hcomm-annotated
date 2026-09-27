/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "dispatcher_pub.h"
#ifndef HCCD
#include "dispatcher_graph_pub.h"
#endif
#include "dispatcher_virtural_pub.h"
#include "dispatcher_aicpu_pub.h"
#include "dispatcher.h"
#include "externalinput_pub.h"
#include "adapter_hal.h"

using namespace hccl;
using FftsCounterCallBack = HcclResult (*)(const HcclDispatcher&, Stream&);
FftsCounterCallBack g_InitTaskCallback = nullptr;
FftsCounterCallBack g_LaunchTaskCallback = nullptr;
void RegisterInitTaskCallBack(HcclResult (*p1)(const HcclDispatcher&, hccl::Stream&)) { g_InitTaskCallback = p1; }

void RegisterLaunchTaskCallBack(HcclResult (*p1)(const HcclDispatcher&, hccl::Stream&)) { g_LaunchTaskCallback = p1; }

HcclResult RegisterLoadTaskCallBack(
    HcclDispatcher dispatcherPtr, void* userPtr, void (*p1)(void* userPtr, void* param, u32 length))

{
    CHK_PTR_NULL(dispatcherPtr);
    reinterpret_cast<DispatcherPub*>(dispatcherPtr)->RegLoadTaskCallBack(userPtr, p1);
    return HCCL_SUCCESS;
}

HcclResult ForceProfOn(HcclDispatcher& dispatcherPtr, bool isForce)
{
    CHK_PTR_NULL(dispatcherPtr);
    reinterpret_cast<DispatcherPub*>(dispatcherPtr)->ForceProf(isForce);
    return HCCL_SUCCESS;
}
HcclResult HcclDispatcherInit(DispatcherType type, const s32 devicePhyId, HcclDispatcher* dispatcher)
{
    CHK_RET(DlProfFunc::GetInstance().DlProfFunctionInit());
    CHK_PTR_NULL(dispatcher);
    u32 deviceLogicId = INVALID_UINT;
    if (static_cast<s32>(devicePhyId) != HOST_DEVICE_ID) {
        CHK_RET(hrtGetDeviceIndexByPhyId(devicePhyId, deviceLogicId));
    } else {
        deviceLogicId = devicePhyId;
    }

    DispatcherPub* pDispatcher = nullptr;
    if (type == DispatcherType::DISPATCHER_NORMAL) {
        if (GetExternalInputHcclEnableFfts()) {
// DispatcherGraph 不编到device侧的so里面
#ifndef HCCD
            pDispatcher = new (std::nothrow) DispatcherGraph(deviceLogicId);
#endif
        } else {
            pDispatcher = new (std::nothrow) DispatcherPub(deviceLogicId);
        }
    } else if (type == DispatcherType::DISPATCHER_VIRTURAL) {
        pDispatcher = new (std::nothrow) DispatcherVirtural(deviceLogicId);
    } else {
        HCCL_ERROR("Not support the dispatcher type[%d]", type);
        return HCCL_E_NOT_SUPPORT;
    }
    CHK_PTR_NULL(pDispatcher);
    HcclResult ret = pDispatcher->Init();
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("Dispatcher init failed, type[%d]", type);
        delete pDispatcher;
        pDispatcher = nullptr;
        return ret;
    }
    *dispatcher = pDispatcher;
    return HCCL_SUCCESS;
}

HcclResult HcclDispatcherDestroy(HcclDispatcher dispatcherPtr)
{
    if (dispatcherPtr != nullptr) {
        DispatcherPub* dispatcher = reinterpret_cast<DispatcherPub*>(dispatcherPtr);
        delete dispatcher;
        dispatcherPtr = nullptr;
    }
    return HCCL_SUCCESS;
}

HcclResult HcclSetGlobalWorkSpace(HcclDispatcher dispatcherPtr, std::vector<void*>& globalWorkSpaceAddr)
{
    CHK_PTR_NULL(dispatcherPtr);
    return reinterpret_cast<DispatcherPub*>(dispatcherPtr)->SetGlobalWorkSpace(globalWorkSpaceAddr);
}

HcclResult HcclSetNotifyWaitMode(HcclDispatcher dispatcherPtr, const SyncMode notifyWaitMode)
{
    CHK_PTR_NULL(dispatcherPtr);
    return reinterpret_cast<DispatcherPub*>(dispatcherPtr)->SetNotifyWaitMode(notifyWaitMode);
}

HcclResult HcclGetNotifyWaitMode(HcclDispatcher dispatcherPtr, SyncMode* notifyWaitMode)
{
    CHK_PTR_NULL(dispatcherPtr);
    CHK_PTR_NULL(notifyWaitMode);

    *notifyWaitMode = reinterpret_cast<DispatcherPub*>(dispatcherPtr)->GetNotifyWaitMode();
    return HCCL_SUCCESS;
}

HcclResult HcclD2DMemcpyAsync(
    HcclDispatcher dispatcherPtr, DeviceMem& dst, const DeviceMem& src, Stream& stream, u32 remoteUserRank,
    LinkType inLinkType)
{
    CHK_PTR_NULL(dispatcherPtr);
    CHK_PTR_NULL(dst.ptr());
    CHK_PTR_NULL(src.ptr());

    return reinterpret_cast<DispatcherPub*>(dispatcherPtr)->MemcpyAsync(dst, src, stream, remoteUserRank, inLinkType);
}

HcclResult HcclMemcpyAsync(
    HcclDispatcher dispatcherPtr, void* dst, const uint64_t destMax, const void* src, const uint64_t count,
    const HcclRtMemcpyKind kind, Stream& stream, const u32 remoteUserRank, LinkType linkType)
{
    CHK_PTR_NULL(dispatcherPtr);
    CHK_PTR_NULL(dst);
    CHK_PTR_NULL(src);

    return reinterpret_cast<DispatcherPub*>(dispatcherPtr)
        ->MemcpyAsync(dst, destMax, src, count, kind, stream, remoteUserRank, linkType);
}

HcclResult HcclReduceAsync(
    HcclDispatcher dispatcherPtr, void* src, uint64_t count, const HcclDataType datatype, const HcclReduceOp reduceOp,
    Stream& stream, void* dst, const u32 remoteUserRank, const LinkType linkType, const u64 reduceAttr)
{
    CHK_PTR_NULL(dispatcherPtr);
    CHK_PTR_NULL(dst);
    CHK_PTR_NULL(src);

    return reinterpret_cast<DispatcherPub*>(dispatcherPtr)
        ->ReduceAsync(src, count, datatype, reduceOp, stream, dst, remoteUserRank, linkType, reduceAttr);
}

HcclResult
HcclDispatcherWaitValue(HcclDispatcher dispatcherPtr, hccl::Stream& stream, u64 waitAddr, u64 valueAddr, bool reset)
{
    CHK_PTR_NULL(dispatcherPtr);
    return reinterpret_cast<DispatcherPub*>(dispatcherPtr)->WaitValue(stream, waitAddr, valueAddr, reset);
}
HcclResult HcclDispatcherWriteValue(HcclDispatcher dispatcherPtr, hccl::Stream& stream, u64 writeAddr, u64 valueAddr)
{
    CHK_PTR_NULL(dispatcherPtr);
    return reinterpret_cast<DispatcherPub*>(dispatcherPtr)->WriteValue(stream, writeAddr, valueAddr);
}

HcclResult HcclSignalRecord(
    HcclDispatcher dispatcherPtr, HcclRtNotify signal, Stream& stream, u32 userRank, u64 offset, s32 stage, bool inchip,
    u64 signalAddr)
{
    CHK_PTR_NULL(dispatcherPtr);
    CHK_PTR_NULL(signal);

    return reinterpret_cast<DispatcherPub*>(dispatcherPtr)
        ->SignalRecord(signal, stream, userRank, offset, stage, inchip, signalAddr);
}
HcclResult HcclSignalWait(
    HcclDispatcher dispatcherPtr, HcclRtNotify signal, Stream& stream, u32 userRank, u32 remoteUserRank, s32 stage,
    bool inchip)
{
    CHK_PTR_NULL(dispatcherPtr);
    CHK_PTR_NULL(signal);

    return reinterpret_cast<DispatcherPub*>(dispatcherPtr)
        ->SignalWait(signal, stream, userRank, remoteUserRank, stage, inchip);
}

HcclResult LaunchTask(HcclDispatcher dispatcherPtr, Stream& stream)
{
    CHK_PTR_NULL(dispatcherPtr);
    if (g_LaunchTaskCallback != nullptr) {
        CHK_RET(g_LaunchTaskCallback(dispatcherPtr, stream));
    }
    std::vector<Stream> subStreams;
    HcclResult ret = reinterpret_cast<DispatcherPub*>(dispatcherPtr)->LaunchTasksEx(stream, subStreams);
    return ret;
}

HcclResult LaunchTaskExtend(HcclDispatcher dispatcherPtr, Stream& stream, std::vector<Stream>& subStreams)
{
    CHK_PTR_NULL(dispatcherPtr);
    if (g_LaunchTaskCallback != nullptr) {
        CHK_RET(g_LaunchTaskCallback(dispatcherPtr, stream));
    }
    return reinterpret_cast<DispatcherPub*>(dispatcherPtr)->LaunchTasksEx(stream, subStreams);
}

HcclResult InitTask(
    HcclDispatcher dispatcherPtr, hccl::Stream& stream, const bool enableCache, const std::string& key,
    bool useGraphConstructorV2)
{
    CHK_PTR_NULL(dispatcherPtr);

    CHK_RET(reinterpret_cast<DispatcherPub*>(dispatcherPtr)->ResetGraphCtx(enableCache, key, useGraphConstructorV2));
    if (g_InitTaskCallback != nullptr) {
        CHK_RET(g_InitTaskCallback(dispatcherPtr, stream));
    }
    return HCCL_SUCCESS;
}

HcclResult AddRetryPreamble(HcclDispatcher dispatcherPtr, hccl::Stream& stream)
{
    CHK_PTR_NULL(dispatcherPtr);

    CHK_RET(reinterpret_cast<DispatcherPub*>(dispatcherPtr)->AddRetryPreamble(stream));
    return HCCL_SUCCESS;
}

HcclResult SetNormalMode(HcclDispatcher dispatcherPtr)
{
    CHK_PTR_NULL(dispatcherPtr);

    reinterpret_cast<DispatcherPub*>(dispatcherPtr)->SetNormalMode();

    return HCCL_SUCCESS;
}

HcclResult HcclGetCallbackResult(HcclDispatcher dispatcherPtr)
{
    CHK_PTR_NULL(dispatcherPtr);

    return reinterpret_cast<DispatcherPub*>(dispatcherPtr)->GetCallbackResult();
}

HcclResult StreamSync(HcclDispatcher dispatcherPtr, Stream& stream)
{
    CHK_PTR_NULL(dispatcherPtr);

    CHK_RET(reinterpret_cast<DispatcherPub*>(dispatcherPtr)->StreamSync(stream));
    return HCCL_SUCCESS;
}

HcclResult
HcclSetOpExecStatusCallback(HcclDispatcher dispatcherPtr, std::function<HcclResult()> checkOpExecStatusCallback)
{
    CHK_RET(CheckRunSideIsDevice());
    CHK_PTR_NULL(dispatcherPtr);

    reinterpret_cast<DispatcherAiCpu*>(dispatcherPtr)->SetOpExecStatusCallback(checkOpExecStatusCallback);
    return HCCL_SUCCESS;
}

HcclResult HcclSetSqeTimeOut(HcclDispatcher dispatcherPtr, const u64 timeOut)
{
    CHK_RET(CheckRunSideIsDevice());
    CHK_PTR_NULL(dispatcherPtr);

    reinterpret_cast<DispatcherAiCpu*>(dispatcherPtr)->SetSqeTimeOut(timeOut);
    return HCCL_SUCCESS;
}

HcclResult HcclSetSqFullWaitTimeOut(HcclDispatcher dispatcherPtr, const u64 timeOut)
{
    CHK_RET(CheckRunSideIsDevice());
    CHK_PTR_NULL(dispatcherPtr);

    reinterpret_cast<DispatcherAiCpu*>(dispatcherPtr)->SetSqFullWaitTimeOut(timeOut);
    return HCCL_SUCCESS;
}

HcclResult HcclSetOpRingBufferIdx(HcclDispatcher dispatcherPtr, const u32 opRingBufferIdx)
{
    CHK_RET(CheckRunSideIsDevice());
    CHK_PTR_NULL(dispatcherPtr);

    reinterpret_cast<DispatcherAiCpu*>(dispatcherPtr)->SetOpRingBufferIdx(opRingBufferIdx);
    return HCCL_SUCCESS;
}

HcclResult HcclSetExecTimeOut(HcclDispatcher dispatcherPtr, s32 execTimeOut)
{
    CHK_PTR_NULL(dispatcherPtr);
    CHK_RET(reinterpret_cast<DispatcherPub*>(dispatcherPtr)->SetHcclExecTimeOut(execTimeOut));
    return HCCL_SUCCESS;
}
HcclResult SetMultiQpMode(HcclDispatcher dispatcherPtr, bool multiQpMode)
{
    CHK_PTR_NULL(dispatcherPtr);

    CHK_RET(reinterpret_cast<DispatcherPub*>(dispatcherPtr)->SetMultiQpMode(multiQpMode));
    return HCCL_SUCCESS;
}
