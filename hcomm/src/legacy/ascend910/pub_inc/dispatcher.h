/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_INC_DISPATCHER_H
#define HCCL_INC_DISPATCHER_H

#include "adapter_pub.h"
#include "mem_host_pub.h"
#include "mem_device_pub.h"
#include "stream_pub.h"
#include "hccl_common.h"
#include "dispatcher_task_types.h"
#include <functional>

enum class DispatcherType {
    DISPATCHER_NORMAL = 0,
    DISPATCHER_VIRTURAL,
    DISPATCHER_AICPU,
};

constexpr u64 ATTR_POS_INLINE_REDUCE = 0x00;
constexpr u64 ATTR_POS_SUPPORT_RDMA = 0x01;
constexpr u64 ATTR_POS_SUPPORT_RDMA_ASYNC = 0x02;
constexpr u64 ATTR_POS_SUPPORT_RDMA_REDUCE = 0x03;
constexpr u64 INLINE_REDUCE_BITMASK = 0x01;
constexpr u64 RDMA_REDUCE_BITMASK = 0x08;
constexpr u64 INLINE_REDUCE_BIT = 0x01;
typedef void (*LoadTaskCallBack)(void* userPtr, void* param, u32 length);

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus
HcclResult HcclDispatcherInit(DispatcherType type, const s32 devicePhyId, HcclDispatcher* dispatcher);

HcclResult HcclDispatcherDestroy(HcclDispatcher dispatcher);

HcclResult HcclSetGlobalWorkSpace(HcclDispatcher dispatcherPtr, std::vector<void*>& globalWorkSpaceAddr);

HcclResult HcclSetNotifyWaitMode(HcclDispatcher dispatcherPtr, const SyncMode notifyWaitMode);
HcclResult HcclGetNotifyWaitMode(HcclDispatcher dispatcherPtr, SyncMode* notifyWaitMode);

HcclResult HcclD2DMemcpyAsync(
    HcclDispatcher dispatcherPtr, hccl::DeviceMem& dst, const hccl::DeviceMem& src, hccl::Stream& stream,
    u32 remoteUserRank = INVALID_VALUE_RANKID, hccl::LinkType inLinkType = hccl::LinkType::LINK_ONCHIP);

HcclResult HcclMemcpyAsync(
    HcclDispatcher dispatcherPtr, void* dst, const uint64_t destMax, const void* src, const uint64_t count,
    const HcclRtMemcpyKind kind, hccl::Stream& stream, const u32 remoteUserRank, hccl::LinkType linkType);

HcclResult HcclReduceAsync(
    HcclDispatcher dispatcherPtr, void* src, uint64_t count, const HcclDataType datatype, const HcclReduceOp reduceOp,
    hccl::Stream& stream, void* dst, const u32 remoteUserRank, const hccl::LinkType linkType, const u64 reduceAttr);

HcclResult
HcclDispatcherWaitValue(HcclDispatcher dispatcherPtr, hccl::Stream& stream, u64 waitAddr, u64 valueAddr, bool reset);
HcclResult HcclDispatcherWriteValue(HcclDispatcher dispatcherPtr, hccl::Stream& stream, u64 writeAddr, u64 valueAddr);

// 仅限task loader使用
HcclResult HcclSignalRecord(
    HcclDispatcher dispatcherPtr, HcclRtNotify signal, hccl::Stream& stream, u32 userRank, u64 offset, s32 stage,
    bool inchip, u64 signalAddr);
HcclResult HcclSignalWait(
    HcclDispatcher dispatcherPtr, HcclRtNotify signal, hccl::Stream& stream, u32 userRank, u32 remoteUserRank,
    s32 stage, bool inchip);

HcclResult LaunchTask(HcclDispatcher dispatcherPtr, hccl::Stream& stream);
HcclResult LaunchTaskExtend(HcclDispatcher dispatcherPtr, hccl::Stream& stream, std::vector<hccl::Stream>& subStreams);
HcclResult InitTask(
    HcclDispatcher dispatcherPtr, hccl::Stream& stream, const bool enableCache, const std::string& key,
    bool useGraphConstructorV2 = false);
HcclResult SetNormalMode(HcclDispatcher dispatcherPtr);
HcclResult HcclGetCallbackResult(HcclDispatcher dispatcherPtr);
void RegisterInitTaskCallBack(HcclResult (*p1)(const HcclDispatcher&, hccl::Stream&));
void RegisterLaunchTaskCallBack(HcclResult (*p1)(const HcclDispatcher&, hccl::Stream&));
HcclResult RegisterLoadTaskCallBack(
    HcclDispatcher dispatcherPtr, void* userPtr, void (*p1)(void* userPtr, void* param, u32 length));
HcclResult HcclDispatcherAicpuInit(
    HcclDispatcher* dispatcher, const u32 devPhyId, uint32_t hcclQos,
    DispatcherType type = DispatcherType::DISPATCHER_AICPU);
HcclResult AddRetryPreamble(HcclDispatcher dispatcherPtr, hccl::Stream& stream);
HcclResult StreamSync(HcclDispatcher dispatcherPtr, hccl::Stream& stream);
HcclResult
HcclSetOpExecStatusCallback(HcclDispatcher dispatcherPtr, std::function<HcclResult()> checkOpExecStatusCallback);
HcclResult HcclSetSqeTimeOut(HcclDispatcher dispatcherPtr, const u64 timeOut);
HcclResult HcclSetSqFullWaitTimeOut(HcclDispatcher dispatcherPtr, const u64 timeOut);
HcclResult HcclSetOpRingBufferIdx(HcclDispatcher dispatcherPtr, const u32 opRingBufferIdx);
HcclResult ForceProfOn(HcclDispatcher& dispatcherPtr, bool isForce);
HcclResult HcclSetExecTimeOut(HcclDispatcher dispatcherPtr, s32 execTimeOut = NOTIFY_DEFAULT_WAIT_TIME);
HcclResult SetMultiQpMode(HcclDispatcher dispatcherPtr, bool multiQpMode);
#ifdef __cplusplus
}
#endif // __cplusplus
#endif //  HCCL_INC_DISPATCHER_H
