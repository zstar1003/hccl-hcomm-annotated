/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "dispatcher_virtural.h"
#include "hccl_tbe_task.h"

namespace hccl {
DispatcherVirtural::DispatcherVirtural(const s32 deviceLogicId) : DispatcherPub(deviceLogicId) {}

DispatcherVirtural::~DispatcherVirtural() {}

HcclResult DispatcherVirtural::Init()
{
#ifndef HCCD
    if (deviceLogicId_ == HOST_DEVICE_ID) {
        return HCCL_SUCCESS;
    }

    aclrtContext ctx = nullptr;
    CHK_RET(hrtCtxGetCurrent(&ctx));
    if (ctx == nullptr) {
        CHK_RET(hrtSetDevice(deviceLogicId_));
        setDeviceFlag_ = true;
    }

    CHK_RET(HcclTbeTaskInit(deviceLogicId_));
#else
    HCCL_ERROR("does not support this interface.");
    return HCCL_E_PARA;
#endif
    return HCCL_SUCCESS;
}

HcclResult DispatcherVirtural::SignalRecord(
    HcclRtNotify signal, Stream& stream, u32 userRank, u64 offset, s32 stage, [[maybe_unused]] bool inchip,
    [[maybe_unused]] u64 signalAddr, [[maybe_unused]] u32 notifyId)
{
    TaskLogicInfo info(
        0, TaskLogicType::DISPATCHER_TYPE, TaskLogicFuncType::DISPATCHER_SIGNALRECORD_TYPE, signal, userRank, offset,
        stage);
    stream.PushTaskLogicInfo(info);

    return HCCL_SUCCESS;
}

HcclResult DispatcherVirtural::SignalWait(
    HcclRtNotify signal, Stream& stream, u32 userRank, u32 remoteUserRank, s32 stage, [[maybe_unused]] bool inchip,
    [[maybe_unused]] u32 notifyId, [[maybe_unused]] u32 timeOut)
{
    TaskLogicInfo info(
        0, TaskLogicType::DISPATCHER_TYPE, TaskLogicFuncType::DISPATCHER_SIGNALWAIT_TYPE, signal, userRank,
        remoteUserRank, stage);
    stream.PushTaskLogicInfo(info);
    return HCCL_SUCCESS;
}

HcclResult DispatcherVirtural::MemcpyAsync(
    hccl::DeviceMem& dst, const hccl::DeviceMem& src, hccl::Stream& stream, [[maybe_unused]] u32 remoteUserRank,
    [[maybe_unused]] hccl::LinkType inLinkType)
{
    if (dst.size() < src.size()) {
        HCCL_ERROR(
            "The size of dst is smaller than that of src. dst addr[%p], dst size[%llu], src addr[%p], src size[%llu]",
            dst.ptr(), dst.size(), src.ptr(), src.size());
        return HCCL_E_PTR;
    }

    TaskLogicInfo info(
        0, TaskLogicType::DISPATCHER_TYPE, TaskLogicFuncType::DISPATCHER_MEMCPYASYNC_TYPE, dst.ptr(), dst.size(),
        src.ptr(), src.size(), HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_DEVICE_TO_DEVICE);

    stream.PushTaskLogicInfo(info);
    return HCCL_SUCCESS;
}
} // namespace hccl
