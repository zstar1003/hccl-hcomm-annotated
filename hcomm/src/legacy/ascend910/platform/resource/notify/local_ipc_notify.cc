/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "local_ipc_notify.h"
#include "local_notify_impl.h"

namespace hccl {
LocalIpcNotify::LocalIpcNotify() {}

LocalIpcNotify::~LocalIpcNotify() { Destroy(); }

HcclResult LocalIpcNotify::Init(const s32 localDeviceId, const s32 remoteDeviceId, const NotifyLoadType type)
{
    pimpl_.reset((new (std::nothrow) LocalNotifyImpl()));
    CHK_SMART_PTR_NULL(pimpl_);
    HcclResult ret = pimpl_->Init(localDeviceId, remoteDeviceId, type);
    if (ret != HCCL_SUCCESS) {
        pimpl_ = nullptr;
        HCCL_ERROR("[LocalNotify]Init failed, ret[%p]", ret);
        return ret;
    }

    notifyPtr = pimpl_->ptr();

    if (localDeviceId == HOST_DEVICE_ID || remoteDeviceId == HOST_DEVICE_ID) {
        return HCCL_SUCCESS;
    }

    CHK_RET(pimpl_->GetNotifyOffset(offset));
    HcclSignalInfo notifyInfo;
    CHK_RET(pimpl_->GetNotifyData(notifyInfo));
    address = notifyInfo.addr;
    notifyId_ = static_cast<u32>(notifyInfo.resId);

    return HCCL_SUCCESS;
}

HcclResult LocalIpcNotify::Init(const HcclSignalInfo& notifyInfo, const NotifyLoadType type)
{
    notifyOwner_ = false; // aicpu侧不需要申请新的notify资源
    pimpl_.reset((new (std::nothrow) LocalNotifyImpl()));
    CHK_SMART_PTR_NULL(pimpl_);
    HcclResult ret = pimpl_->Init(notifyInfo, type);
    if (ret != HCCL_SUCCESS) {
        pimpl_ = nullptr;
        HCCL_ERROR("[LocalNotify]Init failed, ret[%p]", ret);
        return ret;
    }

    CHK_RET(pimpl_->GetNotifyOffset(offset));
    address = notifyInfo.addr;
    notifyId_ = static_cast<u32>(notifyInfo.resId);

    notifyPtr = pimpl_->ptr();
    return HCCL_SUCCESS;
}

HcclResult LocalIpcNotify::Serialize(std::vector<u8>& byteVector) { return pimpl_->Serialize(byteVector); }

HcclResult LocalIpcNotify::Wait(
    Stream& stream, HcclDispatcher dispatcher, s32 stage, u32 timeOut, u32 userRank, u32 remoteUserRank)
{
    return pimpl_->Wait(stream, dispatcher, stage, timeOut, userRank, remoteUserRank);
}

HcclResult LocalIpcNotify::Post(Stream& stream, HcclDispatcher dispatcher, s32 stage, u32 remoteUserRank)
{
    return pimpl_->Post(stream, dispatcher, stage, remoteUserRank);
}

HcclResult LocalIpcNotify::Wait(
    Stream& stream, HcclDispatcher dispatcherPtr, const std::shared_ptr<LocalIpcNotify>& notify, s32 stage, u32 timeOut,
    u32 userRank, u32 remoteUserRank)
{
    CHK_PTR_NULL(dispatcherPtr);
    CHK_SMART_PTR_NULL(notify);

    return reinterpret_cast<DispatcherPub*>(dispatcherPtr)
        ->SignalWait(notify->ptr(), stream, userRank, remoteUserRank, stage, false, INVALID_UINT, timeOut);
}
HcclResult LocalIpcNotify::Post(
    Stream& stream, HcclDispatcher dispatcherPtr, const std::shared_ptr<LocalIpcNotify>& notify, s32 stage,
    u32 remoteUserRank)
{
    CHK_PTR_NULL(dispatcherPtr);
    CHK_SMART_PTR_NULL(notify);

    return reinterpret_cast<DispatcherPub*>(dispatcherPtr)
        ->SignalRecord(notify->ptr(), stream, remoteUserRank, notify->offset, stage, false, notify->address);
}

HcclResult LocalIpcNotify::Grant(s64 recvId) { return pimpl_->Grant(recvId); }

void LocalIpcNotify::Break() { return pimpl_->Break(); }

void LocalIpcNotify::SetEventIdAndTid(const u32 eventId, const u32 tid)
{
    LocalNotifyImpl impl;
    return impl.SetEventIdAndTid(eventId, tid);
}
} // namespace hccl
