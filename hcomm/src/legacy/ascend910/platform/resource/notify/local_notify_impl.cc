/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "local_notify_impl.h"
#include "rts_notify.h"
#include "bare_notify.h"
#include "esched_notify.h"

namespace hccl {
std::atomic<bool> LocalNotifyImpl::tidQueueInit_ = {false};
LocalNotifyImpl::LocalNotifyImpl() {}

LocalNotifyImpl::~LocalNotifyImpl() {}

HcclResult LocalNotifyImpl::Init(const NotifyLoadType type)
{
    if (type == NotifyLoadType::DEVICE_NOTIFY) {
        notify_.reset(new (std::nothrow) RtsNotify(NotifyType::RUNTIME_NOTIFY_MC2));
    } else {
        notify_.reset(new (std::nothrow) RtsNotify(NotifyType::RUNTIME_NOTIFY));
    }

    HCCL_DEBUG("[Create][LocalNotify]notify load type[%d].", type);

    CHK_SMART_PTR_NULL(notify_);

    CHK_RET(notify_->Alloc());
    return HCCL_SUCCESS;
}

HcclResult LocalNotifyImpl::Init(const s32 localDeviceId, const s32 remoteDeviceId, const NotifyLoadType type)
{
    if (type == NotifyLoadType::DEVICE_NOTIFY) {
        notify_.reset(new (std::nothrow) RtsNotify(NotifyType::RUNTIME_NOTIFY_MC2));
    } else {
        if (localDeviceId != HOST_DEVICE_ID && remoteDeviceId != HOST_DEVICE_ID) {
            notify_.reset(new (std::nothrow) RtsNotify(NotifyType::RUNTIME_NOTIFY));
        } else if (localDeviceId != HOST_DEVICE_ID && remoteDeviceId == HOST_DEVICE_ID) {
            notify_.reset(new (std::nothrow) BareNotify(NotifyType::BARE_NOTIFY));
        } else if (localDeviceId == HOST_DEVICE_ID && remoteDeviceId != HOST_DEVICE_ID) {
            if (!tidQueueInit_) {
                EschedNotify::ThreadIdQueInit();
                tidQueueInit_ = true;
            }
            notify_.reset(new (std::nothrow) EschedNotify(NotifyType::ESCHED_EVENT));
        } else {
            HCCL_ERROR(
                "[Create][LocalNotify]not support create notify, notify load type[%d], localDeviceId[%d], "
                "remoteDeviceId[%d]",
                type, localDeviceId, remoteDeviceId);
        }
    }

    HCCL_DEBUG(
        "[Create][LocalNotify]notify load type[%d], localDeviceId[%d], remoteDeviceId[%d].", type, localDeviceId,
        remoteDeviceId);

    CHK_SMART_PTR_NULL(notify_);

    CHK_RET(notify_->Alloc());
    return HCCL_SUCCESS;
}

HcclResult LocalNotifyImpl::Init(const HcclSignalInfo& notifyInfo, const NotifyLoadType type)
{
    if (type == NotifyLoadType::DEVICE_NOTIFY) {
        notify_.reset(new (std::nothrow) RtsNotify(NotifyType::RUNTIME_NOTIFY_MC2, notifyInfo));
    } else {
        notify_.reset(new (std::nothrow) RtsNotify(NotifyType::RUNTIME_NOTIFY, notifyInfo));
    }

    HCCL_DEBUG("[Create][LocalNotify]notify load type[%d].", type);

    CHK_SMART_PTR_NULL(notify_);

#ifdef CCL_KERNEL
    if (type == NotifyLoadType::DEVICE_NOTIFY) {
        CHK_RET(static_cast<RtsNotify*>(notify_.get())->InitAndVerifySingleSignal(true));
    }
#endif

    return HCCL_SUCCESS;
}

HcclResult LocalNotifyImpl::Wait(Stream& stream, HcclDispatcher dispatcher, s32 stage, u32 timeOut)
{
    return notify_->Wait(stream, dispatcher, stage, timeOut);
}

HcclResult LocalNotifyImpl::Post(Stream& stream, HcclDispatcher dispatcher, s32 stage)
{
    return notify_->Post(stream, dispatcher, stage);
}

HcclResult LocalNotifyImpl::Post(Stream& stream) { return notify_->Post(stream); }

HcclResult LocalNotifyImpl::Wait(
    Stream& stream, HcclDispatcher dispatcher, s32 stage, u32 timeOut, u32 userRank, u32 remoteUserRank)
{
    return notify_->Wait(stream, dispatcher, stage, timeOut, userRank, remoteUserRank);
}

HcclResult LocalNotifyImpl::Wait(Stream& stream, u32 timeOut) { return notify_->Wait(stream, timeOut); }

HcclResult LocalNotifyImpl::Post(Stream& stream, HcclDispatcher dispatcher, s32 stage, u32 remoteUserRank)
{
    return notify_->Post(stream, dispatcher, stage, remoteUserRank);
}

HcclResult LocalNotifyImpl::SetIpc() { return notify_->SetIpc(); }

HcclResult LocalNotifyImpl::Grant(s64 recvId) { return notify_->Grant(recvId); }

HcclResult LocalNotifyImpl::Destroy()
{
    HcclResult ret = HCCL_SUCCESS;
    if (notify_) {
        ret = notify_->Destroy();
    }
    notify_ = nullptr;
    return ret;
}

HcclResult LocalNotifyImpl::Serialize(std::vector<u8>& byteVector) { return notify_->Serialize(byteVector); }

HcclResult LocalNotifyImpl::GetNotifyData(HcclSignalInfo& notifyInfo) { return notify_->GetNotifyData(notifyInfo); }

HcclResult LocalNotifyImpl::SetNotifyData(HcclSignalInfo& notifyInfo) { return notify_->SetNotifyData(notifyInfo); }

HcclResult LocalNotifyImpl::GetNotifyOffset(u64& notifyOffset) { return notify_->GetNotifyOffset(notifyOffset); }

void LocalNotifyImpl::Break() { return notify_->Break(); }

void LocalNotifyImpl::SetEventIdAndTid(const u32 eventId, const u32 tid)
{
    EschedNotify::SetEventIdAndTid(eventId, tid);
    return;
}
} // namespace hccl
