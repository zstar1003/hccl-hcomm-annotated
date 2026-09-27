/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "remote_notify_impl.h"
#include "rts_notify.h"
#include "bare_notify.h"
#include "esched_notify.h"

namespace hccl {
RemoteNotifyImpl::RemoteNotifyImpl() {}

RemoteNotifyImpl::~RemoteNotifyImpl() {}

HcclResult RemoteNotifyImpl::Init(const std::vector<u8>& byteVector)
{
    HcclNotifyInfo notifyInfo;
    CHK_RET(NotifyBase::Deserialize(byteVector, notifyInfo));
    NotifyType notifyType = static_cast<NotifyType>(notifyInfo.type);

    HCCL_DEBUG("[RemoteNotifyImpl]notifyType[%u], remote withIpc[%d].", notifyInfo.type, notifyInfo.ipcNotify.withIpc);

    switch (notifyType) {
        case NotifyType::RUNTIME_NOTIFY:
        case NotifyType::RUNTIME_NOTIFY_MC2: {
            notify_.reset(new (std::nothrow) RtsNotify(notifyType, notifyInfo));
            break;
        }

        case NotifyType::BARE_NOTIFY: {
            notify_.reset(new (std::nothrow) BareNotify(notifyType, notifyInfo));
            break;
        }

        case NotifyType::ESCHED_EVENT: {
            notify_.reset(new (std::nothrow) EschedNotify(notifyType, notifyInfo));
            break;
        }

        default: {
            HCCL_ERROR("[Create][RemoteNotify]No specified notify type[%d]!", notifyType);
            break;
        }
    }

    CHK_SMART_PTR_NULL(notify_);
    return HCCL_SUCCESS;
}

HcclResult RemoteNotifyImpl::Init(const HcclSignalInfo& notifyInfo, const NotifyLoadType type)
{
    if (type == NotifyLoadType::DEVICE_NOTIFY) {
        notify_.reset(new (std::nothrow) RtsNotify(NotifyType::RUNTIME_NOTIFY_MC2, notifyInfo));
    } else {
        notify_.reset(new (std::nothrow) RtsNotify(NotifyType::RUNTIME_NOTIFY, notifyInfo));
    }

    HCCL_DEBUG("[Create][LocalNotify]notify load type[%d]. notifyInfo.resId[%u]", type, notifyInfo.resId);

    CHK_SMART_PTR_NULL(notify_);

#ifdef CCL_KERNEL
    CHK_RET(static_cast<RtsNotify*>(notify_.get())->InitAndVerifySingleSignal(false));
#endif

    return HCCL_SUCCESS;
}

HcclResult RemoteNotifyImpl::Open() { return notify_->Open(); }

HcclResult RemoteNotifyImpl::Close()
{
    if (notify_) {
        return notify_->Close();
    }
    notify_ = nullptr;
    return HCCL_SUCCESS;
}

HcclResult RemoteNotifyImpl::Post(Stream& stream, HcclDispatcher dispatcher, s32 stage)
{
    return notify_->Post(stream, dispatcher, stage);
}

HcclResult RemoteNotifyImpl::GetNotifyData(HcclSignalInfo& notifyInfo) { return notify_->GetNotifyData(notifyInfo); }

HcclResult RemoteNotifyImpl::SetNotifyData(HcclSignalInfo& notifyInfo) { return notify_->SetNotifyData(notifyInfo); }

HcclResult RemoteNotifyImpl::GetNotifyOffset(u64& notifyOffset) { return notify_->GetNotifyOffset(notifyOffset); }
} // namespace hccl
