/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "local_notify.h"
#include "local_notify_impl.h"

namespace hccl {

LocalNotify::LocalNotify() {}

LocalNotify::~LocalNotify() { Destroy(); }

HcclResult LocalNotify::Init(const NotifyLoadType type)
{
    notifyOwner_ = true; // host侧需要申请新的notify资源
    pimpl_.reset((new (std::nothrow) LocalNotifyImpl()));
    CHK_SMART_PTR_NULL(pimpl_);
    HcclResult ret = pimpl_->Init(type);
    if (ret != HCCL_SUCCESS) {
        pimpl_ = nullptr;
        HCCL_ERROR("[LocalNotify]Init failed, ret[%d]", ret);
        return ret;
    }

    HcclSignalInfo notifyInfo;
    CHK_RET(pimpl_->GetNotifyData(notifyInfo));
    notifyId_ = static_cast<u32>(notifyInfo.resId);

    notifyPtr = pimpl_->ptr();
    return HCCL_SUCCESS;
}

HcclResult LocalNotify::Init(const HcclSignalInfo& signalInfo, const NotifyLoadType type)
{
    notifyOwner_ = false; // aicpu侧不需要申请新的notify资源
    pimpl_.reset((new (std::nothrow) LocalNotifyImpl()));
    CHK_SMART_PTR_NULL(pimpl_);
    HcclResult ret = pimpl_->Init(signalInfo, type);
    if (ret != HCCL_SUCCESS) {
        pimpl_ = nullptr;
        HCCL_ERROR("[LocalNotify]Init failed, ret[%d]", ret);
        return ret;
    }

    HcclSignalInfo notifyInfo;
    CHK_RET(pimpl_->GetNotifyData(notifyInfo));
    notifyId_ = static_cast<u32>(notifyInfo.resId);

    notifyPtr = pimpl_->ptr();
    return HCCL_SUCCESS;
}

HcclResult LocalNotify::InitNotifyLite(const HcclSignalInfo& notifyInfo)
{
    notifyOwner_ = false; // aicpu侧不需要申请新的notify资源
    notifyId_ = static_cast<u32>(notifyInfo.resId);
    HCCL_INFO("[LocalNotify]Init success. notify id [%u]", notifyId_);
    return HCCL_SUCCESS;
}

HcclResult LocalNotify::Wait(Stream& stream, HcclDispatcher dispatcher, s32 stage, u32 timeOut)
{
    return pimpl_->Wait(stream, dispatcher, stage, timeOut);
}

HcclResult LocalNotify::Wait(Stream& stream, u32 timeOut) { return pimpl_->Wait(stream, timeOut); }

HcclResult LocalNotify::Post(Stream& stream, HcclDispatcher dispatcher, s32 stage)
{
    return pimpl_->Post(stream, dispatcher, stage);
}

HcclResult LocalNotify::Post(Stream& stream) { return pimpl_->Post(stream); }

HcclResult LocalNotify::Wait(
    Stream& stream, HcclDispatcher dispatcherPtr, const std::shared_ptr<LocalNotify>& notify, s32 stage, u32 timeOut)
{
    CHK_PTR_NULL(dispatcherPtr);
    CHK_SMART_PTR_NULL(notify);

    return reinterpret_cast<DispatcherPub*>(dispatcherPtr)
        ->SignalWait(
            notify->ptr(), stream, INVALID_VALUE_RANKID, INVALID_VALUE_RANKID, stage, true, notify->notifyId_, timeOut);
}

HcclResult
LocalNotify::Post(Stream& stream, HcclDispatcher dispatcherPtr, const std::shared_ptr<LocalNotify>& notify, s32 stage)
{
    CHK_PTR_NULL(dispatcherPtr);
    CHK_SMART_PTR_NULL(notify);

    return reinterpret_cast<DispatcherPub*>(dispatcherPtr)
        ->SignalRecord(
            notify->ptr(), stream, INVALID_VALUE_RANKID, INVALID_U64, stage, true, INVALID_U64, notify->notifyId_);
}

HcclResult LocalNotify::Destroy()
{
    if (notifyOwner_ == true && pimpl_ != nullptr) {
        pimpl_->Destroy();
        pimpl_ = nullptr;
        notifyPtr = nullptr;
    }
    return HCCL_SUCCESS;
}

HcclResult LocalNotify::SetIpc() { return pimpl_->SetIpc(); }

HcclResult LocalNotify::GetNotifyData(HcclSignalInfo& notifyInfo) { return pimpl_->GetNotifyData(notifyInfo); }

HcclResult LocalNotify::SetNotifyData(HcclSignalInfo& notifyInfo) { return pimpl_->SetNotifyData(notifyInfo); }

HcclResult LocalNotify::GetNotifyOffset(u64& notifyOffset) { return pimpl_->GetNotifyOffset(notifyOffset); }
} // namespace hccl
