/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "remote_notify.h"
#include "remote_notify_impl.h"

namespace hccl {

RemoteNotify::RemoteNotify() {}

RemoteNotify::~RemoteNotify() {}

HcclResult RemoteNotify::Init(const std::vector<u8>& byteVector)
{
    pimpl_.reset((new (std::nothrow) RemoteNotifyImpl()));
    CHK_SMART_PTR_NULL(pimpl_);
    HcclResult ret = pimpl_->Init(byteVector);
    if (ret != HCCL_SUCCESS) {
        pimpl_ = nullptr;
        HCCL_ERROR("[RemoteNotify]Init failed, ret[%p]", ret);
        return ret;
    }

    return HCCL_SUCCESS;
}

HcclResult RemoteNotify::Init(const HcclSignalInfo& notifyInfo, const NotifyLoadType type)
{
    pimpl_.reset((new (std::nothrow) RemoteNotifyImpl()));
    CHK_SMART_PTR_NULL(pimpl_);
    HcclResult ret = pimpl_->Init(notifyInfo, type);
    if (ret != HCCL_SUCCESS) {
        pimpl_ = nullptr;
        HCCL_ERROR("[LocalNotify]Init failed, ret[%d]", ret);
        return ret;
    }

    notifyPtr = pimpl_->ptr();
    return HCCL_SUCCESS;
}

HcclResult RemoteNotify::Open()
{
    CHK_RET(pimpl_->Open());

    notifyPtr = pimpl_->ptr();

    return HCCL_SUCCESS;
}

HcclResult RemoteNotify::Close()
{
    if (pimpl_ != nullptr) {
        pimpl_->Close();
        pimpl_ = nullptr;
        notifyPtr = nullptr;
    }
    return HCCL_SUCCESS;
}

HcclResult RemoteNotify::Post(Stream& stream, HcclDispatcher dispatcher, s32 stage)
{
    CHK_SMART_PTR_NULL(pimpl_);
    return pimpl_->Post(stream, dispatcher, stage);
}

HcclResult RemoteNotify::GetNotifyData(HcclSignalInfo& notifyInfo)
{
    CHK_SMART_PTR_NULL(pimpl_);
    return pimpl_->GetNotifyData(notifyInfo);
}

HcclResult RemoteNotify::SetNotifyData(HcclSignalInfo& notifyInfo)
{
    CHK_SMART_PTR_NULL(pimpl_);
    return pimpl_->SetNotifyData(notifyInfo);
}

HcclResult RemoteNotify::GetNotifyOffset(u64& notifyOffset)
{
    CHK_SMART_PTR_NULL(pimpl_);
    return pimpl_->GetNotifyOffset(notifyOffset);
}
} // namespace hccl
