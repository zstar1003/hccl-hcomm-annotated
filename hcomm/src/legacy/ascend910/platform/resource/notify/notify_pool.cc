/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "notify_pool.h"
#include "notify_pool_impl.h"

namespace hccl {
NotifyPool::NotifyPool() {}

NotifyPool::~NotifyPool() { pimpl_ = nullptr; }

HcclResult NotifyPool::Init(const s32 devicePhyId)
{
    pimpl_.reset((new (std::nothrow) NotifyPoolImpl(devicePhyId)));
    CHK_SMART_PTR_NULL(pimpl_);
    return pimpl_->Init();
}

HcclResult NotifyPool::Destroy()
{
    CHK_SMART_PTR_NULL(pimpl_);
    return pimpl_->Destroy();
}

HcclResult NotifyPool::RegisterOp(const std::string& tag)
{
    CHK_SMART_PTR_NULL(pimpl_);
    return pimpl_->RegisterOp(tag);
}

HcclResult NotifyPool::UnregisterOp(const std::string& tag)
{
    CHK_SMART_PTR_NULL(pimpl_);
    return pimpl_->UnregisterOp(tag);
}

HcclResult NotifyPool::Alloc(
    const std::string& tag, const RemoteRankInfo& info, std::shared_ptr<LocalIpcNotify>& localNotify,
    const NotifyLoadType type, u32 offsetAlignSize)
{
    CHK_SMART_PTR_NULL(pimpl_);
    return pimpl_->Alloc(tag, info, type, localNotify, offsetAlignSize);
}

HcclResult NotifyPool::ResetNotify()
{
    CHK_SMART_PTR_NULL(pimpl_);
    return pimpl_->ResetNotify();
}

HcclResult NotifyPool::ResetNotifyForDestRank(s64 destRank)
{
    CHK_SMART_PTR_NULL(pimpl_);
    return pimpl_->ResetNotifyForDestRank(destRank);
}
} // namespace hccl
