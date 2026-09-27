/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "bare_notify.h"

namespace hccl {

BareNotify::BareNotify(NotifyType notifyType) : NotifyBase(notifyType) {}

BareNotify::BareNotify(NotifyType notifyType, HcclNotifyInfo notifyInfo) : NotifyBase(notifyType, notifyInfo) {}

BareNotify::~BareNotify()
{
    (void)Close();
    (void)Destroy();
}

HcclResult BareNotify::Open() { return HCCL_SUCCESS; }

HcclResult BareNotify::Close() { return HCCL_SUCCESS; }

HcclResult BareNotify::Wait(
    [[maybe_unused]] Stream& stream, [[maybe_unused]] HcclDispatcher dispatcher, [[maybe_unused]] s32 stage,
    [[maybe_unused]] u32 timeOut)
{
    return HCCL_SUCCESS;
}

HcclResult BareNotify::Post(
    [[maybe_unused]] Stream& stream, [[maybe_unused]] HcclDispatcher dispatcher, [[maybe_unused]] s32 stage)
{
    return HCCL_SUCCESS;
}

HcclResult BareNotify::Wait(
    Stream& stream, HcclDispatcher dispatcher, s32 stage, u32 timeOut, [[maybe_unused]] u32 userRank,
    [[maybe_unused]] u32 remoteUserRank)
{
    return Wait(stream, dispatcher, stage, timeOut);
}

HcclResult BareNotify::Wait([[maybe_unused]] Stream& stream, [[maybe_unused]] u32 timeOut) { return HCCL_SUCCESS; }

HcclResult BareNotify::Post(Stream& stream, HcclDispatcher dispatcher, s32 stage, [[maybe_unused]] u32 remoteUserRank)
{
    return Post(stream, dispatcher, stage);
}

HcclResult BareNotify::Post([[maybe_unused]] Stream& stream) { return HCCL_SUCCESS; }

HcclResult BareNotify::SetIpc() { return HCCL_SUCCESS; }

HcclResult BareNotify::Grant([[maybe_unused]] s64 recvId) { return HCCL_SUCCESS; }

HcclResult BareNotify::Alloc() { return HCCL_SUCCESS; }

HcclResult BareNotify::Destroy() { return HCCL_SUCCESS; }

void BareNotify::Break() { return; }
} // namespace hccl
