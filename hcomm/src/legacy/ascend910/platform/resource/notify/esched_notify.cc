/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "esched_notify.h"

namespace hccl {
constexpr u32 DEFAULT_NTC_EVENTID = 50;
uint32_t EschedNotify::eventId_ = DEFAULT_NTC_EVENTID;
uint32_t EschedNotify::initialThreadId_ = 0U;

EschedNotify::EschedNotify(NotifyType notifyType) : NotifyBase(notifyType) {}

EschedNotify::EschedNotify(NotifyType notifyType, HcclNotifyInfo notifyInfo) : NotifyBase(notifyType, notifyInfo) {}

EschedNotify::~EschedNotify() { (void)Destroy(); }

HcclResult EschedNotify::Open() { return HCCL_SUCCESS; }

HcclResult EschedNotify::Close() { return HCCL_SUCCESS; }

HcclResult EschedNotify::Wait(
    [[maybe_unused]] Stream& stream, [[maybe_unused]] HcclDispatcher dispatcher, [[maybe_unused]] s32 stage,
    [[maybe_unused]] u32 timeOut)
{
    return HCCL_SUCCESS;
}

HcclResult EschedNotify::Post(
    [[maybe_unused]] Stream& stream, [[maybe_unused]] HcclDispatcher dispatcher, [[maybe_unused]] s32 stage)
{
    return HCCL_SUCCESS;
}

HcclResult EschedNotify::Wait(
    Stream& stream, HcclDispatcher dispatcher, s32 stage, u32 timeOut, [[maybe_unused]] u32 userRank,
    [[maybe_unused]] u32 remoteUserRank)
{
    return Wait(stream, dispatcher, stage, timeOut);
}

HcclResult EschedNotify::Wait([[maybe_unused]] Stream& stream, [[maybe_unused]] u32 timeOut) { return HCCL_SUCCESS; }

HcclResult EschedNotify::Post(Stream& stream, HcclDispatcher dispatcher, s32 stage, [[maybe_unused]] u32 remoteUserRank)
{
    return Post(stream, dispatcher, stage);
}

HcclResult EschedNotify::Post([[maybe_unused]] Stream& stream) { return HCCL_SUCCESS; }

HcclResult EschedNotify::SetIpc() { return HCCL_SUCCESS; }

HcclResult EschedNotify::Grant([[maybe_unused]] s64 recvId) { return HCCL_SUCCESS; }

HcclResult EschedNotify::Alloc() { return HCCL_SUCCESS; }

HcclResult EschedNotify::Destroy() { return HCCL_SUCCESS; }

HcclResult EschedNotify::ThreadIdCreate([[maybe_unused]] uint32_t& threadId) { return HCCL_SUCCESS; }

HcclResult EschedNotify::InitGroupId() { return HCCL_SUCCESS; }

void EschedNotify::ThreadIdQueInit() { return; }

HcclResult EschedNotify::GetGroupId() { return HCCL_SUCCESS; }

void EschedNotify::Break() {}
} // namespace hccl
