/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_DISPATCHER_VIRTUAL_PUB_H
#define HCCL_DISPATCHER_VIRTUAL_PUB_H

#include "dispatcher_pub.h"
#include "hccl_common.h"

namespace hccl {
class DispatcherVirtural : public DispatcherPub {
public:
    explicit DispatcherVirtural(const s32 deviceLogicId);
    ~DispatcherVirtural() override;
    HcclResult Init() override;
    HcclResult SignalRecord(
        HcclRtNotify signal, hccl::Stream& stream, u32 userRank, u64 offset = INVALID_U64,
        s32 stage = INVALID_VALUE_STAGE, bool inchip = false, u64 signalAddr = INVALID_U64,
        u32 notifyId = INVALID_UINT) override;
    HcclResult SignalWait(
        HcclRtNotify signal, hccl::Stream& stream, u32 userRank, u32 remoteUserRank, s32 stage = INVALID_VALUE_STAGE,
        bool inchip = false, u32 notifyId = INVALID_UINT, u32 timeOut = NOTIFY_INVALID_WAIT_TIME) override;
    HcclResult MemcpyAsync(
        hccl::DeviceMem& dst, const hccl::DeviceMem& src, hccl::Stream& stream,
        u32 remoteUserRank = INVALID_VALUE_RANKID, hccl::LinkType inLinkType = hccl::LinkType::LINK_ONCHIP) override;

private:
};
} // namespace hccl
#endif // HCCL_DISPATCHER_VIRTUAL_PUB_H
