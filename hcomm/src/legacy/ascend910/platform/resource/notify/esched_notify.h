/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ESCHED_NOTIFY_H
#define ESCHED_NOTIFY_H

#include <atomic>
#include "notify_base.h"

namespace hccl {
// 订阅事件相关的参数
struct CpuSendInfo {
    int32_t devId;
    uint32_t grpId;
    uint32_t threadId;
    uint64_t eventBitMap;
};

class EschedNotify : public NotifyBase {
public:
    explicit EschedNotify(NotifyType notifyType);
    EschedNotify(NotifyType notifyType, HcclNotifyInfo notifyInfo);
    ~EschedNotify() override;

    HcclResult Alloc() override;
    HcclResult Open() override;
    HcclResult Close() override;

    HcclResult Wait(Stream& stream, HcclDispatcher dispatcher, s32 stage, u32 timeOut) override;
    HcclResult Post(Stream& stream, HcclDispatcher dispatcher, s32 stage) override;

    HcclResult
    Wait(Stream& stream, HcclDispatcher dispatcher, s32 stage, u32 timeOut, u32 userRank, u32 remoteUserRank) override;
    HcclResult Wait(Stream& stream, u32 timeOut) override;
    HcclResult Post(Stream& stream, HcclDispatcher dispatcher, s32 stage, u32 remoteUserRank) override;
    HcclResult Post(Stream& stream) override;

    HcclResult SetIpc() override;
    HcclResult Grant(s64 recvId) override;
    HcclResult Destroy() override;
    void Break() override;

    static void SetEventIdAndTid(const u32& eventId, const u32& tid)
    {
        eventId_ = eventId;
        initialThreadId_ = tid;
        return;
    }

    static void ThreadIdQueInit();

private:
    HcclResult ThreadIdCreate(uint32_t& threadId);

    static HcclResult InitGroupId();
    static HcclResult GetGroupId();

    bool isDestroyed{false};
    CpuSendInfo sendinfo_{};
    std::atomic<bool> break_ = {false};
    std::atomic<bool> isWaiting_ = {false};

    static u32 groupId_;
    static u32 eventId_;
    static u32 initialThreadId_;
};

} // namespace hccl

#endif // ESCHED_NOTIFY_H
