/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef LOCAL_NOTIFY_IMPL_H
#define LOCAL_NOTIFY_IMPL_H

#include "notify_base.h"

namespace hccl {
class NotifyBase;
class LocalNotifyImpl {
public:
    LocalNotifyImpl();
    ~LocalNotifyImpl();
    HcclResult Init(const NotifyLoadType type);
    HcclResult Init(const s32 localDeviceId, const s32 remoteDeviceId, const NotifyLoadType type);
    HcclResult Init(
        const HcclSignalInfo& notifyInfo,
        const NotifyLoadType type = NotifyLoadType::DEVICE_NOTIFY); // aicpu侧依据notifyinfo初始化notify

    HcclResult Wait(Stream& stream, HcclDispatcher dispatcher, s32 stage, u32 timeOut);
    HcclResult Post(Stream& stream, HcclDispatcher dispatcher, s32 stage);

    HcclResult
    Wait(Stream& stream, HcclDispatcher dispatcher, s32 stage, u32 timeOut, u32 userRank, u32 remoteUserRank);
    HcclResult Wait(Stream& stream, u32 timeOut);
    HcclResult Post(Stream& stream, HcclDispatcher dispatcher, s32 stage, u32 remoteUserRank);

    HcclResult Post(Stream& stream);

    HcclResult SetIpc();
    HcclResult Grant(s64 recvId);
    HcclResult Destroy();

    void Break();

    HcclResult Serialize(std::vector<u8>& byteVector);

    // mc2获取aicpu notify信息  local&remote
    HcclResult GetNotifyData(HcclSignalInfo& notifyInfo);
    // aicpu device侧 set notify信息
    HcclResult SetNotifyData(HcclSignalInfo& notifyInfo);

    inline HcclRtNotify ptr() { return notify_->ptr(); }
    // 获取offset
    HcclResult GetNotifyOffset(u64& notifyOffset);

    void SetEventIdAndTid(const u32 eventId, const u32 tid);

private:
    std::unique_ptr<NotifyBase> notify_;
    static std::atomic<bool> tidQueueInit_;
};

} // namespace hccl

#endif // LOCAL_NOTIFY_IMPL_H
