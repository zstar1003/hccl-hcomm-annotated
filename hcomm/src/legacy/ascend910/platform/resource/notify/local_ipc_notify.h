/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef LOCAL_IPC_NOTIFY_H
#define LOCAL_IPC_NOTIFY_H

#include "stream_pub.h"
#include "dispatcher.h"
#include "local_notify.h"

namespace hccl {
class LocalIpcNotify : public LocalNotify {
public:
    LocalIpcNotify();
    ~LocalIpcNotify();
    HcclResult
    Init(const s32 localDeviceId, const s32 remoteDeviceId, const NotifyLoadType type = NotifyLoadType::HOST_NOTIFY);
    HcclResult Init(const HcclSignalInfo& notifyInfo, const NotifyLoadType type = NotifyLoadType::DEVICE_NOTIFY);
    HcclResult Serialize(std::vector<u8>& byteVector);

    HcclResult Wait(
        Stream& stream, HcclDispatcher dispatcher, s32 stage = INVALID_VALUE_STAGE,
        u32 timeOut = NOTIFY_DEFAULT_WAIT_TIME, u32 userRank = INVALID_UINT, u32 remoteUserRank = INVALID_UINT);
    HcclResult
    Post(Stream& stream, HcclDispatcher dispatcher, s32 stage = INVALID_VALUE_STAGE, u32 remoteUserRank = INVALID_UINT);
    HcclResult Grant(s64 recvId);

    void Break();

    void SetEventIdAndTid(const u32 eventId, const u32 tid);

    static HcclResult Wait(
        Stream& stream, HcclDispatcher dispatcherPtr, const std::shared_ptr<LocalIpcNotify>& notify,
        s32 stage = INVALID_VALUE_STAGE, u32 timeOut = NOTIFY_DEFAULT_WAIT_TIME, u32 userRank = INVALID_UINT,
        u32 remoteUserRank = INVALID_UINT);
    static HcclResult Post(
        Stream& stream, HcclDispatcher dispatcherPtr, const std::shared_ptr<LocalIpcNotify>& notify,
        s32 stage = INVALID_VALUE_STAGE, u32 remoteUserRank = INVALID_UINT);

    u64 offset = INVALID_U64;
    u64 address = INVALID_U64;
};
} // namespace hccl

#endif // LOCAL_IPC_NOTIFY_H
