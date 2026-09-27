/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TEST_LOCAL_NOTIFY_H
#define TEST_LOCAL_NOTIFY_H

#include "stream_pub.h"
#include "dispatcher.h"
#include "hccl_common.h"

namespace hccl {
class LocalNotify {
public:
    LocalNotify() = default;
    ~LocalNotify() = default;

    HcclResult Init(const NotifyLoadType type = NotifyLoadType::HOST_NOTIFY);

    static HcclResult Wait(
        Stream& stream, HcclDispatcher dispatcherPtr, const std::shared_ptr<LocalNotify>& notify,
        s32 stage = INVALID_VALUE_STAGE, u32 timeOut = NOTIFY_DEFAULT_WAIT_TIME);
    static HcclResult Post(
        Stream& stream, HcclDispatcher dispatcherPtr, const std::shared_ptr<LocalNotify>& notify,
        s32 stage = INVALID_VALUE_STAGE);

    HcclResult Wait(
        Stream& stream, HcclDispatcher dispatcherPtr, s32 stage = INVALID_VALUE_STAGE,
        u32 timeOut = NOTIFY_DEFAULT_WAIT_TIME);
    HcclResult Post(Stream& stream, HcclDispatcher dispatcherPtr, s32 stage = INVALID_VALUE_STAGE);

    u32 GetNotifyIdx();
    HcclResult SetNotifyId(u32 notifyId);

    HcclResult Destroy();
    HcclResult SetIpc();
    HcclResult GetNotifyData(HcclSignalInfo& notifyInfo);

    inline HcclRtNotify ptr() { return notifyPtr; }

    HcclRtNotify notifyPtr = nullptr;

private:
    u32 notifyidx_;
};
} // namespace hccl

#endif // LOCAL_NOTIFY_H
