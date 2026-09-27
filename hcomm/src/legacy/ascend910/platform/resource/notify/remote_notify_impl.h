/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef REMOTE_NOTIFY_IMPL_H
#define REMOTE_NOTIFY_IMPL_H

#include "notify_base.h"

namespace hccl {
class RemoteNotifyImpl {
public:
    RemoteNotifyImpl();
    ~RemoteNotifyImpl();

    HcclResult Init(const std::vector<u8>& byteVector);
    HcclResult Init(const HcclSignalInfo& notifyInfo, const NotifyLoadType type = NotifyLoadType::DEVICE_NOTIFY);

    HcclResult Open();
    HcclResult Close();
    HcclResult Post(Stream& stream, HcclDispatcher dispatcher, s32 stage);
    HcclResult GetNotifyData(HcclSignalInfo& notifyInfo);
    HcclResult SetNotifyData(HcclSignalInfo& notifyInfo);
    inline HcclRtNotify ptr() { return notify_->ptr(); }
    // 获取offset
    HcclResult GetNotifyOffset(u64& notifyOffset);

private:
    std::unique_ptr<NotifyBase> notify_;
};
} // namespace hccl

#endif // REMOTE_NOTIFY_IMPL_H
