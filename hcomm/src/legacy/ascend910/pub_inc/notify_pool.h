/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef NOTIFY_POOL_H
#define NOTIFY_POOL_H

#include "dispatcher.h"

namespace hccl {
class NotifyPoolImpl;
class LocalIpcNotify;
class NotifyPool {
public:
    NotifyPool();
    ~NotifyPool();
    HcclResult Init(const s32 devicePhyId);
    HcclResult Destroy();
    HcclResult RegisterOp(const std::string& tag);
    HcclResult UnregisterOp(const std::string& tag);
    // local notify申请
    HcclResult Alloc(
        const std::string& tag, const RemoteRankInfo& info, std::shared_ptr<LocalIpcNotify>& localNotify,
        const NotifyLoadType type = NotifyLoadType::HOST_NOTIFY, u32 offsetAlignSize = INVALID_UINT);
    HcclResult ResetNotify();
    HcclResult ResetNotifyForDestRank(s64 destRank);

protected:
    std::unique_ptr<NotifyPoolImpl> pimpl_;
};
} // namespace hccl
#endif //  NOTIFY_POOL_H
