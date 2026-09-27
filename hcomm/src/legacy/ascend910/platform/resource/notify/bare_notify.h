/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BARE_NOTIFY_H
#define BARE_NOTIFY_H

#include "notify_base.h"

namespace hccl {

class BareNotify : public NotifyBase {
public:
    explicit BareNotify(NotifyType notifyType);
    BareNotify(NotifyType notifyType, HcclNotifyInfo notifyInfo);
    ~BareNotify() override;

    HcclResult Alloc() override;
    HcclResult Destroy() override;

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
    void Break() override;

private:
    bool isOpen{false};
    drvIpcNotifyInfo drvinfo{};
};

} // namespace hccl

#endif // BARE_NOTIFY_H
