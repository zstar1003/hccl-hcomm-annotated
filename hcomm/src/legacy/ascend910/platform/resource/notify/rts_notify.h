/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef RTS_NOTIFY_H
#define RTS_NOTIFY_H

#include "notify_base.h"

namespace hccl {

class RtsNotify : public NotifyBase {
public:
    explicit RtsNotify(NotifyType notifyType);
    RtsNotify(NotifyType notifyType, HcclNotifyInfo notifyInfo);
    RtsNotify(NotifyType notifyType, const HcclSignalInfo& notifyInfo);
    ~RtsNotify() override;

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

    HcclResult GetNotifyData(HcclSignalInfo& notifyInfo) override
    {
        notifyInfo.resId = static_cast<u64>(id);
        notifyInfo.addr = address;
        notifyInfo.devId = devId;
        notifyInfo.tsId = tsId;
        notifyInfo.flag = flag;
        HCCL_DEBUG(
            "GetNotifyData resId[%lld], addr[0x%lx], devId[%u], tsId[%u]", notifyInfo.resId, notifyInfo.addr,
            notifyInfo.devId, notifyInfo.tsId);
        return HCCL_SUCCESS;
    }

    HcclResult SetNotifyData(const HcclSignalInfo& notifyInfo) override
    {
        id = notifyInfo.resId;
        address = notifyInfo.addr;
        devId = notifyInfo.devId;
        tsId = notifyInfo.tsId;
        flag = notifyInfo.flag;
        HCCL_DEBUG("SetNotifyData resId[%lld], addr[0x%lx], devId[%u], tsId[%u]", id, address, devId, tsId);
        return HCCL_SUCCESS;
    }

    HcclResult GetNotifyOffset(u64& notifyOffset) override
    {
        notifyOffset = notifyInfo_.ipcNotify.offset;
        return HCCL_SUCCESS;
    }

    HcclResult InitAndVerifySingleSignal(bool isLocalSignal);

private:
    HcclResult UpdateNotifyInfo();

    bool isDestroyed{false};
    bool inchip{true};
    bool isLocal{true};

    u32 id{INVALID_UINT};
    u64 address{INVALID_U64};
    u32 size{INVALID_UINT};
    u32 devId{INVALID_UINT};
    u32 tsId{INVALID_UINT};
    u32 flag{INVALID_UINT};
};

} // namespace hccl

#endif // RTS_NOTIFY_H
