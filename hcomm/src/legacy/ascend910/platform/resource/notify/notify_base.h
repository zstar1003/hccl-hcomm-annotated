/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef NOTIFY_BASE_H
#define NOTIFY_BASE_H

#include "adapter_rts.h"
#include "adapter_hal.h"
#include "hccl/base.h"

#include "mem_name_repository_pub.h"
#include "dispatcher_pub.h"

namespace hccl {

enum class NotifyType { RUNTIME_NOTIFY = 0, RUNTIME_NOTIFY_MC2, BARE_NOTIFY, ESCHED_EVENT, NOTIFY_TYPE_RESERVED };

constexpr s32 IPC_NOTIFY_PID_ARRAY_SIZE = 1;

using HcclIpcRtsNotify = struct TagHcclIpcRtsNotify {
    u8 ipcName[HCCL_IPC_MEM_NAME_LEN] = {0};
    bool withIpc;
    HcclRtNotify ptr;
    u32 id;
    u64 offset;

    TagHcclIpcRtsNotify() : withIpc(false), ptr(nullptr), id(INVALID_UINT), offset(INVALID_U64) {}
};

using HcclNotifyInfo = struct TagHcclNotifyInfo {
    u32 type;
    HcclIpcRtsNotify ipcNotify;

    TagHcclNotifyInfo() : type(INVALID_UINT) {}

    TagHcclNotifyInfo(const TagHcclNotifyInfo& that) : type(that.type), ipcNotify(that.ipcNotify) {}

    TagHcclNotifyInfo(const TagHcclNotifyInfo&& that) : type(that.type), ipcNotify(that.ipcNotify) {}

    TagHcclNotifyInfo& operator=(const TagHcclNotifyInfo& that)
    {
        if (&that != this) {
            type = that.type;
            ipcNotify = that.ipcNotify;
        }
        return *this;
    }
};

class NotifyBase {
public:
    explicit NotifyBase(NotifyType notifyType) : notifyType(notifyType)
    {
        notifyInfo_.type = static_cast<u32>(notifyType);
    }

    NotifyBase(NotifyType notifyType, HcclNotifyInfo notifyInfo) : notifyType(notifyType), notifyInfo_(notifyInfo) {}

    NotifyBase(NotifyType notifyType, [[maybe_unused]] HcclSignalInfo notifyInfo) : notifyType(notifyType)
    {
        HCCL_ERROR("[NotifyConstructor]Does not support this interface.");
    }

    virtual ~NotifyBase() {}

    HcclResult Serialize(std::vector<u8>& byteVector)
    {
        std::vector<u8> data = CustomTypeToVectorByte<HcclNotifyInfo>(notifyInfo_);
        if (data.empty() || data.size() > NOTIFY_INFO_LENGTH) {
            HCCL_ERROR("serialize msgSize[%u] > NOTIFY_INFO_LENGTH[%u]", data.size(), NOTIFY_INFO_LENGTH);
            return HCCL_E_INTERNAL;
        }

        std::vector<u8> paddingData(NOTIFY_INFO_LENGTH - data.size(), 0);
        HCCL_DEBUG("[Serialize]data size[%u], paddingData[%u].", data.size(), paddingData.size());
        data.insert(data.end(), paddingData.begin(), paddingData.end());

        byteVector = data;

        return HCCL_SUCCESS;
    }

    static HcclResult Deserialize(const std::vector<u8>& byteVector, HcclNotifyInfo& notifyInfo)
    {
        CHK_PRT_RET(
            byteVector.size() < sizeof(HcclNotifyInfo),
            HCCL_ERROR(
                "[Deserialize]expected byteVector size[%zu] < sizeof(HcclNotifyInfo)[%zu]", byteVector.size(),
                sizeof(HcclNotifyInfo)),
            HCCL_E_PARA);
        CHK_SAFETY_FUNC_RET(memcpy_s((u8*)&notifyInfo, sizeof(HcclNotifyInfo), &byteVector[0], sizeof(HcclNotifyInfo)));
        return HCCL_SUCCESS;
    }

    virtual HcclResult Alloc() = 0;
    virtual HcclResult Destroy() = 0;

    virtual HcclResult Open() = 0;
    virtual HcclResult Close() = 0;

    virtual HcclResult Wait(Stream& stream, HcclDispatcher dispatcher, s32 stage, u32 timeOut) = 0;
    virtual HcclResult Post(Stream& stream, HcclDispatcher dispatcher, s32 stage) = 0;

    virtual HcclResult
    Wait(Stream& stream, HcclDispatcher dispatcher, s32 stage, u32 timeOut, u32 userRank, u32 remoteUserRank)
        = 0;
    virtual HcclResult Wait(Stream& stream, u32 timeOut) = 0;
    virtual HcclResult Post(Stream& stream, HcclDispatcher dispatcher, s32 stage, u32 remoteUserRank) = 0;
    virtual HcclResult Post(Stream& stream) = 0;
    virtual HcclResult SetIpc() = 0;
    virtual HcclResult Grant(s64 recvId) = 0;
    virtual void Break()
    {
        HCCL_ERROR("[Break]Does not support this interface.");
        return;
    }

    virtual HcclResult GetNotifyData([[maybe_unused]] HcclSignalInfo& notifyInfo)
    {
        HCCL_ERROR("[GetNotifyData]Does not support this interface.");
        return HCCL_E_NOT_SUPPORT;
    }

    virtual HcclResult SetNotifyData([[maybe_unused]] const HcclSignalInfo& notifyInfo)
    {
        HCCL_ERROR("[SetNotifyData]Does not support this interface.");
        return HCCL_E_NOT_SUPPORT;
    }

    virtual HcclResult GetNotifyOffset([[maybe_unused]] u64& offset)
    {
        HCCL_ERROR("[GetNotifyOffset]Does not support this interface.");
        return HCCL_E_NOT_SUPPORT;
    }

    inline HcclRtNotify ptr() { return notifyPtr; }

protected:
    NotifyType notifyType;
    HcclNotifyInfo notifyInfo_;
    HcclRtNotify notifyPtr{nullptr};
};
} // namespace hccl

#endif // NOTIFY_BASE_H
