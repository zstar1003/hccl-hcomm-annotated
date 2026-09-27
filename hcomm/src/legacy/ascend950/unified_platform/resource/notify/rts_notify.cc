/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "cast_utils.h"
#include "rts_notify_v2.h"
#include "log.h"
#include "dev_capability.h"
#include "binary_stream.h"
namespace Hccl {

RtsNotify::RtsNotify(bool devUsed) : devPhyId(HrtGetDevicePhyIdByUserDevId(HrtGetDevice())), devUsed(devUsed)
{
    s32 deviceID = HrtGetDevice();

    if (devUsed) {
        handle = HrtNotifyCreateWithFlag(deviceID, ACL_NOTIFY_DEVICE_USE_ONLY);
    } else {
        handle = HrtNotifyCreate(deviceID);
    }

    id = HrtGetNotifyID(handle);
}

RtsNotify::~RtsNotify() { DECTOR_TRY_CATCH("RtsNotify", HrtNotifyDestroy(handle)); }

std::string RtsNotify::SetIpcName() const
{
    char ipcName[RTS_IPC_MEM_NAME_LEN] = {0};
    HrtIpcSetNotifyName(handle, ipcName, RTS_IPC_MEM_NAME_LEN);
    return ipcName;
}

void RtsNotify::SetIpcPid(s32 pid) const { HrtSetIpcNotifyPid(handle, pid); }

u32 RtsNotify::GetId() const { return id; }

u64 RtsNotify::GetOffset() const { return HrtNotifyGetOffset(handle); }

u64 RtsNotify::GetHandleAddr() const { return ReinterpretAs<u64>(handle); }

u32 RtsNotify::GetSize() const { return DevCapability::GetInstance().GetNotifySize(); }

bool RtsNotify::IsDevUsed() const { return devUsed; }

u32 RtsNotify::GetDevPhyId() const { return devPhyId; }

std::vector<char> RtsNotify::GetUniqueId() const
{
    BinaryStream binaryStream;
    binaryStream << id;
    binaryStream << devPhyId;
    HCCL_INFO("[RtsNotify][GetUniqueId]id[%u] devPhyId[%u]", id, devPhyId);
    std::vector<char> result;
    binaryStream.Dump(result);
    return result;
}

void RtsNotify::Wait(const Stream& stream, u32 timeout) const
{
    HrtNotifyWaitWithTimeOut(handle, stream.GetPtr(), timeout);
}

void RtsNotify::Post(const Stream& stream) const { HrtNotifyRecord(handle, stream.GetPtr()); }

string RtsNotify::Describe() const
{
    return StringFormat("RtsNotify[devUsed=%d, id=%u, handle=%p]", devUsed, id, handle);
}

} // namespace Hccl
