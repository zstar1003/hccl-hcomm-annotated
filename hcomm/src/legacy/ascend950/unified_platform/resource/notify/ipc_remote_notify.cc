/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ipc_remote_notify.h"
#include "exchange_ipc_notify_dto.h"
#include "dev_capability.h"
namespace Hccl {

IpcRemoteNotify::IpcRemoteNotify() : BaseRemoteNotify(RmaType::IPC) {}

IpcRemoteNotify::IpcRemoteNotify(const Serializable& rmtDto) : BaseRemoteNotify(RmaType::IPC)
{
    auto dto = dynamic_cast<const ExchangeIpcNotifyDto&>(rmtDto);
    rmtPid = dto.pid;
    handleAddr = dto.handleAddr;
    id = dto.id;
    devUsed = dto.devUsed;
    rmtDevPhyId = dto.devPhyId;
    (void)memcpy_s(name, RTS_IPC_MEM_NAME_LEN, dto.name, RTS_IPC_MEM_NAME_LEN);

    // OpenIpc
    u32 myPid = HrtDeviceGetBareTgid();
    if (rmtPid == myPid) {
        handle = reinterpret_cast<void*>(handleAddr);
    } else {
        if (devUsed) {
            handle = HrtIpcOpenNotifyWithFlag(name, RT_NOTIFY_FLAG_DOWNLOAD_TO_DEV);
        } else {
            handle = HrtIpcOpenNotify(name);
        }
    }
    addr = HrtNotifyGetAddr(handle);
    size = DevCapability::GetInstance().GetNotifySize();
    HCCL_INFO(
        "IpcRemoteNotify[name=%s, handleAddr=0x%llx, id=%u, rmtPid=%u, rmtDevPhyId=%u, devUsed=%d, addr=0x%llx, "
        "handle=%p]",
        name, handleAddr, id, rmtPid, rmtDevPhyId, devUsed, addr, handle);
}

void IpcRemoteNotify::Post(const Stream& stream) const { HrtNotifyRecord(handle, stream.GetPtr()); }

string IpcRemoteNotify::Describe() const
{
    return StringFormat(
        "IpcRemoteNotify[name=%s, handleAddr=0x%llx, id=%u, rmtPid=%u, rmtDevPhyId=%u, devUsed=%d, "
        "handle=%p, addr=0x%llx, size=%llu]",
        name, handleAddr, id, rmtPid, rmtDevPhyId, devUsed, handle, addr, size);
}

} // namespace Hccl
