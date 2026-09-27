/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ipc_local_notify.h"
#include "runtime_api_exception.h"
#include "exchange_ipc_notify_dto.h"

namespace Hccl {

IpcLocalNotify::IpcLocalNotify(bool devUsed) : BaseLocalNotify(RmaType::IPC, devUsed)
{
    // SetIpc
    auto name = GetNotify()->SetIpcName();
    s32 sRet = memcpy_s(ipcName, RTS_IPC_MEM_NAME_LEN, name.c_str(), name.size());
    if (sRet != EOK) {
        THROW<RuntimeApiException>(StringFormat("[IpcLocalNotify]memcpy_s ipcname failed, cname[%s]", name.c_str()));
    }
}

void IpcLocalNotify::Wait(const Stream& stream, u32 timeout) const { GetNotify()->Wait(stream, timeout); }

void IpcLocalNotify::Post(const Stream& stream) const { GetNotify()->Post(stream); }

std::unique_ptr<Serializable> IpcLocalNotify::GetExchangeDto()
{
    std::unique_ptr<ExchangeIpcNotifyDto> dto = make_unique<ExchangeIpcNotifyDto>(
        GetNotify()->GetHandleAddr(), GetNotify()->GetId(), HrtDeviceGetBareTgid(), GetNotify()->GetDevPhyId(),
        GetNotify()->IsDevUsed());
    (void)memcpy_s(dto->name, RTS_IPC_MEM_NAME_LEN, ipcName, RTS_IPC_MEM_NAME_LEN);
    return std::unique_ptr<Serializable>(dto.release());
}

string IpcLocalNotify::Describe() const
{
    return StringFormat("IpcLocalNotify[notify=%s]", GetNotify()->Describe().c_str());
}

void IpcLocalNotify::Grant(u32 pid) const
{
    u32 myPid = HrtDeviceGetBareTgid();
    if (pid != myPid) {
        GetNotify()->SetIpcPid(static_cast<s32>(pid));
    }
}

} // namespace Hccl
