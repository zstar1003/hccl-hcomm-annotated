/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "host_device_sync_notify_manager.h"
namespace Hccl {
HostDeviceSyncNotifyManager::HostDeviceSyncNotifyManager()
{
    notifys[0] = std::make_unique<RtsNotify>(false);
    notifys[1] = std::make_unique<RtsNotify>(false);
}

RtsNotify* HostDeviceSyncNotifyManager::GetDeviceWaitNotify() { return notifys[0].get(); }

RtsNotify* HostDeviceSyncNotifyManager::GetHostWaitNotify() { return notifys[1].get(); }

void HostDeviceSyncNotifyManager::GetMc2AiCpuNotifys(u8 aicpuNotifyNum, void** aicpuNotify)
{
    if (mc2AicpuNotifys_.size() < aicpuNotifyNum) {
        for (u16 i = mc2AicpuNotifys_.size(); i < aicpuNotifyNum; i++) {
            shared_ptr<RtsNotify> rtsNotify = make_shared<RtsNotify>(true);
            mc2AicpuNotifys_.push_back(rtsNotify);
        }
    }
    for (u16 i = 0; i < aicpuNotifyNum; i++) {
        *(aicpuNotify + i) = reinterpret_cast<void*>(mc2AicpuNotifys_[i].get()->GetHandleAddr());
    }
}

std::vector<char> HostDeviceSyncNotifyManager::GetPackedData()
{
    std::vector<char> result;
    BinaryStream binaryStream;
    binaryStream << notifys[0]->GetUniqueId();
    binaryStream << notifys[1]->GetUniqueId();
    binaryStream.Dump(result);

    return result;
}
} // namespace Hccl
