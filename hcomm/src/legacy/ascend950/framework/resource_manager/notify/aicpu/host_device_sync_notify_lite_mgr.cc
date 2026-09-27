/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "host_device_sync_notify_lite_mgr.h"
#include "binary_stream.h"

namespace Hccl {

NotifyLite* HostDeviceSyncNotifyLiteMgr::GetDeviceWaitNotify() { return notifys[0].get(); }

NotifyLite* HostDeviceSyncNotifyLiteMgr::GetHostWaitNotify() { return notifys[1].get(); }

void HostDeviceSyncNotifyLiteMgr::ParsePackedData(std::vector<char>& data)
{
    BinaryStream binaryStream(data);

    std::vector<char> uniqueId0;
    std::vector<char> uniqueId1;
    binaryStream >> uniqueId0;
    binaryStream >> uniqueId1;

    if (!isReady) {
        notifys[0] = std::make_unique<NotifyLite>(uniqueId0);
        notifys[1] = std::make_unique<NotifyLite>(uniqueId1);
        isReady = true;
    }
}

} // namespace Hccl
