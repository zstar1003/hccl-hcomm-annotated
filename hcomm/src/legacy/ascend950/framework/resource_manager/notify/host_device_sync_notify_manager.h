/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_HOST_DEVICE_SYNC_NOTIFY_MANAGER_H
#define HCCLV2_HOST_DEVICE_SYNC_NOTIFY_MANAGER_H

#include <vector>
#include <memory>
#include "local_notify_v2.h"
#include "rts_notify_v2.h"

namespace Hccl {

class HostDeviceSyncNotifyManager {
public:
    HostDeviceSyncNotifyManager();

    RtsNotify* GetDeviceWaitNotify();

    RtsNotify* GetHostWaitNotify();

    void GetMc2AiCpuNotifys(u8 aicpuNotifyNum, void** aicpuNotify);

    std::vector<char> GetPackedData();

private:
    static constexpr u32 HOST_DEVICE_SYNC_NOTIFY_NUM = 2;

    std::unique_ptr<RtsNotify> notifys[HOST_DEVICE_SYNC_NOTIFY_NUM];

    std::vector<shared_ptr<RtsNotify>> mc2AicpuNotifys_{};
};
} // namespace Hccl

#endif
