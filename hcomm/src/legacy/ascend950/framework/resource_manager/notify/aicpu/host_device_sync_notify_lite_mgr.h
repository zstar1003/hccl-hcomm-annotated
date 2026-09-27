/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_HOST_DEVICE_SYNC_NOTIFY_LITE_MGR_H
#define HCCLV2_HOST_DEVICE_SYNC_NOTIFY_LITE_MGR_H

#include <map>
#include <memory>
#include "types.h"
#include "notify_lite.h"
namespace Hccl {

class HostDeviceSyncNotifyLiteMgr {
public:
    NotifyLite* GetDeviceWaitNotify();

    NotifyLite* GetHostWaitNotify();

    void ParsePackedData(std::vector<char>& data);

private:
    static constexpr u32 HOST_DEVICE_SYNC_NOTIFY_NUM = 2;

    bool isReady{false};

    std::unique_ptr<NotifyLite> notifys[HOST_DEVICE_SYNC_NOTIFY_NUM];
};

} // namespace Hccl
#endif
