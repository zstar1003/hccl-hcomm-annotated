/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_TYPICAL_SYNC_MEM_H
#define HCCL_TYPICAL_SYNC_MEM_H

#include <mutex>
#include <map>
#include "hccl_common.h"
#include "network/hccp.h"
#include "mem_device.h"

namespace hccl {
using HcclRtSignal = void*;

class TypicalSyncMem {
public:
    static TypicalSyncMem& GetInstance();

    // Alloc and free notify memory for sync
    HcclResult AllocSyncMem(int32_t** ptr);
    HcclResult FreeSyncMem(int32_t* ptr);
    HcclResult GetNotifyHandle(u64 notifyVa, HcclRtNotify& notifyHandle);
    HcclResult GetNotifySrcMem(struct MrInfoT& mrInfo);

private:
    TypicalSyncMem();
    ~TypicalSyncMem();
    // Delete copy and move constructors and assign operators
    TypicalSyncMem(TypicalSyncMem const&) = delete;            // Copy construct
    TypicalSyncMem(TypicalSyncMem&&) = delete;                 // Move construct
    TypicalSyncMem& operator=(TypicalSyncMem const&) = delete; // Copy assign
    TypicalSyncMem& operator=(TypicalSyncMem&&) = delete;      // Move assign

    HcclResult InitNotifySrcMem();
    HcclResult DeInitNotifySrcMem();
    HcclResult CreateEmptyNotify(HcclRtNotify& notifyHandle);
    HcclResult DestroyNotify(HcclRtNotify notifyHandle);
    HcclResult FreeAllSyncMem();

    RdmaHandle rdmaHandle_ = nullptr;
    struct MrInfoT notifySrcMrInfo_ {};
    MrHandle notifySrcMrHandle_{nullptr};
    std::mutex syncMemMapMutex_;
    DeviceMem srcDevMem_;
    std::map<u64, HcclRtSignal> syncMemMap_{}; // notifyVa - notify map
};
} // namespace hccl
#endif // HCCL_TYPICAL_WINDOW_MEM_H
