/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_RESMGRFETCHER_H
#define HCCL_RESMGRFETCHER_H

#include "stream_lite_mgr.h"
#include "connected_link_mgr.h"
#include "coll_operator.h"
#include "mem_transport_lite_mgr.h"
#include "host_device_sync_notify_lite_mgr.h"
#include "queue_notify_lite_mgr.h"
#include "cnt_1ton_notify_lite_mgr.h"
#include "cnt_nto1_notify_lite_mgr.h"
#include "mirror_task_manager.h"
#include "data_buffer.h"
#include "kernel_param_lite.h"

namespace Hccl {
class ResMgrFetcher {
public:
    virtual ~ResMgrFetcher() = default;
    virtual HostDeviceSyncNotifyLiteMgr* GetHostDeviceSyncNotifyLiteMgr() = 0;
    virtual StreamLiteMgr* GetStreamLiteMgr() = 0;
    virtual QueueNotifyLiteMgr* GetQueueNotifyLiteMgr() = 0;
    virtual Cnt1tonNotifyLiteMgr* GetCnt1tonNotifyLiteMgr() = 0;
    virtual CntNto1NotifyLiteMgr* GetCntNto1NotifyLiteMgr() = 0;
    virtual ConnectedLinkMgr* GetConnectedLinkMgr() = 0;
    virtual DevId GetDevPhyId() = 0;
    virtual u64 GetLocAddr(BufferType type) = 0;
    virtual u32 GetExecTimeOut() = 0;

    virtual CollOperator GetCurrentOp() = 0;
    virtual RmaBufferLite* GetRmaBufferLite(BufferType type) = 0;
    virtual u64 GetCounterAddr() = 0;

    virtual MemTransportLiteMgr* GetTransportLiteMgr() = 0;
    virtual MirrorTaskManagerLite* GetMirrorTaskMgrLite() = 0;
};
} // namespace Hccl

#endif // HCCL_RESMGRFETCHER_H
