/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_CCU_STREAM_SYNC_NOTIFY_MANAGER_H
#define HCCLV2_CCU_STREAM_SYNC_NOTIFY_MANAGER_H

#include "types.h"
#include "rts_notify_v2.h"
#include "queue_bcast_post_cnt_notify_manager.h"
#include "queue_wait_group_cnt_notify_manager.h"

namespace Hccl {

class CcuStreamSyncNotifyManager {
public:
    RtsCntNotify* GetRtsNTo1CntNotify(u32 streamId, u32 topicId = 0);
    Rts1ToNCntNotify* GetRts1ToNCntNotify(u32 streamId, u32 topicId = 0);

private:
    QueueWaitGroupCntNotifyManager queueWaitGroupCntNotifyManager;
    QueueBcastPostCntNotifyManager queueBcastPostCntNotifyManager;
};

} // namespace Hccl

#endif // HCCLV2_CCU_STREAM_SYNC_NOTIFY_MANAGER_H
