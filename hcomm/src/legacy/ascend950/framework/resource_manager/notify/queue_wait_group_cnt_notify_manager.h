/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_QUEUE_CNT_NOTIFY_MANAGER_H
#define HCCLV2_QUEUE_CNT_NOTIFY_MANAGER_H

#include <map>
#include "types.h"
#include "rts_cnt_notify.h"

using namespace std;
namespace Hccl {

class QueueWaitGroupCntNotifyManager {
public:
    explicit QueueWaitGroupCntNotifyManager();

    ~QueueWaitGroupCntNotifyManager();

    void ApplyFor(QId qid, u32 topicId);

    bool Release(QId qid, u32 topicId);

    RtsCntNotify* Get(QId qid, u32 topicId);

    void Destroy();

    vector<char> GetPackedData();

private:
    bool IsExist(QId qid, u32 topicId);
    map<pair<QId, u32>, unique_ptr<RtsCntNotify>> notifyPool;
};
} // namespace Hccl

#endif // HCCLV2_QUEUE_CNT_NOTIFY_MANAGER_H
