/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_QUEUE_NOTIFY_MANAGER_H
#define HCCLV2_QUEUE_NOTIFY_MANAGER_H

#include <map>
#include <vector>
#include "local_notify_v2.h"
#include "types.h"

namespace Hccl {

class CommunicatorImpl;

class QueueNotifyManager {
    using QueueNotifyPool = std::map<std::tuple<QId, QId, u32>, std::unique_ptr<RtsNotify>>;

public:
    static constexpr u32 MAX_NUM_FOR_QPAIR = 12;

    explicit QueueNotifyManager(const CommunicatorImpl& comm);

    ~QueueNotifyManager();

    void ApplyFor(QId postQid, QId waitQid, u32 topicId);

    bool Release(QId postQid, QId waitQid, u32 topicId);

    RtsNotify* Get(QId postQid, QId waitQid, u32 topicId);

    bool Destroy();

    vector<char> GetPackedData();

private:
    const CommunicatorImpl* comm;
    QueueNotifyPool notifyPool;

    bool IsExist(QId postQid, QId waitQid, u32 topicId);
};
} // namespace Hccl

#endif // !HCCLV2_QUEUE_NOTIFY_MANAGER_H
