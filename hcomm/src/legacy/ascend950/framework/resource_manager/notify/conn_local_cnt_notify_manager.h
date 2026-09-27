/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_CONN_LOCAL_CNT_NOTIFY_MANAGER_H
#define HCCLV2_CONN_LOCAL_CNT_NOTIFY_MANAGER_H

#include <unordered_map>
#include <vector>
#include "local_cnt_notify.h"
#include "virtual_topo.h"
namespace Hccl {
class CommunicatorImpl;

class ConnLocalCntNotifyManager {
    using RtsCntNotifyPool = unordered_map<u32, vector<unique_ptr<RtsCntNotify>>>;
    using ConnLocalCntNotifyPool = unordered_map<PortData, unordered_map<u32, vector<unique_ptr<LocalCntNotify>>>>;

public:
    explicit ConnLocalCntNotifyManager(CommunicatorImpl* communicator);

    ~ConnLocalCntNotifyManager();

    void ApplyFor(u32 topicId, vector<LinkData> links);

    vector<RtsCntNotify*> Get(u32 topicId);

    bool Destroy();

    unordered_map<u32, vector<LocalCntNotify*>> GetTopicIdCntNotifyMap(const PortData& portData);

private:
    CommunicatorImpl* comm;
    RtsCntNotifyPool rtsNotifyPool;
    ConnLocalCntNotifyPool localCntNotifyPool;
};
} // namespace Hccl

#endif // HCCLV2_CONN_LOCAL_CNT_NOTIFY_MANAGER_H
