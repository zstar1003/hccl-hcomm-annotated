/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_CONN_LOCAL_NOTIFY_MANAGER_H
#define HCCLV2_CONN_LOCAL_NOTIFY_MANAGER_H

#include <unordered_map>
#include <vector>
#include <list>
#include "types.h"
#include "orion_adapter_rts.h"
#include "local_notify_v2.h"
#include "virtual_topo.h"

namespace Hccl {
class CommunicatorImpl;

class ConnLocalNotifyManager {
    using ConnLocalNotifyPool = unordered_map<RankId, unordered_map<LinkData, vector<unique_ptr<BaseLocalNotify>>>>;

public:
    explicit ConnLocalNotifyManager(CommunicatorImpl* communicator);

    ~ConnLocalNotifyManager();

    void ApplyFor(RankId remoteRankId, const LinkData& linkData);

    bool Release(RankId remoteRankId, const LinkData& linkData);

    vector<BaseLocalNotify*> Get(RankId remoteRankId, const LinkData& linkData);

    bool Destroy();

private:
    CommunicatorImpl* comm;
    ConnLocalNotifyPool notifyPool;

    bool IsExist(RankId remoteRankId, const LinkData& linkData);
};
} // namespace Hccl

#endif // !HCCLV2_CONN_LOCAL_NOTIFY_MANAGER_H
