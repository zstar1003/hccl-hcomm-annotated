/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef UB_MEMORY_TRANSPORT_MANAGER_H
#define UB_MEMORY_TRANSPORT_MANAGER_H
#include "virtual_topo.h"
#include <vector>
#include "ub_memory_transport.h"

namespace Hccl {
class CommunicatorImpl;
class UbMemoryTransportMgr {
public:
    explicit UbMemoryTransportMgr(const CommunicatorImpl& communicator);
    virtual ~UbMemoryTransportMgr();
    UbMemoryTransportMgr* Get(const LinkData& link);
    set<UbMemoryTransportMgr*> Get(RankId rank);
    HcclResult BatchCreateTransport(const std::vector<LinkData>& links);
    void TransportsConnect();
    std::vector<std::pair<RankId, RemoteIpcRmaBuffer*>> GetRmtRankId2RmtIpcRmaBufList();
    std::vector<std::pair<RankId, uintptr_t>> GetAllRankId2AivTagBufAddrList();
    std::vector<std::pair<RankId, uintptr_t>> GetAllRankId2AivOffloadTagBufAddrList();

private:
    vector<LinkData> tempTransport;
    const CommunicatorImpl* comm;
    unordered_map<LinkData, unique_ptr<UbMemoryTransport>> ubMemLink2TransportMap;

    HcclResult CreateTransportByLink(const LinkData& link);

    void WaitTransportsReady(vector<std::pair<UbMemoryTransport*, LinkData>>& transports) const;
    vector<std::pair<UbMemoryTransport*, LinkData>> GetUnconfirmedTrans();
};
} // namespace Hccl
#endif
