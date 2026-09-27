/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CCU_TRANS_MANAGER_H
#define HCCL_CCU_TRANS_MANAGER_H

#include <unordered_map>
#include <vector>
#include "types.h"
#include "virtual_topo.h"
#include "ccu_transport.h"
#include "socket_manager.h"

namespace Hccl {

class CcuTransportMgr {
public:
    CcuTransportMgr(const CommunicatorImpl& comm, const int32_t devLogicId);
    virtual ~CcuTransportMgr();
    CcuTransport* Get(const LinkData& link);
    set<CcuTransport*> Get(RankId rank);
    HcclResult PrepareCreate(const LinkData& link, CcuTransport*& transport);

    void Confirm(); // 用于正常建联流程，失败需要回退
    void Fallback();
    void Destroy();
    void RecoverConfirm(); // 用于快照恢复特性，失败报错不回退

    // 以下接口用于N秒快恢特性
    void Clean();
    void Resume();

private:
    const CommunicatorImpl* comm{nullptr};
    const int32_t devLogicId_{0};
    bool isDestroyed{false};

    unordered_map<LinkData, unique_ptr<CcuTransport>> ccuLink2TransportMap;
    unordered_map<RankId, set<CcuTransport*>> ccuRank2TransportsMap;
    vector<LinkData> tempTransport;

    vector<std::pair<CcuTransport*, LinkData>> GetUnConfirmedTrans();
    HcclResult CreateTransportByLink(const LinkData& link, CcuTransport*& transport);
    void TransportsConnect();
    void WaitTransportsReady(vector<std::pair<CcuTransport*, LinkData>>& transports) const;
    void DumpNotReadyTransports(vector<std::pair<CcuTransport*, LinkData>>& transports) const;

    void RecoverTransportsConnect();
    void WaitTransportsRecoverReady(vector<std::pair<CcuTransport*, LinkData>>& transports) const;
};

} // namespace Hccl

#endif // HCCL_CCU_TRANS_MANAGER_H
