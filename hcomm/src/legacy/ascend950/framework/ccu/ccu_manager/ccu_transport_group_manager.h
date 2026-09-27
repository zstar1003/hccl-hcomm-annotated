/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CCU_TRANS_GROUP_MANAGER_H
#define HCCL_CCU_TRANS_GROUP_MANAGER_H

#include <unordered_map>
#include <vector>
#include "ccu_transport_group.h"
#include "ccu_rank_group.h"

namespace Hccl {
class CommunicatorImpl;
class CcuTransportGroupMgr {
public:
    explicit CcuTransportGroupMgr(CommunicatorImpl& comm);
    virtual ~CcuTransportGroupMgr();
    CcuTransportGroup* PrepareCreate(const LinkGroup& linkGrp, u32 cntCkeNum);
    void Confirm();
    void Fallback();
    void Destroy();
    vector<LinkGroup> GetAllTransportGroups();
    void Clean();
    void ResumeAll(u32 cntCkeNum);

private:
    unordered_map<LinkGroup, unique_ptr<CcuTransportGroup>> linkGrp2TransportGrpMap;
    vector<LinkGroup> tempTransportGrp;

    bool isDestroyed{false};
    CommunicatorImpl* comm;
    CcuTransportGroup* CreateTransportGroupByLinkGrp(const LinkGroup& linkGrp, u32 cntCkeNum);
    CcuTransportGroup* Get(const LinkGroup& linkGrp);
};

} // namespace Hccl

#endif // HCCL_CCU_TRANS_GROUP_MANAGER_H
