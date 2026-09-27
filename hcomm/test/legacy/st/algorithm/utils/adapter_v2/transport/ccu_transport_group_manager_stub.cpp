/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_transport_group_manager.h"
#include "ccu_transport_manager.h"
#include "coll_service_stub.h"
#include "communicator_impl_stub.h"

namespace Hccl {

CcuTransportGroup* CcuTransportGroupMgr::PrepareCreate(const LinkGroup& linkGrp, u32 cntCkeNum)
{
    // 如果ccuRankGrp2TransportGrpMap中存在rankGrp对应的transportGroup，则直接返回
    auto linkGrpIter = linkGrp2TransportGrpMap.find(linkGrp);
    if (linkGrpIter != linkGrp2TransportGrpMap.end()) {
        return linkGrpIter->second.get();
    }

    CcuTransportMgr* ccuTransportMgr = dynamic_cast<CollServiceStub*>(comm->GetCollService())
                                           ->GetCcuInsPreprocessor()
                                           ->GetCcuComm()
                                           ->GetCcuTransportMgr();
    vector<CcuTransport*> ccuTransports;
    for (auto& linkInfo : linkGrp.GetLinks()) {
        const auto transportsPerRemoteRank = ccuTransportMgr->Get(linkInfo.rankId);
        if (transportsPerRemoteRank.size() != 0) {
            for (auto& transport : transportsPerRemoteRank) {
                if (transport->GetDieId() == linkInfo.dieId) {
                    // 待修改，CcuTransport::GetDieId()为平台层接口，框架层不能用
                    ccuTransports.emplace_back(transport);
                    // 如果找到，先break（避免算法返回的linkGroup中的单个linkInfo对应多条transport）
                    break;
                }
            }
        }
    }

    std::unique_ptr<CcuTransportGroup> newTransportGroup
        = std::make_unique<CcuTransportGroup>(ccuTransports, cntCkeNum);

    // TransportGroup如果创建失败，则返回nullptr，并抛异
    if (newTransportGroup->GetGrpStatus() != TransportGrpStatus::INIT) {
        auto msg = StringFormat("[CcuTransportGroupMgr::%s] Fail to create transportGroup", __func__);
        HCCL_WARNING(msg.c_str());
        return nullptr;
    }
    linkGrp2TransportGrpMap[linkGrp] = std::move(newTransportGroup);
    tempTransportGrp.emplace_back(linkGrp);

    return linkGrp2TransportGrpMap[linkGrp].get();
}

CcuTransportGroupMgr::CcuTransportGroupMgr(CommunicatorImpl& comm) : comm(&comm) { isDestroyed = false; }

CcuTransportGroupMgr::~CcuTransportGroupMgr() {}

void CcuTransportGroupMgr::Confirm() { tempTransportGrp.clear(); }

void CcuTransportGroupMgr::Fallback() {}

} // namespace Hccl
