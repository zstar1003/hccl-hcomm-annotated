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
#include "ccu_transport_group.h"
#include "exception_util.h"
#include "internal_exception.h"
#include "communicator_impl.h"
#include "coll_service_device_mode.h"

namespace Hccl {

CcuTransportGroupMgr::CcuTransportGroupMgr(CommunicatorImpl& comm) : comm(&comm) { isDestroyed = false; }

CcuTransportGroupMgr::~CcuTransportGroupMgr()
{
    if (!isDestroyed) {
        DECTOR_TRY_CATCH("CcuTransportGroupMgr", Destroy());
    }
}

CcuTransportGroup* CcuTransportGroupMgr::Get(const LinkGroup& linkGrp)
{
    auto linkGrpIter = linkGrp2TransportGrpMap.find(linkGrp);
    if (linkGrpIter != linkGrp2TransportGrpMap.end()) {
        return linkGrpIter->second.get();
    }
    HCCL_WARNING(
        "[CcuTransportGroupMgr::%s] CcuTransportGroup does not existed, "
        "errNo[0x%016llx], RankGroup size:%u",
        __func__, HCCL_ERROR_CODE(HcclResult::HCCL_E_PTR), linkGrp.GetLinks().size());
    return nullptr;
}

CcuTransportGroup* CcuTransportGroupMgr::PrepareCreate(const LinkGroup& linkGrp, u32 cntCkeNum)
{
    // 如果linkGrp2TransportGrpMap中存在linkGrp对应的transportGroup，则直接返回
    auto ccuTransportGrp = Get(linkGrp);
    if (ccuTransportGrp != nullptr) {
        return ccuTransportGrp;
    }

    return CreateTransportGroupByLinkGrp(linkGrp, cntCkeNum);
}

CcuTransportGroup* CcuTransportGroupMgr::CreateTransportGroupByLinkGrp(const LinkGroup& linkGrp, u32 cntCkeNum)
{
    CHECK_NULLPTR(comm, "[CcuTransportGroupMgr::CreateTransportGroupByLinkGrp] comm is nullptr!");
    CcuTransportMgr* ccuTransportMgr = dynamic_cast<CollServiceDeviceMode*>(comm->GetCollService())
                                           ->GetCcuInsPreprocessor()
                                           ->GetCcuComm()
                                           ->GetCcuTransportMgr();
    vector<CcuTransport*> ccuTransports;
    for (auto& linkInfo : linkGrp.GetLinks()) {
        const auto transportsPerRemoteRank = ccuTransportMgr->Get(linkInfo.rankId);
        if (transportsPerRemoteRank.size() != 0) {
            for (auto& transport : transportsPerRemoteRank) {
                if (transport->GetDieId() == linkInfo.dieId) {
                    ccuTransports.emplace_back(transport);
                    // 如果找到，先break（避免算法返回的linkGroup中的单个linkInfo对应多条transport）
                    break;
                }
            }
        }
    }

    std::unique_ptr<CcuTransportGroup> newTransportGroup
        = std::make_unique<CcuTransportGroup>(ccuTransports, cntCkeNum);

    // TransportGroup如果创建失败，则返回nullptr，不抛异
    if (newTransportGroup->GetGrpStatus() != TransportGrpStatus::INIT) {
        auto msg = StringFormat("[CcuTransportGroupMgr::%s] Fail to create transportGroup", __func__);
        HCCL_WARNING(msg.c_str());
        return nullptr;
    }
    linkGrp2TransportGrpMap[linkGrp] = std::move(newTransportGroup);
    tempTransportGrp.emplace_back(linkGrp);

    return linkGrp2TransportGrpMap[linkGrp].get();
}

void CcuTransportGroupMgr::Confirm() { tempTransportGrp.clear(); }

void CcuTransportGroupMgr::Clean()
{
    for (auto& linkGrpTransPair : linkGrp2TransportGrpMap) {
        linkGrpTransPair.second = nullptr;
    }
}

void CcuTransportGroupMgr::ResumeAll(u32 cntCkeNum)
{
    vector<LinkGroup> linkGroups;
    for (auto iter = linkGrp2TransportGrpMap.begin(); iter != linkGrp2TransportGrpMap.end(); ++iter) {
        linkGroups.push_back(iter->first);
    }
    if (linkGroups.size() == 0) {
        HCCL_WARNING("[CcuTransportGroupMgr][%s] used linkGroups vec is empty", __func__);
        return;
    }

    for (auto& linkGroup : linkGroups) {
        CcuTransportGroup* transportGrp = CreateTransportGroupByLinkGrp(linkGroup, cntCkeNum);
        if (transportGrp == nullptr) {
            THROW<InternalException>(
                "[CcuTransportGroupMgr][%s] transportGrp alloc resource fail, "
                "linkGroup size[%zu], cntCkeNum[%u]",
                __func__, linkGroup.GetLinks().size(), cntCkeNum);
        }
    }
}

void CcuTransportGroupMgr::Fallback()
{
    for (auto& linkGrp : tempTransportGrp) {
        auto iterLinkGrp = linkGrp2TransportGrpMap.find(linkGrp);
        linkGrp2TransportGrpMap.erase(iterLinkGrp);
    }
    tempTransportGrp.clear();
}

void CcuTransportGroupMgr::Destroy()
{
    isDestroyed = true;
    linkGrp2TransportGrpMap.clear();
}

vector<LinkGroup> CcuTransportGroupMgr::GetAllTransportGroups()
{
    vector<LinkGroup> linkGroups;
    for (auto iter = linkGrp2TransportGrpMap.begin(); iter != linkGrp2TransportGrpMap.end(); ++iter) {
        linkGroups.push_back(iter->first);
    }
    if (linkGroups.size() == 0) {
        THROW<InternalException>("[CcuTransportGroupMgr][%s] used linkGroups vec is empty", __func__);
    }
    return linkGroups;
}

} // namespace Hccl
