/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "conn_local_cnt_notify_manager.h"
#include <set>
#include "communicator_impl.h"
#include "rdma_handle_manager.h"

namespace Hccl {

ConnLocalCntNotifyManager::ConnLocalCntNotifyManager(CommunicatorImpl* communicator) : comm(communicator) {}

ConnLocalCntNotifyManager::~ConnLocalCntNotifyManager() { DECTOR_TRY_CATCH("ConnLocalCntNotifyManager", Destroy()); }

void ConnLocalCntNotifyManager::ApplyFor(u32 topicId, vector<LinkData> links)
{
    HCCL_INFO("in topicId=%u apply for now1", topicId);

    if (rtsNotifyPool.count(topicId) != 0) {
        HCCL_WARNING("Local notify for topicId[%u] already exists, no need to alloc.", topicId);
        return;
    }

    HCCL_INFO("topicId=%u apply for now", topicId);

    // 拿到并存储全部的portData
    set<std::pair<PortData, LinkProtocol>> ports;
    for (auto& link : links) {
        auto linkProtocol = link.GetLinkProtocol();
        bool ifUbProto = linkProtocol == LinkProtocol::UB_TP || linkProtocol == LinkProtocol::UB_CTP;
        if (link.GetType() != PortDeploymentType::DEV_NET || !ifUbProto) {
            string msg
                = StringFormat("Unsupported %s of link %s", link.GetType().Describe().c_str(), link.Describe().c_str());
            THROW<InvalidParamsException>(msg);
        }

        HCCL_INFO("topicId=%u, linkData=%s", topicId, link.Describe().c_str());

        auto portData = link.GetLocalPort();
        ports.insert(std::make_pair(portData, linkProtocol));
    }

    u32 count = 2;
    rtsNotifyPool[topicId].resize(count);

    for (u32 i = 0; i < count; ++i) {
        rtsNotifyPool[topicId][i] = std::make_unique<RtsCntNotify>();
        for (auto& portLinkProtoPair : ports) {
            auto port = portLinkProtoPair.first;
            auto linkProtocol = portLinkProtoPair.second;
            RdmaHandle rdmaHandle = RdmaHandleManager::GetInstance().Get(comm->GetDevicePhyId(), port, linkProtocol);
            if (rdmaHandle == nullptr) {
                string msg = StringFormat(
                    "Failed to get rdma handle for devicePhyId %u, port %u", comm->GetDevicePhyId(), port);
                THROW<NullPtrException>(msg);
            }
            localCntNotifyPool[port][topicId].push_back(
                std::make_unique<LocalCntNotify>(rdmaHandle, rtsNotifyPool[topicId][i].get()));
        }
    };
}

vector<RtsCntNotify*> ConnLocalCntNotifyManager::Get(u32 topicId)
{
    if (rtsNotifyPool.count(topicId) == 0) {
        HCCL_WARNING("Local notify for topicId[%u] not exists, no need to get.", topicId);
        return {};
    }

    vector<RtsCntNotify*> v;
    v.push_back(rtsNotifyPool[topicId][0].get());
    v.push_back(rtsNotifyPool[topicId][1].get());
    return v;
}

bool ConnLocalCntNotifyManager::Destroy()
{
    localCntNotifyPool.clear();
    rtsNotifyPool.clear();
    return true;
}

unordered_map<u32, vector<LocalCntNotify*>> ConnLocalCntNotifyManager::GetTopicIdCntNotifyMap(const PortData& portData)
{
    HCCL_INFO("GetTopicIdCntNotifyMap portData=%s", portData.Describe().c_str());
    unordered_map<u32, vector<LocalCntNotify*>> result;
    for (auto& it : localCntNotifyPool) {
        if (it.first != portData) {
            continue;
        }
        for (auto& topicIdIt : it.second) {
            for (auto& notify : topicIdIt.second) {
                result[topicIdIt.first].push_back(notify.get());
            }
        }
    }

    return result;
}

} // namespace Hccl
