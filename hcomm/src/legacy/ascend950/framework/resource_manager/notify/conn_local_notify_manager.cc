/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "conn_local_notify_manager.h"

#include "rdma_handle_manager.h"
#include "communicator_impl.h"
#include "invalid_params_exception.h"

#include "ipc_local_notify.h"
#include "rdma_local_notify.h"
#include "ub_local_notify.h"

namespace Hccl {

ConnLocalNotifyManager::ConnLocalNotifyManager(CommunicatorImpl* communicator) : comm(communicator) {}

ConnLocalNotifyManager::~ConnLocalNotifyManager() { DECTOR_TRY_CATCH("ConnLocalNotifyManager", Destroy()); }

bool ConnLocalNotifyManager::IsExist(RankId remoteRankId, const LinkData& linkData)
{
    return notifyPool.count(remoteRankId) != 0 && notifyPool[remoteRankId].count(linkData) != 0;
}

void ConnLocalNotifyManager::ApplyFor(RankId remoteRankId, const LinkData& linkData)
{
    HCCL_INFO("Local notify for remoteRankId[%d] and linkData[%s] alloc.", remoteRankId, linkData.Describe().c_str());
    if (IsExist(remoteRankId, linkData)) {
        HCCL_WARNING(
            "Local notify for remoteRankId[%d] and linkData[%s] already exists, no need to alloc.", remoteRankId,
            linkData.Describe().c_str());
        return;
    }

    u32 count = 3; // 待修改: 需要定义GetCount()
    notifyPool[remoteRankId][linkData].resize(count);

    for (u32 i = 0; i < count; ++i) {
        if (linkData.GetType() == PortDeploymentType::P2P) {
            notifyPool[remoteRankId][linkData][i]
                = make_unique<IpcLocalNotify>(comm->GetOpAiCpuTSFeatureFlag()); // 算子粒度
            continue;
        } else if (linkData.GetType() == PortDeploymentType::DEV_NET) {
            auto linkProtocol = linkData.GetLinkProtocol();
            if (linkProtocol == LinkProtocol::ROCE) {
                RdmaHandle rdmaHandle = RdmaHandleManager::GetInstance().Get(
                    comm->GetDevicePhyId(), linkData.GetLocalPort(), linkProtocol);
                notifyPool[remoteRankId][linkData][i]
                    = make_unique<RdmaLocalNotify>(rdmaHandle, comm->GetOpAiCpuTSFeatureFlag()); // 算子粒度
                continue;
            } else if (
                linkProtocol == LinkProtocol::UB_CTP || linkProtocol == LinkProtocol::UB_TP
                || linkProtocol == LinkProtocol::UBOE || linkProtocol == LinkProtocol::UB_RTP) {
                RdmaHandle rdmaHandle = RdmaHandleManager::GetInstance().Get(
                    comm->GetDevicePhyId(), linkData.GetLocalPort(), linkProtocol);
                notifyPool[remoteRankId][linkData][i]
                    = make_unique<UbLocalNotify>(rdmaHandle, comm->GetOpAiCpuTSFeatureFlag()); // 算子粒度
            } else {
                // 待修改: 仅支持 P2P 和 RDMA 申请 notify
                string msg = StringFormat(
                    "Unsupported %s of link %s", linkProtocol.Describe().c_str(), linkData.Describe().c_str());
                THROW<InvalidParamsException>(msg);
            }
        } else {
            // 待修改: 仅支持 P2P 和 RDMA 申请 notify
            string msg = StringFormat(
                "Unsupported %s of link %s", linkData.GetType().Describe().c_str(), linkData.Describe().c_str());
            THROW<InvalidParamsException>(msg);
        }
    };
}

bool ConnLocalNotifyManager::Release(RankId remoteRankId, const LinkData& linkData)
{
    if (!IsExist(remoteRankId, linkData)) {
        HCCL_WARNING(
            "Notify for remoteRankId[%d] and linkData[%p] does not exist, no need to release.", remoteRankId,
            &linkData);
        return true;
    }

    notifyPool[remoteRankId].erase(linkData);
    if (notifyPool[remoteRankId].empty()) {
        notifyPool.erase(remoteRankId);
    }
    return true;
}

vector<BaseLocalNotify*> ConnLocalNotifyManager::Get(RankId remoteRankId, const LinkData& linkData)
{
    vector<BaseLocalNotify*> v;

    if (!IsExist(remoteRankId, linkData)) {
        HCCL_WARNING(
            "Get Local notify for remoteRankId[%d] and linkData[%s] does not exist", remoteRankId,
            linkData.Describe().c_str());
        return v;
    }

    for (const auto& i : notifyPool[remoteRankId][linkData]) {
        v.emplace_back(i.get());
    }

    return v;
}

bool ConnLocalNotifyManager::Destroy()
{
    notifyPool.clear();
    return true;
}

} // namespace Hccl
