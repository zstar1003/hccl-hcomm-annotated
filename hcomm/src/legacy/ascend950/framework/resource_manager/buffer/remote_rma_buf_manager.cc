/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "remote_rma_buf_manager.h"
#include "communicator_impl.h"
#include "rdma_handle_manager.h"
#include "internal_exception.h"
namespace Hccl {

RemoteRmaBufManager::RemoteRmaBufManager(const CommunicatorImpl& communicator)
    : comm(const_cast<CommunicatorImpl*>(&communicator))
{}

RemoteRmaBuffer*
RemoteRmaBufManager::GetRemoteRmaBuffer(const string& opTag, const LinkData& linkData, BufferType bufType)
{
    auto tagIter = remoteBufMap.find(opTag);
    if (tagIter != remoteBufMap.end()) {
        auto linkDataIter = tagIter->second.find(linkData);
        if (linkDataIter != tagIter->second.end()) {
            auto bufTypeIter = linkDataIter->second.find(bufType);
            if (bufTypeIter != linkDataIter->second.end()) {
                return bufTypeIter->second.get();
            }
        }
    }
    HCCL_WARNING(
        "WARNING: RemoteRmaBuffer does not exist, "
        "errNo[0x%016llx], opTag[%s], linkData[%s], bufType[%s]",
        HCCL_ERROR_CODE(HcclResult::HCCL_E_PTR), opTag.c_str(), linkData.Describe().c_str(),
        bufType.Describe().c_str());

    return nullptr;
}

unique_ptr<RemoteRmaBuffer> RemoteRmaBufManager::Create(const LinkData& linkData) const
{
    if (linkData.GetType() == PortDeploymentType::P2P) {
        HCCL_INFO("Create remote Ipc RMA buffer.");
        return make_unique<RemoteIpcRmaBuffer>();
    } else {
        auto linkProtocol = linkData.GetLinkProtocol();
        if (linkProtocol == LinkProtocol::ROCE) {
            RdmaHandle rdmaHandle = RdmaHandleManager::GetInstance().Get(
                comm->GetDevicePhyId(), linkData.GetLocalPort(), linkData.GetLinkProtocol());
            return make_unique<RemoteRdmaRmaBuffer>(rdmaHandle);
        } else if (linkProtocol == LinkProtocol::UB_CTP || linkProtocol == LinkProtocol::UB_TP) {
            RdmaHandle rdmaHandle = RdmaHandleManager::GetInstance().Get(
                comm->GetDevicePhyId(), linkData.GetLocalPort(), linkData.GetLinkProtocol());
            return make_unique<RemoteUbRmaBuffer>(rdmaHandle);
        }
        string msg = StringFormat("LinkData[%s] is error", linkData.Describe().c_str());
        THROW<InternalException>(msg);
        return nullptr;
    }
}

void RemoteRmaBufManager::Bind(
    unique_ptr<RemoteRmaBuffer> remoteRmaBuf, const string& opTag, const LinkData& linkData, BufferType bufType)
{
    remoteBufMap[opTag][linkData][bufType] = std::move(remoteRmaBuf);
}

} // namespace Hccl
