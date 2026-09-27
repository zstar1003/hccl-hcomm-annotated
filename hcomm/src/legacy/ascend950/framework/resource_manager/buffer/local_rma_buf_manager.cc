/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "local_rma_buf_manager.h"
#include "internal_exception.h"
#include "rdma_handle_manager.h"
#include "local_ipc_rma_buffer_v2.h"
#include "local_rdma_rma_buffer_v2.h"
#include "local_ub_rma_buffer.h"

#include "log.h"
#include "stl_util.h"
#include "exception_util.h"
#include "communicator_impl.h"

namespace Hccl {

LocalRmaBufManager::LocalRmaBufManager(const CommunicatorImpl& communicator)
    : comm(const_cast<CommunicatorImpl*>(&communicator))
{}

LocalRmaBufManager::~LocalRmaBufManager() { DECTOR_TRY_CATCH("LocalRmaBufManager", Destroy()); }

bool LocalRmaBufManager::IsExist(const string& opTag, const PortData& portData, BufferType bufferType)
{
    return bufs.find(opTag) != bufs.end() && bufs[opTag].find(portData) != bufs[opTag].end()
           && bufs[opTag][portData].find(bufferType) != bufs[opTag][portData].end();
}

LocalRmaBuffer* LocalRmaBufManager::Reg(
    const string& opTag, BufferType bufferType, std::shared_ptr<Buffer> buffer, const PortData& portData,
    LinkProtocol linkProtocol)
{
    if (buffer == nullptr) {
        HCCL_ERROR("input buffer is null");
        return nullptr;
    }
    HCCL_INFO(
        "LocalRmaBufManager::Reg, buffer[%s], opTag[%s], bufferType[%u], portData[%s]", buffer->Describe().c_str(),
        opTag.c_str(), bufferType, portData.Describe().c_str());
    if (IsExist(opTag, portData, bufferType)) {
        string msg = StringFormat(
            "opTag=%s bufferType=%s, buffer=%s already reg to portData=%s", opTag.c_str(),
            bufferType.Describe().c_str(), buffer->Describe().c_str(), portData.Describe().c_str());
        HCCL_DEBUG(msg.c_str());
        return bufs[opTag][portData][bufferType].get();
    }
    if (portData.GetType() == PortDeploymentType::P2P) {
        bufs[opTag][portData][bufferType] = make_unique<LocalIpcRmaBuffer>(buffer);
        return bufs[opTag][portData][bufferType].get();
    } else {
        if (portData.GetProto() == LinkProtoType::RDMA) {
            RdmaHandle rdmaHandle
                = RdmaHandleManager::GetInstance().Get(comm->GetDevicePhyId(), portData, linkProtocol);
            bufs[opTag][portData][bufferType] = make_unique<LocalRdmaRmaBuffer>(buffer, rdmaHandle);
            return bufs[opTag][portData][bufferType].get();
        } else if (portData.GetProto() == LinkProtoType::UB) {
            HCCL_INFO(
                "LocalRmaBufManager::Reg, comm->GetOpAiCpuTSFeatureFlag[%d], comm->GetOpAivFeatureFlag[%d]",
                comm->GetOpAiCpuTSFeatureFlag(), comm->GetOpAivFeatureFlag());
            if (comm->GetOpAiCpuTSFeatureFlag() || comm->GetOpAivFeatureFlag()) { // 算子粒度
                bufs[opTag][portData][bufferType] = make_unique<LocalUbRmaBuffer>(buffer);
            } else {
                RdmaHandle rdmaHandle
                    = RdmaHandleManager::GetInstance().Get(comm->GetDevicePhyId(), portData, linkProtocol);
                bufs[opTag][portData][bufferType] = make_unique<LocalUbRmaBuffer>(buffer, rdmaHandle);
            }
            return bufs[opTag][portData][bufferType].get();
        }
        // 待修改: 仅支持 P2P 和 RDMA
        string msg = StringFormat("PortData=%s is error", portData.Describe().c_str());
        MACRO_THROW(InternalException, msg);
    }
}

LocalRmaBuffer* LocalRmaBufManager::Get(const string& opTag, const PortData& portData, BufferType bufferType)
{
    if (IsExist(opTag, portData, bufferType)) { // if localRmaBuffer exists
        HCCL_INFO(
            "[LocalRmaBufManager][%s] LocalUbRmaBuffer[%s]", __func__,
            bufs[opTag][portData][bufferType]->Describe().c_str());
        return bufs[opTag][portData][bufferType].get();
    }
    HCCL_WARNING(
        "LocalRmaBuffer doesn't exist:opTag[%s], bufferType[%u], portData[%s]", opTag.c_str(), bufferType,
        portData.Describe().c_str());
    return nullptr;
}

void LocalRmaBufManager::Destroy() { bufs.clear(); }

LocalRmaBuffer* LocalRmaBufManager::Get(const PortData& portData)
{
    if (Contain(ccuBufs, portData)) {
        return ccuBufs[portData].get();
    }
    HCCL_WARNING("LocalRmaBuffer doesn't exist at port[%s].", portData.Describe().c_str());
    return nullptr;
}

HcclResult LocalRmaBufManager::Dereg(const string& opTag)
{
    if (bufs.find(opTag) == bufs.end()) {
        HCCL_WARNING("[LocalRmaBufManager::%s] opTag[%s] Cannot find Buffer in bufs.", __func__, opTag.c_str());
        return HCCL_SUCCESS;
    }
    bufs.erase(opTag);
    return HCCL_SUCCESS;
}

} // namespace Hccl
