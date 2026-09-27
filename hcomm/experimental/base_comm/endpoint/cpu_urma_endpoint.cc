/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "cpu_urma_endpoint.h"
#include <algorithm>
#include "endpoint_mgr.h"
#include "host_peer_ra_init.h"
#include "log.h"
#include "urma_mem.h"
#include "adapter_rts_common.h"
#include "orion_adapter_hccp.h"
#include "server_socket_manager.h"
#include "hccp_peer_manager.h"

namespace hcomm_experimental {
namespace {
    constexpr uint32_t kHostResourceId = 0U;
}

CpuUrmaEndpoint::CpuUrmaEndpoint(const EndpointDesc& endpointDesc) : Endpoint(endpointDesc) {}

CpuUrmaEndpoint::~CpuUrmaEndpoint() noexcept
{
    std::lock_guard<std::mutex> lock(portMutex_);
    if (dynamicPort_ != HCCL_INVALID_PORT) {
        ServerSocketStopListen(dynamicPort_);
    }
    dynamicPort_ = HCCL_INVALID_PORT;
}

HcclResult CpuUrmaEndpoint::Init()
{
    HCCL_INFO("[%s] localEndpoint protocol[%d]", __func__, endpointDesc_.protocol);

    Hccl::IpAddress ipAddr{};
    CHK_RET(hcomm::CommAddrToIpAddress(endpointDesc_.commAddr, ipAddr));
    RaSocketSetWhiteListStatus(1); // PEER模式需要手动开启白名单模式
    CHK_RET(InitHostPeerRaOnce(kHostResourceId, "CpuUrmaEndpoint"));
    auto& rdmaHandleMgr = Hccl::RdmaHandleManager::GetInstance();
    ctxHandle_ = static_cast<void*>(
        rdmaHandleMgr.GetByAddr(kHostResourceId, Hccl::LinkProtoType::UB, ipAddr, Hccl::PortDeploymentType::HOST_NET));
    CHK_PTR_NULL(ctxHandle_);
    HCCL_INFO(
        "CpuUrmaEndpoint::%s success, hostResourceId[%u], ipAddr[%s], ctxHandle[%p]", __func__, kHostResourceId,
        ipAddr.Describe().c_str(), ctxHandle_);

    EXCEPTION_CATCH(regedMemMgr_ = std::make_unique<UbRegedMemMgr>(), return HCCL_E_PARA);
    this->regedMemMgr_->rdmaHandle_ = this->ctxHandle_;
    return HCCL_SUCCESS;
}

HcclResult CpuUrmaEndpoint::ServerSocketListen(const uint32_t port)
{
    Hccl::IpAddress ipAddr{};
    CHK_RET(hcomm::CommAddrToIpAddress(endpointDesc_.commAddr, ipAddr));

    Hccl::DevNetPortType type = Hccl::DevNetPortType(Hccl::ConnectProtoType::UB);
    Hccl::PortData localPort = Hccl::PortData(kHostResourceId, type, 0, ipAddr);

    HCCL_INFO(
        "[CpuUrmaEndpoint::%s] hostResourceId[%u] ipAddress[%s]", __func__, kHostResourceId, ipAddr.Describe().c_str());

    uint32_t requestPort = port;
    CHK_RET(hcomm::ServerSocketManager::GetInstance().ServerSocketStartListen(
        localPort, Hccl::NicType::HOST_NIC_TYPE, kHostResourceId, &requestPort));

    return HCCL_SUCCESS;
}

HcclResult CpuUrmaEndpoint::ServerSocketStopListen(const uint32_t port)
{
    Hccl::IpAddress ipAddr{};
    CHK_RET(hcomm::CommAddrToIpAddress(endpointDesc_.commAddr, ipAddr));

    Hccl::DevNetPortType type = Hccl::DevNetPortType(Hccl::ConnectProtoType::UB);
    Hccl::PortData localPort = Hccl::PortData(kHostResourceId, type, 0, ipAddr);
    CHK_RET(hcomm::ServerSocketManager::GetInstance().ServerSocketStopListen(
        localPort, Hccl::NicType::HOST_NIC_TYPE, port));

    return HCCL_SUCCESS;
}

HcclResult CpuUrmaEndpoint::ServerSocketGetListenPort(uint32_t* port)
{
    std::lock_guard<std::mutex> lock(portMutex_);
    CHK_PTR_NULL(port);
    Hccl::IpAddress localIpAddr{};
    CHK_RET(hcomm::CommAddrToIpAddress(endpointDesc_.commAddr, localIpAddr));

    Hccl::DevNetPortType portType = Hccl::DevNetPortType(Hccl::ConnectProtoType::UB);
    Hccl::PortData portData = Hccl::PortData(kHostResourceId, portType, 0, localIpAddr);

    HCCL_INFO(
        "[CpuUrmaEndpoint::%s] hostResourceId[%u] ipAddress[%s]", __func__, kHostResourceId,
        localIpAddr.Describe().c_str());

    if (dynamicPort_ != HCCL_INVALID_PORT) {
        *port = dynamicPort_;
        HCCL_INFO("[CpuUrmaEndpoint::%s] already listening, return existing port[%u]", __func__, dynamicPort_);
        return HCCL_SUCCESS;
    }

    uint32_t requestPort = 0;
    CHK_RET(hcomm::ServerSocketManager::GetInstance().ServerSocketStartListen(
        portData, Hccl::NicType::HOST_NIC_TYPE, kHostResourceId, &requestPort));
    if (requestPort == 0 || requestPort == HCCL_INVALID_PORT) {
        HCCL_ERROR("[CpuUrmaEndpoint::%s] get listen port failed, port is invalid.", __func__);
        return HCCL_E_NETWORK;
    }
    dynamicPort_ = requestPort;
    *port = dynamicPort_;

    return HCCL_SUCCESS;
}

HcclResult CpuUrmaEndpoint::RegisterMemory(HcommMem mem, const char* memTag, void** memHandle)
{
    CHK_RET(this->regedMemMgr_->RegisterMemory(mem, memTag, memHandle));
    return HCCL_SUCCESS;
}

HcclResult CpuUrmaEndpoint::UnregisterMemory(void* memHandle)
{
    CHK_RET(this->regedMemMgr_->UnregisterMemory(memHandle));
    return HCCL_SUCCESS;
}

HcclResult CpuUrmaEndpoint::MemoryExport(void* memHandle, void** memDesc, uint32_t* memDescLen)
{
    CHK_RET(this->regedMemMgr_->MemoryExport(this->endpointDesc_, memHandle, memDesc, memDescLen));
    return HCCL_SUCCESS;
}

HcclResult CpuUrmaEndpoint::MemoryImport(const void* memDesc, uint32_t descLen, HcommMem* outMem)
{
    CHK_RET(this->regedMemMgr_->MemoryImport(memDesc, descLen, outMem));
    return HCCL_SUCCESS;
}

HcclResult CpuUrmaEndpoint::MemoryUnimport(const void* memDesc, uint32_t descLen)
{
    CHK_RET(this->regedMemMgr_->MemoryUnimport(memDesc, descLen));
    return HCCL_SUCCESS;
}

HcclResult CpuUrmaEndpoint::GetAllMemHandles(void** memHandles, uint32_t* memHandleNum)
{
    CHK_RET(this->regedMemMgr_->GetAllMemHandles(memHandles, memHandleNum));
    return HCCL_SUCCESS;
}

} // namespace hcomm_experimental
