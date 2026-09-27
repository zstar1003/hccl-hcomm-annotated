/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CCU_CHANNEL_MOCK_UTILS_H
#define CCU_CHANNEL_MOCK_UTILS_H

#include <memory>

#define private public
#define protected public

#include "hcom_common.h"
#include "hcomm_primitives.h"
#include "hccl_rank_graph.h"

#include "socket/socket.h"
#include "local_ub_rma_buffer.h"

#include "adapter_rts.h"
#include "orion_adpt_utils.h"

#include "ccu_transport_.h"
#include "ccu_conn.h"

#include "endpoint_monitor.h"

#undef private
#undef protected

inline EndpointDesc MockEndpointDesc(const CommAddr& commAddr, uint32_t devPhyId)
{
    EndpointDesc epDesc{};
    (void)memset_s(&epDesc, sizeof(epDesc), 0, sizeof(epDesc));

    (void)EndpointDescInit(&epDesc, 1);

    epDesc.protocol = CommProtocol::COMM_PROTOCOL_UB_CTP;
    epDesc.commAddr = commAddr;
    epDesc.loc.locType = EndpointLocType::ENDPOINT_LOC_TYPE_DEVICE;
    epDesc.loc.device.devPhyId = devPhyId;
    epDesc.loc.device.superDevId = 0;
    epDesc.loc.device.serverIdx = 0;
    epDesc.loc.device.superPodIdx = 0;

    return epDesc;
}

inline HcommChannelDesc MockHcommChannelDesc(const EndpointDesc& destEpDesc, const HcommSocket socket, void*& memHandle)
{
    HcommChannelDesc channelDesc{};
    (void)memset_s(&channelDesc, sizeof(channelDesc), 0, sizeof(channelDesc));

    (void)HcommChannelDescInit(&channelDesc, 1);

    channelDesc.remoteEndpoint = destEpDesc;
    channelDesc.notifyNum = 1;
    channelDesc.exchangeAllMems = false;
    channelDesc.memHandles = &memHandle;
    channelDesc.memHandleNum = 1;

    channelDesc.socket = socket;
    channelDesc.port = 0;

    return channelDesc;
}

inline std::unique_ptr<Hccl::Socket> MockHcclSocket(const CommAddr& srcAddr, const CommAddr& dstAddr)
{
    Hccl::SocketHandle socketHandle = (Hccl::SocketHandle)(0x12345678);
    Hccl::IpAddress srcIp{}, dstIp{};
    (void)hcomm::CommAddrToIpAddress(srcAddr, srcIp);
    (void)hcomm::CommAddrToIpAddress(dstAddr, dstIp);
    auto role = srcIp < dstIp ? Hccl::SocketRole::SERVER : Hccl::SocketRole::CLIENT;
    constexpr uint32_t listenPort = 60001;
    const std::string tag = "test";
    constexpr Hccl::NicType nicType = Hccl::NicType::DEVICE_NIC_TYPE;
    return make_unique<Hccl::Socket>(socketHandle, srcIp, listenPort, dstIp, tag, role, nicType);
}

inline std::unique_ptr<Hccl::LocalUbRmaBuffer> MockUbRmaBuffer()
{
    constexpr uint64_t addr = 0x12345678;
    constexpr uint64_t size = 0x1234;
    const HcclMemType memType = HcclMemType::HCCL_MEM_TYPE_DEVICE;

    auto localBuffer = std::make_shared<Hccl::Buffer>(reinterpret_cast<uintptr_t>(addr), size, memType);
    return make_unique<Hccl::LocalUbRmaBuffer>(localBuffer);
}

inline CommMemInfo MockCommMemInfo(Hccl::LocalUbRmaBuffer* bufferPtr)
{
    CommMemInfo memInfo{};
    auto& mem = memInfo.mem;
    mem.type = CommMemType::COMM_MEM_TYPE_DEVICE;
    mem.addr = reinterpret_cast<void*>(bufferPtr->GetAddr());
    mem.size = bufferPtr->GetSize();

    memInfo.bufferHandle = static_cast<void*>(bufferPtr);
    return memInfo;
}

inline HcclResult MockCcuConnectionImportJettys(hcomm::CcuConnection* This)
{
    This->innerStatus_ = hcomm::CcuConnection::InnerStatus::CONNECTED;
    This->status_ = hcomm::CcuConnStatus::CONNECTED;
    return HcclResult::HCCL_SUCCESS;
}

inline HcclResult MockCcuTransportGetRmtCke(hcomm::CcuTransport* This, const uint32_t index, uint32_t& rmtCkeId)
{
    if (index >= This->locRes_.ckes.size()) {
        return HcclResult::HCCL_E_PARA;
    }

    rmtCkeId = This->locRes_.ckes[index];
    return HcclResult::HCCL_SUCCESS;
}

inline HcclResult MockCcuTransportGetRmtXn(hcomm::CcuTransport* This, const uint32_t index, uint32_t& rmtCkeId)
{
    if (index >= This->locRes_.xns.size()) {
        return HcclResult::HCCL_E_PARA;
    }

    rmtCkeId = This->locRes_.xns[index];
    return HcclResult::HCCL_SUCCESS;
}

inline void MockCcuChannelGetRes()
{
    MOCKER_CPP(&hcomm::EndpointMonitor::RegisterToEndpointMonitor).stubs().will(returnValue(HcclResult::HCCL_SUCCESS));
    MOCKER_CPP(&hcomm::CcuConnection::ImportJetty).stubs().will(invoke(MockCcuConnectionImportJettys));
    MOCKER_CPP(&hcomm::CcuTransport::RecvDataSize).stubs().will(returnValue(HcclResult::HCCL_SUCCESS));
    MOCKER_CPP(&hcomm::CcuTransport::RecvConnAndTransInfo).stubs().will(returnValue(HcclResult::HCCL_SUCCESS));
    MOCKER_CPP(&hcomm::CcuTransport::RecvDataProcess).stubs().will(returnValue(HcclResult::HCCL_SUCCESS));
    MOCKER_CPP(&hcomm::CcuTransport::CheckFinish).stubs().will(returnValue(HcclResult::HCCL_SUCCESS));
    MOCKER_CPP(&hcomm::CcuTransport::GetRmtCkeByIndex).stubs().will(invoke(MockCcuTransportGetRmtCke));
    MOCKER_CPP(&hcomm::CcuTransport::GetRmtXnByIndex).stubs().will(invoke(MockCcuTransportGetRmtXn));
    MOCKER_CPP(&hcomm::CcuTransport::GetRmtSignalAddrByIndex).stubs().will(returnValue(HcclResult::HCCL_SUCCESS));
    MOCKER_CPP(&hcomm::CcuTransport::GetRmtVarAddrByIndex).stubs().will(returnValue(HcclResult::HCCL_SUCCESS));
}

#endif // CCU_CHANNEL_MOCK_UTILS_H
