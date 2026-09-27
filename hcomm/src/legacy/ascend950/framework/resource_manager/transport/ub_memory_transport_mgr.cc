/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ub_memory_transport_mgr.h"
#include "timeout_exception.h"
#include "communicator_impl.h"
#include "adapter_error_manager_pub.h"
#include "hccl_log_keywords.h"

namespace Hccl {

constexpr u32 AIV_TAG_BUF_INDEX = 1;         // aiv tag buf的下标
constexpr u32 AIV_OFFLOAD_TAG_BUF_INDEX = 2; // aiv offload tag buf的下标
UbMemoryTransportMgr::UbMemoryTransportMgr(const CommunicatorImpl& communicator) : comm(&communicator) {}

UbMemoryTransportMgr::~UbMemoryTransportMgr()
{
    tempTransport.clear();
    ubMemLink2TransportMap.clear();
}
HcclResult UbMemoryTransportMgr::BatchCreateTransport(const std::vector<LinkData>& links)
{
    HCCL_INFO("[%s] start", __func__);
    for (auto& link : links) {
        auto ret = CreateTransportByLink(link);
        if (ret != HcclResult::HCCL_SUCCESS) {
            HCCL_ERROR(
                "[UbMemoryTransportMgr::%s] CreateTransportByLink fail link[%s]", __func__, link.Describe().c_str());
            return ret;
        }
    }

    return HcclResult::HCCL_SUCCESS;
}

std::vector<std::pair<RankId, RemoteIpcRmaBuffer*>> UbMemoryTransportMgr::GetRmtRankId2RmtIpcRmaBufList()
{
    HCCL_INFO("[%s] start", __func__);
    std::vector<std::pair<RankId, RemoteIpcRmaBuffer*>> rankId2RmtIpcRmaBufList{};

    for (const auto& ubMemLink2TransportIter : ubMemLink2TransportMap) {
        auto rmtRank = ubMemLink2TransportIter.first.GetRemoteRankId();
        auto rmtMemBuffer = ubMemLink2TransportIter.second->GetRmtMemBuffer(0);
        rankId2RmtIpcRmaBufList.push_back(std::make_pair(rmtRank, rmtMemBuffer));
    }

    return rankId2RmtIpcRmaBufList;
}

std::vector<std::pair<RankId, uintptr_t>> UbMemoryTransportMgr::GetAllRankId2AivTagBufAddrList()
{
    HCCL_INFO("[%s] start", __func__);
    std::vector<std::pair<RankId, uintptr_t>> rankId2AivTagBufList{};

    for (const auto& ubMemLink2TransportIter : ubMemLink2TransportMap) {
        auto rmtRank = ubMemLink2TransportIter.first.GetRemoteRankId();
        uintptr_t rmtAivTagufferAddr = ubMemLink2TransportIter.second->GetRmtMemBuffer(AIV_TAG_BUF_INDEX)->GetAddr();
        rankId2AivTagBufList.push_back(std::make_pair(rmtRank, rmtAivTagufferAddr));
    }
    rankId2AivTagBufList.push_back(std::make_pair(comm->GetMyRank(), comm->GetAivTagBuffer()->GetAddr()));

    return rankId2AivTagBufList;
}

std::vector<std::pair<RankId, uintptr_t>> UbMemoryTransportMgr::GetAllRankId2AivOffloadTagBufAddrList()

{
    HCCL_INFO("[%s] start", __func__);

    std::vector<std::pair<RankId, uintptr_t>> rankId2AivOffloadTagBufList{};

    for (const auto& ubMemLink2TransportIter : ubMemLink2TransportMap) {
        auto rmtRank = ubMemLink2TransportIter.first.GetRemoteRankId();
        uintptr_t rmtAivTagBufferAddr
            = ubMemLink2TransportIter.second->GetRmtMemBuffer(AIV_OFFLOAD_TAG_BUF_INDEX)->GetAddr();
        rankId2AivOffloadTagBufList.push_back(std::make_pair(rmtRank, rmtAivTagBufferAddr));
    }
    rankId2AivOffloadTagBufList.push_back(std::make_pair(comm->GetMyRank(), comm->GetAivOffloadTagBuffer()->GetAddr()));

    return rankId2AivOffloadTagBufList;
}

HcclResult UbMemoryTransportMgr::CreateTransportByLink(const LinkData& link)
{
    HCCL_INFO("[%s] start", __func__);
    auto linkIter = ubMemLink2TransportMap.find(link);
    if (linkIter != ubMemLink2TransportMap.end()) {
        return HcclResult::HCCL_SUCCESS;
    }
    // 创建socket
    std::string socketTag = comm->GetEstablishLinkSocketTag();
    SocketConfig socketConfig(link.GetRemoteRankId(), link, socketTag);
    Socket* socket = comm->GetSocketManager().GetConnectedSocket(socketConfig);
    if (socket == nullptr) {
        HCCL_WARNING("[UbMemoryTransportMgr::%s] Fail to get socket via link %s, ", __func__, link.Describe().c_str());

        return HcclResult::HCCL_E_INTERNAL;
    }

    std::unique_ptr<UbMemoryTransport> transport = make_unique<UbMemoryTransport>(
        comm->GetCclBuffer(), comm->GetAivTagBuffer(), comm->GetAivOffloadTagBuffer(), socket,
        comm->GetDeviceLogicId());

    if (transport->Init() != HcclResult::HCCL_SUCCESS) {
        HCCL_ERROR("[UbMemoryTransportMgr][%s] transport init fail, link %s", __func__, link.Describe().c_str());
        return HCCL_E_INTERNAL;
    }

    tempTransport.emplace_back(link); // 插入TempTransport中表明Transport并未真正创建成功，需要等待握手确认
    ubMemLink2TransportMap[link] = std::move(transport);
    return HcclResult::HCCL_SUCCESS;
}
void UbMemoryTransportMgr::WaitTransportsReady(vector<std::pair<UbMemoryTransport*, LinkData>>& transports) const
{
    HCCL_INFO("[%s] start", __func__);

    auto timeout = std::chrono::seconds(EnvConfig::GetInstance().GetSocketConfig().GetLinkTimeOut());
    HcclUs startTime = std::chrono::steady_clock::now();
    while (!transports.empty()) {
        for (auto transIter = transports.begin(); transIter != transports.end();) {
            auto status = (*transIter).first->GetStatus();
            if (status == UbMemoryTransport::UBTransportStatus::READY) {
                transIter = transports.erase(transIter);
            } else if (status == UbMemoryTransport::UBTransportStatus::CONNECT_FAILED) {
                THROW<InternalException>(StringFormat(
                    "Invalid status occurs when creating transport connection %s!",
                    (*transIter).first->Describe().c_str()));
            } else if (status == UbMemoryTransport::UBTransportStatus::SOCKET_TIMEOUT) {
                RPT_INPUT_ERR(
                    true, "EI0006", std::vector<std::string>({"reason"}),
                    std::vector<std::string>({"UbMemoryTransport wait SOCKET_TIMEOUT."}));
                HCCL_ERROR(
                    "[%s][%s] wait socket establish timeout, UbMemoryTransport wait SOCKET_TIMEOUT, "
                    "commId[%s].",
                    LOG_KEYWORDS_INIT_CHANNEL.c_str(), LOG_KEYWORDS_TIMEOUT.c_str(), comm->GetId().c_str());
                THROW<TimeoutException>(StringFormat(
                    "[UbMemoryTransportMgr][%s] [UbMemoryTransport]%s [LinkData]%s "
                    "socket timeout, commId[%s], please check",
                    __func__, (*transIter).first->Describe().c_str(), (*transIter).second.Describe().c_str(),
                    comm->GetId().c_str()));
            } else {
                ++transIter;
            }
        }

        if ((std::chrono::steady_clock::now() - startTime) >= timeout) {
            // 上报故障码EI0006
            RPT_INPUT_ERR(
                true, "EI0006", std::vector<std::string>({"reason"}),
                std::vector<std::string>({"UbMemoryTransportMgr wait transports ready timeout."}));
            HCCL_ERROR(
                "[%s][%s] wait socket establish timeout, UbMemoryTransportMgr wait transports ready timeout, "
                "commId[%s].",
                LOG_KEYWORDS_INIT_CHANNEL.c_str(), LOG_KEYWORDS_TIMEOUT.c_str(), comm->GetId().c_str());
            THROW<InternalException>(
                "UbMemoryTransportMgr::WaitTransportReady timeout, commId[%s]", comm->GetId().c_str());
        }
    }
}

vector<std::pair<UbMemoryTransport*, LinkData>> UbMemoryTransportMgr::GetUnconfirmedTrans()
{
    HCCL_INFO("[%s] start", __func__);
    if (tempTransport.size() == 0) {
        HCCL_WARNING("[UbMemoryTransportMgr::%s] UnConfirmedTrans does not exist, please check.", __func__);
        return vector<std::pair<UbMemoryTransport*, LinkData>>();
    }

    vector<std::pair<UbMemoryTransport*, LinkData>> unConfirmedTrans;
    for (const auto& linkId : tempTransport) {
        auto iterLink = ubMemLink2TransportMap.find(linkId);
        unConfirmedTrans.emplace_back(std::make_pair(iterLink->second.get(), linkId));
    }
    return unConfirmedTrans;
}

void UbMemoryTransportMgr::TransportsConnect()
{
    HCCL_INFO("[%s] start", __func__);
    // transport建链
    vector<std::pair<UbMemoryTransport*, LinkData>> transLinkPairs = GetUnconfirmedTrans();
    auto op = comm->GetCurrentCollOperator();
    auto accelerator = comm->GetOpExecuteConfig().accState;
    HCCL_INFO("[UbMemoryTransportMgr::TransportsConnect] accelerator[%s]", accelerator.Describe().c_str());
    for (auto& pair : transLinkPairs) {
        auto transport = pair.first;
        transport->SetLocalOpAcceState(accelerator);
        transport->SetHandshakeMsg(op->GetUniqueId());

        HCCL_INFO("[UbMemoryTransport::%s] transport=[%s]", __func__, transport->Describe().c_str());
        HCCL_INFO("[UbMemoryTransport::%s] links=[%s]", __func__, pair.second.Describe().c_str());
    }

    // 轮询Connect
    WaitTransportsReady(transLinkPairs);

    tempTransport.clear();
    HCCL_INFO("[UbMemoryTransport::%s] transports connect end.", __func__);
}
} // namespace Hccl
