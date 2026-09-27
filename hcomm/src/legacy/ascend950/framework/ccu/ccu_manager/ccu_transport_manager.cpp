/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_transport_manager.h"

#include <chrono>

#include "coll_operator.h"
#include "exception_util.h"
#include "socket_manager.h"
#include "ccu_communicator.h"
#include "communicator_impl.h"
#include "timeout_exception.h"
#include "internal_exception.h"
#include "coll_service_device_mode.h"
#include "adapter_error_manager_pub.h"
#include "hccl_log_keywords.h"

namespace Hccl {

CcuTransportMgr::CcuTransportMgr(const CommunicatorImpl& comm, const int32_t devLogicId)
    : comm(&comm),
      devLogicId_(devLogicId)
{}

CcuTransportMgr::~CcuTransportMgr()
{
    if (!isDestroyed) {
        DECTOR_TRY_CATCH("CcuTransportMgr", Destroy());
    }
}

CcuTransport* CcuTransportMgr::Get(const LinkData& link)
{
    auto linkIter = ccuLink2TransportMap.find(link);
    if (linkIter != ccuLink2TransportMap.end()) {
        return linkIter->second.get();
    }
    HCCL_WARNING(
        "[CcuTransportMgr::%s] CcuTransport does not existed, "
        "errNo[0x%016llx], localRank[%d], remoteRank[%d]",
        __func__, HCCL_ERROR_CODE(HcclResult::HCCL_E_PTR), link.GetLocalRankId(), link.GetRemoteRankId());

    return nullptr;
}

set<CcuTransport*> CcuTransportMgr::Get(RankId rank)
{
    auto rankIter = ccuRank2TransportsMap.find(rank);
    if (rankIter != ccuRank2TransportsMap.end()) {
        return rankIter->second;
    }
    HCCL_WARNING(
        "[CcuTransportMgr::%s] CcuTransport does not existed, "
        "errNo[0x%016llx], remoteRank[%d]",
        __func__, HCCL_ERROR_CODE(HcclResult::HCCL_E_PTR), rank);
    return set<CcuTransport*>();
}

HcclResult CcuTransportMgr::PrepareCreate(const LinkData& link, CcuTransport*& transport)
{
    auto linkIter = ccuLink2TransportMap.find(link);
    if (linkIter != ccuLink2TransportMap.end()) {
        transport = linkIter->second.get();
        return HcclResult::HCCL_SUCCESS;
    }

    auto ret = CreateTransportByLink(link, transport);
    if (ret == HcclResult::HCCL_E_UNAVAIL) {
        HCCL_WARNING(
            "[CcuTransportMgr::%s]Fail to create CcuTransport. "
            "The above error log can be ignored.",
            __func__);
        comm->PrintChannelInfoCallback();
    }

    return ret;
}

static HcclResult CheckIfLinkProtocolSupport(const LinkData& link)
{
    const auto linkProtocol = link.GetLinkProtocol();
    if (link.GetLinkProtocol() != LinkProtocol::UB_CTP && linkProtocol != LinkProtocol::UB_TP) {
        HCCL_ERROR(
            "[CcuTransportMgr][%s] %s is not supported now, only ub_ctp/ub_tp can be created, "
            "please check, link[%s].",
            __func__, linkProtocol.Describe().c_str(), link.Describe().c_str());
        return HcclResult::HCCL_E_NOT_SUPPORT;
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTransportMgr::CreateTransportByLink(const LinkData& link, CcuTransport*& transport)
{
    HCCL_INFO("[CcuTransportMgr][%s] begin", __func__);
    CHECK_NULLPTR(comm, "[CcuTransportMgr::CreateTransportByLink] comm is nullptr!");
    CHK_RET(CheckIfLinkProtocolSupport(link));

    std::string socketTag = comm->GetEstablishLinkSocketTag();
    SocketConfig socketConfig(link.GetRemoteRankId(), link, socketTag);
    Socket* socket = comm->GetSocketManager().GetConnectedSocket(socketConfig);
    if (socket == nullptr) {
        HCCL_WARNING("[CcuTransportMgr::%s] Fail to get socket via link %s, ", __func__, link.Describe().c_str());
        return HcclResult::HCCL_E_INTERNAL;
    }

    CcuJettyMgr* ccuJettyMgr = dynamic_cast<CollServiceDeviceMode*>(comm->GetCollService())
                                   ->GetCcuInsPreprocessor()
                                   ->GetCcuComm()
                                   ->GetCcuJettyMgr();
    const auto channelJettys = ccuJettyMgr->GetChannelJettys(link);
    const CcuChannelInfo& channelInfo = channelJettys.first;
    const std::vector<CcuJetty*>& ccuJettys = channelJettys.second;

    const auto& locAddr = link.GetLocalAddr();
    const auto& rmtAddr = link.GetRemoteAddr();
    CcuTransport::CcuConnectionType type = link.GetLinkProtocol() == LinkProtocol::UB_CTP ?
                                               CcuTransport::CcuConnectionType::UB_CTP :
                                               CcuTransport::CcuConnectionType::UBC_TP;
    CcuTransport::CcuConnectionInfo connectionInfo{type, locAddr, rmtAddr, channelInfo, ccuJettys};

    std::shared_ptr<LocalUbRmaBuffer> locCclRmaBuffer;
    if (comm->GetCclBuffer() == nullptr) {
        HCCL_ERROR("dataBuf[type=SCRATCH] is nullptr");
        return HcclResult::HCCL_E_INTERNAL;
    }
    HCCL_INFO("[CcuTransportMgr][%s] comm cclBuf[%s]", __func__, comm->GetCclBuffer()->Describe().c_str());
    locCclRmaBuffer = make_shared<LocalUbRmaBuffer>(comm->GetCclBuffer());

    HCCL_INFO("[CcuTransportMgr::CreateTransportByLink] locCclRmaBuffer[%s]", locCclRmaBuffer->Describe().c_str());
    const CcuTransport::CclBufferInfo locCclBufInfo{
        locCclRmaBuffer->GetBuf()->GetAddr(), static_cast<uint32_t>(locCclRmaBuffer->GetBuf()->GetSize()),
        locCclRmaBuffer->GetTokenId(), locCclRmaBuffer->GetTokenValue()};

    // 当前不支持创建非UBC协议的链路
    std::unique_ptr<CcuTransport> transportPtr = nullptr;
    auto ret = CcuCreateTransport(socket, connectionInfo, locCclBufInfo, transportPtr);
    if (ret == HcclResult::HCCL_E_UNAVAIL) {
        HCCL_WARNING(
            "[CcuTransportMgr][%s] failed, some ccu resources are unavailable, "
            "locAddr[%s] rmtAddr[%s].",
            __func__, locAddr.Describe().c_str(), rmtAddr.Describe().c_str());
        return ret;
    }
    CHK_RET(ret);

    tempTransport.emplace_back(link);
    ccuLink2TransportMap[link] = std::move(transportPtr);
    const auto& rawTransportPtr = ccuLink2TransportMap[link].get();
    ccuRank2TransportsMap[link.GetRemoteRankId()].insert(rawTransportPtr);

    transport = rawTransportPtr;
    HCCL_INFO("[CcuTransportMgr][%s] end", __func__);
    return HcclResult::HCCL_SUCCESS;
}

void CcuTransportMgr::WaitTransportsReady(vector<std::pair<CcuTransport*, LinkData>>& transports) const
{
    auto timeout = std::chrono::seconds(EnvConfig::GetInstance().GetSocketConfig().GetLinkTimeOut());
    HcclUs startTime = std::chrono::steady_clock::now();
    while (!transports.empty()) {
        for (auto transIter = transports.begin(); transIter != transports.end();) {
            auto status = (*transIter).first->GetStatus();
            if (status == CcuTransport::TransStatus::CONNECT_FAILED) {
                THROW<InternalException>(
                    "Invalid status occurs when creating transport connection %s!",
                    (*transIter).first->Describe().c_str());
            }

            if (status == CcuTransport::TransStatus::SOCKET_TIMEOUT) {
                RPT_INPUT_ERR(
                    true, "EI0006", std::vector<std::string>({"reason"}),
                    std::vector<std::string>({"CcuTransport wait SOCKET_TIMEOUT."}));
                HCCL_ERROR(
                    "[%s][%s] wait socket establish timeout, CcuTransport wait SOCKET_TIMEOUT, "
                    "commId[%s].",
                    LOG_KEYWORDS_INIT_CHANNEL.c_str(), LOG_KEYWORDS_TIMEOUT.c_str(), comm->GetId().c_str());
                THROW<TimeoutException>(
                    "[CcuTransportMgr][%s] [CcuTransport]%s [LinkData]%s socket timeout, "
                    "commId[%s], please check.",
                    __func__, (*transIter).first->Describe().c_str(), (*transIter).second.Describe().c_str(),
                    comm->GetId().c_str());
            }

            if (status != CcuTransport::TransStatus::READY) {
                ++transIter;
                continue;
            }
            transIter = transports.erase(transIter);
        }

        if ((std::chrono::steady_clock::now() - startTime) >= timeout) {
            string timeoutMsg
                = StringFormat("CcuTransportMgr::WaitTransportReady timeout, commId[%s]", comm->GetId().c_str());
            HCCL_ERROR(
                "[%s][%s] wait socket establish timeout, %s", LOG_KEYWORDS_INIT_CHANNEL.c_str(),
                LOG_KEYWORDS_TIMEOUT.c_str(), timeoutMsg.c_str());
            DumpNotReadyTransports(transports);
            // 上报EI0006
            RPT_INPUT_ERR(
                true, "EI0006", std::vector<std::string>({"reason"}),
                std::vector<std::string>({"CcuTransportMgr wait transports ready timeout."}));
            THROW<InternalException>(timeoutMsg);
        }
    }
}

void CcuTransportMgr::DumpNotReadyTransports(vector<std::pair<CcuTransport*, LinkData>>& transports) const
{
    HCCL_ERROR("Dump ccu timeout transport info, transport size[%u]", transports.size());
    for (auto transIter = transports.begin(); transIter != transports.end(); ++transIter) {
        string allStr = (*transIter).first->Describe();
        size_t pos = allStr.find("Socket");
        if (pos != string::npos) {
            HCCL_ERROR("CcuTransport[%s]", allStr.substr(0, pos).c_str());
            allStr = allStr.substr(pos);
        }
        HCCL_ERROR("CcuTransport[%s]", allStr.c_str());
        HCCL_ERROR("LinkData[%s]", (*transIter).second.Describe().c_str());
    }
}

void CcuTransportMgr::TransportsConnect()
{
    vector<std::pair<CcuTransport*, LinkData>> transLinkPairs = GetUnConfirmedTrans();
    auto op = comm->GetCurrentCollOperator();
    auto accelerator = comm->GetOpExecuteConfig().accState;
    HCCL_INFO("[CcuTransportMgr::TransportsConnect] accelerator[%s]", accelerator.Describe().c_str());

    for (auto& pair : transLinkPairs) {
        auto transport = pair.first;
        transport->SetLocalOpAcceState(accelerator);
        transport->SetHandshakeMsg(op->GetUniqueId());

        HCCL_INFO("[CcuTransportMgr::%s] transport=[%s]", __func__, transport->Describe().c_str());
        HCCL_INFO("[CcuTransportMgr::%s] links=[%s]", __func__, pair.second.Describe().c_str());
        HCCL_INFO("[CcuTransportMgr::%s] opInfo=[%s]", __func__, CollOpToString(*op).c_str());
    }

    WaitTransportsReady(transLinkPairs);

    HCCL_INFO("[CcuTransportMgr::%s] transports connect end.", __func__);
}

void CcuTransportMgr::Confirm()
{
    TransportsConnect();
    tempTransport.clear();
}

vector<std::pair<CcuTransport*, LinkData>> CcuTransportMgr::GetUnConfirmedTrans()
{
    if (tempTransport.size() == 0) {
        HCCL_WARNING("[CcuTransportMgr::%s] UnConfirmedTrans does not exist, please check.", __func__);
        return vector<std::pair<CcuTransport*, LinkData>>();
    }

    vector<std::pair<CcuTransport*, LinkData>> unConfirmedTrans;
    for (const auto& linkData : tempTransport) {
        auto iterLink = ccuLink2TransportMap.find(linkData);
        if (iterLink == ccuLink2TransportMap.end()) {
            THROW<InternalException>(
                "[CcuTransportMgr::%s]Link can't find, linkData[%s]", __func__, linkData.Describe().c_str());
        }
        unConfirmedTrans.emplace_back(std::make_pair(iterLink->second.get(), linkData));
    }
    return unConfirmedTrans;
}

void CcuTransportMgr::Clean()
{
    BatchDeleteJettyInfo batchDeleteJettyInfo;
    // 获取所有transport的unimportJetty和deleteJetty
    for (auto& linkTransPair : ccuLink2TransportMap) {
        if (linkTransPair.second == nullptr) {
            continue;
        }
        auto partDeleteInfo = linkTransPair.second->GetDeleteJettyInfo();
        for (auto& jettyInfo : partDeleteInfo) {
            if (jettyInfo.localJetty != 0) {
                batchDeleteJettyInfo.deleteJettyList[jettyInfo.rdmaHandle].insert(jettyInfo.localJetty);
            }
        }
        auto partUnimportInfo = linkTransPair.second->GetUnimportJettyInfo();
        for (auto& jettyInfo : partUnimportInfo) {
            if (jettyInfo.remoteJetty != 0) {
                batchDeleteJettyInfo.unimportJettyList[jettyInfo.rdmaHandle].insert(jettyInfo.remoteJetty);
            }
        }
    }

    // 循环unimport
    for (auto& tmp : batchDeleteJettyInfo.unimportJettyList) {
        auto& unimportJettys = tmp.second;
        for (auto& unimportJetty : unimportJettys) {
            HrtRaUbUnimportJetty(tmp.first, unimportJetty);
        }
    }

    // 批量销毁jetty
    std::vector<JettyHandle> failJettyHandles;
    for (auto& tmp : batchDeleteJettyInfo.deleteJettyList) {
        auto& rdmaHandle = tmp.first;
        auto& delJettys = tmp.second;
        auto ret = HrtRaCtxQpDestoryBatch(rdmaHandle, delJettys, failJettyHandles);
        for (u64 failJetty : failJettyHandles) {
            HCCL_ERROR("[%s]delete jetty[%llu] fail", __func__, failJetty);
        }
        if (ret == HCCL_E_INTERNAL || ret == HCCL_E_TIMEOUT) {
            HCCL_ERROR(
                "[%s]HrtRaCtxQpDestoryBatch finish, ret[%u], rdmaHandle[%p], originalJettyCount[%u], "
                "undeleteJettyCount[%u]",
                __func__, ret, rdmaHandle, delJettys.size(), failJettyHandles.size());
            continue;
        } else {
            HCCL_INFO(
                "[%s]HrtRaCtxQpDestoryBatch finish, ret[%u], rdmaHandle[%p], originalJettyCount[%u], "
                "undeleteJettyCount[%u]",
                __func__, ret, rdmaHandle, delJettys.size(), failJettyHandles.size());
        }
        failJettyHandles.clear();
    }

    // 清理transport
    for (auto& linkTransPair : ccuLink2TransportMap) {
        if (linkTransPair.second == nullptr) {
            continue;
        }
        if (linkTransPair.second->Clean() != HCCL_SUCCESS) {
            THROW<CcuApiException>("[CcuTransportMgr::%s]CcuTransport clean failed.", __func__);
        }
    }
}

void CcuTransportMgr::Resume()
{
    for (auto iter = ccuLink2TransportMap.begin(); iter != ccuLink2TransportMap.end(); iter++) {
        tempTransport.push_back(iter->first);
    }
}

void CcuTransportMgr::Fallback()
{
    // 遍历TempTransport所有link，分别在ccuLink2TransportMap和ccuRank2TransportsMap删除对应的Transport
    for (const auto& linkId : tempTransport) {
        auto iterLink = ccuLink2TransportMap.find(linkId);
        // 在ccuRank2TransportsMap中要删除的Transport
        auto prepareDelTransport = std::move(iterLink->second);
        ccuLink2TransportMap.erase(iterLink);

        auto iterRank = ccuRank2TransportsMap.find(linkId.GetRemoteRankId());
        auto iterRankSet = std::move(iterRank->second).find(prepareDelTransport.get());
        if (iterRankSet != std::move(iterRank->second).end()) {
            iterRank->second.erase(iterRankSet);
            if (iterRank->second.size() == 0) {
                ccuRank2TransportsMap.erase(iterRank);
            }
        }
    }

    tempTransport.clear();
}

void CcuTransportMgr::Destroy()
{
    isDestroyed = true;
    Clean();
    ccuLink2TransportMap.clear();
    ccuRank2TransportsMap.clear();
}

void CcuTransportMgr::RecoverTransportsConnect()
{
    vector<std::pair<CcuTransport*, LinkData>> transLinkPairs = GetUnConfirmedTrans();
    auto accelerator = comm->GetOpExecuteConfig().accState;
    HCCL_INFO("[CcuTransportMgr::TransportsConnect] accelerator[%s]", accelerator.Describe().c_str());
    for (auto& pair : transLinkPairs) {
        auto transport = pair.first;

        u32 crcValue{0};
        HCCL_INFO("[RecoverMemTransport]commptr=%p", comm);

        if (comm->IsWorldGroup()) {
            // 判断是否在框内
            if (comm->GetNeighboorRanks().find(pair.second.GetRemoteRankId()) != comm->GetNeighboorRanks().end()) {
                // 在框内使用带LocalID的CRC值
                crcValue = comm->GetRanktableCrc(true);
            } else {
                // 不在框内使用不带LocalID的CRC值
                crcValue = comm->GetRanktableCrc(false);
            }
        }

        // 握手消息定义，包括 通信算子数目，rankTable CRC，通信步骤字段
        CollOperator op{};
        op.opTag = std::to_string(comm->GetCollOpIndex()) + "_" + std::to_string(crcValue) + "_"
                   + std::to_string(comm->GetStep());
        transport->SetLocalOpAcceState(accelerator);
        transport->SetHandshakeMsg(op.GetUniqueId());
        HCCL_INFO("[CcuTransportMgr::%s] transport=[%s]", __func__, transport->Describe().c_str());
        HCCL_INFO("[CcuTransportMgr::%s] links=[%s]", __func__, pair.second.Describe().c_str());
    }

    WaitTransportsRecoverReady(transLinkPairs);

    HCCL_INFO("[CcuTransportMgr::%s] transports connect end.", __func__);
}

void CcuTransportMgr::RecoverConfirm()
{
    RecoverTransportsConnect();
    tempTransport.clear();
}

void CcuTransportMgr::WaitTransportsRecoverReady(vector<std::pair<CcuTransport*, LinkData>>& transports) const
{
    constexpr u32 waitTransportReadyTimeoutMs = 10 * 1000; // 待修改，定义最大等待10秒

    auto timeout = std::chrono::milliseconds(waitTransportReadyTimeoutMs);
    HcclUs startTime = std::chrono::steady_clock::now();
    while (!transports.empty()) {
        for (auto transIter = transports.begin(); transIter != transports.end();) {
            auto status = (*transIter).first->GetStatus();
            if (status == CcuTransport::TransStatus::CONNECT_FAILED) {
                THROW<InternalException>(
                    "Invalid status occurs when creating transport connection %s!",
                    (*transIter).first->Describe().c_str());
            }

            if (status != CcuTransport::TransStatus::READY) {
                ++transIter;
                continue;
            }
            transIter = transports.erase(transIter);
        }

        if ((std::chrono::steady_clock::now() - startTime) >= timeout) {
            THROW<InternalException>("WaitTransportReady timeout, commId[%s]", comm->GetId().c_str());
        }
    }
}

} // namespace Hccl
