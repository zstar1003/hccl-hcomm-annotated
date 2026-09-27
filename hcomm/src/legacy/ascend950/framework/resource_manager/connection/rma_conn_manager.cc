/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "rma_conn_manager.h"
#include "p2p_connection.h"
#include "communicator_impl.h"
#include "rdma_handle_manager.h"
#include "dev_rdma_connection.h"
#include "dev_ub_connection.h"
#include "null_ptr_exception.h"
#include "exception_util.h"
#include "socket_manager.h"
#include "timeout_exception.h"
namespace Hccl {

RmaConnManager::RmaConnManager(const CommunicatorImpl& comm) : isDestroyed(false), comm(&comm)
{
    HCCL_INFO("AICPU: RmaConnManager init");
}

RmaConnManager::~RmaConnManager()
{
    if (!isDestroyed) {
        DECTOR_TRY_CATCH("RmaConnManager", Destroy());
    }
}

unique_ptr<RmaConnection>
RmaConnManager::CreateRdmaConn(Socket* socket, const std::string& tag, const LinkData& linkData) const
{
    CHECK_NULLPTR(socket, "[RmaConnManager::CreateRdmaConn] socket is nullptr!");
    RdmaHandle rdmaHandle = RdmaHandleManager::GetInstance().Get(
        comm->GetDevicePhyId(), linkData.GetLocalPort(), linkData.GetLinkProtocol());

    OpMode opMode = comm->GetCurrentCollOperator()->opMode;
    unique_ptr<DevRdmaConnection> rmaNetConn = make_unique<DevRdmaConnection>(socket, rdmaHandle, opMode);
    QpHandle qpHandle = rmaNetConn->GetHandle();

    auto buffer = comm->GetDataBufferManager().Get(tag, BufferType::SCRATCH);
    if (buffer == nullptr) {
        THROW<NullPtrException>(StringFormat("RmaConnManager::CreateRdmaConn ptr is null"));
    }
    RaMrInfo bufInfo{};
    bufInfo.addr = reinterpret_cast<void*>(buffer->GetAddr());
    bufInfo.size = buffer->GetSize();
    bufInfo.access = static_cast<u32>(RA_ACCESS_LOCAL_WRITE) | static_cast<u32>(RA_ACCESS_REMOTE_WRITE);
    HrtRaMrReg(qpHandle, bufInfo);
    return std::unique_ptr<RmaConnection>(rmaNetConn.release());
}

unique_ptr<RmaConnection> RmaConnManager::CreateUbConn(
    Socket* socket, const std::string& tag, const LinkData& linkData, const HrtUbJfcMode jfcMode)
{
    RdmaHandle rdmaHandle = RdmaHandleManager::GetInstance().Get(
        comm->GetDevicePhyId(), linkData.GetLocalPort(), linkData.GetLinkProtocol());
    OpMode opMode = comm->GetCurrentCollOperator()->opMode;
    HCCL_INFO(
        "[RmaConnManager][%s]opMode[%d],linkData[%s],devicePhyId[%u], tag[%s]", __func__, static_cast<int32_t>(opMode),
        linkData.Describe().c_str(), comm->GetDevicePhyId(), tag.c_str());

    unique_ptr<DevUbConnection> ubConn = nullptr;
    locAddr = linkData.GetLocalAddr();
    rmtAddr = linkData.GetRemoteAddr();
    IpAddress locIpv4Addr = locAddr;
    IpAddress rmtIpv4Addr = rmtAddr;
    HCCL_INFO(
        "[RmaConnManager][%s] LinkProtocol[%s], locAddr[%s], rmtAddr[%s]", __func__,
        linkData.GetLinkProtocol().Describe().c_str(), locAddr.Describe().c_str(), rmtAddr.Describe().c_str());
    if (linkData.GetLinkProtocol() == LinkProtocol::UBOE || linkData.GetLinkProtocol() == LinkProtocol::UB_RTP) {
        // socket建链状态ok，并交换数据
        WaitUboeSocketReady(socket, linkData);
    }

    bool devUsed = comm->GetOpAiCpuTSFeatureFlag();
    if (linkData.GetLinkProtocol() == LinkProtocol::UB_TP) {
        ubConn = make_unique<DevUbTpConnection>(rdmaHandle, locAddr, rmtAddr, opMode, devUsed, jfcMode);
    } else if (linkData.GetLinkProtocol() == LinkProtocol::UBOE) {
        ubConn = make_unique<DevUbUboeConnection>(
            rdmaHandle, locAddr, rmtAddr, opMode, devUsed, jfcMode, locIpv4Addr, rmtIpv4Addr);
    } else if (linkData.GetLinkProtocol() == LinkProtocol::UB_RTP) {
        ubConn
            = make_unique<DevUbRtpConnection>(rdmaHandle, locAddr, rmtAddr, opMode, devUsed, jfcMode, locAddr, rmtAddr);
    } else {
        ubConn = make_unique<DevUbCtpConnection>(rdmaHandle, locAddr, rmtAddr, opMode, devUsed, jfcMode);
    }
    return std::unique_ptr<RmaConnection>(ubConn.release());
}

RmaConnection* RmaConnManager::Create(const std::string& tag, const LinkData& linkData, const HrtUbJfcMode jfcMode)
{
    HCCL_INFO(
        "Create tag = [%s], remoteRank[%d] LinkData[%s] ", tag.c_str(), linkData.GetRemoteRankId(),
        linkData.Describe().c_str());
    RmaConnection* rmaConnPtr = Get(tag, linkData);
    if (rmaConnPtr != nullptr) {
        HCCL_INFO("has inited");
        return rmaConnPtr;
    }

    std::string socketTag = comm->GetEstablishLinkSocketTag();
    SocketConfig socketConfig(linkData.GetRemoteRankId(), linkData, socketTag);
    Socket* socket = comm->GetSocketManager().GetConnectedSocket(socketConfig);
    HCCL_INFO("socketTag = [%s]", socketTag.c_str());
    HCCL_INFO("[RmaConnManager::%s] linkData Type[%s]", __func__, linkData.GetType().Describe().c_str());
    std::unique_ptr<RmaConnection> rmaConn = nullptr;
    if (linkData.GetType() == PortDeploymentType::P2P) {
        rmaConn = make_unique<P2PConnection>(socket, tag);
    } else if (linkData.GetType() == PortDeploymentType::DEV_NET) {
        auto linkProtocol = linkData.GetLinkProtocol();
        HCCL_INFO("[RmaConnManager::%s] linkData linkProtocol[%s]", __func__, linkProtocol.Describe().c_str());
        if (linkProtocol == LinkProtocol::ROCE) {
            rmaConn = CreateRdmaConn(socket, tag, linkData);
        } else if (
            linkProtocol == LinkProtocol::UB_TP || linkProtocol == LinkProtocol::UB_CTP
            || linkProtocol == LinkProtocol::UBOE || linkProtocol == LinkProtocol::UB_RTP) {
            rmaConn = CreateUbConn(socket, tag, linkData, jfcMode);
        }
    }

    if (rmaConn == nullptr) {
        auto msg = StringFormat("Fail to create RmaConnection via link %s", linkData.Describe().c_str());
        THROW<NullPtrException>(msg.c_str());
    }

    rmaConn->Connect();
    rmaConnectionMap[tag][linkData] = std::move(rmaConn);

    return rmaConnectionMap[tag][linkData].get();
}

void RmaConnManager::RecreateAllConns()
{
    for (const auto& connPair : rmaConnectionMap) {
        const string& tag = connPair.first;
        for (const auto& linkDataConnPair : connPair.second) {
            const LinkData& linkData = linkDataConnPair.first;
            if (linkDataConnPair.second != nullptr) {
                rmaConnectionMap[tag][linkData] = nullptr;
                Create(tag, linkData);
            }
        }
    }
}

RmaConnection* RmaConnManager::Get(const std::string& tag, const LinkData& linkData)
{
    auto tagIter = rmaConnectionMap.find(tag);
    if (tagIter != rmaConnectionMap.end()) {
        auto linkDataIter = tagIter->second.find(linkData);
        if (linkDataIter != tagIter->second.end()) {
            return linkDataIter->second.get();
        }
    }
    HCCL_WARNING(
        "WARNING: RmaConnection not existed, "
        "errNo[0x%016llx], localRank[%d], remoteRank[%d], tag[%s]",
        HCCL_ERROR_CODE(HcclResult::HCCL_E_PTR), comm->GetMyRank(), linkData.GetRemoteRankId(), tag.c_str());

    return nullptr;
}

std::vector<RmaConnection*> RmaConnManager::GetOpTagConns(const std::string& tag) const
{
    std::vector<RmaConnection*> rmaConnList;
    auto opTagIter = rmaConnectionMap.find(tag);
    if (opTagIter != rmaConnectionMap.end()) {
        for (auto& linkDataConn : opTagIter->second) {
            rmaConnList.emplace_back(linkDataConn.second.get());
        }
        return rmaConnList;
    }
    HCCL_WARNING(
        "WARNING: RmaConnection not existed, "
        "errNo[0x%016llx], localRank[%d], tag[%s]",
        HCCL_ERROR_CODE(HcclResult::HCCL_E_PTR), comm->GetMyRank(), tag.c_str());
    return rmaConnList;
}

void RmaConnManager::Release(const std::string& tag, const LinkData& linkData)
{
    auto tagIter = rmaConnectionMap.find(tag);
    if (tagIter != rmaConnectionMap.end()) {
        auto linkDataIter = tagIter->second.find(linkData);
        if (linkDataIter != tagIter->second.end()) {
            tagIter->second.erase(linkDataIter);
        }
    }
}

void RmaConnManager::GetDeleteJettys(BatchDeleteJettyInfo& batchDeleteJettyInfo)
{
    // 获取要删除的连接
    DevUbConnection* ubConn = nullptr;
    for (auto& connPair : rmaConnectionMap) {
        for (auto& linkDataConnPair : connPair.second) {
            if (linkDataConnPair.second != nullptr) {
                ubConn = dynamic_cast<DevUbConnection*>(linkDataConnPair.second.get());
                if (ubConn == nullptr) {
                    continue;
                }

                const auto& rdmaHandle = ubConn->GetRdmaHandle();
                auto& remoteJettyHandle = ubConn->GetRemoteJettyHandle();
                if (rdmaHandle && remoteJettyHandle != 0) {
                    batchDeleteJettyInfo.unimportJettyList[rdmaHandle].insert(remoteJettyHandle);
                    remoteJettyHandle = 0;
                }
                ubConn->ReleaseTp();
                auto& jettyHandle = ubConn->GetJettyHandle();
                if (jettyHandle != 0) {
                    batchDeleteJettyInfo.deleteJettyList[rdmaHandle].insert(jettyHandle);
                    jettyHandle = 0;
                }
                linkDataConnPair.second = nullptr;
            }
        }
    }
}

void RmaConnManager::BatchDeleteJettys()
{
    BatchDeleteJettyInfo batchDeleteJettyInfo;
    GetDeleteJettys(batchDeleteJettyInfo);
    for (auto& unimportJettys : batchDeleteJettyInfo.unimportJettyList) {
        if (!RdmaHandleManager::GetInstance().IsHandleValid(unimportJettys.first)) {
            HCCL_WARNING(
                "[RmaConnManager][%s] skip HrtRaUbUnimportJetty batch, "
                "rdmaHandle=%p invalid (DeInit/DestroyAll done)",
                __func__, unimportJettys.first);
            continue;
        }
        for (auto& unimportJetty : unimportJettys.second) {
            HrtRaUbUnimportJetty(unimportJettys.first, unimportJetty);
        }
    }

    std::vector<JettyHandle> failJettyHandles;
    for (const auto& deleteJettys : batchDeleteJettyInfo.deleteJettyList) {
        if (!RdmaHandleManager::GetInstance().IsHandleValid(deleteJettys.first)) {
            HCCL_WARNING(
                "[RmaConnManager][%s] skip HrtRaCtxQpDestroyBatch, "
                "rdmaHandle=%p invalid (DeInit/DestroyAll done)",
                __func__, deleteJettys.first);
            continue;
        }
        auto ret = HrtRaCtxQpDestoryBatch(deleteJettys.first, deleteJettys.second, failJettyHandles);
        for (u64 failJetty : failJettyHandles) {
            HCCL_ERROR("[%s]delete jetty[%llu] fail", __func__, failJetty);
        }
        if (ret == HCCL_E_INTERNAL || ret == HCCL_E_TIMEOUT) {
            HCCL_ERROR(
                "[%s]HrtRaCtxQpDestoryBatch finish, ret[%u], rdmaHandle[%p], originalJettyCount[%u], "
                "undeleteJettyCount[%u]",
                __func__, ret, deleteJettys.first, deleteJettys.second.size(), failJettyHandles.size());
            continue;
        } else {
            HCCL_INFO(
                "[%s]HrtRaCtxQpDestoryBatch finish, ret[%u], rdmaHandle[%p], originalJettyCount[%u], "
                "undeleteJettyCount[%u]",
                __func__, ret, deleteJettys.first, deleteJettys.second.size(), failJettyHandles.size());
        }
        failJettyHandles.clear();
    }
}

void RmaConnManager::Destroy()
{
    isDestroyed = true;
    BatchDeleteJettys();
    rmaConnectionMap.clear();
}

void RmaConnManager::Clear()
{
    BatchDeleteJettys();
    rmaConnectionMap.clear();
}

std::vector<RmaConnection*> RmaConnManager::GetAllConns() const
{
    std::vector<RmaConnection*> rmaConnList;
    for (const auto& connPair : rmaConnectionMap) {
        for (const auto& linkDataConnPair : connPair.second) {
            if (linkDataConnPair.second != nullptr) {
                rmaConnList.push_back(linkDataConnPair.second.get());
            }
        }
    }
    return rmaConnList;
}

const std::vector<BufferType> BUF_TYPES = {BufferType::SCRATCH, BufferType::INPUT, BufferType::OUTPUT};

void RmaConnManager::BindRemoteRmaBuffers()
{
    for (const auto& connPair : rmaConnectionMap) {
        const string& tag = connPair.first;
        for (const auto& linkDataConnPair : connPair.second) {
            const LinkData& linkData = linkDataConnPair.first;
            for (auto& bufType : BUF_TYPES) {
                RemoteRmaBuffer* remoteRmaBuf
                    = comm->GetRemoteRmaBufManager().GetRemoteRmaBuffer(tag, linkData, bufType);
                if (remoteRmaBuf != nullptr) {
                    rmaConnectionMap[tag][linkData]->Bind(remoteRmaBuf, bufType);
                }
            }
        }
    }
}

void RmaConnManager::BatchCreate(vector<LinkData>& links)
{
    HCCL_INFO(
        "[NsRecovery][Resume]RmaConnManager::BatchCreate, before Create, rmaConnectionMap size[%u]",
        rmaConnectionMap.size());
    const string& tag = comm->GetId();
    for (const auto& linkData : links) {
        if (rmaConnectionMap[tag][linkData] == nullptr) {
            Create(tag, linkData);
        } else {
            HCCL_WARNING(
                "[NsRecovery][Resume]RmaConnManager::BatchCreate, connection has existed, will not recreate, "
                "linkData[%s]",
                linkData.Describe().c_str());
        }
    }
    HCCL_INFO(
        "[NsRecovery][Resume]RmaConnManager::BatchCreate, after Create, rmaConnectionMap size[%u], "
        "rmaConnectionMap[comm->GetId()] size[%u]",
        rmaConnectionMap.size(), rmaConnectionMap[tag].size());
}

bool RmaConnManager::IsSocketReady(Socket* socket, const LinkData& linkData)
{
    if (socket == nullptr) {
        MACRO_THROW(InternalException, StringFormat("%s socket is nullptr, please check", linkData.Describe().c_str()));
    }

    SocketStatus socketStatus = socket->GetAsyncStatus();
    if (socketStatus == SocketStatus::OK) {
        uboeStatus = UboeStatus::SOCKET_OK;
        return true;
    } else if (socketStatus == SocketStatus::TIMEOUT) {
        uboeStatus = UboeStatus::SOCKET_TIMEOUT;
        return false;
    }

    return false;
}

UboeStatus RmaConnManager::GetUboeSocketStatus(Socket* socket, const LinkData& linkData)
{
    if (uboeStatus == UboeStatus::READY) {
        return uboeStatus;
    } else if (uboeStatus == UboeStatus::INIT) {
        ubStatus = UbStatus::INIT;
    }

    if (!IsSocketReady(socket, linkData)) {
        return uboeStatus;
    }

    switch (ubStatus) {
        case UbStatus::INIT:
            ubStatus = UbStatus::SOCKET_OK;
            uboeStatus = UboeStatus::SOCKET_OK;
            break;
        case UbStatus::SOCKET_OK:
            ubStatus = UbStatus::SEND_DATA;
            SendExchangeData(socket, linkData);
            break;
        case UbStatus::SEND_DATA:
            RecvExchangeData(socket, linkData);
            ubStatus = UbStatus::RECV_DATA;
            break;
        case UbStatus::RECV_DATA:
            RecvDataProcess(linkData);
            ubStatus = UbStatus::RECV_FIN;
            uboeStatus = UboeStatus::READY;
            break;
        default:
            break;
    }
    return uboeStatus;
}

void RmaConnManager::WaitUboeSocketReady(Socket* socket, const LinkData& linkData)
{
    HCCL_INFO("[RmaConnManager][%s] begin", __func__);
    auto timeout = std::chrono::seconds(EnvConfig::GetInstance().GetSocketConfig().GetLinkTimeOut());
    HcclUs startTime = std::chrono::steady_clock::now();
    while (true) {
        auto status = GetUboeSocketStatus(socket, linkData);
        if (status == UboeStatus::READY) {
            break;
        }
        if (status == UboeStatus::SOCKET_TIMEOUT) {
            MACRO_THROW(
                TimeoutException, StringFormat(
                                      "[RmaConnManager][%s] %s socket timeout, commId[%s], please check", __func__,
                                      linkData.Describe().c_str(), comm->GetId().c_str()));
        }
        if ((std::chrono::steady_clock::now() - startTime) >= timeout) {
            string timeoutMsg = StringFormat("WaitUboeSocketReady timeout, commId[%s].", comm->GetId().c_str());
            HCCL_ERROR(timeoutMsg.c_str());
            THROW<InternalException>(timeoutMsg);
        }
    }
    HCCL_INFO("[RmaConnManager][%s] end", __func__);
}

void RmaConnManager::Ipv4Pack()
{
    IpAddress locEidAddr;
    RdmaHandleManager::GetInstance().GetEidByIpv4Addr(locAddr, locEidAddr);
    locAddr = locEidAddr;
    HCCL_INFO("[RmaConnManager::%s] locAddr[%s]", __func__, locAddr.Describe().c_str());
    sendData = locAddr.GetUniqueId();
}

void RmaConnManager::Ipv4UnPack(BinaryStream& binaryStream)
{
    IpAddress rmtEidAddr(binaryStream);
    rmtAddr = rmtEidAddr;
    HCCL_INFO("[RmaConnManager::%s] rmtAddr[%s]", __func__, rmtAddr.Describe().c_str());
}

void RmaConnManager::SendExchangeData(Socket* socket, const LinkData& linkData)
{
    Ipv4Pack();
    socket->SendAsync(sendData.data(), sendData.size());
    exchangeDataSize = sendData.size();

    HCCL_INFO("send data %s, size=%llu", linkData.Describe().c_str(), exchangeDataSize);
}

void RmaConnManager::RecvExchangeData(Socket* socket, const LinkData& linkData)
{
    recvData.resize(exchangeDataSize);
    socket->RecvAsync(reinterpret_cast<u8*>(recvData.data()), recvData.size());

    HCCL_INFO("recv data %s, size=%llu", linkData.Describe().c_str(), recvData.size());
}

void RmaConnManager::RecvDataProcess(const LinkData& linkData)
{
    HCCL_INFO(
        "RecvDataProcess: link=%s, size=%llu, exchangeDataSize=%u", linkData.Describe().c_str(), recvData.size(),
        exchangeDataSize);
    BinaryStream binaryStream(recvData);
    Ipv4UnPack(binaryStream);
}

} // namespace Hccl
