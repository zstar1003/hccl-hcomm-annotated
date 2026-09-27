/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "mem_transport_manager.h"
#include "rdma_handle_manager.h"
#include "communicator_impl.h"
#include "ub_mem_transport.h"
#include "urma_direct_transport.h"
#include "p2p_transport.h"
#include "cnt_notify_res_helper.h"
#include "timeout_exception.h"
#include "local_rma_buf_manager.h"
namespace Hccl {
MemTransportManager::MemTransportManager(const CommunicatorImpl& communicator) : comm(&communicator) {}

MemTransportManager::~MemTransportManager() {}

std::vector<BaseLocalNotify*> MemTransportManager::GetNotifyVec(const LinkData& linkData) const
{
    return comm->GetConnLocalNotifyManager().Get(linkData.GetRemoteRankId(), linkData);
}

const std::vector<BufferType> PIPE_BUFFER_TYPE = {BufferType::INPUT, BufferType::OUTPUT, BufferType::SCRATCH};

std::vector<LocalRmaBuffer*>
MemTransportManager::GetBufferVec(const std::string& opTag, const LinkData& linkData, OpMode opMode) const
{
    HCCL_DEBUG("[MemTransportManager][%s] opMode[%s]", __func__, opMode.Describe().c_str());
    std::vector<LocalRmaBuffer*> result;
    if (opMode == OpMode::OPBASE) {
        result.push_back(nullptr); // 单算子 input/output 为null,
        result.push_back(nullptr);
        auto res = comm->GetLocalRmaBufManager().Get(opTag, linkData.GetLocalPort(), BufferType::SCRATCH);
        result.push_back(res);
    } else {
        for (auto& bufType : PIPE_BUFFER_TYPE) {
            auto res = comm->GetLocalRmaBufManager().Get(opTag, linkData.GetLocalPort(), bufType);
            result.push_back(res); // INPUT/OUTPUT/SCRATCH 都交换
        }
    }
    return result;
}

std::vector<RmaConnection*> MemTransportManager::GetConnVec(const std::string& opTag, const LinkData& linkData) const
{
    std::vector<RmaConnection*> result;
    result.push_back(comm->GetRmaConnManager().Get(opTag, linkData));
    return result;
}

void MemTransportManager::CreateOpbasedUbMemTransport(
    BaseMemTransport::CommonLocRes& locRes, BaseMemTransport::Attribution& attr, const LinkData& linkData,
    const Socket& socket)
{
    auto topicIdCntNotifyVecMap = comm->GetConnLocalCntNotifyManager().GetTopicIdCntNotifyMap(linkData.GetLocalPort());
    CntNotifyResHelper tool;
    BaseMemTransport::LocCntNotifyRes locCntNotifyRes = tool.GetCntNotifyRes(topicIdCntNotifyVecMap);
    HCCL_INFO("locCntNotifyRes=%s, linkData=%s", locCntNotifyRes.Describe().c_str(), linkData.Describe().c_str());
    RdmaHandle rdmaHandle = RdmaHandleManager::GetInstance().Get(
        comm->GetDevicePhyId(), linkData.GetLocalPort(), linkData.GetLinkProtocol());

    // DFX：注册transportCallBack, 用于信息保存
    auto transportCallBack = MemTransportCallback(linkData, comm->GetMirrorTaskManager());
    auto ubMemTransport
        = make_unique<UbMemTransport>(locRes, attr, linkData, socket, rdmaHandle, locCntNotifyRes, transportCallBack);
    opTagOpbasedMap[linkData] = std::move(ubMemTransport);
}

void MemTransportManager::CreateOffloadUbMemTransport(
    const string& opTag, BaseMemTransport::CommonLocRes& locRes, BaseMemTransport::Attribution& attr,
    const LinkData& linkData, const Socket& socket)
{
    auto topicIdCntNotifyVecMap = comm->GetConnLocalCntNotifyManager().GetTopicIdCntNotifyMap(linkData.GetLocalPort());
    CntNotifyResHelper tool;
    BaseMemTransport::LocCntNotifyRes locCntNotifyRes = tool.GetCntNotifyRes(topicIdCntNotifyVecMap);
    HCCL_INFO("locCntNotifyRes=%s, linkData=%s", locCntNotifyRes.Describe().c_str(), linkData.Describe().c_str());
    RdmaHandle rdmaHandle = RdmaHandleManager::GetInstance().Get(
        comm->GetDevicePhyId(), linkData.GetLocalPort(), linkData.GetLinkProtocol());

    // DFX：注册transportCallBack, 用于信息保存
    auto transportCallBack = MemTransportCallback(linkData, comm->GetMirrorTaskManager());
    auto ubMemTransport
        = make_unique<UbMemTransport>(locRes, attr, linkData, socket, rdmaHandle, locCntNotifyRes, transportCallBack);
    opTagOffloadMap[opTag][linkData] = std::move(ubMemTransport);
}

BaseMemTransport* MemTransportManager::CreateOpbasedMemTransport(const LinkData& linkData)
{
    auto op = comm->GetCurrentCollOperator();
    HCCL_INFO("link=%s Entry CreateMemTransport", linkData.Describe().c_str());
    HCCL_INFO("Entry CreateMemTransport, opInfo=[%s]", CollOpToString(*op).c_str());
    BaseMemTransport::CommonLocRes locRes;
    locRes.notifyVec = GetNotifyVec(linkData);
    HCCL_INFO("link=%s get notifyVec OK", linkData.Describe().c_str());

    // buffer 来自localRmaBufferManager, input/output/scratch
    locRes.bufferVec = GetBufferVec(comm->GetId(), linkData, OpMode::OPBASE);
    HCCL_INFO("link=%s get bufferVec OK", linkData.Describe().c_str());

    // connection是一个，来自 RmaConnManager
    locRes.connVec = GetConnVec(comm->GetId(), linkData);
    HCCL_INFO("link=%s get connVec OK", linkData.Describe().c_str());

    HCCL_INFO("locRes=%s", locRes.Describe().c_str());

    BaseMemTransport::Attribution attr;
    attr.devicePhyId = linkData.GetLocalPort().GetId();
    // 握手消息定义，未来包括 cann版本号，rankTable CRC等字段
    auto accelerator = comm->GetOpExecuteConfig().accState;
    HCCL_INFO("[MemTransportManager::%s] accelerator[%s]", __func__, accelerator.Describe().c_str());
    attr.opAcceState = accelerator;
    attr.handshakeMsg = op->GetUniqueId();

    SocketConfig socketConfig(linkData.GetRemoteRankId(), linkData, comm->GetEstablishLinkSocketTag());
    auto socket = comm->GetSocketManager().GetConnectedSocket(socketConfig);
    if (socket == nullptr) {
        throw std::runtime_error("CreateMemTransport GetConnectedSocket failed, socket is nullptr");
    }
    if (linkData.GetType() == PortDeploymentType::P2P) {
        opTagOpbasedMap[linkData] = make_unique<P2PTransport>(locRes, attr, linkData, *socket);
    } else if (linkData.GetType() == PortDeploymentType::DEV_NET) {
        auto linkProtocol = linkData.GetLinkProtocol();
        if (linkProtocol == LinkProtocol::UB_CTP || linkProtocol == LinkProtocol::UB_TP
            || linkProtocol == LinkProtocol::UBOE || linkProtocol == LinkProtocol::UB_RTP) {
            CreateOpbasedUbMemTransport(locRes, attr, linkData, *socket);
        } else {
            THROW<NullPtrException>(StringFormat("linkData=%s is error", linkData.Describe().c_str()));
        }
    } else {
        THROW<NullPtrException>(StringFormat("linkData=%s is error", linkData.Describe().c_str()));
    }

    opTagOpbasedMap[linkData]->Establish();

    newOpbasedTransports[linkData] = 0;

    HCCL_INFO("link=%s OK.", linkData.Describe().c_str());
    HCCL_INFO("create transport %s OK.", opTagOpbasedMap[linkData]->Describe().c_str());

    return opTagOpbasedMap[linkData].get();
}
BaseMemTransport* MemTransportManager::CreateOffloadMemTransport(const std::string& opTag, const LinkData& linkData)
{
    auto op = comm->GetCurrentCollOperator();
    HCCL_INFO("link=%s Entry CreateMemTransport", linkData.Describe().c_str());
    HCCL_INFO("Entry CreateMemTransport, opInfo=[%s]", CollOpToString(*op).c_str());
    BaseMemTransport::CommonLocRes locRes;
    locRes.notifyVec = GetNotifyVec(linkData);
    HCCL_INFO("link=%s get notifyVec OK", linkData.Describe().c_str());

    // buffer 来自localRmaBufferManager, input/output/scratch
    locRes.bufferVec = GetBufferVec(opTag, linkData, OpMode::OFFLOAD);
    HCCL_INFO("link=%s get bufferVec OK", linkData.Describe().c_str());

    // connection是一个，来自 RmaConnManager
    std::string tag = comm->GetOpAiCpuTSFeatureFlag() == true ? comm->GetId() : opTag; // 算子粒度
    locRes.connVec = GetConnVec(tag, linkData);
    HCCL_INFO("link=%s get connVec OK", linkData.Describe().c_str());

    HCCL_INFO("locRes=%s", locRes.Describe().c_str());

    BaseMemTransport::Attribution attr;
    attr.devicePhyId = linkData.GetLocalPort().GetId();
    // 握手消息定义，未来包括 cann版本号，rankTable CRC等字段
    auto accelerator = comm->GetOpExecuteConfig().accState;
    HCCL_INFO("[MemTransportManager::%s] accelerator[%s]", __func__, accelerator.Describe().c_str());
    attr.opAcceState = accelerator;
    attr.handshakeMsg = op->GetUniqueId();

    SocketConfig socketConfig(linkData.GetRemoteRankId(), linkData, comm->GetEstablishLinkSocketTag());
    auto socket = comm->GetSocketManager().GetConnectedSocket(socketConfig);
    if (socket == nullptr) {
        throw std::runtime_error("CreateMemTransport GetConnectedSocket failed, socket is nullptr");
    }

    if (linkData.GetType() == PortDeploymentType::P2P) {
        opTagOffloadMap[opTag][linkData] = make_unique<P2PTransport>(locRes, attr, linkData, *socket);
    } else if (linkData.GetType() == PortDeploymentType::DEV_NET) {
        auto linkProtocol = linkData.GetLinkProtocol();
        if (linkProtocol == LinkProtocol::UB_CTP || linkProtocol == LinkProtocol::UB_TP
            || linkProtocol == LinkProtocol::UBOE || linkProtocol == LinkProtocol::UB_RTP) {
            CreateOffloadUbMemTransport(opTag, locRes, attr, linkData, *socket);
        } else {
            THROW<NullPtrException>(StringFormat("linkData=%s is error", linkData.Describe().c_str()));
        }
    } else {
        THROW<NullPtrException>(StringFormat("linkData=%s is error", linkData.Describe().c_str()));
    }

    opTagOffloadMap[opTag][linkData]->Establish();

    newOffloadTransports[opTag][linkData] = 0;

    HCCL_INFO("link=%s OK.", linkData.Describe().c_str());
    HCCL_INFO("create transport %s OK.", opTagOffloadMap[opTag][linkData]->Describe().c_str());

    return opTagOffloadMap[opTag][linkData].get();
}

void MemTransportManager::DumpNotReadyTransportsOpbased()
{
    HCCL_ERROR("Dump opbased timeout transport info, transport size[%u]", newOpbasedTransports.size());
    for (auto linkIt = newOpbasedTransports.begin(); linkIt != newOpbasedTransports.end(); ++linkIt) {
        auto transportPtr = opTagOpbasedMap[linkIt->first].get();
        HCCL_ERROR("Transport info[%s]", transportPtr->Describe().c_str());
        HCCL_ERROR("Linkdata info[%s]", transportPtr->GetLinkDescInfo().c_str());
        HCCL_ERROR("Socket info[%s]", transportPtr->DescribeSocket().c_str());
    }
}

void MemTransportManager::DumpNotReadyTransportsOffload(const std::string& opTag)
{
    HCCL_ERROR("Dump offload timeout transport info, transport size[%u]", newOffloadTransports[opTag].size());
    for (auto linkIt = newOffloadTransports[opTag].begin(); linkIt != newOffloadTransports[opTag].end(); ++linkIt) {
        auto transportPtr = opTagOffloadMap[opTag][linkIt->first].get();
        HCCL_ERROR("Transport info[%s]", transportPtr->Describe().c_str());
        HCCL_ERROR("Linkdata info[%s]", transportPtr->GetLinkDescInfo().c_str());
        HCCL_ERROR("Socket info[%s]", transportPtr->DescribeSocket().c_str());
    }
}

void MemTransportManager::DumpNotReadyTransportsUrma()
{
    HCCL_RUN_INFO("[MemTransportManager][%s] start", __func__);
    for (auto& it : urmaDirectMap_) {
        auto status = it.second->GetStatus();
        if (status != TransportStatus::READY) {
            HCCL_INFO("linkData[%s] status[%s]", it.first.Describe().c_str(), status.Describe().c_str());
        }
    }
}

bool MemTransportManager::IsAllOpbasedTransportReady()
{
    bool result = true;
    // 当前只针对新增的transports做资源交换和op校验
    for (auto linkIt = newOpbasedTransports.begin(); linkIt != newOpbasedTransports.end();) {
        auto status = opTagOpbasedMap[linkIt->first]->GetStatus();
        if (status != TransportStatus::READY) { // 任意一个没有ready，结果为 false
            if (status == TransportStatus::SOCKET_TIMEOUT) {
                MACRO_THROW(
                    TimeoutException,
                    StringFormat(
                        "[MemTransportManager][%s] %s socket timeout, commId[%s], please check", __func__,
                        opTagOpbasedMap[linkIt->first]->GetLinkDescInfo().c_str(), comm->GetId().c_str()));
            }
            result = false;
            ++linkIt;
        } else {
            HCCL_INFO("linkData[%s], status[%s].", linkIt->first.Describe().c_str(), status.Describe().c_str());
            linkIt = newOpbasedTransports.erase(linkIt);
        }
    }
    return result;
}

bool MemTransportManager::IsAllOneSidedTransportReady()
{
    bool result = true;
    // 当前只针对新增的transports做资源交换和op校验
    for (auto linkIt = newOneSidedTransports.begin(); linkIt != newOneSidedTransports.end();) {
        auto status = oneSidedMap[linkIt->first]->GetStatus();
        if (status != TransportStatus::READY) { // 任意一个没有ready，结果为 false
            result = false;
            ++linkIt;
        } else {
            HCCL_INFO("linkData[%s] status[%s]", linkIt->first.Describe().c_str(), status.Describe().c_str());
            linkIt = newOneSidedTransports.erase(linkIt);
        }
    }
    return result;
}

bool MemTransportManager::IsAllOffloadTransportReady(const std::string& opTag)
{
    bool result = true;
    // 当前只针对新增的transports做资源交换和op校验
    for (auto linkIt = newOffloadTransports[opTag].begin(); linkIt != newOffloadTransports[opTag].end();) {
        auto status = opTagOffloadMap[opTag][linkIt->first]->GetStatus();
        if (status != TransportStatus::READY) { // 任意一个没有ready，结果为 false
            if (status == TransportStatus::SOCKET_TIMEOUT) {
                MACRO_THROW(
                    TimeoutException,
                    StringFormat(
                        "[MemTransportManager][%s] %s socket timeout, commId[%s], please check", __func__,
                        opTagOffloadMap[opTag][linkIt->first]->GetLinkDescInfo().c_str(), comm->GetId().c_str()));
            }
            result = false;
            ++linkIt;
        } else {
            HCCL_INFO(
                "opTag[%s] linkData[%s] status[%s]", opTag.c_str(), linkIt->first.Describe().c_str(),
                status.Describe().c_str());
            linkIt = newOffloadTransports[opTag].erase(linkIt);
        }
    }
    return result;
}

bool MemTransportManager::IsAllTransportReady()
{
    bool result = true;
    for (auto& tagIt : opTagOffloadMap) {
        for (auto& it : tagIt.second) {
            auto status = it.second->GetStatus();
            if (status != TransportStatus::READY) { // 任意一个没有ready，结果为 false
                if (status == TransportStatus::SOCKET_TIMEOUT) {
                    MACRO_THROW(
                        TimeoutException, StringFormat(
                                              "[MemTransportManager][%s] %s socket timeout, commId[%s], please check",
                                              __func__, it.second->GetLinkDescInfo().c_str(), comm->GetId().c_str()));
                }
                result = false;
            }
        }
    }
    for (auto& it : opTagOpbasedMap) {
        auto status = it.second->GetStatus();
        if (status != TransportStatus::READY) { // 任意一个没有ready，结果为 false
            if (status == TransportStatus::SOCKET_TIMEOUT) {
                MACRO_THROW(
                    TimeoutException, StringFormat(
                                          "[MemTransportManager][%s] %s socket timeout, commId[%s], please check",
                                          __func__, it.second->GetLinkDescInfo().c_str(), comm->GetId().c_str()));
            }
            result = false;
        }
    }
    for (auto& it : urmaDirectMap_) {
        auto status = it.second->GetStatus();
        if (status != TransportStatus::READY) { // 任意一个没有ready，结果为 false
            if (status == TransportStatus::SOCKET_TIMEOUT) {
                MACRO_THROW(
                    TimeoutException, StringFormat(
                                          "[MemTransportManager][%s] %s socket timeout, commId[%s], please check",
                                          __func__, it.second->GetLinkDescInfo().c_str(), comm->GetId().c_str()));
            }
            result = false;
        }
    }
    return result;
}

void MemTransportManager::BatchBuildOpbasedTransports(const vector<LinkData>& links)
{
    HCCL_INFO("Batch build opbased transports start, link num is [%u]", links.size());
    for (auto& link : links) {
        if (opTagOpbasedMap.find(link) != opTagOpbasedMap.end()) {
            HCCL_WARNING("linkData=%s already exists, do not need to create transport", link.Describe().c_str());
            continue;
        }
        CreateOpbasedMemTransport(link);
    }
}

void MemTransportManager::BatchBuildOffloadTransports(const std::string& opTag, const vector<LinkData>& links)
{
    HCCL_INFO("Batch build offload transports start, link num is [%u]", links.size());
    for (auto& link : links) {
        if (opTagOffloadMap.find(opTag) != opTagOffloadMap.end()
            && opTagOffloadMap[opTag].find(link) != opTagOffloadMap[opTag].end()) {
            HCCL_WARNING(
                "opTag=%s, linkData=%s already exists, do not need to create transport", opTag.c_str(),
                link.Describe().c_str());
            continue;
        }
        CreateOffloadMemTransport(opTag, link);
    }
}

BaseMemTransport* MemTransportManager::GetOpbasedTransport(const LinkData& linkData)
{
    if (opTagOpbasedMap.find(linkData) == opTagOpbasedMap.end()) {
        HCCL_WARNING("GetOpbasedTransport, linkData=%s find transport is null", linkData.Describe().c_str());
        return nullptr;
    }
    return opTagOpbasedMap[linkData].get();
}

BaseMemTransport* MemTransportManager::GetOffloadTransport(const std::string& opTag, const LinkData& linkData)
{
    if (opTagOffloadMap.find(opTag) == opTagOffloadMap.end()) {
        HCCL_WARNING(
            "GetOffloadTransport, opTag=%s, linkData=%s find transport is null", opTag.c_str(),
            linkData.Describe().c_str());
        return nullptr;
    }
    if (opTagOffloadMap[opTag].find(linkData) == opTagOffloadMap[opTag].end()) {
        HCCL_WARNING(
            "GetOffloadTransport, opTag=%s, linkData=%s find transport is null", opTag.c_str(),
            linkData.Describe().c_str());
        return nullptr;
    }

    return opTagOffloadMap[opTag][linkData].get();
}

BaseMemTransport* MemTransportManager::GetUrmaDirectTransport(const LinkData& linkData)
{
    if (urmaDirectMap_.find(linkData) == urmaDirectMap_.end()) {
        HCCL_WARNING("GetUrmaDirectTransport, linkData=%s find transport is null", linkData.Describe().c_str());
        return nullptr;
    }
    return urmaDirectMap_[linkData].get();
}

std::vector<char> MemTransportManager::GetOneSidedPackedData()
{
    if (!IsAllOneSidedTransportReady()) {
        std::string msg = StringFormat("status of some transports is not ready, please check.");
        THROW<InternalException>(msg);
    }

    std::vector<char> result;
    BinaryStream binaryStream;
    u32 mapSize = oneSidedMap.size();
    binaryStream << mapSize;

    if (mapSize == 0) {
        HCCL_WARNING("mem transport oneSidedMap is empty");
    }

    for (auto& it : oneSidedMap) {
        binaryStream << it.first.GetUniqueId();
        binaryStream << it.second->GetUniqueId();
        HCCL_INFO(
            "MemTransportManager::GetOneSidedPackedData: %s %s.", it.first.Describe().c_str(),
            it.second->Describe().c_str());
    }

    binaryStream.Dump(result);
    return result;
}

std::vector<HcclAiRMAWQ> MemTransportManager::GetUrmaWqs()
{
    if (!IsAllTransportReady()) {
        std::string msg = StringFormat("status of some transports is not ready, please check.");
        THROW<InternalException>(msg);
    }

    std::vector<HcclAiRMAWQ> wqs;
    auto links = comm->GetFullMeshLinks();
    for (auto& link : links) {
        if (urmaDirectMap_.find(link) == urmaDirectMap_.end()) {
            HCCL_WARNING(
                "[MemTransportManager][GetUrmaWqs]GetUrmaDirectTransport, linkData=%s find transport is null",
                link.Describe().c_str());
            continue;
        }
        UrmaDirectTransport* urmaTransport = reinterpret_cast<UrmaDirectTransport*>(urmaDirectMap_[link].get());

        wqs.push_back(urmaTransport->GetAiRMAWQ());
        HCCL_INFO("MemTransportManager::GetUrmaWq: %s.", link.Describe().c_str());
    }
    return wqs;
}

std::vector<HcclAiRMACQ> MemTransportManager::GetUrmaCqs()
{
    if (!IsAllTransportReady()) {
        std::string msg = StringFormat("status of some transports is not ready, please check.");
        THROW<InternalException>(msg);
    }

    std::vector<HcclAiRMACQ> cqs;
    auto links = comm->GetFullMeshLinks();
    for (auto& link : links) {
        if (urmaDirectMap_.find(link) == urmaDirectMap_.end()) {
            HCCL_WARNING(
                "[MemTransportManager][GetUrmaWqs]GetUrmaDirectTransport, linkData=%s find transport is null",
                link.Describe().c_str());
            continue;
        }
        UrmaDirectTransport* urmaTransport = reinterpret_cast<UrmaDirectTransport*>(urmaDirectMap_[link].get());

        cqs.push_back(urmaTransport->GetAiRMACQ());
        HCCL_INFO("MemTransportManager::GetUrmaCq: %s.", link.Describe().c_str());
    }

    return cqs;
}

std::vector<char> MemTransportManager::GetOpbasedPackedData()
{
    if (!IsAllOpbasedTransportReady()) {
        std::string msg = StringFormat("status of some transports is not ready, please check.");
        THROW<InternalException>(msg);
    }

    std::vector<char> result;
    BinaryStream binaryStream;
    u32 mapSize = opTagOpbasedMap.size();
    binaryStream << mapSize;

    if (mapSize == 0) {
        HCCL_WARNING("mem transport opTagOpbasedMap is empty");
    }

    for (auto& it : opTagOpbasedMap) {
        binaryStream << it.first.GetUniqueId();
        binaryStream << it.second->GetUniqueId();
        HCCL_INFO("MemTransportManager::GetOpbasedPackedData: %s.", it.first.Describe().c_str());
    }

    binaryStream.Dump(result);
    return result;
}

std::vector<char> MemTransportManager::GetOffloadPackedData(const std::string& opTag)
{
    if (!IsAllOffloadTransportReady(opTag)) {
        std::string msg
            = StringFormat("status of some transports is not ready, please check. opTag[%s]", opTag.c_str());
        THROW<InternalException>(msg);
    }

    std::vector<char> result;
    BinaryStream binaryStream;
    u32 mapSize = 0;

    auto transpMap = opTagOffloadMap.find(opTag);
    if (transpMap != opTagOffloadMap.end()) {
        mapSize = transpMap->second.size();
        binaryStream << mapSize;
        for (auto& it : transpMap->second) {
            binaryStream << it.first.GetUniqueId();
            binaryStream << it.second->GetUniqueId();
            HCCL_INFO("MemTransportManager::GetOffloadPackedData: %s.", it.first.Describe().c_str());
        }
    } else {
        HCCL_WARNING("mem transport opTagOffloadMap is empty for opTag[%s]", opTag.c_str());
        binaryStream << mapSize;
    }

    binaryStream.Dump(result);
    return result;
}

std::vector<char> MemTransportManager::GetPackedAllTransportData()
{
    /* 打包的数据：
    {
        u32 opbasedMapSize
        对opTagOpbasedMap里的每个pair:
            vector<char> Opbase linkdata
            vector<char> Opbase transport
        u32 opTagNum
        对opTagOffloadMap里的每个opTag:
            vector<char> opTag
            u32 offloadMapSize
            对opTagOffloadMap[opTag]里的每个pair:
                vector<char> Offload linkdata
                vector<char> Offload transport
    }
    */

    std::vector<char> result;
    BinaryStream binaryStream;

    u32 opbasedMapSize = opTagOpbasedMap.size();
    HCCL_INFO("GetPackedAllTransportData: opbasedMapSize=%u", opbasedMapSize);
    binaryStream << opbasedMapSize;
    for (auto& it : opTagOpbasedMap) {
        binaryStream << it.first.GetUniqueId();
        binaryStream << it.second->GetUniqueId();
    }

    u32 opTagNum = opTagOffloadMap.size();
    HCCL_INFO("GetPackedAllTransportData: opTagNum=%u", opTagNum);
    binaryStream << opTagNum;
    for (auto& opTagIt : opTagOffloadMap) {
        std::string opTag = opTagIt.first;
        std::vector<char> opTagVec(opTag.begin(), opTag.end());
        binaryStream << opTagVec;
        u32 offloadMapSize = opTagIt.second.size();
        binaryStream << offloadMapSize;
        for (auto& it : opTagIt.second) {
            binaryStream << it.first.GetUniqueId();
            binaryStream << it.second->GetUniqueId();
        }
    }

    binaryStream.Dump(result);
    return result;
}

BaseMemTransport* MemTransportManager::RecoverOpbasedMemTransport(const LinkData& linkData)
{
    HCCL_INFO("link=%s Entry CreateMemTransport", linkData.Describe().c_str());
    BaseMemTransport::CommonLocRes locRes;
    locRes.notifyVec = GetNotifyVec(linkData);
    HCCL_INFO("link=%s get notifyVec OK", linkData.Describe().c_str());

    // buffer 来自localRmaBufferManager, input/output/scratch
    locRes.bufferVec = GetBufferVec(comm->GetId(), linkData, OpMode::OPBASE);
    HCCL_INFO("link=%s get bufferVec OK", linkData.Describe().c_str());

    // connection是一个，来自 RmaConnManager
    locRes.connVec = GetConnVec(comm->GetId(), linkData);
    HCCL_INFO("link=%s get connVec OK", linkData.Describe().c_str());

    HCCL_INFO("locRes=%s", locRes.Describe().c_str());

    BaseMemTransport::Attribution attr;
    attr.devicePhyId = linkData.GetLocalPort().GetId();

    u32 crcValue{0};
    HCCL_INFO("[RecoverMemTransport]commptr=%p", comm);

    if (comm->IsWorldGroup()) {
        // 判断是否在框内
        if (comm->GetNeighboorRanks().find(linkData.GetRemoteRankId()) != comm->GetNeighboorRanks().end()) {
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
    auto accelerator = comm->GetOpExecuteConfig().accState;
    HCCL_INFO("[MemTransportManager::CreateOpbasedMemTransport] accelerator[%s]", accelerator.Describe().c_str());
    attr.opAcceState = accelerator;
    attr.handshakeMsg = op.GetUniqueId();

    SocketConfig socketConfig(linkData.GetRemoteRankId(), linkData, comm->GetEstablishLinkSocketTag());
    auto socket = comm->GetSocketManager().GetConnectedSocket(socketConfig);
    if (socket == nullptr) {
        throw std::runtime_error("CreateMemTransport GetConnectedSocket failed, socket is nullptr");
    }
    if (linkData.GetType() == PortDeploymentType::P2P) {
        opTagOpbasedMap[linkData] = make_unique<P2PTransport>(locRes, attr, linkData, *socket);
    } else if (linkData.GetType() == PortDeploymentType::DEV_NET) {
        auto linkProtocol = linkData.GetLinkProtocol();
        if (linkProtocol == LinkProtocol::UB_CTP || linkProtocol == LinkProtocol::UB_TP) {
            CreateOpbasedUbMemTransport(locRes, attr, linkData, *socket);
        } else {
            THROW<NullPtrException>(StringFormat("linkData=%s is error", linkData.Describe().c_str()));
        }
    } else {
        THROW<NullPtrException>(StringFormat("linkData=%s is error", linkData.Describe().c_str()));
    }

    opTagOpbasedMap[linkData]->Establish();

    newOpbasedTransports[linkData] = 0;

    HCCL_INFO("link=%s OK.", linkData.Describe().c_str());
    HCCL_INFO("create transport %s OK.", opTagOpbasedMap[linkData]->Describe().c_str());

    return opTagOpbasedMap[linkData].get();
}

BaseMemTransport* MemTransportManager::RecoverOffloadMemTransport(const std::string& opTag, const LinkData& linkData)
{
    HCCL_INFO("link=%s Entry CreateMemTransport", linkData.Describe().c_str());
    BaseMemTransport::CommonLocRes locRes;
    locRes.notifyVec = GetNotifyVec(linkData);
    HCCL_INFO("link=%s get notifyVec OK", linkData.Describe().c_str());

    // buffer 来自localRmaBufferManager, input/output/scratch
    locRes.bufferVec = GetBufferVec(opTag, linkData, OpMode::OFFLOAD);
    HCCL_INFO("link=%s get bufferVec OK", linkData.Describe().c_str());

    // connection是一个，来自 RmaConnManager
    locRes.connVec = GetConnVec(opTag, linkData);
    HCCL_INFO("link=%s get connVec OK", linkData.Describe().c_str());

    HCCL_INFO("locRes=%s", locRes.Describe().c_str());

    BaseMemTransport::Attribution attr;
    attr.devicePhyId = linkData.GetLocalPort().GetId();

    u32 crcValue{0};
    HCCL_INFO("[RecoverMemTransport]commptr=%p", comm);

    if (comm->IsWorldGroup()) {
        // 判断是否在框内
        if (comm->GetNeighboorRanks().find(linkData.GetRemoteRankId()) != comm->GetNeighboorRanks().end()) {
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
    auto accelerator = comm->GetOpExecuteConfig().accState;
    HCCL_INFO("[MemTransportManager::CreateOpbasedMemTransport] accelerator[%s]", accelerator.Describe().c_str());
    attr.opAcceState = accelerator;
    attr.handshakeMsg = op.GetUniqueId();

    SocketConfig socketConfig(linkData.GetRemoteRankId(), linkData, comm->GetEstablishLinkSocketTag());
    auto socket = comm->GetSocketManager().GetConnectedSocket(socketConfig);
    if (socket == nullptr) {
        throw std::runtime_error("CreateMemTransport GetConnectedSocket failed, socket is nullptr");
    }
    if (linkData.GetType() == PortDeploymentType::P2P) {
        opTagOffloadMap[opTag][linkData] = make_unique<P2PTransport>(locRes, attr, linkData, *socket);
    } else if (linkData.GetType() == PortDeploymentType::DEV_NET) {
        auto linkProtocol = linkData.GetLinkProtocol();
        if (linkProtocol == LinkProtocol::UB_CTP || linkProtocol == LinkProtocol::UB_TP) {
            CreateOffloadUbMemTransport(opTag, locRes, attr, linkData, *socket);
        } else {
            THROW<NullPtrException>(StringFormat("linkData=%s is error", linkData.Describe().c_str()));
        }
    } else {
        THROW<NullPtrException>(StringFormat("linkData=%s is error", linkData.Describe().c_str()));
    }

    opTagOffloadMap[opTag][linkData]->Establish();

    newOffloadTransports[opTag][linkData] = 0;

    HCCL_INFO("link=%s OK.", linkData.Describe().c_str());
    HCCL_INFO("create transport %s OK.", opTagOffloadMap[opTag][linkData]->Describe().c_str());

    return opTagOffloadMap[opTag][linkData].get();
}

// 功能说明：根据输入的CommID和LinkData信息，恢复单算子Transport对象，并将通信域一致信息改为RecoverInfo
// 输入说明：vector<LinkData> &links：linkData数据
void MemTransportManager::BatchRecoverOpbasedTransports(const vector<LinkData>& links)
{
    HCCL_INFO("BatchRecoverOpbasedTransports start, link num is [%u]", links.size());
    for (auto& link : links) {
        // 校验transport是否已经构建
        if (opTagOpbasedMap.find(link) != opTagOpbasedMap.end()) {
            HCCL_WARNING("linkData=%s already exists, do not need to create transport", link.Describe().c_str());
            continue;
        }
        // 创建transport
        RecoverOpbasedMemTransport(link);
    }
}

// 功能说明：根据输入的CommID和LinkData信息，恢复图模式Transport对象，并将通信域一致信息改为RecoverInfo
// 输入说明：vector<LinkData> &links：linkData数据
//          std::string &opTag：commId，通信域标记
void MemTransportManager::BatchRecoverOffloadTransports(const std::string& opTag, const vector<LinkData>& links)
{
    HCCL_INFO("BatchRecoverOffloadTransports start, link num is [%u]", links.size());
    for (auto& link : links) {
        // 校验transport是否已经构建
        if (opTagOffloadMap.find(opTag) != opTagOffloadMap.end()
            && opTagOffloadMap[opTag].find(link) != opTagOffloadMap[opTag].end()) {
            HCCL_WARNING(
                "opTag=%s, linkData=%s already exists, do not need to create transport", opTag.c_str(),
                link.Describe().c_str());
            continue;
        }
        // 创建transport
        RecoverOffloadMemTransport(opTag, link);
    }
}

// 功能说明：单算子场景，推动式建链，建链成功后，使用RankConsistent校验通信域一致性
bool MemTransportManager::IsAllOpbasedTransportRecoveredReady()
{
    bool isAllTransportRecoveredReady = true;
    // 当前只针对新增的transports做资源交换和op校验
    for (auto linkIt = newOpbasedTransports.begin(); linkIt != newOpbasedTransports.end();) {
        // 尝试建链
        auto status = opTagOpbasedMap[linkIt->first]->GetStatus();
        if (status != TransportStatus::READY) {
            if (status == TransportStatus::SOCKET_TIMEOUT) {
                MACRO_THROW(
                    TimeoutException,
                    StringFormat(
                        "[MemTransportManager][%s] %s socket timeout, commId[%s], please check", __func__,
                        opTagOpbasedMap[linkIt->first]->GetLinkDescInfo().c_str(), comm->GetId().c_str()));
            }
            // 只要任意transport一个没有ready，整体建链结果为 false
            isAllTransportRecoveredReady = false;
            ++linkIt;
        } else {
            HCCL_INFO("linkData[%s], status[%s].", linkIt->first.Describe().c_str(), status.Describe().c_str());
            linkIt = newOpbasedTransports.erase(linkIt);
        }
    }
    return isAllTransportRecoveredReady;
}

// 功能说明：图模式场景，推动式建链，建链成功后，使用RankConsistent校验通信域一致性
// 输入说明：std::string &opTag：commId，通信域标记
bool MemTransportManager::IsAllOffloadTransportRecoveredReady(const std::string& opTag)
{
    bool isAllTransportRecoveredReady = true;
    // 当前只针对新增的transports做资源交换和op校验
    for (auto linkIt = newOffloadTransports[opTag].begin(); linkIt != newOffloadTransports[opTag].end();) {
        // 尝试建链
        auto status = opTagOffloadMap[opTag][linkIt->first]->GetStatus();
        if (status != TransportStatus::READY) {
            if (status == TransportStatus::SOCKET_TIMEOUT) {
                MACRO_THROW(
                    TimeoutException,
                    StringFormat(
                        "[MemTransportManager][%s] %s socket timeout, commId[%s], please check", __func__,
                        opTagOffloadMap[opTag][linkIt->first]->GetLinkDescInfo().c_str(), comm->GetId().c_str()));
            }
            // 只要任意transport一个没有ready，整体建链结果为 false
            isAllTransportRecoveredReady = false;
            ++linkIt;
        } else {
            HCCL_INFO(
                "opTag[%s] linkData[%s] status[%s]", opTag.c_str(), linkIt->first.Describe().c_str(),
                status.Describe().c_str());
            linkIt = newOffloadTransports[opTag].erase(linkIt);
        }
    }
    return isAllTransportRecoveredReady;
}

void MemTransportManager::Clear()
{
    opTagOpbasedMap.clear();
    std::vector<RmaConnection*> emptyVec;
    for (auto& offloadMapIt : opTagOffloadMap) {
        for (auto& memTransportMapIt : offloadMapIt.second) {
            memTransportMapIt.second->SetConnVec(emptyVec);
        }
    }
}

void MemTransportManager::UpdateOffloadTransports()
{
    HCCL_INFO("[UpdateOffloadTransports] start, opTagOffloadMap size is [%u]", opTagOffloadMap.size());
    for (auto& it : opTagOffloadMap) {
        std::string opTag = it.first;
        HCCL_INFO("[UpdateOffloadTransports] start, opTag[%s]", opTag.c_str());
        for (auto& linkTransPair : it.second) {
            auto connectVec = GetConnVec(comm->GetId(), linkTransPair.first);
            linkTransPair.second->SetConnVec(connectVec);
        }
    }
}
BaseMemTransport* MemTransportManager::GetOneSidedTransport(const LinkData& linkData)
{
    if (oneSidedMap.find(linkData) == oneSidedMap.end()) {
        HCCL_WARNING("GetOpbasedTransport, linkData=%s find transport is null", linkData.Describe().c_str());
        return nullptr;
    }
    return oneSidedMap[linkData].get();
}

void MemTransportManager::CreateOneSidedUbMemTransport(
    BaseMemTransport::CommonLocRes& locRes, BaseMemTransport::Attribution& attr, const LinkData& linkData,
    const Socket& socket)
{
    auto topicIdCntNotifyVecMap = comm->GetConnLocalCntNotifyManager().GetTopicIdCntNotifyMap(linkData.GetLocalPort());
    CntNotifyResHelper tool;
    BaseMemTransport::LocCntNotifyRes locCntNotifyRes = tool.GetCntNotifyRes(topicIdCntNotifyVecMap);
    HCCL_INFO("locCntNotifyRes=%s, linkData=%s", locCntNotifyRes.Describe().c_str(), linkData.Describe().c_str());
    RdmaHandle rdmaHandle = RdmaHandleManager::GetInstance().Get(
        comm->GetDevicePhyId(), linkData.GetLocalPort(), linkData.GetLinkProtocol());

    // DFX：注册transportCallBack, 用于信息保存
    auto transportCallBack = MemTransportCallback(linkData, comm->GetMirrorTaskManager());
    auto ubMemTransport
        = make_unique<UbMemTransport>(locRes, attr, linkData, socket, rdmaHandle, locCntNotifyRes, transportCallBack);
    HCCL_INFO("[CreateOneSidedUbMemTransport] Add oneSidedMap");
    oneSidedMap[linkData] = std::move(ubMemTransport);
}

BaseMemTransport* MemTransportManager::CreateOneSidedTransport(const LinkData& linkData)
{
    HCCL_INFO("link=%s Entry CreateMemTransport", linkData.Describe().c_str());
    BaseMemTransport::CommonLocRes locRes;
    locRes.notifyVec = GetNotifyVec(linkData);
    HCCL_INFO("link=%s get notifyVec OK", linkData.Describe().c_str());

    // buffer 来自localRmaBufferManager, input/output/scratch
    locRes.bufferVec = GetBufferVec(comm->GetId(), linkData, OpMode::OFFLOAD);
    HCCL_INFO("link=%s get bufferVec OK", linkData.Describe().c_str());

    // connection是一个，来自 RmaConnManager
    locRes.connVec = GetConnVec(comm->GetId(), linkData);
    HCCL_INFO("link=%s get connVec OK", linkData.Describe().c_str());

    HCCL_INFO("locRes=%s", locRes.Describe().c_str());

    BaseMemTransport::Attribution attr;
    attr.devicePhyId = linkData.GetLocalPort().GetId();

    auto accelerator = comm->GetOpExecuteConfig().accState;
    HCCL_INFO("[MemTransportManager::CreateOneSidedTransport] accelerator[%s]", accelerator.Describe().c_str());
    attr.opAcceState = accelerator;

    SocketConfig socketConfig(linkData.GetRemoteRankId(), linkData, comm->GetEstablishLinkSocketTag());
    auto socket = comm->GetSocketManager().GetConnectedSocket(socketConfig);
    if (socket == nullptr) {
        throw std::runtime_error("CreateMemTransport GetConnectedSocket failed, socket is nullptr");
    }
    if (linkData.GetType() == PortDeploymentType::P2P) {
        oneSidedMap[linkData] = make_unique<P2PTransport>(locRes, attr, linkData, *socket);
    } else if (linkData.GetType() == PortDeploymentType::DEV_NET) {
        if (linkData.GetLinkProtocol() == LinkProtocol::UB_CTP || linkData.GetLinkProtocol() == LinkProtocol::UB_TP) {
            HCCL_INFO("CreateOneSidedUbMemTransport start");
            CreateOneSidedUbMemTransport(locRes, attr, linkData, *socket);
            HCCL_INFO("CreateOneSidedUbMemTransport end");
        } else {
            THROW<NullPtrException>(StringFormat("linkData=%s is error", linkData.Describe().c_str()));
        }
    } else {
        THROW<NullPtrException>(StringFormat("linkData=%s is error", linkData.Describe().c_str()));
    }

    HCCL_INFO("CreateOneSidedTransport Establish");
    oneSidedMap[linkData]->Establish();

    HCCL_INFO("CreateOneSidedTransport equal 0");
    newOneSidedTransports[linkData] = 0;

    HCCL_INFO("link=%s OK.", linkData.Describe().c_str());
    HCCL_INFO("create transport %s OK", oneSidedMap[linkData]->Describe().c_str());

    return oneSidedMap[linkData].get();
}

void MemTransportManager::BatchBuildOneSidedTransports(const vector<LinkData>& links)
{
    HCCL_INFO("Batch build opbased transports start, link num is [%u]", links.size());
    for (auto& link : links) {
        if (opTagOpbasedMap.find(link) != opTagOpbasedMap.end()) {
            HCCL_WARNING("linkData=%s already exists, do not need to create transport", link.Describe().c_str());
            continue;
        }
        CreateOneSidedTransport(link);
    }
}

void MemTransportManager::CreateUrmaDirectTransport(
    BaseMemTransport::CommonLocRes& locRes, BaseMemTransport::Attribution& attr, const LinkData& linkData,
    const Socket& socket)
{
    RdmaHandle rdmaHandle = RdmaHandleManager::GetInstance().Get(
        comm->GetDevicePhyId(), linkData.GetLocalPort(), linkData.GetLinkProtocol());

    // DFX：注册transportCallBack, 用于信息保存
    auto transportCallBack = MemTransportCallback(linkData, comm->GetMirrorTaskManager());
    auto transport = make_unique<UrmaDirectTransport>(locRes, attr, linkData, socket, rdmaHandle, transportCallBack);
    HCCL_INFO("[CreateUrmaDirectTransport] Add urmaDirectMap_");
    urmaDirectMap_[linkData] = std::move(transport);
}

BaseMemTransport* MemTransportManager::CreateUrmaDirectTransport(const LinkData& linkData)
{
    auto op = comm->GetCurrentCollOperator();
    HCCL_INFO("link=%s Entry CreateMemTransport", linkData.Describe().c_str());
    HCCL_INFO("Entry CreateMemTransport, opInfo=[%s]", CollOpToString(*op).c_str());
    BaseMemTransport::CommonLocRes locRes;

    // buffer 来自localRmaBufferManager, input/output/scratch
    locRes.bufferVec = GetBufferVec(GetAivUrmaBufferTag(comm->GetId()), linkData, OpMode::OPBASE);
    HCCL_INFO("link=%s get bufferVec OK", linkData.Describe().c_str());

    // connection是一个，来自 RmaConnManager
    locRes.connVec = GetConnVec(comm->GetId(), linkData);
    HCCL_INFO("link=%s get connVec OK", linkData.Describe().c_str());

    HCCL_INFO("locRes=%s", locRes.Describe().c_str());

    BaseMemTransport::Attribution attr;
    attr.devicePhyId = linkData.GetLocalPort().GetId();
    // 握手消息定义，未来包括 cann版本号，rankTable CRC等字段
    attr.handshakeMsg = op->GetUniqueId();

    SocketConfig socketConfig(linkData.GetRemoteRankId(), linkData, comm->GetEstablishLinkSocketTag());
    auto socket = comm->GetSocketManager().GetConnectedSocket(socketConfig);
    if (socket == nullptr) {
        throw std::runtime_error("CreateMemTransport GetConnectedSocket failed, socket is nullptr");
    }

    CreateUrmaDirectTransport(locRes, attr, linkData, *socket);

    urmaDirectMap_[linkData]->Establish();

    HCCL_INFO("link=%s OK.", linkData.Describe().c_str());
    HCCL_INFO("create transport %s OK.", urmaDirectMap_[linkData]->Describe().c_str());

    return urmaDirectMap_[linkData].get();
}

void MemTransportManager::BatchBuildUrmaDirectTransports(const vector<LinkData>& links)
{
    HCCL_INFO("Batch build urma direct transports start, link num is [%u]", links.size());
    for (auto& link : links) {
        if (urmaDirectMap_.find(link) != urmaDirectMap_.end()) {
            HCCL_WARNING("linkData=%s already exists, do not need to create transport", link.Describe().c_str());
            continue;
        }
        CreateUrmaDirectTransport(link);
    }
}

HcclResult MemTransportManager::ClearOpTransport(const std::string& opTag)
{
    if (opTagOffloadMap.find(opTag) == opTagOffloadMap.end()) {
        HCCL_WARNING(
            "[LocalRmaBufManager::%s] opTag[%s] Cannot find Transport in opTagOffloadMap.", __func__, opTag.c_str());
    }
    if (newOffloadTransports.find(opTag) == newOffloadTransports.end()) {
        HCCL_WARNING(
            "[LocalRmaBufManager::%s] opTag[%s] Cannot find Transport in newOffloadTransports.", __func__,
            opTag.c_str());
    }
    opTagOffloadMap.erase(opTag);
    newOffloadTransports.erase(opTag);
    return HCCL_SUCCESS;
}

} // namespace Hccl
