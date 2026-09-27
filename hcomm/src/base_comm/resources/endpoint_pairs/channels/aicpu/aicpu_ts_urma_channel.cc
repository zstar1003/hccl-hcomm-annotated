/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "cast_utils.h"
#include "aicpu_ts_urma_channel.h"
#include "endpoint.h"
#include "../../sockets/socket_mgr.h"
#include "orion_adpt_utils.h"
#include "hcomm_c_adpt.h"
#include "config_log.h"
#include "config_plf_log_v2.h"
#include "hcomm_res_mgr.h"
#include "endpoint.h"

// Orion
#include "adapter_rts_common.h"
#include "topo_common_types.h"
#include "virtual_topo.h"
#include "aicpu_res_package_helper.h"
#include "makebufs_helper.h"

namespace hcomm {
using Hccl::PLF_CHANNEL;
constexpr uint16_t DEFAULT_LISTENING_PORT = 60001;

AicpuTsUrmaChannel::AicpuTsUrmaChannel(EndpointHandle endpointHandle, const HcommChannelDesc& channelDesc)
    : endpointHandle_(endpointHandle),
      channelDesc_(channelDesc)
{}

AicpuTsUrmaChannel::~AicpuTsUrmaChannel()
{
    if (channelDesc_.socket == nullptr && socket_ != nullptr) {
        SocketMgr::GetInstance(devicePhyId_).PutSocket(socketConfig_, socket_);
        socket_ = nullptr;
    }
}

HcclResult AicpuTsUrmaChannel::ParseInputParam()
{
    // 1. 从 endpointHandle_，获得 localEp_ 和 rdmaHandle_
    // TODO: 使用 HcommEndpointGet
    Endpoint* localEpPtr = static_cast<Endpoint*>(endpointHandle_);
    CHK_PTR_NULL(localEpPtr);
    localEp_ = localEpPtr->GetEndpointDesc();
    rdmaHandle_ = localEpPtr->GetRdmaHandle();

    HCCL_INFO("[AicpuTsUrmaChannel][%s] localProtocol[%d]", __func__, localEp_.protocol);

    // 2. 从 channelDesc_，获得 remoteEp_, socket_ 和 notifyNum
    remoteEp_ = channelDesc_.remoteEndpoint;
    socket_ = static_cast<Hccl::Socket*>(channelDesc_.socket);
    notifyNum_ = channelDesc_.notifyNum;
    commonRes_.bufferVec.clear();

    if (channelDesc_.exchangeAllMems) {
        // 3. Get memHandles from endpoint
        HCCL_INFO("[AicpuTsUrmaChannel][%s] exchangeAllMems == True. Get memHandles from endpoint.", __func__);
        std::shared_ptr<Hccl::LocalUbRmaBuffer>* memHandles = nullptr;
        uint32_t memHandleNum = 0;
        CHK_RET(static_cast<HcclResult>(
            HcommMemGetAllMemHandles(endpointHandle_, ReinterpretAs<void**>(&memHandles), &memHandleNum)));
        HCCL_INFO("[AicpuTsUrmaChannel][%s] Got memHandleNum[%u].", __func__, memHandleNum);
        for (uint32_t i = 0; i < memHandleNum; ++i) {
            std::shared_ptr<Hccl::LocalUbRmaBuffer>& localUbRmaBuffer = memHandles[i];
            CHK_SMART_PTR_NULL(localUbRmaBuffer);
            Hccl::Buffer* buf = localUbRmaBuffer->GetBuf();
            CHK_PTR_NULL(buf);
            HCCL_INFO(
                "[AicpuTsUrmaChannel][%s] Got memHandle No.%u: addr[0x%llx], size[0x%llx], type[%d], memInfo[%s].",
                __func__, i, static_cast<unsigned long long>(localUbRmaBuffer->GetAddr()),
                static_cast<unsigned long long>(localUbRmaBuffer->GetSize()), static_cast<int>(buf->GetMemType()),
                buf->GetMemInfo().c_str());
            commonRes_.bufferVec.push_back(localUbRmaBuffer.get());
        }
    } else {
        // 3. 从 channelDesc 的 memHandle，获得 bufs_
        HCCL_INFO("[AicpuTsUrmaChannel][%s] exchangeAllMems == false. Get memHandles from channelDesc.", __func__);
        CHK_RET(MakeRmaBufferVecFromMemHandles(
            channelDesc_.memHandles, channelDesc_.memHandleNum, commonRes_.bufferVec, "AicpuTsUrmaChannel"));
    }

    return HCCL_SUCCESS;
}

HcclResult AicpuTsUrmaChannel::BuildAttr()
{
    attr_.devicePhyId = localEp_.loc.device.devPhyId;
    attr_.opMode = Hccl::OpMode::OPBASE;
    return HCCL_SUCCESS;
}

HcclResult AicpuTsUrmaChannel::BuildConnection()
{
    UbConnBuildContext ctx;
    CHK_RET(PrepareUbConnBuildContext(localEp_, remoteEp_, channelDesc_, ctx));

    Hccl::OpMode opMode = Hccl::OpMode::OPBASE;
    bool devUsed = true; // aicpu 为 true
    // UB_CTP → HCOMM_TA_CTP_UB_TIMEOUT，UB_TP → HCOMM_TA_RTP_UB_TIMEOUT
    u8 taTimeOut = 0;
    uint32_t taTimeOutValue = 0;
    if (ctx.protocol == Hccl::LinkProtocol::UB_CTP) {
        CHK_RET(hcomm::HcommResMgr::GetInstance().GetConfigMgr().GetRdmaConfig().GetTaCtpUbTimeOut(taTimeOutValue));
    } else {
        CHK_RET(hcomm::HcommResMgr::GetInstance().GetConfigMgr().GetRdmaConfig().GetTaRtpUbTimeOut(taTimeOutValue));
    }
    taTimeOut = static_cast<u8>(taTimeOutValue);
    std::unique_ptr<Hccl::DevUbConnection> ubConn = nullptr;
    switch (ctx.protocol) {
        case Hccl::LinkProtocol::UB_TP:
            EXCEPTION_CATCH(
                ubConn = std::make_unique<Hccl::DevUbTpConnection>(
                    rdmaHandle_, ctx.locAddr, ctx.rmtAddr, opMode, devUsed, Hccl::HrtUbJfcMode::STARS_POLL,
                    Hccl::IpAddress(), Hccl::IpAddress(), ctx.qosPre, taTimeOut, COMM_ENGINE_AICPU_TS, ctx.sqDepth),
                return HCCL_E_PTR);
            break;
        case Hccl::LinkProtocol::UB_CTP:
            EXCEPTION_CATCH(
                ubConn = std::make_unique<Hccl::DevUbCtpConnection>(
                    rdmaHandle_, ctx.locAddr, ctx.rmtAddr, opMode, devUsed, Hccl::HrtUbJfcMode::STARS_POLL,
                    Hccl::IpAddress(), Hccl::IpAddress(), ctx.qosPre, taTimeOut, COMM_ENGINE_AICPU_TS, ctx.sqDepth),
                return HCCL_E_PTR);
            break;
        default:
            HCCL_ERROR("%s No LinkProtocol to match", __func__);
            break;
    }
    CHK_SMART_PTR_NULL(ubConn);

    if (devBaseAttr_.maxReadSize == 0 || devBaseAttr_.maxWriteSize == 0) {
        HCCL_ERROR(
            "%s maxReadSize[%u] or maxWriteSize[%u] must not be zero", __func__, devBaseAttr_.maxReadSize,
            devBaseAttr_.maxWriteSize);
        return HCCL_E_PARA;
    }
    ubConn->SetMaxReadSize(devBaseAttr_.maxReadSize);
    ubConn->SetMaxWriteSize(devBaseAttr_.maxWriteSize);
    HCCL_INFO("%s maxReadSize[%u], maxWriteSize[%u]", __func__, devBaseAttr_.maxReadSize, devBaseAttr_.maxWriteSize);

    commonRes_.connVec.clear();
    commonRes_.connVec.emplace_back(ubConn.get());
    connections_.clear();
    connections_.push_back(std::move(ubConn));
    PLF_CONFIG_INFO(
        PLF_CHANNEL, "[AicpuTsUrmaChannel] build DevUbConnection, protocol[%s] sqDepth[%u].",
        ctx.protocol.Describe().c_str(), ctx.sqDepth);
    return HCCL_SUCCESS;
}

HcclResult AicpuTsUrmaChannel::BuildNotify()
{
    localNotifies_.clear();
    commonRes_.notifyVec.clear();
    bool devUsed = true;
    for (uint32_t i = 0; i < notifyNum_; ++i) {
        std::unique_ptr<Hccl::UbLocalNotify> notifyPtr = nullptr;
        EXCEPTION_CATCH(notifyPtr = std::make_unique<Hccl::UbLocalNotify>(rdmaHandle_, devUsed), return HCCL_E_PTR);
        commonRes_.notifyVec.push_back(notifyPtr.get());
        localNotifies_.push_back(std::move(notifyPtr));
    }
    PLF_CONFIG_INFO(PLF_CHANNEL, "[AicpuTsUrmaChannel] create notify, notifyNum[%u].", notifyNum_);
    return HCCL_SUCCESS;
}

HcclResult AicpuTsUrmaChannel::BuildUbMemTransport()
{
    Hccl::BaseMemTransport::LocCntNotifyRes locCntNotifyRes{};
    locCntNotifyRes.vec.clear();
    locCntNotifyRes.desc.clear();
    const Hccl::Socket& socket = *socket_;

    Hccl::LinkData linkData = BuildDefaultLinkData();
    CHK_RET(EndpointDescPairToLinkData(localEp_, remoteEp_, linkData));

    bool isRecvFirst = socket.GetRole() == Hccl::SocketRole::CLIENT ? true : false;

    // make_unique / make_shared / release 包一层抛异常的宏
    EXCEPTION_CATCH(
        memTransport_ = std::make_unique<Hccl::UbMemTransport>(
            commonRes_, attr_, linkData, socket, rdmaHandle_, locCntNotifyRes, isRecvFirst),
        return HCCL_E_PTR);
    PLF_CONFIG_INFO(
        PLF_CHANNEL, "[AicpuTsUrmaChannel] create UbMemTransport, socket[%s], linkData[%s].",
        socket_->Describe().c_str(), linkData.Describe().c_str());
    return HCCL_SUCCESS;
}

HcclResult AicpuTsUrmaChannel::BuildSocket()
{
    if (socket_ != nullptr) {
        return HCCL_SUCCESS;
    }
    HCCL_INFO("[AicpuTsUrmaChannel][%s] socket ptr is NULL, rebuildSocket", __func__);
    Hccl::IpAddress ipaddr{};
    CHK_RET(CommAddrToIpAddress(localEp_.commAddr, ipaddr));
    Hccl::DevNetPortType type = Hccl::DevNetPortType(Hccl::ConnectProtoType::UB);
    Hccl::PortData localPort = Hccl::PortData(static_cast<Hccl::RankId>(localEp_.loc.device.devPhyId), type, 0, ipaddr);
    if (channelDesc_.role == HCOMM_SOCKET_ROLE_RESERVED) {
        Hccl::SocketHandle socketHandle
            = Hccl::SocketHandleManager::GetInstance().Create(localEp_.loc.device.devPhyId, localPort);
        EXCEPTION_CATCH(
            serverSocket_ = std::make_unique<Hccl::Socket>(
                socketHandle, ipaddr, DEFAULT_LISTENING_PORT, ipaddr, "server", Hccl::SocketRole::SERVER,
                Hccl::NicType::DEVICE_NIC_TYPE),
            return HCCL_E_PARA);
        HCCL_INFO("[AicpuTsUrmaChannel][%s] listen_socket_info[%s]", __func__, serverSocket_->Describe().c_str());
        EXCEPTION_CATCH(serverSocket_->Listen(), return HCCL_E_INTERNAL);
        Hccl::LinkData linkData = BuildDefaultLinkData();
        CHK_RET(EndpointDescPairToLinkData(localEp_, remoteEp_, linkData));
        HCCL_INFO("[AicpuTsUrmaChannel][%s] built linkData: %s", __func__, linkData.Describe().c_str());
        std::string socketTag
            = (channelDesc_.channelName != nullptr) ? std::string(channelDesc_.channelName) : "AUTOMATIC_SOCKET_TAG";
        bool noRankId = true;
        Hccl::SocketConfig socketConfig = Hccl::SocketConfig(linkData, socketTag, noRankId);
        CHK_RET(SocketMgr::GetInstance(devicePhyId_).GetSocket(socketConfig, socket_));
        socketConfigHolder_ = std::make_unique<Hccl::SocketConfig>(socketConfig);
        socketConfig_ = socketConfigHolder_.get();
    } else {
        uint16_t port = channelDesc_.port;
        if (port == 0) {
            port = DEFAULT_LISTENING_PORT;
            HCCL_INFO("[AicpuTsUrmaChannel::%s] channelDesc port is 0, use default port [%u]", __func__, port);
        }
        Hccl::LinkData linkData = BuildDefaultLinkData();
        CHK_RET(EndpointDescPairToLinkData(localEp_, remoteEp_, linkData));
        HCCL_INFO("[AicpuTsUrmaChannel][%s] built linkData: %s", __func__, linkData.Describe().c_str());
        std::string socketTag
            = (channelDesc_.channelName != nullptr) ? std::string(channelDesc_.channelName) : "AUTOMATIC_SOCKET_TAG";
        bool isServer = (channelDesc_.role == HCOMM_SOCKET_ROLE_SERVER);
        Hccl::SocketConfig socketConfig = Hccl::SocketConfig(linkData, port, socketTag, isServer);
        CHK_RET(SocketMgr::GetInstance(devicePhyId_).GetSocket(socketConfig, socket_));
        socketConfigHolder_ = std::make_unique<Hccl::SocketConfig>(socketConfig);
        socketConfig_ = socketConfigHolder_.get();
    }
    return HCCL_SUCCESS;
}

HcclResult AicpuTsUrmaChannel::Init()
{
    /*
        Argue result: make_unique 配合一场捕获的宏 EXCEPTION CATCH
        Attention: const 和引用
    */
    // TODO: 处理抛异常
    s32 devLogicId;
    CHK_RET(ParseInputParam());
    CHK_RET(hrtGetDevice(&devLogicId));
    CHK_RET(hrtGetDevicePhyIdByIndex(static_cast<u32>(devLogicId), devicePhyId_));
    CHK_RET(StartListen());
    CHK_RET(BuildSocket());
    CHK_RET(BuildAttr());
    /*
        HccpRaGetDevBaseAttr
        获取urma read/write 单个wr的最大传输数据大小
        调用前,rdmaHandle_要在ParseInputParam中被赋值好,之后BuildConnection会使用获取的属性
    */
    CHK_RET(HccpRaGetDevBaseAttr(rdmaHandle_, &devBaseAttr_));
    CHK_RET(BuildConnection());
    CHK_RET(BuildNotify());
    CHK_RET(BuildUbMemTransport());

    return HCCL_SUCCESS;
}

HcclResult AicpuTsUrmaChannel::GetNotifyNum(uint32_t* notifyNum) const
{
    *notifyNum = this->notifyNum_;
    return HCCL_SUCCESS;
}

HcclResult AicpuTsUrmaChannel::GetRemoteMems(uint32_t* memNum, CommMem** remoteMem, char*** memInfos)
{
    return memTransport_->GetRemoteMems(memNum, remoteMem, memInfos);
}

ChannelStatus AicpuTsUrmaChannel::GetStatus()
{
    ChannelStatus out = Channel::TransportStatusToChannelStatus(memTransport_->GetStatus(), localEp_, GetChannelDesc());

    if (isFirstPrintChannelInfo_ && out == ChannelStatus::READY) {
        std::string channelInfo = "create channel info:channel handle[";
        channelInfo.append(std::to_string(ReinterpretAs<uint64_t>(this)));
        channelInfo.append("] ");
        HcclResult ret = memTransport_->Describe(channelInfo);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR("[AicpuTsUrmaChannel][%s] Describe channel info failed, ret=%d", __func__, ret);
            out = ChannelStatus::FAILED;
        } else {
            channelInfo.append(" TA[RM]"); // 目前TA只支持RM
            HCCL_CONFIG_DEBUG(hccl::HCCL_RES, "%s", channelInfo.c_str());
        }
        isFirstPrintChannelInfo_ = false;
    }
    return out;
}

HcclResult SetModuleDataName(Hccl::ModuleData& module, const std::string& name)
{
    int ret = strcpy_s(module.name, sizeof(module.name), name.c_str());
    if (ret != 0) {
        HCCL_ERROR("[SetModuleDataName] strcpy_s name %s failed", name.c_str());
        return HCCL_E_INTERNAL;
    }

    return HCCL_SUCCESS;
}

HcclResult AicpuTsUrmaChannel::PackOpData(std::vector<char>& data)
{
    std::vector<Hccl::ModuleData> dataVec;
    dataVec.resize(Hccl::AicpuResMgrType::__COUNT__);

    Hccl::AicpuResMgrType resType = Hccl::AicpuResMgrType::STREAM;
    CHK_RET(SetModuleDataName(dataVec[resType], "UbMemTransport"));

    std::vector<char> result;
    Hccl::BinaryStream binaryStream;
    binaryStream << memTransport_->GetUniqueIdV2();

    binaryStream.Dump(result);

    dataVec[resType].data = result;

    Hccl::AicpuResPackageHelper helper;
    data = helper.GetPackedData(dataVec);

    return HCCL_SUCCESS;
}

HcclResult AicpuTsUrmaChannel::H2DResPack(std::vector<char>& buffer)
{
    CHK_RET(PackOpData(buffer));
    HCCL_INFO(
        "[AicpuTsUrmaChannel][%s] Pack Buffer data[%p], Pack Buffer size[%zu].", __func__, buffer.data(),
        buffer.size());
    return HCCL_SUCCESS;
}

HcclResult AicpuTsUrmaChannel::Clean()
{
    PLF_CONFIG_INFO(PLF_CHANNEL, "[AicpuTsUrmaChannel] clean channel resource.");
    memTransport_.reset();
    return HCCL_SUCCESS;
}

HcclResult AicpuTsUrmaChannel::ResetLocalNotifies()
{
    for (size_t i = 0; i < localNotifies_.size(); ++i) {
        if (localNotifies_[i] == nullptr) {
            continue;
        }
        Hccl::RtsNotify* rtsNotify = localNotifies_[i]->GetNotify();
        if (rtsNotify == nullptr) {
            continue;
        }
        HcclRtNotify rtNotify = ReinterpretAs<HcclRtNotify>(rtsNotify->GetHandleAddr());
        if (rtNotify == nullptr) {
            continue;
        }
        HcclResult ret = hrtNotifyReset(rtNotify);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR(
                "[AicpuTsUrmaChannel][ResetLocalNotifies] hrtNotifyReset failed, channel[%p], notifyIdx[%zu], "
                "notifyId[%u], ret[0x%016llx]",
                this, i, rtsNotify->GetId(), HCCL_ERROR_CODE(ret)),
            ret);
        HCCL_INFO(
            "[AicpuTsUrmaChannel][ResetLocalNotifies] reset notify success, channel[%p], notifyIdx[%zu], "
            "notifyId[%u], notifyPtr[%p]",
            this, i, rtsNotify->GetId(), rtNotify);
    }
    return HCCL_SUCCESS;
}

HcclResult AicpuTsUrmaChannel::Resume()
{
    CHK_RET(BuildSocket());
    CHK_RET(BuildConnection());
    CHK_RET(ResetLocalNotifies());
    CHK_RET(BuildUbMemTransport());
    return HCCL_SUCCESS;
}

HcclResult AicpuTsUrmaChannel::UpdateMemInfo(HcommMemHandle* memHandles, uint32_t memHandleNum)
{
    std::vector<Hccl::LocalRmaBuffer*> bufferVecTemp;
    CHK_RET(MakeRmaBufferVecFromMemHandles(memHandles, memHandleNum, bufferVecTemp, "AicpuTsUrmaChannel"));
    CHK_RET(memTransport_->UpdateMemInfo(bufferVecTemp));
    commonRes_.bufferVec.insert(commonRes_.bufferVec.end(), bufferVecTemp.begin(), bufferVecTemp.end());
    return HCCL_SUCCESS;
}

// 返回当前 channel 类型，供上层区分不同 channel 的能力和行为
HcommChannelKind AicpuTsUrmaChannel::GetChannelKind() const { return HcommChannelKind::AICPU_TS_URMA; }

HcclResult AicpuTsUrmaChannel::NotifyRecord([[maybe_unused]] const uint32_t remoteNotifyIdx)
{
    HCCL_INFO("[AicpuTsUrmaChannel::%s] not supported yet.", __func__);
    return HCCL_E_NOT_SUPPORT;
}

HcclResult
AicpuTsUrmaChannel::NotifyWait([[maybe_unused]] const uint32_t localNotifyIdx, [[maybe_unused]] const uint32_t timeout)
{
    HCCL_INFO("[AicpuTsUrmaChannel::%s] not supported yet.", __func__);
    return HCCL_E_NOT_SUPPORT;
}

HcclResult AicpuTsUrmaChannel::WriteWithNotify(
    [[maybe_unused]] void* dst, [[maybe_unused]] const void* src, [[maybe_unused]] const uint64_t len,
    [[maybe_unused]] uint32_t remoteNotifyIdx)
{
    HCCL_INFO("[AicpuTsUrmaChannel::%s] not supported yet.", __func__);
    return HCCL_E_NOT_SUPPORT;
}

HcclResult
AicpuTsUrmaChannel::Write([[maybe_unused]] void* dst, [[maybe_unused]] const void* src, [[maybe_unused]] uint64_t len)
{
    HCCL_INFO("[AicpuTsUrmaChannel::%s] not supported yet.", __func__);
    return HCCL_E_NOT_SUPPORT;
}

HcclResult
AicpuTsUrmaChannel::Read([[maybe_unused]] void* dst, [[maybe_unused]] const void* src, [[maybe_unused]] uint64_t len)
{
    HCCL_INFO("[AicpuTsUrmaChannel::%s] not supported yet.", __func__);
    return HCCL_E_NOT_SUPPORT;
}

HcclResult AicpuTsUrmaChannel::ChannelFence()
{
    HCCL_INFO("[AicpuTsUrmaChannel::%s] not supported yet.", __func__);
    return HCCL_E_NOT_SUPPORT;
}

HcclResult AicpuTsUrmaChannel::StartListen()
{
    if (channelDesc_.role != HCOMM_SOCKET_ROLE_SERVER) {
        return HCCL_SUCCESS;
    }

    uint16_t port = channelDesc_.port;
    HCCL_INFO(
        "[AicpuTsUrmaChannel::%s] Start. EndpointHandle[0x%llx], port[%u]", __func__,
        ReinterpretAs<uint64_t>(endpointHandle_), port);
    if (port == 0) {
        port = DEFAULT_LISTENING_PORT;
        HCCL_INFO("[AicpuTsUrmaChannel::%s] channelDesc port is 0, use default port [%u]", __func__, port);
    }
    CHK_RET(static_cast<HcclResult>(HcommEndpointStartListen(endpointHandle_, port, nullptr)));
    HCCL_INFO("[AicpuTsUrmaChannel::%s] SUCCESS. port[%u].", __func__, port);
    return HCCL_SUCCESS;
}

} // namespace hcomm
