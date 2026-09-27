/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aicpu_ts_roce_channel.h"

#include <algorithm>
#include <chrono>
#include <arpa/inet.h>
#include <climits>
#include <cstdio>
#include <cstdint>
#include <memory>
#include <string>
#include <securec.h>
#include "log.h"
#include "endpoint.h"
#include "../../../endpoints/aicpu_ts_roce_endpoint.h"
#include "server_socket_context/aicpu_ts_roce_server_socket_context.h"
#include "aicpu_ts_roce_reged_mem_mgr.h"
#include "adapter_rts_common.h"
#include "channel_param.h"
#include "dispatcher_ctx.h"
#include "adapter_hccp_common.h"
#include "hccl_dispatcher_ctx.h"
#include "hccl_network.h"
#include "mem_device_pub.h"
#include "sal_pub.h"
#include "env_config.h"

namespace hcomm {

namespace {
    constexpr uint32_t kDefaultRocePort = 16666;
    constexpr uint8_t kHcommTrafficClassConfigNotSet = 0xff;
    constexpr uint8_t kHcommServiceLevelConfigNotSet = 0xff;
    constexpr uint32_t kAicpuTsRoceSqCqDepth = 2048U; // SQ/发送CQ默认深度

    HcclResult CommAddrToHcclIp(const CommAddr& ca, hccl::HcclIpAddress& out)
    {
        if (ca.type == COMM_ADDR_TYPE_IP_V4) {
            out = hccl::HcclIpAddress(ca.addr);
            return HCCL_SUCCESS;
        }
        if (ca.type == COMM_ADDR_TYPE_IP_V6) {
            out = hccl::HcclIpAddress(ca.addr6);
            return HCCL_SUCCESS;
        }
        HCCL_ERROR("[AicpuTsRoceChannel] unsupported CommAddr type[%d]", ca.type);
        return HCCL_E_NOT_SUPPORT;
    }

    HcclResult
    DecideLocalIsClientByEndpointIps(const EndpointDesc& local, const EndpointDesc& remote, bool& outLocalIsClient)
    {
        hccl::HcclIpAddress localIp{};
        hccl::HcclIpAddress remoteIp{};
        CHK_RET(CommAddrToHcclIp(local.commAddr, localIp));
        CHK_RET(CommAddrToHcclIp(remote.commAddr, remoteIp));
        const std::string localStr(localIp.GetReadableIP());
        const std::string remoteStr(remoteIp.GetReadableIP());
        if (localStr < remoteStr) {
            outLocalIsClient = true;
        } else if (localStr > remoteStr) {
            outLocalIsClient = false;
        } else {
            HCCL_ERROR("[AicpuTsRoceChannel] same readable IP but loc not DEVICE; cannot decide socket role");
            return HCCL_E_PARA;
        }
        return HCCL_SUCCESS;
    }
} // namespace

HcclResult AicpuTsRoceChannel::BuildSocketTagName(std::string& outTag) const
{
    if (channelDesc_.channelName != nullptr) {
        outTag = std::string(channelDesc_.channelName);
        if (outTag.size() + 1U > SOCK_CONN_TAG_SIZE) {
            HCCL_ERROR(
                "[AicpuTsRoceChannel] channelName too long (max %u bytes)",
                static_cast<unsigned int>(SOCK_CONN_TAG_SIZE - 1U));
            return HCCL_E_PARA;
        }
        return HCCL_SUCCESS;
    }

    hccl::HcclIpAddress localIp{};
    hccl::HcclIpAddress remoteIp{};
    CHK_RET(CommAddrToHcclIp(localEp_.commAddr, localIp));
    CHK_RET(CommAddrToHcclIp(remoteEp_.commAddr, remoteIp));
    const std::string clientStr(isLocalIpClient_ ? localIp.GetReadableIP() : remoteIp.GetReadableIP());
    const std::string serverStr(isLocalIpClient_ ? remoteIp.GetReadableIP() : localIp.GetReadableIP());
    const uint32_t port = channelDesc_.port != 0 ? channelDesc_.port : kDefaultRocePort;
    outTag = clientStr + "_" + serverStr + ":" + std::to_string(port);
    if (outTag.size() + 1U > SOCK_CONN_TAG_SIZE) {
        HCCL_ERROR(
            "[AicpuTsRoceChannel] socketTag too long (max %u bytes)",
            static_cast<unsigned int>(SOCK_CONN_TAG_SIZE - 1U));
        return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}

AicpuTsRoceChannel::AicpuTsRoceChannel(EndpointHandle endpointHandle, const HcommChannelDesc& channelDesc)
    : endpointHandle_(endpointHandle),
      channelDesc_(channelDesc)
{}

AicpuTsRoceChannel::~AicpuTsRoceChannel()
{
    transport_.reset();
    if (ownsDispatcherCtx_ && dispatcherCtx_ != nullptr) {
        HcclResult ret = DestroyDispatcherCtx(dispatcherCtx_, dispatcherCommId_.c_str());
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR("[AicpuTsRoceChannel][%s] DestroyDispatcherCtx failed, ret[%d]", SocketRoleTag(), ret);
        }
        dispatcherCtx_ = nullptr;
        ownsDispatcherCtx_ = false;
    }
    dataSocket_.reset();
    HCCL_INFO("[AicpuTsRoceChannel][%s] destroyed", SocketRoleTag());
}

HcclResult AicpuTsRoceChannel::ParseInputParam()
{
    auto* localEpPtr = static_cast<Endpoint*>(endpointHandle_);
    CHK_PTR_NULL(localEpPtr);
    localEp_ = localEpPtr->GetEndpointDesc();
    rdmaHandle_ = localEpPtr->GetRdmaHandle();
    CHK_PTR_NULL(rdmaHandle_);

    remoteEp_ = channelDesc_.remoteEndpoint;
    if (channelDesc_.role == HCOMM_SOCKET_ROLE_CLIENT) {
        isLocalIpClient_ = true;
    } else if (channelDesc_.role == HCOMM_SOCKET_ROLE_SERVER) {
        isLocalIpClient_ = false;
    } else {
        if (channelDesc_.role != HCOMM_SOCKET_ROLE_RESERVED) {
            HCCL_WARNING(
                "[AicpuTsRoceChannel] unexpected channelDesc.role[%d]; "
                "using inner logic to decide socket role based on endpoint IPs",
                static_cast<int>(channelDesc_.role));
        }
        CHK_RET(DecideLocalIsClientByEndpointIps(localEp_, remoteEp_, isLocalIpClient_));
    }
    HCCL_INFO("[AicpuTsRoceChannel][%s] ParseInputParam start", SocketRoleTag());

    notifyNum_ = channelDesc_.notifyNum;
    if (notifyNum_ != 0) {
        HCCL_WARNING(
            "[AicpuTsRoceChannel][%s] channelDesc.notifyNum[%u] ignored; transport uses notifyNum=0 for now.",
            SocketRoleTag(), notifyNum_);
    }
    HCCL_INFO("[AicpuTsRoceChannel][%s] ParseInputParam done", SocketRoleTag());
    return HCCL_SUCCESS;
}

HcclResult AicpuTsRoceChannel::BuildDataSocket()
{
    HCCL_INFO("[AicpuTsRoceChannel][%s] BuildDataSocket start", SocketRoleTag());
    auto* roceEp = dynamic_cast<AicpuTsRoceEndpoint*>(static_cast<Endpoint*>(endpointHandle_));
    CHK_PTR_NULL(roceEp);

    HcclNetDevCtx netDevCtx = static_cast<HcclNetDevCtx>(roceEp->GetNetDev());
    CHK_PTR_NULL(netDevCtx);

    auto* netDevCtxPtr = static_cast<hccl::NetDevContext*>(netDevCtx);
    machinePara_.localIpAddr = netDevCtxPtr->GetLocalIp();

    hccl::HcclIpAddress remoteIp{};
    CHK_RET(CommAddrToHcclIp(remoteEp_.commAddr, remoteIp));

    uint32_t port = channelDesc_.port != 0 ? channelDesc_.port : kDefaultRocePort;
    std::string socketTag;
    CHK_RET(BuildSocketTagName(socketTag));

    HCCL_INFO(
        "[AicpuTsRoceChannel][%s] BuildDataSocket localIp[%s] remoteIp[%s] port[%u] socketTag[%s]", SocketRoleTag(),
        machinePara_.localIpAddr.GetReadableIP(), remoteIp.GetReadableIP(), port, socketTag.c_str());

    if (isLocalIpClient_) {
        CHK_RET(BuildClientDataSocket(netDevCtx, remoteIp, port, socketTag));
    } else {
        CHK_RET(BuildServerDataSocket(roceEp, remoteIp, port, socketTag));
    }

    machinePara_.remoteIpAddr = remoteIp;
    machinePara_.localSocketPort = dataSocket_->GetLocalPort();
    machinePara_.remoteSocketPort = dataSocket_->GetRemotePort();
    HCCL_INFO(
        "[AicpuTsRoceChannel][%s] BuildDataSocket done localPort[%u] remotePort[%u]", SocketRoleTag(),
        machinePara_.localSocketPort, machinePara_.remoteSocketPort);
    return HCCL_SUCCESS;
}

HcclResult AicpuTsRoceChannel::BuildClientDataSocket(
    HcclNetDevCtx netDevCtx, const hccl::HcclIpAddress& remoteIp, uint32_t port, const std::string& socketTag)
{
    HCCL_INFO("[AicpuTsRoceChannel][client] BuildClientDataSocket connect to server");
    EXCEPTION_CATCH(
        dataSocket_ = std::make_shared<hccl::HcclSocket>(
            socketTag, netDevCtx, remoteIp, port, hccl::HcclSocketRole::SOCKET_ROLE_CLIENT),
        return HCCL_E_PTR);
    CHK_SMART_PTR_NULL(dataSocket_);
    CHK_RET(dataSocket_->Init());
    CHK_RET(dataSocket_->Connect());
    HCCL_INFO("[AicpuTsRoceChannel][client] BuildClientDataSocket TCP link ready");
    return HCCL_SUCCESS;
}

HcclResult AicpuTsRoceChannel::BuildServerDataSocket(
    AicpuTsRoceEndpoint* roceEp, const hccl::HcclIpAddress& remoteIp, uint32_t port, const std::string& socketTag)
{
    HCCL_INFO("[AicpuTsRoceChannel][server] BuildDataSocket listen and accept");
    auto* roceCtx = static_cast<AicpuTsRoceServerSocketContext*>(roceEp->GetServerSocketContext());
    CHK_PTR_NULL(roceCtx);
    CHK_RET(roceCtx->ServerSocketListen(port));
    SocketWlistInfo wlistEntry{};
    wlistEntry.connLimit = 1U;
    const auto bin = remoteIp.GetBinaryAddress();
    wlistEntry.remoteIp.addr = bin.addr;
    wlistEntry.remoteIp.addr6 = bin.addr6;
    s32 mw = memcpy_s(wlistEntry.tag, sizeof(wlistEntry.tag), socketTag.c_str(), socketTag.size() + 1U);
    CHK_PRT_RET(
        mw != EOK, HCCL_ERROR("[AicpuTsRoceChannel][%s] memcpy_s whitelist tag failed", SocketRoleTag()),
        HCCL_E_MEMORY);
    const std::vector<SocketWlistInfo> wlistVec = {wlistEntry};
    CHK_RET(roceCtx->AddListenSocketWhiteList(port, wlistVec));
    CHK_RET(roceCtx->GetSocket(port, socketTag, dataSocket_));
    CHK_SMART_PTR_NULL(dataSocket_);
    HCCL_INFO("[AicpuTsRoceChannel][server] BuildDataSocket accepted client connection");
    return HCCL_SUCCESS;
}

HcclResult AicpuTsRoceChannel::AssignDispatcherCommId()
{
    char commBuf[160];
    int nc = snprintf_s(commBuf, sizeof(commBuf), sizeof(commBuf) - 1U, "hcomm_roce_ch_%p", static_cast<void*>(this));
    CHK_PRT_RET(nc < 0, HCCL_ERROR("[AicpuTsRoceChannel] snprintf_s commId failed"), HCCL_E_INTERNAL);
    dispatcherCommId_.assign(commBuf);
    return HCCL_SUCCESS;
}

HcclResult AicpuTsRoceChannel::EnsureDispatcherCtx(u32 devPhyId)
{
    DispatcherCtxPtr ctx = nullptr;
    if (!FindDispatcherByCommId(&ctx, dispatcherCommId_.c_str())) {
        CHK_RET(CreateDispatcherCtx(&ctx, devPhyId, dispatcherCommId_.c_str()));
        ownsDispatcherCtx_ = true;
    } else {
        ownsDispatcherCtx_ = false;
    }
    dispatcherCtx_ = ctx;
    CHK_PTR_NULL(dispatcherCtx_);
    return HCCL_SUCCESS;
}

HcclResult AicpuTsRoceChannel::ConfigureMachineParaForTransport()
{
    machinePara_.machineType
        = isLocalIpClient_ ? hccl::MachineType::MACHINE_CLIENT_TYPE : hccl::MachineType::MACHINE_SERVER_TYPE;
    machinePara_.linkMode = hccl::LinkMode::LINK_DUPLEX_MODE;
    machinePara_.tag = dispatcherCommId_;
    machinePara_.localDeviceId = localEp_.loc.device.devPhyId;
    machinePara_.remoteDeviceId = remoteEp_.loc.device.devPhyId;
    CHK_RET(hrtGetDevice(&machinePara_.deviceLogicId));
    DevType devType = DevType::DEV_TYPE_COUNT;
    CHK_RET(hrtGetDeviceType(devType));
    machinePara_.deviceType = devType;
    machinePara_.nicDeploy = NICDeployment::NIC_DEPLOYMENT_DEVICE;
    machinePara_.userMemEnable = false;
    machinePara_.drainEnable = true;
    machinePara_.isIndOp = true;
    machinePara_.isAicpuModeEn = true;
    machinePara_.notifyNum = 0;
    // 解析用户配置的队列深度：0/INVALID_UINT表示使用默认值，否则使用用户配置（已在CheckRoceAttr校验）
    auto resolveQueueDepth = [](uint32_t userVal, uint32_t defaultVal) -> u32 {
        return (userVal != 0U && userVal != INVALID_UINT) ? userVal : defaultVal;
    };
    machinePara_.queueDepthAttr.sqDepth = resolveQueueDepth(channelDesc_.roceAttr.sqDepth, kAicpuTsRoceSqCqDepth);
    machinePara_.queueDepthAttr.sendCqDepth = resolveQueueDepth(channelDesc_.roceAttr.scqDepth, kAicpuTsRoceSqCqDepth);
    machinePara_.sockets.clear();
    machinePara_.sockets.push_back(dataSocket_);
    if (channelDesc_.roceAttr.tc != kHcommTrafficClassConfigNotSet) {
        machinePara_.tc = channelDesc_.roceAttr.tc;
    } else {
        machinePara_.tc = EnvConfig::HCCL_RDMA_TC_DEFAULT;
    }
    if (channelDesc_.roceAttr.sl != kHcommServiceLevelConfigNotSet) {
        machinePara_.sl = channelDesc_.roceAttr.sl;
    } else {
        machinePara_.sl = EnvConfig::HCCL_RDMA_SL_DEFAULT;
    }
    if (channelDesc_.roceAttr.retryCnt != INVALID_UINT) {
        machinePara_.retryCnt = channelDesc_.roceAttr.retryCnt;
    }
    if (channelDesc_.roceAttr.retryInterval != INVALID_UINT) {
        machinePara_.retryInterval = channelDesc_.roceAttr.retryInterval;
    }
    return HCCL_SUCCESS;
}

constexpr u32 TRANSPORT_PARA_DEFAULT_TIMEOUT = 120000; // 默认超时时间
void AicpuTsRoceChannel::ConfigureTransportParaForRoce()
{
    transportPara_.timeout = std::chrono::milliseconds(TRANSPORT_PARA_DEFAULT_TIMEOUT);
    transportPara_.nicDeploy = NICDeployment::NIC_DEPLOYMENT_DEVICE;
}

HcclResult AicpuTsRoceChannel::CreateAndInitTransport(HcclDispatcher dispatcher)
{
    if (machinePara_.drainEnable) {
        notifyPool_.reset(new (std::nothrow) hccl::NotifyPool());
        CHK_SMART_PTR_NULL(notifyPool_);
        CHK_RET(notifyPool_->Init(localEp_.loc.device.devPhyId));
        CHK_RET(notifyPool_->RegisterOp(machinePara_.tag));
    }

    EXCEPTION_CATCH(
        transport_ = std::make_unique<hccl::Transport>(
            hccl::TransportType::TRANS_TYPE_IBV_EXP, transportPara_, dispatcher, notifyPool_, machinePara_),
        return HCCL_E_PTR);
    CHK_SMART_PTR_NULL(transport_);
    HCCL_INFO("[AicpuTsRoceChannel][%s] Transport Init start", SocketRoleTag());
    HcclResult tr = transport_->Init();
    if (tr != HCCL_SUCCESS) {
        transport_.reset();
        return tr;
    }
    return HCCL_SUCCESS;
}

HcclResult AicpuTsRoceChannel::BuildDispatcherAndTransport()
{
    const u32 devPhyId = static_cast<u32>(localEp_.loc.device.devPhyId);
    CHK_RET(AssignDispatcherCommId());
    HCCL_INFO(
        "[AicpuTsRoceChannel][%s] BuildDispatcherAndTransport commId[%s]", SocketRoleTag(), dispatcherCommId_.c_str());

    CHK_RET(EnsureDispatcherCtx(devPhyId));
    auto* dctx = static_cast<hccl::DispatcherCtx*>(dispatcherCtx_);
    const HcclDispatcher dispatcher = dctx->GetDispatcher();
    CHK_PTR_NULL(dispatcher);

    CHK_RET(ConfigureMachineParaForTransport());
    ConfigureTransportParaForRoce();
    CHK_RET(CreateAndInitTransport(dispatcher));
    inited_ = true;
    HCCL_INFO("[AicpuTsRoceChannel][%s] BuildDispatcherAndTransport done, transport inited", SocketRoleTag());
    return HCCL_SUCCESS;
}

HcclResult AicpuTsRoceChannel::Init()
{
    HCCL_INFO("[AicpuTsRoceChannel] Init start");
    CHK_RET(ParseInputParam());
    CHK_RET(BuildDataSocket());
    roceStatus_ = RoceStatus::SOCKET_CONNECTING;
    HCCL_INFO("[AicpuTsRoceChannel][%s] Init success", SocketRoleTag());
    return HCCL_SUCCESS;
}

ChannelStatus AicpuTsRoceChannel::GetStatus()
{
    switch (roceStatus_) {
        case RoceStatus::INIT:
            return ChannelStatus::INIT;
        case RoceStatus::SOCKET_CONNECTING:
            if (dataSocket_->GetStatus() == hccl::HcclSocketStatus::SOCKET_OK) {
                roceStatus_ = RoceStatus::SOCKET_OK;
                return GetStatus();
            }
            if (dataSocket_->GetStatus() == hccl::HcclSocketStatus::SOCKET_TIMEOUT) {
                roceStatus_ = RoceStatus::FAILED;
                dataSocket_->Close();
                return ChannelStatus::SOCKET_TIMEOUT;
            }
            if (dataSocket_->GetStatus() == hccl::HcclSocketStatus::SOCKET_ERROR) {
                roceStatus_ = RoceStatus::FAILED;
                dataSocket_->Close();
                return ChannelStatus::FAILED;
            }
            return ChannelStatus::INIT; // socket尚未建立连接
        case RoceStatus::SOCKET_OK:
            if (BuildDispatcherAndTransport() == HCCL_SUCCESS) {
                roceStatus_ = RoceStatus::READY;
                return ChannelStatus::READY;
            }
            roceStatus_ = RoceStatus::FAILED;
            dataSocket_->Close();
            return ChannelStatus::FAILED;
        case RoceStatus::READY:
            return ChannelStatus::READY;
        case RoceStatus::FAILED:
            dataSocket_->Close();
            return ChannelStatus::FAILED;
    }
    return ChannelStatus::INIT;
}

HcommChannelKind AicpuTsRoceChannel::GetChannelKind() const { return HcommChannelKind::AICPU_TS_ROCE; }

HcclResult AicpuTsRoceChannel::GetNotifyNum(uint32_t* notifyNum) const
{
    CHK_PTR_NULL(notifyNum);
    *notifyNum = notifyNum_;
    return HCCL_SUCCESS;
}

// 单边通信暂未使用，接口先保留但返回不支持
HcclResult AicpuTsRoceChannel::GetRemoteMems(uint32_t* memNum, CommMem** remoteMem, char*** memInfos)
{
    (void)remoteMem;
    (void)memInfos;
    (void)memNum;
    HCCL_DEBUG("[AicpuTsRoceChannel][%s] GetRemoteMems not supported for AICPU TS RoCE channel", SocketRoleTag());
    return HCCL_E_NOT_SUPPORT;
}

// 单边通信暂未使用，接口先保留但返回不支持
HcclResult AicpuTsRoceChannel::Clean()
{
    HCCL_INFO("[AicpuTsRoceChannel][%s] Clean not supported for AICPU TS RoCE channel", SocketRoleTag());
    return HCCL_E_NOT_SUPPORT;
}

// 单边通信暂未使用，接口先保留但返回不支持
HcclResult AicpuTsRoceChannel::Resume()
{
    HCCL_INFO(
        "[AicpuTsRoceChannel][%s] Resume not implemented, no resume needed for AICPU TS RoCE channel", SocketRoleTag());
    return HCCL_E_NOT_SUPPORT;
}

HcclResult AicpuTsRoceChannel::ValidateSerializeParams(u32 qpNum, size_t localMemCount, size_t remoteMemCount) const
{
    CHK_PRT_RET(
        qpNum > RDMA_QP_MAX_NUM || qpNum < 1U, HCCL_ERROR("[AicpuTsRoceChannel] bad qpNum[%u]", qpNum),
        HCCL_E_INTERNAL);
    CHK_PRT_RET(
        localMemCount > 0U && localMemCount > (SIZE_MAX / sizeof(RoceMemDetails)),
        HCCL_ERROR("[AicpuTsRoceChannel][Serialize] localMem count overflow"), HCCL_E_PARA);
    CHK_PRT_RET(
        remoteMemCount > 0U && remoteMemCount > (SIZE_MAX / sizeof(RoceMemDetails)),
        HCCL_ERROR("[AicpuTsRoceChannel][Serialize] remoteMem count overflow"), HCCL_E_PARA);
    const u64 localBytes = static_cast<u64>(localMemCount * sizeof(RoceMemDetails));
    const u64 remoteBytes = static_cast<u64>(remoteMemCount * sizeof(RoceMemDetails));
    CHK_PRT_RET(
        localBytes > static_cast<u64>(UINT32_MAX) || remoteBytes > static_cast<u64>(UINT32_MAX),
        HCCL_ERROR("[AicpuTsRoceChannel][Serialize] mem detail blob too large"), HCCL_E_PARA);
    return HCCL_SUCCESS;
}

HcclResult AicpuTsRoceChannel::InitSerializeRoceChannelRes(
    HcommRoceChannelRes& res, size_t localMemCount, size_t remoteMemCount, void* localMem, void* remoteMem,
    const std::vector<HcclQpInfoV2>& aiQpInfos, u32 qpNum) const
{
    res = HcommRoceChannelRes{};
    res.localMemCount = static_cast<u32>(localMemCount);
    res.remoteMemCount = static_cast<u32>(remoteMemCount);
    res.localMem = localMem;
    res.remoteMem = remoteMem;
    res.chipId = LLONG_MAX;
    std::copy_n(aiQpInfos.begin(), static_cast<std::ptrdiff_t>(qpNum), res.QpInfo);
    res.qpsPerConnection = qpNum - static_cast<u32>(qpNum > 1U);
    CHK_RET(SerializeDrainNotifyInfo(res));
    return HCCL_SUCCESS;
}

HcclResult AicpuTsRoceChannel::BuildSerializeChannelMem(
    AicpuTsRoceChannelMem& bundle, const std::vector<RoceMemDetails>& localMd,
    const std::vector<RoceMemDetails>& remoteMd, const std::vector<HcclQpInfoV2>& aiQpInfos, u32 qpNum) const
{
    const size_t nL = localMd.size();
    const size_t nR = remoteMd.size();
    const u64 localBytes = static_cast<u64>(nL * sizeof(RoceMemDetails));
    const u64 remoteBytes = static_cast<u64>(nR * sizeof(RoceMemDetails));

    EXCEPTION_CATCH(bundle.resAlloc = hccl::DeviceMem::alloc(sizeof(HcommRoceChannelRes)), return HCCL_E_PTR);
    CHK_PTR_NULL(bundle.resAlloc.ptr());
    if (nL > 0U) {
        EXCEPTION_CATCH(bundle.localAlloc = hccl::DeviceMem::alloc(localBytes), return HCCL_E_PTR);
        CHK_PTR_NULL(bundle.localAlloc.ptr());
        CHK_RET(hrtMemSyncCopy(
            bundle.localAlloc.ptr(), localBytes, localMd.data(), localBytes,
            HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE));
    }
    if (nR > 0U) {
        EXCEPTION_CATCH(bundle.remoteAlloc = hccl::DeviceMem::alloc(remoteBytes), return HCCL_E_PTR);
        CHK_PTR_NULL(bundle.remoteAlloc.ptr());
        CHK_RET(hrtMemSyncCopy(
            bundle.remoteAlloc.ptr(), remoteBytes, remoteMd.data(), remoteBytes,
            HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE));
    }

    HcommRoceChannelRes res{};
    CHK_RET(InitSerializeRoceChannelRes(
        res, nL, nR, nL > 0U ? bundle.localAlloc.ptr() : nullptr, nR > 0U ? bundle.remoteAlloc.ptr() : nullptr,
        aiQpInfos, qpNum));

    CHK_RET(hrtMemSyncCopy(
        bundle.resAlloc.ptr(), sizeof(res), &res, sizeof(res), HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE));
    return HCCL_SUCCESS;
}

HcclResult AicpuTsRoceChannel::Serialize(std::shared_ptr<hccl::DeviceMem>& out)
{
    out.reset();
    HCCL_INFO("[AicpuTsRoceChannel][%s] Serialize start", SocketRoleTag());
    CHK_PRT_RET(
        !inited_, HCCL_ERROR("[AicpuTsRoceChannel][%s][Serialize] channel not inited", SocketRoleTag()),
        HCCL_E_INTERNAL);

    std::vector<RoceMemDetails> localMd;
    std::vector<RoceMemDetails> remoteMd;
    std::vector<HcclQpInfoV2> aiQpInfos;
    u32 qpNum = 0;

    auto* ep = static_cast<Endpoint*>(endpointHandle_);
    CHK_PTR_NULL(ep);
    auto* mgr = dynamic_cast<AicpuTsRoceRegedMemMgr*>(ep->GetRegedMemMgr());
    CHK_PTR_NULL(mgr);
    CHK_RET(mgr->GetAllMemDetails(localMd, remoteMd));
    CHK_RET(transport_->GetAiQpInfo(aiQpInfos));
    qpNum = static_cast<u32>(aiQpInfos.size());
    const size_t nL = localMd.size();
    const size_t nR = remoteMd.size();
    CHK_RET(ValidateSerializeParams(qpNum, nL, nR));

    AicpuTsRoceChannelMem bundle;
    CHK_RET(BuildSerializeChannelMem(bundle, localMd, remoteMd, aiQpInfos, qpNum));

    std::shared_ptr<AicpuTsRoceChannelMem> bundleKeep;
    EXCEPTION_CATCH(bundleKeep = std::make_shared<AicpuTsRoceChannelMem>(std::move(bundle)), return HCCL_E_PTR);

    hccl::DeviceMem* viewPtr = nullptr;
    EXCEPTION_CATCH(
        viewPtr = new hccl::DeviceMem(hccl::DeviceMem::create(bundleKeep->resAlloc.ptr(), sizeof(HcommRoceChannelRes))),
        return HCCL_E_PTR);

    out = std::shared_ptr<hccl::DeviceMem>(viewPtr, [bundleKeep](hccl::DeviceMem* p) {
        delete p;
    });
    HCCL_INFO(
        "[AicpuTsRoceChannel][%s] Serialize done qpNum[%u] localMem[%zu] remoteMem[%zu]", SocketRoleTag(), qpNum, nL,
        nR);
    return HCCL_SUCCESS;
}

HcclResult AicpuTsRoceChannel::NotifyRecord([[maybe_unused]] const uint32_t remoteNotifyIdx)
{
    HCCL_INFO("[AicpuTsRoceChannel::%s] not supported yet.", __func__);
    return HCCL_E_NOT_SUPPORT;
}

HcclResult
AicpuTsRoceChannel::NotifyWait([[maybe_unused]] const uint32_t localNotifyIdx, [[maybe_unused]] const uint32_t timeout)
{
    HCCL_INFO("[AicpuTsRoceChannel::%s] not supported yet.", __func__);
    return HCCL_E_NOT_SUPPORT;
}

HcclResult AicpuTsRoceChannel::WriteWithNotify(
    [[maybe_unused]] void* dst, [[maybe_unused]] const void* src, [[maybe_unused]] const uint64_t len,
    [[maybe_unused]] uint32_t remoteNotifyIdx)
{
    HCCL_INFO("[AicpuTsRoceChannel::%s] not supported yet.", __func__);
    return HCCL_E_NOT_SUPPORT;
}

HcclResult
AicpuTsRoceChannel::Write([[maybe_unused]] void* dst, [[maybe_unused]] const void* src, [[maybe_unused]] uint64_t len)
{
    HCCL_INFO("[AicpuTsRoceChannel::%s] not supported yet.", __func__);
    return HCCL_E_NOT_SUPPORT;
}

HcclResult
AicpuTsRoceChannel::Read([[maybe_unused]] void* dst, [[maybe_unused]] const void* src, [[maybe_unused]] uint64_t len)
{
    HCCL_INFO("[AicpuTsRoceChannel::%s] not supported yet.", __func__);
    return HCCL_E_NOT_SUPPORT;
}

HcclResult AicpuTsRoceChannel::ChannelFence()
{
    HCCL_INFO("[AicpuTsRoceChannel::%s] not supported yet.", __func__);
    return HCCL_E_NOT_SUPPORT;
}

HcclResult AicpuTsRoceChannel::SerializeDrainNotifyInfo(HcommRoceChannelRes& res) const
{
    void* remoteAddr = nullptr;
    uint32_t remoteKey = 0;
    uint32_t notifySize = 0;
    void* localAddr = nullptr;
    uint32_t localKey = 0;
    CHK_SMART_PTR_NULL(transport_);
    CHK_RET(transport_->GetDrainRemSrcMem(remoteAddr, remoteKey, notifySize));
    CHK_RET(transport_->GetDrainLocalDataNotify(localAddr, localKey, res.localDataSignal));

    res.remoteNotifyAddr = remoteAddr;
    res.remoteNotifyKey = remoteKey;
    res.localDataNotifyAddr = localAddr;
    res.localDataNotifyKey = localKey;
    res.notifySize = notifySize;
    HCCL_DEBUG(
        "[%s] remoteNotifyAddr[%p], remoteNotifyKey[%u], localDataNotifyAddr[%p], localDataNotifyKey[%u],"
        "notifySize[%u].",
        __func__, res.remoteNotifyAddr, res.remoteNotifyKey, res.localDataNotifyAddr, res.localDataNotifyKey,
        res.notifySize);
    return HCCL_SUCCESS;
}
} // namespace hcomm
