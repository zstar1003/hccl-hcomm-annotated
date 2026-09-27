/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <memory>
#include "aicpu_ts_hccs_channel.h"
#include "endpoint.h"
#include "../../../endpoints/aicputs_hccs_endpoint.h"
#include "hccs_reged_mem_mgr.h"
#include "../../../endpoints/net_dev/global_net_dev_manager.h"
#include "channel_param.h"
#include "inner/remote_ipc_rma_buffer.h"
#include "inner/local_ipc_rma_buffer.h"
#include "hcomm_c_adpt.h"
#include "hccl_socket_manager.h"
#include "externalinput_pub.h"
// for hccl_network.h
#include "inner/local_rdma_rma_buffer.h"
#include "inner/remote_rdma_rma_buffer.h"
#include "hccl_network.h"

using LocalIpcRmaBufferMgr
    = hcomm::RmaBufferMgr<hccl::BufferKey<uintptr_t, u64>, std::shared_ptr<hccl::LocalIpcRmaBuffer>>;
using RemoteIpcRmaBufferMgr
    = hcomm::RmaBufferMgr<hccl::BufferKey<uintptr_t, u64>, std::shared_ptr<hccl::RemoteIpcRmaBuffer>>;

using namespace hccl;

namespace hcomm {
AicpuTsHccsChannel::AicpuTsHccsChannel(EndpointHandle endpointHandle, const HcommChannelDesc& channelDesc)
    : endpointHandle_(endpointHandle),
      channelDesc_(channelDesc)
{}

AicpuTsHccsChannel::~AicpuTsHccsChannel()
{
    try {
        TransportDeInit();
    } catch (...) {
    }

    try {
        DisableMemAccess();
    } catch (...) {
    }

    try {
        DestroyConnection();
    } catch (...) {
    }

    try {
        DisableP2P();
    } catch (...) {
    }
}

HcclResult AicpuTsHccsChannel::ParseInputParam()
{
    CHK_RET(static_cast<HcclResult>(HcommEndpointGet(endpointHandle_, reinterpret_cast<void**>(&localEpPtr_))));
    CHK_PTR_NULL(localEpPtr_);

    localEp_ = localEpPtr_->GetEndpointDesc();

    remoteEp_ = channelDesc_.remoteEndpoint;
    notifyNum_ = channelDesc_.notifyNum;

    serverPort_ = channelDesc_.port != 0 ? channelDesc_.port : AICPU_CHANNEL_DEFAULT_PORT;

    CHK_RET(GetFirstIpByPhyId(localEp_.loc.device.devPhyId, localEp_.loc.device.superDevId, localIp_));
    CHK_RET(GetFirstIpByPhyId(remoteEp_.loc.device.devPhyId, remoteEp_.loc.device.superDevId, remoteIp_));
    std::string localReadableAddress = localIp_.GetReadableAddress();
    std::string remoteReadableAddress = remoteIp_.GetReadableAddress();

    if (channelDesc_.role == HCOMM_SOCKET_ROLE_SERVER) {
        isSocketServer_ = true;
    } else if (channelDesc_.role != HCOMM_SOCKET_ROLE_CLIENT) {
        HCCL_WARNING(
            "[AicpuTsHccsChannel] unexpected channelDesc.role[%d]; "
            "using inner logic to decide socket role based on endpoint IPs",
            static_cast<int>(channelDesc_.role));
        if (localReadableAddress < remoteReadableAddress) {
            isSocketServer_ = true;
        }
    }

    HCCL_INFO(
        "[AicpuTsHccsChannel][ParseInputParam] local devPhyId [%u] ip[%s] remote devPhyId[%u] ip[%s], "
        "isSocketServer_[%u], serverPort_[%u]",
        localEp_.loc.device.devPhyId, localReadableAddress.c_str(), remoteEp_.loc.device.devPhyId,
        remoteReadableAddress.c_str(), static_cast<u32>(isSocketServer_), serverPort_);

    return HCCL_SUCCESS;
}

HcclResult AicpuTsHccsChannel::GetFirstIpByPhyId(u32 devicePhyId, u32 superDevId, HcclIpAddress& ip) const
{
    CHK_RET(GlobalNetDevMgr::GetDeviceVnicIP(devicePhyId, superDevId, ip));
    HCCL_INFO(
        "[AicpuTsHccsChannel][GetFirstIpByPhyId]devicePhyId[%u] superDevId[%u] linkInfo.ip[%s]", devicePhyId,
        superDevId, ip.GetReadableAddress());
    return HCCL_SUCCESS;
}

HcclResult AicpuTsHccsChannel::BuildConnection()
{
    /* delay start server here, upper layer may not call ServerSocketListen of endpoint,
    and here can get the port from channel desc*/
    CHK_RET(hccl::GlobalNetDevMgr::GetInstance(localEp_.loc.device.devPhyId).ServerInit(serverPort_));
    serverInited_ = true;

    std::string localReadableAddress = localIp_.GetReadableAddress();
    std::string remoteReadableAddress = remoteIp_.GetReadableAddress();

    HCCL_INFO(
        "[AicpuTsHccsChannel][BuildConnection] local devPhyId [%u] ip[%s] remote devPhyId[%u] ip[%s]",
        localEp_.loc.device.devPhyId, localReadableAddress.c_str(), remoteEp_.loc.device.devPhyId,
        remoteReadableAddress.c_str());

    if (channelDesc_.channelName != nullptr) {
        socketTag_ = std::string(channelDesc_.channelName);
    } else if (isSocketServer_) {
        GlobalNetDevMgr::MakeSocketTag(localIp_, serverPort_, remoteIp_, socketTag_);
    } else {
        GlobalNetDevMgr::MakeSocketTag(remoteIp_, serverPort_, localIp_, socketTag_);
    }

    if (isSocketServer_) {
        CHK_RET(GlobalNetDevMgr::GetInstance(localEp_.loc.device.devPhyId)
                    .AcceptClient(serverPort_, remoteIp_, socketTag_, socket_));
    } else {
        CHK_RET(GlobalNetDevMgr::GetInstance(localEp_.loc.device.devPhyId)
                    .ConnectToServer(serverPort_, remoteIp_, serverPort_, socketTag_, socket_));
    }
    HCCL_INFO(
        "[AicpuTsHccsChannel][BuildConnection] local devPhyId [%u] ip[%s] "
        "remote devPhyId[%u] ip[%s] socketTag_[%s]",
        localEp_.loc.device.devPhyId, localReadableAddress.c_str(), remoteEp_.loc.device.devPhyId,
        remoteReadableAddress.c_str(), socketTag_.c_str());
    return HCCL_SUCCESS;
}

void AicpuTsHccsChannel::DestroyConnection()
{
    if (socket_ != nullptr) {
        GlobalNetDevMgr::GetInstance(localEp_.loc.device.devPhyId).CloseSocket(socket_);
    }

    if (serverInited_) {
        (void)hccl::GlobalNetDevMgr::GetInstance(localEp_.loc.device.devPhyId).ServerDeInit(serverPort_);
        serverInited_ = false;
    }
    HCCL_INFO("[AicpuTsHccsChannel][%s] finish DestroyConnection", __func__);
}

HcclResult AicpuTsHccsChannel::SetMachinePara(hccl::MachinePara& machinePara)
{
    CHK_RET(hrtGetDeviceType(machinePara.deviceType));

    u32 deviceLogicId;
    CHK_RET(hrtGetDeviceIndexByPhyId(localEp_.loc.device.devPhyId, deviceLogicId));
    machinePara.deviceLogicId = static_cast<s32>(deviceLogicId);
    machinePara.tag = socketTag_;
    machinePara.notifyNum = channelDesc_.notifyNum;
    machinePara.linkMode = hccl::LinkMode::LINK_DUPLEX_MODE;
    ;
    machinePara.specifyLink = LinkTypeInServer::RESERVED_LINK_TYPE;
    machinePara.machineType
        = isSocketServer_ ? hccl::MachineType::MACHINE_SERVER_TYPE : hccl::MachineType::MACHINE_CLIENT_TYPE;
    machinePara.serverId = localEp_.loc.device.serverIdx;
    machinePara.localDeviceId = localEp_.loc.device.devPhyId;
    machinePara.remoteDeviceId = remoteEp_.loc.device.devPhyId;
    machinePara.localIpAddr = socket_->GetLocalIp();
    machinePara.remoteIpAddr = socket_->GetRemoteIp();
    machinePara.localSocketPort = socket_->GetLocalPort();
    machinePara.remoteSocketPort = socket_->GetRemotePort();
    machinePara.srcPorts = std::vector<std::uint16_t>(1, 0); /* 默认填充一个元素，0代表默认不配置 */
    machinePara.mem.clear();
    machinePara.linkAttribute = 0x03; /* 0x03同时支持目的端和源端发起 */
    machinePara.sockets.push_back(socket_);
    machinePara.exchangeInfo.resize(sizeof(HccsExchangeInfo));
    machinePara.isNewOneSide = true;
    return HCCL_SUCCESS;
}

void AicpuTsHccsChannel::SetTransportParam(hccl::TransportPara& para) const
{
    std::chrono::milliseconds kdefaultTimeout = std::chrono::seconds(GetExternalInputHcclLinkTimeOut());
    para.timeout = kdefaultTimeout;
    para.virtualFlag = false;
}

HcclResult AicpuTsHccsChannel::TransportInit()
{
    hccl::MachinePara machinePara = {};
    CHK_RET(SetMachinePara(machinePara));

    hccl::TransportPara para = {};
    SetTransportParam(para);

    CHK_RET(HcclDispatcherInit(DispatcherType::DISPATCHER_NORMAL, localEp_.loc.device.devPhyId, &dispatcher_));
    CHK_SMART_PTR_NULL(dispatcher_);

    if (!FindDispatcherByCommId(&dispatcherCtx_, DEFAULT_DISPATCH_NAME)) {
        CHK_RET(CreateDispatcherCtx(&dispatcherCtx_, localEp_.loc.device.devPhyId, DEFAULT_DISPATCH_NAME));
    }
    CHK_PTR_NULL(dispatcherCtx_);

    notifyPool_.reset(new (std::nothrow) hccl::NotifyPool());
    CHK_SMART_PTR_NULL(notifyPool_);
    CHK_RET(notifyPool_->Init(localEp_.loc.device.devPhyId));
    CHK_RET(notifyPool_->RegisterOp(machinePara.tag));

    transport_.reset(new (std::nothrow)
                         Transport(TransportType::TRANS_TYPE_P2P, para, dispatcher_, notifyPool_, machinePara));

    CHK_RET(transport_->Init());

    HCCL_INFO("[AicpuTsHccsChannel][%s] finish TransportInit", __func__);
    return HCCL_SUCCESS;
}

void AicpuTsHccsChannel::TransportDeInit()
{
    if (transport_ != nullptr) {
        transport_ = nullptr;
    }
    if (notifyPool_ != nullptr) {
        notifyPool_ = nullptr;
    }
    if (dispatcherCtx_ != nullptr) {
        (void)DestroyDispatcherCtx(dispatcherCtx_, DEFAULT_DISPATCH_NAME);
        dispatcherCtx_ = nullptr;
    }
    if (dispatcher_ != nullptr) {
        (void)HcclDispatcherDestroy(dispatcher_);
        dispatcher_ = nullptr;
    }
    HCCL_INFO("[AicpuTsHccsChannel][%s] finish TransportDeInit", __func__);
}

HcclResult AicpuTsHccsChannel::EnableP2P()
{
    CHK_PTR_NULL(localEpPtr_);
    auto* mgr = static_cast<HccsRegedMemMgr*>(localEpPtr_->GetRegedMemMgr());
    CHK_PTR_NULL(mgr);
    CHK_RET(mgr->MemoryEnableP2P(localEpPtr_->GetEndpointDesc(), remoteEp_));
    HCCL_INFO("[AicpuTsHccsChannel][%s] finish EnableP2P", __func__);
    return HCCL_SUCCESS;
}

void AicpuTsHccsChannel::DisableP2P()
{
    if (localEpPtr_ != nullptr) {
        auto* mgr = static_cast<HccsRegedMemMgr*>(localEpPtr_->GetRegedMemMgr());
        if (mgr != nullptr) {
            (void)mgr->MemoryDisableP2P(localEpPtr_->GetEndpointDesc(), remoteEp_);
        }
    }

    HCCL_INFO("[AicpuTsHccsChannel][%s] finish DisableP2P", __func__);
}

HcclResult AicpuTsHccsChannel::EnableMemAccess()
{
    s32 pid = 0;
    CHK_RET(SalGetBareTgid(&pid));
    // switch first
    HcommMemGrantInfo localGrantInfo = {localEp_.loc.device.superDevId, pid};
    HcommMemGrantInfo remoteGrantInfo = {};
    if (isSocketServer_) {
        CHK_RET(socket_->Recv(&remoteGrantInfo, sizeof(HcommMemGrantInfo)));
        CHK_RET(socket_->Send(&localGrantInfo, sizeof(HcommMemGrantInfo)));
    } else {
        CHK_RET(socket_->Send(&localGrantInfo, sizeof(HcommMemGrantInfo)));
        CHK_RET(socket_->Recv(&remoteGrantInfo, sizeof(HcommMemGrantInfo)));
    }
    CHK_PTR_NULL(localEpPtr_);
    // MemoryGrant 由 HccsRegedMemMgr 承载；本 channel 只服务 HCCS endpoint，直接 static_cast
    auto* mgr = static_cast<HccsRegedMemMgr*>(localEpPtr_->GetRegedMemMgr());
    CHK_PTR_NULL(mgr);
    CHK_RET(mgr->MemoryGrant(&remoteGrantInfo));
    // need to wait peer grant for me end, not need to check value, just make sure grant process end
    u32 localGrantSync = 1;
    u32 remoteGrantSync = 1;
    if (isSocketServer_) {
        CHK_RET(socket_->Recv(&remoteGrantSync, sizeof(u32)));
        CHK_RET(socket_->Send(&localGrantSync, sizeof(u32)));
    } else {
        CHK_RET(socket_->Send(&localGrantSync, sizeof(u32)));
        CHK_RET(socket_->Recv(&remoteGrantSync, sizeof(u32)));
    }
    CHK_RET(mgr->MemoryOpenRemoteIpc());
    HCCL_INFO("[AicpuTsHccsChannel][%s] finish EnableMemAccess", __func__);
    return HCCL_SUCCESS;
}

void AicpuTsHccsChannel::DisableMemAccess() const
{
    if (localEpPtr_ != nullptr) {
        auto* mgr = static_cast<HccsRegedMemMgr*>(localEpPtr_->GetRegedMemMgr());
        if (mgr != nullptr) {
            (void)mgr->MemoryCloseRemoteIpc();
        }
    }
    HCCL_INFO("[AicpuTsHccsChannel][%s] finish DisableMemAccess", __func__);
}

HcclResult AicpuTsHccsChannel::Init()
{
    CHK_RET(ParseInputParam());
    CHK_RET(EnableP2P());
    HcclResult ret = BuildConnection();
    if (ret != HCCL_SUCCESS) {
        DestroyConnection();
        DisableP2P();
        return ret;
    }

    ret = EnableMemAccess();
    if (ret != HCCL_SUCCESS) {
        DisableMemAccess();
        DestroyConnection();
        DisableP2P();
        return ret;
    }

    ret = TransportInit();
    if (ret != HCCL_SUCCESS) {
        TransportDeInit();
        DisableMemAccess();
        DestroyConnection();
        DisableP2P();
        return ret;
    }
    HCCL_INFO("[AicpuTsHccsChannel][%s] finish Init", __func__);
    return HCCL_SUCCESS;
}

HcclResult AicpuTsHccsChannel::GetRemoteMems(uint32_t* memNum, CommMem** remoteMem, [[maybe_unused]] char*** memInfos)
{
    remoteIpcRmaBufferVec_.clear();
    auto* mgr = static_cast<HccsRegedMemMgr*>(localEpPtr_->GetRegedMemMgr());
    CHK_PTR_NULL(mgr);
    CHK_RET(mgr->GetRemoteIpcRmaBuffer(remoteIpcRmaBufferVec_));
    *remoteMem = remoteIpcRmaBufferVec_.data();
    *memNum = remoteIpcRmaBufferVec_.size();
    return HCCL_SUCCESS;
}

ChannelStatus AicpuTsHccsChannel::GetStatus()
{
    ChannelStatus out = ChannelStatus::READY;
    return out;
}

HcclResult AicpuTsHccsChannel::GetNotifyNum(uint32_t* notifyNum) const
{
    *notifyNum = notifyNum_;
    return HCCL_SUCCESS;
}

HcclResult AicpuTsHccsChannel::BuildHcclChannelHccsRes(HcclChannelHccsRes& channelHccsRes)
{
    HcclChannelP2p& linkp2p = channelHccsRes.channelP2p;

    CHK_SAFETY_FUNC_RET(memcpy_s(
        channelHccsRes.channelTag, sizeof(channelHccsRes.channelTag) - 1, socketTag_.c_str(), socketTag_.length()));
    HCCL_DEBUG("[AicpuTsHccsChannel][%s] channelHccsRes.channelTag[%s]", __func__, channelHccsRes.channelTag);

    linkp2p.remoteHcclbuffer.addr = nullptr;
    linkp2p.remoteHcclbuffer.size = 0;
    linkp2p.remoteUserMem = nullptr;
    linkp2p.remoteUserMemCount = 0;
    linkp2p.qos = channelDesc_.qos;

    HCCL_DEBUG("[AicpuTsHccsChannel][%s] finish set remoteMem info", __func__);

    u64 notifyNum = 0;
    channelHccsRes.p2pNotifyNum = transport_->GetNotifyNum();
    HCCL_DEBUG(
        "[AicpuTsHccsChannel][%s] finish set localnotify & remotenotify info, "
        "notifyNum[%llu], p2pNotifyNum[%llu]",
        __func__, notifyNum, channelHccsRes.p2pNotifyNum);
    CHK_RET(transport_->GetTransportAttr(linkp2p.transportAttr));

    DevType devType;
    CHK_RET(hrtGetDeviceType(devType));
    channelHccsRes.deviceType = static_cast<u32>(devType);
    channelHccsRes.remoteDevicePhyId = remoteEp_.loc.device.devPhyId;
    channelHccsRes.localDevicePhyId = localEp_.loc.device.devPhyId;
    channelHccsRes.machineType
        = isSocketServer_ ? hccl::MachineType::MACHINE_SERVER_TYPE : hccl::MachineType::MACHINE_CLIENT_TYPE;
    u32 deviceLogicId;
    CHK_RET(hrtGetDeviceIndexByPhyId(localEp_.loc.device.devPhyId, deviceLogicId));
    channelHccsRes.localDeviceLogicId = static_cast<s32>(deviceLogicId);

    auto* mgr = static_cast<HccsRegedMemMgr*>(localEpPtr_->GetRegedMemMgr());
    CHK_PTR_NULL(mgr);

    remoteIpcRmaBufferVecEx_.clear();
    CHK_RET(mgr->GetRemoteIpcRmaBufferEx(remoteIpcRmaBufferVecEx_));
    channelHccsRes.remoteBufSize = remoteIpcRmaBufferVecEx_.size();
    channelHccsRes.remoteBufMem = remoteIpcRmaBufferVecEx_.data();

    localIpcRmaBufferVecEx_.clear();
    CHK_RET(mgr->GetLocalIpcRmaBufferEx(localIpcRmaBufferVecEx_));
    channelHccsRes.localBufSize = localIpcRmaBufferVecEx_.size();
    channelHccsRes.localBufMem = localIpcRmaBufferVecEx_.data();

    HCCL_DEBUG("[AicpuTsHccsChannel][%s] finish set RemoteChannelP2pResParam info", __func__);
    return HCCL_SUCCESS;
}

HcclResult AicpuTsHccsChannel::Serialize(std::shared_ptr<hccl::DeviceMem>& out)
{
    HCCL_DEBUG("[AicpuTsHccsChannel][%s] start", __func__);
    HcclChannelHccsRes hostChannelHccsRes;
    CHK_RET(BuildHcclChannelHccsRes(hostChannelHccsRes));

    // 临时缓存信息
    HcclChannelHccsRes deviceChannelHccsRes = hostChannelHccsRes;

    // 计算设备内存分配的空间，包括需要深度拷贝的子域的信息内的内存，然后分配整块设备地址内存
    u64 outSize = 0;
    // cal base info
    u64 baseSize = sizeof(HcclChannelHccsRes);
    outSize += baseSize;
    // cal local buf mem
    size_t localBufSize = hostChannelHccsRes.localBufSize * sizeof(HcclMemEx);
    outSize += localBufSize;
    // cal remote buf mem
    size_t remoteBufSize = hostChannelHccsRes.remoteBufSize * sizeof(HcclMemEx);
    outSize += remoteBufSize;
    EXCEPTION_CATCH((out = std::make_shared<hccl::DeviceMem>(hccl::DeviceMem::alloc(outSize))), return HCCL_E_PTR);

    void* dstPtr = nullptr;
    // 复制 local buf
    if (hostChannelHccsRes.localBufSize > 0 && hostChannelHccsRes.localBufMem != nullptr) {
        // 使用设备地址重置 local buf的地址
        dstPtr = reinterpret_cast<uint8_t*>(out.get()->ptr()) + baseSize;
        deviceChannelHccsRes.localBufMem = reinterpret_cast<HcclMemEx*>(dstPtr);
        CHK_RET(hrtMemSyncCopy(
            deviceChannelHccsRes.localBufMem, localBufSize, hostChannelHccsRes.localBufMem, localBufSize,
            HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE));
    }

    // 复制 remote buf
    if (hostChannelHccsRes.remoteBufSize > 0 && hostChannelHccsRes.remoteBufMem != nullptr) {
        // 使用设备地址重置 remote buf的地址
        dstPtr = reinterpret_cast<uint8_t*>(out.get()->ptr()) + baseSize + localBufSize;
        deviceChannelHccsRes.remoteBufMem = reinterpret_cast<HcclMemEx*>(dstPtr);
        CHK_RET(hrtMemSyncCopy(
            deviceChannelHccsRes.remoteBufMem, remoteBufSize, hostChannelHccsRes.remoteBufMem, remoteBufSize,
            HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE));
    }

    // 复制 base
    CHK_RET(hrtMemSyncCopy(
        out.get()->ptr(), sizeof(HcclChannelHccsRes), &deviceChannelHccsRes, sizeof(HcclChannelHccsRes),
        HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE));

    HCCL_DEBUG("[AicpuTsHccsChannel][%s] end", __func__);
    return HCCL_SUCCESS;
}

HcclResult AicpuTsHccsChannel::Clean()
{
    HCCL_INFO("[AicpuTsHccsChannel][%s] Clean not implemented, no resume needed for AICPU TS Hccs channel", __func__);
    return HCCL_E_NOT_SUPPORT;
}

HcclResult AicpuTsHccsChannel::Resume()
{
    HCCL_INFO("[AicpuTsHccsChannel][%s] Resume not implemented, no resume needed for AICPU TS Hccs channel", __func__);
    return HCCL_E_NOT_SUPPORT;
}

HcommChannelKind AicpuTsHccsChannel::GetChannelKind() const { return HcommChannelKind::AICPU_TS_HCCS; }

HcclResult AicpuTsHccsChannel::NotifyRecord([[maybe_unused]] const uint32_t remoteNotifyIdx)
{
    HCCL_INFO("[AicpuTsHccsChannel::%s] not supported yet.", __func__);
    return HCCL_E_NOT_SUPPORT;
}

HcclResult
AicpuTsHccsChannel::NotifyWait([[maybe_unused]] const uint32_t localNotifyIdx, [[maybe_unused]] const uint32_t timeout)
{
    HCCL_INFO("[AicpuTsHccsChannel::%s] not supported yet.", __func__);
    return HCCL_E_NOT_SUPPORT;
}

HcclResult AicpuTsHccsChannel::WriteWithNotify(
    [[maybe_unused]] void* dst, [[maybe_unused]] const void* src, [[maybe_unused]] const uint64_t len,
    [[maybe_unused]] uint32_t remoteNotifyIdx)
{
    HCCL_INFO("[AicpuTsHccsChannel::%s] not supported yet.", __func__);
    return HCCL_E_NOT_SUPPORT;
}

HcclResult
AicpuTsHccsChannel::Write([[maybe_unused]] void* dst, [[maybe_unused]] const void* src, [[maybe_unused]] uint64_t len)
{
    HCCL_INFO("[AicpuTsHccsChannel::%s] not supported yet.", __func__);
    return HCCL_E_NOT_SUPPORT;
}

HcclResult
AicpuTsHccsChannel::Read([[maybe_unused]] void* dst, [[maybe_unused]] const void* src, [[maybe_unused]] uint64_t len)
{
    HCCL_INFO("[AicpuTsHccsChannel::%s] not supported yet.", __func__);
    return HCCL_E_NOT_SUPPORT;
}

HcclResult AicpuTsHccsChannel::ChannelFence()
{
    HCCL_INFO("[AicpuTsHccsChannel::%s] not supported yet.", __func__);
    return HCCL_E_NOT_SUPPORT;
}
} // namespace hcomm
