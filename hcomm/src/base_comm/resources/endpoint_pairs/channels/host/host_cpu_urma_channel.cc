/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "cast_utils.h"
#include "host_cpu_urma_channel.h"
#include "endpoint.h"
#include "../../sockets/socket_mgr.h"
#include "orion_adpt_utils.h"
#include "hcomm_adapter_urma.h"

// Orion
#include "topo_common_types.h"
#include "virtual_topo.h"
#include "env_config/env_config_v2.h"

namespace hcomm {
constexpr uint16_t DEFAULT_LISTENING_PORT = 60001;

HostCpuUrmaChannel::HostCpuUrmaChannel(EndpointHandle endpointHandle, const HcommChannelDesc& channelDesc)
    : endpointHandle_(endpointHandle),
      channelDesc_(channelDesc)
{}

HostCpuUrmaChannel::~HostCpuUrmaChannel()
{
    if (channelDesc_.socket == nullptr && socket_ != nullptr) {
        SocketMgr::GetInstance(devicePhyId_).PutSocket(socketConfig_, socket_);
        socket_ = nullptr;
    }
}

HcclResult HostCpuUrmaChannel::ParseInputParam()
{
    // 1. 从 endpointHandle_，获得 localEp_ 和 rdmaHandle_
    Endpoint* localEpPtr = static_cast<Endpoint*>(endpointHandle_);
    CHK_PTR_NULL(localEpPtr);
    localEp_ = localEpPtr->GetEndpointDesc();
    rdmaHandle_ = localEpPtr->GetRdmaHandle();

    HCCL_INFO("[HostCpuUrmaChannel][%s] localProtocol[%d]", __func__, localEp_.protocol);

    // 2. 从 channelDesc_，获得 remoteEp_, socket_ 和 notifyNum
    remoteEp_ = channelDesc_.remoteEndpoint;
    socket_ = static_cast<Hccl::Socket*>(channelDesc_.socket);
    commonRes_.bufferVec.clear();

    if (channelDesc_.exchangeAllMems) {
        // 3. Get memHandles from endpoint
        HCCL_INFO("[HostCpuUrmaChannel][%s] exchangeAllMems == True. Get memHandles from endpoint.", __func__);
        std::shared_ptr<Hccl::LocalUbRmaBuffer>* memHandles = nullptr;
        uint32_t memHandleNum = 0;
        CHK_RET(static_cast<HcclResult>(
            HcommMemGetAllMemHandles(endpointHandle_, ReinterpretAs<void**>(&memHandles), &memHandleNum)));
        HCCL_INFO("[HostCpuUrmaChannel][%s] Got memHandleNum[%u].", __func__, memHandleNum);
        for (uint32_t i = 0; i < memHandleNum; ++i) {
            std::shared_ptr<Hccl::LocalUbRmaBuffer>& localUbRmaBuffer = memHandles[i];
            CHK_SMART_PTR_NULL(localUbRmaBuffer);
            Hccl::Buffer* buf = localUbRmaBuffer->GetBuf();
            CHK_PTR_NULL(buf);
            HCCL_INFO(
                "[HostCpuUrmaChannel][%s] Got memHandle No.%u: addr[0x%llx], size[0x%llx], memType[%d], memInfo[%s].",
                __func__, i, static_cast<unsigned long long>(localUbRmaBuffer->GetAddr()),
                static_cast<unsigned long long>(localUbRmaBuffer->GetSize()), static_cast<int>(buf->GetMemType()),
                buf->GetMemInfo().c_str());
            commonRes_.bufferVec.push_back(localUbRmaBuffer.get());
        }
    } else {
        // 3. 从 channelDesc 的 memHandle，获得 bufs_
        HCCL_WARNING("[HostCpuUrmaChannel][%s] exchangeAllMems is false.", __func__);
    }

    return HCCL_SUCCESS;
}

HcclResult HostCpuUrmaChannel::StartListen()
{
    uint16_t port = channelDesc_.port;
    HCCL_INFO(
        "[HostCpuUrmaChannel::%s] Start. EndpointHandle[0x%llx], port[%u]", __func__,
        ReinterpretAs<uint64_t>(endpointHandle_), port);
    if (port == 0) {
        port = DEFAULT_LISTENING_PORT;
        HCCL_INFO("[HostCpuUrmaChannel::%s] channelDesc port is 0, use default port [%u]", __func__, port);
    }
    CHK_RET(static_cast<HcclResult>(HcommEndpointStartListen(endpointHandle_, port, nullptr)));
    HCCL_INFO("[HostCpuUrmaChannel::%s] SUCCESS. port[%u].", __func__, port);
    return HCCL_SUCCESS;
}

HcclResult HostCpuUrmaChannel::BuildSocket()
{
    if (socket_ != nullptr) {
        return HCCL_SUCCESS;
    }
    HCCL_INFO("[HostCpuUrmaChannel::%s] socket ptr is NULL, rebuild Socket", __func__);

    Hccl::LinkData linkData = BuildDefaultLinkData();
    CHK_RET(EndpointDescPairToLinkData(localEp_, remoteEp_, linkData));
    HCCL_INFO("[HostCpuUrmaChannel::%s] built linkData: %s", __func__, linkData.Describe().c_str());
    uint16_t port = channelDesc_.port;
    if (port == 0) {
        port = DEFAULT_LISTENING_PORT;
        HCCL_INFO("[HostCpuUrmaChannel::%s] channelDesc port is 0, use default port [%u]", __func__, port);
    }

    std::string socketTag
        = (channelDesc_.channelName != nullptr) ? std::string(channelDesc_.channelName) : "AUTOMATIC_SOCKET_TAG";
    Hccl::SocketConfig socketConfig
        = (channelDesc_.role != HCOMM_SOCKET_ROLE_RESERVED) ?
              Hccl::SocketConfig(linkData, port, socketTag, channelDesc_.role == HCOMM_SOCKET_ROLE_SERVER) :
              Hccl::SocketConfig(linkData, socketTag, true);
    CHK_RET(SocketMgr::GetInstance(devicePhyId_).GetSocket(socketConfig, socket_));

    HCCL_INFO("[HostCpuUrmaChannel::%s] SUCCESS. port[%u].", __func__, port);
    return HCCL_SUCCESS;
}

HcclResult HostCpuUrmaChannel::BuildConnection()
{
    UbConnBuildContext ctx;
    CHK_RET(PrepareUbConnBuildContext(localEp_, remoteEp_, channelDesc_, ctx));

    Hccl::OpMode opMode = Hccl::OpMode::OPBASE;
    std::unique_ptr<Hccl::HostUbConnection> ubConn = nullptr;
    switch (ctx.protocol) {
        case Hccl::LinkProtocol::UB_TP:
            EXCEPTION_CATCH(
                ubConn = std::make_unique<Hccl::HostUbTpConnection>(
                    rdmaHandle_, ctx.locAddr, ctx.rmtAddr, opMode, Hccl::HrtUbJfcMode::NORMAL, ctx.qosPre),
                return HCCL_E_PTR);
            break;
        case Hccl::LinkProtocol::UB_CTP:
            EXCEPTION_CATCH(
                ubConn = std::make_unique<Hccl::HostUbCtpConnection>(
                    rdmaHandle_, ctx.locAddr, ctx.rmtAddr, opMode, Hccl::HrtUbJfcMode::NORMAL, ctx.qosPre),
                return HCCL_E_PTR);
            break;
        default:
            HCCL_ERROR("%s No LinkProtocol protocol[%s] to match", __func__, ctx.protocol.Describe().c_str());
            break;
    }
    CHK_SMART_PTR_NULL(ubConn);

    commonRes_.connVec.clear();
    commonRes_.connVec.emplace_back(ubConn.get());
    connections_.clear();
    connections_.push_back(std::move(ubConn));

    return HCCL_SUCCESS;
}

HcclResult HostCpuUrmaChannel::BuildUbMemTransport()
{
    Hccl::BaseMemTransport::LocCntNotifyRes locCntNotifyRes{};
    const Hccl::Socket& socket = *socket_;
    bool isRecvFirst = socket.GetRole() == Hccl::SocketRole::CLIENT ? true : false;

    Hccl::LinkData linkData = BuildDefaultLinkData();
    CHK_RET(EndpointDescPairToLinkData(localEp_, remoteEp_, linkData));

    // make_unique / make_shared / release 包一层抛异常的宏
    EXCEPTION_CATCH(
        memTransport_ = std::make_unique<Hccl::UbMemTransport>(
            commonRes_, attr_, linkData, socket, rdmaHandle_, locCntNotifyRes, isRecvFirst),
        return HCCL_E_PTR);
    return HCCL_SUCCESS;
}

HcclResult HostCpuUrmaChannel::Init()
{
    s32 devLogicId;
    CHK_RET(hrtGetDevice(&devLogicId));
    CHK_RET(hrtGetDevicePhyIdByIndex(static_cast<u32>(devLogicId), devicePhyId_));
    CHK_RET(ParseInputParam());
    if (channelDesc_.role != HCOMM_SOCKET_ROLE_CLIENT) {
        CHK_RET(StartListen());
    }
    CHK_RET(BuildSocket());
    CHK_RET(BuildConnection());
    CHK_RET(BuildUbMemTransport());
    // urma函数初始化
    CHK_RET(DlUrmaFunction::GetInstance().DlUrmaFunctionInit());
    // 获取urma read/write 单个wr的最大传输数据大小
    CHK_RET(HccpRaGetDevBaseAttr(rdmaHandle_, &devBaseAttr_));

    return HCCL_SUCCESS;
}

HcclResult HostCpuUrmaChannel::GetNotifyNum([[maybe_unused]] uint32_t* notifyNum) const
{
    HCCL_INFO("[HostCpuUrmaChannel::%s] not supported yet.", __func__);
    return HCCL_SUCCESS;
}

HcclResult HostCpuUrmaChannel::GetRemoteMems(uint32_t* memNum, CommMem** remoteMem, char*** memInfos)
{
    return memTransport_->GetRemoteMems(memNum, remoteMem, memInfos);
}

ChannelStatus HostCpuUrmaChannel::GetStatus()
{
    memTransport_->SetIsHost();
    ChannelStatus out = Channel::TransportStatusToChannelStatus(memTransport_->GetStatus(), localEp_, GetChannelDesc());
    return out;
}

HcclResult hcomm::HostCpuUrmaChannel::NotifyRecord([[maybe_unused]] const uint32_t remoteNotifyIdx)
{
    HCCL_INFO("[HostCpuUrmaChannel::%s] not supported yet.", __func__);
    return HCCL_E_NOT_SUPPORT;
}

HcclResult hcomm::HostCpuUrmaChannel::NotifyWait(
    [[maybe_unused]] const uint32_t localNotifyIdx, [[maybe_unused]] const uint32_t timeout)
{
    HCCL_INFO("[HostCpuUrmaChannel::%s] not supported yet.", __func__);
    return HCCL_E_NOT_SUPPORT;
}

HcclResult hcomm::HostCpuUrmaChannel::WriteWithNotify(
    [[maybe_unused]] void* dst, [[maybe_unused]] const void* src, [[maybe_unused]] const uint64_t len,
    [[maybe_unused]] uint32_t remoteNotifyIdx)
{
    HCCL_INFO("[HostCpuUrmaChannel::%s] not supported yet.", __func__);
    return HCCL_E_NOT_SUPPORT;
}

HcclResult HostCpuUrmaChannel::GetLocSeg(const void* addr, const size_t size, u64* seg)
{
    if (commonRes_.bufferVec.empty()) {
        HCCL_ERROR("[HostCpuUrmaChannel::%s] commonRes_.bufferVec is empty.", __func__);
        return HCCL_E_INTERNAL;
    }

    bool isAddrInRange = false;
    for (auto& it : commonRes_.bufferVec) {
        CHK_PTR_NULL(it);
        Hccl::Buffer iterBuf(it->GetAddr(), it->GetSize());
        if (iterBuf.Contains(ReinterpretAs<uintptr_t>(addr), size)) {
            auto localUbRmaBuffer = dynamic_cast<Hccl::LocalUbRmaBuffer*>(it);
            CHK_PTR_NULL(localUbRmaBuffer);
            *seg = localUbRmaBuffer->GetTargetSeg();
            isAddrInRange = true;
            break;
        }
    }

    if (!isAddrInRange) {
        HCCL_ERROR("GetLocSeg addr[%p] size[%llu] is not in commonRes_.bufferVec", addr, size);
        return HCCL_E_INTERNAL;
    }
    return HCCL_SUCCESS;
}

HcclResult HostCpuUrmaChannel::GetSplitNum(uint64_t len, uint64_t maxJettyWrDataLen, uint64_t& splitNum) const
{
    if (len == 0 || maxJettyWrDataLen == 0) {
        HCCL_ERROR(
            "[HostCpuUrmaChannel::%s] invalid len[%llu] or maxJettyWrDataLen[%llu].", __func__, len, maxJettyWrDataLen);
        return HCCL_E_PARA;
    }
    if ((len % maxJettyWrDataLen) == 0) {
        splitNum = len / maxJettyWrDataLen;
    } else {
        splitNum = (len / maxJettyWrDataLen) + 1;
    }
    return HCCL_SUCCESS;
}

HcclResult HostCpuUrmaChannel::GetLocalAndRemoteSeg(
    urma_opcode_t opcode, void* dst, const void* src, uint64_t len, u64& localSeg, u64& remoteSeg)
{
    if (opcode == URMA_OPC_WRITE) {
        CHK_RET(GetLocSeg(src, len, &localSeg));
        CHK_RET(memTransport_->GetRemoteSeg(dst, len, &remoteSeg));
    } else if (opcode == URMA_OPC_READ) {
        CHK_RET(GetLocSeg(dst, len, &localSeg));
        CHK_RET(memTransport_->GetRemoteSeg(src, len, &remoteSeg));
    }
    return HCCL_SUCCESS;
}

constexpr u32 RELAX_ORDER = 1;  // Relax Order
constexpr u32 STRONG_ORDER = 2; // Strong Order
HcclResult HostCpuUrmaChannel::UrmaPostJettySendWr(urma_opcode_t opcode, void* dst, const void* src, uint64_t len)
{
    // 构造urma的wr
    urma_jfs_wr_t urmaWriteWr{};
    urmaWriteWr.opcode = opcode;
    urmaWriteWr.flag.bs.place_order = (fenceFlag_ == true ? STRONG_ORDER : RELAX_ORDER);
    urmaWriteWr.flag.bs.comp_order = 1; // comp_order要一直保持为1,
    urmaWriteWr.flag.bs.fence = (fenceFlag_ == true ? 1 : 0);
    urmaWriteWr.flag.bs.complete_enable = 0;
    urmaWriteWr.flag.bs.inline_flag = 0;
    urmaWriteWr.tjetty = ReinterpretAs<urma_target_jetty_t*>(connections_[0]->GetTJettyVa());
    urmaWriteWr.user_ctx = 0; // 跟ibvs中的wr_id对应
    urmaWriteWr.next = nullptr;

    //  获取切片数量
    uint64_t splitNum = 0;
    uint64_t maxJettyWrDataLen = (opcode == URMA_OPC_WRITE) ? devBaseAttr_.maxWriteSize : devBaseAttr_.maxReadSize;
    CHK_RET(GetSplitNum(len, maxJettyWrDataLen, splitNum));

    u64 localSeg;
    u64 remoteSeg;
    CHK_RET(GetLocalAndRemoteSeg(opcode, dst, src, len, localSeg, remoteSeg));

    uint64_t offset = 0;
    for (uint64_t i = 0; i < splitNum; i++) {
        urma_jfs_wr_t* badWr = nullptr;
        uint64_t chunkLen = std::min(len - offset, maxJettyWrDataLen);
        // 源地址 数据长度 tseg
        urma_sge_t srclist = {};
        urmaWriteWr.rw.src.sge = &srclist;
        urmaWriteWr.rw.src.sge->addr = ReinterpretAs<uint64_t>(static_cast<char*>(const_cast<void*>(src)) + offset);
        urmaWriteWr.rw.src.sge->len = chunkLen;
        urmaWriteWr.rw.src.sge->tseg = (opcode == URMA_OPC_WRITE) ? ReinterpretAs<urma_target_seg_t*>(localSeg) :
                                                                    ReinterpretAs<urma_target_seg_t*>(remoteSeg);
        urmaWriteWr.rw.src.num_sge = 1;

        // 目的地址 数据长度 tseg
        urma_sge_t dstlist = {};
        urmaWriteWr.rw.dst.sge = &dstlist;
        urmaWriteWr.rw.dst.sge->addr = ReinterpretAs<uint64_t>(static_cast<const char*>(dst) + offset); // 远端地址
        urmaWriteWr.rw.dst.sge->len = chunkLen;
        urmaWriteWr.rw.dst.sge->tseg = (opcode == URMA_OPC_WRITE) ? ReinterpretAs<urma_target_seg_t*>(remoteSeg) :
                                                                    ReinterpretAs<urma_target_seg_t*>(localSeg);
        urmaWriteWr.rw.dst.num_sge = 1;

        // 只有最后一个wr上报cqe
        if (i == splitNum - 1) {
            urmaWriteWr.flag.bs.complete_enable = 1;
            urmaWriteWr.flag.bs.place_order = STRONG_ORDER; // 最后一个wr设置为strong order
        }
        CHK_RET(
            HrtUrmaPostJettySendWr(ReinterpretAs<urma_jetty_t*>(connections_[0]->GetJettyVa()), &urmaWriteWr, &badWr));
        offset += chunkLen;
    }
    fenceFlag_ = false;
    wqeNum_++;
    HCCL_INFO(
        "UrmaPostJettySendWr opencode[%u] fenceFlag_[%u] wqeNum_[%u] splitNum[%llu] SUCCESS.", opcode, fenceFlag_,
        wqeNum_, splitNum);
    return HCCL_SUCCESS;
}

HcclResult hcomm::HostCpuUrmaChannel::Write(void* dst, const void* src, uint64_t len)
{
    CHK_RET(UrmaPostJettySendWr(URMA_OPC_WRITE, dst, src, len));
    return HCCL_SUCCESS;
}

HcclResult hcomm::HostCpuUrmaChannel::Read(void* dst, const void* src, uint64_t len)
{
    CHK_RET(UrmaPostJettySendWr(URMA_OPC_READ, dst, src, len));
    return HCCL_SUCCESS;
}

HcclResult HostCpuUrmaChannel::WaitForWqeCompletion()
{
    CHK_PRT_RET(
        wqeNum_ == 0, HCCL_INFO("[HostCpuUrmaChannel::%s] no need to wait since wqeNum_ is 0.", __func__),
        HCCL_SUCCESS);
    std::vector<urma_cr_t> wc(wqeNum_);

    auto timeout = std::chrono::milliseconds(
        static_cast<uint64_t>(Hccl::EnvConfig::GetInstance().GetRtsConfig().GetExecTimeOut())
        * 1000ULL); // 乘1000转为毫秒
    auto startTime = std::chrono::steady_clock::now();
    while (true) {
        auto actualNum = HrtUrmaPollJfc(ReinterpretAs<urma_jfc_t*>(connections_[0]->GetCqVa()), wqeNum_, wc.data());
        if (actualNum < 0) {
            HCCL_ERROR("[HostCpuUrmaChannel::%s] urma_poll_jfc failed. actualNum=%d", __func__, actualNum);
            return HCCL_E_NETWORK;
        }

        uint32_t actualNum32 = static_cast<uint32_t>(actualNum);
        if (actualNum32 > wqeNum_) {
            HCCL_ERROR(
                "[HostCpuUrmaChannel::%s] urma_poll_jfc polled more completions (%u) than expected (%u).", __func__,
                actualNum32, wqeNum_);
            return HCCL_E_INTERNAL;
        } else if (actualNum32 > 0) {
            for (uint32_t i = 0; i < actualNum32; i++) {
                if (wc[i].status != URMA_CR_SUCCESS) {
                    HCCL_ERROR(
                        "[HostCpuUrmaChannel::%s] urma_poll_jfc error. wc[%u] status:%d", __func__, i, wc[i].status);
                    return HCCL_E_NETWORK;
                }
            }
            wqeNum_ -= actualNum32; // 减去已完成的数量，继续等待剩余的完成
            if (wqeNum_ == 0) {
                break; // 所有的wqe都已完成，退出循环
            }
        }

        if ((std::chrono::steady_clock::now() - startTime) >= timeout) {
            HCCL_ERROR("[HostCpuUrmaChannel::%s] call urma_poll_jfc timeout.", __func__);
            return HCCL_E_TIMEOUT;
        }
    }

    wqeNum_ = 0; // 所有wqe都已完成，重置计算器
    return HCCL_SUCCESS;
}

HcclResult hcomm::HostCpuUrmaChannel::ChannelFence()
{
    std::lock_guard<std::mutex> lock(fenceMutex_);
    HCCL_INFO("[HostCpuUrmaChannel::%s] start, wqeNum_ = %u va[%llu]", __func__, wqeNum_, connections_[0]->GetCqVa());
    CHK_PRT_RET(
        wqeNum_ == 0, HCCL_INFO("[HostCpuUrmaChannel::%s] no need to fence since no wqeNum[%u].", __func__),
        HCCL_SUCCESS);
    CHK_RET(WaitForWqeCompletion());
    fenceFlag_ = true;
    return HCCL_SUCCESS;
}

HcclResult HostCpuUrmaChannel::ChannelDrain()
{
    std::lock_guard<std::mutex> lock(fenceMutex_);
    HCCL_INFO("[HostCpuUrmaChannel::%s] start, wqeNum_ = %u", __func__, wqeNum_);
    return WaitForWqeCompletion();
}

HcclResult hcomm::HostCpuUrmaChannel::Clean()
{
    HCCL_INFO("[HostCpuUrmaChannel::%s] not supported yet.", __func__);
    return HCCL_E_NOT_SUPPORT;
}

HcclResult hcomm::HostCpuUrmaChannel::Resume()
{
    HCCL_INFO("[HostCpuUrmaChannel::%s] not supported yet.", __func__);
    return HCCL_E_NOT_SUPPORT;
}

} // namespace hcomm
