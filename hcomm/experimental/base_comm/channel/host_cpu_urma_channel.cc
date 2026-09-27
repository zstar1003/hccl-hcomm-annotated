/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "host_cpu_urma_channel.h"
#include "endpoint.h"
#include "orion_adpt_utils.h"
#include "hcomm_res.h"
#include "hcomm_adapter_urma.h"
#include "urma_api.h"

// Orion
#include "coll_alg_param.h"
#include "topo_common_types.h"
#include "virtual_topo.h"
#include "aicpu_res_package_helper.h"

namespace hcomm_experimental {
constexpr u32 FENCE_TIMEOUT_MS = 30 * 1000; // 定义最大等待30秒
constexpr u32 MEMORY_BLOCK_SIZE = 128;
constexpr uint16_t DEFAULT_LISTENING_PORT = 60001;
constexpr uint32_t kHostResourceId = 0U;

HostCpuUrmaChannel::HostCpuUrmaChannel(EndpointHandle endpointHandle, const HcommChannelDesc& channelDesc)
    : endpointHandle_(endpointHandle),
      channelDesc_(channelDesc)
{}

HostCpuUrmaChannel::~HostCpuUrmaChannel()
{
    if (socket_ != nullptr) {
        hcomm::SocketMgr::GetInstance(devicePhyId_).PutSocket(socketConfig_, socket_);
        socket_ = nullptr;
    }
}

HcclResult HostCpuUrmaChannel::ParseInputParam()
{
    // 1. 从 endpointHandle_，获得 localEp_ 和 rdmaHandle_
    Endpoint* localEpPtr = reinterpret_cast<Endpoint*>(endpointHandle_);
    CHK_PTR_NULL(localEpPtr);
    localEp_ = localEpPtr->GetEndpointDesc();
    rdmaHandle_ = localEpPtr->GetRdmaHandle();

    HCCL_INFO("[HostCpuUrmaChannel][%s] localProtocol[%d]", __func__, localEp_.protocol);

    // 2. 从 channelDesc_，获得 remoteEp_, socket_ 和 notifyNum
    remoteEp_ = channelDesc_.remoteEndpoint;
    socket_ = reinterpret_cast<Hccl::Socket*>(channelDesc_.socket);

    if (channelDesc_.exchangeAllMems) {
        // 3. Get memHandles from endpoint
        HCCL_INFO("[HostCpuUrmaChannel][%s] exchangeAllMems == True. Get memHandles from endpoint.", __func__);
        std::shared_ptr<Hccl::LocalUbRmaBuffer>* memHandles = nullptr;
        uint32_t memHandleNum = 0;
        CHK_RET(localEpPtr->GetAllMemHandles(reinterpret_cast<void**>(&memHandles), &memHandleNum));
        HCCL_INFO("[HostCpuUrmaChannel][%s] Got memHandleNum[%u].", __func__, memHandleNum);
        for (uint32_t i = 0; i < memHandleNum; ++i) {
            std::shared_ptr<Hccl::LocalUbRmaBuffer>& localUbRmaBuffer = memHandles[i];
            auto buf = localUbRmaBuffer->GetBuf();
            CHK_PTR_NULL(buf);
            HCCL_INFO(
                "[HostCpuUrmaChannel][%s] Got memHandle No.%u: addr[0x%llx], size[0x%llx], memInfo[%s].", __func__, i,
                localUbRmaBuffer->GetAddr(), localUbRmaBuffer->GetSize(), buf->GetMemInfo().c_str());
            bufs_.emplace_back(std::move(std::make_shared<Hccl::Buffer>(
                localUbRmaBuffer->GetAddr(), localUbRmaBuffer->GetSize(), buf->GetMemType(),
                buf->GetMemInfo().c_str())));
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
        reinterpret_cast<uint64_t>(endpointHandle_), port);
    if (port == 0) {
        port = DEFAULT_LISTENING_PORT;
        HCCL_INFO("[HostCpuUrmaChannel::%s] channelDesc port is 0, use default port [%u]", __func__, port);
    }
    Endpoint* localEpPtr = reinterpret_cast<Endpoint*>(endpointHandle_);
    CHK_PTR_NULL(localEpPtr);
    CHK_RET(localEpPtr->ServerSocketListen(port));
    HCCL_INFO("[HostCpuUrmaChannel::%s] SUCCESS. port[%u].", __func__, port);
    return HCCL_SUCCESS;
}

HcclResult HostCpuUrmaChannel::BuildSocket()
{
    if (socket_ != nullptr) {
        return HCCL_SUCCESS;
    }
    HCCL_INFO("[HostCpuUrmaChannel::%s] socket ptr is NULL, rebuild Socket", __func__);

    Hccl::LinkData linkData = hcomm::BuildDefaultLinkData();
    CHK_RET(hcomm::EndpointDescPairToLinkData(localEp_, remoteEp_, linkData));
    HCCL_INFO("[HostCpuUrmaChannel::%s] built linkData: %s", __func__, linkData.Describe().c_str());
    uint16_t port = channelDesc_.port;
    if (port == 0) {
        port = DEFAULT_LISTENING_PORT;
        HCCL_INFO("[HostCpuUrmaChannel::%s] channelDesc port is 0, use default port [%u]", __func__, port);
    }
    std::string socketTag = "AUTOMATIC_SOCKET_TAG";
    Hccl::SocketConfig socketConfig
        = (channelDesc_.role != HCOMM_SOCKET_ROLE_RESERVED) ?
              Hccl::SocketConfig(linkData, port, socketTag, channelDesc_.role == HCOMM_SOCKET_ROLE_SERVER) :
              Hccl::SocketConfig(linkData, socketTag, true);
    CHK_RET(hcomm::SocketMgr::GetInstance(devicePhyId_).GetSocket(socketConfig, socket_));
    HCCL_INFO("[HostCpuUrmaChannel::%s] SUCCESS. port[%u].", __func__, port);
    return HCCL_SUCCESS;
}

HcclResult HostCpuUrmaChannel::BuildConnection()
{
    Hccl::OpMode opMode = Hccl::OpMode::OPBASE;
    Hccl::LinkProtocol protocol;
    CHK_RET(hcomm::CommProtocolToLinkProtocol(localEp_.protocol, protocol));

    Hccl::IpAddress locAddr;
    Hccl::IpAddress rmtAddr;
    CHK_RET(hcomm::CommAddrToIpAddress(localEp_.commAddr, locAddr));
    CHK_RET(hcomm::CommAddrToIpAddress(remoteEp_.commAddr, rmtAddr));

    HCCL_INFO("[HostCpuUrmaChannel::%s] init TpManager with hostResourceId[%u].", __func__, kHostResourceId);
    Hccl::TpManager::GetInstance(kHostResourceId).Init();

    std::unique_ptr<Hccl::HostUbConnection> ubConn = nullptr;
    switch (protocol) {
        case Hccl::LinkProtocol::UB_TP:
            EXCEPTION_CATCH(
                ubConn = std::make_unique<Hccl::HostUbTpConnection>(rdmaHandle_, locAddr, rmtAddr, opMode),
                return HCCL_E_PTR);
            break;
        case Hccl::LinkProtocol::UB_CTP:
            EXCEPTION_CATCH(
                ubConn = std::make_unique<Hccl::HostUbCtpConnection>(rdmaHandle_, locAddr, rmtAddr, opMode),
                return HCCL_E_PTR);
            break;
        default:
            HCCL_ERROR("%s No LinkProtocol protocol[%s] to match", __func__, protocol.Describe().c_str());
            break;
    }
    CHK_SMART_PTR_NULL(ubConn);

    commonRes_.connVec.clear();
    commonRes_.connVec.emplace_back(ubConn.get());
    connections_.clear();
    connections_.push_back(std::move(ubConn));

    return HCCL_SUCCESS;
}

HcclResult HostCpuUrmaChannel::BuildBuffer()
{
    localRmaBuffers_.clear();
    commonRes_.bufferVec.clear();
    for (size_t i = 0; i < bufs_.size(); i++) {
        std::unique_ptr<Hccl::LocalUbRmaBuffer> bufferPtr = nullptr;
        EXCEPTION_CATCH(bufferPtr = std::make_unique<Hccl::LocalUbRmaBuffer>(bufs_[i], rdmaHandle_), return HCCL_E_PTR);
        commonRes_.bufferVec.push_back(bufferPtr.get());
        localRmaBuffers_.push_back(std::move(bufferPtr));
    }
    return HCCL_SUCCESS;
}

HcclResult HostCpuUrmaChannel::BuildUbMemTransport()
{
    Hccl::BaseMemTransport::LocCntNotifyRes locCntNotifyRes{};
    const Hccl::Socket& socket = *socket_;
    bool isRecvFirst = socket.GetRole() == Hccl::SocketRole::CLIENT ? true : false;

    Hccl::LinkData linkData = hcomm::BuildDefaultLinkData();
    CHK_RET(hcomm::EndpointDescPairToLinkData(localEp_, remoteEp_, linkData));

    // make_unique / make_shared / release 包一层抛异常的宏
    EXCEPTION_CATCH(
        memTransport_ = std::make_unique<Hccl::UbMemTransport>(
            commonRes_, attr_, linkData, socket, rdmaHandle_, locCntNotifyRes, isRecvFirst),
        return HCCL_E_PTR);
    return HCCL_SUCCESS;
}

HcclResult HostCpuUrmaChannel::Init()
{
    devicePhyId_ = kHostResourceId;
    CHK_RET(ParseInputParam());
    if (channelDesc_.role != HCOMM_SOCKET_ROLE_CLIENT) {
        CHK_RET(StartListen());
    }
    CHK_RET(BuildSocket());
    CHK_RET(BuildConnection());
    CHK_RET(BuildBuffer());
    CHK_RET(BuildUbMemTransport());
    // urma函数初始化
    CHK_RET(hcomm::DlUrmaFunction::GetInstance().DlUrmaFunctionInit());
    // 获取urma read/write 单个wr的最大传输数据大小
    CHK_RET(hcomm::HccpRaGetDevBaseAttr(rdmaHandle_, &devBaseAttr_));

    return HCCL_SUCCESS;
}

HcclResult HostCpuUrmaChannel::GetNotifyNum(uint32_t* notifyNum) const
{
    (void)notifyNum;
    HCCL_INFO("[HostCpuUrmaChannel::%s] not supported yet.", __func__);
    return HCCL_E_NOT_SUPPORT;
}

HcclResult HostCpuUrmaChannel::GetRemoteMem(HcclMem** remoteMem, uint32_t* memNum, char** memTags)
{
    CommMem* remoteCommMem = nullptr;
    char** memInfos = nullptr;
    CHK_RET(memTransport_->GetRemoteMems(memNum, &remoteCommMem, &memInfos));
    *remoteMem = reinterpret_cast<HcclMem*>(remoteCommMem);
    if (memTags != nullptr && memInfos != nullptr) {
        for (uint32_t i = 0; i < *memNum; ++i) {
            memTags[i] = memInfos[i];
        }
    }
    return HCCL_SUCCESS;
}

HcclResult HostCpuUrmaChannel::GetUserRemoteMem(CommMem** remoteMem, char*** memTags, uint32_t* memNum)
{
    return memTransport_->GetRemoteMems(memNum, remoteMem, memTags);
}

ChannelStatus HostCpuUrmaChannel::GetStatus()
{
    memTransport_->SetIsHost();
    ChannelStatus out = Channel::TransportStatusToChannelStatus(memTransport_->GetStatus(), localEp_, channelDesc_);
    if (out == ChannelStatus::READY && socket_ != nullptr) {
        hcomm::SocketMgr::GetInstance(devicePhyId_).PutSocket(socketConfig_, socket_);
    }
    return out;
}

HcclResult HostCpuUrmaChannel::NotifyRecord(const uint32_t remoteNotifyIdx)
{
    (void)remoteNotifyIdx;
    HCCL_INFO("[HostCpuUrmaChannel::%s] not supported yet.", __func__);
    return HCCL_E_NOT_SUPPORT;
}

HcclResult HostCpuUrmaChannel::NotifyWait(const uint32_t localNotifyIdx, const uint32_t timeout)
{
    (void)localNotifyIdx;
    (void)timeout;
    HCCL_INFO("[HostCpuUrmaChannel::%s] not supported yet.", __func__);
    return HCCL_E_NOT_SUPPORT;
}

HcclResult HostCpuUrmaChannel::WriteWithNotify(void* dst, const void* src, const uint64_t len, uint32_t remoteNotifyIdx)
{
    (void)dst;
    (void)src;
    (void)len;
    (void)remoteNotifyIdx;
    HCCL_INFO("[HostCpuUrmaChannel::%s] not supported yet.", __func__);
    return HCCL_E_NOT_SUPPORT;
}

HcclResult HostCpuUrmaChannel::GetLocSeg(const void* addr, const size_t size, u64* seg)
{
    if (localRmaBuffers_.empty()) {
        HCCL_ERROR("[HostCpuUrmaChannel::%s] localRmaBuffers is empty.", __func__);
        return HCCL_E_INTERNAL;
    }

    bool isAddrInRange = false;
    for (auto& it : localRmaBuffers_) {
        Hccl::Buffer iterBuf(it->GetBufferInfo().first, it->GetBufferInfo().second);
        if (iterBuf.Contains(reinterpret_cast<uintptr_t>(addr), size)) {
            *seg = it->GetTargetSeg();
            isAddrInRange = true;
            break;
        }
    }

    if (!isAddrInRange) {
        HCCL_ERROR("GetLocSeg addr[%p] size[%llu] is not in localRmaBuffers_", addr, size);
        return HCCL_E_INTERNAL;
    }
    return HCCL_SUCCESS;
}

HcclResult HostCpuUrmaChannel::GetSplitNum(uint64_t len, uint64_t maxJettyWrDataLen, uint64_t& splitNum)
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

HcclResult HostCpuUrmaChannel::UrmaPostJettySendWr(urma_opcode_t opcode, void* dst, const void* src, uint64_t len)
{
    // 构造urma的wr
    urma_jfs_wr_t urmaWriteWr{};
    urmaWriteWr.opcode = opcode;
    urmaWriteWr.flag.bs.place_order = (fenceFlag_ == true ? 2 : 1);
    urmaWriteWr.flag.bs.comp_order = 1; // comp_order要一直保持为1,
    urmaWriteWr.flag.bs.fence = (fenceFlag_ == true ? 1 : 0);
    urmaWriteWr.flag.bs.complete_enable = 0;
    urmaWriteWr.flag.bs.inline_flag = 0;
    urmaWriteWr.tjetty = reinterpret_cast<urma_target_jetty_t*>(connections_[0]->GetTJettyVa());
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
        urma_sge_t srclist{};
        urmaWriteWr.rw.src.sge = &srclist;
        urmaWriteWr.rw.src.sge->addr = reinterpret_cast<uint64_t>(static_cast<char*>(const_cast<void*>(src)) + offset);
        urmaWriteWr.rw.src.sge->len = chunkLen;
        urmaWriteWr.rw.src.sge->tseg = (opcode == URMA_OPC_WRITE) ? reinterpret_cast<urma_target_seg_t*>(localSeg) :
                                                                    reinterpret_cast<urma_target_seg_t*>(remoteSeg);
        urmaWriteWr.rw.src.num_sge = 1;

        // 目的地址 数据长度 tseg
        urma_sge_t dstlist{};
        urmaWriteWr.rw.dst.sge = &dstlist;
        urmaWriteWr.rw.dst.sge->addr = reinterpret_cast<uint64_t>(static_cast<const char*>(dst) + offset); // 远端地址
        urmaWriteWr.rw.dst.sge->len = chunkLen;
        urmaWriteWr.rw.dst.sge->tseg = (opcode == URMA_OPC_WRITE) ? reinterpret_cast<urma_target_seg_t*>(remoteSeg) :
                                                                    reinterpret_cast<urma_target_seg_t*>(localSeg);
        urmaWriteWr.rw.dst.num_sge = 1;

        // 只有最后一个wr上报cqe
        if (i == splitNum - 1) {
            urmaWriteWr.flag.bs.complete_enable = 1;
            urmaWriteWr.flag.bs.place_order = 2; // 最后一个wr设置为strong order
        }
        CHK_RET(hcomm::HrtUrmaPostJettySendWr(
            reinterpret_cast<urma_jetty_t*>(connections_[0]->GetJettyVa()), &urmaWriteWr, &badWr));
        offset += chunkLen;
    }
    fenceFlag_ = false;
    wqeNum_++;
    HCCL_INFO(
        "UrmaPostJettySendWr opencode[%u] fenceFlag_[%u] wqeNum_[%u] splitNum[%llu] SUCCESS.", opcode, fenceFlag_,
        wqeNum_, splitNum);
    return HCCL_SUCCESS;
}

HcclResult HostCpuUrmaChannel::Write(void* dst, const void* src, uint64_t len)
{
    CHK_RET(UrmaPostJettySendWr(URMA_OPC_WRITE, dst, src, len));
    return HCCL_SUCCESS;
}

HcclResult HostCpuUrmaChannel::Read(void* dst, const void* src, uint64_t len)
{
    CHK_RET(UrmaPostJettySendWr(URMA_OPC_READ, dst, src, len));
    return HCCL_SUCCESS;
}

HcclResult HostCpuUrmaChannel::ChannelFence()
{
    std::lock_guard<std::mutex> lock(fenceMutex_);
    HCCL_INFO("[HostCpuUrmaChannel::%s] start, wqeNum_ = %u va[%llu]", __func__, wqeNum_, connections_[0]->GetCqVa());
    CHK_PRT_RET(
        wqeNum_ == 0, HCCL_INFO("[HostCpuUrmaChannel::%s] no need to fence since no wqeNum[%u].", __func__),
        HCCL_SUCCESS);
    std::vector<urma_cr_t> wc(wqeNum_);

    auto timeout = std::chrono::milliseconds(FENCE_TIMEOUT_MS);
    auto startTime = std::chrono::steady_clock::now();
    while (true) {
        auto actualNum
            = hcomm::HrtUrmaPollJfc(reinterpret_cast<urma_jfc_t*>(connections_[0]->GetCqVa()), wqeNum_, wc.data());
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
    fenceFlag_ = true;
    return HCCL_SUCCESS;
}

HcclResult HostCpuUrmaChannel::Clean()
{
    HCCL_INFO("[HostCpuUrmaChannel::%s] not supported yet.", __func__);
    return HCCL_E_NOT_SUPPORT;
}

HcclResult HostCpuUrmaChannel::Resume()
{
    HCCL_INFO("[HostCpuUrmaChannel::%s] not supported yet.", __func__);
    return HCCL_E_NOT_SUPPORT;
}

} // namespace hcomm_experimental
