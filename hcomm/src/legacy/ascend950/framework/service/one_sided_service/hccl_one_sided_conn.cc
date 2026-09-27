/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hccl_one_sided_conn_v2.h"
#include "connections_builder.h"
#include "hccl_net_dev.h"
#include "hccl_mem.h"
#include "communicator_impl.h"
#include "transport_urma_mem.h"
#include "hccl_log_keywords.h"
namespace Hccl {
using namespace std;

HcclOneSidedConn::HcclOneSidedConn(CommunicatorImpl* comm, LinkData linkData) : comm_(comm), linkData_(linkData) {}

HcclOneSidedConn::~HcclOneSidedConn()
{
    for (const auto& pair : desc2netDevMap_) {
        const HcclNetDev& hcclNetDev = pair.second;
        HcclResult ret = HcclNetDevClose(hcclNetDev);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR(
                "[HcclOneSidedConn][~HcclOneSidedConn]HcclNetDevClose failed, descStr[%s], ret[%d].",
                pair.first.c_str(), ret);
        }
    }
}
HcclResult HcclOneSidedConn::Connect([[maybe_unused]] const std::string& commId)
{
    HCCL_INFO("[HcclOneSidedConn]Connect start");

    // Socket/RmaConnection建链
    vector<LinkData> links;
    links.push_back(linkData_);
    comm_->GetSocketManager().BatchCreateSockets(links);
    make_unique<ConnectionsBuilder>(*comm_)->BatchBuild(comm_->GetId(), links);
    comm_->GetMemTransportManager()->BatchBuildOneSidedTransports(links);

    // Transport粒度申请notify，aicpu76行那个
    for (auto& link : links) {
        comm_->GetConnLocalNotifyManager().ApplyFor(link.GetRemoteRankId(), link);
    }

    // 推动式建链
    WaitOneSidedTransportReady();

    // 保存socket
    SocketConfig socketConfig(linkData_.GetRemoteRankId(), linkData_, comm_->GetEstablishLinkSocketTag());
    socket_ = comm_->GetSocketManager().GetConnectedSocket(socketConfig);
    if (socket_ == nullptr) {
        HCCL_ERROR("[HcclOneSidedConn]socket_ is nullptr");
        return HCCL_E_PTR;
    }

    // 创建TransportUrmaMem并保存
    transportMemPtr_ = make_shared<TransportUrmaMem>(
        comm_->GetMemTransportManager()->GetOneSidedTransport(linkData_), remoteHcclBufMgr_);
    return HCCL_SUCCESS;
}

void HcclOneSidedConn::WaitOneSidedTransportReady()
{
    auto timeout = std::chrono::seconds(EnvConfig::GetInstance().GetSocketConfig().GetLinkTimeOut());
    HcclUs startTime = std::chrono::steady_clock::now();
    while (true) {
        if (comm_->GetMemTransportManager()->IsAllOneSidedTransportReady()) {
            break;
        }
        if ((std::chrono::steady_clock::now() - startTime) >= timeout) {
            RPT_INPUT_ERR(
                true, "EI0006", std::vector<std::string>({"reason"}),
                std::vector<std::string>({"WaitOneSidedTransportReady timeout, SOCKET_TIMEOUT."}));
            HCCL_ERROR(
                "[%s][%s] wait socket establish timeout, WaitOneSidedTransportReady timeout.",
                LOG_KEYWORDS_INIT_CHANNEL.c_str(), LOG_KEYWORDS_TIMEOUT.c_str());
            THROW<InternalException>("WaitOneSidedTransportReady timeout.");
        }
    }
}

HcclResult HcclOneSidedConn::SendLocalMemDesc(const HcclMemDescs& localMemDescs)
{
    HCCL_INFO("[HcclOneSidedConn]SendLocalMemDesc start");
    socket_->Send((u8*)(&localMemDescs.arrayLength), sizeof(u32));
    HCCL_INFO("send localMemDescs.arrayLength:%u", localMemDescs.arrayLength);
    if (localMemDescs.arrayLength == 0) {
        HCCL_INFO("localMemDescs.arrayLength[%u], no need to send data", localMemDescs.arrayLength);
    } else {
        HCCL_INFO("send descSize:%u", localMemDescs.arrayLength * sizeof(HcclMemDesc));
        if (static_cast<u64>(localMemDescs.arrayLength) > static_cast<u64>(UINT32_MAX) / sizeof(HcclMemDesc)) {
            THROW<InternalException>("integer overflow occurs");
        }
        socket_->Send((u8*)(localMemDescs.array), localMemDescs.arrayLength * sizeof(HcclMemDesc));
    }
    return HCCL_SUCCESS;
}

HcclResult HcclOneSidedConn::ReceiveRemoteMemDesc(HcclMemDescs& remoteMemDescs, u32& actualNumOfRemote)
{
    HCCL_INFO("[HcclOneSidedConn]ReceiveRemoteMemDesc start");
    socket_->Recv((u8*)(&actualNumOfRemote), sizeof(u32));
    remoteMemDescs.arrayLength = actualNumOfRemote;
    HCCL_INFO("receive actualNumOfRemote:%u", actualNumOfRemote);
    if (remoteMemDescs.arrayLength == 0) {
        HCCL_INFO("actualNumOfRemote[%u], no need to receive data", remoteMemDescs.arrayLength);
    } else {
        if (remoteMemDescs.array == nullptr) {
            HCCL_ERROR(
                "[HcclOneSidedConn]remoteMemDescs.array is nullptr but actualNumOfRemote[%u] > 0", actualNumOfRemote);
            return HCCL_E_PTR;
        }
        HCCL_INFO("receive descSize:%u", actualNumOfRemote * sizeof(HcclMemDesc));
        socket_->Recv((u8*)remoteMemDescs.array, actualNumOfRemote * sizeof(HcclMemDesc));
    }
    return HCCL_SUCCESS;
}

HcclResult HcclOneSidedConn::ExchangeMemDesc(
    const HcclMemDescs& localMemDescs, HcclMemDescs& remoteMemDescs, u32& actualNumOfRemote)
{
    HCCL_INFO("[HcclOneSidedConn][ExchangeMemDesc]localRank[%u] exchange memDesc begin", comm_->GetMyRank());
    CHK_PRT_RET(
        (localMemDescs.array == nullptr) && (remoteMemDescs.array == nullptr),
        HCCL_ERROR(
            "[HcclOneSidedConn]localMemDesc array and remoteMemDesc array are both nullptr, do not need to exchange"),
        HCCL_E_PARA);
    CHK_PRT_RET(
        (localMemDescs.arrayLength == 0) && (remoteMemDescs.arrayLength == 0),
        HCCL_ERROR(
            "[HcclOneSidedConn]localMemDesc arrayLength = %u and remoteMemDescs arrayLength = %u , do "
            "not need to exchange",
            localMemDescs.arrayLength, remoteMemDescs.arrayLength),
        HCCL_E_PARA);

    if (socket_ == nullptr) {
        CHK_RET(Connect(comm_->GetId()));
    }

    if (socket_->GetRole() == SocketRole::CLIENT) {
        // 先收后发
        CHK_RET(ReceiveRemoteMemDesc(remoteMemDescs, actualNumOfRemote));
        CHK_RET(SendLocalMemDesc(localMemDescs));
    } else {
        // 先发后收
        CHK_RET(SendLocalMemDesc(localMemDescs));
        CHK_RET(ReceiveRemoteMemDesc(remoteMemDescs, actualNumOfRemote));
    }

    // 校验remoteDescs中的remoteRankId和conn对象中保存的localRankId是否一样
    for (u32 i = 0; i < actualNumOfRemote; i++) {
        CHK_PTR_NULL(remoteMemDescs.array);
        const RmaMemDesc* remoteRmaMemDesc = reinterpret_cast<const RmaMemDesc*>(remoteMemDescs.array[i].desc);
        CHK_PTR_NULL(remoteRmaMemDesc);
        RankId tempRankId = remoteRmaMemDesc->remoteRankId;
        HCCL_INFO("[TransportMem][ExchangeMemDesc]tempRankId:%u, userRank:%u", tempRankId, comm_->GetMyRank());
        if (tempRankId != comm_->GetMyRank()) {
            HCCL_ERROR(
                "[TransportMem][ExchangeMemDesc]localRank[%u] receive remoteMemDesc from wrong localRank[%u], "
                "connection is for localRank[%u]",
                comm_->GetMyRank(), tempRankId, comm_->GetMyRank());
            return HCCL_E_INTERNAL;
        }
    }

    return HCCL_SUCCESS;
}

HcclResult HcclOneSidedConn::EnableMemAccess(const HcclMemDesc& remoteMemDesc, HcclMem& remoteMem)
{
    HCCL_INFO("[HcclOneSidedConn]EnableMemAccess start");
    // 反序列化remoteMemDesc
    const RmaMemDesc* remoteRmaMemDesc = reinterpret_cast<const RmaMemDesc*>(remoteMemDesc.desc);
    std::vector<char> tempDesc(TRANSPORT_EMD_ESC_SIZE);
    tempDesc.assign(remoteRmaMemDesc->memDesc, remoteRmaMemDesc->memDesc + TRANSPORT_EMD_ESC_SIZE);
    ExchangeUbBufferDto dto;
    BinaryStream remoteRdmaRmaBufferStream(tempDesc);
    dto.Deserialize(remoteRdmaRmaBufferStream);

    // 导入内存描述符
    shared_ptr<HcclBuf> outBuf = make_shared<HcclBuf>();
    string tempStr = string(remoteRmaMemDesc->memDesc, TRANSPORT_EMD_ESC_SIZE);
    auto iter = desc2HcclBufMapRemoteUb_.find(tempStr);
    if (iter != desc2HcclBufMapRemoteUb_.end()) {
        outBuf = iter->second;
    } else {
        HcclNetDevInfos info;
        info.addr.protoType = HcclNetDevice::ConvertHcclProtoToLinkProto(linkData_.GetLocalPort().GetProto());
        info.addr.type = HCCL_ADDR_TYPE_IP_V4;
        info.netdevDeployment = HcclNetDevice::ConvertDeploymentType(linkData_.GetLocalPort().GetType());
        info.devicePhyId = comm_->GetDevicePhyId();
        info.addr.addr = linkData_.GetLocalPort().GetAddr().GetBinaryAddress().addr;
        HcclNetDev netDev;
        HcclResult ret = HcclNetDevOpen(&info, &netDev);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR("[HcclOneSidedConn][EnableMemAccess]HcclNetDevOpen failed, ret[%d]", ret);
            return ret;
        }
        desc2netDevMap_.emplace(tempStr, netDev);
        ret = HcclMemImport(remoteRmaMemDesc->memDesc, TRANSPORT_EMD_ESC_SIZE, true, outBuf.get(), netDev);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR("[HcclOneSidedConn][EnableMemAccess]EnableMemAccess failed, ret [%d]", ret);
            return ret;
        }
    }

    // 填充remoteMem
    remoteMem.type = static_cast<HcclMemType>(dto.memType);
    remoteMem.addr = outBuf->addr;
    remoteMem.size = outBuf->len;

    // 添加计数器
    BufferKey<uintptr_t, u64> tempKey(reinterpret_cast<uintptr_t>(outBuf->addr), outBuf->len);
    auto resultPair = remoteHcclBufMgr_.Add(tempKey, outBuf);
    if (resultPair.first == remoteHcclBufMgr_.End()) {
        HCCL_ERROR("[HcclOneSidedConn][EnableMemAccess]The memory overlaps with the memory has been enabled");
        return HCCL_E_INTERNAL;
    }

    // 存储remoteHcclBuf
    desc2HcclBufMapRemoteUb_.emplace(tempStr, outBuf);
    HCCL_INFO("[HcclOneSidedConn][EnableMemAccess] Enable memory access success.");
    return HCCL_SUCCESS;
}

HcclResult HcclOneSidedConn::DisableMemAccess(const HcclMemDesc& remoteMemDesc)
{
    HCCL_INFO("[HcclOneSidedConn]DisableMemAccess start");
    // 将HcclMemDesc转化为RmaMemDesc
    const RmaMemDesc* remoteRmaMemDesc = reinterpret_cast<const RmaMemDesc*>(remoteMemDesc.desc);
    string tempStr = string(remoteRmaMemDesc->memDesc, TRANSPORT_EMD_ESC_SIZE);
    auto it = desc2HcclBufMapRemoteUb_.find(tempStr);
    if (it == desc2HcclBufMapRemoteUb_.end()) {
        HCCL_ERROR("[HcclOneSidedConn][DisableMemAccess]Can't find hcclmem by key.");
        return HCCL_E_INTERNAL;
    }

    // 计数器删除HcclBuf
    HcclBuf* buf = it->second.get();
    BufferKey<uintptr_t, u64> tempKey(reinterpret_cast<uintptr_t>(it->second->addr), it->second->len);
    // 删除成功：输入key是表中某一最相近key的全集，计数-1后为0，返回true
    // 删除失败：输入key是表中某一最相近key的全集，计数-1后不为0（说明存在其他remoteRank使用），返回false
    auto resultPair = remoteHcclBufMgr_.Del(tempKey);
    if (resultPair) {
        HcclResult ret = HcclMemClose(buf);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR("[HcclOneSidedConn][DisableMemAccess]Close remote memory failed. ret[%d]", ret);
            return HCCL_E_INTERNAL;
        }
        desc2HcclBufMapRemoteUb_.erase(remoteRmaMemDesc->memDesc);
    }
    HCCL_INFO("[HcclOneSidedConn][DisableMemAccess] Disable memory access success.");
    return HCCL_SUCCESS;
}

HcclResult HcclOneSidedConn::BatchBufferSlice(
    const HcclOneSideOpDesc* oneSideDescs, u32 descNum,
    vector<HcclAicpuLocBufLite>& hostBatchPutGetLocalBufferSliceBufs,
    vector<HcclAicpuLocBufLite>& hostBatchPutGetRemoteBufferSliceBufs)
{
    RmaBufferSlice localRmaBufferSlice[descNum] = {};
    RmtRmaBufferSlice remoteRmaBufferSlice[descNum] = {};

    if (transportMemPtr_ != nullptr) {
        CHK_RET(transportMemPtr_->BatchBufferSlice(oneSideDescs, descNum, localRmaBufferSlice, remoteRmaBufferSlice));
    } else {
        THROW<InternalException>("transportMemPtr is nullptr");
    }

    for (u32 i = 0; i < descNum; i++) {
        hostBatchPutGetLocalBufferSliceBufs[i].addr = localRmaBufferSlice[i].addr;
        hostBatchPutGetLocalBufferSliceBufs[i].size = localRmaBufferSlice[i].size;
        hostBatchPutGetLocalBufferSliceBufs[i].tokenId
            = static_cast<LocalUbRmaBuffer*>(localRmaBufferSlice[i].buf)->GetTokenId();
        hostBatchPutGetLocalBufferSliceBufs[i].tokenValue
            = static_cast<LocalUbRmaBuffer*>(localRmaBufferSlice[i].buf)->GetTokenValue();
        HCCL_INFO(
            "hostBatchPutGetLocalBufferSliceBufs, addr=0x%llx, size=0x%llx",
            hostBatchPutGetLocalBufferSliceBufs[i].addr, hostBatchPutGetLocalBufferSliceBufs[i].size);

        hostBatchPutGetRemoteBufferSliceBufs[i].addr = remoteRmaBufferSlice[i].addr;
        hostBatchPutGetRemoteBufferSliceBufs[i].size = remoteRmaBufferSlice[i].size;
        hostBatchPutGetRemoteBufferSliceBufs[i].tokenId
            = static_cast<RemoteUbRmaBuffer*>(remoteRmaBufferSlice[i].buf)->GetTokenId();
        hostBatchPutGetRemoteBufferSliceBufs[i].tokenValue
            = static_cast<RemoteUbRmaBuffer*>(remoteRmaBufferSlice[i].buf)->GetTokenValue();
        HCCL_INFO(
            "hostBatchPutGetRemoteBufferSliceBufs, addr=0x%llx, size=0x%llx",
            hostBatchPutGetRemoteBufferSliceBufs[i].addr, hostBatchPutGetRemoteBufferSliceBufs[i].size);
    }
    return HCCL_SUCCESS;
}
} // namespace Hccl
