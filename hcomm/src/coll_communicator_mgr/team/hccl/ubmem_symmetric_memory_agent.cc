/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ubmem_symmetric_memory_agent.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <thread>

#include "env_config/env_config_v2.h"
#include "hccl_common.h"
#include "log.h"

namespace hccl {
namespace {
    constexpr const char* UB_MEM_SYMMETRIC_SOCKET_TAG_PREFIX = "ub_mem_sym";

    uint64_t HashCommId(const std::string& commId)
    {
        constexpr uint64_t fnvOffset = 1469598103934665603ULL;
        constexpr uint64_t fnvPrime = 1099511628211ULL;
        uint64_t hash = fnvOffset;
        for (unsigned char value : commId) {
            hash ^= value;
            hash *= fnvPrime;
        }
        return hash;
    }
} // namespace

static_assert(sizeof(UbmemPacket) == UBMEM_PACKET_TOTAL_LEN, "UB Memory exchange packet layout is invalid");

UbMemSymmetricMemoryAgent::UbMemSymmetricMemoryAgent(
    RankGraph* rankGraph, uint32_t selfRank, const std::vector<uint32_t>& worldRankIds, uint32_t netLayer,
    const std::string& commId)
    : rankGraph_(rankGraph),
      selfRank_(selfRank),
      lsaTeamSize_(static_cast<uint32_t>(worldRankIds.size())),
      worldRankIds_(worldRankIds),
      commId_(commId),
      commHash_(HashCommId(commId)),
      netLayer_(netLayer)
{
    auto selfIter = std::find(worldRankIds_.begin(), worldRankIds_.end(), selfRank_);
    if (selfIter != worldRankIds_.end()) {
        selfMember_ = static_cast<uint32_t>(selfIter - worldRankIds_.begin());
        leftRank_ = worldRankIds_[(selfMember_ + lsaTeamSize_ - 1U) % lsaTeamSize_];
        rightRank_ = worldRankIds_[(selfMember_ + 1U) % lsaTeamSize_];
    }
}

UbMemSymmetricMemoryAgent::~UbMemSymmetricMemoryAgent() { Finalize(); }

HcclResult UbMemSymmetricMemoryAgent::GetLink(uint32_t peerRank, CommLink& link) const
{
    CommLink* links = nullptr;
    uint32_t linkNum = 0;
    CHK_RET(rankGraph_->GetLinks(netLayer_, selfRank_, peerRank, &links, &linkNum));
    CHK_PRT_RET(
        links == nullptr || linkNum == 0,
        HCCL_ERROR("[%s] no link, layer[%u], selfRank[%u], peerRank[%u]", __func__, netLayer_, selfRank_, peerRank),
        HCCL_E_NOT_FOUND);
    for (uint32_t index = 0; index < linkNum; ++index) {
        if (links[index].linkAttr.linkProtocol == COMM_PROTOCOL_UB_MEM
            && links[index].srcEndpointDesc.loc.locType != ENDPOINT_LOC_TYPE_HOST
            && links[index].dstEndpointDesc.loc.locType != ENDPOINT_LOC_TYPE_HOST) {
            link = links[index];
            return HCCL_SUCCESS;
        }
    }
    HCCL_ERROR(
        "[%s] no UB Memory device link, layer[%u], selfRank[%u], peerRank[%u]", __func__, netLayer_, selfRank_,
        peerRank);
    return HCCL_E_NOT_FOUND;
}

HcclResult UbMemSymmetricMemoryAgent::CheckNeighborLinksAvailable()
{
    if (lsaTeamSize_ <= 1U) {
        return HCCL_SUCCESS;
    }
    CommLink leftLink{};
    CommLink rightLink{};
    CHK_RET(GetLink(leftRank_, leftLink));
    if (rightRank_ != leftRank_) {
        CHK_RET(GetLink(rightRank_, rightLink));
    }
    return HCCL_SUCCESS;
}

HcclResult UbMemSymmetricMemoryAgent::CheckNeighborLinks()
{
    if (neighborLinksChecked_) {
        return HCCL_SUCCESS;
    }
    CHK_PTR_NULL(rankGraph_);
    CHK_PRT_RET(
        lsaTeamSize_ == 0 || selfMember_ >= lsaTeamSize_ || worldRankIds_[selfMember_] != selfRank_,
        HCCL_ERROR("[%s] self rank[%u] is not in UB Memory LSA team", __func__, selfRank_), HCCL_E_PARA);
    CHK_RET(CheckNeighborLinksAvailable());
    neighborLinksChecked_ = true;
    return HCCL_SUCCESS;
}

std::string UbMemSymmetricMemoryAgent::BuildSocketTag(uint32_t peerRank) const
{
    uint32_t lowRank = std::min(selfRank_, peerRank);
    uint32_t highRank = std::max(selfRank_, peerRank);
    return std::string(UB_MEM_SYMMETRIC_SOCKET_TAG_PREFIX) + "_" + std::to_string(commHash_) + "_"
           + std::to_string(netLayer_) + "_" + std::to_string(lowRank) + "_" + std::to_string(highRank);
}

HcclResult
UbMemSymmetricMemoryAgent::CreateNeighborSocket(uint32_t peerRank, const CommLink& link, SocketHandler& socket)
{
    SocketDesc desc{};
    desc.localEndpoint = link.srcEndpointDesc;
    desc.remoteEndpoint = link.dstEndpointDesc;
    desc.role = selfRank_ < peerRank ? HCOMM_SOCKET_ROLE_SERVER : HCOMM_SOCKET_ROLE_CLIENT;

    uint32_t port = 0;
    uint32_t portRank = desc.role == HCOMM_SOCKET_ROLE_SERVER ? selfRank_ : peerRank;
    CHK_RET(rankGraph_->GetDevicePort(portRank, &port));
    CHK_PRT_RET(port > UINT16_MAX, HCCL_ERROR("[%s] invalid port[%u]", __func__, port), HCCL_E_PARA);
    desc.listenPort = static_cast<uint16_t>(port);

    std::string tag;
    EXCEPTION_CATCH(tag = BuildSocketTag(peerRank), return HCCL_E_MEMORY);
    CHK_PRT_RET(tag.size() >= sizeof(desc.tag), HCCL_ERROR("[%s] socket tag is too long", __func__), HCCL_E_PARA);
    CHK_PRT_RET(
        memcpy_s(desc.tag, sizeof(desc.tag), tag.c_str(), tag.size() + 1U) != EOK,
        HCCL_ERROR("[%s] copy socket tag failed", __func__), HCCL_E_MEMORY);
    return SocketCreate(&desc, &socket);
}

HcclResult UbMemSymmetricMemoryAgent::WaitSocketReady(SocketHandler socket) const
{
    auto connectTimeout = std::chrono::seconds(Hccl::EnvConfig::GetInstance().GetSocketConfig().GetLinkTimeOut());
    auto deadline = std::chrono::steady_clock::now() + connectTimeout;
    while (std::chrono::steady_clock::now() < deadline) {
        SocketStates status = SOCKET_CONNECTING;
        CHK_RET(SocketGetStatus(socket, &status));
        if (status == SOCKET_OK) {
            return HCCL_SUCCESS;
        }
        CHK_PRT_RET(status == SOCKET_TIMEOUT, HCCL_ERROR("[%s] socket connect timeout", __func__), HCCL_E_TIMEOUT);
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    HCCL_ERROR("[%s] wait socket ready timeout", __func__);
    return HCCL_E_TIMEOUT;
}

HcclResult UbMemSymmetricMemoryAgent::Init()
{
    CHK_RET(CheckNeighborLinks());
    if (initialized_ || lsaTeamSize_ <= 1U) {
        initialized_ = true;
        return HCCL_SUCCESS;
    }

    CommLink leftLink{};
    CHK_RET(GetLink(leftRank_, leftLink));
    HcclResult ret = CreateNeighborSocket(leftRank_, leftLink, leftSocket_);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR(
            "[%s] create left neighbor socket failed, selfRank[%u], peerRank[%u], netLayer[%u], ret[%d]", __func__,
            selfRank_, leftRank_, netLayer_, ret);
        Finalize();
        return ret;
    }
    if (rightRank_ == leftRank_) {
        rightSocket_ = leftSocket_;
    } else {
        CommLink rightLink{};
        ret = GetLink(rightRank_, rightLink);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR(
                "[%s] get right neighbor UB Memory link failed, selfRank[%u], peerRank[%u], netLayer[%u], ret[%d]",
                __func__, selfRank_, rightRank_, netLayer_, ret);
            Finalize();
            return ret;
        }
        ret = CreateNeighborSocket(rightRank_, rightLink, rightSocket_);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR(
                "[%s] create right neighbor socket failed, selfRank[%u], peerRank[%u], netLayer[%u], ret[%d]", __func__,
                selfRank_, rightRank_, netLayer_, ret);
            Finalize();
            return ret;
        }
    }
    ret = WaitSocketReady(leftSocket_);
    if (ret == HCCL_SUCCESS && rightSocket_ != leftSocket_) {
        ret = WaitSocketReady(rightSocket_);
    }
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR(
            "[%s] wait UB Memory LSA ring socket ready failed, selfRank[%u], leftRank[%u], rightRank[%u], "
            "netLayer[%u], ret[%d]",
            __func__, selfRank_, leftRank_, rightRank_, netLayer_, ret);
        Finalize();
        return ret;
    }
    initialized_ = true;
    HCCL_RUN_INFO(
        "[%s] UB Memory LSA ring ready, comm[%s], rank[%u], left[%u], right[%u], layer[%u]", __func__, commId_.c_str(),
        selfRank_, leftRank_, rightRank_, netLayer_);
    return HCCL_SUCCESS;
}

void UbMemSymmetricMemoryAgent::Finalize()
{
    if (rightSocket_ != nullptr && rightSocket_ != leftSocket_) {
        CHK_PRT(SocketDestroy(rightSocket_));
    }
    if (leftSocket_ != nullptr) {
        CHK_PRT(SocketDestroy(leftSocket_));
    }
    rightSocket_ = nullptr;
    leftSocket_ = nullptr;
    initialized_ = false;
}

HcclResult UbMemSymmetricMemoryAgent::TransferBuffer(
    const uint8_t* sendBuffer, uint8_t* recvBuffer, size_t size, SocketHandler sendSocket,
    SocketHandler recvSocket) const
{
    uint64_t sentSize = 0;
    uint64_t receivedSize = 0;
    auto transferTimeout = std::chrono::seconds(Hccl::EnvConfig::GetInstance().GetSocketConfig().GetLinkTimeOut());
    auto deadline = std::chrono::steady_clock::now() + transferTimeout;
    while ((sentSize < size || receivedSize < size) && std::chrono::steady_clock::now() < deadline) {
        bool progressed = false;
        if (sentSize < size) {
            uint64_t completedSize = 0;
            CHK_RET(
                SocketSendNb(sendSocket, const_cast<uint8_t*>(sendBuffer) + sentSize, size - sentSize, &completedSize));
            CHK_PRT_RET(
                completedSize > size - sentSize,
                HCCL_ERROR(
                    "[%s] invalid socket send completion[%llu]", __func__,
                    static_cast<unsigned long long>(completedSize)),
                HCCL_E_INTERNAL);
            sentSize += completedSize;
            progressed = progressed || completedSize > 0;
        }
        if (receivedSize < size) {
            uint64_t completedSize = 0;
            CHK_RET(SocketRecvNb(recvSocket, recvBuffer + receivedSize, size - receivedSize, &completedSize));
            CHK_PRT_RET(
                completedSize > size - receivedSize,
                HCCL_ERROR(
                    "[%s] invalid socket receive completion[%llu]", __func__,
                    static_cast<unsigned long long>(completedSize)),
                HCCL_E_INTERNAL);
            receivedSize += completedSize;
            progressed = progressed || completedSize > 0;
        }
        if (!progressed) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
    CHK_PRT_RET(
        sentSize != size || receivedSize != size,
        HCCL_ERROR(
            "[%s] ring transfer timeout, sent[%llu/%zu], received[%llu/%zu]", __func__,
            static_cast<unsigned long long>(sentSize), size, static_cast<unsigned long long>(receivedSize), size),
        HCCL_E_TIMEOUT);
    return HCCL_SUCCESS;
}

HcclResult UbMemSymmetricMemoryAgent::ExchangeInfo(void* inputPtr, void* outputPtr, uint64_t inputSize)
{
    CHK_PTR_NULL(inputPtr);
    CHK_PTR_NULL(outputPtr);
    CHK_PRT_RET(!initialized_, HCCL_ERROR("[%s] UB Memory LSA ring is not initialized", __func__), HCCL_E_UNAVAIL);
    CHK_PRT_RET(inputSize == 0, HCCL_ERROR("[%s] input size is zero", __func__), HCCL_E_PARA);
    CHK_PRT_RET(
        inputSize > UBMEM_PACKET_DATA_MAX_LEN,
        HCCL_ERROR(
            "[%s] input size[%llu] exceeds maximum[%u]", __func__, static_cast<unsigned long long>(inputSize),
            UBMEM_PACKET_DATA_MAX_LEN),
        HCCL_E_PARA);

    auto* output = static_cast<uint8_t*>(outputPtr);
    CHK_SAFETY_FUNC_RET(memcpy_s(output + selfMember_ * inputSize, inputSize, inputPtr, inputSize));
    if (lsaTeamSize_ <= 1U) {
        return HCCL_SUCCESS;
    }

    UbmemPacket sendPacket{};
    sendPacket.type = UbmemPacketType::DATA;
    sendPacket.memberId = selfMember_;
    CHK_SAFETY_FUNC_RET(memcpy_s(sendPacket.data, sizeof(sendPacket.data), inputPtr, inputSize));
    // 对齐A3的Ring AllGather语义；A5 UB_MEM在当前线程执行固定轮次交换，以适配SocketSendNb/SocketRecvNb接口。
    // 每轮把左邻居收到的数据继续发往右邻居，完成后输出数组包含全部LSA成员的信息。
    for (uint32_t round = 0; round < lsaTeamSize_ - 1U; ++round) {
        UbmemPacket recvPacket{};
        CHK_RET(TransferBuffer(
            reinterpret_cast<const uint8_t*>(&sendPacket), reinterpret_cast<uint8_t*>(&recvPacket), sizeof(UbmemPacket),
            rightSocket_, leftSocket_));
        CHK_PRT_RET(
            recvPacket.type != UbmemPacketType::DATA || recvPacket.memberId >= lsaTeamSize_,
            HCCL_ERROR(
                "[%s] invalid packet, type[%u], memberId[%u]", __func__, static_cast<uint32_t>(recvPacket.type),
                recvPacket.memberId),
            HCCL_E_PARA);
        CHK_SAFETY_FUNC_RET(memcpy_s(output + recvPacket.memberId * inputSize, inputSize, recvPacket.data, inputSize));
        sendPacket = recvPacket;
    }
    HCCL_INFO("[%s] exchanged information for[%u] LSA members", __func__, lsaTeamSize_);
    return HCCL_SUCCESS;
}

} // namespace hccl
