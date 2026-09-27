/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "symmetric_memory_agent.h"
#include <chrono>

namespace hccl {
using namespace std;

const string STR_IPC_MEM_EXCHANGE = "Exchange_Info";
constexpr u32 USLEEP_ONE_THOUSAND = 1000;
constexpr u32 RING_RANK_SIZE_MIN = 2;

SymmetricMemoryAgent::SymmetricMemoryAgent(
    const std::unique_ptr<HcclSocketManager>& socketManager, u32 devicePhyId, s32 deviceLogicId,
    const HcclIpAddress& localVnicIp, const std::vector<RankInfo>& rankInfoList, u32 userRank, bool useSuperPodMode,
    const std::string& identifier)
    : socketManager_(socketManager),
      devicePhyId_(devicePhyId),
      deviceLogicId_(deviceLogicId),
      localVnicIp_(localVnicIp),
      rankInfoList_(rankInfoList),
      userRank_(userRank),
      rankSize_(rankInfoList.size()),
      useSuperPodMode_(useSuperPodMode),
      identifier_(identifier)
{
    if (rankSize_ >= RING_RANK_SIZE_MIN) { // 当前数据交换算法使用超节点内大平面ring算法，需要和“左右”两边的rank建链
        leftRank_ = (userRank_ - 1 + rankSize_) % rankSize_;
        rightRank_ = (userRank_ + 1) % rankSize_;
    }
}

SymmetricMemoryAgent::~SymmetricMemoryAgent()
{
    threadRun_ = false;
    if (recvThread_ && recvThread_->joinable()) {
        recvThread_->join();
        recvThread_ = nullptr;
    }
    if (vnicPortCtx_ != nullptr) {
        HcclNetCloseDev(vnicPortCtx_);
        vnicPortCtx_ = nullptr;
    }
}

HcclResult SymmetricMemoryAgent::Init()
{
    CHK_PRT_RET(
        rankSize_ < RING_RANK_SIZE_MIN, HCCL_ERROR("[SymmetricMemoryAgent][Init] single rank communicator"),
        HCCL_E_PARA);
    CHK_RET(EstablishSockets());
    CHK_RET(InitRecvThread());
    return HCCL_SUCCESS;
}

HcclResult SymmetricMemoryAgent::InitRecvThread()
{
    threadRun_ = true;
    recvThread_.reset(new (std::nothrow) std::thread(&SymmetricMemoryAgent::DealWithRequest, std::ref(*this)));
    CHK_SMART_PTR_NULL(recvThread_);
    return HCCL_SUCCESS;
}

HcclResult SymmetricMemoryAgent::EstablishSockets()
{
    CHK_PRT_RET((vnicPortCtx_ != nullptr), HCCL_ERROR("[SymmetricMemoryAgent][Init] already initd"), HCCL_E_PARA);
    CHK_RET(HcclNetOpenDev(&vnicPortCtx_, NicType::VNIC_TYPE, devicePhyId_, deviceLogicId_, localVnicIp_));
    CHK_PTR_NULL(vnicPortCtx_);

    HCCL_INFO(
        "[SymmetricMemoryAgent][EstablishSockets] userRank[%u], leftRank_[%u], rightRank_[%u], rankSize_[%u]",
        userRank_, leftRank_, rightRank_, rankSize_);
    for (size_t i = 0; i < rankInfoList_.size(); i++) {
        if (rankInfoList_[i].userRank == leftRank_ || rankInfoList_[i].userRank == rightRank_) {
            HcclRankLinkInfo remoteLinkInfo;
            RankInfo dstRankInfo = rankInfoList_[i];
            remoteLinkInfo.userRank = dstRankInfo.userRank;
            remoteLinkInfo.devicePhyId = dstRankInfo.devicePhyId;
            remoteLinkInfo.ip = HcclIpAddress(dstRankInfo.devicePhyId);
            if (useSuperPodMode_) {
                CHK_RET(hrtRaGetSingleSocketVnicIpInfo(
                    devicePhyId_, DeviceIdType::DEVICE_ID_TYPE_SDID, dstRankInfo.superDeviceId, remoteLinkInfo.ip));
            } else {
                CHK_RET(hrtRaGetSingleSocketVnicIpInfo(
                    devicePhyId_, DeviceIdType::DEVICE_ID_TYPE_PHY_ID, dstRankInfo.devicePhyId, remoteLinkInfo.ip));
            }
            // 通信域未分配端口则使用默认端口
            remoteLinkInfo.port
                = dstRankInfo.deviceVnicPort == HCCL_INVALID_PORT ? HETEROG_CCL_PORT : dstRankInfo.deviceVnicPort;
            remoteLinkInfo.socketsPerLink = 1;
            string newTag = GenerateSocketTag(devicePhyId_, rankInfoList_[i].devicePhyId);
            std::vector<std::shared_ptr<HcclSocket>> tmpSockets;
            HcclResult ret
                = socketManager_->CreateSingleLinkSocket(newTag, vnicPortCtx_, remoteLinkInfo, tmpSockets, false, true);
            CHK_PRT_RET(
                ret != HCCL_SUCCESS,
                HCCL_ERROR(
                    "[Create][DestSockets]Create single link sockets failed, "
                    "local rank[%u], remote rank[%u]",
                    userRank_, rankInfoList_[i].userRank),
                ret);
            if (tmpSockets.size() != 1) {
                HCCL_ERROR(
                    "[SymmetricMemoryAgent][CreateVnic] socket number[%llu] is not 1 as expected!", tmpSockets.size());
                return HCCL_E_INTERNAL;
            }
            // 设置强制断链为关闭，避免进程退出时recv失败
            tmpSockets[0]->SetForceClose(false);
            mapRankIdconnectedSockets_[remoteLinkInfo.userRank] = (tmpSockets[0]);
            mapRankId2DevPhyId_[remoteLinkInfo.userRank] = remoteLinkInfo.devicePhyId;
        }
    }

    for (const auto& kv : mapRankIdconnectedSockets_) {
        CHK_PRT_RET(
            socketManager_->WaitLinkEstablish(kv.second) != HCCL_SUCCESS,
            HCCL_ERROR(
                "[SymmetricMemoryAgent][EstablishSockets] tag[%s] socket establish failed",
                kv.second->GetTag().c_str()),
            HCCL_E_INTERNAL);
    }
    return HCCL_SUCCESS;
}

std::string SymmetricMemoryAgent::GenerateSocketTag(u32 localRank, u32 remoteRank)
{
    u32 small = localRank;
    u32 large = remoteRank;

    if (localRank > remoteRank) {
        small = remoteRank;
        large = localRank;
    }

    // Socket构造规则：前缀 + identifier + small + large
    std::string tag
        = STR_IPC_MEM_EXCHANGE + "_" + identifier_ + "_" + std::to_string(small) + ":" + std::to_string(large);
    return tag;
}

HcclResult SymmetricMemoryAgent::ExchangeInfo(void* inputPtr, void* outputPtr, u64 inputSize)
{
    CHK_PTR_NULL(inputPtr);
    CHK_PTR_NULL(outputPtr);
    CHK_PRT_RET(inputSize == 0, HCCL_ERROR("Input size is 0"), HCCL_E_PARA);
    // 校验 inputSize 是否超过协议载荷上限
    CHK_PRT_RET(
        inputSize > PACKET_DATA_MAX_LEN,
        HCCL_ERROR("Input size %lu exceeds max payload %u", inputSize, PACKET_DATA_MAX_LEN), HCCL_E_PARA);
    // 校验是否建链成功
    CHK_PRT_RET(
        mapRankIdconnectedSockets_.find(rightRank_) == mapRankIdconnectedSockets_.end(),
        HCCL_ERROR("[ExchangeInfo] rightRank_%u socket not found in map", rightRank_), HCCL_E_INTERNAL);
    CHK_PRT_RET(
        mapRankIdconnectedSockets_.find(leftRank_) == mapRankIdconnectedSockets_.end(),
        HCCL_ERROR("[ExchangeInfo] leftRank_%u socket not found in map", leftRank_), HCCL_E_INTERNAL);

    HCCL_INFO(
        "[SymmetricMemoryAgent] start to ExchangeInfo, inputPtr[%p], outputPtr[%p], inputSize[%llu]", inputPtr,
        outputPtr, inputSize);

    // 重置本轮状态
    outputDataPtr_ = static_cast<u8*>(outputPtr);
    currentInputSize_ = inputSize; // 记录实际有效长度
    collectedCount_ = 0;
    // 本地数据处理：先把自己的一份拷到 Output 对应位置
    u8* selfDstPtr = outputDataPtr_ + (userRank_ * inputSize);
    CHK_SAFETY_FUNC_RET(memcpy_s(selfDstPtr, inputSize, inputPtr, inputSize));
    collectedCount_++;

    Packet dataPkt;
    dataPkt.type = MsgType::MSG_TYPE_DATA;
    dataPkt.rankId = userRank_;
    CHK_SAFETY_FUNC_RET(memset_s(dataPkt.data, PACKET_DATA_MAX_LEN, 0, PACKET_DATA_MAX_LEN));
    CHK_SAFETY_FUNC_RET(memcpy_s(dataPkt.data, PACKET_DATA_MAX_LEN, inputPtr, inputSize));
    {
        std::lock_guard<std::mutex> lock(queueMutex_);
        requestQueue_.push(dataPkt);
    }
    isProcessingTask_ = true;

    CHK_RET(WaitForCollectionComplete());
    HCCL_INFO("[SymmetricMemoryAgent] ExchangeInfo end");
    return HCCL_SUCCESS;
}

HcclResult SymmetricMemoryAgent::WaitForCollectionComplete()
{
    std::unique_lock<std::mutex> lock(completionMutex_);
    auto timeout = std::chrono::seconds(GetExternalInputHcclLinkTimeOut());
    auto status = completionCv_.wait_for(lock, timeout);
    if (status == std::cv_status::timeout) {
        HCCL_ERROR("[SymmetricMemoryAgent] ExchangeInfo Timeout! Collected: %u/%u", collectedCount_.load(), rankSize_);
        return HCCL_E_TCP_TRANSFER;
    }
    return HCCL_SUCCESS;
}

void SymmetricMemoryAgent::DealWithRequest()
{
    if (hrtSetDevice(deviceLogicId_) != HCCL_SUCCESS) {
        return;
    }

    std::vector<u8> leftRecvBuf(PACKET_TOTAL_LEN, 0);
    u32 leftRecvLen = 0;

    while (threadRun_) {
        if (isProcessingTask_) {
            if (collectedCount_ < rankSize_) {
                u64 received = 0;
                std::unique_lock<std::mutex> lock(socketMutex_);
                HcclResult ret = mapRankIdconnectedSockets_[leftRank_]->IRecv(
                    leftRecvBuf.data() + leftRecvLen, PACKET_TOTAL_LEN - leftRecvLen, received);

                CHK_PRT_CONT(
                    (ret != HCCL_SUCCESS) && (ret != HCCL_E_AGAIN),
                    HCCL_ERROR(
                        "[SymmetricMemoryAgent][DealWithRequest] IRecv failed, ret[%d] remoteRank[%u] "
                        "receivedSize[%llu]",
                        ret, leftRank_, leftRecvLen));

                leftRecvLen += received;
                if (leftRecvLen == PACKET_TOTAL_LEN) {
                    Packet* pkt = reinterpret_cast<Packet*>(leftRecvBuf.data());
                    ProcessReceivedPacket(*pkt);
                    leftRecvLen = 0;
                }
            }
            std::lock_guard<std::mutex> lock(queueMutex_);
            if (!requestQueue_.empty()) {
                Packet pkt = requestQueue_.front();
                std::unique_lock<std::mutex> sockLock(socketMutex_);
                HcclResult ret
                    = mapRankIdconnectedSockets_[rightRank_]->Send(static_cast<void*>(&pkt), PACKET_TOTAL_LEN);
                if (ret == HCCL_SUCCESS) {
                    requestQueue_.pop();
                } else {
                    HCCL_ERROR(
                        "[SymmetricMemoryAgent][DealWithRequest] Data(from rank[%u]) Send to rank[%u] failed.",
                        pkt.rankId, rightRank_);
                }
            }
            // 检查是否完全结束, 退出条件: 数据全齐 && 队列空闲
            if (requestQueue_.empty() && collectedCount_ == rankSize_) {
                std::unique_lock<std::mutex> lock(completionMutex_);
                HCCL_INFO("[SymmetricMemoryAgent] ExchangeInfo Complete.");
                isProcessingTask_ = false;
                completionCv_.notify_all();
            }
        }
        SaluSleep(USLEEP_ONE_THOUSAND);
    }

    hrtResetDevice(deviceLogicId_);
}

HcclResult SymmetricMemoryAgent::ProcessReceivedPacket(Packet& pkt)
{
    if (pkt.rankId < rankSize_ && pkt.rankId != userRank_) {
        u8* dest = outputDataPtr_ + (pkt.rankId * currentInputSize_);
        CHK_SAFETY_FUNC_RET(memcpy_s(dest, currentInputSize_, pkt.data, currentInputSize_));
        collectedCount_++;
    }
    HCCL_INFO(
        "[SymmetricMemoryAgent][ProcessReceivedPacket] Data Recv from rank[%u]. Collected[%u / %u].", pkt.rankId,
        collectedCount_.load(), rankSize_);
    // Ring 转发逻辑：如果数据不是自己的，也不是右边Rank发出的(转了一圈)，则转发给右边
    if (pkt.rankId != userRank_ && pkt.rankId != rightRank_) {
        std::lock_guard<std::mutex> lock(queueMutex_);
        requestQueue_.push(pkt);
    }
    return HCCL_SUCCESS;
}
} // namespace hccl
