/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SYMMETRIC_MEMORY_AGENT_H
#define SYMMETRIC_MEMORY_AGENT_H

#include <vector>
#include <map>
#include <mutex>
#include <thread>
#include <atomic>
#include <queue>
#include <condition_variable>
#include <string>
#include <memory>

#include "hccl/hccl_types.h"
#include "hccl/base.h"
#include "hccl_socket_manager.h"
#include "adapter_hccp_common.h"

namespace hccl {

// 协议常量
constexpr u32 PACKET_DATA_MAX_LEN = 144;
constexpr u32 PACKET_TOTAL_LEN = 152; // 4(Type) + 4(Rank) + 144(Data)

// 消息类型
enum class MsgType : u32 {
    MSG_TYPE_DATA = 0,
};

// 协议包结构
struct Packet {
    MsgType type;
    u32 rankId;
    u8 data[PACKET_DATA_MAX_LEN];
};

class SymmetricMemoryAgent {
public:
    SymmetricMemoryAgent(
        const std::unique_ptr<HcclSocketManager>& socketManager, u32 devicePhyId, s32 deviceLogicId,
        const HcclIpAddress& localVnicIp, const std::vector<RankInfo>& rankInfoList, u32 userRank, bool useSuperPodMode,
        const std::string& identifier);
    virtual ~SymmetricMemoryAgent();

    HcclResult Init();
    HcclResult ExchangeInfo(void* inputPtr, void* outputPtr, u64 inputSize);

private:
    HcclResult InitRecvThread();
    HcclResult EstablishSockets();
    std::string GenerateSocketTag(u32 localRank, u32 remoteRank);

    void DealWithRequest();
    HcclResult WaitForCollectionComplete();
    HcclResult ProcessReceivedPacket(Packet& pkt);

    HcclNetDevCtx vnicPortCtx_{nullptr};
    const std::unique_ptr<HcclSocketManager>& socketManager_;
    u32 devicePhyId_;
    s32 deviceLogicId_;
    HcclIpAddress localVnicIp_;
    const std::vector<RankInfo>& rankInfoList_;
    u32 userRank_;
    u32 leftRank_{0};
    u32 rightRank_{0};
    u32 rankSize_;
    bool useSuperPodMode_;
    std::string identifier_{};

    std::unique_ptr<std::thread> recvThread_;
    std::atomic<bool> threadRun_{false};

    std::mutex socketMutex_;
    std::unordered_map<u32, std::shared_ptr<HcclSocket>> mapRankIdconnectedSockets_;
    std::unordered_map<u32, u32> mapRankId2DevPhyId_;

    u8* outputDataPtr_{nullptr};
    u64 currentInputSize_{0}; // 记录当前交换数据的实际有效长度
    std::atomic<u32> collectedCount_{0};

    std::queue<Packet> requestQueue_;
    std::mutex queueMutex_;

    // 完成通知 (用于 WaitForCollectionComplete)
    std::mutex completionMutex_;
    std::condition_variable completionCv_;

    std::atomic<bool> isProcessingTask_{false};
};

} // namespace hccl
#endif // SYMMETRIC_MEMORY_AGENT_H
