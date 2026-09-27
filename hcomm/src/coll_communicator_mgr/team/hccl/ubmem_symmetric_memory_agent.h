/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef UBMEM_SYMMETRIC_MEMORY_AGENT_H
#define UBMEM_SYMMETRIC_MEMORY_AGENT_H

#include <cstdint>
#include <string>
#include <vector>

#include "hccl_comm_socket_c_adpt.h"
#include "rank_graph_base.h"

namespace hccl {

constexpr uint32_t UBMEM_PACKET_DATA_MAX_LEN = 144U;
constexpr uint32_t UBMEM_PACKET_TOTAL_LEN = 152U;

enum class UbmemPacketType : uint32_t {
    DATA = 0,
};

// 与A3 Packet语义一致，memberId表示数据在LSA Team输出数组中的位置。
struct UbmemPacket {
    UbmemPacketType type;
    uint32_t memberId;
    uint8_t data[UBMEM_PACKET_DATA_MAX_LEN];
};

class UbMemSymmetricMemoryAgent {
public:
    UbMemSymmetricMemoryAgent(
        RankGraph* rankGraph, uint32_t selfRank, const std::vector<uint32_t>& worldRankIds, uint32_t netLayer,
        const std::string& commId);
    ~UbMemSymmetricMemoryAgent();

    HcclResult CheckNeighborLinks();
    HcclResult Init();
    void Finalize();
    HcclResult ExchangeInfo(void* inputPtr, void* outputPtr, uint64_t inputSize);

    uint32_t GetNetLayer() const { return netLayer_; }

private:
    HcclResult CheckNeighborLinksAvailable();
    HcclResult GetLink(uint32_t peerRank, CommLink& link) const;
    HcclResult CreateNeighborSocket(uint32_t peerRank, const CommLink& link, SocketHandler& socket);
    HcclResult WaitSocketReady(SocketHandler socket) const;
    HcclResult TransferBuffer(
        const uint8_t* sendBuffer, uint8_t* recvBuffer, size_t size, SocketHandler sendSocket,
        SocketHandler recvSocket) const;
    std::string BuildSocketTag(uint32_t peerRank) const;

    RankGraph* rankGraph_{nullptr};
    uint32_t selfRank_{0};
    uint32_t selfMember_{0};
    uint32_t lsaTeamSize_{0};
    std::vector<uint32_t> worldRankIds_;
    std::string commId_;
    uint64_t commHash_{0};
    uint32_t netLayer_{0};
    uint32_t leftRank_{0};
    uint32_t rightRank_{0};
    SocketHandler leftSocket_{nullptr};
    SocketHandler rightSocket_{nullptr};
    bool neighborLinksChecked_{false};
    bool initialized_{false};
};

} // namespace hccl

#endif // UBMEM_SYMMETRIC_MEMORY_AGENT_H
