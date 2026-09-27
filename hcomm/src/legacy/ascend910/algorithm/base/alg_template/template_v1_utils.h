/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_TEMPLATE_UTILS_H
#define HCCL_TEMPLATE_UTILS_H

#include <condition_variable>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include "alltoall_utils_pub.h"
#include "device_capacity.h"
#include "ffts_common_pub.h"
#include "hccl_socket.h"
#include "local_notify.h"
#include "stream_pub.h"
#include "transport_pub.h"
#include "comm_utils.h"

struct SendRecvInfo {
    // 存放数据长度和偏移长度
    std::vector<u64> sendLength;
    std::vector<u64> sendOffset;
    std::vector<u64> recvLength;
    std::vector<u64> recvOffset;
    // 存放数据个数和偏移个数
    std::vector<u64> sendCounts;
    std::vector<u64> sendDispls;
    std::vector<u64> recvCounts;
    std::vector<u64> recvDispls;
};

struct ZCopySendRecvInfo {
    // 存放数据长度和偏移长度
    std::vector<u64> localRecvLength;
    std::vector<u64> localRecvOffset;
    std::vector<u64> remoteSendOffset;
};

struct Slice {
    u64 offset{0}; // Slice相对于input/output的偏移字节数，gather类操作取output，scatter类操作取input
    u64 size{0};   // Slice的数据大小，单位：字节
};

struct MemBlockInfo {
    std::vector<u64> size;             // 每块数据块的字节大小
    std::vector<u64> userInputOffsets; // 每个输入块的起始偏移字节数(UserIn)
    std::vector<u64> inputOffsets;     // 每个输入块的起始偏移字节数(CclIn)
    std::vector<u64> outputOffsets;    // 每个输出块的起始偏移字节数
};

namespace hccl {
// common.h
constexpr s64 HCCL_SMALL_COUNT_32_KB = 32 * 1024; // hccl小数据量标准，暂定512KB

struct SubCommInfo {
    u32 localRank;
    u32 localRankSize;
    std::vector<LINK> links;
    std::vector<LINK> virtualLinks; // for alltoall 多线程性能提升使用
};

struct NslbDpAdjInfo {
    uint16_t dstLocalRankId;
    uint8_t phaseId;
    uint8_t rev;
};

// 算法信息表AdjInfo
struct AdjInfo {
    uint16_t dstRankNum;
    uint16_t rev;
    std::vector<NslbDpAdjInfo> nsAdjInfo;
};

// broadcast_nb_binary_pub.h
bool ShouldUseBinaryBroadcastOfNB(
    const u64 dataSize, const u32 rankSize, const u32 userRankSize, const float deviceNumPerAggregation);

// all_reduce_nb_pub.h
u64 GetSliceSizeOfNB(const u64 dataSize, const u32 rankSize);
} // namespace hccl

#endif /* HCCL_TEMPLATE_UTILS_H */
