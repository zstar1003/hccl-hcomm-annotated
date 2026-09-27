/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIV_NPU_DIRECT_BASE_H
#define AIV_NPU_DIRECT_BASE_H

#include "kernel_operator.h"

using namespace AscendC;

// AIV直驱Roce所需的rmaInfo信息
// Transport 内存类型
enum class HcclAiRMAMemType : uint32_t {
    LOCAL_INPUT = 0,
    REMOTE_INPUT,

    LOCAL_OUTPUT,
    REMOTE_OUTPUT,

    // 可透传更多的内存，可在MAX_NUM之前追加，例如：
    // LOCAL_EXP,
    // REMOTE_EXP,
    MAX_NUM
};

// Transport 内存信息
struct HcclAiRMAMemInfo {
    uint32_t memMaxNum{0};        // 最大内存数量，等于 HcclAiRMAMemType::MAX_NUM
    uint32_t sizeOfMemDetails{0}; // sizeof(MemDetails)，用于内存校验和偏移计算
    uint64_t memDetailPtr{0};     // MemDetails数组首地址, 个数: HcclAiRMAMemType::MAX_NUM
    // 可往后追加字段
};

// 全部 Transport QP/Mem 信息
struct HcclRMAInfo {
    uint32_t curRankId{0}; // 当前rankId
    uint32_t rankNum{0};   // rank数量
    uint32_t qpNum{0};     // 单个Transport的QP数量

    uint32_t sizeOfRMAWQ{0};  // sizeof(HcclAiRMAWQ)
    uint32_t sizeOfRMACQ{0};  // sizeof(HcclAiRMACQ)
    uint32_t sizeOfRMAMem{0}; // sizeof(HcclAiRMAMemInfo)

    // HcclAiRMAWQ二维数组首地址
    // QP个数: rankNum * qpNum
    // 计算偏移获取SQ指针：sqPtr + (dstRankId * qpNum + qpIndex) * sizeOfRMAWQ
    // 0 <= qpIndex < qpNum
    uint64_t sqPtr{0};

    // HcclAiRMACQ二维数组首地址
    // QP个数: rankNum * qpNum
    // 计算偏移获取SCQ指针：scqPtr + (dstRankId * qpNum + qpIndex) * sizeOfRMACQ
    // 0 <= qpIndex < qpNum
    uint64_t scqPtr{0};

    // HcclAiRMAWQ二维数组首地址
    // QP个数: rankNum * qpNum
    // 计算偏移获取RQ指针：rqPtr + (dstRankId * qpNum + qpIndex) * sizeOfRMAWQ
    // 0 <= qpIndex < qpNum
    uint64_t rqPtr{0};

    // HcclAiRMACQ二维数组首地址
    // QP个数: rankNum * qpNum
    // 计算偏移获取RCQ指针: rcqPtr + (dstRankId * qpNum + qpIndex) * sizeOfRMACQ
    // 0 <= qpIndex < qpNum
    uint64_t rcqPtr{0};

    // HcclAivMemInfo一维数组
    // 内存信息个数: rankNum
    // 计算偏移获取内存信息指针: memPtr + rankId * sizeOfRMAMem
    // srcRankId 获取自身内存信息，dstRankId 获取 Transport 内存信息
    uint64_t memPtr{0};
    // 可往后追加字段
};

enum class DBMode : int32_t { INVALID_DB = -1, HW_DB = 0, SW_DB };

struct HcclAiRMAWQ {
    uint32_t wqn{0};
    uint64_t bufAddr{0};
    uint32_t wqeSize{0};
    uint32_t depth{0};
    uint64_t headAddr{0};
    uint64_t tailAddr{0};
    DBMode dbMode{DBMode::INVALID_DB}; // 0-hw/1-sw
    uint64_t dbAddr{0};
    uint32_t sl{0};
};

struct HcclAiRMACQ {
    uint32_t cqn{0};
    uint64_t bufAddr{0};
    uint32_t cqeSize{0};
    uint32_t depth{0};
    uint64_t headAddr{0};
    uint64_t tailAddr{0};
    DBMode dbMode{DBMode::INVALID_DB}; // 0-hw/1-sw
    uint64_t dbAddr{0};
};

struct hns_roce_rc_sq_wqe {
    uint32_t byte_4;
    uint32_t msg_len;
    uint32_t immtdata;
    uint32_t byte_16;
    uint32_t byte_20;
    uint32_t rkey;
    uint64_t remoteVA;
};

struct hns_roce_lite_wqe_data_seg {
    uint32_t len;
    uint32_t lkey;
    uint64_t localVA;
};

__aicore__ inline void cacheWriteThrough(__gm__ uint8_t* sourceAddr, uint64_t length)
{
    __gm__ uint8_t* start = (__gm__ uint8_t*)((uint64_t)sourceAddr / CACHE_LINE_SIZE * CACHE_LINE_SIZE);
    __gm__ uint8_t* end = (__gm__ uint8_t*)(((uint64_t)sourceAddr + length) / CACHE_LINE_SIZE * CACHE_LINE_SIZE);
    GlobalTensor<uint8_t> global;
    global.SetGlobalBuffer(start);
    for (uint32_t i = 0; i <= end - start; i += CACHE_LINE_SIZE) {
        DataCacheCleanAndInvalid<uint8_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(global[i]);
    }
}

#endif /* AIV_NPU_DIRECT_BASE_H */
