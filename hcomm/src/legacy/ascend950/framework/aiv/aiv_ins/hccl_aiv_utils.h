/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_AIV_UTILS_H
#define HCCL_AIV_UTILS_H

#include "string"

#include "hccl_types.h"
#include "orion_adapter_rts.h"
#include "template_utils.h"
#include "acl/acl_rt.h"

namespace Hccl {
constexpr u32 MAX_RANK_SIZE_ = 64; // 注意要和device侧的一致
constexpr u32 MAX_NUM_BLOCKS = 56; // 56-72

constexpr s32 TAG_INIT_VALUE = 1;
constexpr s32 TAG_RESET_COUNT = 1000;
constexpr s32 TOPO_LEN = 32;

constexpr u32 AIV_TAG_MOVE_LEFT_BITS = 16;
constexpr u32 AIV_TAG_ADDR_OFFSET = 16 * 1024;
constexpr u32 AIV_FLAG_ADDR_OFFSET = 40 * 1024;
constexpr u32 AIV_FLAG_AREA_SIZE = 1000 * 1024;
constexpr u32 AIV_FLAG_CLEAR_OFFSET = 1040 * 1024;
constexpr u32 AIV_LOW_16_BITS = 0xFFFF;

constexpr u32 CACHEMAP_MAXSIZE = 65536;
constexpr float CACHEMAP_CLEARPERCENT = 0.1;

constexpr u32 BATCH_SEND_RECV_ITEM_SIZE = 16; // 注意要和device侧的BATCH_SEND_RECV_ITEM_SIZE保持一致

enum class KernelArgsType {
    ARGS_TYPE_SERVER = 0, // kernel参数为单机内
    ARGS_TYPE_TWO_SHOT = 1,
    ARGS_TYPE_DEFAULT
};

struct all2AllDataDes {
    DataType sendType;
    DataType recvType;
    u64 sendCount;
    u64 recvCount;
    // 重载 != 操作符
    bool operator!=(const all2AllDataDes& other) const
    {
        return sendType != other.sendType || recvType != other.recvType || sendCount != other.sendCount
               || recvCount != other.recvCount;
    }

    // 重载 < 操作符
    bool operator<(const all2AllDataDes& other) const
    {
        if (sendType != other.sendType)
            return sendType < other.sendType;
        if (recvType != other.recvType)
            return recvType < other.recvType;
        if (sendCount != other.sendCount)
            return sendCount < other.sendCount;
        return recvCount < other.recvCount;
    }
};

struct all2AllVDataDes {
    DataType sendType;
    DataType recvType;
    void* sendCounts;
    void* recvCounts;
    void* sdispls;
    void* rdispls;
    // 重载 != 操作符
    bool operator!=(const all2AllVDataDes& other) const
    {
        return sendType != other.sendType || recvType != other.recvType || sendCounts != other.sendCounts
               || recvCounts != other.recvCounts || sdispls != other.sdispls || rdispls != other.rdispls;
    }

    // 重载 < 操作符
    bool operator<(const all2AllVDataDes& other) const
    {
        if (sendType != other.sendType)
            return sendType < other.sendType;
        if (recvType != other.recvType)
            return recvType < other.recvType;
        if (sendCounts != other.sendCounts)
            return sendCounts < other.sendCounts;
        if (recvCounts != other.recvCounts)
            return recvCounts < other.recvCounts;
        if (sdispls != other.sdispls)
            return sdispls < other.sdispls;
        return rdispls < other.rdispls;
    }
};

// 表示算子属性的参数，相对固定
struct AivOpCacheArgs {
    std::string algName = "default";
    u64 count = 0;
    DataType dataType = DataType::INT32;
    uint16_t opType = 0;
    ReduceOp reduceOp = ReduceOp::INVALID;
    u32 root = 0;
    u32 numBlocksLimit = 0;
    DataType outputDataType = DataType::INT32;
    all2AllDataDes all2allDataDes{};
    all2AllVDataDes all2allVDataDes{};

    AivOpCacheArgs(
        const std::string& algName_, u64 count_, DataType dataType_, uint16_t opType_, ReduceOp reduceOp_, u32 root_,
        u32 numBlocksLimit_, DataType outputDataType_, const all2AllDataDes& all2allDataDes_ = {},
        const all2AllVDataDes& all2allVDataDes_ = {})
        : algName(algName_),
          count(count_),
          dataType(dataType_),
          opType(opType_),
          reduceOp(reduceOp_),
          root(root_),
          numBlocksLimit(numBlocksLimit_),
          outputDataType(outputDataType_),
          all2allDataDes(all2allDataDes_),
          all2allVDataDes(all2allVDataDes_)
    {}
    // 自定义 operator<
    bool operator<(const AivOpCacheArgs& other) const
    {
        if (algName != other.algName)
            return algName < other.algName;
        if (count != other.count)
            return count < other.count;
        if (dataType != other.dataType)
            return dataType < other.dataType;
        if (opType != other.opType)
            return opType < other.opType;
        if (reduceOp != other.reduceOp)
            return reduceOp < other.reduceOp;
        if (root != other.root)
            return root < other.root;
        if (numBlocksLimit != other.numBlocksLimit)
            return numBlocksLimit < other.numBlocksLimit;
        if (all2allDataDes != other.all2allDataDes)
            return all2allDataDes < other.all2allDataDes;
        if (all2allVDataDes != other.all2allVDataDes)
            return all2allVDataDes < other.all2allVDataDes;
        return outputDataType < other.outputDataType;
    }
};

struct HcclSendRecvItemHost {
    uint32_t sendRecvType;
    uint64_t bufAddr;
    uint64_t count;
    uint32_t dataTypeSize;
    uint32_t remoteRank;
};

// 非均匀算子AlltoAllV/AlltoAllVC/AllGatherV/ReduceScatterV需要的额外参数信息，A3场景
struct ExtraArgsA2A { // 后面考虑把这个参数名换一下，或者直接独立弄个参数出来
    u64 sendCounts[MAX_RANK_SIZE_] = {};
    u64 sendDispls[MAX_RANK_SIZE_] = {};
    u64 recvCounts[MAX_RANK_SIZE_] = {};
    u64 recvDispls[MAX_RANK_SIZE_] = {};
    uint64_t itemNum = 0;
    HcclSendRecvItemHost sendRecvInfo[BATCH_SEND_RECV_ITEM_SIZE] = {};
};

// 算子计数信息
struct OpCounterInfo {
    u64 headCountMem = 0;
    u64 tailCountMem = 0;
    u64 addOneMem = 0;
    u32 memSize = 0;
    bool isEnableCounter = false;
};

// 表示算子属性的参数，相对固定
struct AivOpArgs {
    HcclCMDType cmdType = HcclCMDType::HCCL_CMD_MAX;
    std::string comm = {};
    u32 numBlocks = MAX_NUM_BLOCKS;
    rtStream_t stream = nullptr;
    uint64_t beginTime = 0;
    OpCounterInfo counter = {};
    const void* buffersIn = nullptr;
    u64 input = 0;
    u64 output = 0;
    u32 rank = 0;
    u32 sendRecvRemoteRank = 0;
    u32 rankSize = 0;
    u64 xRankSize = 0;
    u64 yRankSize = 0;
    u64 zRankSize = 0;
    u64 count = 0;
    DataType dataType = DataType::INT32;
    ReduceOp op = ReduceOp::SUM;
    u32 root = 0;
    u32 aivTag = 0;
    u64 inputSliceStride = 0;
    u64 outputSliceStride = 0;
    u64 repeatNum = 0;
    u64 inputRepeatStride = 0;
    u64 outputRepeatStride = 0;
    bool isOpBase = false;
    ExtraArgsA2A extraArgs = {};
    uint64_t topo_[TOPO_LEN] = {0};
    AivOpArgs() {};
    KernelArgsType argsType = KernelArgsType::ARGS_TYPE_SERVER;
};

using AivSuperKernelArgs = struct AivSuperKernelArgsDef {
    const void* buffersIn = nullptr; // 注册的CCLIN地址，所有卡可访问
    u64 rank{};
    u64 rankSize{};
    u64 len{};
    u64 dataType{};
    u64 unitSize{};
    u64 reduceOp{};
    u64 numBlocks{};
    s64 tag{}; // 第几次调用，定时重置成1
    s64 clearEnable{};
    uint64_t inputSliceStride{};
    uint64_t outputSliceStride{};
    uint64_t repeatNum{};
    uint64_t inputRepeatStride{};
    uint64_t outputRepeatStride{};
    u64 input{};
    u64 output{};
    u64 cclBufferSize{};
    AivSuperKernelArgsDef(
        u64 input, u64 output, u32 rank, u32 rankSize, u64 len, u32 dataType, u64 unitSize, u32 reduceOp,
        u32 numBlocks = 0, s32 tag = 0, bool clearEnable = true, uint64_t inputSliceStride = 0,
        uint64_t outputSliceStride = 0, uint64_t repeatNum = 0, uint64_t inputRepeatStride = 0,
        uint64_t outputRepeatStride = 0, u64 cclBufferSize = 0)
        : rank(rank),
          rankSize(rankSize),
          len(len),
          dataType(dataType),
          unitSize(unitSize),
          reduceOp(reduceOp),
          numBlocks(numBlocks),
          tag(tag),
          clearEnable(clearEnable),
          inputSliceStride(inputSliceStride),
          outputSliceStride(outputSliceStride),
          repeatNum(repeatNum),
          inputRepeatStride(inputRepeatStride),
          outputRepeatStride(outputRepeatStride),
          input(input),
          output(output),
          cclBufferSize(cclBufferSize)
    {}
    AivSuperKernelArgsDef() {}
};

HcclResult RegisterKernel();

HcclResult ExecuteKernelLaunchInner(const AivOpArgs& opArgs, void* args, u32 argsSize);

HcclResult ExecuteKernelLaunch(const AivOpArgs& opArgs);
} // namespace Hccl

#endif // HCCL_AIV_UTILS_H
