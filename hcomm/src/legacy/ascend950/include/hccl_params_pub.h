/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_HCCL_PARAMS_PUB_H
#define HCCL_HCCL_PARAMS_PUB_H

#include <string>
#include <functional>
#include <unordered_map>
#include "types.h"
#include "enum_factory.h"
#include "data_type.h"
#include "op_type.h"
#include "reduce_op.h"
#include "dev_type.h"

namespace Hccl {

constexpr u8 MAX_SUB_COMM_DEPTH = 2;

class CommParams {
public:
    std::string commId{""};
    RankId myRank{0};
    u32 rankSize{0};
    /* rankInParentGroup: 子通信域(group)内的rank在父通信域(hccl_world_group)中的rankId.
       创建hccl_world_group通信域时，myRank与rankInParentGroup相等;
       CreateGroup创建子通信域时，myRank为子通信域内的rankId，此时myRank与rankInParentGroup不一定相等 */
    RankId rankInParentComm{0};
    DevType devType{DevType::DEV_TYPE_950};
    bool devUsed{false};
    bool isWorldGroup{false};
    u8 commDepth{0}; // 0:world, 1:child, 2:grandchild

    CommParams(
        std::string commId, RankId myRank, u32 rankSize, RankId rankInParentComm, const DevType& devType,
        bool devUsed = false, bool isWorldGroup = false)
        : commId(std::move(commId)),
          myRank(myRank),
          rankSize(rankSize),
          rankInParentComm(rankInParentComm),
          devType(devType),
          devUsed(devUsed),
          isWorldGroup(isWorldGroup)
    {}

    CommParams() {}
};

class CollOpParams {
public:
    OpType opType;
    DataType dataType;
    ReduceOp reduceOp;
    u32 dstRank;
    void* sendBuf;
    void* recvBuf;
    u64 count{0};
    u32 root{0};
    bool staticAddr{false};
    bool staticShape{false};
    DataType outputDataType{DataType::INVALID};
    std::string opTag;
    bool isMc2{false};
    std::string algConfig;
    HcclAccelerator commEngine;
    union {
        struct {
            u64 dataCount;
            DataType dataType;
            u64 strideCount;
        } dataDes;
        struct {
            void* counts;
            void* displs;
            DataType dataType;
        } vDataDes;
        struct {
            DataType sendType;
            DataType recvType;
            u64 sendCount;
            u64 recvCount;
        } all2AllDataDes;
        struct {
            DataType sendType;
            DataType recvType;
            void* sendCounts;
            void* recvCounts;
            void* sdispls;
            void* rdispls;
        } all2AllVDataDes;
        struct {
            DataType sendType;
            DataType recvType;
            void* sendCountMatrix;
        } all2AllVCDataDes;
        struct {
            void* sendRecvItemsPtr;
            u32 itemNum;
        } batchSendRecvDataDes;
    };
    // 使用初始化列表
    CollOpParams()
        : opType(),
          dataType(),
          reduceOp(),
          dstRank(),
          sendBuf(),
          recvBuf(),
          count(),
          root(),
          staticAddr(),
          staticShape(),
          outputDataType()
    {
        // 显式初始化 union 的默认成员
        dataDes = {0, DataType::INVALID, 0}; // 假设 dataDes 是默认使用的成员
    }

    std::string Describe() const;

private:
    std::string DescReduceScatter(const CollOpParams& opParams);

    std::string DescAllreduce(const CollOpParams& opParams);

    std::string DescAllgather(const CollOpParams& opParams);

    std::string DescScatter(const CollOpParams& opParams);

    std::string DescAlltoall(const CollOpParams& opParams);

    std::string DescAlltoallV(const CollOpParams& opParams);

    std::string DescAlltoallVC(const CollOpParams& opParams);

    std::string DescSend(const CollOpParams& opParams);

    std::string DescRecv(const CollOpParams& opParams);

    std::string DescReduce(const CollOpParams& opParams);

    std::string DescBroadcast(const CollOpParams& opParams);

    std::string DescBatchSendRecv(const CollOpParams& opParams);

    std::string DescAllGatherV(const CollOpParams& opParams);

    std::string DescReduceScatterV(const CollOpParams& opParams);

    std::unordered_map<OpType, std::function<std::string(const CollOpParams&)>, std::EnumClassHash> descOpMap{
        {OpType::REDUCESCATTER, std::bind(&CollOpParams::DescReduceScatter, this, std::placeholders::_1)},
        {OpType::ALLREDUCE, std::bind(&CollOpParams::DescAllreduce, this, std::placeholders::_1)},
        {OpType::ALLGATHER, std::bind(&CollOpParams::DescAllgather, this, std::placeholders::_1)},
        {OpType::SCATTER, std::bind(&CollOpParams::DescScatter, this, std::placeholders::_1)},
        {OpType::ALLTOALL, std::bind(&CollOpParams::DescAlltoall, this, std::placeholders::_1)},
        {OpType::ALLTOALLV, std::bind(&CollOpParams::DescAlltoallV, this, std::placeholders::_1)},
        {OpType::ALLTOALLVC, std::bind(&CollOpParams::DescAlltoallVC, this, std::placeholders::_1)},
        {OpType::SEND, std::bind(&CollOpParams::DescSend, this, std::placeholders::_1)},
        {OpType::RECV, std::bind(&CollOpParams::DescRecv, this, std::placeholders::_1)},
        {OpType::REDUCE, std::bind(&CollOpParams::DescReduce, this, std::placeholders::_1)},
        {OpType::BROADCAST, std::bind(&CollOpParams::DescBroadcast, this, std::placeholders::_1)},
        {OpType::BATCHSENDRECV, std::bind(&CollOpParams::DescBatchSendRecv, this, std::placeholders::_1)},
        {OpType::ALLGATHERV, std::bind(&CollOpParams::DescAllGatherV, this, std::placeholders::_1)},
        {OpType::REDUCESCATTERV, std::bind(&CollOpParams::DescReduceScatterV, this, std::placeholders::_1)}
        // 后续待补充其他算子信息
    };
};

struct CollOffloadOpResReq {
    u64 requiredSubQueNum{0};
    u64 requiredScratchMemSize{0};
};
} // namespace Hccl

#endif // HCCL_HCCL_PARAMS_PUB_H
