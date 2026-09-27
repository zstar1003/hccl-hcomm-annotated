/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_COLL_OPERATOR_H
#define HCCLV2_COLL_OPERATOR_H
#include <memory>
#include <string>
#include <vector>
#include "types.h"
#include "op_type.h"
#include "op_mode.h"
#include "data_type.h"
#include "reduce_op.h"
#include "buffer_type.h"
#include "buffer.h"
#include "const_val.h"
namespace Hccl {
using BaseCollOperator = struct BaseCollOperatorDef {
    OpMode opMode{OpMode::INVALID};
    OpType opType{OpType::DEBUGCASE}; // A5的类型
    u32 oldOpType{0};                 // A3的类型
    ReduceOp reduceOp{ReduceOp::INVALID};
    u32 oldReduceOp{0}; // A3的类型
    DataType dataType{DataType::INVALID};
    u32 oldDataType{0};                         // 仅用于A3的场景
    DataType outputDataType{DataType::INVALID}; // 低精度场景，存在指定输出数据类型
    u64 dataCount{0};
    u32 root{0};
    u32 numBlocksLimit{0};
    RankId sendRecvRemoteRank{0};
    u64 inputAddr{0};
    u64 inputSize{0};
    u64 outputAddr{0};
    u64 outputSize{0};
    std::shared_ptr<Buffer> inputMem{nullptr};
    std::shared_ptr<Buffer> outputMem{nullptr};
    std::shared_ptr<Buffer> scratchMem{nullptr};
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
    BaseCollOperatorDef() : opMode(), opType(), reduceOp(), dataType(), dataCount(0), root(0), sendRecvRemoteRank()
    {
        // 显式初始化 union 的默认成员
        dataDes = {0, DataType::INVALID, 0}; // 假设 dataDes 是默认使用的成员
    }
    Buffer* GetBuffer(const BufferType type)
    {
        if (type == BufferType::INPUT) {
            return inputMem.get();
        } else if (type == BufferType::OUTPUT) {
            return outputMem.get();
        } else if (type == BufferType::SCRATCH) {
            return scratchMem.get();
        } else {
            return nullptr;
        }
    }
};

using CollAlgOperator = BaseCollOperator;

using CollOperator = struct CollOperatorDef : public BaseCollOperator {
    std::string opTag;
    bool staticAddr{false};
    bool staticShape{false};
    RankId myRank{INVALID_RANKID};
    std::vector<char> GetUniqueId() const;
    static CollOperatorDef GetPackedData(std::vector<char>& byteVector);
};

std::string MemBufferDesc(const BaseCollOperator& collOp);
std::string OpDesc(const BaseCollOperator& collOp);
std::string DescReduceScatter(const BaseCollOperator& collOp);
std::string DescAllreduce(const BaseCollOperator& collOp);
std::string DescAllgather(const BaseCollOperator& collOp);
std::string DescScatter(const BaseCollOperator& collOp);
std::string DescAlltoall(const BaseCollOperator& collOp);
std::string DescAlltoallV(const BaseCollOperator& collOp);
std::string DescAlltoallVC(const BaseCollOperator& collOp);
std::string DescSend(const BaseCollOperator& collOp);
std::string DescRecv(const BaseCollOperator& collOp);
std::string DescReduce(const BaseCollOperator& collOp);
std::string DescBroadcast(const BaseCollOperator& collOp);
std::string DescBatchSendRecv(const BaseCollOperator& collOp);
std::string DescHalfAlltoAllV(const BaseCollOperator& collOp);
std::string DescReduceScatterV(const BaseCollOperator& collOp);
std::string DescAllGatherV(const BaseCollOperator& collOp);

std::string CollOpToString(const BaseCollOperator& collOp);

} // namespace Hccl

#endif // !HCCLV2_COLL_OPERATOR_H
