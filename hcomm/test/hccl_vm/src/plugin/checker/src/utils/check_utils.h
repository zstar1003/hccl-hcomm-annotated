/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV1_CHECK_UTILS_H
#define HCCLV1_CHECK_UTILS_H

#include <map>
#include <memory>
#include <set>
#include <vector>

#include "binary_data_type_pub.h"
#include "checker_def.h"
#include "enum_factory.h"
#include "log.h"
#include "string_util.h"
#include "task_graph_generator.h"

namespace HcclSim {
const std::string FOUR_INDENT_SPACE = "    ";

inline std::string BufferTypeToString(BufferType bufferType)
{
    switch (bufferType) {
        case BufferType::INPUT:
            return "INPUT";
        case BufferType::OUTPUT:
            return "OUTPUT";
        case BufferType::CCL:
            return "CCL";
        case BufferType::SCRATCH:
            return "SCRATCH";
        case BufferType::INPUT_AIV:
            return "INPUT_AIV";
        case BufferType::OUTPUT_AIV:
            return "OUTPUT_AIV";
        case BufferType::AIV_COMMINFO:
            return "AIV_COMMINFO";
        case BufferType::USERBUF_AIV:
            return "USERBUF_AIV";
        case BufferType::MS:
            return "MS";
        default:
            return "INVALID";
    }
}

inline std::string HcclCmdTypeToString(HcclCMDType cmdType)
{
    switch (cmdType) {
        case HCCL_CMD_BROADCAST:
            return "Broadcast";
        case HCCL_CMD_ALLREDUCE:
            return "AllReduce";
        case HCCL_CMD_REDUCE:
            return "Reduce";
        case HCCL_CMD_SEND:
            return "Send";
        case HCCL_CMD_RECEIVE:
            return "Recv";
        case HCCL_CMD_ALLGATHER:
            return "AllGather";
        case HCCL_CMD_REDUCE_SCATTER:
            return "ReduceScatter";
        case HCCL_CMD_ALLTOALLV:
            return "AllToAllV";
        case HCCL_CMD_ALLTOALLVC:
            return "AllToAllVC";
        case HCCL_CMD_ALLTOALL:
            return "AllToAll";
        case HCCL_CMD_SCATTER:
            return "Scatter";
        case HCCL_CMD_BATCH_SEND_RECV:
            return "BatchSendRecv";
        case HCCL_CMD_ALLGATHER_V:
            return "AllGatherV";
        case HCCL_CMD_REDUCE_SCATTER_V:
            return "ReduceScatterV";
        default:
            return "Unknown";
    }
}

inline std::string HcclDataTypeToString(HcclDataType dataType)
{
    switch (dataType) {
        case HCCL_DATA_TYPE_INT8:
            return "INT8";
        case HCCL_DATA_TYPE_INT16:
            return "INT16";
        case HCCL_DATA_TYPE_INT32:
            return "INT32";
        case HCCL_DATA_TYPE_FP16:
            return "FP16";
        case HCCL_DATA_TYPE_FP32:
            return "FP32";
        case HCCL_DATA_TYPE_INT64:
            return "INT64";
        case HCCL_DATA_TYPE_UINT64:
            return "UINT64";
        case HCCL_DATA_TYPE_UINT8:
            return "UINT8";
        case HCCL_DATA_TYPE_UINT16:
            return "UINT16";
        case HCCL_DATA_TYPE_UINT32:
            return "UINT32";
        case HCCL_DATA_TYPE_FP64:
            return "FP64";
        case HCCL_DATA_TYPE_BFP16:
            return "BFP16";
        default:
            return "Unknown";
    }
}

inline std::string ReduceOpToString(HcclReduceOp reduceOp)
{
    switch (reduceOp) {
        case HCCL_REDUCE_SUM:
            return "HCCL_REDUCE_SUM";
        case HCCL_REDUCE_PROD:
            return "HCCL_REDUCE_PROD";
        case HCCL_REDUCE_MAX:
            return "HCCL_REDUCE_MAX";
        case HCCL_REDUCE_MIN:
            return "HCCL_REDUCE_MIN";
        case HCCL_REDUCE_RESERVED:
            return "HCCL_REDUCE_RESERVED";
        default:
            return "INVALID";
    }
}

struct SrcBufDes {
    RankId rankId;      // 数据源的rankId
    BufferType bufType; // 数据源的内存类型
    u64 srcAddr;        // 数据源的地址
    SrcBufDes(RankId id, BufferType type, u64 addr) : rankId(id), bufType(type), srcAddr(addr) {}
    inline bool operator<(const SrcBufDes& another) const { return rankId < another.rankId; }

    std::string Describe() const
    {
        std::stringstream ret;
        ret << "    - sourceRank=" << rankId << ", sourceBufferType=" << BufferTypeToString(bufType)
            << ", sourceAddr=0x" << std::hex << srcAddr << std::dec << '\n';
        return ret.str();
    }
};

struct BufferSemantic {
    u64 startAddr;                       // 起始地址
    mutable u64 size;                    // 大小
    mutable bool isReduce;               // 是否做了reduce操作
    mutable HcclReduceOp reduceType;     // reduce操作的类型
    mutable std::set<SrcBufDes> srcBufs; // 这块数据来自哪个或哪些rank
    std::vector<u32> affectedGlobalSteps; // 表示这个语义块被哪个、哪些节点影响了，用于图形化界面展示

    BufferSemantic(
        u64 startAddr, u64 size, bool isReduce = false, HcclReduceOp reduceType = HcclReduceOp::HCCL_REDUCE_RESERVED)
        : startAddr(startAddr),
          size(size),
          isReduce(isReduce),
          reduceType(reduceType)
    {}

    BufferSemantic(u64 startAddr, u64 size, bool isReduce, HcclReduceOp reduceType, std::set<SrcBufDes> srcBufs)
        : startAddr(startAddr),
          size(size),
          isReduce(isReduce),
          reduceType(reduceType),
          srcBufs(srcBufs)
    {}

    inline bool operator<(const BufferSemantic& another) const { return startAddr < another.startAddr; }

    std::string Describe() const
    {
        std::stringstream ret;
        ret << "  range=[0x" << std::hex << startAddr << ",0x" << (startAddr + size) << ")"
            << ", size=0x" << size << std::dec;
        if (isReduce) {
            ret << ", reduce=" << ReduceOpToString(reduceType);
        }
        ret << ", sourceCount=" << srcBufs.size() << '\n';
        ret << "  sources:\n";
        for (const auto& ele : srcBufs) {
            ret << ele.Describe();
        }
        return ret.str();
    }
};

using RankMemorySemantics = std::map<BufferType, std::set<BufferSemantic>>;

// SrcBufDes 会放进 std::set 里，地址平移时必须重建集合，不能原地修改 key。
std::set<SrcBufDes> OffsetSrcBufs(const std::set<SrcBufDes>& srcBufs, u64 offset);

TaskTypeStub GetNodeType(const TaskNode* node);
void CalcInputOutputSize(
    HcclCMDType opType, uint32_t rankSize, uint64_t count, HcclDataType dataType, u64& inputSize, u64& outputSize,
    RankId myRank, RankId srcRank = 0, RankId dstRank = 0, VDataDesTagInner vDataDes = VDataDesTagInner{},
    All2AllDataDesTagInner all2AllDataDes = All2AllDataDesTagInner{});
void CalcDataSize(HcclCMDType opType, uint64_t count, HcclDataType dataType, u64& dataSize);
bool IsAllToAllSeries(HcclCMDType opType);
void GenTopoMeta(TopoMeta& topoMate, int superPodNum, int serverNum, int rankNum);
u32 CalRankSize(const TopoMeta& topoMeta);
} // namespace HcclSim

#endif
