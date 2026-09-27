/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_operator.h"
#include <string>
#include <unordered_map>
#include <algorithm>
#include <functional>
#include "op_type.h"
#include "string_util.h"
#include "binary_stream.h"
namespace Hccl {
constexpr u32 MAX_OP_TAG_LEN = 191;            // 最大的tag 长度
constexpr u32 MAX_HANDSHAKEMSGPACK_LEN = 1024; // 最大握手消息长度

std::string MemBufferDesc(const BaseCollOperator& collOp)
{
    std::string memDesc = "";
    memDesc += "inputMem=" + (collOp.inputMem ? collOp.inputMem->Describe() : "nullptr") + ", ";
    memDesc += "outputMem=" + (collOp.outputMem ? collOp.outputMem->Describe() : "nullptr") + ", ";
    memDesc += "scratchMem=" + (collOp.scratchMem ? collOp.scratchMem->Describe() : "nullptr");
    return memDesc;
}

std::string OpDesc(const BaseCollOperator& collOp)
{
    std::string opDesc = "";
    opDesc += "opType=" + collOp.opType.Describe() + ", ";
    opDesc += "opMode=" + collOp.opMode.Describe() + ", ";
    opDesc += "dataType=" + collOp.dataType.Describe() + ", ";
    opDesc += "sendRecvRemoteRank=" + std::to_string(collOp.sendRecvRemoteRank) + ", ";
    opDesc += "Buffers=[" + MemBufferDesc(collOp) + "]";
    return opDesc;
}

std::string DescReduceScatter(const BaseCollOperator& collOp)
{
    return StringFormat(
        "BaseCollOperator[%s, reduceOp=%s, dataCount=%llu]", OpDesc(collOp).c_str(), collOp.reduceOp.Describe().c_str(),
        collOp.dataCount);
}

std::string DescAllreduce(const BaseCollOperator& collOp)
{
    return StringFormat(
        "BaseCollOperator[%s, reduceOp=%s, dataCount=%llu]", OpDesc(collOp).c_str(), collOp.reduceOp.Describe().c_str(),
        collOp.dataCount);
}

std::string DescAllgather(const BaseCollOperator& collOp)
{
    return StringFormat("BaseCollOperator[%s, dataCount=%llu]", OpDesc(collOp).c_str(), collOp.dataCount);
}

std::string DescScatter(const BaseCollOperator& collOp)
{
    return StringFormat(
        "BaseCollOperator[%s, dataCount=%llu, root=%u]", OpDesc(collOp).c_str(), collOp.dataCount, collOp.root);
}

std::string DescAlltoall(const BaseCollOperator& collOp)
{
    return StringFormat(
        "BaseCollOperator[opType=%s, opMode=%s, sendCount=%llu, recvCount=%llu, sendType=%s, recvType=%s, "
        "sendRecvRemoteRank=%u, Buffers=[%s]]",
        collOp.opType.Describe().c_str(), collOp.opMode.Describe().c_str(), collOp.all2AllDataDes.sendCount,
        collOp.all2AllDataDes.recvCount, collOp.all2AllDataDes.sendType.Describe().c_str(),
        collOp.all2AllDataDes.recvType.Describe().c_str(), collOp.sendRecvRemoteRank, MemBufferDesc(collOp).c_str());
}

std::string DescAlltoallV(const BaseCollOperator& collOp)
{
    return StringFormat(
        "BaseCollOperator[opType=%s, opMode=%s, sendType=%s, recvType=%s, sendRecvRemoteRank=%u, Buffers=[%s]]",
        collOp.opType.Describe().c_str(), collOp.opMode.Describe().c_str(),
        collOp.all2AllVDataDes.sendType.Describe().c_str(), collOp.all2AllVDataDes.recvType.Describe().c_str(),
        collOp.sendRecvRemoteRank, MemBufferDesc(collOp).c_str());
}

std::string DescAlltoallVC(const BaseCollOperator& collOp)
{
    return StringFormat(
        "BaseCollOperator[opType=%s, opMode=%s, sendType=%s, recvType=%s, sendRecvRemoteRank=%u, Buffers=[%s]]",
        collOp.opType.Describe().c_str(), collOp.opMode.Describe().c_str(),
        collOp.all2AllVCDataDes.sendType.Describe().c_str(), collOp.all2AllVCDataDes.recvType.Describe().c_str(),
        collOp.sendRecvRemoteRank, MemBufferDesc(collOp).c_str());
}

std::string DescSend(const BaseCollOperator& collOp)
{
    return StringFormat("BaseCollOperator[%s]", OpDesc(collOp).c_str());
}

std::string DescRecv(const BaseCollOperator& collOp)
{
    return StringFormat("BaseCollOperator[%s]", OpDesc(collOp).c_str());
}

std::string DescReduce(const BaseCollOperator& collOp)
{
    return StringFormat(
        "BaseCollOperator[%s, reduceOp=%s, dataCount=%llu, root=%u]", OpDesc(collOp).c_str(),
        collOp.reduceOp.Describe().c_str(), collOp.dataCount, collOp.root);
}

std::string DescBroadcast(const BaseCollOperator& collOp)
{
    return StringFormat(
        "BaseCollOperator[%s, dataCount=%llu, root=%u]", OpDesc(collOp).c_str(), collOp.dataCount, collOp.root);
}

std::string DescBatchSendRecv(const BaseCollOperator& collOp)
{
    return StringFormat(
        "BaseCollOperator[%s, dataCount=%llu, root=%u]", OpDesc(collOp).c_str(), collOp.dataCount, collOp.root);
}

std::string DescHalfAlltoAllV(const BaseCollOperator& collOp)
{
    return StringFormat("BaseCollOperator[%s]", OpDesc(collOp).c_str());
}

std::string DescReduceScatterV(const BaseCollOperator& collOp)
{
    return StringFormat("BaseCollOperator[%s]", OpDesc(collOp).c_str());
}

std::string DescAllGatherV(const BaseCollOperator& collOp)
{
    return StringFormat("BaseCollOperator[%s]", OpDesc(collOp).c_str());
}

std::unordered_map<OpType, std::function<std::string(const BaseCollOperator&)>, std::EnumClassHash> descOpMap{
    {OpType::REDUCESCATTER, std::bind(&DescReduceScatter, std::placeholders::_1)},
    {OpType::ALLREDUCE, std::bind(&DescAllreduce, std::placeholders::_1)},
    {OpType::ALLGATHER, std::bind(&DescAllgather, std::placeholders::_1)},
    {OpType::SCATTER, std::bind(&DescScatter, std::placeholders::_1)},
    {OpType::ALLTOALL, std::bind(&DescAlltoall, std::placeholders::_1)},
    {OpType::ALLTOALLV, std::bind(&DescAlltoallV, std::placeholders::_1)},
    {OpType::ALLTOALLVC, std::bind(&DescAlltoallVC, std::placeholders::_1)},
    {OpType::SEND, std::bind(&DescSend, std::placeholders::_1)},
    {OpType::RECV, std::bind(&DescRecv, std::placeholders::_1)},
    {OpType::REDUCE, std::bind(&DescReduce, std::placeholders::_1)},
    {OpType::BROADCAST, std::bind(&DescBroadcast, std::placeholders::_1)},
    {OpType::BATCHSENDRECV, std::bind(&DescBatchSendRecv, std::placeholders::_1)},
    {OpType::HALFALLTOALLV, std::bind(&DescHalfAlltoAllV, std::placeholders::_1)},
    {OpType::REDUCESCATTERV, std::bind(&DescReduceScatterV, std::placeholders::_1)},
    {OpType::ALLGATHERV, std::bind(&DescAllGatherV, std::placeholders::_1)},
};

std::string CollOpToString(const BaseCollOperator& collOp)
{
    auto it = descOpMap.find(collOp.opType);
    if (it != descOpMap.end()) {
        return it->second.operator()(collOp);
    } else {
        return "unknown";
    }
}

inline std::vector<char> DumpByteVector(BinaryStream& binaryStream)
{
    std::vector<char> byteVector;
    binaryStream.Dump(byteVector);

    auto remainLen = MAX_HANDSHAKEMSGPACK_LEN - byteVector.size();
    byteVector.insert(byteVector.end(), remainLen, '\0');

    return byteVector;
}

std::vector<char> opTagToVector(const std::string& opTag)
{
    std::vector<char> result(MAX_OP_TAG_LEN, '\0');
    auto copyLen = opTag.size() < MAX_OP_TAG_LEN ? opTag.size() : MAX_OP_TAG_LEN;
    std::copy_n(opTag.begin(), copyLen, result.begin());

    return result;
}

std::string vectorToOpTag(const std::vector<char>& opTagvector)
{
    auto validSize = opTagvector.size() < MAX_OP_TAG_LEN ? opTagvector.size() : MAX_OP_TAG_LEN;
    auto firstNul = std::find(opTagvector.begin(), opTagvector.begin() + validSize, '\0');

    return std::string(opTagvector.begin(), firstNul);
}

std::vector<char> CollOperator::GetUniqueId() const
{
    HCCL_INFO(
        "[CollOperator::%s] opMode[%s], opType[%s], reduceOp[%s], dataType[%s], dataCount[%llu], root[%u], "
        "myRank[%d], sendRecvRemoteRank[%d], opTag[%s], staticAddr[%d], staticShape[%d], outputDataType[%s], ",
        __func__, opMode.Describe().c_str(), opType.Describe().c_str(), reduceOp.Describe().c_str(),
        dataType.Describe().c_str(), dataCount, root, myRank, sendRecvRemoteRank, opTag.c_str(), staticAddr,
        staticShape, outputDataType.Describe().c_str());
    BinaryStream binaryStream;
    binaryStream << opMode;
    binaryStream << opType;
    binaryStream << reduceOp;
    binaryStream << dataType;
    binaryStream << dataCount;
    binaryStream << root;
    binaryStream << myRank;
    binaryStream << sendRecvRemoteRank;
    binaryStream << opTagToVector(opTag);
    binaryStream << staticAddr;
    binaryStream << staticShape;
    binaryStream << outputDataType;

    if (opType == OpType::BATCHSENDRECV) {
        return DumpByteVector(binaryStream);
    }

    if (opType == OpType::ALLTOALL) {
        binaryStream << all2AllDataDes.sendType;
        binaryStream << all2AllDataDes.recvType;
        binaryStream << all2AllDataDes.sendCount;
        binaryStream << all2AllDataDes.recvCount;
        return DumpByteVector(binaryStream);
    }

    if (opType == OpType::ALLTOALLV) {
        binaryStream << all2AllVDataDes.sendType;
        binaryStream << all2AllVDataDes.recvType;
        return DumpByteVector(binaryStream);
    }

    if (opType == OpType::ALLTOALLVC) {
        binaryStream << all2AllVCDataDes.sendType;
        binaryStream << all2AllVCDataDes.recvType;
        return DumpByteVector(binaryStream);
    }

    if (opType == OpType::ALLGATHERV || opType == OpType::REDUCESCATTERV) {
        binaryStream << vDataDes.dataType;
        return DumpByteVector(binaryStream);
    }

    binaryStream << dataDes.dataCount;
    binaryStream << dataDes.dataType;
    binaryStream << dataDes.strideCount;

    return DumpByteVector(binaryStream);
}

CollOperatorDef CollOperator::GetPackedData(std::vector<char>& byteVector)
{
    CollOperator op;
    BinaryStream binaryStream(byteVector);
    std::vector<char> vectorOpTag;
    binaryStream >> op.opMode;
    binaryStream >> op.opType;
    binaryStream >> op.reduceOp;
    binaryStream >> op.dataType;
    binaryStream >> op.dataCount;
    binaryStream >> op.root;
    binaryStream >> op.myRank;
    binaryStream >> op.sendRecvRemoteRank;
    binaryStream >> vectorOpTag;
    binaryStream >> op.staticAddr;
    binaryStream >> op.staticShape;
    binaryStream >> op.outputDataType;

    op.opTag = vectorToOpTag(vectorOpTag);

    if (op.opType == OpType::BATCHSENDRECV) {
        return op;
    }

    if (op.opType == OpType::ALLTOALL) {
        binaryStream >> op.all2AllDataDes.sendType;
        binaryStream >> op.all2AllDataDes.recvType;
        binaryStream >> op.all2AllDataDes.sendCount;
        binaryStream >> op.all2AllDataDes.recvCount;
        return op;
    }

    if (op.opType == OpType::ALLTOALLV) {
        binaryStream >> op.all2AllVDataDes.sendType;
        binaryStream >> op.all2AllVDataDes.recvType;
        return op;
    }

    if (op.opType == OpType::ALLTOALLVC) {
        binaryStream >> op.all2AllVCDataDes.sendType;
        binaryStream >> op.all2AllVCDataDes.recvType;
        return op;
    }

    if (op.opType == OpType::ALLGATHERV || op.opType == OpType::REDUCESCATTERV) {
        binaryStream >> op.vDataDes.dataType;
        return op;
    }

    binaryStream >> op.dataDes.dataCount;
    binaryStream >> op.dataDes.dataType;
    binaryStream >> op.dataDes.strideCount;

    return op;
}
} // namespace Hccl
