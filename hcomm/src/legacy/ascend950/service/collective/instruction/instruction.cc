/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "instruction.h"
#include "types.h"
#include "inttypes.h"

namespace Hccl {
string InsLocalCopy::Describe() const
{
    return StringFormat(
        "InsLocalCopy[srcSlice=%s, dstSlice=%s]", srcSlice_.Describe().c_str(), dstSlice_.Describe().c_str());
}
const DataSlice& InsLocalCopy::GetSrcSlice() const { return srcSlice_; }
const DataSlice& InsLocalCopy::GetDstSlice() const { return dstSlice_; }

string InsLocalCopyExtend::Describe() const
{
    return StringFormat(
        "InsLocalCopyExtend[srcBuffer=%s, dstBuffer=%s]", srcBuffer_.Describe().c_str(), dstBuffer_.Describe().c_str());
}
const DataBuffer& InsLocalCopyExtend::GetSrcBuffer() const { return srcBuffer_; }
const DataBuffer& InsLocalCopyExtend::GetDstBuffer() const { return dstBuffer_; }

string InsLocalReduce::Describe() const
{
    return StringFormat(
        "InsLocalReduce[dataType=%s, reduceOp=%s, srcSlice=%s, dstSlice=%s]", dataType_.Describe().c_str(),
        reduceOp_.Describe().c_str(), srcSlice_.Describe().c_str(), dstSlice_.Describe().c_str());
}
const DataSlice& InsLocalReduce::GetSrcSlice() const { return srcSlice_; }
const DataSlice& InsLocalReduce::GetDstSlice() const { return dstSlice_; }
const DataType InsLocalReduce::GetDataType() const { return dataType_; }
const ReduceOp InsLocalReduce::GetReduceOp() const { return reduceOp_; }

string InsLocalPostTo::Describe() const
{
    return StringFormat(
        "InsLocalPostTo[notifyType=%s, postQid=%u, waitQid=%u, topicId=%u]", notifyType_.Describe().c_str(), postQid_,
        waitQid_, topicId_);
}

void InsLocalPostTo::SetPostQid(QId qid)
{
    if (waitQid_ == qid) {
        THROW<InvalidParamsException>("post Qid is equal to wait Qid");
    }
    postQid_ = qid;
}

QId InsLocalPostTo::GetPostQid() const { return postQid_; }
QId InsLocalPostTo::GetWaitQid() const { return waitQid_; }
u32 InsLocalPostTo::GetTopicId() const { return topicId_; }
NotifyType InsLocalPostTo::GetNotifyType() const { return notifyType_; }

string InsLocalWaitFrom::Describe() const
{
    return StringFormat("InsLocalWaitFrom[waitQid=%u, postQid=%u, topicId=%u]", waitQid_, postQid_, topicId_);
}

void InsLocalWaitFrom::SetWaitQid(QId qid)
{
    if (postQid_ == qid) {
        THROW<InvalidParamsException>("post Qid is equal to wait Qid");
    }
    waitQid_ = qid;
}

QId InsLocalWaitFrom::GetPostQid() const { return postQid_; }
QId InsLocalWaitFrom::GetWaitQid() const { return waitQid_; }
u32 InsLocalWaitFrom::GetTopicId() const { return topicId_; }
NotifyType InsLocalWaitFrom::GetNotifyType() const { return notifyType_; }

using Iterator = BaseConstIterator<vector, QId>;
void InsLocalWaitGroup::Append(QId postQid) { postQids_.push_back(postQid); }
string InsLocalWaitGroup::Describe() const
{
    std::string postQidsStr;
    for (u32 idx = 0; idx < postQids_.size(); idx++) {
        postQidsStr += StringFormat("%u, ", postQids_[idx]);
    }
    if (!postQidsStr.empty()) {
        u32 redundantLen = 2;
        postQidsStr = postQidsStr.substr(0, postQidsStr.size() - redundantLen);
    }

    return StringFormat(
        "InsLocalWaitGroup[waitQid=%u, topicId=%u, postQidNum=%zu, postQids=postQidList[%s]]", waitQid_, topicId_,
        postQids_.size(), postQidsStr.c_str());
}
QId InsLocalWaitGroup::GetWaitQid() const { return waitQid_; }
u32 InsLocalWaitGroup::GetTopicId() const { return topicId_; }
void InsLocalWaitGroup::SetWaitQid(QId qId)
{
    for (auto iter = Iter(); iter.HasNext(); ++iter) {
        if (*iter == qId) {
            THROW<InvalidParamsException>("One of post Qids is equal to wait Qid");
        }
    }

    waitQid_ = qId;
}

void InsLocalBcastPost::Append(QId waitQid) { waitQids_.push_back(waitQid); }
string InsLocalBcastPost::Describe() const
{
    std::string waitQidsStr;
    for (u32 idx = 0; idx < waitQids_.size(); idx++) {
        waitQidsStr += StringFormat("%u, ", waitQids_[idx]);
    }
    if (!waitQidsStr.empty()) {
        u32 redundantLen = 2;
        waitQidsStr = waitQidsStr.substr(0, waitQidsStr.size() - redundantLen);
    }

    return StringFormat(
        "InsLocalBcastPost[postQid=%d, topicId=%d, waitQidNum=%u, waitQids=waitQidList[%s]]", postQid_, topicId_,
        waitQids_.size(), waitQidsStr.c_str());
}
QId InsLocalBcastPost::GetPostQid() const { return postQid_; }
u32 InsLocalBcastPost::GetTopicId() const { return topicId_; }
void InsLocalBcastPost::SetPostQid(QId qId)
{
    for (auto iter = Iter(); iter.HasNext(); ++iter) {
        if (*iter == qId) {
            THROW<InvalidParamsException>("One of post Qids is equal to wait Qid");
        }
    }

    postQid_ = qId;
}

string InsPostReady::Describe() const
{
    return StringFormat("InsPostReady:remoteRank=%d, link=%s", remoteRank_, link_.Describe().c_str());
}
RankId InsPostReady::GetRemoteRank() const { return remoteRank_; }
const LinkData* InsPostReady::GetLink() const { return &link_; }

string InsWaitReady::Describe() const
{
    return StringFormat("InsWaitReady:remoteRank=%d, link=%s", remoteRank_, link_.Describe().c_str());
}
RankId InsWaitReady::GetRemoteRank() const { return remoteRank_; }
const LinkData* InsWaitReady::GetLink() const { return &link_; }

string InsPostFin::Describe() const
{
    return StringFormat("InsPostFin:remoteRank=%d, link=%s", remoteRank_, link_.Describe().c_str());
}
RankId InsPostFin::GetRemoteRank() const { return remoteRank_; }
const LinkData* InsPostFin::GetLink() const { return &link_; }

string InsWaitFin::Describe() const
{
    return StringFormat("InsWaitFin:remoteRank=%d, link=%s", remoteRank_, link_.Describe().c_str());
}
RankId InsWaitFin::GetRemoteRank() const { return remoteRank_; }
const LinkData* InsWaitFin::GetLink() const { return &link_; }

string InsWaitGroupFin::Describe() const
{
    string linksStr;
    for (auto iter = links_.begin(); iter != links_.end(); ++iter) {
        linksStr += iter->Describe();
    }
    if (!linksStr.empty()) {
        u32 redundantLen = 2;
        linksStr = linksStr.substr(0, linksStr.size() - redundantLen);
    }
    return StringFormat("InsWaitGroupFin[topicId=%u, value=0x%x, links=%s]", topicId_, value_, linksStr.c_str());
}
u32 InsWaitGroupFin::GetTopicId() const { return topicId_; }
void InsWaitGroupFin::Append(LinkData link) { links_.push_back(link); }

u32 InsWaitGroupFin::GetValue() const { return value_; }

string InsPostFinAck::Describe() const
{
    return StringFormat("InsPostFinAck[remoteRank=%d, link=%s]", remoteRank_, link_.Describe().c_str());
}
RankId InsPostFinAck::GetRemoteRank() const { return remoteRank_; }
const LinkData* InsPostFinAck::GetLink() const { return &link_; }

string InsWaitFinAck::Describe() const
{
    return StringFormat("InsWaitFinAck[remoteRank=%d, link=%s]", remoteRank_, link_.Describe().c_str());
}
RankId InsWaitFinAck::GetRemoteRank() const { return remoteRank_; }
const LinkData* InsWaitFinAck::GetLink() const { return &link_; }

string InsRead::Describe() const
{
    return StringFormat(
        "InsRead[remoteRank=%d, link=%s, localSlice=%s, remoteSlice=%s]", remoteRank_, link_.Describe().c_str(),
        localSlice_.Describe().c_str(), remoteSlice_.Describe().c_str());
}

RankId InsRead::GetRemoteRank() const { return remoteRank_; }
const LinkData* InsRead::GetLink() const { return &link_; }
const DataSlice& InsRead::GetLocalSlice() const { return localSlice_; }
const DataSlice& InsRead::GetRemoteSlice() const { return remoteSlice_; }

string InsReadReduce::Describe() const
{
    return StringFormat(
        "InsReadReduce[remoteRank=%d, link=%s, dataType=%s, reduceOp=%s, localSlice=%s, remoteSlice=%s]", remoteRank_,
        link_.Describe().c_str(), dataType_.Describe().c_str(), reduceOp_.Describe().c_str(),
        localSlice_.Describe().c_str(), remoteSlice_.Describe().c_str());
}

string InsReadExtend::Describe() const
{
    return StringFormat(
        "InsReadExtend[remoteRank=%d, link=%s, localBuffer=%s, remoteBuffer=%s]", remoteRank_, link_.Describe().c_str(),
        localBuffer_.Describe().c_str(), remoteBuffer_.Describe().c_str());
}

RankId InsReadExtend::GetRemoteRank() const { return remoteRank_; }
const LinkData* InsReadExtend::GetLink() const { return &link_; }
const DataBuffer& InsReadExtend::GetLocalBuffer() const { return localBuffer_; }
const DataBuffer& InsReadExtend::GetRemoteBuffer() const { return remoteBuffer_; }

RankId InsReadReduce::GetRemoteRank() const { return remoteRank_; }
const LinkData* InsReadReduce::GetLink() const { return &link_; }
const DataSlice& InsReadReduce::GetLocalSlice() const { return localSlice_; }
const DataSlice& InsReadReduce::GetRemoteSlice() const { return remoteSlice_; }
const DataType InsReadReduce::GetDataType() const { return dataType_; }
const ReduceOp InsReadReduce::GetReduceOp() const { return reduceOp_; }

string InsBatchRead::Describe() const
{
    return StringFormat(
        "InsBatchRead[remoteRank=%d, link=%s, readInsVec size=%zu]", remoteRank, link.Describe().c_str(),
        readInsVec.size());
}

RankId InsBatchRead::GetRemoteRank() const { return remoteRank; }

const LinkData* InsBatchRead::GetLink() const { return &link; }

void InsBatchRead::PushReadIns(unique_ptr<Instruction> readIns)
{
    if (readIns->GetType() != InstructionType::READ && readIns->GetType() != InstructionType::READ_REDUCE) {
        THROW<NotSupportException>(StringFormat(
            "[InsBatchRead][%s] only support read and readReduce instruction type, "
            "but get instruction type[%s]",
            __func__, readIns->GetType().Describe().c_str()));
    }

    readInsVec.push_back(std::move(readIns));
}

string InsReadReduceExtend::Describe() const
{
    return StringFormat(
        "InsReadReduceExtend[remoteRank=%d, link=%s, dataType=%s, reduceOp=%s, localBuffer=%s, remoteBuffer=%s]",
        remoteRank_, link_.Describe().c_str(), dataType_.Describe().c_str(), reduceOp_.Describe().c_str(),
        localBuffer_.Describe().c_str(), remoteBuffer_.Describe().c_str());
}

RankId InsReadReduceExtend::GetRemoteRank() const { return remoteRank_; }
const LinkData* InsReadReduceExtend::GetLink() const { return &link_; }
const DataBuffer& InsReadReduceExtend::GetLocalBuffer() const { return localBuffer_; }
const DataBuffer& InsReadReduceExtend::GetRemoteBuffer() const { return remoteBuffer_; }
const DataType InsReadReduceExtend::GetDataType() const { return dataType_; }
const ReduceOp InsReadReduceExtend::GetReduceOp() const { return reduceOp_; }

string InsWrite::Describe() const
{
    return StringFormat(
        "InsWrite[remoteRank=%d, link=%s, localSlice=%s, remoteSlice=%s]", remoteRank_, link_.Describe().c_str(),
        localSlice_.Describe().c_str(), remoteSlice_.Describe().c_str());
}

RankId InsWrite::GetRemoteRank() const { return remoteRank_; }
const LinkData* InsWrite::GetLink() const { return &link_; }
const DataSlice& InsWrite::GetLocalSlice() const { return localSlice_; }
const DataSlice& InsWrite::GetRemoteSlice() const { return remoteSlice_; }

string InsWriteExtend::Describe() const
{
    return StringFormat(
        "InsWriteExtend[remoteRank=%d, link=%s, localBuffer=%s, remoteBuffer=%s]", remoteRank_,
        link_.Describe().c_str(), localBuffer_.Describe().c_str(), remoteBuffer_.Describe().c_str());
}

RankId InsWriteExtend::GetRemoteRank() const { return remoteRank_; }
const LinkData* InsWriteExtend::GetLink() const { return &link_; }
const DataBuffer& InsWriteExtend::GetLocalBuffer() const { return localBuffer_; }
const DataBuffer& InsWriteExtend::GetRemoteBuffer() const { return remoteBuffer_; }

string InsWriteReduce::Describe() const
{
    return StringFormat(
        "InsWriteReduce[remoteRank=%d, link=%s, dataType=%s, reduceOp=%s, localSlice=%s, remoteSlice=%s]", remoteRank_,
        link_.Describe().c_str(), dataType_.Describe().c_str(), reduceOp_.Describe().c_str(),
        localSlice_.Describe().c_str(), remoteSlice_.Describe().c_str());
}

RankId InsWriteReduce::GetRemoteRank() const { return remoteRank_; }
const LinkData* InsWriteReduce::GetLink() const { return &link_; }
const DataSlice& InsWriteReduce::GetLocalSlice() const { return localSlice_; }
const DataSlice& InsWriteReduce::GetRemoteSlice() const { return remoteSlice_; }
const DataType InsWriteReduce::GetDataType() const { return dataType_; }
const ReduceOp InsWriteReduce::GetReduceOp() const { return reduceOp_; }

string InsBatchWrite::Describe() const
{
    return StringFormat(
        "InsBatchWrite[remoteRank=%d, link=%s, writeInsVec size=%zu]", remoteRank, link.Describe().c_str(),
        writeInsVec.size());
}

RankId InsBatchWrite::GetRemoteRank() const { return remoteRank; }

const LinkData* InsBatchWrite::GetLink() const { return &link; }

void InsBatchWrite::PushWriteIns(unique_ptr<Instruction> writeIns)
{
    if (writeIns->GetType() != InstructionType::WRITE && writeIns->GetType() != InstructionType::WRITE_REDUCE) {
        THROW<NotSupportException>(StringFormat(
            "[InsBatchWrite][%s] only support Write and WriteReduce instruction "
            "type, but get instruction type[%s]",
            __func__, writeIns->GetType().Describe().c_str()));
    }

    writeInsVec.push_back(std::move(writeIns));
}

string InsWriteReduceExtend::Describe() const
{
    return StringFormat(
        "InsWriteReduceExtend[remoteRank=%d, link=%s, dataType=%s, reduceOp=%s, localBuffer=%s, remoteBuffer=%s]",
        remoteRank_, link_.Describe().c_str(), dataType_.Describe().c_str(), reduceOp_.Describe().c_str(),
        localBuffer_.Describe().c_str(), remoteBuffer_.Describe().c_str());
}

RankId InsWriteReduceExtend::GetRemoteRank() const { return remoteRank_; }
const LinkData* InsWriteReduceExtend::GetLink() const { return &link_; }
const DataBuffer& InsWriteReduceExtend::GetLocalBuffer() const { return localBuffer_; }
const DataBuffer& InsWriteReduceExtend::GetRemoteBuffer() const { return remoteBuffer_; }
const DataType InsWriteReduceExtend::GetDataType() const { return dataType_; }
const ReduceOp InsWriteReduceExtend::GetReduceOp() const { return reduceOp_; }

string InsWriteWithFin::Describe() const
{
    return StringFormat(
        "InsWriteWithFin[remoteRank=%d, link=%s, localSlice=%s, remoteSlice=%s, bitValue=0x%x]", remoteRank_,
        link_.Describe().c_str(), localSlice_.Describe().c_str(), remoteSlice_.Describe().c_str(), bitValue_);
}

RankId InsWriteWithFin::GetRemoteRank() const { return remoteRank_; }
const LinkData* InsWriteWithFin::GetLink() const { return &link_; }
const DataSlice& InsWriteWithFin::GetLocalSlice() const { return localSlice_; }
const DataSlice& InsWriteWithFin::GetRemoteSlice() const { return remoteSlice_; }

string InsWriteWithFinExtend::Describe() const
{
    return StringFormat(
        "InsWriteWithFinExtend[remoteRank=%d, link=%s, localBuffer=%s, remoteBuffer=%s, bitValue=0x%x]", remoteRank_,
        link_.Describe().c_str(), localBuffer_.Describe().c_str(), remoteBuffer_.Describe().c_str(), bitValue_);
}

RankId InsWriteWithFinExtend::GetRemoteRank() const { return remoteRank_; }
const LinkData* InsWriteWithFinExtend::GetLink() const { return &link_; }
const DataBuffer& InsWriteWithFinExtend::GetLocalBuffer() const { return localBuffer_; }
const DataBuffer& InsWriteWithFinExtend::GetRemoteBuffer() const { return remoteBuffer_; }

string InsWriteReduceWithFin::Describe() const
{
    return StringFormat(
        "InsWriteReduceWithFin[remoteRank=%d, link=%s, dataType=%s, reduceOp=%s, localSlice=%s, remoteSlice=%s, "
        "bitValue=0x%u]",
        remoteRank_, link_.Describe().c_str(), dataType_.Describe().c_str(), reduceOp_.Describe().c_str(),
        localSlice_.Describe().c_str(), remoteSlice_.Describe().c_str(), bitValue_);
}

RankId InsWriteReduceWithFin::GetRemoteRank() const { return remoteRank_; }
const LinkData* InsWriteReduceWithFin::GetLink() const { return &link_; }
const DataSlice& InsWriteReduceWithFin::GetLocalSlice() const { return localSlice_; }
const DataSlice& InsWriteReduceWithFin::GetRemoteSlice() const { return remoteSlice_; }
const DataType InsWriteReduceWithFin::GetDataType() const { return dataType_; }
const ReduceOp InsWriteReduceWithFin::GetReduceOp() const { return reduceOp_; }

string InsWriteReduceWithFinExtend::Describe() const
{
    return StringFormat(
        "InsWriteReduceWithFin[remoteRank=%d, link=%s, dataType=%s, reduceOp=%s, localBuffer=%s, remoteBuffer=%s, "
        "bitValue=0x%u]",
        remoteRank_, link_.Describe().c_str(), dataType_.Describe().c_str(), reduceOp_.Describe().c_str(),
        localBuffer_.Describe().c_str(), remoteBuffer_.Describe().c_str(), bitValue_);
}

RankId InsWriteReduceWithFinExtend::GetRemoteRank() const { return remoteRank_; }
const LinkData* InsWriteReduceWithFinExtend::GetLink() const { return &link_; }
const DataBuffer& InsWriteReduceWithFinExtend::GetLocalBuffer() const { return localBuffer_; }
const DataBuffer& InsWriteReduceWithFinExtend::GetRemoteBuffer() const { return remoteBuffer_; }
const DataType InsWriteReduceWithFinExtend::GetDataType() const { return dataType_; }
const ReduceOp InsWriteReduceWithFinExtend::GetReduceOp() const { return reduceOp_; }
const NotifyType& InsWriteReduceWithFinExtend::GetNotifyType() const { return notifyType_; }
const u32& InsWriteReduceWithFinExtend::GetTopicId() const { return topicId_; }
const u32& InsWriteReduceWithFinExtend::GetBitValue() const { return bitValue_; }

const InstructionType Instruction::GetType() const { return type_; }

const NotifyType& InsWriteWithFin::GetNotifyType() const { return notifyType_; }

const u32& InsWriteWithFin::GetTopicId() const { return topicId_; }

const u32& InsWriteWithFin::GetBitValue() const { return bitValue_; }

const NotifyType& InsWriteReduceWithFin::GetNotifyType() const { return notifyType_; }
const u32& InsWriteReduceWithFin::GetTopicId() const { return topicId_; }

const u32& InsWriteReduceWithFin::GetBitValue() const { return bitValue_; }

string InsBatchOneSidedRead::Describe() const
{
    return StringFormat("InsBatchOneSidedRead[remoteRank=%d, link=%s]", remoteRank_, link_.Describe().c_str());
}

RankId InsBatchOneSidedRead::GetRemoteRank() const { return remoteRank_; }
const LinkData* InsBatchOneSidedRead::GetLink() const { return &link_; }

const vector<RmaBufSliceLite>& InsBatchOneSidedRead::GetLocalSlice() const { return localSlice_; }

const vector<RmtRmaBufSliceLite>& InsBatchOneSidedRead::GetRemoteSlice() const { return remoteSlice_; }

string InsBatchOneSidedWrite::Describe() const
{
    return StringFormat("InsBatchOneSidedWrite[remoteRank=%d, link=%s]", remoteRank_, link_.Describe().c_str());
}

RankId InsBatchOneSidedWrite::GetRemoteRank() const { return remoteRank_; }
const LinkData* InsBatchOneSidedWrite::GetLink() const { return &link_; }
const vector<RmaBufSliceLite>& InsBatchOneSidedWrite::GetLocalSlice() const { return localSlice_; }

const vector<RmtRmaBufSliceLite>& InsBatchOneSidedWrite::GetRemoteSlice() const { return remoteSlice_; }

string InsStreamSync::Describe() const { return StringFormat("InsStreamSync"); }

string InsAicpuReduce::Describe() const
{
    return StringFormat(
        "InsAicpuReduce[dataType=%s, reduceOp=%s, srcSlice=%s, dstSlice=%s]", dataType_.Describe().c_str(),
        reduceOp_.Describe().c_str(), srcSlice_.Describe().c_str(), dstSlice_.Describe().c_str());
}
const DataSlice& InsAicpuReduce::GetSrcSlice() const { return srcSlice_; }
const DataSlice& InsAicpuReduce::GetDstSlice() const { return dstSlice_; }
const DataType InsAicpuReduce::GetDataType() const { return dataType_; }
const ReduceOp InsAicpuReduce::GetReduceOp() const { return reduceOp_; }

template <typename T>
void InsAicpuReduce::AicpuReduceTemplate(T* dst, u64 dstSize, T* src, u64 srcSize, ReduceOp reduceOp)
{
    if (dst == nullptr || src == nullptr) {
        THROW<NullPtrException>(StringFormat("nsAicpuReduce::AicpuReduceTemplate dst or src is nullptr"));
    }
    if (dstSize != srcSize) {
        string msg = StringFormat("srcSize[" PRIu64 "] should be equal to dstSize[" PRIu64 "]", srcSize, dstSize);
        THROW<InternalException>(msg);
    }
    u64 count = dstSize / u64(sizeof(T));
    for (u64 i = 0; i < count; ++i) {
        T dstData = *(dst + i);
        T srcData = *(src + i);
        switch (reduceOp) {
            case ReduceOp::SUM:
                *(dst + i) = srcData + dstData;
                break;
            case ReduceOp::PROD:
                *(dst + i) = srcData * dstData;
                break;
            case ReduceOp::MAX:
                *(dst + i) = std::max(srcData, dstData);
                break;
            case ReduceOp::MIN:
                *(dst + i) = std::min(srcData, dstData);
                break;
            default:
                string msg = StringFormat("ReduceOp[%d] not support", int(reduceOp));
                THROW<NotSupportException>(msg);
                break;
        }
    }
}

void InsAicpuReduce::RunAicpuReduce(
    void* dst, u64 dstSize, void* src, u64 srcSize, DataType dataType, ReduceOp reduceOp)
{
    switch (dataType) {
        case DataType::INT64:
            AicpuReduceTemplate<int64_t>((int64_t*)(dst), dstSize, (int64_t*)(src), srcSize, reduceOp);
            break;
        case DataType::UINT64:
            AicpuReduceTemplate<uint64_t>((uint64_t*)(dst), dstSize, (uint64_t*)(src), srcSize, reduceOp);
            break;
        case DataType::FP64:
            AicpuReduceTemplate<double>((double*)(dst), dstSize, (double*)(src), srcSize, reduceOp);
            break;
        default:
            string msg = StringFormat("DataType[%d] not support", int(dataType));
            THROW<NotSupportException>(msg);
            break;
    }
}

string InsPreStreamSync::Describe() const { return StringFormat("InsPreStreamSync"); }
} // namespace Hccl
