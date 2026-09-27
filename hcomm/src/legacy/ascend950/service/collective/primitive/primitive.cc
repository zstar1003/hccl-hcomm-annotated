/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "primitive.h"
#include "null_ptr_exception.h"
#include "prim_queue.h"

#include <set>

namespace Hccl {
PrimPostTo::PrimPostTo(const weak_ptr<PrimQueue> queue, NotifyType notifyType, u32 topicId)
    : Primitive(PrimType::POST_TO),
      queue(queue),
      notifyType(notifyType),
      topicId(topicId)
{
    if (queue.lock().get() == nullptr) {
        THROW<NullPtrException>("queue");
    }
}

void PrimPostTo::SetParent(const weak_ptr<PrimQueue>& que)
{
    if (que.lock().get() == nullptr) {
        THROW<NullPtrException>("parent");
    }
    if (GetQid() == que.lock()->GetId()) {
        THROW<InvalidParamsException>("parent Qid is equal to queue Qid");
    }
    parent = que;
}

std::string PrimPostTo::Describe() const
{
    if (parent.lock().get() == nullptr) {
        return StringFormat(
            "%s Qid[%u] NotifyType[%s]", type.Describe().c_str(), queue.lock()->GetId(), notifyType.Describe().c_str());
    } else {
        return StringFormat(
            "%s parent[%u] postTo Qid[%u] NotifyType[%s]", type.Describe().c_str(), parent.lock()->GetId(),
            queue.lock()->GetId(), notifyType.Describe().c_str());
    }
}

QId PrimPostTo::GetQid() const { return queue.lock()->GetId(); }

QId PrimPostTo::GetParentQid() const
{
    if (parent.lock().get() == nullptr) {
        return INVALID_PRIM_QID;
    } else {
        return parent.lock()->GetId();
    }
}

PrimWaitFrom::PrimWaitFrom(const weak_ptr<PrimQueue> queue, u32 topicId)
    : Primitive(PrimType::WAIT_FROM),
      queue(queue),
      topicId(topicId)
{
    if (queue.lock().get() == nullptr) {
        THROW<NullPtrException>("queue");
    }
}

std::string PrimWaitFrom::Describe() const
{
    if (parent.lock().get() == nullptr) {
        return StringFormat("%s Qid[%u]", type.Describe().c_str(), queue.lock()->GetId());
    } else {
        return StringFormat(
            "%s parent[%u] waitFrom Qid[%u]", type.Describe().c_str(), parent.lock()->GetId(), queue.lock()->GetId());
    }
}

void PrimWaitFrom::SetParent(const weak_ptr<PrimQueue>& que)
{
    if (que.lock().get() == nullptr) {
        THROW<NullPtrException>("parent");
    }
    if (GetQid() == que.lock()->GetId()) {
        THROW<InvalidParamsException>("parent Qid is equal to queue Qid");
    }
    parent = que;
}

QId PrimWaitFrom::GetQid() const { return queue.lock()->GetId(); }

QId PrimWaitFrom::GetParentQid() const
{
    if (parent.lock().get() == nullptr) {
        return INVALID_PRIM_QID;
    } else {
        return parent.lock()->GetId();
    }
}

PrimWaitGroup::PrimWaitGroup(u32 topicId) : Primitive(PrimType::WAIT_GROUP), topicId(topicId) {}

void PrimWaitGroup::Append(const weak_ptr<PrimQueue> queue)
{
    if (queue.lock().get() == nullptr) {
        THROW<NullPtrException>("queue");
    }
    qids.push_back(queue.lock()->GetId());
}

std::string PrimWaitGroup::Describe() const
{
    std::string qidsStr;
    for (u32 idx = 0; idx < qids.size(); idx++) {
        qidsStr += StringFormat("qid[%u], ", qids[idx]);
    }
    if (!qidsStr.empty()) {
        u32 redundantLen = 2;
        qidsStr = qidsStr.substr(0, qidsStr.size() - redundantLen);
    }

    if (parent.lock().get() == nullptr) {
        return StringFormat("%s: qidNum[%zu] qids[%s]", type.Describe().c_str(), qids.size(), qidsStr.c_str());
    } else {
        return StringFormat(
            "%s: parent[%u] qidNum[%zu] qids[%s]", type.Describe().c_str(), parent.lock()->GetId(), qids.size(),
            qidsStr.c_str());
    }
}

void PrimWaitGroup::SetParent(const weak_ptr<PrimQueue>& que)
{
    if (que.lock().get() == nullptr) {
        THROW<NullPtrException>("parent");
    }

    QId parentQid = que.lock()->GetId();
    for (auto qid = qids.begin(); qid != qids.end(); ++qid) {
        if (*qid == parentQid) {
            THROW<InvalidParamsException>("parent Qid is equal to one of queue Qids");
        }
    }

    parent = que;
}

QId PrimWaitGroup::GetParentQid() const
{
    if (parent.lock().get() == nullptr) {
        return INVALID_PRIM_QID;
    } else {
        return parent.lock()->GetId();
    }
}

PrimLocalCopy::PrimLocalCopy(const DataSlice& srcSlice, const DataSlice& dstSlice)
    : Primitive(PrimType::LOCAL_COPY),
      srcSlice(srcSlice),
      dstSlice(dstSlice)
{
    if (srcSlice.GetSize() != dstSlice.GetSize()) {
        THROW<InvalidParamsException>("The size of dstSlice is not equal to srcSlice");
    }
    if (srcSlice.GetType() == dstSlice.GetType()) {
        u64 srcStart = srcSlice.GetOffset();
        u64 srcEnd = srcStart + srcSlice.GetSize();
        u64 dstStart = dstSlice.GetOffset();
        u64 dstEnd = dstStart + dstSlice.GetSize();
        if (srcStart >= dstStart && srcStart < dstEnd) {
            THROW<InvalidParamsException>("The addresses of dstSlice and srcSlice overlap");
        }
        if (dstStart >= srcStart && dstStart < srcEnd) {
            THROW<InvalidParamsException>("The addresses of dstSlice and srcSlice overlap");
        }
    }
}

std::string PrimLocalCopy::Describe() const
{
    return StringFormat(
        "%s: src[%s], dst[%s]", type.Describe().c_str(), srcSlice.Describe().c_str(), dstSlice.Describe().c_str());
}

PrimLocalReduce::PrimLocalReduce(
    const DataSlice& srcSlice, const DataSlice& dstSlice, DataType dataType, ReduceOp reduceOp)
    : Primitive(PrimType::LOCAL_REDUCE),
      srcSlice(srcSlice),
      dstSlice(dstSlice),
      dataType(dataType),
      reduceOp(reduceOp)
{
    if (srcSlice.GetSize() != dstSlice.GetSize()) {
        THROW<InvalidParamsException>("The size of dstSlice is not equal to srcSlice");
    }
    if (srcSlice.GetType() == dstSlice.GetType()) {
        u64 srcStart = srcSlice.GetOffset();
        u64 srcEnd = srcStart + srcSlice.GetSize();
        u64 dstStart = dstSlice.GetOffset();
        u64 dstEnd = dstStart + dstSlice.GetSize();
        if (srcStart >= dstStart && srcStart < dstEnd) {
            THROW<InvalidParamsException>("The addresses of dstSlice and srcSlice overlap");
        }
        if (dstStart >= srcStart && dstStart < srcEnd) {
            THROW<InvalidParamsException>("The addresses of dstSlice and srcSlice overlap");
        }
    }
}

std::string PrimLocalReduce::Describe() const
{
    return StringFormat(
        "%s: %s, %s, src[%s], dst[%s]", type.Describe().c_str(), reduceOp.Describe().c_str(),
        dataType.Describe().c_str(), srcSlice.Describe().c_str(), dstSlice.Describe().c_str());
}

PrimSend::PrimSend(
    RankId remoteRank, const LinkData& link, const DataSlice& localSlice, const DataSlice& remoteSlice, DmaMode dmaMode)
    : Primitive(PrimType::SEND),
      remoteRank(remoteRank),
      link(link),
      dmaMode(dmaMode)
{
    if (localSlice.GetSize() != remoteSlice.GetSize()) {
        THROW<InvalidParamsException>("The size of remoteSlice is not equal to localSlice");
    }
    localSlices.push_back(localSlice);
    remoteSlices.push_back(remoteSlice);
}

std::string PrimSend::Describe() const
{
    string desc = StringFormat(
        "%s: remoteRank[%u], %s, %s, sliceNUm[%u]", type.Describe().c_str(), remoteRank, link.Describe().c_str(),
        dmaMode.Describe().c_str(), localSlices.size());
    for (u32 idx = 0; idx < localSlices.size(); idx++) {
        desc += StringFormat(
            " sliceIdx[%d]: local%s, remote%s;", idx, localSlices[idx].Describe().c_str(),
            remoteSlices[idx].Describe().c_str());
    }
    return desc;
}

void PrimSend::Append(const DataSlice& localSlice, const DataSlice& remoteSlice)
{
    if (localSlice.GetSize() != remoteSlice.GetSize()) {
        THROW<InvalidParamsException>("The size of remoteSlice is not equal to localSlice");
    }
    localSlices.push_back(localSlice);
    remoteSlices.push_back(remoteSlice);
}

PrimRecv::PrimRecv(
    RankId remoteRank, const LinkData& link, const DataSlice& localSlice, const DataSlice& remoteSlice, DmaMode dmaMode)
    : Primitive(PrimType::RECV),
      remoteRank(remoteRank),
      link(link),
      dmaMode(dmaMode)
{
    if (localSlice.GetSize() != remoteSlice.GetSize()) {
        THROW<InvalidParamsException>("The size of remoteSlice is not equal to localSlice");
    }
    localSlices.push_back(localSlice);
    remoteSlices.push_back(remoteSlice);
}

std::string PrimRecv::Describe() const
{
    string desc = StringFormat(
        "%s: remoteRank[%u], %s, %s, sliceNUm[%u]", type.Describe().c_str(), remoteRank, link.Describe().c_str(),
        dmaMode.Describe().c_str(), localSlices.size());
    for (u32 idx = 0; idx < localSlices.size(); idx++) {
        desc += StringFormat(
            " sliceIdx[%d]: local%s, remote%s;", idx, localSlices[idx].Describe().c_str(),
            remoteSlices[idx].Describe().c_str());
    }
    return desc;
}

void PrimRecv::Append(const DataSlice& localSlice, const DataSlice& remoteSlice)
{
    if (localSlice.GetSize() != remoteSlice.GetSize()) {
        THROW<InvalidParamsException>("The size of remoteSlice is not equal to localSlice");
    }
    localSlices.push_back(localSlice);
    remoteSlices.push_back(remoteSlice);
}

PrimSendReduce::PrimSendReduce(
    RankId remoteRank, const LinkData& link, const DataSlice& localSlice, const DataSlice& remoteSrcSlice,
    const DataSlice& remoteDstSlice, const DataType& dataType, const ReduceOp& reduceOp, DmaMode dmaMode)
    : Primitive(PrimType::SEND_REDUCE),
      remoteRank(remoteRank),
      link(link),
      dataType(dataType),
      reduceOp(reduceOp),
      dmaMode(dmaMode)
{
    if (localSlice.GetSize() != remoteSrcSlice.GetSize() || remoteSrcSlice.GetSize() != remoteDstSlice.GetSize()) {
        THROW<InvalidParamsException>("The size of localSlice, remoteSrcSlice, and remoteDstSlice are not equal");
    }
    localSlices.push_back(localSlice);
    remoteSrcSlices.push_back(remoteSrcSlice);
    remoteDstSlices.push_back(remoteDstSlice);
}

std::string PrimSendReduce::Describe() const
{
    string desc = StringFormat(
        "%s: remoteRank[%u], %s, %s, sliceNum[%u]", type.Describe().c_str(), remoteRank, link.Describe().c_str(),
        dmaMode.Describe().c_str(), localSlices.size());
    for (u32 idx = 0; idx < localSlices.size(); idx++) {
        desc += StringFormat(
            " sliceIdx[%d]: local%s, remoteSrc%s, remoteDst%s;", idx, localSlices[idx].Describe().c_str(),
            remoteSrcSlices[idx].Describe().c_str(), remoteDstSlices[idx].Describe().c_str());
    }
    return desc;
}

void PrimSendReduce::Append(
    const DataSlice& localSlice, const DataSlice& remoteSrcSlice, const DataSlice& remoteDstSlice)
{
    if (localSlice.GetSize() != remoteSrcSlice.GetSize() || remoteSrcSlice.GetSize() != remoteDstSlice.GetSize()) {
        THROW<InvalidParamsException>("The size of localSlice, remoteSrcSlice, and remoteDstSlice are not equal");
    }
    localSlices.push_back(localSlice);
    remoteSrcSlices.push_back(remoteSrcSlice);
    remoteDstSlices.push_back(remoteDstSlice);
}

PrimRecvReduce::PrimRecvReduce(
    RankId remoteRank, const LinkData& link, const DataSlice& remoteSlice, const DataSlice& localSrcSlice,
    const DataSlice& localDstSlice, const DataType& dataType, const ReduceOp& reduceOp, DmaMode dmaMode)
    : Primitive(PrimType::RECV_REDUCE),
      remoteRank(remoteRank),
      link(link),
      dataType(dataType),
      reduceOp(reduceOp),
      dmaMode(dmaMode)
{
    if (remoteSlice.GetSize() != localSrcSlice.GetSize() || localSrcSlice.GetSize() != localDstSlice.GetSize()) {
        THROW<InvalidParamsException>("The size of remoteSlice, localSrcSlice, and localDstSlice are not equal");
    }
    remoteSlices.push_back(remoteSlice);
    localSrcSlices.push_back(localSrcSlice);
    localDstSlices.push_back(localDstSlice);
}

std::string PrimRecvReduce::Describe() const
{
    string desc = StringFormat(
        "%s: remoteRank[%u], %s, %s, sliceNum[%u]", type.Describe().c_str(), remoteRank, link.Describe().c_str(),
        dmaMode.Describe().c_str(), remoteSlices.size());
    for (u32 idx = 0; idx < remoteSlices.size(); idx++) {
        desc += StringFormat(
            " sliceIdx[%d]: local%s, localSrc%s, localDst%s;", idx, remoteSlices[idx].Describe().c_str(),
            localSrcSlices[idx].Describe().c_str(), localDstSlices[idx].Describe().c_str());
    }
    return desc;
}

void PrimRecvReduce::Append(
    const DataSlice& remoteSlice, const DataSlice& localSrcSlice, const DataSlice& localDstSlice)
{
    if (remoteSlice.GetSize() != localSrcSlice.GetSize() || localSrcSlice.GetSize() != localDstSlice.GetSize()) {
        THROW<InvalidParamsException>("The size of remoteSlice, localSrcSlice, and localDstSlice are not equal");
    }
    remoteSlices.push_back(remoteSlice);
    localSrcSlices.push_back(localSrcSlice);
    localDstSlices.push_back(localDstSlice);
}

std::string PrimGroup::Describe() const
{
    string desc = StringFormat("%s: primSize[%zu]", type.Describe().c_str(), prims.size());
    auto primIter = prims.begin();
    for (; primIter != prims.end(); primIter++) {
        desc += (*primIter)->Describe() + "\n";
    }
    return desc;
}

void PrimGroup::CheckValid() const
{
    std::set<LinkData> sendLink;
    std::set<LinkData> recvLink;
    for (auto iter = Iter(); iter.HasNext(); ++iter) {
        if (iter->GetType() == PrimType::SEND) {
            PrimSend* primSend = dynamic_cast<PrimSend*>(const_cast<Primitive*>(&(*iter)));
            if (sendLink.count(primSend->GetLink()) > 0) {
                THROW<InvalidParamsException>("One link has two Send Prims");
            }
            sendLink.insert(primSend->GetLink());
        } else if (iter->GetType() == PrimType::SEND_REDUCE) {
            PrimSendReduce* primSendReduce = dynamic_cast<PrimSendReduce*>(const_cast<Primitive*>(&(*iter)));
            if (sendLink.count(primSendReduce->GetLink()) > 0) {
                THROW<InvalidParamsException>("One link has two Send Prims");
            }
            sendLink.insert(primSendReduce->GetLink());
        } else if (iter->GetType() == PrimType::RECV) {
            PrimRecv* primRecv = dynamic_cast<PrimRecv*>(const_cast<Primitive*>(&(*iter)));
            if (recvLink.count(primRecv->GetLink()) > 0) {
                THROW<InvalidParamsException>("One link has two Recv Prims");
            }
            recvLink.insert(primRecv->GetLink());
        } else { // when come here , the PrimType is PrimType::RECV_REDUCE
            PrimRecvReduce* primRecvReduce = dynamic_cast<PrimRecvReduce*>(const_cast<Primitive*>(&(*iter)));
            if (recvLink.count(primRecvReduce->GetLink()) > 0) {
                THROW<InvalidParamsException>("One link has two Recv Prims");
            }
            recvLink.insert(primRecvReduce->GetLink());
        }
    }
    return;
}

void PrimGroup::Append(unique_ptr<Primitive> prim)
{
    if (prim->GetType() != PrimType::SEND && prim->GetType() != PrimType::RECV
        && prim->GetType() != PrimType::SEND_REDUCE && prim->GetType() != PrimType::RECV_REDUCE) {
        THROW<InvalidParamsException>("PrimGroup only support PrimSend or PrimRecv or "
                                      "PrimSendReduce or PrimRecvReduce");
    }
    prims.push_back(std::move(prim));
    return;
}
} // namespace Hccl
