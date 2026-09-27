/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_PRIMITIVE_H
#define HCCLV2_PRIMITIVE_H

#include <string>
#include <memory>
#include <list>
#include "types.h"
#include "data_slice.h"
#include "notify_type.h"
#include "data_type.h"
#include "dma_mode.h"
#include "reduce_op.h"
#include "string_util.h"
#include "virtual_topo.h"
#include "invalid_params_exception.h"

namespace Hccl {

using namespace std;

MAKE_ENUM(
    PrimType, POST_TO, WAIT_FROM, WAIT_GROUP, LOCAL_COPY, LOCAL_REDUCE, SEND, RECV, GROUP, SEND_REDUCE, RECV_REDUCE)

class Primitive {
public:
    explicit Primitive(PrimType type) : type(type) {};

    virtual ~Primitive() = default;

    virtual string Describe() const = 0;
    PrimType GetType() const { return type; }

protected:
    PrimType type;
};

class PrimQueue;
constexpr u32 INVALID_PRIM_QID = 0xffffff; // 无效的原语队列
class PrimPostTo : public Primitive {
public:
    PrimPostTo(const weak_ptr<PrimQueue> queue, NotifyType notifyType = NotifyType::NORMAL, u32 topicId = 0);

    string Describe() const override;

    void SetParent(const weak_ptr<PrimQueue>& que);

    QId GetQid() const;
    QId GetParentQid() const;
    u32 GetTopicId() const { return topicId; }
    NotifyType GetNotifyType() const { return notifyType; }

private:
    weak_ptr<PrimQueue> queue;
    NotifyType notifyType;
    u32 topicId;
    weak_ptr<PrimQueue> parent;
};

class PrimWaitFrom : public Primitive {
public:
    PrimWaitFrom(const weak_ptr<PrimQueue> queue, u32 topicId = 0);

    string Describe() const override;

    void SetParent(const weak_ptr<PrimQueue>& que);

    QId GetQid() const;
    QId GetParentQid() const;
    u32 GetTopicId() const { return topicId; }

private:
    weak_ptr<PrimQueue> queue;
    u32 topicId;
    weak_ptr<PrimQueue> parent;
};

class PrimWaitGroup : public Primitive {
public:
    PrimWaitGroup(u32 topicId = 0);

    using Iterator = BaseConstIterator<vector, QId>;

    void Append(const weak_ptr<PrimQueue> queue);

    string Describe() const override;

    void SetParent(const weak_ptr<PrimQueue>& que);

    QId GetParentQid() const;

    u32 GetTopicId() const { return topicId; }

    Iterator Iter() const { return Iterator(qids); }

private:
    vector<QId> qids;
    u32 topicId;
    weak_ptr<PrimQueue> parent;
};

class PrimLocalCopy : public Primitive {
public:
    PrimLocalCopy(const DataSlice& srcSlice, const DataSlice& dstSlice);

    string Describe() const override;

    const DataSlice& GetSrcSlice() const { return srcSlice; }
    const DataSlice& GetDstSlice() const { return dstSlice; }

private:
    DataSlice srcSlice;
    DataSlice dstSlice;
};

class PrimLocalReduce : public Primitive {
public:
    PrimLocalReduce(const DataSlice& srcSlice, const DataSlice& dstSlice, DataType dataType, ReduceOp reduceOp);

    string Describe() const override;

    const DataType& GetDataType() const { return dataType; }
    const ReduceOp& GetReduceOp() const { return reduceOp; }
    const DataSlice& GetSrcSlice() const { return srcSlice; }
    const DataSlice& GetDstSlice() const { return dstSlice; }

private:
    DataSlice srcSlice;
    DataSlice dstSlice;
    DataType dataType;
    ReduceOp reduceOp;
};

class PrimSend : public Primitive {
public:
    PrimSend(
        RankId remoteRank, const LinkData& link, const DataSlice& localSlice, const DataSlice& remoteSlice,
        DmaMode dmaMode = DmaMode::DEFAULT);

    string Describe() const override;
    void Append(const DataSlice& localSlice, const DataSlice& remoteSlice);

    void SetRemoteRank(RankId remote) { remoteRank = remote; }
    void SetLink(const LinkData& l) { this->link = l; }
    RankId GetRemoteRank() const { return remoteRank; }
    const LinkData& GetLink() const { return link; }
    DmaMode GetDmaMode() const { return dmaMode; }
    u32 Size() const { return localSlices.size(); }
    const DataSlice& GetLocalSlice(u32 pos) const
    {
        if (pos >= localSlices.size()) {
            throw InvalidParamsException("pos is out of range of localSlices");
        }
        return localSlices[pos];
    }
    const DataSlice& GetRemoteSlice(u32 pos) const
    {
        if (pos >= remoteSlices.size()) {
            throw InvalidParamsException("pos is out of range of remoteSlices");
        }
        return remoteSlices[pos];
    }

private:
    RankId remoteRank;
    LinkData link;
    vector<DataSlice> localSlices;
    vector<DataSlice> remoteSlices;
    DmaMode dmaMode{DmaMode::DEFAULT};
};

class PrimRecv : public Primitive {
public:
    PrimRecv(
        RankId remoteRank, const LinkData& link, const DataSlice& localSlice, const DataSlice& remoteSlice,
        DmaMode dmaMode = DmaMode::DEFAULT);

    string Describe() const override;
    void Append(const DataSlice& localSlice, const DataSlice& remoteSlice);

    void SetRemoteRank(RankId remote) { remoteRank = remote; }
    void SetLink(const LinkData& l) { this->link = l; }

    RankId GetRemoteRank() const { return remoteRank; }
    const LinkData& GetLink() const { return link; }
    DmaMode GetDmaMode() const { return dmaMode; }
    u32 Size() const { return localSlices.size(); }
    const DataSlice& GetLocalSlice(u32 pos) const
    {
        if (pos >= localSlices.size()) {
            throw InvalidParamsException("pos is out of range of localSlices");
        }
        return localSlices[pos];
    }
    const DataSlice& GetRemoteSlice(u32 pos) const
    {
        if (pos >= remoteSlices.size()) {
            throw InvalidParamsException("pos is out of range of remoteSlices");
        }
        return remoteSlices[pos];
    }

private:
    RankId remoteRank;
    LinkData link;
    vector<DataSlice> localSlices;
    vector<DataSlice> remoteSlices;
    DmaMode dmaMode{DmaMode::DEFAULT};
};

class PrimSendReduce : public Primitive {
public:
    PrimSendReduce(
        RankId remoteRank, const LinkData& link, const DataSlice& localSlice, const DataSlice& remoteSrcSlice,
        const DataSlice& remoteDstSlice, const DataType& dataType, const ReduceOp& reduceOp,
        DmaMode dmaMode = DmaMode::DEFAULT);

    string Describe() const override;
    void Append(const DataSlice& localSlice, const DataSlice& remoteSrcSlice, const DataSlice& remoteDstSlice);

    void SetRemoteRank(RankId remote) { remoteRank = remote; }
    void SetLink(const LinkData& l) { this->link = l; }

    RankId GetRemoteRank() const { return remoteRank; }
    const LinkData& GetLink() const { return link; }
    DmaMode GetDmaMode() const { return dmaMode; }
    const DataType& GetDataType() const { return dataType; }
    const ReduceOp& GetReduceOp() const { return reduceOp; }
    u32 Size() const { return localSlices.size(); }
    const DataSlice& GetLocalSlice(u32 pos) const
    {
        if (pos >= localSlices.size()) {
            throw InvalidParamsException("pos is out of range of localSlices");
        }
        return localSlices[pos];
    }
    const DataSlice& GetRemoteSrcSlice(u32 pos) const
    {
        if (pos >= remoteSrcSlices.size()) {
            throw InvalidParamsException("pos is out of range of remoteSrcSlices");
        }
        return remoteSrcSlices[pos];
    }
    const DataSlice& GetRemoteDstSlice(u32 pos) const
    {
        if (pos >= remoteDstSlices.size()) {
            throw InvalidParamsException("pos is out of range of remoteDstSlices");
        }
        return remoteDstSlices[pos];
    }

private:
    RankId remoteRank;
    LinkData link;
    vector<DataSlice> localSlices;
    vector<DataSlice> remoteSrcSlices;
    vector<DataSlice> remoteDstSlices;
    DataType dataType;
    ReduceOp reduceOp;
    DmaMode dmaMode{DmaMode::DEFAULT};
};

class PrimRecvReduce : public Primitive {
public:
    PrimRecvReduce(
        RankId remoteRank, const LinkData& link, const DataSlice& remoteSlice, const DataSlice& localSrcSlice,
        const DataSlice& localDstSlice, const DataType& dataType, const ReduceOp& reduceOp,
        DmaMode dmaMode = DmaMode::DEFAULT);

    string Describe() const override;
    void Append(const DataSlice& remoteSlice, const DataSlice& localSrcSlice, const DataSlice& localDstSlice);

    void SetRemoteRank(RankId remote) { remoteRank = remote; }
    void SetLink(const LinkData& l) { this->link = l; }

    RankId GetRemoteRank() const { return remoteRank; }
    const LinkData& GetLink() const { return link; }
    DmaMode GetDmaMode() const { return dmaMode; }
    const DataType& GetDataType() const { return dataType; }
    const ReduceOp& GetReduceOp() const { return reduceOp; }
    u32 Size() const { return remoteSlices.size(); }
    const DataSlice& GetRemoteSlice(u32 pos) const
    {
        if (pos >= remoteSlices.size()) {
            throw InvalidParamsException("pos is out of range of remoteSlices");
        }
        return remoteSlices[pos];
    }
    const DataSlice& GetLocalSrcSlice(u32 pos) const
    {
        if (pos >= localSrcSlices.size()) {
            throw InvalidParamsException("pos is out of range of localSrcSlices");
        }
        return localSrcSlices[pos];
    }
    const DataSlice& GetLocalDstSlice(u32 pos) const
    {
        if (pos >= localDstSlices.size()) {
            throw InvalidParamsException("pos is out of range of localDstSlices");
        }
        return localDstSlices[pos];
    }

private:
    RankId remoteRank;
    LinkData link;
    vector<DataSlice> remoteSlices;
    vector<DataSlice> localSrcSlices;
    vector<DataSlice> localDstSlices;
    DataType dataType;
    ReduceOp reduceOp;
    DmaMode dmaMode{DmaMode::DEFAULT};
};

class PrimGroup : public Primitive {
public:
    PrimGroup() : Primitive(PrimType::GROUP) {}

    using Iterator = BaseConstIterator<vector, unique_ptr<Primitive>>;

    string Describe() const override;
    void CheckValid() const;
    void Append(unique_ptr<Primitive> prim);

    Iterator Iter() const { return Iterator(prims); }

    u32 GetSize() const { return prims.size(); }

private:
    vector<unique_ptr<Primitive>> prims;
};
} // namespace Hccl
#endif
