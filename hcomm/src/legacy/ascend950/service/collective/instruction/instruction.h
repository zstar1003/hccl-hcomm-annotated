/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_INSTRUCTION_H
#define HCCLV2_INSTRUCTION_H

#include <string>
#include <map>
#include <memory>
#include <list>
#include "coll_operator.h"
#include "types.h"
#include "data_slice.h"
#include "virtual_topo.h"
#include "notify_type.h"
#include "data_buffer.h"
#include "rmt_rma_buf_slice_lite.h"
#include "rma_buf_slice_lite.h"

namespace Hccl {

MAKE_ENUM(
    InstructionType, LOCAL_COPY, LOCAL_REDUCE, LOCAL_POST_TO, LOCAL_WAIT_FROM, LOCAL_WAIT_GROUP, LOCAL_BCAST_POST,
    POST_READY, WAIT_READY, POST_FIN, WAIT_FIN, WAIT_GROUP_FIN, POST_FIN_ACK, WAIT_FIN_ACK, READ, READ_REDUCE,
    BATCH_READ, WRITE, WRITE_REDUCE, BATCH_WRITE, WRITE_WITH_FIN, WRITE_REDUCE_WITH_FIN, CCU_INS, AICPU_INS,
    LOCAL_COPY_EXTEND, WRITE_EXTEND, WRITE_REDUCE_EXTEND, WRITE_REDUCE_WITH_FIN_EXTEND, READ_EXTEND, READ_REDUCE_EXTEND,
    WRITE_WITH_FIN_EXTEND, BATCH_ONE_SIDED_WRITE, BATCH_ONE_SIDED_READ, AIV_INS, STREAM_SYNC, AICPU_REDUCE,
    PRE_STREAM_SYNC)

constexpr u32 INVALID_TOPICID = 0xFFFFFFFF;
constexpr uint32_t NOTIFY_INDEX_READY = 0;
constexpr uint32_t NOTIFY_INDEX_FIN = 1;
constexpr uint32_t NOTIFY_INDEX_FIN_ACK = 2;
class Instruction {
public:
    explicit Instruction(InstructionType type) : type_(type) {}
    virtual ~Instruction() = default;
    virtual string Describe() const = 0;

    const InstructionType GetType() const;
    virtual const LinkData* GetLink() const { return nullptr; }

protected:
    InstructionType type_;
};

class InsLocalCopy : public Instruction {
public:
    InsLocalCopy(const DataSlice& srcSlice, const DataSlice& dstSlice)
        : Instruction(InstructionType::LOCAL_COPY),
          srcSlice_(srcSlice),
          dstSlice_(dstSlice)
    {}
    string Describe() const override;

    const DataSlice& GetSrcSlice() const;
    const DataSlice& GetDstSlice() const;

private:
    DataSlice srcSlice_;
    DataSlice dstSlice_;
};

class InsLocalCopyExtend : public Instruction {
public:
    InsLocalCopyExtend(const DataBuffer& srcBuffer, const DataBuffer& dstBuffer)
        : Instruction(InstructionType::LOCAL_COPY_EXTEND),
          srcBuffer_(srcBuffer),
          dstBuffer_(dstBuffer)
    {}
    string Describe() const override;

    const DataBuffer& GetSrcBuffer() const;
    const DataBuffer& GetDstBuffer() const;

private:
    DataBuffer srcBuffer_;
    DataBuffer dstBuffer_;
};

class InsLocalReduce : public Instruction {
public:
    InsLocalReduce(const DataSlice& srcSlice, const DataSlice& dstSlice, DataType dataType, ReduceOp reduceOp)
        : Instruction(InstructionType::LOCAL_REDUCE),
          srcSlice_(srcSlice),
          dstSlice_(dstSlice),
          dataType_(dataType),
          reduceOp_(reduceOp)
    {}
    string Describe() const override;

    const DataSlice& GetSrcSlice() const;
    const DataSlice& GetDstSlice() const;
    const DataType GetDataType() const;
    const ReduceOp GetReduceOp() const;

private:
    DataSlice srcSlice_;
    DataSlice dstSlice_;
    DataType dataType_;
    ReduceOp reduceOp_;
};
constexpr u32 INVALID_INSTRUCTION_QID = 0xffffff; // 无效的指令队列
class InsLocalPostTo : public Instruction {
public:
    explicit InsLocalPostTo(QId waitQid, NotifyType notifyType = NotifyType::NORMAL, u32 topicId = 0)
        : Instruction(InstructionType::LOCAL_POST_TO),
          waitQid_(waitQid),
          notifyType_(notifyType),
          topicId_(topicId)
    {}
    string Describe() const override;

    void SetPostQid(QId qid);

    QId GetPostQid() const;
    QId GetWaitQid() const;
    u32 GetTopicId() const;
    NotifyType GetNotifyType() const;

private:
    QId postQid_{INVALID_INSTRUCTION_QID};
    QId waitQid_;
    NotifyType notifyType_;
    u32 topicId_;
};

class InsLocalWaitFrom : public Instruction {
public:
    explicit InsLocalWaitFrom(QId postQid, NotifyType notifyType = NotifyType::NORMAL, u32 topicId = 0)
        : Instruction(InstructionType::LOCAL_WAIT_FROM),
          postQid_(postQid),
          notifyType_(notifyType),
          topicId_(topicId)
    {}
    string Describe() const override;

    void SetWaitQid(QId qid);

    QId GetPostQid() const;
    QId GetWaitQid() const;
    u32 GetTopicId() const;
    NotifyType GetNotifyType() const;

private:
    QId postQid_;
    QId waitQid_{INVALID_INSTRUCTION_QID};
    NotifyType notifyType_;
    u32 topicId_;
};

class InsLocalWaitGroup : public Instruction {
public:
    explicit InsLocalWaitGroup(u32 topicId = 0) : Instruction(InstructionType::LOCAL_WAIT_GROUP), topicId_(topicId) {}

    using Iterator = BaseConstIterator<vector, QId>;

    void Append(QId postQid);
    string Describe() const override;
    QId GetWaitQid() const;
    u32 GetTopicId() const;
    void SetWaitQid(QId qId);

    Iterator Iter() const { return Iterator(postQids_); }

private:
    vector<QId> postQids_;
    QId waitQid_;
    u32 topicId_;
};

class InsLocalBcastPost : public Instruction {
public:
    explicit InsLocalBcastPost(u32 topicId = 0) : Instruction(InstructionType::LOCAL_BCAST_POST), topicId_(topicId) {}

    using Iterator = BaseConstIterator<vector, QId>;

    void Append(QId waitQid);
    string Describe() const override;
    u32 GetPostQid() const;
    QId GetTopicId() const;
    void SetPostQid(QId qId);

    Iterator Iter() const { return Iterator(waitQids_); }

private:
    vector<QId> waitQids_;
    QId postQid_{0};
    u32 topicId_;
};

class InsPostReady : public Instruction {
public:
    InsPostReady(RankId remoteRank, const LinkData& link)
        : Instruction(InstructionType::POST_READY),
          remoteRank_(remoteRank),
          link_(link)
    {}
    string Describe() const override;

    RankId GetRemoteRank() const;
    const LinkData* GetLink() const override;

private:
    RankId remoteRank_;
    LinkData link_;
};

class InsWaitReady : public Instruction {
public:
    InsWaitReady(RankId remoteRank, const LinkData& link)
        : Instruction(InstructionType::WAIT_READY),
          remoteRank_(remoteRank),
          link_(link)
    {}
    string Describe() const override;

    RankId GetRemoteRank() const;
    const LinkData* GetLink() const override;

private:
    RankId remoteRank_;
    LinkData link_;
};

class InsPostFin : public Instruction {
public:
    InsPostFin(RankId remoteRank, const LinkData& link)
        : Instruction(InstructionType::POST_FIN),
          remoteRank_(remoteRank),
          link_(link)
    {}
    string Describe() const override;

    RankId GetRemoteRank() const;
    const LinkData* GetLink() const override;

private:
    RankId remoteRank_;
    LinkData link_;
};

class InsWaitFin : public Instruction {
public:
    InsWaitFin(RankId remoteRank, const LinkData& link)
        : Instruction(InstructionType::WAIT_FIN),
          remoteRank_(remoteRank),
          link_(link)
    {}
    string Describe() const override;

    RankId GetRemoteRank() const;
    const LinkData* GetLink() const override;

private:
    RankId remoteRank_;
    LinkData link_;
};

class InsPostFinAck : public Instruction {
public:
    InsPostFinAck(RankId remoteRank, const LinkData& link)
        : Instruction(InstructionType::POST_FIN_ACK),
          remoteRank_(remoteRank),
          link_(link)
    {}
    string Describe() const override;

    RankId GetRemoteRank() const;
    const LinkData* GetLink() const override;

private:
    RankId remoteRank_;
    LinkData link_;
};

class InsWaitGroupFin : public Instruction {
public:
    explicit InsWaitGroupFin(u32 topicId = 0)
        : Instruction(InstructionType::WAIT_GROUP_FIN),
          value_(0),
          topicId_(topicId)
    {}
    using Iterator = BaseConstIterator<vector, LinkData>;

    string Describe() const override;
    u32 GetTopicId() const;
    void Append(LinkData link);

    void SetValue(u32 givenValue) { value_ = givenValue; }

    u32 GetValue() const;

    Iterator Iter() const { return Iterator(links_); }

private:
    u32 value_;
    u32 topicId_;
    vector<LinkData> links_;
};

class InsWaitFinAck : public Instruction {
public:
    InsWaitFinAck(RankId remoteRank, const LinkData& link)
        : Instruction(InstructionType::WAIT_FIN_ACK),
          remoteRank_(remoteRank),
          link_(link)
    {}
    string Describe() const override;

    RankId GetRemoteRank() const;
    const LinkData* GetLink() const override;

private:
    RankId remoteRank_;
    LinkData link_;
};

class InsRead : public Instruction {
public:
    InsRead(RankId remoteRank, const LinkData& link, const DataSlice& localSlice, const DataSlice& remoteSlice)
        : Instruction(InstructionType::READ),
          remoteRank_(remoteRank),
          link_(link),
          localSlice_(localSlice),
          remoteSlice_(remoteSlice)
    {}
    string Describe() const override;

    RankId GetRemoteRank() const;
    const LinkData* GetLink() const override;
    const DataSlice& GetLocalSlice() const;
    const DataSlice& GetRemoteSlice() const;

private:
    RankId remoteRank_;
    LinkData link_;
    DataSlice localSlice_;
    DataSlice remoteSlice_;
};

class InsReadExtend : public Instruction {
public:
    InsReadExtend(
        RankId remoteRank, const LinkData& link, const DataBuffer& localBuffer, const DataBuffer& remoteBuffer)
        : Instruction(InstructionType::READ_EXTEND),
          remoteRank_(remoteRank),
          link_(link),
          localBuffer_(localBuffer),
          remoteBuffer_(remoteBuffer)
    {}
    string Describe() const override;

    RankId GetRemoteRank() const;
    const LinkData* GetLink() const override;
    const DataBuffer& GetLocalBuffer() const;
    const DataBuffer& GetRemoteBuffer() const;

private:
    RankId remoteRank_;
    LinkData link_;
    DataBuffer localBuffer_;
    DataBuffer remoteBuffer_;
};

class InsReadReduce : public Instruction {
public:
    InsReadReduce(
        RankId remoteRank, const LinkData& link, const DataSlice& localSlice, const DataSlice& remoteSlice,
        DataType dataType, ReduceOp reduceOp)
        : Instruction(InstructionType::READ_REDUCE),
          remoteRank_(remoteRank),
          link_(link),
          dataType_(dataType),
          reduceOp_(reduceOp),
          localSlice_(localSlice),
          remoteSlice_(remoteSlice)
    {}
    string Describe() const override;

    RankId GetRemoteRank() const;
    const LinkData* GetLink() const override;
    const DataSlice& GetLocalSlice() const;
    const DataSlice& GetRemoteSlice() const;
    const DataType GetDataType() const;
    const ReduceOp GetReduceOp() const;

private:
    RankId remoteRank_;
    LinkData link_;
    DataType dataType_;
    ReduceOp reduceOp_;
    DataSlice localSlice_;
    DataSlice remoteSlice_;
};

class InsBatchRead : public Instruction {
public:
    using Iterator = BaseConstIterator<vector, unique_ptr<Instruction>>;
    InsBatchRead(RankId remoteRank, const LinkData& link)
        : Instruction(InstructionType::BATCH_READ),
          remoteRank(remoteRank),
          link(link)
    {}
    string Describe() const override;

    RankId GetRemoteRank() const;
    const LinkData* GetLink() const override;
    Iterator Iter() const { return Iterator(readInsVec); };
    void PushReadIns(unique_ptr<Instruction> readIns);

private:
    RankId remoteRank;
    LinkData link;
    vector<unique_ptr<Instruction>> readInsVec;
};

class InsReadReduceExtend : public Instruction {
public:
    InsReadReduceExtend(
        RankId remoteRank, const LinkData& link, const DataBuffer& localBuffer, const DataBuffer& remoteBuffer,
        DataType dataType, ReduceOp reduceOp)
        : Instruction(InstructionType::READ_REDUCE_EXTEND),
          remoteRank_(remoteRank),
          link_(link),
          dataType_(dataType),
          reduceOp_(reduceOp),
          localBuffer_(localBuffer),
          remoteBuffer_(remoteBuffer)
    {}
    string Describe() const override;

    RankId GetRemoteRank() const;
    const LinkData* GetLink() const override;
    const DataBuffer& GetLocalBuffer() const;
    const DataBuffer& GetRemoteBuffer() const;
    const DataType GetDataType() const;
    const ReduceOp GetReduceOp() const;

private:
    RankId remoteRank_;
    LinkData link_;
    DataType dataType_;
    ReduceOp reduceOp_;
    DataBuffer localBuffer_;
    DataBuffer remoteBuffer_;
};

class InsWrite : public Instruction {
public:
    InsWrite(RankId remoteRank, const LinkData& link, const DataSlice& localSlice, const DataSlice& remoteSlice)
        : Instruction(InstructionType::WRITE),
          remoteRank_(remoteRank),
          link_(link),
          localSlice_(localSlice),
          remoteSlice_(remoteSlice)
    {}
    string Describe() const override;

    RankId GetRemoteRank() const;
    const LinkData* GetLink() const override;
    const DataSlice& GetLocalSlice() const;
    const DataSlice& GetRemoteSlice() const;

private:
    RankId remoteRank_;
    LinkData link_;
    DataSlice localSlice_;
    DataSlice remoteSlice_;
};

class InsWriteExtend : public Instruction {
public:
    InsWriteExtend(
        RankId remoteRank, const LinkData& link, const DataBuffer& localBuffer, const DataBuffer& remoteBuffer)
        : Instruction(InstructionType::WRITE_EXTEND),
          remoteRank_(remoteRank),
          link_(link),
          localBuffer_(localBuffer),
          remoteBuffer_(remoteBuffer)
    {}
    string Describe() const override;

    RankId GetRemoteRank() const;
    const LinkData* GetLink() const override;
    const DataBuffer& GetLocalBuffer() const;
    const DataBuffer& GetRemoteBuffer() const;

private:
    RankId remoteRank_;
    LinkData link_;
    DataBuffer localBuffer_;
    DataBuffer remoteBuffer_;
};

class InsWriteReduce : public Instruction {
public:
    InsWriteReduce(
        RankId remoteRank, const LinkData& link, const DataSlice& localSlice, const DataSlice& remoteSlice,
        DataType dataType, ReduceOp reduceOp)
        : Instruction(InstructionType::WRITE_REDUCE),
          remoteRank_(remoteRank),
          link_(link),
          dataType_(dataType),
          reduceOp_(reduceOp),
          localSlice_(localSlice),
          remoteSlice_(remoteSlice)
    {}
    string Describe() const override;

    RankId GetRemoteRank() const;
    const LinkData* GetLink() const override;
    const DataSlice& GetLocalSlice() const;
    const DataSlice& GetRemoteSlice() const;
    const DataType GetDataType() const;
    const ReduceOp GetReduceOp() const;

private:
    RankId remoteRank_;
    LinkData link_;
    DataType dataType_;
    ReduceOp reduceOp_;
    DataSlice localSlice_;
    DataSlice remoteSlice_;
};

class InsBatchWrite : public Instruction {
public:
    using Iterator = BaseConstIterator<vector, unique_ptr<Instruction>>;
    InsBatchWrite(RankId remoteRank, const LinkData& link)
        : Instruction(InstructionType::BATCH_WRITE),
          remoteRank(remoteRank),
          link(link)
    {}
    string Describe() const override;

    RankId GetRemoteRank() const;
    const LinkData* GetLink() const override;
    Iterator Iter() const { return Iterator(writeInsVec); };
    void PushWriteIns(unique_ptr<Instruction> writeIns);

private:
    RankId remoteRank;
    LinkData link;
    vector<unique_ptr<Instruction>> writeInsVec;
};

class InsWriteReduceExtend : public Instruction {
public:
    InsWriteReduceExtend(
        RankId remoteRank, const LinkData& link, const DataBuffer& localBuffer, const DataBuffer& remoteBuffer,
        DataType dataType, ReduceOp reduceOp)
        : Instruction(InstructionType::WRITE_REDUCE_EXTEND),
          remoteRank_(remoteRank),
          link_(link),
          dataType_(dataType),
          reduceOp_(reduceOp),
          localBuffer_(localBuffer),
          remoteBuffer_(remoteBuffer)
    {}
    string Describe() const override;

    RankId GetRemoteRank() const;
    const LinkData* GetLink() const override;
    const DataBuffer& GetLocalBuffer() const;
    const DataBuffer& GetRemoteBuffer() const;
    const DataType GetDataType() const;
    const ReduceOp GetReduceOp() const;

private:
    RankId remoteRank_;
    LinkData link_;
    DataType dataType_;
    ReduceOp reduceOp_;
    DataBuffer localBuffer_;
    DataBuffer remoteBuffer_;
};

class InsWriteWithFin : public Instruction {
public:
    InsWriteWithFin(
        RankId remoteRank, const LinkData& link, const DataSlice& localSlice, const DataSlice& remoteSlice,
        NotifyType notifyType = NotifyType::NORMAL, u32 bitValue = 0, u32 topicId = 0)
        : Instruction(InstructionType::WRITE_WITH_FIN),
          remoteRank_(remoteRank),
          link_(link),
          localSlice_(localSlice),
          remoteSlice_(remoteSlice),
          notifyType_(notifyType),
          bitValue_(bitValue),
          topicId_(topicId)
    {}
    string Describe() const override;

    RankId GetRemoteRank() const;
    const LinkData* GetLink() const override;
    const DataSlice& GetLocalSlice() const;
    const DataSlice& GetRemoteSlice() const;
    const NotifyType& GetNotifyType() const;
    const u32& GetTopicId() const;
    const u32& GetBitValue() const;

private:
    RankId remoteRank_;
    LinkData link_;
    DataSlice localSlice_;
    DataSlice remoteSlice_;
    NotifyType notifyType_;
    u32 bitValue_;
    u32 topicId_;
};

class InsWriteWithFinExtend : public Instruction {
public:
    InsWriteWithFinExtend(
        RankId remoteRank, const LinkData& link, const DataBuffer& localBuffer, const DataBuffer& remoteBuffer,
        NotifyType notifyType = NotifyType::NORMAL, u32 bitValue = 0, u32 topicId = 0)
        : Instruction(InstructionType::WRITE_WITH_FIN_EXTEND),
          remoteRank_(remoteRank),
          link_(link),
          localBuffer_(localBuffer),
          remoteBuffer_(remoteBuffer),
          notifyType_(notifyType),
          bitValue_(bitValue),
          topicId_(topicId)
    {}
    string Describe() const override;

    RankId GetRemoteRank() const;
    const LinkData* GetLink() const override;
    const DataBuffer& GetLocalBuffer() const;
    const DataBuffer& GetRemoteBuffer() const;
    const NotifyType& GetNotifyType() const;
    const u32& GetTopicId() const;
    const u32& GetBitValue() const;

private:
    RankId remoteRank_;
    LinkData link_;
    DataBuffer localBuffer_;
    DataBuffer remoteBuffer_;
    NotifyType notifyType_;
    u32 bitValue_;
    u32 topicId_;
};

class InsWriteReduceWithFin : public Instruction {
public:
    InsWriteReduceWithFin(
        RankId remoteRank, const LinkData& link, const DataSlice& localSlice, const DataSlice& remoteSlice,
        DataType dataType, ReduceOp reduceOp, NotifyType notifyType = NotifyType::NORMAL, u32 bitValue = 0,
        u32 topicId = 0)
        : Instruction(InstructionType::WRITE_REDUCE_WITH_FIN),
          remoteRank_(remoteRank),
          link_(link),
          dataType_(dataType),
          reduceOp_(reduceOp),
          localSlice_(localSlice),
          remoteSlice_(remoteSlice),
          notifyType_(notifyType),
          bitValue_(bitValue),
          topicId_(topicId)
    {}
    string Describe() const override;

    RankId GetRemoteRank() const;
    const LinkData* GetLink() const override;
    const DataSlice& GetLocalSlice() const;
    const DataSlice& GetRemoteSlice() const;
    const DataType GetDataType() const;
    const ReduceOp GetReduceOp() const;
    const NotifyType& GetNotifyType() const;
    const u32& GetTopicId() const;
    const u32& GetBitValue() const;

private:
    RankId remoteRank_;
    LinkData link_;
    DataType dataType_;
    ReduceOp reduceOp_;
    DataSlice localSlice_;
    DataSlice remoteSlice_;
    NotifyType notifyType_;
    u32 bitValue_;
    u32 topicId_;
};

class InsWriteReduceWithFinExtend : public Instruction {
public:
    InsWriteReduceWithFinExtend(
        RankId remoteRank, const LinkData& link, const DataBuffer& localBuffer, const DataBuffer& remoteBuffer,
        DataType dataType, ReduceOp reduceOp, NotifyType notifyType = NotifyType::NORMAL, u32 bitValue = 0,
        u32 topicId = 0)
        : Instruction(InstructionType::WRITE_REDUCE_WITH_FIN_EXTEND),
          remoteRank_(remoteRank),
          link_(link),
          dataType_(dataType),
          reduceOp_(reduceOp),
          localBuffer_(localBuffer),
          remoteBuffer_(remoteBuffer),
          notifyType_(notifyType),
          bitValue_(bitValue),
          topicId_(topicId)
    {}
    string Describe() const override;

    RankId GetRemoteRank() const;
    const LinkData* GetLink() const override;
    const DataBuffer& GetLocalBuffer() const;
    const DataBuffer& GetRemoteBuffer() const;
    const DataType GetDataType() const;
    const ReduceOp GetReduceOp() const;
    const NotifyType& GetNotifyType() const;
    const u32& GetTopicId() const;
    const u32& GetBitValue() const;

private:
    RankId remoteRank_;
    LinkData link_;
    DataType dataType_;
    ReduceOp reduceOp_;
    DataBuffer localBuffer_;
    DataBuffer remoteBuffer_;
    NotifyType notifyType_;
    u32 bitValue_;
    u32 topicId_;
};

class InsBatchOneSidedRead : public Instruction {
public:
    InsBatchOneSidedRead(
        RankId remoteRank, const LinkData& link, const vector<RmaBufSliceLite>& localSlice,
        const vector<RmtRmaBufSliceLite>& remoteSlice)
        : Instruction(InstructionType::BATCH_ONE_SIDED_READ),
          remoteRank_(remoteRank),
          link_(link),
          localSlice_(localSlice),
          remoteSlice_(remoteSlice)
    {}
    string Describe() const override;

    RankId GetRemoteRank() const;
    const LinkData* GetLink() const override;
    const vector<RmaBufSliceLite>& GetLocalSlice() const;
    const vector<RmtRmaBufSliceLite>& GetRemoteSlice() const;

private:
    RankId remoteRank_;
    LinkData link_;
    vector<RmaBufSliceLite> localSlice_;
    vector<RmtRmaBufSliceLite> remoteSlice_;
};

class InsBatchOneSidedWrite : public Instruction {
public:
    InsBatchOneSidedWrite(
        RankId remoteRank, const LinkData& link, const vector<RmaBufSliceLite>& localSlice,
        const vector<RmtRmaBufSliceLite>& remoteSlice)
        : Instruction(InstructionType::BATCH_ONE_SIDED_WRITE),
          remoteRank_(remoteRank),
          link_(link),
          localSlice_(localSlice),
          remoteSlice_(remoteSlice)
    {}
    string Describe() const override;

    RankId GetRemoteRank() const;
    const LinkData* GetLink() const override;
    const vector<RmaBufSliceLite>& GetLocalSlice() const;
    const vector<RmtRmaBufSliceLite>& GetRemoteSlice() const;

private:
    RankId remoteRank_;
    LinkData link_;
    vector<RmaBufSliceLite> localSlice_;
    vector<RmtRmaBufSliceLite> remoteSlice_;
};

class InsStreamSync : public Instruction {
public:
    InsStreamSync() : Instruction(InstructionType::STREAM_SYNC) {}
    string Describe() const override;
};

class InsAicpuReduce : public Instruction {
public:
    InsAicpuReduce(const DataSlice& srcSlice, const DataSlice& dstSlice, DataType dataType, ReduceOp reduceOp)
        : Instruction(InstructionType::AICPU_REDUCE),
          srcSlice_(srcSlice),
          dstSlice_(dstSlice),
          dataType_(dataType),
          reduceOp_(reduceOp)
    {}
    string Describe() const override;

    const DataSlice& GetSrcSlice() const;
    const DataSlice& GetDstSlice() const;
    const DataType GetDataType() const;
    const ReduceOp GetReduceOp() const;
    static void RunAicpuReduce(void* dst, u64 dstSize, void* src, u64 srcSize, DataType dataType, ReduceOp reduceOp);

private:
    template <typename T>
    static void AicpuReduceTemplate(T* dst, u64 dstSize, T* src, u64 srcSize, ReduceOp reduceOp);
    DataSlice srcSlice_;
    DataSlice dstSlice_;
    DataType dataType_;
    ReduceOp reduceOp_;
};

class InsPreStreamSync : public Instruction {
public:
    InsPreStreamSync() : Instruction(InstructionType::PRE_STREAM_SYNC) {}
    string Describe() const override;
};

} // namespace Hccl
#endif
