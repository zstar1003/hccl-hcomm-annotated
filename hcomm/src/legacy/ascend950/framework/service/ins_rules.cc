/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_rules.h"
#include "null_ptr_exception.h"
#include "not_support_exception.h"
#include "queue_wait_group_cnt_notify_manager.h"
#include "queue_bcast_post_cnt_notify_manager.h"
#include "cnt_notify_res_helper.h"
#include "data_type.h"
#include "reduce_op.h"
#include "aicpu_kernel_launcher.h"
#include "coll_service_device_mode.h"
#include "dlprof_function_v2.h"
#include "hccl_aiv_utils.h"
#include "ccu_ins_group.h"

#include <limits>

namespace Hccl {

constexpr u32 BASE_BIT = 1; // 用于左移设置二进制数的特定位
constexpr u32 TOKEN_VALUE_INDEX = 2;

template <typename INS_TYPE>
inline void VerifyDataSliceIsEqual(const INS_TYPE& ins)
{
    const DataSlice& localSlice = ins.GetLocalSlice();
    const DataSlice& remoteSlice = ins.GetRemoteSlice();

    if (localSlice.GetSize() != remoteSlice.GetSize()) {
        string msg = StringFormat("%s slice size is different", ins.Describe().c_str());
        THROW<NotSupportException>(msg);
    }
}

template <typename INS_TYPE>
inline RmaBufferSlice PrepareP2PRmaBufferSlice(const INS_TYPE& ins, CommunicatorImpl& comm)
{
    CollOperator op = *comm.GetCurrentCollOperator();
    const DataSlice& localSlice = ins.GetLocalSlice();
    auto buffer = comm.GetDataBufferManager().Get(op.opTag, localSlice.GetType());
    if (buffer == nullptr) {
        string msg = StringFormat(
            "%s DataBuffer is nullptr, opTag[%s], bufferType[%s]", op.opTag.c_str(), ins.Describe().c_str(),
            localSlice.GetType().Describe().c_str());
        THROW<NullPtrException>(msg);
    }
    u64 addrLocal = buffer->GetAddr() + localSlice.GetOffset();
    return RmaBufferSlice{.addr = addrLocal, .size = localSlice.GetSize(), .buf = nullptr};
}

template <typename INS_TYPE>
inline RmaBufferSlice PrepareRmaBufferSlice(const INS_TYPE& ins, CommunicatorImpl& comm)
{
    CollOperator op = *comm.GetCurrentCollOperator();
    const DataSlice& localSlice = ins.GetLocalSlice();
    LocalRmaBuffer* localRmaBuffer
        = comm.GetLocalRmaBufManager().Get(op.opTag, ins.GetLink()->GetLocalPort(), localSlice.GetType());
    if (localRmaBuffer == nullptr) {
        string msg = StringFormat(
            "%s LocalRmaBuffer Get is nullptr, localBufType[%u]", ins.Describe().c_str(),
            static_cast<u32>(localSlice.GetType()));
        THROW<NullPtrException>(msg);
    }
    u64 addrLocal = localRmaBuffer->GetBuf()->GetAddr() + localSlice.GetOffset();

    return RmaBufferSlice{.addr = addrLocal, .size = localSlice.GetSize(), .buf = localRmaBuffer};
}

template <typename INS_TYPE>
inline RmtRmaBufferSlice PrepareRmtRmaBufferSlice(const INS_TYPE& ins, BaseMemTransport& transport)
{
    const DataSlice& remoteSlice = ins.GetRemoteSlice();

    RemoteRmaBuffer* remoteRmaBuffer = transport.GetRmtRmaBuffer(remoteSlice.GetType());
    if (remoteRmaBuffer == nullptr) {
        string msg = StringFormat(
            "%s RemoteRmaBuffer Get is nullptr, remoteBufType[%u]", ins.Describe().c_str(),
            static_cast<u32>(remoteSlice.GetType()));
        THROW<NullPtrException>(msg);
    }
    u64 addrRemote = 0;
    if (remoteRmaBuffer != nullptr) {
        addrRemote = remoteRmaBuffer->GetAddr() + remoteSlice.GetOffset();
    }
    return RmtRmaBufferSlice{.addr = addrRemote, .size = remoteSlice.GetSize(), .buf = remoteRmaBuffer};
}

template <typename INS_TYPE>
inline BaseMemTransport* GetTransport(const INS_TYPE& ins, CommunicatorImpl& comm)
{
    CollOperator op = *comm.GetCurrentCollOperator();
    BaseMemTransport* transport = nullptr;
    if (ins.GetLink() == nullptr) {
        THROW<NullPtrException>(StringFormat("[%s] ins.GetLink() is nullptr", __func__));
    }
    if (op.opMode == OpMode::OPBASE) {
        transport = comm.GetMemTransportManager()->GetOpbasedTransport(*ins.GetLink());
    } else if (op.opMode == OpMode::OFFLOAD) {
        transport = comm.GetMemTransportManager()->GetOffloadTransport(op.opTag, *ins.GetLink());
    }
    if (transport == nullptr) {
        string msg = StringFormat(
            "%s MemTransport Get is nullptr, opTag[%s], remoteRank[%d], linkData[%s]", ins.Describe().c_str(),
            op.opTag.c_str(), ins.GetRemoteRank(), ins.GetLink()->Describe().c_str());
        THROW<NullPtrException>(msg);
    }

    return transport;
}

template <typename INS_TYPE>
inline ReduceIn GetReduceIn(const INS_TYPE& ins)
{
    return ReduceIn(ins.GetDataType(), ins.GetReduceOp());
}

template <typename INS_TYPE>
inline WithNotifyIn GetFinWithNotify(const INS_TYPE& ins, BaseMemTransport& transport)
{
    if (ins.GetNotifyType() == NotifyType::NORMAL) {
        return WithNotifyIn(TransportNotifyType::NORMAL, NOTIFY_INDEX_FIN);
    } else if (ins.GetNotifyType() == NotifyType::COUNTER) {
        auto desc = transport.GetRmtCntNotifyDesc();
        CntNotifyResHelper tool;
        u32 index = tool.GetIndex(desc, ins.GetTopicId(), NOTIFY_INDEX_FIN);
        return WithNotifyIn(TransportNotifyType::COUNT, index, ins.GetBitValue());
    } else {
        string msg = StringFormat("only support NORMAL or COUNTER notifyType, ins=%s", ins.Describe().c_str());
        MACRO_THROW(NotSupportException, msg);
    }
}

inline RtsNotify*
RtsNotifyGet(QueueNotifyManager& queueNotifyManager, QId postQid, QId waitQid, u32 topicId, const string& desc)
{
    auto* notify = queueNotifyManager.Get(postQid, waitQid, topicId);
    if (notify == nullptr) {
        string msg = StringFormat(
            "%s BaseLocalNotify Get nullptr, postQid[%u], waitQid[%u], topicId[%u]", desc.c_str(), postQid, waitQid,
            topicId);
        THROW<NullPtrException>(msg);
    }
    return notify;
}

inline RtsCntNotify* RtsCntNotifyGet(
    QueueWaitGroupCntNotifyManager& queueWaitGroupCntNotifyManager, QId waitQid, u32 topicId, const string& desc)
{
    RtsCntNotify* notify = queueWaitGroupCntNotifyManager.Get(waitQid, topicId);
    if (notify == nullptr) {
        string msg
            = StringFormat("%s RtsCntNotify Get nullptr, waitQid[%u], topicId[%u]", desc.c_str(), waitQid, topicId);
        THROW<NullPtrException>(msg);
    }
    return notify;
}

inline Rts1ToNCntNotify* Rts1ToNCntNotifyGet(
    QueueBcastPostCntNotifyManager& queueBcastPostCntNotifyManager, QId postQid, u32 topicId, const string& desc)
{
    Rts1ToNCntNotify* notify = queueBcastPostCntNotifyManager.Get(postQid, topicId);
    if (notify == nullptr) {
        string msg
            = StringFormat("%s Rts1ToNCntNotify Get nullptr, postQid[%u], topicId[%u]", desc.c_str(), postQid, topicId);
        THROW<NullPtrException>(msg);
    }
    return notify;
}

inline vector<RtsCntNotify*>
LocalCntNotifyGet(ConnLocalCntNotifyManager& connLocalCntNotifyManager, u32 topicId, const string& desc)
{
    u32 listSize = 2;
    auto notifyList = connLocalCntNotifyManager.Get(topicId);
    if (notifyList.size() != listSize || notifyList[0] == nullptr || notifyList[1] == nullptr) {
        string msg = StringFormat("%s LocalCntNotify Get nullptr, topicId[%u]", desc.c_str(), topicId);
        THROW<NullPtrException>(msg);
    }
    return notifyList;
}

static void
SaveDfxTaskInfo(const CommunicatorImpl& comm, const TaskParam& taskParam, const u32 remoteRankId, bool isMaster = false)
{
    u32 taskId;
    u32 streamId;
    HrtGetTaskIdAndStreamID(taskId, streamId);

    std::unique_ptr<TaskInfo> taskInfo = std::make_unique<TaskInfo>(
        streamId, taskId, remoteRankId, taskParam, comm.GetMirrorTaskManager().GetCurrDfxOpInfo(), isMaster);

    HCCL_INFO("Begin to AddTaskInfo: streamId[%lu], taskId[%lu], remoteRankId[%u].", streamId, taskId, remoteRankId);
    comm.GetMirrorTaskManager().AddTaskInfo(std::move(taskInfo));
}

void Interpret(
    const InsPostReady& insPostReady, CommunicatorImpl& comm, const Stream& stream,
    [[maybe_unused]] const OpTaskConfig& taskConfig)
{
    GetTransport(insPostReady, comm)->Post(NOTIFY_INDEX_READY, stream);
}

void Interpret(
    const InsWaitReady& insWaitReady, CommunicatorImpl& comm, const Stream& stream, const OpTaskConfig& taskConfig)
{
    GetTransport(insWaitReady, comm)->Wait(NOTIFY_INDEX_READY, stream, taskConfig.GetNotifyWaitTime());
}

void Interpret(
    const InsPostFin& insPostFin, CommunicatorImpl& comm, const Stream& stream,
    [[maybe_unused]] const OpTaskConfig& taskConfig)
{
    GetTransport(insPostFin, comm)->Post(NOTIFY_INDEX_FIN, stream);
}

void Interpret(
    const InsWaitFin& insWaitFin, CommunicatorImpl& comm, const Stream& stream, const OpTaskConfig& taskConfig)
{
    GetTransport(insWaitFin, comm)->Wait(NOTIFY_INDEX_FIN, stream, taskConfig.GetNotifyWaitTime());
}

void Interpret(
    const InsPostFinAck& insPostFinAck, CommunicatorImpl& comm, const Stream& stream,
    [[maybe_unused]] const OpTaskConfig& taskConfig)
{
    GetTransport(insPostFinAck, comm)->Post(NOTIFY_INDEX_FIN_ACK, stream);
}

void Interpret(
    const InsWaitGroupFin& insWaitGroupFin, CommunicatorImpl& comm, const Stream& stream,
    const OpTaskConfig& taskConfig)
{
    TaskParam taskParam{};
    taskParam.beginTime = DlProfFunction::GetInstance().dlMsprofSysCycleTime();

    auto notifyList = LocalCntNotifyGet(
        comm.GetConnLocalCntNotifyManager(), insWaitGroupFin.GetTopicId(), insWaitGroupFin.Describe());
    notifyList[NOTIFY_INDEX_FIN]->WaitValue(insWaitGroupFin.GetValue(), taskConfig.GetNotifyWaitTime(), stream);

    taskParam.taskType = TaskParamType::TASK_NOTIFY_WAIT;
    taskParam.endTime = DlProfFunction::GetInstance().dlMsprofSysCycleTime();
    taskParam.taskPara.Notify.notifyID = notifyList[NOTIFY_INDEX_FIN]->GetId();
    taskParam.taskPara.Notify.value = insWaitGroupFin.GetValue();

    SaveDfxTaskInfo(comm, taskParam, INVALID_VALUE_RANKID); // 本地填充rmt rankId， 为0xffff
}

void Interpret(
    const InsWaitFinAck& insWaitFinAck, CommunicatorImpl& comm, const Stream& stream, const OpTaskConfig& taskConfig)
{
    GetTransport(insWaitFinAck, comm)->Wait(NOTIFY_INDEX_FIN_ACK, stream, taskConfig.GetNotifyWaitTime());
}

void Interpret(const InsRead& insRead, CommunicatorImpl& comm, const Stream& stream, const OpTaskConfig& taskConfig)
{
    (void)taskConfig;
    VerifyDataSliceIsEqual(insRead);
    RmaBufferSlice locSlice;
    if (insRead.GetLink()->GetType() == PortDeploymentType::P2P) {
        locSlice = PrepareP2PRmaBufferSlice(insRead, comm);
    } else {
        locSlice = PrepareRmaBufferSlice(insRead, comm);
    }
    auto transport = GetTransport(insRead, comm);
    RmtRmaBufferSlice rmtSlice = PrepareRmtRmaBufferSlice(insRead, *transport);
    transport->Read(locSlice, rmtSlice, stream);
}

void Interpret(const InsWrite& insWrite, CommunicatorImpl& comm, const Stream& stream, const OpTaskConfig& taskConfig)
{
    (void)taskConfig;
    VerifyDataSliceIsEqual(insWrite);
    RmaBufferSlice locSlice;
    if (insWrite.GetLink()->GetType() == PortDeploymentType::P2P) {
        locSlice = PrepareP2PRmaBufferSlice(insWrite, comm);
    } else {
        locSlice = PrepareRmaBufferSlice(insWrite, comm);
    }
    auto transport = GetTransport(insWrite, comm);
    RmtRmaBufferSlice rmtSlice = PrepareRmtRmaBufferSlice(insWrite, *transport);
    transport->Write(locSlice, rmtSlice, stream);
}

void Interpret(
    const InsReadReduce& insReadReduce, CommunicatorImpl& comm, const Stream& stream,
    [[maybe_unused]] const OpTaskConfig& taskConfig)
{
    VerifyDataSliceIsEqual(insReadReduce);
    RmaBufferSlice locSlice;
    if (insReadReduce.GetLink()->GetType() == PortDeploymentType::P2P) {
        locSlice = PrepareP2PRmaBufferSlice(insReadReduce, comm);
    } else {
        locSlice = PrepareRmaBufferSlice(insReadReduce, comm);
    }
    auto transport = GetTransport(insReadReduce, comm);
    RmtRmaBufferSlice rmtSlice = PrepareRmtRmaBufferSlice(insReadReduce, *transport);
    transport->ReadReduce(locSlice, rmtSlice, GetReduceIn(insReadReduce), stream);
}

void Interpret(
    const InsWriteReduce& insWriteReduce, CommunicatorImpl& comm, const Stream& stream,
    [[maybe_unused]] const OpTaskConfig& taskConfig)
{
    VerifyDataSliceIsEqual(insWriteReduce);
    RmaBufferSlice locSlice;
    if (insWriteReduce.GetLink()->GetType() == PortDeploymentType::P2P) {
        locSlice = PrepareP2PRmaBufferSlice(insWriteReduce, comm);
    } else {
        locSlice = PrepareRmaBufferSlice(insWriteReduce, comm);
    }
    auto transport = GetTransport(insWriteReduce, comm);
    RmtRmaBufferSlice rmtSlice = PrepareRmtRmaBufferSlice(insWriteReduce, *transport);
    transport->WriteReduce(locSlice, rmtSlice, GetReduceIn(insWriteReduce), stream);
}

void Interpret(
    const InsWriteWithFin& insWriteWithFin, CommunicatorImpl& comm, const Stream& stream,
    [[maybe_unused]] const OpTaskConfig& taskConfig)
{
    VerifyDataSliceIsEqual(insWriteWithFin);
    RmaBufferSlice locSlice = PrepareRmaBufferSlice(insWriteWithFin, comm); // InsWriteWithFin当前不支持P2P
    auto transport = GetTransport(insWriteWithFin, comm);
    RmtRmaBufferSlice rmtSlice = PrepareRmtRmaBufferSlice(insWriteWithFin, *transport);
    transport->WriteWithNotify(locSlice, rmtSlice, GetFinWithNotify(insWriteWithFin, *transport), stream);
}

void Interpret(
    const InsWriteReduceWithFin& insWriteReduceWithFin, CommunicatorImpl& comm, const Stream& stream,
    [[maybe_unused]] const OpTaskConfig& taskConfig)
{
    VerifyDataSliceIsEqual(insWriteReduceWithFin);
    RmaBufferSlice locSlice = PrepareRmaBufferSlice(insWriteReduceWithFin, comm); // InsWriteReduceWithFin当前不支持P2P
    auto transport = GetTransport(insWriteReduceWithFin, comm);
    RmtRmaBufferSlice rmtSlice = PrepareRmtRmaBufferSlice(insWriteReduceWithFin, *transport);
    transport->WriteReduceWithNotify(
        locSlice, rmtSlice, GetReduceIn(insWriteReduceWithFin), GetFinWithNotify(insWriteReduceWithFin, *transport),
        stream);
}

void Interpret(
    const InsLocalPostTo& insLocalPostTo, CommunicatorImpl& comm, const Stream& stream,
    [[maybe_unused]] const OpTaskConfig& taskConfig)
{
    TaskParam taskParam{};
    taskParam.beginTime = DlProfFunction::GetInstance().dlMsprofSysCycleTime();
    u32 bitValue = BASE_BIT;
    u64 notifyID;

    if (insLocalPostTo.GetNotifyType() == NotifyType::NORMAL) {
        auto notify = RtsNotifyGet(
            comm.GetCcuQueueNotifyManager(), insLocalPostTo.GetPostQid(), insLocalPostTo.GetWaitQid(),
            insLocalPostTo.GetTopicId(), insLocalPostTo.Describe());
        notify->Post(stream);
        notifyID = notify->GetId();
    } else if (insLocalPostTo.GetNotifyType() == NotifyType::COUNTER) {
        RtsCntNotify* notify = RtsCntNotifyGet(
            comm.GetQueueWaitGroupCntNotifyManager(), insLocalPostTo.GetWaitQid(), insLocalPostTo.GetTopicId(),
            insLocalPostTo.Describe());
        auto postQid = insLocalPostTo.GetPostQid();
        bitValue = BASE_BIT << postQid;
        notify->PostBits(bitValue, stream);
        notifyID = notify->GetId();
    } else {
        string msg = StringFormat(
            "only support NORMAL or COUNTER notifyType, %s", insLocalPostTo.GetNotifyType().Describe().c_str());
        MACRO_THROW(NotSupportException, msg);
    }

    taskParam.taskType = TaskParamType::TASK_NOTIFY_RECORD;
    taskParam.endTime = DlProfFunction::GetInstance().dlMsprofSysCycleTime();
    taskParam.taskPara.Notify.notifyID = notifyID;
    taskParam.taskPara.Notify.value = bitValue;

    SaveDfxTaskInfo(comm, taskParam, INVALID_VALUE_RANKID); // 本地填充rmt rankId， 为0xffff
}

void Interpret(
    const InsLocalWaitFrom& insLocalWaitFrom, CommunicatorImpl& comm, const Stream& stream,
    const OpTaskConfig& taskConfig)
{
    TaskParam taskParam{};
    taskParam.beginTime = DlProfFunction::GetInstance().dlMsprofSysCycleTime();
    u32 bitValue = BASE_BIT;
    u64 notifyID;

    if (insLocalWaitFrom.GetNotifyType() == NotifyType::NORMAL) {
        auto notify = RtsNotifyGet(
            comm.GetCcuQueueNotifyManager(), insLocalWaitFrom.GetPostQid(), insLocalWaitFrom.GetWaitQid(),
            insLocalWaitFrom.GetTopicId(), insLocalWaitFrom.Describe());
        notify->Wait(stream, taskConfig.GetNotifyWaitTime());
        notifyID = notify->GetId();
    } else if (insLocalWaitFrom.GetNotifyType() == NotifyType::COUNTER) {
        Rts1ToNCntNotify* notify = Rts1ToNCntNotifyGet(
            comm.GetBcastPostCntNotifyManager(), insLocalWaitFrom.GetPostQid(), insLocalWaitFrom.GetTopicId(),
            insLocalWaitFrom.Describe());
        auto waitQid = insLocalWaitFrom.GetWaitQid();
        bitValue = BASE_BIT << waitQid;
        notify->WaitBits(bitValue, taskConfig.GetNotifyWaitTime(), stream);
        notifyID = notify->GetId();
    } else {
        string msg = StringFormat(
            "only support NORMAL or COUNTER notifyType, %s", insLocalWaitFrom.GetNotifyType().Describe().c_str());
        MACRO_THROW(NotSupportException, msg);
    }

    taskParam.taskType = TaskParamType::TASK_NOTIFY_WAIT;
    taskParam.endTime = DlProfFunction::GetInstance().dlMsprofSysCycleTime();
    taskParam.taskPara.Notify.notifyID = notifyID;
    taskParam.taskPara.Notify.value = bitValue;

    SaveDfxTaskInfo(comm, taskParam, INVALID_VALUE_RANKID); // 本地填充rmt rankId， 为0xffff
}

void Interpret(
    const InsLocalWaitGroup& insLocalWaitGroup, CommunicatorImpl& comm, const Stream& stream,
    const OpTaskConfig& taskConfig)
{
    TaskParam taskParam{};
    taskParam.beginTime = DlProfFunction::GetInstance().dlMsprofSysCycleTime();

    RtsCntNotify* notify = RtsCntNotifyGet(
        comm.GetQueueWaitGroupCntNotifyManager(), insLocalWaitGroup.GetWaitQid(), insLocalWaitGroup.GetTopicId(),
        insLocalWaitGroup.Describe());

    u32 value = 0;
    for (auto iter = insLocalWaitGroup.Iter(); iter.HasNext(); ++iter) {
        value |= BASE_BIT << *iter;
    }
    notify->WaitValue(value, taskConfig.GetNotifyWaitTime(), stream);

    taskParam.taskType = TaskParamType::TASK_NOTIFY_WAIT;
    taskParam.endTime = DlProfFunction::GetInstance().dlMsprofSysCycleTime();
    taskParam.taskPara.Notify.notifyID = notify->GetId();
    taskParam.taskPara.Notify.value = value;

    SaveDfxTaskInfo(comm, taskParam, INVALID_VALUE_RANKID); // 本地填充rmt rankId， 为0xffff
}

void Interpret(
    const InsLocalBcastPost& insLocalBcastPost, CommunicatorImpl& comm, const Stream& stream,
    [[maybe_unused]] const OpTaskConfig& taskConfig)
{
    TaskParam taskParam{};
    taskParam.beginTime = DlProfFunction::GetInstance().dlMsprofSysCycleTime();

    Rts1ToNCntNotify* notify = Rts1ToNCntNotifyGet(
        comm.GetBcastPostCntNotifyManager(), insLocalBcastPost.GetPostQid(), insLocalBcastPost.GetTopicId(),
        insLocalBcastPost.Describe());

    u32 value = 0;
    for (auto iter = insLocalBcastPost.Iter(); iter.HasNext(); ++iter) {
        value |= BASE_BIT << *iter;
    }
    notify->PostValue(value, stream);

    taskParam.taskType = TaskParamType::TASK_NOTIFY_WAIT;
    taskParam.endTime = DlProfFunction::GetInstance().dlMsprofSysCycleTime();
    taskParam.taskPara.Notify.notifyID = notify->GetId();
    taskParam.taskPara.Notify.value = value;

    SaveDfxTaskInfo(comm, taskParam, INVALID_VALUE_RANKID); // 本地填充rmt rankId， 为0xffff
}

void Interpret(
    const InsLocalCopy& insLocalCopy, CommunicatorImpl& comm, const Stream& stream,
    [[maybe_unused]] const OpTaskConfig& taskConfig)
{
    if (insLocalCopy.GetSrcSlice().GetSize() == 0) {
        return;
    }

    TaskParam taskParam{};
    taskParam.beginTime = DlProfFunction::GetInstance().dlMsprofSysCycleTime();

    auto dstBuffer = comm.GetCurrentCollOperator()->GetBuffer(insLocalCopy.GetDstSlice().GetType());
    if (dstBuffer == nullptr) {
        THROW<NullPtrException>(StringFormat("LocalCopy Interpret dstBuffer ptr is null"));
    }
    auto srcBuffer = comm.GetCurrentCollOperator()->GetBuffer(insLocalCopy.GetSrcSlice().GetType());
    if (srcBuffer == nullptr) {
        THROW<NullPtrException>(StringFormat("LocalCopy Interpret srcBuffer ptr is null"));
    }
    void* dst = reinterpret_cast<void*>(dstBuffer->GetAddr() + insLocalCopy.GetDstSlice().GetOffset());
    void* src = reinterpret_cast<void*>(srcBuffer->GetAddr() + insLocalCopy.GetSrcSlice().GetOffset());

    HrtMemAsyncCopy(
        dst, insLocalCopy.GetDstSlice().GetSize(), src, insLocalCopy.GetSrcSlice().GetSize(),
        ACL_MEMCPY_DEVICE_TO_DEVICE, stream.GetPtr());

    taskParam.taskType = TaskParamType::TASK_SDMA;
    taskParam.endTime = DlProfFunction::GetInstance().dlMsprofSysCycleTime();

    taskParam.taskPara.DMA.src = src;
    taskParam.taskPara.DMA.dst = dst;
    taskParam.taskPara.DMA.size = insLocalCopy.GetSrcSlice().GetSize();
    taskParam.taskPara.DMA.notifyID = 0; // 填充无效值
    taskParam.taskPara.DMA.linkType = DfxLinkType::ONCHIP;
    taskParam.taskPara.DMA.dmaOp = DmaOp::HCCL_DMA_READ;

    SaveDfxTaskInfo(comm, taskParam, comm.GetMyRank());
}

inline void CheckLocalReduceIns(const InsLocalReduce& ins)
{
    if (ins.GetDataType() == DataType::INT64) {
        THROW<InvalidParamsException>(StringFormat(
            "%s LocalReduce SDMAInlineReduce does not support INT64, "
            "need use TBE reduce.",
            __func__));
    }
}

void Interpret(
    const InsLocalReduce& insLocalReduce, CommunicatorImpl& comm, const Stream& stream,
    [[maybe_unused]] const OpTaskConfig& taskConfig)
{
    HCCL_INFO("%s Instruction %s", __func__, insLocalReduce.Describe().c_str());
    // SDMA支持的Reduce，则使用 sdmaReduce
    // SDMA不支持的Reduce，则使用 TBE算子(Asend C算子）

    if (insLocalReduce.GetSrcSlice().GetSize() == 0) {
        HCCL_WARNING("%s InsLocalReduce srcSlice size is 0, return", __func__);
        return;
    }

    if (insLocalReduce.GetSrcSlice().GetSize() != insLocalReduce.GetDstSlice().GetSize()) {
        HCCL_WARNING("%s InsLocalReduce srcSlice size is not equal to dstSlice size, return", __func__);
        return;
    }

    CheckLocalReduceIns(insLocalReduce);

    TaskParam taskParam{};
    taskParam.beginTime = DlProfFunction::GetInstance().dlMsprofSysCycleTime();

    auto dstBuffer = comm.GetCurrentCollOperator()->GetBuffer(insLocalReduce.GetDstSlice().GetType());
    if (dstBuffer == nullptr) {
        THROW<NullPtrException>(StringFormat("LocalReduce Interpret dstBuffer ptr is null"));
    }
    auto srcBuffer = comm.GetCurrentCollOperator()->GetBuffer(insLocalReduce.GetSrcSlice().GetType());
    if (srcBuffer == nullptr) {
        THROW<NullPtrException>(StringFormat("LocalReduce Interpret srcBuffer ptr is null"));
    }
    void* dst = reinterpret_cast<void*>(dstBuffer->GetAddr() + insLocalReduce.GetDstSlice().GetOffset());
    void* src = reinterpret_cast<void*>(srcBuffer->GetAddr() + insLocalReduce.GetSrcSlice().GetOffset());

    ReduceIn reduceIn(insLocalReduce.GetDataType(), insLocalReduce.GetReduceOp());

    aclrtReduceKind rtReduceOp
        = static_cast<aclrtReduceKind>(static_cast<int>(RtReduceOpGet(insLocalReduce.GetReduceOp())));
    aclDataType rtDataType = static_cast<aclDataType>(static_cast<int>(RtDataTypeGet(insLocalReduce.GetDataType())));
    HrtReduceAsync(
        dst, insLocalReduce.GetDstSlice().GetSize(), src, insLocalReduce.GetSrcSlice().GetSize(), rtReduceOp,
        rtDataType, stream.GetPtr());

    taskParam.taskType = TaskParamType::TASK_REDUCE_INLINE;
    taskParam.endTime = DlProfFunction::GetInstance().dlMsprofSysCycleTime();

    taskParam.taskPara.Reduce.src = src;
    taskParam.taskPara.Reduce.dst = dst;
    taskParam.taskPara.Reduce.size = insLocalReduce.GetSrcSlice().GetSize();
    taskParam.taskPara.Reduce.notifyID = INVALID_VALUE_NOTIFYID;
    taskParam.taskPara.Reduce.linkType = DfxLinkType::ONCHIP;
    taskParam.taskPara.Reduce.dataType = DataTypeToHcclDataType(insLocalReduce.GetDataType());
    taskParam.taskPara.Reduce.reduceOp = ReduceOpToHcclReduceOp(insLocalReduce.GetReduceOp());

    SaveDfxTaskInfo(comm, taskParam, comm.GetMyRank());
}

static void
LaunchCcuTasks(vector<CcuTaskParam> params, const Stream* stream, TaskParam& taskParam, const OpTaskConfig& taskConfig)
{
    taskParam.beginTime = DlProfFunction::GetInstance().dlMsprofSysCycleTime();

    for (auto it = params.begin(); it != params.end(); ++it) {
        rtCcuTaskInfo_t taskInfo{};
        taskInfo.dieId = it->dieId;
        taskInfo.missionId = it->missionId;
        taskInfo.instStartId = it->instStartId;
        taskInfo.instCnt = it->instCnt;
        taskInfo.key = it->key;
        taskInfo.argSize = it->argSize;
        taskInfo.timeout = taskConfig.GetNotifyWaitTime() >= std::numeric_limits<u16>::max() ?
                               std::numeric_limits<u16>::max() :
                               static_cast<u16>(taskConfig.GetNotifyWaitTime());
        std::copy(std::begin(it->args), std::end(it->args), std::begin(taskInfo.args));

        HCCL_INFO(
            "start ccu task, dieId[%u] missionId[%u] instStartId[%u] instCnt[%u], argSize[%u], timeout[%u]s",
            taskInfo.dieId, taskInfo.missionId, taskInfo.instStartId, taskInfo.instCnt, taskInfo.argSize,
            taskInfo.timeout);

        for (std::size_t i = 0; i < taskInfo.argSize; i++) { // args 大小为 13
            if (i == TOKEN_VALUE_INDEX) {
                continue;
            }
            HCCL_INFO("arg[%lu] = %lu", i, taskInfo.args[i]);
        }
        HrtCcuLaunch(taskInfo, stream->GetPtr());
    }
    taskParam.endTime = DlProfFunction::GetInstance().dlMsprofSysCycleTime();
}

static void ReportCcuProfilingInfo(
    uint64_t execId, std::vector<CcuProfilingInfo>& streamProfilingInfo, const CommunicatorImpl& comm,
    TaskParam& taskParam, bool isMaster)
{
    if (streamProfilingInfo.empty()) {
        HCCL_INFO("There is no ccu profiling info.");
        return;
    }
    taskParam.taskPara.Ccu.dieId = streamProfilingInfo[0].dieId;
    taskParam.taskPara.Ccu.missionId = streamProfilingInfo[0].missionId;
    taskParam.taskPara.Ccu.execMissionId = streamProfilingInfo[0].missionId;
    taskParam.taskPara.Ccu.instrId = streamProfilingInfo[0].instrId;
    taskParam.taskPara.Ccu.executeId = execId;

    CcuJettyMgr* ccuJettyMgr = dynamic_cast<CollServiceDeviceMode*>(comm.GetCollService())
                                   ->GetCcuInsPreprocessor()
                                   ->GetCcuComm()
                                   ->GetCcuJettyMgr();
    for (auto& profInfo : streamProfilingInfo) {
        for (int idx = 0; idx < CCU_MAX_CHANNEL_NUM; idx++) {
            if (profInfo.channelId[idx] == INVALID_VALUE_CHANNELID) {
                break;
            }
            profInfo.remoteRankId[idx]
                = ccuJettyMgr->GetRemoteRankIdByChannelId(profInfo.dieId, profInfo.channelId[idx]);
        }
    }
    taskParam.ccuDetailInfo = std::make_shared<std::vector<CcuProfilingInfo>>(streamProfilingInfo);
    HCCL_INFO("Begin to SaveDfxTaskInfo. taskType[%d]", static_cast<int32_t>(TaskParamType::TASK_CCU));
    SaveDfxTaskInfo(comm, taskParam, INVALID_VALUE_RANKID, isMaster);
}

static void GetCcuProfilingInfo(
    const CcuInstruction& ccuInstruction, const vector<vector<CcuTaskParam>>& ccuParams,
    std::vector<std::vector<CcuProfilingInfo>>& ccuProfilingInfo)
{
    HcclResult res = CcuCtxMgr::GetProfilingInfo(
        HrtGetDevice(), *(ccuInstruction.GetTaskArg()), ccuInstruction.GetExecId(), ccuProfilingInfo);
    if (res != HcclResult::HCCL_SUCCESS) {
        string msg = StringFormat("Get ccu profiling info failed, res[%d]", res);
        THROW<NotSupportException>(msg);
    }
    if (ccuProfilingInfo.size() != ccuParams.size()) {
        string msg
            = StringFormat("Get ccu profiling info size error(%u-%u).", ccuProfilingInfo.size(), ccuParams.size());
        THROW<NotSupportException>(msg);
    }
}

static void FastLoadSaveParams(
    const CcuInstruction& ccuInstruction, CommunicatorImpl& comm, [[maybe_unused]] const OpTaskConfig& taskConfig,
    const Stream& stream, std::vector<std::vector<CcuTaskParam>>& ccuParams,
    std::vector<std::vector<CcuProfilingInfo>>& ccuProfilingInfo)
{
    std::size_t totalSize = 0;
    for (const auto& ccuParam : ccuParams) {
        totalSize += ccuParam.size();
    }
    if (totalSize != 0 && comm.isEnableSuperFasterLoad()) {
        CcuInstType insType = ccuInstruction.GetInstType();
        if (ccuInstruction.GetInstType() == CcuInstType::CCU_INS_GROUP) {
            const CcuInsGroup* insGroup = dynamic_cast<const CcuInsGroup*>(&ccuInstruction);
            if (insGroup == nullptr) {
                THROW<NullPtrException>(StringFormat("%s CcuInsGroup trans failed", __func__));
            }
            if (insGroup->GetCcuInstructions().empty()) {
                THROW<InvalidParamsException>(StringFormat("%s insGroup CcuInstructions isEmpty", __func__));
            }
            insType = insGroup->GetCcuInstructions()[0]->GetInstType();
        }
        HCCL_RUN_INFO("current CcuInstType: %d", static_cast<int>(insType));
        comm.saveCCUParams(
            std::move(ccuParams), std::move(ccuProfilingInfo), ccuInstruction.GetExecId(), insType,
            stream.GetId() != comm.GetStreamManager().GetMaster()->GetId());
    }
}

void SubmitCcuInsGroupTasks(
    const CcuInstruction& ccuInstruction, CommunicatorImpl& comm, const OpTaskConfig& taskConfig, const Stream& stream,
    std::vector<std::vector<CcuTaskParam>>& ccuParams)
{
    TaskParam taskParam = {};
    taskParam.taskType = TaskParamType::TASK_CCU;
    std::vector<std::vector<CcuProfilingInfo>> ccuProfilingInfo;
    GetCcuProfilingInfo(ccuInstruction, ccuParams, ccuProfilingInfo);

    u32 timeout = taskConfig.GetNotifyWaitTime();
    u32 reqStreamNum = ccuParams.size() - 1;
    u32 value = 0;
    for (u32 i = 0; i < reqStreamNum; ++i) {
        value |= BASE_BIT << i;
    }

    // launch LocalPostTo on stream
    Rts1ToNCntNotify* cntNotify1ToN = comm.GetCcuStreamSyncNotifyManager().GetRts1ToNCntNotify(stream.GetId());
    if (cntNotify1ToN == nullptr) {
        HCCL_ERROR("[SubmitCcuInsGroupTasks] GetRts1ToNCntNotify returned nullptr");
        return;
    }
    cntNotify1ToN->PostValue(value, stream);

    // launch ccu task
    LaunchCcuTasks(*ccuParams.begin(), &stream, taskParam, taskConfig);
    ReportCcuProfilingInfo(ccuInstruction.GetExecId(), ccuProfilingInfo[0], comm, taskParam, stream.IsMaster());

    // launch LocalWaitFrom on stream
    RtsCntNotify* cntNotifyNTo1 = comm.GetCcuStreamSyncNotifyManager().GetRtsNTo1CntNotify(stream.GetId());
    if (cntNotifyNTo1 == nullptr) {
        HCCL_ERROR("[SubmitCcuInsGroupTasks] GetRtsNTo1CntNotify returned nullptr");
        return;
    }
    cntNotifyNTo1->WaitValue(value, timeout, stream);

    auto& streamMgr = comm.GetStreamManager();
    // 查询当前从流持有的子从流
    auto streamIndex = streamMgr.GetStreamIndex(stream.GetId());
    auto& candidateSubSlaveStreamIndexes = streamMgr.GetSubSlaveIndexes(streamIndex);
    for (u32 ccuProfIdx = 1; ccuProfIdx <= reqStreamNum; ++ccuProfIdx) {
        Stream* slave;
        if (ccuProfIdx > candidateSubSlaveStreamIndexes.size()) {
            // 子从流不足，添加(主)从流->(子)从流对应关系, 并创建流
            streamMgr.RegisterBucket(streamIndex, streamMgr.GetSlaveIndex());
            slave = streamMgr.GetSlave();
        } else {
            slave = streamMgr.GetSlaveByIndex(candidateSubSlaveStreamIndexes[ccuProfIdx - 1]);
        }

        // 捕获slaveStream
        auto masterStream = comm.GetStreamManager().GetMaster();
        comm.GetStreamManager().CaptureSlaveStream(masterStream, slave); // 捕获slaveStream
        u32 bitValue = BASE_BIT << (ccuProfIdx - 1);
        cntNotify1ToN->WaitBits(bitValue, timeout, *slave);

        // launch ccu task
        LaunchCcuTasks(ccuParams[ccuProfIdx], slave, taskParam, taskConfig);
        ReportCcuProfilingInfo(
            ccuInstruction.GetExecId(), ccuProfilingInfo[ccuProfIdx], comm, taskParam, slave->IsMaster());

        // launch localPostTo on extra streams
        cntNotifyNTo1->PostBits(bitValue, *slave);
    }
    FastLoadSaveParams(ccuInstruction, comm, taskConfig, stream, ccuParams, ccuProfilingInfo);
}

static void SubmitCcuTasks(
    const CcuInstruction& ccuInstruction, CommunicatorImpl& comm, const OpTaskConfig& taskConfig, const Stream& stream)
{
    std::vector<std::vector<CcuTaskParam>> ccuParams;
    ccuInstruction.Translate(ccuParams);
    if (ccuParams.size() == 0) {
        HCCL_INFO("There is no ccu mission ccuParams.");
        return;
    }

    if (ccuParams.size() > 1) {
        SubmitCcuInsGroupTasks(ccuInstruction, comm, taskConfig, stream, ccuParams);
        return;
    }

    TaskParam taskParam = {};
    taskParam.taskType = TaskParamType::TASK_CCU;
    std::vector<std::vector<CcuProfilingInfo>> ccuProfilingInfo;
    GetCcuProfilingInfo(ccuInstruction, ccuParams, ccuProfilingInfo);

    // esl 2die适配，先申请从流再启动task
    LaunchCcuTasks(*ccuParams.begin(), &stream, taskParam, taskConfig);
    ReportCcuProfilingInfo(ccuInstruction.GetExecId(), ccuProfilingInfo[0], comm, taskParam, stream.IsMaster());
    FastLoadSaveParams(ccuInstruction, comm, taskConfig, stream, ccuParams, ccuProfilingInfo);
}

void Interpret(
    const CcuInstruction& ccuInstruction, CommunicatorImpl& comm, const Stream& stream, const OpTaskConfig& taskConfig)
{
    SubmitCcuTasks(ccuInstruction, comm, taskConfig, stream);
}

void Interpret(
    const AicpuInstruction& aicpuInstruction, CommunicatorImpl& comm, const Stream& stream,
    const OpTaskConfig& taskConfig)
{
    (void)taskConfig;

    AicpuKernelLauncher aicpuKernelLauncher(comm);
    aicpuKernelLauncher.AicpuKernelLaunch(stream, aicpuInstruction.GetAlgName());
}

static void ReportAivTaskInfo(const CommunicatorImpl& comm, AivOpArgs& aivOpArgs, bool isMaster)
{
    HCCL_DEBUG("Begin to SaveAivDfxTaskInfo taskType[%d]", static_cast<int32_t>(TaskParamType::TASK_AIV));
    // flagMem每个stream的中的任务复用，异常时只有最后一个task的信息
    TaskParam taskParam = {
        .taskType  = TaskParamType::TASK_AIV,
        .beginTime = aivOpArgs.beginTime,
        .endTime   = DlProfFunction::GetInstance().dlMsprofSysCycleTime(),
        .aicpuTaskId = 0,
        .npuDevId = 0,
        .isMaster = isMaster,
        .tid = 0,
        .taskPara  = {
            .Aiv = {
                    .cmdType     = aivOpArgs.cmdType,
                    .tag         = aivOpArgs.aivTag,
                    .count       = aivOpArgs.count,
                    .numBlocks    = aivOpArgs.numBlocks,
                    .rankSize    = aivOpArgs.rankSize,
                    .flagMem     = aivOpArgs.isOpBase ? reinterpret_cast<void *>(comm.GetAivTagBuffer()->GetAddr() + AIV_FLAG_ADDR_OFFSET):
                                           reinterpret_cast<void *>(comm.GetAivOffloadTagBuffer()->GetAddr() + AIV_FLAG_ADDR_OFFSET),
                    .flagMemSize = AIV_FLAG_AREA_SIZE,
                    .rank        = aivOpArgs.rank,
                    .sendRecvRemoteRank = aivOpArgs.sendRecvRemoteRank,
                    .dataType    = DataTypeToHcclDataType(aivOpArgs.dataType),
            }
        },
        .ccuDetailInfo  = nullptr
    };

    SaveDfxTaskInfo(comm, taskParam, INVALID_VALUE_RANKID, isMaster);
}

void Interpret(
    const AivInstruction& aivInstruction, const CommunicatorImpl& comm, const Stream& stream,
    const OpTaskConfig& taskConfig)
{
    (void)taskConfig;
    AivOpArgs aivOpArgs;
    aivInstruction.GetAivInsArgs(aivOpArgs);

    aivOpArgs.stream = stream.GetPtr();

    aivOpArgs.aivTag = aivOpArgs.isOpBase ? (static_cast<uint32_t>(comm.GetAivTag()) << AIV_TAG_MOVE_LEFT_BITS)
                                                | static_cast<uint32_t>(aivOpArgs.aivTag) :
                                            (static_cast<uint32_t>(comm.GetAivOffloadTag()) << AIV_TAG_MOVE_LEFT_BITS)
                                                | static_cast<uint32_t>(aivOpArgs.aivTag);
    HCCL_INFO("%s AivTag[%u]", __func__, aivOpArgs.aivTag);
    void* buffersInAddr = aivOpArgs.isOpBase ? reinterpret_cast<void*>(comm.GetAivTagBuffer()->GetAddr()) :
                                               reinterpret_cast<void*>(comm.GetAivOffloadTagBuffer()->GetAddr());
    aivOpArgs.buffersIn = buffersInAddr;

    if ((aivOpArgs.aivTag & AIV_LOW_16_BITS) == 1 && (aivOpArgs.aivTag >> AIV_TAG_MOVE_LEFT_BITS) == 1) {
        void* buffersInAddrSrc;
        u64 buffersIn[MAX_RANK_SIZE_] = {};
        buffersIn[comm.GetMyRank()] = comm.GetCclBuffer()->GetAddr();
        auto ubMemLink2TransportMap = comm.GetUbMemoryTransportMgr()->GetRmtRankId2RmtIpcRmaBufList();
        for (auto ubMemLink2TransportIter : ubMemLink2TransportMap) {
            auto rmtRank = ubMemLink2TransportIter.first;
            auto rmtMemBuffer = ubMemLink2TransportIter.second->GetAddr();
            buffersIn[rmtRank] = rmtMemBuffer;
        }
        HrtMemcpy(
            buffersInAddr, MAX_RANK_SIZE_ * sizeof(uint64_t), buffersIn, MAX_RANK_SIZE_ * sizeof(uint64_t),
            RT_MEMCPY_HOST_TO_DEVICE);
        u64 buffersOut[MAX_RANK_SIZE_] = {};
        auto ubMemLink2TransportMap_ = aivOpArgs.isOpBase ?
                                           comm.GetUbMemoryTransportMgr()->GetAllRankId2AivTagBufAddrList() :
                                           comm.GetUbMemoryTransportMgr()->GetAllRankId2AivOffloadTagBufAddrList();
        for (auto ubMemLink2TransportIter : ubMemLink2TransportMap_) {
            auto rmtRank = ubMemLink2TransportIter.first;
            auto rmtMemBuffer = ubMemLink2TransportIter.second;
            buffersOut[rmtRank] = rmtMemBuffer;
        }
        buffersInAddr = aivOpArgs.isOpBase ?
                            reinterpret_cast<void*>(comm.GetAivTagBuffer()->GetAddr() + AIV_TAG_ADDR_OFFSET) :
                            reinterpret_cast<void*>(comm.GetAivOffloadTagBuffer()->GetAddr() + AIV_TAG_ADDR_OFFSET);
        HrtMemcpy(
            buffersInAddr, MAX_RANK_SIZE_ * sizeof(uint64_t), buffersOut, MAX_RANK_SIZE_ * sizeof(uint64_t),
            RT_MEMCPY_HOST_TO_DEVICE);

        buffersInAddr = aivOpArgs.isOpBase ?
                            reinterpret_cast<void*>(comm.GetAivTagBuffer()->GetAddr() + AIV_FLAG_ADDR_OFFSET) :
                            reinterpret_cast<void*>(comm.GetAivOffloadTagBuffer()->GetAddr() + AIV_FLAG_ADDR_OFFSET);
        buffersInAddrSrc
            = aivOpArgs.isOpBase ?
                  reinterpret_cast<void*>(comm.GetAivTagBuffer()->GetAddr() + AIV_FLAG_CLEAR_OFFSET) :
                  reinterpret_cast<void*>(comm.GetAivOffloadTagBuffer()->GetAddr() + AIV_FLAG_CLEAR_OFFSET);
        bool isAivClearEnable = comm.GetAivClearEnable();
        if (isAivClearEnable) {
            HrtMemAsyncCopy(
                buffersInAddr, AIV_FLAG_AREA_SIZE, buffersInAddrSrc, AIV_FLAG_AREA_SIZE, ACL_MEMCPY_DEVICE_TO_DEVICE,
                stream.GetPtr());
        }
    }

    if (comm.GetCurrentCollOperator()->inputMem == nullptr) {
        HCCL_INFO("%s comm.GetCurrentCollOperator()->inputMem is nullptr", __func__);
    } else {
        u64 localInputAddr = static_cast<uint64_t>(comm.GetCurrentCollOperator()->inputMem->GetAddr());
        aivOpArgs.input += localInputAddr;
    }

    if (comm.GetCurrentCollOperator()->outputMem == nullptr) {
        HCCL_INFO("%s comm.GetCurrentCollOperator()->outputMem is nullptr", __func__);
    } else {
        u64 localOutputAddr = static_cast<uint64_t>(comm.GetCurrentCollOperator()->outputMem->GetAddr());
        aivOpArgs.output += localOutputAddr;
    }
    aivOpArgs.beginTime = DlProfFunction::GetInstance().dlMsprofSysCycleTime();
    ExecuteKernelLaunch(aivOpArgs);
    ReportAivTaskInfo(comm, aivOpArgs, stream.IsMaster());
}

} // namespace Hccl
