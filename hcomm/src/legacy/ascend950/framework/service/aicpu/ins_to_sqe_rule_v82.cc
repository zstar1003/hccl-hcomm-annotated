/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <regex>
#include "ins_to_sqe_rule.h"
#include "null_ptr_exception.h"
#include "internal_exception.h"
#include "invalid_params_exception.h"
#include "mem_transport_lite.h"
#include "sal.h"
#include "task_info.h"
#include "lite_res_mgr_fetcher.h"
#include "timeout_exception.h"
#include "not_support_exception.h"

namespace Hccl {

constexpr u32 BASE_BIT = 1; // 用于左移设置二进制数的特定位

template <typename INS_TYPE>
MemTransportLite& GetTransportLite(const INS_TYPE& ins, ResMgrFetcher* resMgrFetcher)
{
    if (ins.GetLink() == nullptr) {
        THROW<NullPtrException>(StringFormat("%s ins.GetLink() is nullptr", ins.Describe().c_str()));
    }
    MemTransportLite* transport = nullptr;
    if (resMgrFetcher->GetCurrentOp().opMode == OpMode::OPBASE) {
        transport = resMgrFetcher->GetTransportLiteMgr()->GetOpbase(*ins.GetLink()); // 单算子，采用 GetOpBase
    } else if (resMgrFetcher->GetCurrentOp().opMode == OpMode::OFFLOAD) {
        // 图下沉算子，需要采用 GetOffload(opTag, linkData) 获取transport
        transport
            = resMgrFetcher->GetTransportLiteMgr()->GetOffload(resMgrFetcher->GetCurrentOp().opTag, *ins.GetLink());
    }

    if (UNLIKELY(transport == nullptr)) {
        string msg = StringFormat(
            "%s MemTransportLite Get is nullptr, remoteRank[%d], linkData[%s]", ins.Describe().c_str(),
            ins.GetRemoteRank(), ins.GetLink()->Describe().c_str());
        THROW<NullPtrException>(msg);
    }
    return *transport;
}

template <typename INS_TYPE>
RmaBufferLite GetLocRmaBufferLite(const INS_TYPE& ins, ResMgrFetcher* resMgrFetcher)
{
    auto lite = resMgrFetcher->GetRmaBufferLite(ins.GetLocalSlice().GetType());
    if (UNLIKELY(lite == nullptr)) {
        string msg = StringFormat("[%s] lite Get nullptr", __func__);
        THROW<NullPtrException>(msg);
    }
    Buffer buf(lite->GetAddr(), lite->GetSize());
    auto range = buf.Range(ins.GetLocalSlice().GetOffset(), ins.GetLocalSlice().GetSize());
    return RmaBufferLite(range.GetAddr(), range.GetSize(), lite->GetTokenId(), lite->GetTokenValue());
}

template <typename INS_TYPE>
Buffer GetRmtBuffer(const INS_TYPE& ins, MemTransportLite& transport, ResMgrFetcher* resMgrFetcher)
{
    (void)resMgrFetcher;
    auto buf = transport.GetRmtBuffer(ins.GetRemoteSlice().GetType());
    return buf.Range(ins.GetRemoteSlice().GetOffset(), ins.GetRemoteSlice().GetSize());
}

template <typename INS_TYPE>
NotifyLite& GetNotifyLite(const INS_TYPE& ins, ResMgrFetcher* resMgrFetcher)
{
    auto notify = resMgrFetcher->GetQueueNotifyLiteMgr()->Get(ins.GetPostQid(), ins.GetWaitQid(), ins.GetTopicId());
    if (UNLIKELY(notify == nullptr)) {
        string msg = StringFormat(
            "%s NotifyLite Get nullptr, postQid[%d], waitQid[%d], topicId[%d]", ins.Describe().c_str(),
            ins.GetPostQid(), ins.GetWaitQid(), ins.GetTopicId());
        THROW<NullPtrException>(msg);
    }
    return *notify;
}

template <typename INS_TYPE>
Cnt1tonNotifyLite& GetCnt1toNNotifyLite(const INS_TYPE& ins, ResMgrFetcher* resMgrFetcher)
{
    auto notify = resMgrFetcher->GetCnt1tonNotifyLiteMgr()->Get(ins.GetPostQid(), ins.GetTopicId());
    if (UNLIKELY(notify == nullptr)) {
        string msg = StringFormat(
            "%s Cnt1tonNotifyLite Get nullptr, postQid[%d], topicId[%d]", ins.Describe().c_str(), ins.GetPostQid(),
            ins.GetTopicId());
        THROW<NullPtrException>(msg);
    }
    return *notify;
}

template <typename INS_TYPE>
CntNto1NotifyLite& GetCntNto1NotifyLite(const INS_TYPE& ins, ResMgrFetcher* resMgrFetcher)
{
    auto notify = resMgrFetcher->GetCntNto1NotifyLiteMgr()->Get(ins.GetWaitQid(), ins.GetTopicId());
    if (UNLIKELY(notify == nullptr)) {
        string msg = StringFormat(
            "%s CntNto1NotifyLite Get nullptr, waitQid[%d], topicId[%d]", ins.Describe().c_str(), ins.GetWaitQid(),
            ins.GetTopicId());
        THROW<NullPtrException>(msg);
    }
    return *notify;
}

void Interpret(const InsLocalPostTo& ins, const StreamLite& stream, ResMgrFetcher* resMgrFetcher)
{
    auto taskId = stream.GetRtsq()->GetTaskId();
    u32 value = 1;
    u32 notifyId = 0;
    if (ins.GetNotifyType() == NotifyType::NORMAL) {
        auto& notify = GetNotifyLite(ins, resMgrFetcher);
        notifyId = notify.GetId();
        stream.GetRtsq()->NotifyRecordLoc(notify.GetId());
    } else if (ins.GetNotifyType() == NotifyType::COUNTER) {
        auto& notify = GetCntNto1NotifyLite(ins, resMgrFetcher);
        notifyId = notify.GetId();
        value = BASE_BIT << (ins.GetPostQid());
        stream.GetRtsq()->CntNto1NotifyRecord(notify.GetId(), value);
    } else {
        std::string msg
            = StringFormat("only support NORMAL or COUNTER notifyType, %s", ins.GetNotifyType().Describe().c_str());
        MACRO_THROW(NotSupportException, msg);
    }

    TaskParam taskParam{};
    taskParam.taskType = TaskParamType::TASK_NOTIFY_RECORD;
    taskParam.beginTime = ProfGetCurCpuTimestamp();
    taskParam.taskPara.Notify.notifyID = notifyId;
    taskParam.taskPara.Notify.value = value;
    auto taskInfo = std::make_unique<TaskInfo>(stream.GetSqId(), taskId, INVALID_VALUE_RANKID, taskParam);
    resMgrFetcher->GetMirrorTaskMgrLite()->AddTaskInfo(std::move(taskInfo));
}

void Interpret(const InsLocalWaitFrom& ins, const StreamLite& stream, ResMgrFetcher* resMgrFetcher)
{
    auto taskId = stream.GetRtsq()->GetTaskId();
    u32 value = 1;
    u32 notifyId = 0;
    if (ins.GetNotifyType() == NotifyType::NORMAL) {
        auto& notify = GetNotifyLite(ins, resMgrFetcher);
        notifyId = notify.GetId();
        stream.GetRtsq()->NotifyWait(notify.GetId());
    } else if (ins.GetNotifyType() == NotifyType::COUNTER) {
        auto& notify = GetCnt1toNNotifyLite(ins, resMgrFetcher);
        notifyId = notify.GetId();
        value = BASE_BIT << (ins.GetWaitQid());
        stream.GetRtsq()->Cnt1toNNotifyWait(notify.GetId(), value);
    } else {
        std::string msg
            = StringFormat("only support NORMAL or COUNTER notifyType, %s", ins.GetNotifyType().Describe().c_str());
        MACRO_THROW(NotSupportException, msg);
    }

    TaskParam taskParam{};
    taskParam.taskType = TaskParamType::TASK_NOTIFY_WAIT;
    taskParam.beginTime = ProfGetCurCpuTimestamp();
    taskParam.taskPara.Notify.notifyID = notifyId;
    taskParam.taskPara.Notify.value = value;
    auto taskInfo = std::make_unique<TaskInfo>(stream.GetSqId(), taskId, INVALID_VALUE_RANKID, taskParam);
    resMgrFetcher->GetMirrorTaskMgrLite()->AddTaskInfo(std::move(taskInfo));
}

void Interpret(const InsLocalCopy& ins, const StreamLite& stream, ResMgrFetcher* resMgrFetcher)
{
    if (UNLIKELY(ins.GetSrcSlice().GetSize() == 0)) {
        return;
    }

    // 传入数据大小不能超过 u32最大值， 需要进行切分
    u64 u32Max = UINT32_MAX;
    double countSplitingTimes = static_cast<double>(ins.GetSrcSlice().GetSize()) / static_cast<double>(u32Max);
    u64 splitingTimes = static_cast<int>(std::ceil(countSplitingTimes));
    u64 src = resMgrFetcher->GetRmaBufferLite(ins.GetSrcSlice().GetType())->GetAddr() + ins.GetSrcSlice().GetOffset();
    u64 dst = resMgrFetcher->GetRmaBufferLite(ins.GetDstSlice().GetType())->GetAddr() + ins.GetDstSlice().GetOffset();
    u64 blockSize = u32Max;
    u64 offset = u32Max;
    for (u64 i = 0; i < splitingTimes; i++) {
        // 处理尾块数据
        if (i == splitingTimes - 1) {
            blockSize = ins.GetSrcSlice().GetSize() - u32Max * (splitingTimes - 1);
            offset = 0;
        }

        auto taskId = stream.GetRtsq()->GetTaskId();
        stream.GetRtsq()->SdmaCopy(src, dst, blockSize, 0); // 待确认， PART_ID是否固定设置为 0
        HCCL_INFO("InsLocalCopy srcA:0x%llx dstA:0x%llx,size=0x%llx", src, dst, blockSize);
        TaskParam taskParam{};
        taskParam.taskType = TaskParamType::TASK_SDMA;
        taskParam.beginTime = ProfGetCurCpuTimestamp();
        taskParam.taskPara.DMA.src = reinterpret_cast<void*>(src);
        taskParam.taskPara.DMA.dst = reinterpret_cast<void*>(dst);
        taskParam.taskPara.DMA.size = blockSize;
        taskParam.taskPara.DMA.notifyID = INVALID_VALUE_NOTIFYID;
        taskParam.taskPara.DMA.linkType = DfxLinkType::ONCHIP;
        taskParam.taskPara.DMA.dmaOp = DmaOp::HCCL_DMA_READ;
        auto taskInfo = std::make_unique<TaskInfo>(stream.GetSqId(), taskId, INVALID_VALUE_RANKID, taskParam);
        resMgrFetcher->GetMirrorTaskMgrLite()->AddTaskInfo(std::move(taskInfo));
        src += offset;
        dst += offset;
    }
}

void Interpret(const InsLocalCopyExtend& ins, const StreamLite& stream, ResMgrFetcher* resMgrFetcher)
{
    if (UNLIKELY(ins.GetSrcBuffer().GetSize() == 0)) {
        HCCL_WARNING("%s insLocalCopyExtend srcBuffer size is 0, return", __func__);
        return;
    }

    // 传入数据大小不能超过 u32最大值， 需要进行切分
    u64 u32Max = UINT32_MAX;
    double countSplitingTimes = static_cast<double>(ins.GetSrcBuffer().GetSize()) / static_cast<double>(u32Max);
    u64 splitingTimes = static_cast<int>(std::ceil(countSplitingTimes));
    u64 src = ins.GetSrcBuffer().GetAddr();
    u64 dst = ins.GetDstBuffer().GetAddr();
    u64 blockSize = u32Max;
    u64 offset = u32Max;
    for (u64 i = 0; i < splitingTimes; i++) {
        // 处理尾块数据
        if (i == splitingTimes - 1) {
            blockSize = ins.GetSrcBuffer().GetSize() - u32Max * (splitingTimes - 1);
            offset = 0;
        }

        auto taskId = stream.GetRtsq()->GetTaskId();
        stream.GetRtsq()->SdmaCopy(src, dst, blockSize, 0); // 待确认， PART_ID是否固定设置为 0
        HCCL_INFO("InsLocalCopyExtend srcA:0x%llx dstA:0x%llx,size=0x%llx", src, dst, blockSize);
        TaskParam taskParam{};
        taskParam.taskType = TaskParamType::TASK_SDMA;
        taskParam.beginTime = ProfGetCurCpuTimestamp();
        taskParam.taskPara.DMA.src = reinterpret_cast<void*>(src);
        taskParam.taskPara.DMA.dst = reinterpret_cast<void*>(dst);
        taskParam.taskPara.DMA.size = blockSize;
        taskParam.taskPara.DMA.notifyID = INVALID_VALUE_NOTIFYID;
        taskParam.taskPara.DMA.linkType = DfxLinkType::ONCHIP;
        taskParam.taskPara.DMA.dmaOp = DmaOp::HCCL_DMA_READ;
        auto taskInfo = std::make_unique<TaskInfo>(stream.GetSqId(), taskId, INVALID_VALUE_RANKID, taskParam);
        resMgrFetcher->GetMirrorTaskMgrLite()->AddTaskInfo(std::move(taskInfo));
        src += offset;
        dst += offset;
    }
}

inline void AicpuCheckLocalReduceIns(const InsLocalReduce& ins)
{
    if (UNLIKELY(ins.GetDataType() == DataType::INT64)) {
        THROW<InvalidParamsException>(
            StringFormat("%s LocalReduce SDMA InlineReduce does not support INT64, need use TBE.", __func__));
    }
}

void Interpret(const InsLocalReduce& ins, const StreamLite& stream, ResMgrFetcher* resMgrFetcher)
{
    // SDMA支持的Reduce，则使用 sdmaReduce
    // SDMA不支持的Reduce，则使用 TBE算子(Asend C算子）

    if (UNLIKELY(ins.GetSrcSlice().GetSize() == 0)) {
        HCCL_WARNING("%s InsLocalReduce srcSlice size is 0, return", __func__);
        return;
    }

    if (UNLIKELY(ins.GetSrcSlice().GetSize() != ins.GetDstSlice().GetSize())) {
        HCCL_WARNING("%s InsLocalReduce srcSlice size is not equal to dstSlice size, return", __func__);
        return;
    }

    AicpuCheckLocalReduceIns(ins);
    RmaBufferLite* srcPtr = resMgrFetcher->GetRmaBufferLite(ins.GetSrcSlice().GetType());
    RmaBufferLite* dstPtr = resMgrFetcher->GetRmaBufferLite(ins.GetDstSlice().GetType());
    u64 srcOffset = ins.GetSrcSlice().GetOffset();
    u64 dstOffset = ins.GetDstSlice().GetOffset();
    if (UNLIKELY((srcPtr->GetSize() < srcOffset) && (dstPtr->GetSize() < dstOffset))) {
        THROW<InvalidParamsException>(StringFormat(
            "Interpret: offset exceeds memSize, srcPtr size[%llu], srcOffset[%llu], dstPtr size[%llu], dstOffset[%llu]",
            srcPtr->GetSize(), srcOffset, dstPtr->GetSize(), dstOffset));
    }
    u64 src = srcPtr->GetAddr() + srcOffset;
    u64 dst = dstPtr->GetAddr() + dstOffset;
    ReduceIn reduceIn(ins.GetDataType(), ins.GetReduceOp());

    auto taskId = stream.GetRtsq()->GetTaskId();
    stream.GetRtsq()->SdmaReduce(
        src, dst, ins.GetSrcSlice().GetSize(), 0, reduceIn); // 待确认， PART_ID是否固定设置为 0

    HCCL_INFO("InsLocalReduce srcA:0x%llx dstA:0x%llx,size=0x%llx", src, dst, ins.GetSrcSlice().GetSize());
    TaskParam taskParam{};
    taskParam.taskType = TaskParamType::TASK_REDUCE_INLINE;
    taskParam.beginTime = ProfGetCurCpuTimestamp();
    taskParam.taskPara.Reduce.src = reinterpret_cast<void*>(src);
    taskParam.taskPara.Reduce.dst = reinterpret_cast<void*>(dst);
    taskParam.taskPara.Reduce.size = ins.GetSrcSlice().GetSize();
    taskParam.taskPara.Reduce.notifyID = INVALID_VALUE_NOTIFYID;
    taskParam.taskPara.Reduce.linkType = DfxLinkType::ONCHIP;
    taskParam.taskPara.Reduce.dataType = DataTypeToHcclDataType(ins.GetDataType());
    taskParam.taskPara.Reduce.reduceOp = ReduceOpToHcclReduceOp(ins.GetReduceOp());
    auto taskInfo = std::make_unique<TaskInfo>(stream.GetSqId(), taskId, INVALID_VALUE_RANKID, taskParam);
    resMgrFetcher->GetMirrorTaskMgrLite()->AddTaskInfo(std::move(taskInfo));
}

void Interpret(const InsLocalWaitGroup& ins, const StreamLite& stream, ResMgrFetcher* resMgrFetcher)
{
    auto taskId = stream.GetRtsq()->GetTaskId();
    auto& notify = GetCntNto1NotifyLite(ins, resMgrFetcher);
    u32 value = 0;
    u32 offsetNum = 32;
    for (auto iter = ins.Iter(); iter.HasNext(); ++iter) {
        if (UNLIKELY(*iter >= offsetNum)) {
            THROW<InternalException>("Invalid iter value: %d. Must be in [0, 31].", *iter);
        }
        value |= BASE_BIT << *iter;
    }
    HCCL_INFO("InsLocalBcastPost notifyId=%u, value %u", notify.GetId(), value);
    stream.GetRtsq()->CntNto1NotifyWait(notify.GetId(), value);

    TaskParam taskParam{};
    taskParam.taskType = TaskParamType::TASK_NOTIFY_WAIT;
    taskParam.beginTime = ProfGetCurCpuTimestamp();
    taskParam.taskPara.Notify.notifyID = notify.GetId();
    taskParam.taskPara.Notify.value = value;
    auto taskInfo = std::make_unique<TaskInfo>(stream.GetSqId(), taskId, INVALID_VALUE_RANKID, taskParam);
    resMgrFetcher->GetMirrorTaskMgrLite()->AddTaskInfo(std::move(taskInfo));
}

void Interpret(const InsLocalBcastPost& ins, const StreamLite& stream, ResMgrFetcher* resMgrFetcher)
{
    auto taskId = stream.GetRtsq()->GetTaskId();
    auto& notify = GetCnt1toNNotifyLite(ins, resMgrFetcher);
    u32 value = 0;
    for (auto iter = ins.Iter(); iter.HasNext(); ++iter) {
        value |= BASE_BIT << *iter;
    }
    HCCL_INFO("InsLocalBcastPost notifyId=%u, value %u", notify.GetId(), value);
    stream.GetRtsq()->Cnt1toNNotifyRecord(notify.GetId(), value);

    TaskParam taskParam{};
    taskParam.taskType = TaskParamType::TASK_NOTIFY_RECORD;
    taskParam.beginTime = ProfGetCurCpuTimestamp();
    taskParam.taskPara.Notify.notifyID = notify.GetId();
    taskParam.taskPara.Notify.value = value;
    auto taskInfo = std::make_unique<TaskInfo>(stream.GetSqId(), taskId, INVALID_VALUE_RANKID, taskParam);
    resMgrFetcher->GetMirrorTaskMgrLite()->AddTaskInfo(std::move(taskInfo));
}

void Interpret(const InsPostReady& ins, const StreamLite& stream, ResMgrFetcher* resMgrFetcher)
{
    auto& transport = GetTransportLite(ins, resMgrFetcher);
    transport.Post(NOTIFY_INDEX_READY, stream);
}

void Interpret(const InsWaitReady& ins, const StreamLite& stream, ResMgrFetcher* resMgrFetcher)
{
    auto& transport = GetTransportLite(ins, resMgrFetcher);
    transport.Wait(NOTIFY_INDEX_READY, stream);
}

void Interpret(const InsPostFin& ins, const StreamLite& stream, ResMgrFetcher* resMgrFetcher)
{
    auto& transport = GetTransportLite(ins, resMgrFetcher);
    transport.Post(NOTIFY_INDEX_FIN, stream);
}

void Interpret(const InsWaitFin& ins, const StreamLite& stream, ResMgrFetcher* resMgrFetcher)
{
    auto& transport = GetTransportLite(ins, resMgrFetcher);
    transport.Wait(NOTIFY_INDEX_FIN, stream);
}

void Interpret(const InsPostFinAck& ins, const StreamLite& stream, ResMgrFetcher* resMgrFetcher)
{
    auto& transport = GetTransportLite(ins, resMgrFetcher);
    transport.Post(NOTIFY_INDEX_FIN_ACK, stream);
}

void Interpret(const InsWaitFinAck& ins, const StreamLite& stream, ResMgrFetcher* resMgrFetcher)
{
    auto& transport = GetTransportLite(ins, resMgrFetcher);
    transport.Wait(NOTIFY_INDEX_FIN_ACK, stream);
}

void Interpret(const InsRead& ins, const StreamLite& stream, ResMgrFetcher* resMgrFetcher)
{
    if (UNLIKELY(ins.GetLocalSlice().GetSize() == 0 && ins.GetRemoteSlice().GetSize() == 0)) {
        HCCL_WARNING("%s InsRead localSlice size is 0 and remoteSlice size is 0, return", __func__);
        return;
    } else if (UNLIKELY(ins.GetLocalSlice().GetSize() != ins.GetRemoteSlice().GetSize())) {
        THROW<InvalidParamsException>(
            StringFormat("%s InsRead either localSlice size or remoteSlice size is not zero", __func__));
    }

    auto& transport = GetTransportLite(ins, resMgrFetcher);
    transport.Read(GetLocRmaBufferLite(ins, resMgrFetcher), GetRmtBuffer(ins, transport, resMgrFetcher), stream);
}

void Interpret(const InsReadReduce& ins, const StreamLite& stream, ResMgrFetcher* resMgrFetcher)
{
    if (UNLIKELY(ins.GetLocalSlice().GetSize() == 0 && ins.GetRemoteSlice().GetSize() == 0)) {
        HCCL_WARNING("%s InsReadReduce localSlice size is 0 and remoteSlice size is 0, return", __func__);
        return;
    } else if (UNLIKELY(ins.GetLocalSlice().GetSize() != ins.GetRemoteSlice().GetSize())) {
        THROW<InvalidParamsException>(StringFormat(
            "%s InsReadReduce either localSlice size or remoteSlice size "
            "is not zero",
            __func__));
    }

    auto& transport = GetTransportLite(ins, resMgrFetcher);
    transport.ReadReduce(
        GetLocRmaBufferLite(ins, resMgrFetcher), GetRmtBuffer(ins, transport, resMgrFetcher),
        ReduceIn(ins.GetDataType(), ins.GetReduceOp()), stream);
}

void Interpret(const InsBatchRead& ins, const StreamLite& stream, ResMgrFetcher* resMgrFetcher)
{
    auto& transport = GetTransportLite(ins, resMgrFetcher);
    std::vector<RmaBufferLite> locRmaBufferLites;
    std::vector<Buffer> rmtBuffers;
    std::vector<BaseTransportLiteImpl::TransferOp> transferOp;
    if (UNLIKELY(!ins.Iter().HasNext())) {
        THROW<InvalidParamsException>(StringFormat("[%s] the number of InsBatchRead is zero.", __func__));
    }

    for (auto iter = ins.Iter(); iter.HasNext(); ++iter) {
        if (iter->GetType() == InstructionType::READ) {
            const InsRead& insRead = dynamic_cast<const InsRead&>(*iter);
            if (UNLIKELY(insRead.GetLocalSlice().GetSize() == 0 && insRead.GetRemoteSlice().GetSize() == 0)) {
                HCCL_WARNING(
                    "%s InsRead in InsBatchRead localSlice size is 0 and remoteSlice size is 0, return", __func__);
                continue;
            } else if (UNLIKELY(insRead.GetLocalSlice().GetSize() != insRead.GetRemoteSlice().GetSize())) {
                THROW<InvalidParamsException>(StringFormat(
                    "%s InsRead in InsBatchRead either localSlice size or "
                    "remoteSlice size is not zero",
                    __func__));
            }
            locRmaBufferLites.push_back(GetLocRmaBufferLite(insRead, resMgrFetcher));
            rmtBuffers.push_back(GetRmtBuffer(insRead, transport, resMgrFetcher));
            transferOp.push_back({TransferType(TransferType::READ), ReduceIn(DataType::INVALID, ReduceOp::INVALID)});
        } else if (iter->GetType() == InstructionType::READ_REDUCE) {
            const InsReadReduce& insReadReduce = dynamic_cast<const InsReadReduce&>(*iter);
            if (UNLIKELY(
                    insReadReduce.GetLocalSlice().GetSize() == 0 && insReadReduce.GetRemoteSlice().GetSize() == 0)) {
                HCCL_WARNING(
                    "%s InsReadReduce in InsBatchRead localSlice size is 0 and remoteSlice size is 0, return",
                    __func__);
                continue;
            } else if (UNLIKELY(insReadReduce.GetLocalSlice().GetSize() != insReadReduce.GetRemoteSlice().GetSize())) {
                THROW<InvalidParamsException>(StringFormat(
                    "%s InsReadReduce in InsBatchRead either localSlice size or "
                    "remoteSlice size is not 0",
                    __func__));
            }
            locRmaBufferLites.push_back(GetLocRmaBufferLite(insReadReduce, resMgrFetcher));
            rmtBuffers.push_back(GetRmtBuffer(insReadReduce, transport, resMgrFetcher));
            transferOp.push_back(
                {TransferType(TransferType::READ), ReduceIn(insReadReduce.GetDataType(), insReadReduce.GetReduceOp())});
        }
    }

    if (UNLIKELY(locRmaBufferLites.empty())) {
        return;
    }
    transport.BatchTransfer(locRmaBufferLites, rmtBuffers, transferOp, stream);
}

void Interpret(const InsReadExtend& ins, const StreamLite& stream, ResMgrFetcher* resMgrFetcher)
{
    DataBuffer localBuffer = ins.GetLocalBuffer();
    if (UNLIKELY(localBuffer.GetSize() == 0)) {
        HCCL_WARNING("%s insReadExtend localSlice size is 0, return", __func__);
        return;
    }
    DataBuffer remoteBuffer = ins.GetRemoteBuffer();
    u64 scratchAddr = resMgrFetcher->GetRmaBufferLite(BufferType::SCRATCH)->GetAddr();
    u64 scratchSize = resMgrFetcher->GetRmaBufferLite(BufferType::SCRATCH)->GetSize();
    HCCL_INFO("%s scratchAddr = %llu, scratchSize = %llu", __func__, scratchAddr, scratchSize);
    RmaBufferLite loc(
        localBuffer.GetAddr(), localBuffer.GetSize(),
        resMgrFetcher->GetRmaBufferLite(BufferType::SCRATCH)->GetTokenId(),
        resMgrFetcher->GetRmaBufferLite(BufferType::SCRATCH)->GetTokenValue());
    Buffer rmt(remoteBuffer.GetAddr(), remoteBuffer.GetSize());
    HCCL_INFO("%s RmaBufferLite = %s, Buffer = %s", __func__, loc.Describe().c_str(), rmt.Describe().c_str());
    auto& transport = GetTransportLite(ins, resMgrFetcher);
    transport.Read(loc, rmt, stream);
}

void Interpret(const InsWrite& ins, const StreamLite& stream, ResMgrFetcher* resMgrFetcher)
{
    if (UNLIKELY(ins.GetLocalSlice().GetSize() == 0 && ins.GetRemoteSlice().GetSize() == 0)) {
        HCCL_WARNING("%s InsWrite localSlice size is 0 and remoteSlice size is 0, return", __func__);
        return;
    } else if (UNLIKELY(ins.GetLocalSlice().GetSize() != ins.GetRemoteSlice().GetSize())) {
        THROW<InvalidParamsException>(
            StringFormat("%s InsWrite either localSlice size or remoteSlice size is not zero", __func__));
    }

    auto& transport = GetTransportLite(ins, resMgrFetcher);
    transport.Write(GetLocRmaBufferLite(ins, resMgrFetcher), GetRmtBuffer(ins, transport, resMgrFetcher), stream);
}

void Interpret(const InsBatchWrite& ins, const StreamLite& stream, ResMgrFetcher* resMgrFetcher)
{
    auto& transport = GetTransportLite(ins, resMgrFetcher);
    std::vector<RmaBufferLite> locRmaBufferLites;
    std::vector<Buffer> rmtBuffers;
    std::vector<BaseTransportLiteImpl::TransferOp> transferOp;
    if (UNLIKELY(!ins.Iter().HasNext())) {
        THROW<InvalidParamsException>(StringFormat("[%s] the number of InsBatchWrite is zero.", __func__));
    }

    for (auto iter = ins.Iter(); iter.HasNext(); ++iter) {
        if (iter->GetType() == InstructionType::WRITE) {
            const InsWrite& insWrite = dynamic_cast<const InsWrite&>(*iter);
            if (UNLIKELY(insWrite.GetLocalSlice().GetSize() == 0 && insWrite.GetRemoteSlice().GetSize() == 0)) {
                HCCL_WARNING(
                    "%s InsWrite in InsBatchWrite localSlice size is 0 and remoteSlice size is 0, return", __func__);
                continue;
            } else if (UNLIKELY(insWrite.GetLocalSlice().GetSize() != insWrite.GetRemoteSlice().GetSize())) {
                THROW<InvalidParamsException>(StringFormat(
                    "%s InsWrite in InsBatchWrite either localSlice size or "
                    "remoteSlice size is not zero",
                    __func__));
            }
            locRmaBufferLites.push_back(GetLocRmaBufferLite(insWrite, resMgrFetcher));
            rmtBuffers.push_back(GetRmtBuffer(insWrite, transport, resMgrFetcher));
            transferOp.push_back({TransferType(TransferType::WRITE), ReduceIn(DataType::INVALID, ReduceOp::INVALID)});
        } else if (iter->GetType() == InstructionType::WRITE_REDUCE) {
            const InsWriteReduce& insWriteReduce = dynamic_cast<const InsWriteReduce&>(*iter);
            if (UNLIKELY(
                    insWriteReduce.GetLocalSlice().GetSize() == 0 && insWriteReduce.GetRemoteSlice().GetSize() == 0)) {
                HCCL_WARNING(
                    "%s InsWriteReduce in InsBatchWrite localSlice size is 0 and remoteSlice size is 0, "
                    "return",
                    __func__);
                continue;
            } else if (UNLIKELY(
                           insWriteReduce.GetLocalSlice().GetSize() != insWriteReduce.GetRemoteSlice().GetSize())) {
                THROW<InvalidParamsException>(StringFormat(
                    "%s InsWriteReduce in InsBatchWrite either localSlice size "
                    "or remoteSlice size is not 0",
                    __func__));
            }
            locRmaBufferLites.push_back(GetLocRmaBufferLite(insWriteReduce, resMgrFetcher));
            rmtBuffers.push_back(GetRmtBuffer(insWriteReduce, transport, resMgrFetcher));
            transferOp.push_back(
                {TransferType(TransferType::WRITE),
                 ReduceIn(insWriteReduce.GetDataType(), insWriteReduce.GetReduceOp())});
        }
    }

    if (UNLIKELY(locRmaBufferLites.empty())) {
        return;
    }
    transport.BatchTransfer(locRmaBufferLites, rmtBuffers, transferOp, stream);
}

void Interpret(const InsWriteExtend& ins, const StreamLite& stream, ResMgrFetcher* resMgrFetcher)
{
    DataBuffer localBuffer = ins.GetLocalBuffer();
    if (UNLIKELY(localBuffer.GetSize() == 0)) {
        HCCL_WARNING("%s insWriteExtend localSlice size is 0, return", __func__);
        return;
    }
    DataBuffer remoteBuffer = ins.GetRemoteBuffer();
    u64 scratchAddr = resMgrFetcher->GetRmaBufferLite(BufferType::SCRATCH)->GetAddr();
    u64 scratchSize = resMgrFetcher->GetRmaBufferLite(BufferType::SCRATCH)->GetSize();
    HCCL_INFO("%s scratchAddr = %llu, scratchSize = %llu", __func__, scratchAddr, scratchSize);
    RmaBufferLite loc(
        localBuffer.GetAddr(), localBuffer.GetSize(),
        resMgrFetcher->GetRmaBufferLite(BufferType::SCRATCH)->GetTokenId(),
        resMgrFetcher->GetRmaBufferLite(BufferType::SCRATCH)->GetTokenValue());
    Buffer rmt(remoteBuffer.GetAddr(), remoteBuffer.GetSize());
    HCCL_INFO("%s RmaBufferLite = %s, Buffer = %s", __func__, loc.Describe().c_str(), rmt.Describe().c_str());
    auto& transport = GetTransportLite(ins, resMgrFetcher);
    transport.Write(loc, rmt, stream);
}

void Interpret(const InsWriteWithFin& ins, const StreamLite& stream, ResMgrFetcher* resMgrFetcher)
{
    auto& transport = GetTransportLite(ins, resMgrFetcher);

    if (UNLIKELY(ins.GetLocalSlice().GetSize() == 0)) {
        HCCL_WARNING("%s insWriteWithFin localSlice size is 0, transform to insPostFin", __func__);
        transport.Post(NOTIFY_INDEX_FIN, stream);
        return;
    }

    transport.WriteWithNotify(
        GetLocRmaBufferLite(ins, resMgrFetcher), GetRmtBuffer(ins, transport, resMgrFetcher),
        WithNotifyIn(TransportNotifyType::NORMAL, NOTIFY_INDEX_FIN), stream);
}

void Interpret(const InsWriteWithFinExtend& ins, const StreamLite& stream, ResMgrFetcher* resMgrFetcher)
{
    DataBuffer localBuffer = ins.GetLocalBuffer();
    DataBuffer remoteBuffer = ins.GetRemoteBuffer();
    auto& transport = GetTransportLite(ins, resMgrFetcher);

    if (UNLIKELY(localBuffer.GetSize() == 0)) {
        HCCL_WARNING("%s insWriteWithFinExtend localBuffer size is 0, transform to insPostFin", __func__);
        transport.Post(NOTIFY_INDEX_FIN, stream);
        return;
    }
    u64 scratchAddr = resMgrFetcher->GetRmaBufferLite(BufferType::SCRATCH)->GetAddr();
    u64 scratchSize = resMgrFetcher->GetRmaBufferLite(BufferType::SCRATCH)->GetSize();
    HCCL_INFO("%s scratchAddr = %llu, scratchSize = %llu", __func__, scratchAddr, scratchSize);

    RmaBufferLite loc(
        localBuffer.GetAddr(), localBuffer.GetSize(),
        resMgrFetcher->GetRmaBufferLite(BufferType::SCRATCH)->GetTokenId(),
        resMgrFetcher->GetRmaBufferLite(BufferType::SCRATCH)->GetTokenValue());
    Buffer rmt(remoteBuffer.GetAddr(), remoteBuffer.GetSize());
    HCCL_INFO("%s RmaBufferLite = %s, Buffer = %s", __func__, loc.Describe().c_str(), rmt.Describe().c_str());

    transport.WriteWithNotify(loc, rmt, WithNotifyIn(TransportNotifyType::NORMAL, NOTIFY_INDEX_FIN), stream);
}

void Interpret(const InsWriteReduce& ins, const StreamLite& stream, ResMgrFetcher* resMgrFetcher)
{
    if (UNLIKELY(ins.GetLocalSlice().GetSize() == 0 && ins.GetRemoteSlice().GetSize() == 0)) {
        HCCL_WARNING("%s InsWriteReduce localSlice size is 0 and remoteSlice size is 0, return", __func__);
        return;
    } else if (UNLIKELY(ins.GetLocalSlice().GetSize() != ins.GetRemoteSlice().GetSize())) {
        THROW<InvalidParamsException>(StringFormat(
            "%s InsWriteReduce either localSlice size or remoteSlice size "
            "is not zero",
            __func__));
    }

    auto& transport = GetTransportLite(ins, resMgrFetcher);
    transport.WriteReduce(
        GetLocRmaBufferLite(ins, resMgrFetcher), GetRmtBuffer(ins, transport, resMgrFetcher),
        ReduceIn(ins.GetDataType(), ins.GetReduceOp()), stream);
}

void Interpret(const InsWriteReduceWithFin& ins, const StreamLite& stream, ResMgrFetcher* resMgrFetcher)
{
    auto& transport = GetTransportLite(ins, resMgrFetcher);

    if (UNLIKELY(ins.GetLocalSlice().GetSize() == 0)) {
        HCCL_WARNING("%s insWriteReduceWithFin localSlice size is 0, transform to insPostFin", __func__);
        transport.Post(NOTIFY_INDEX_FIN, stream);
        return;
    }

    transport.WriteReduceWithNotify(
        GetLocRmaBufferLite(ins, resMgrFetcher), GetRmtBuffer(ins, transport, resMgrFetcher),
        ReduceIn(ins.GetDataType(), ins.GetReduceOp()), WithNotifyIn(TransportNotifyType::NORMAL, NOTIFY_INDEX_FIN),
        stream);
}

void Interpret(const InsBatchOneSidedRead& ins, const StreamLite& stream, ResMgrFetcher* resMgrFetcher)
{
    auto& transport = GetTransportLite(ins, resMgrFetcher);
    transport.BatchOneSidedRead(ins.GetLocalSlice(), ins.GetRemoteSlice(), stream);
}

void Interpret(const InsBatchOneSidedWrite& ins, const StreamLite& stream, ResMgrFetcher* resMgrFetcher)
{
    auto& transport = GetTransportLite(ins, resMgrFetcher);
    transport.BatchOneSidedWrite(ins.GetLocalSlice(), ins.GetRemoteSlice(), stream);
}

using InsToSqeRule91095 = std::function<void(const Instruction&, const StreamLite&, ResMgrFetcher* resMgrFetcher)>;

template <class InsType>
InsToSqeRule91095 Rule91095()
{
    return [](const Instruction& ins, const StreamLite& stream, ResMgrFetcher* resMgrFetcher) {
        return Interpret(static_cast<const InsType&>(ins), stream, resMgrFetcher);
    };
}

void Interpret(const InsStreamSync& insStreamSync, const StreamLite& stream, ResMgrFetcher* resMgrFetcher)
{
    (void)insStreamSync;
    constexpr uint64_t NANOSECOND_TO_SECOND = 1000000000U;
    const uint64_t kPrintSqInterval = 30U;
    uint32_t head = 0;
    uint32_t tail = 0;
    u32 timeout = resMgrFetcher->GetExecTimeOut() + 10;
    u64 startUsec = GetCurAicpuTimestamp();
    u64 lastUsec = startUsec;
    u32 sqId = stream.GetSqId();
    tail = stream.GetRtsq()->QuerySqTail();
    HCCL_INFO("StreamSync aicpu stream sqid[%d] tail[%u]", sqId, tail);
    do {
        head = stream.GetRtsq()->QuerySqHead();
        u64 curUsec = GetCurAicpuTimestamp();
        if (UNLIKELY(curUsec - startUsec > NANOSECOND_TO_SECOND * timeout)) {
            string msg
                = StringFormat("stream sync timeout %lus. curhead:%u, curtail:%u, sqId:%u", timeout, head, tail, sqId);
            THROW<TimeoutException>(msg);
        }

        // 等待下发阶段，每隔30s打印一次状态
        if (curUsec - lastUsec > NANOSECOND_TO_SECOND * kPrintSqInterval) {
            lastUsec = curUsec;
            HCCL_INFO("[StreamSync]Current state. sqid:%d, head:%u, tail:%u", sqId, head, tail);
        }
    } while (head != tail);
}

void Interpret(
    [[maybe_unused]] const InsPreStreamSync& insPreStreamSync, const StreamLite& stream,
    [[maybe_unused]] ResMgrFetcher* resMgrFetcher)
{
    HcclResult ret = stream.GetRtsq()->SetPreStreamSyncReady();
    stream.GetRtsq()->LaunchTask();
    if (UNLIKELY(ret != HCCL_SUCCESS)) {
        string msg = StringFormat("[Interpret]SetPreStreamSyncReady failed");
        THROW<InternalException>(msg);
    }
}

void Interpret(const InsAicpuReduce& insAicpuReduce, const StreamLite& stream, ResMgrFetcher* resMgrFetcher)
{
    // 使用aicpu进行reduce运算，支持int64 uint64 fp64
    if (UNLIKELY(insAicpuReduce.GetSrcSlice().GetSize() == 0)) {
        HCCL_WARNING("%s InsAicpuReduce srcSlice size is 0, return", __func__);
        return;
    }

    if (UNLIKELY(insAicpuReduce.GetSrcSlice().GetSize() != insAicpuReduce.GetDstSlice().GetSize())) {
        HCCL_WARNING("%s InsAicpuReduce srcSlice size is not equal to dstSlice size, return", __func__);
        return;
    }

    RmaBufferLite* srcPtr = resMgrFetcher->GetRmaBufferLite(insAicpuReduce.GetSrcSlice().GetType());
    RmaBufferLite* dstPtr = resMgrFetcher->GetRmaBufferLite(insAicpuReduce.GetDstSlice().GetType());
    u64 srcOffset = insAicpuReduce.GetSrcSlice().GetOffset();
    u64 dstOffset = insAicpuReduce.GetDstSlice().GetOffset();
    if (UNLIKELY((srcPtr->GetSize() < srcOffset) && (dstPtr->GetSize() < dstOffset))) {
        THROW<InvalidParamsException>(StringFormat(
            "Interpret: offset exceeds memSize, srcPtr size[%llu], srcOffset[%llu], dstPtr size[%llu], dstOffset[%llu]",
            srcPtr->GetSize(), srcOffset, dstPtr->GetSize(), dstOffset));
    }
    void* dst = reinterpret_cast<void*>(dstPtr->GetAddr() + insAicpuReduce.GetDstSlice().GetOffset());
    void* src = reinterpret_cast<void*>(srcPtr->GetAddr() + insAicpuReduce.GetSrcSlice().GetOffset());
    insAicpuReduce.RunAicpuReduce(
        dst, insAicpuReduce.GetDstSlice().GetSize(), src, insAicpuReduce.GetSrcSlice().GetSize(),
        insAicpuReduce.GetDataType(), insAicpuReduce.GetReduceOp());
    HCCL_INFO("InsAicpuReduce srcA:0x%p dstA:0x%p, size=0x%llx", src, dst, insAicpuReduce.GetSrcSlice().GetSize());
    auto taskId = stream.GetRtsq()->GetTaskId();
    TaskParam taskParam{};
    taskParam.taskType = TaskParamType::TASK_REDUCE_INLINE;
    taskParam.beginTime = ProfGetCurCpuTimestamp();
    taskParam.taskPara.Reduce.src = reinterpret_cast<void*>(src);
    taskParam.taskPara.Reduce.dst = reinterpret_cast<void*>(dst);
    taskParam.taskPara.Reduce.size = insAicpuReduce.GetSrcSlice().GetSize();
    taskParam.taskPara.Reduce.notifyID = INVALID_VALUE_NOTIFYID;
    taskParam.taskPara.Reduce.linkType = DfxLinkType::ONCHIP;
    taskParam.taskPara.Reduce.dataType = DataTypeToHcclDataType(insAicpuReduce.GetDataType());
    taskParam.taskPara.Reduce.reduceOp = ReduceOpToHcclReduceOp(insAicpuReduce.GetReduceOp());
    auto taskInfo = std::make_unique<TaskInfo>(stream.GetSqId(), taskId, INVALID_VALUE_RANKID, taskParam);
    resMgrFetcher->GetMirrorTaskMgrLite()->AddTaskInfo(std::move(taskInfo));
}

const std::unordered_map<InstructionType, InsToSqeRule91095, std::EnumClassHash> insRule91095Map{
    {InstructionType::LOCAL_COPY, Rule91095<InsLocalCopy>()},
    {InstructionType::LOCAL_POST_TO, Rule91095<InsLocalPostTo>()},
    {InstructionType::LOCAL_WAIT_FROM, Rule91095<InsLocalWaitFrom>()},
    {InstructionType::LOCAL_BCAST_POST, Rule91095<InsLocalBcastPost>()},
    {InstructionType::LOCAL_WAIT_GROUP, Rule91095<InsLocalWaitGroup>()},
    {InstructionType::WAIT_READY, Rule91095<InsWaitReady>()},
    {InstructionType::POST_READY, Rule91095<InsPostReady>()},
    {InstructionType::WAIT_FIN, Rule91095<InsWaitFin>()},
    {InstructionType::POST_FIN, Rule91095<InsPostFin>()},
    {InstructionType::WRITE, Rule91095<InsWrite>()},
    {InstructionType::WRITE_REDUCE, Rule91095<InsWriteReduce>()},
    {InstructionType::BATCH_WRITE, Rule91095<InsBatchWrite>()},
    {InstructionType::BATCH_READ, Rule91095<InsBatchRead>()},
    {InstructionType::READ, Rule91095<InsRead>()},
    {InstructionType::READ_REDUCE, Rule91095<InsReadReduce>()},
    {InstructionType::READ_EXTEND, Rule91095<InsReadExtend>()},
    {InstructionType::WRITE_REDUCE_WITH_FIN, Rule91095<InsWriteReduceWithFin>()},
    {InstructionType::WRITE_WITH_FIN, Rule91095<InsWriteWithFin>()},
    {InstructionType::LOCAL_COPY_EXTEND, Rule91095<InsLocalCopyExtend>()},
    {InstructionType::WRITE_EXTEND, Rule91095<InsWriteExtend>()},
    {InstructionType::WRITE_WITH_FIN_EXTEND, Rule91095<InsWriteWithFinExtend>()},
    {InstructionType::BATCH_ONE_SIDED_WRITE, Rule91095<InsBatchOneSidedWrite>()},
    {InstructionType::BATCH_ONE_SIDED_READ, Rule91095<InsBatchOneSidedRead>()},
    {InstructionType::LOCAL_REDUCE, Rule91095<InsLocalReduce>()},
    {InstructionType::POST_FIN_ACK, Rule91095<InsPostFinAck>()},
    {InstructionType::WAIT_FIN_ACK, Rule91095<InsWaitFinAck>()},
    {InstructionType::STREAM_SYNC, Rule91095<InsStreamSync>()},
    {InstructionType::PRE_STREAM_SYNC, Rule91095<InsPreStreamSync>()},
    {InstructionType::AICPU_REDUCE, Rule91095<InsAicpuReduce>()}};

void Interpret(const Instruction& ins, const StreamLite& stream, ResMgrFetcher* resMgrFetcher)
{
    HCCL_INFO("%s Instruction %s", __func__, ins.Describe().c_str());
    auto iter = insRule91095Map.find(ins.GetType());
    if (iter != insRule91095Map.end()) {
        auto& rule = iter->second;
        return rule(ins, stream, resMgrFetcher);
    }
    THROW<InternalException>(StringFormat("%s: invalid instruction type[%u]", __func__, ins.GetType()));
}

} // namespace Hccl
