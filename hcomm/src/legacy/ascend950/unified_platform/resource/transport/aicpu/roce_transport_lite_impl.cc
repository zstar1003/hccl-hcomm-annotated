/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "cast_utils.h"
#include "roce_transport_lite_impl.h"
#include "binary_stream.h"
#include "log.h"
#include "dfx_profiling_handler_lite.h"
#include "unified_platform/pub_inc/config_plf_log_v2.h"
#include "sal.h"

namespace Hccl {

RoceTransportLiteImpl::RoceTransportLiteImpl(std::vector<char>& uniqueId) { Init(uniqueId); }

RoceTransportLiteImpl::~RoceTransportLiteImpl() {}

void RoceTransportLiteImpl::Init(std::vector<char>& uniqueId)
{
    BinaryStream binaryStream(uniqueId);
    u32 type;
    binaryStream >> type;
    binaryStream >> notifyNum_;
    binaryStream >> bufferNum_;
    binaryStream >> connNum_;

    std::vector<char> locNotifyUniqueIds;
    binaryStream >> locNotifyUniqueIds;
    ParseLocNotifyVec(locNotifyUniqueIds);

    std::vector<char> rmtNotifyUniqueIds;
    binaryStream >> rmtNotifyUniqueIds;
    ParseRmtNotifyVec(rmtNotifyUniqueIds);

    std::vector<char> notifyValueBufferUniqueIds;
    binaryStream >> notifyValueBufferUniqueIds;
    ParseNotifyValueBuffer(notifyValueBufferUniqueIds);

    std::vector<char> locBufferUniqueIds;
    binaryStream >> locBufferUniqueIds;
    ParseLocBufferVec(locBufferUniqueIds);

    std::vector<char> rmtBufferUniqueIds;
    binaryStream >> rmtBufferUniqueIds;
    ParseRmtBufferVec(rmtBufferUniqueIds);

    std::vector<char> connUniqueIds;
    binaryStream >> connUniqueIds;
    ParseConnVec(connUniqueIds);
}

void RoceTransportLiteImpl::ParseLocNotifyVec(std::vector<char>& data)
{
    if (notifyNum_ == 0) {
        HCCL_WARNING("[RoceTransportLiteImpl::%s] notifyNum is 0", __func__);
        return;
    }

    u32 notifySizePerDto = data.size() / notifyNum_;

    for (u32 idx = 0; idx < notifyNum_; idx++) {
        auto start = data.begin() + idx * notifySizePerDto;
        auto end = start + notifySizePerDto;
        std::vector<char> dto(start, end);
        localNotifies_.push_back(std::make_unique<NotifyLite>(dto));
        HCCL_INFO("locNotify idx=%u, %s", idx, localNotifies_.back()->Describe().c_str());
    }
}

void RoceTransportLiteImpl::ParseRmtNotifyVec(std::vector<char>& data)
{
    if (notifyNum_ == 0) {
        HCCL_WARNING("[RoceTransportLiteImpl::%s] notifyNum is 0", __func__);
        return;
    }

    u32 rmtBufferSizePerDto = data.size() / notifyNum_;
    HCCL_INFO(
        "[RoceTransportLiteImpl::%s] Parse remote notify num=%u, sizePerDto=%u", __func__, notifyNum_,
        rmtBufferSizePerDto);

    BinaryStream binaryStream(data);
    remoteNotifies_.clear();
    u64 addr;
    u64 size;
    u32 rkey;
    for (u32 idx = 0; idx < notifyNum_; idx++) {
        binaryStream >> addr;
        binaryStream >> size;
        binaryStream >> rkey;
        RmtRmaBufferLite rdmaBufLite(addr, size, rkey);
        HCCL_INFO("idx=%u, %s", idx, rdmaBufLite.Describe().c_str());
        remoteNotifies_.emplace_back(rdmaBufLite);
    }
}

void RoceTransportLiteImpl::ParseNotifyValueBuffer(std::vector<char>& data)
{
    HCCL_INFO("[RoceTransportLiteImpl::%s] Parse notify value buffer", __func__);

    BinaryStream binaryStream(data);
    u64 addr;
    u64 size;
    u32 lkey;
    binaryStream >> addr;
    binaryStream >> size;
    binaryStream >> lkey;
    notifyValueBuffer_ = std::make_unique<RmaBufferLite>(addr, size, lkey);
}

void RoceTransportLiteImpl::ParseLocBufferVec(std::vector<char>& data)
{
    if (bufferNum_ == 0) {
        HCCL_WARNING("[RoceTransportLiteImpl::%s] bufferNum is 0", __func__);
        return;
    }

    u32 locBufferSizePerDto = data.size() / bufferNum_;
    HCCL_INFO(
        "[RoceTransportLiteImpl::%s] Parse local buffer num=%u, sizePerDto=%u", __func__, bufferNum_,
        locBufferSizePerDto);

    BinaryStream binaryStream(data);
    locBufferVec_.clear();
    u64 addr;
    u64 size;
    u32 lkey;
    for (u32 idx = 0; idx < bufferNum_; idx++) {
        binaryStream >> addr;
        binaryStream >> size;
        binaryStream >> lkey;
        RmaBufferLite rdmaBufLite(addr, size, lkey);
        HCCL_INFO("idx=%u, %s", idx, rdmaBufLite.Describe().c_str());
        locBufferVec_.emplace_back(rdmaBufLite);
    }
}

void RoceTransportLiteImpl::ParseRmtBufferVec(std::vector<char>& data)
{
    if (bufferNum_ == 0) {
        HCCL_WARNING("[RoceTransportLiteImpl::%s] bufferNum is 0", __func__);
        return;
    }

    u32 rmtBufferSizePerDto = data.size() / bufferNum_;
    HCCL_INFO(
        "[RoceTransportLiteImpl::%s] Parse remote buffer num=%u, sizePerDto=%u", __func__, bufferNum_,
        rmtBufferSizePerDto);

    BinaryStream binaryStream(data);
    rmtBufferVec_.clear();
    u64 addr;
    u64 size;
    u32 rkey;
    for (u32 idx = 0; idx < bufferNum_; idx++) {
        binaryStream >> addr;
        binaryStream >> size;
        binaryStream >> rkey;
        RmtRmaBufferLite rdmaBufLite(addr, size, rkey);
        HCCL_INFO("idx=%u, %s", idx, rdmaBufLite.Describe().c_str());
        rmtBufferVec_.emplace_back(rdmaBufLite);
    }
}

void RoceTransportLiteImpl::ParseConnVec(std::vector<char>& data)
{
    if (connNum_ == 0) {
        HCCL_WARNING("[RoceTransportLiteImpl::%s] connNum is 0", __func__);
        return;
    }

    u32 connSizePerDto = data.size() / connNum_;
    HCCL_INFO("[RoceTransportLiteImpl::%s] Parse conn num=%u, sizePerDto=%u", __func__, connNum_, connSizePerDto);
    for (u32 idx = 0; idx < connNum_; idx++) {
        auto start = data.begin() + idx * connSizePerDto;
        auto end = start + connSizePerDto;
        std::vector<char> connUniqueId(start, end);
        connUniqueIdVec_.emplace_back(connUniqueId);
        std::unique_ptr<RdmaConnLiteV2> connLite;
        connLite = std::make_unique<RdmaConnLiteV2>(connUniqueId);
        HCCL_INFO("[RoceTransportLiteImpl::%s] idx=%u, %s", __func__, idx, connLite->Describe().c_str());
        connVec_.emplace_back(std::move(connLite));
    }
}

RmaBufSliceLite RoceTransportLiteImpl::GetRmaBufSlicelite(const RmaBufferLite& lite) const
{
    return RmaBufSliceLite(lite.GetAddr(), lite.GetSize(), lite.GetLkey(), 0);
}

RmaBufSliceLite RoceTransportLiteImpl::GetNotifySlicelite(u32 index) const
{
    (void)index;
    return RmaBufSliceLite(
        notifyValueBuffer_->GetAddr(), notifyValueBuffer_->GetSize(), notifyValueBuffer_->GetLkey(), 0);
}

RmtRmaBufSliceLite RoceTransportLiteImpl::GetRmtRmaBufSliceLite(const Buffer& rmtBuf) const
{
    for (auto& it : rmtBufferVec_) {
        Buffer buf(it.GetAddr(), it.GetSize());
        if (buf.Contains(rmtBuf.GetAddr(), rmtBuf.GetSize())) {
            return RmtRmaBufSliceLite(rmtBuf.GetAddr(), rmtBuf.GetSize(), it.GetRkey(), 0, 0, UINT32_MAX);
        }
    }
    MACRO_THROW(InternalException, StringFormat("%s is not in current transport", rmtBuf.Describe().c_str()));
}

RmtRmaBufSliceLite RoceTransportLiteImpl::GetRmtNotifySliceLite(u32 index) const
{
    auto& lite = remoteNotifies_[index];
    return RmtRmaBufSliceLite(lite.GetAddr(), lite.GetSize(), lite.GetRkey(), 0, 0, UINT32_MAX);
}

std::string RoceTransportLiteImpl::Describe() const
{
    std::string desc = "RoceTransportLiteImpl[";

    u32 idx = 0;
    desc += "localNotifies=[";
    for (auto& it : localNotifies_) {
        desc += StringFormat("idx=%u, %s;", idx, it->Describe().c_str());
        idx++;
    }

    idx = 0;
    desc += "], remoteNotifies=[";
    for (auto& it : remoteNotifies_) {
        desc += StringFormat("idx=%u, %s;", idx, it.Describe().c_str());
        idx++;
    }

    idx = 0;
    desc += "], locBufferVec=[";
    for (auto& it : locBufferVec_) {
        desc += StringFormat("idx=%u, %s;", idx, it.Describe().c_str());
        idx++;
    }

    idx = 0;
    desc += "], rmtBufferVec=[";
    for (auto& it : rmtBufferVec_) {
        desc += StringFormat("idx=%u, %s;", idx, it.Describe().c_str());
        idx++;
    }

    idx = 0;
    desc += "], connVec=[";
    for (auto& it : connVec_) {
        desc += StringFormat("idx=%u, %s;", idx, it->Describe().c_str());
        idx++;
    }

    desc += "]]";
    return desc;
}

HcclResult
RoceTransportLiteImpl::BuildLocRmaBufferLite(const uintptr_t addr, const size_t size, RmaBufferLite& rmaBufferLite)
{
    HCCL_INFO(
        "[RoceTransportLiteImpl::%s] start to find addr[0x%llx], size[0x%llx] in locBufferVec, whose size is %zu. ",
        __func__, addr, size, locBufferVec_.size());

    if (locBufferVec_.empty()) {
        HCCL_ERROR("[RoceTransportLiteImpl::%s] locBufferVec is empty.", __func__);
        return HCCL_E_INTERNAL;
    }

    bool isAddrInRange = false;
    for (auto& it : locBufferVec_) {
        Buffer iterBuf(it.GetAddr(), it.GetSize());
        if (iterBuf.Contains(addr, size)) {
            rmaBufferLite = RmaBufferLite(addr, size, it.GetLkey());
            isAddrInRange = true;
            break;
        }
    }

    if (!isAddrInRange) {
        HCCL_WARNING(
            "[RoceTransportLiteImpl::%s] addr[0x%llx], size[0x%llx] not in any range of locBufferVec. The token of the "
            "first locBuffer is used.",
            __func__, addr, size);
        rmaBufferLite = RmaBufferLite(addr, size, locBufferVec_[0].GetLkey());
        return HCCL_SUCCESS;
    }

    return HCCL_SUCCESS;
}

void RoceTransportLiteImpl::Read(const RmaBufferLite& loc, const Buffer& rmt, const StreamLite& stream)
{
    u64 dbAddr = 0;
    u64 dbValue = 0;
    // 获取Profiling任务ID
    auto taskId = stream.GetRtsq()->GetTaskId();

    // 获取本端和远端Buffer切片
    SqeConfigLite cfg;
    SetFenceConfig(cfg);
    auto locRmaBufSliceLite = GetRmaBufSlicelite(loc);
    auto rmtRmaBufSliceLite = GetRmtRmaBufSliceLite(rmt);

    // Post Wqe && return dbValue
    connVec_[0]->Read(locRmaBufSliceLite, rmtRmaBufSliceLite, cfg, dbAddr, dbValue);

    // Ring Doorbell
    BuildRdmaDbSendTask(stream, dbAddr, dbValue);

    // 上报Profiling任务
    ReportDmaTask(
        ReinterpretAs<const void*>(locRmaBufSliceLite.GetAddr()),
        ReinterpretAs<const void*>(rmtRmaBufSliceLite.GetAddr()), locRmaBufSliceLite.GetSize(), stream, taskId,
        TaskParamType::TASK_RDMA, DmaOp::HCCL_DMA_READ, INVALID_VALUE_NOTIFYID, UINT32_MAX, __func__);

    // Poll Cq
    constexpr int32_t POLL_NUM = 1;     // poll cqe num
    constexpr int32_t POLL_TIMEOUT = 5; // 5 ms
    std::vector<int32_t> errList = {};
    connVec_[0]->PollCq(POLL_NUM, POLL_TIMEOUT, errList, dbAddr, dbValue);
}

void RoceTransportLiteImpl::Write(const RmaBufferLite& loc, const Buffer& rmt, const StreamLite& stream)
{
    u64 dbAddr = 0;
    u64 dbValue = 0;
    // 获取Profiling任务ID
    auto taskId = stream.GetRtsq()->GetTaskId();

    // 获取本端和远端Buffer切片
    auto locRmaBufSliceLite = GetRmaBufSlicelite(loc);
    auto rmtRmaBufSliceLite = GetRmtRmaBufSliceLite(rmt);
    SqeConfigLite cfg;
    SetFenceConfig(cfg);

    // Post Wqe && return dbValue
    connVec_[0]->Write(locRmaBufSliceLite, rmtRmaBufSliceLite, cfg, dbAddr, dbValue);

    // Ring Doorbell
    BuildRdmaDbSendTask(stream, dbAddr, dbValue);

    // 上报Profiling任务
    ReportDmaTask(
        ReinterpretAs<const void*>(locRmaBufSliceLite.GetAddr()),
        ReinterpretAs<const void*>(rmtRmaBufSliceLite.GetAddr()), locRmaBufSliceLite.GetSize(), stream, taskId,
        TaskParamType::TASK_RDMA, DmaOp::HCCL_DMA_WRITE, INVALID_VALUE_NOTIFYID, UINT32_MAX, __func__);

    // Poll Cq
    constexpr int32_t POLL_NUM = 1;     // poll cqe num
    constexpr int32_t POLL_TIMEOUT = 5; // 5 ms
    std::vector<int32_t> errList = {};
    connVec_[0]->PollCq(POLL_NUM, POLL_TIMEOUT, errList, dbAddr, dbValue);
}

void RoceTransportLiteImpl::WriteReduce(
    const RmaBufferLite& loc, const Buffer& rmt, const ReduceIn& reduceIn, const StreamLite& stream)
{
    u64 dbAddr = 0;
    u64 dbValue = 0;
    auto taskId = stream.GetRtsq()->GetTaskId();

    SqeConfigLite cfg;
    SetFenceConfig(cfg);
    auto locRmaBufSliceLite = GetRmaBufSlicelite(loc);
    auto rmtRmaBufSliceLite = GetRmtRmaBufSliceLite(rmt);

    // Post Wqe && return dbValue
    connVec_[0]->WriteReduce(
        locRmaBufSliceLite, rmtRmaBufSliceLite, cfg, reduceIn.dataType, reduceIn.reduceOp, dbAddr, dbValue);

    // Ring Doorbell
    BuildRdmaDbSendTask(stream, dbAddr, dbValue);

    // 上报Profiling任务
    ReportReduceTask(
        ReinterpretAs<const void*>(locRmaBufSliceLite.GetAddr()),
        ReinterpretAs<const void*>(rmtRmaBufSliceLite.GetAddr()), locRmaBufSliceLite.GetSize(), reduceIn, stream,
        taskId, TaskParamType::TASK_REDUCE_INLINE, INVALID_VALUE_NOTIFYID, UINT32_MAX, __func__);

    // Poll Cq
    constexpr int32_t POLL_NUM = 1;     // poll cqe num
    constexpr int32_t POLL_TIMEOUT = 5; // 5 ms
    std::vector<int32_t> errList = {};
    connVec_[0]->PollCq(POLL_NUM, POLL_TIMEOUT, errList, dbAddr, dbValue);
}

void RoceTransportLiteImpl::WriteWithNotify(
    const RmaBufferLite& loc, const Buffer& rmt, const WithNotifyIn& withNotify, const StreamLite& stream)
{
    auto taskId = stream.GetRtsq()->GetTaskId();
    u64 dbAddr = 0;
    u64 dbValue = 0;

    SqeConfigLite cfg;
    SetFenceConfig(cfg);
    auto locRmaBufSliceLite = GetRmaBufSlicelite(loc);
    auto rmtRmaBufSliceLite = GetRmtRmaBufSliceLite(rmt);
    auto locNotifySliceLite = GetNotifySlicelite(withNotify.index_); // 普通Notify
    auto rmtNotifySliceLite = GetRmtNotifySliceLite(withNotify.index_);

    // Post Wqe && return dbValue
    connVec_[0]->WriteWithNotify(
        locRmaBufSliceLite, rmtRmaBufSliceLite, locNotifySliceLite, rmtNotifySliceLite, cfg, dbAddr, dbValue);

    // Ring Doorbell
    BuildRdmaDbSendTask(stream, dbAddr, dbValue);

    // 上报Profiling任务
    ReportDmaTask(
        ReinterpretAs<const void*>(locRmaBufSliceLite.GetAddr()),
        ReinterpretAs<const void*>(rmtRmaBufSliceLite.GetAddr()), locRmaBufSliceLite.GetSize(), stream, taskId,
        TaskParamType::TASK_WRITE_WITH_NOTIFY, DmaOp::HCCL_DMA_WRITE, rmtNotifySliceLite.GetNotifyId(), 1, __func__);

    // Poll Cq
    constexpr int32_t POLL_NUM = 2;     // poll cqe num
    constexpr int32_t POLL_TIMEOUT = 5; // 5 ms
    std::vector<int32_t> errList = {};
    connVec_[0]->PollCq(POLL_NUM, POLL_TIMEOUT, errList, dbAddr, dbValue);
}

void RoceTransportLiteImpl::WriteReduceWithNotify(
    const RmaBufferLite& loc, const Buffer& rmt, const ReduceIn& reduceIn, const WithNotifyIn& withNotify,
    const StreamLite& stream)
{
    u64 dbAddr = 0;
    u64 dbValue = 0;
    auto taskId = stream.GetRtsq()->GetTaskId();

    auto locRmaBufSliceLite = GetRmaBufSlicelite(loc);
    auto rmtRmaBufSliceLite = GetRmtRmaBufSliceLite(rmt);
    auto locNotifySliceLite = GetNotifySlicelite(withNotify.index_); // 普通Notify
    auto rmtNotifySliceLite = GetRmtNotifySliceLite(withNotify.index_);
    SqeConfigLite cfg;
    SetFenceConfig(cfg);

    // Post Wqe && return dbValue
    connVec_[0]->WriteReduceWithNotify(
        locRmaBufSliceLite, rmtRmaBufSliceLite, locNotifySliceLite, rmtNotifySliceLite, cfg, reduceIn.dataType,
        reduceIn.reduceOp, dbAddr, dbValue);

    // Ring Doorbell
    BuildRdmaDbSendTask(stream, dbAddr, dbValue);

    // 上报Profiling任务
    ReportReduceTask(
        ReinterpretAs<const void*>(locRmaBufSliceLite.GetAddr()),
        ReinterpretAs<const void*>(rmtRmaBufSliceLite.GetAddr()), locRmaBufSliceLite.GetSize(), reduceIn, stream,
        taskId, TaskParamType::TASK_WRITE_REDUCE_WITH_NOTIFY, rmtNotifySliceLite.GetNotifyId(), 1, __func__);

    // Poll Cq
    constexpr int32_t POLL_NUM = 2;     // poll cqe num
    constexpr int32_t POLL_TIMEOUT = 5; // 5 ms
    std::vector<int32_t> errList = {};
    connVec_[0]->PollCq(POLL_NUM, POLL_TIMEOUT, errList, dbAddr, dbValue);
}

HcclResult RoceTransportLiteImpl::Fence()
{
    fence_ = true;
    HCCL_INFO("[%s] SUCCESS. fence[%d]", __func__, fence_);
    return HCCL_SUCCESS;
}

void RoceTransportLiteImpl::Post(u32 index, const StreamLite& stream)
{
    u64 dbAddr = 0;
    u64 dbValue = 0;
    auto taskId = stream.GetRtsq()->GetTaskId();

    SqeConfigLite cfg;
    SetFenceConfig(cfg);
    auto locNotifySliceLite = GetNotifySlicelite(index);
    auto rmtNotifySliceLite = GetRmtNotifySliceLite(index);

    // Post Wqe && return dbValue
    connVec_[0]->Write(locNotifySliceLite, rmtNotifySliceLite, cfg, dbAddr, dbValue);

    // Ring Doorbell
    BuildRdmaDbSendTask(stream, dbAddr, dbValue);

    // 上报Profiling任务
    ReportDmaTask(
        ReinterpretAs<const void*>(locNotifySliceLite.GetAddr()),
        ReinterpretAs<const void*>(rmtNotifySliceLite.GetAddr()), locNotifySliceLite.GetSize(), stream, taskId,
        TaskParamType::TASK_RDMA, DmaOp::HCCL_DMA_WRITE, rmtNotifySliceLite.GetNotifyId(), 1, __func__);

    // Poll Cq
    constexpr int32_t POLL_NUM = 1;     // poll cqe num
    constexpr int32_t POLL_TIMEOUT = 5; // 5 ms
    std::vector<int32_t> errList = {};
    connVec_[0]->PollCq(POLL_NUM, POLL_TIMEOUT, errList, dbAddr, dbValue);
}

HcclResult RoceTransportLiteImpl::PollCq(int32_t numEntries, int32_t timeOut, std::vector<int32_t>& errList)
{
    u64 dbAddr = 0;
    u64 cqDbValue = 0;
    HcclResult ret = HCCL_SUCCESS;

    // Poll numEntries个Cqe, 返回异常的status, 同时返回cq的db
    ret = connVec_[0]->PollCq(numEntries, timeOut, errList, dbAddr, cqDbValue);

    return ret;
}

void RoceTransportLiteImpl::WaitWithTimeout(u32 index, const StreamLite& stream, u32 timeout)
{
    auto taskId = stream.GetRtsq()->GetTaskId();
    auto notifyId = localNotifies_[index]->GetId();
    BuildNotifyWaitTask(notifyId, stream, timeout);

    // 上报Profiling任务
    ReportNotifyWaitTask(notifyId, stream, taskId);
}

// 下发Rtsq sqe, 敲DB
void RoceTransportLiteImpl::BuildRdmaDbSendTask(const StreamLite& stream, u64 remoteAddr, u64 dbValue) const
{
    stream.GetRtsq()->RdmaDbSend(remoteAddr, dbValue);
}

// 下发Rtsq sqe, NotifyWait
void RoceTransportLiteImpl::BuildNotifyWaitTask(u32 notifyId, const StreamLite& stream, u32 timeout) const
{
    stream.GetRtsq()->NotifyWait(notifyId, timeout);
}

void RoceTransportLiteImpl::SetFenceConfig(SqeConfigLite& cfg)
{
    cfg.cqeEn = true;
    cfg.fence = fence_ ? 1 : 0;
    fence_ = false;
}

void RoceTransportLiteImpl::ReportDmaTask(
    const void* src, const void* dst, u64 size, const StreamLite& stream, u32 taskId, TaskParamType taskType,
    [[maybe_unused]] DmaOp dmaOp, u64 notifyId, u32 notifyValue, const char* funcName) const
{
    // 未开启任务上报时直接返回
    if (!IsReportTask()) {
        return;
    }

    HCCL_INFO(
        "[RoceTransportLiteImpl::%s][ProfilingTaskParam] sqId[%u], taskId[%u], taskType[%s], "
        "src[%p], dst[%p], size[%zu], notifyId[%llu], notifyValue[%u]",
        funcName, stream.GetSqId(), taskId, taskType.Describe().c_str(), src, dst, size, notifyId, notifyValue);

    DfxTaskInfo* slot = stream.NextTaskSlot();
    slot->taskType = static_cast<u8>(taskType);
    slot->sqId = stream.GetSqId();
    slot->taskId = taskId;
    const void* opInfo = stream.GetLatestDfxOpInfo();
    slot->dfxOpInfo = (opInfo != nullptr) ? ReinterpretAs<u64>(opInfo) : DFX_INVALID_U64;
    slot->linkType = DfxLinkTypeVal::LINK_ROCE;
    slot->transportType = static_cast<u8>(DfxTransportType::DFX_TRANSPORT_TYPE_ROCE);
    slot->channelHandle = ReinterpretAs<u64>(this);
    slot->taskPara.Dma.sqeAddr = stream.GetRtsq()->GetSqeAddr();
    PLF_CONFIG_INFO(Hccl::PLF_TASK, "[%s] %s", __func__, slot->Describe().c_str());
}

void RoceTransportLiteImpl::ReportReduceTask(
    const void* src, const void* dst, u64 size, const ReduceIn& reduceIn, const StreamLite& stream, u32 taskId,
    TaskParamType taskType, u64 notifyId, u32 notifyValue, const char* funcName) const
{
    // 未开启任务上报时直接返回
    if (!IsReportTask()) {
        return;
    }

    HCCL_INFO(
        "[RoceTransportLiteImpl::%s][ProfilingTaskParam] sqId[%u], taskId[%u], taskType[%s], "
        "src[%p], dst[%p], size[%zu], notifyId[%llu], notifyValue[%u], dataType[%d], reduceOp[%d]",
        funcName, stream.GetSqId(), taskId, taskType.Describe().c_str(), src, dst, size, notifyId, notifyValue,
        static_cast<int>(reduceIn.dataType), static_cast<int>(reduceIn.reduceOp));

    DfxTaskInfo* slot = stream.NextTaskSlot();
    slot->taskType = static_cast<u8>(taskType);
    slot->sqId = stream.GetSqId();
    slot->taskId = taskId;
    const void* opInfo = stream.GetLatestDfxOpInfo();
    slot->dfxOpInfo = (opInfo != nullptr) ? ReinterpretAs<u64>(opInfo) : DFX_INVALID_U64;
    slot->linkType = DfxLinkTypeVal::LINK_ROCE;
    slot->transportType = static_cast<u8>(DfxTransportType::DFX_TRANSPORT_TYPE_ROCE);
    slot->channelHandle = ReinterpretAs<u64>(this);
    slot->taskPara.Reduce.sqeAddr = stream.GetRtsq()->GetSqeAddr();
    slot->taskPara.Reduce.srcAddr = ReinterpretAs<u64>(src);
    slot->taskPara.Reduce.dstAddr = ReinterpretAs<u64>(dst);
    slot->taskPara.Reduce.size = size;
    slot->taskPara.Reduce.notifyId = static_cast<u32>(notifyId);
    slot->taskPara.Reduce.reduceOp = static_cast<u8>(ConvertReduceOpToHcclReduceOp(reduceIn.reduceOp));
    PLF_CONFIG_INFO(Hccl::PLF_TASK, "[%s] %s", __func__, slot->Describe().c_str());
}

void RoceTransportLiteImpl::ReportNotifyWaitTask(u64 notifyId, const StreamLite& stream, u32 taskId) const
{
    // 未开启任务上报时直接返回
    if (!IsReportTask()) {
        return;
    }

    HCCL_INFO(
        "[RoceTransportLiteImpl::%s][ProfilingTaskParam] sqId[%u], taskId[%u], notifyId[%llu]", __func__,
        stream.GetSqId(), taskId, notifyId);

    DfxTaskInfo* slot = stream.NextTaskSlot();
    slot->taskType = static_cast<u8>(TaskParamTypeVal::TASK_NOTIFY_WAIT);
    slot->sqId = stream.GetSqId();
    slot->taskId = taskId;
    const void* opInfo = stream.GetLatestDfxOpInfo();
    slot->dfxOpInfo = (opInfo != nullptr) ? ReinterpretAs<u64>(opInfo) : DFX_INVALID_U64;
    slot->linkType = DfxLinkTypeVal::LINK_ROCE;
    slot->transportType = static_cast<u8>(DfxTransportType::DFX_TRANSPORT_TYPE_ROCE);
    slot->channelHandle = ReinterpretAs<u64>(this);
    slot->taskPara.Notify.sqeAddr = stream.GetRtsq()->GetSqeAddr();
    PLF_CONFIG_INFO(Hccl::PLF_TASK, "[%s] %s", __func__, slot->Describe().c_str());
}

bool RoceTransportLiteImpl::IsReportTask() const
{
    // TaskException或Profiling开启时，允许上报
    return taskExceptionEnable_ || DfxProfilingHandlerLite::GetInstance().GetProfL1State();
}

} // namespace Hccl
