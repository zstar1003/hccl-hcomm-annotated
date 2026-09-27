/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_DISPATCHER_AICPU_PUB_H
#define HCCL_DISPATCHER_AICPU_PUB_H

#include <vector>
#include <functional>
#include <mutex>
#include "sal_pub.h"
#include "dispatcher_pub.h"

#include "aicpu/aicpu_hccl_sqcq.h"
#include "aicpu/aicpu_hccl_sqcqv1.h"
#include "aicpu/aicpu_hccl_sqcqv2.h"

#include "op_unfold_cache.h"

namespace hccl {
using AddOneNotifyWaitSqe = void (*)(uint16_t, uint16_t, u64, const uint8_t*, uint8_t*, const dfx::DfxTimeOutConfig&);
using AddOneRecordSqe = void (*)(uint16_t, uint16_t, u64, const uint8_t*, uint8_t*);
using AddOneWriteValueRecordSqe = void (*)(uint16_t, uint16_t, u64, const uint8_t*, uint8_t*);
using AddOneMemcpySqe = void (*)(
    uint16_t, uint16_t, const void*, uint32_t, const aclDataType, aclrtReduceKind, const void*, uint32_t, uint32_t,
    uint32_t, u64, uint8_t, const uint8_t*, uint8_t*, uint32_t);
using AddOneEventResetSqe = void (*)(uint16_t, int32_t, uint16_t, int64_t, int64_t, u64, const uint8_t*, uint8_t*);
using AddOneEventRecordSqe = void (*)(uint16_t, int32_t, uint16_t, const uint8_t*, uint8_t*);
using AddOneEventWaitSqe = void (*)(uint16_t, int32_t, uint16_t, const uint8_t*, uint8_t*);
using AddOneRdmaDbSendSqe
    = void (*)(uint16_t, uint16_t, uint64_t, uint64_t, uint32_t, uint8_t, const uint8_t*, uint8_t*);
using AddOnePlaceHolderSqe = void (*)(uint16_t, uint16_t, uint16_t, const uint8_t*, uint8_t*);
using AddOneCacheMemcpyPlaceHolderSqe
    = void (*)(uint16_t, uint16_t, const void*, const void*, uint8_t, const uint8_t*, uint8_t*, uint32_t);
using AddOneCacheNotifyWaitPlaceholderSqe
    = void (*)(uint16_t, uint16_t, u64, const uint8_t*, uint8_t*, const dfx::DfxTimeOutConfig&);
using AddOneCacheNotifyRecordPlaceholderSqe = void (*)(uint16_t, uint16_t, u64, const uint8_t*, uint8_t*);
using AddOneCacheWriteValuePlaceholderSqe = void (*)(uint16_t, uint16_t, u64, const uint8_t*, uint8_t*);
using AddOneCacheMemcpyRecordPlaceholderSqe = void (*)(
    uint16_t, uint16_t, const void*, uint32_t, const aclDataType, aclrtReduceKind, const void*, uint32_t, uint32_t,
    uint32_t, u64, uint8_t, const uint8_t*, uint8_t*, uint32_t);

class DispatcherAiCpu : public DispatcherPub {
public:
    explicit DispatcherAiCpu(const u32 devPhyId);
    ~DispatcherAiCpu() override;
    HcclResult Init() override;
    HcclResult WaitValue(hccl::Stream& stream, u64 waitAddr, u64 valueAddr, bool reset) override;
    HcclResult WriteValue(hccl::Stream& stream, u64 writeAddr, u64 valueAddr) override;
    HcclResult SignalRecord(
        HcclRtNotify signal, hccl::Stream& stream, u32 userRank, u64 offset = INVALID_U64,
        s32 stage = INVALID_VALUE_STAGE, bool inchip = false, u64 signalAddr = INVALID_U64,
        u32 notifyId = INVALID_UINT) override;
    HcclResult SignalRecord(
        hccl::DeviceMem& dst, hccl::DeviceMem& src, hccl::Stream& stream, u32 remoteUserRank, hccl::LinkType inLinkType,
        u32 notifyId) override;
    HcclResult SignalWait(
        HcclRtNotify signal, hccl::Stream& stream, u32 userRank, u32 remoteUserRank, s32 stage = INVALID_VALUE_STAGE,
        bool inchip = false, u32 notifyId = INVALID_UINT, u32 timeOut = NOTIFY_INVALID_WAIT_TIME) override;
    HcclResult MemcpyAsync(
        hccl::DeviceMem& dst, const hccl::DeviceMem& src, hccl::Stream& stream,
        u32 remoteUserRank = INVALID_VALUE_RANKID, hccl::LinkType inLinkType = hccl::LinkType::LINK_ONCHIP) override;
    HcclResult ReduceAsync(
        const void* src, void* dst, u64 dataCount, const HcclDataType datatype, HcclReduceOp redOp, Stream& stream,
        HcclReduceType reduceType = HcclReduceType::HCCL_TBE_REDUCE) override;
    HcclResult InlineReduceAsync(
        const void* src, u64 dataCount, const HcclDataType datatype, HcclReduceOp redOp, Stream& stream, void* dst,
        u32 remoteUserRank = INVALID_VALUE_RANKID, hccl::LinkType inLinkType = hccl::LinkType::LINK_ONCHIP) override;
    HcclResult RdmaRecord(
        u32 dbindex, u64 dbinfo, const struct SendWr& wr, hccl::Stream& stream, RdmaType rdmaType, u32 userRank,
        u64 offset, u32 notifyId) override;

    HcclResult LaunchTasksEx(Stream& stream, std::vector<Stream>& subStreams) override;
    HcclResult LaunchAllTasks() override;

    HcclResult RdmaSend(u32 dbindex, u64 dbinfo, hccl::Stream& stream, RdmaTaskInfo& taskInfo) override;
    // 新增接口用于算子展开的动态缓存
    HcclResult ClearLaunchContext(); // 当前算子展开不需要使用动态缓存
    // 设置launch context, 在LaunchTask时用于算子展开动态缓存的admission
    // (因为需要在DispatcherAicpu中暂存AlltoallvMetadata, 所以传入指针而不是引用)
    HcclResult SetLaunchContext(
        const OpUnfoldKey& key, OpUnfoldCache* cachePtr, const std::vector<OpUnfoldMemRange>& userInputMemRanges,
        const std::vector<OpUnfoldMemRange>& userOutputMemRanges, const bool isAlltoallv,
        const AlltoallvMetadata* alltoallvMetadataPtr);
    // 缓存命中时, 使用缓存中的SQE信息下发给RTSQ
    HcclResult LaunchNewTask(
        OpUnfoldCacheEntry* entryPtr, const std::vector<OpUnfoldMemRange>& userInputMemRanges,
        const std::vector<OpUnfoldMemRange>& userOutputMemRanges, Stream& mainStream, std::vector<Stream>& slaveStreams,
        const bool profL1Enable, const bool isAlltoallv, const AlltoallvMetadata& alltoallvMetadata,
        const AlltoallvSendRecvInfo& alltoallvSendRecvInfo);

    HcclResult LaunchTask(Stream& stream, bool isBlockLaunch);
    HcclResult TbeReduceAsync(
        const void* src1, const void* src2, u64 count, const HcclDataType datatype, HcclReduceOp redOp, Stream& stream,
        const void* dst);
    HcclResult AddRetryPreamble(Stream& stream) override;
    HcclResult StreamSync(Stream& stream) override;

    void SetOpExecStatusCallback(std::function<HcclResult()> checkOpExecStatusCallback)
    {
        checkOpExecStatusCallback_ = checkOpExecStatusCallback;
        return;
    }

    void SetOpRingBufferIdx(const u32 opRingBufferIdx)
    {
        opRingBufferIdx_ = opRingBufferIdx;
        HCCL_INFO("[DispatcherAiCpu][SetOpRingBufferIdx]DFX opRingBufferIdx: [%u]", opRingBufferIdx);
        return;
    }

    void SetSqeTimeOut(const u64 timeOut)
    {
        if (timeOut > notifyMaxWaitTime_) {
            dfxTimeOutConfig_.sqeTimeOutTimeOut = notifyMaxWaitTime_;
            HCCL_WARNING(
                "[SetSqeTimeOut] timeOut[%llu s] exceeds the maximum allowed value "
                "for notifyMaxWaitTime[%u s].",
                timeOut, notifyMaxWaitTime_);
        } else {
            dfxTimeOutConfig_.sqeTimeOutTimeOut = timeOut;
        }
        HCCL_INFO(
            "[DispatcherAiCpu][SetSqeTimeOut]DFX timeout config init successfully with details: [%s]",
            dfxTimeOutConfig_.ToString().c_str());
        return;
    }

    void GetSqeTimeOut(u64& timeOut)
    {
        timeOut = dfxTimeOutConfig_.sqeWaitTimeOut;
        return;
    }

    HcclResult SetSqFullWaitTimeOut(u64 notifyWaitTime)
    {
        dfxTimeOutConfig_.sqFullWaitTimeOut
            = (notifyWaitTime == 0) ? notifyWaitTime : (notifyWaitTime + AICPU_RTSQ_TIMEOUT_INC);
        HCCL_INFO(
            "[DispatcherAiCpu][SetSqFullWaitTimeOut]DFX timeout config with details: [%s]",
            dfxTimeOutConfig_.ToString().c_str());
        return HCCL_SUCCESS;
    }
    HcclResult SignalRecord(Stream& stream, u64 notifyId)
    {
        return SignalRecord(
            nullptr, stream, INVALID_VALUE_RANKID, INVALID_U64, INVALID_VALUE_STAGE, true, INVALID_U64,
            static_cast<u32>(notifyId));
    }
    HcclResult SignalWait(Stream& stream, u32 notifyId, u32 timeOut)
    {
        return SignalWait(
            nullptr, stream, INVALID_VALUE_RANKID, INVALID_VALUE_RANKID, INVALID_VALUE_STAGE, true,
            static_cast<u32>(notifyId), timeOut);
    }

public:
    dfx::DfxTimeOutConfig dfxTimeOutConfig_ = {};
    uint32_t opRingBufferIdx_ = 0;

private:
    // 新增接口用于算子展开的动态缓存
    HcclResult WaitRtsq(
        Stream& stream, const size_t& sqeCount,
        const bool isBlockLaunch); // 等待RTSQ直到有sqeCount的SQE的空间 (与LaunchTask中相同的逻辑)
    HcclResult MemcpyRtsq(
        Stream& stream, const size_t sqeCount, const uint8_t* sqeArray, const uint8_t* sqeTypeArray,
        const AicpuDfxInfo* sqeDfxInfoArray, const bool profL1Enable, const std::vector<uint64_t>& profTimestamps,
        const size_t profTimestampStartIdx); // 将动态缓存中更新后的SQE的相关信息下发到RTSQ中

    HcclResult AddFlipTask(Stream& stream);
    HcclResult GetStreamSqeBufferAddr(
        hccl::Stream& stream, uint8_t*& sqeBufferAddr, uint8_t*& sqeTypeAddr, uint8_t*& sqeDfxInfoAddr,
        uint16_t& taskId);
    void SaveStreamInfo(hccl::Stream& stream);
    u64 CalcDbAddr(u32 dbindex);
    void InitTimeOutConfig();
    u32 GetMaxNotifyWaitTime() { return notifyMaxWaitTime_; }

    AddOneNotifyWaitSqe addOneNotifyWaitSqe_ = nullptr;
    AddOneRecordSqe addOneRecordSqe_ = nullptr;
    AddOneWriteValueRecordSqe addOneWriteValueRecordSqe_ = nullptr;
    AddOneMemcpySqe addOneMemcpySqe_ = nullptr;
    AddOneEventResetSqe addOneEventResetSqe_ = nullptr;
    AddOneEventRecordSqe addOneEventRecordSqe_ = nullptr;
    AddOneEventWaitSqe addOneEventWaitSqe_ = nullptr;
    AddOneRdmaDbSendSqe addOneRdmaDbSendSqe_ = nullptr;
    AddOnePlaceHolderSqe addOneFlipPlaceHolderSqe_ = nullptr;
    AddOneCacheMemcpyPlaceHolderSqe addOneCacheMemcpyPlaceHolderSqe_ = nullptr;
    AddOneCacheNotifyWaitPlaceholderSqe addOneCacheNotifyWaitPlaceholderSqe_ = nullptr;
    AddOneCacheNotifyRecordPlaceholderSqe addOneCacheNotifyRecordPlaceholderSqe_ = nullptr;
    AddOneCacheWriteValuePlaceholderSqe addOneCacheWriteValuePlaceholderSqe_ = nullptr;
    AddOneCacheMemcpyRecordPlaceholderSqe addOneCacheMemcpyRecordPlaceholderSqe_ = nullptr;
    std::function<HcclResult()> checkOpExecStatusCallback_ = nullptr;

    HcclAicpuDispatcherInfo aicpuInfo_;

    std::recursive_mutex streamMapMutex_;       // 保护streamMap_并发遍历/插入/删除
    std::unordered_map<s32, Stream> streamMap_; // 保存下过task的stream
    u64 notifySize_ = 0;

    // Launch context用于算子展开的动态缓存
    // 注意: cachePtr_初始化为空, needAddSqe_初始化为false, 即暂无算子展开的动态缓存
    OpUnfoldKey key_;                   // 当前展开算子的标识符
    OpUnfoldCache* cachePtr_ = nullptr; // 算子展开的动态缓存
    std::vector<OpUnfoldMemRange>
        userInputMemRanges_; // 当前算子展开执行时, 通信域内各rank分配的user input memory range
    std::vector<OpUnfoldMemRange>
        userOutputMemRanges_; // 当前算子展开执行时, 通信域内各rank分配的user output memory range
    bool isAlltoallv_ = false;
    const AlltoallvMetadata* alltoallvMetadataPtr_ = nullptr; // alltoallv算子对应的metadata (与通信域绑定)
    bool needAddSqe_ = false;
};
} // namespace hccl
#endif // HCCL_DISPATCHER_AICPU_PUB_H
