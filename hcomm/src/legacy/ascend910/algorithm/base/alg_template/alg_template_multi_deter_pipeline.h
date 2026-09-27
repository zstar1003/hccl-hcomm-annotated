/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ALG_TEMPLATE_MULTI_DETER_PIPELINE_H
#define ALG_TEMPLATE_MULTI_DETER_PIPELINE_H

#include <vector>
#include <memory>
#include <hccl/hccl_types.h>
#include "hccl/base.h"
#include "externalinput_pub.h"
#include "mem_device_pub.h"
#include "stream_pub.h"
#include "dispatcher.h"
#include "alg_template_base_pub.h"

namespace hccl {
constexpr u32 STEP_OFFSET_TWO = 2;
// 上游保证最多4条流做规约操作，4条做localreduce，其中1条做localreduce主流
constexpr u32 MAX_REDUCE_STREAM_NUM = 4;
constexpr u32 MIN_SERVER_NUM = 2;
constexpr u32 MIN_INTRA_RANK_NUM = 3;
constexpr u32 SECOND_TO_LAST = 2;
constexpr u32 LOCAL_REDUCE_SERIIAL_ALG_SERVER_NUM = 2;
constexpr u32 PARITY_BASE = 2;
class MultiDeterPipeline : public AlgTemplateBase {
public:
    explicit MultiDeterPipeline(const HcclDispatcher dispatcher);
    ~MultiDeterPipeline() override;
    HcclResult RunAsync() override;
    HcclResult RunAsyncReduceScatterPipeline();
    // ReduceScatterDeterPipeline
    HcclResult Prepare(
        HcomCollOpInfo* opInfo, DeviceMem& buffer, const u64 count, const u64 offset, const std::vector<Slice>& slices,
        const SubCommInfo& level0CommInfo, const SubCommInfo& level1CommInfo, Stream& mainStream,
        std::vector<Stream>& subStream, std::vector<std::shared_ptr<LocalNotify>>& notifyMain,
        std::vector<std::shared_ptr<LocalNotify>>& notifySub) override;

    // AllReduceDeterPipeline
    HcclResult Prepare(
        HcomCollOpInfo* opInfo, DeviceMem& inBuffer, DeviceMem& outBuffer, const u64 count,
        const std::vector<Slice>& slices, const SubCommInfo& level0CommInfo, const SubCommInfo& level1CommInfo,
        Stream& mainStream, std::vector<Stream>& subStream, std::vector<std::shared_ptr<LocalNotify>>& notifyMain,
        std::vector<std::shared_ptr<LocalNotify>>& notifySub) override;

protected:
    HcclResult MainWaitSub(u32 begin, u32 end);
    HcclResult SubRecordMain(u32 begin, u32 end);
    HcclResult MainRecordSub(u32 begin, u32 end);
    HcclResult SubWaitMain(u32 begin, u32 end);
    // 根据step获取 机间或机内的rankId
    constexpr u32 GetPreRankIdByStep(u32 rankId, u32 rankSize, u32 step) const
    {
        return (rankId + rankSize - step) % rankSize;
    }

    constexpr u32 GetNextRankIdByStep(u32 rankId, u32 rankSize, u32 step) const { return (rankId + step) % rankSize; }

    inline u32 GetPreServerIdByStep(u32 step) { return GetPreRankIdByStep(serverId_, serverSize_, step); }

    inline u32 GetNextServerIdByStep(u32 step) { return GetNextRankIdByStep(serverId_, serverSize_, step); }

    inline u32 GetPreIntraRankIdByStep(u32 step) { return GetPreRankIdByStep(intraRankId_, intraRankSize_, step); }

    inline u32 GetNextIntraRankIdByStep(u32 step) { return GetNextRankIdByStep(intraRankId_, intraRankSize_, step); }

    inline u32 GetRankIdx(u32 serverId, u32 intraRankId) const { return serverId * intraRankSize_ + intraRankId; }
    // 获取device内存部分
    virtual HcclResult
    GetRemoteCclbufferDeviceMem(u32 inputSliceIndex, LINK link, u32 outputSliceIndex, DeviceMem& remoteMem);
    virtual HcclResult GetLocalUserInDeviceMem(u32 rankIdInAllRanks, DeviceMem& locaMem);
    virtual HcclResult GetLocalUserOutDeviceMem(u32 rankIdInAllRanks, DeviceMem& localMem);
    virtual HcclResult GetLocalInCclbufferDeviceMem(u32 rankIdInAllRanks, DeviceMem& localMem, bool ifUseLastSize);
    virtual HcclResult GetLocalOutCclbufferDeviceMem(u32 rankIdInAllRanks, DeviceMem& localMem, bool ifUseLastSize);

    virtual HcclResult RunLocalCopy();
    virtual HcclResult RunIntraAlltoallPreSync(u32 step);
    HcclResult RunIntraAlltoall(u32 step);
    // LocalReduce内部函数
    HcclResult GroupTasksByStream(
        u32 activeCount, const std::vector<bool>& isReduceBlock, u32 retIndex,
        std::vector<std::vector<std::vector<std::pair<u32, u32>>>>& batchStreamTasks, std::vector<bool>& processed,
        std::vector<u32>& origIdxMap, u32& newActiveCount) const;
    HcclResult ExecuteStreamTasks(
        const std::vector<std::vector<std::pair<u32, u32>>>& streamTasks, const std::vector<DeviceMem>& validMem,
        std::vector<u32>& origIdxMap, bool useMainStream);
    virtual HcclResult BatchPostNotifyForStreams(
        const std::vector<std::vector<std::pair<u32, u32>>>& streamTasks, bool isStartPhase, bool useMainStream);
    void CompressActiveSet(
        std::vector<DeviceMem>& validMem, std::vector<bool>& isReduceBlock, std::vector<u32>& origIdxMap,
        const std::vector<bool>& processed, u32& trackedTargetIdx, const u32 origRetIndex) const;
    HcclResult
    LocalReduce(std::vector<DeviceMem>& reduceMem, std::vector<bool>& isReduceBlock, u32 retIndex, bool useMainStream);
    virtual HcclResult RunIntraLocalReduce(u32 step);
    virtual HcclResult RunFinalReduce();
    // RDAM send部分
    virtual HcclResult RunInterSend(u32 step);
    // 主从流同步部分
    virtual HcclResult AlltoallSync(u32 step, bool isStartPhase);
    virtual HcclResult LocalReduceSync(u32 step, bool isStartPhase);
    HcclResult AlltoallLocalReduceSync(u32 step, bool isStartPhase);
    // local reduce串行算法
    HcclResult RunAsyncLocalReduceSerial();
    // 初始化部分
    void InitAlltoallRecvBlockIdxMap();
    HcclResult PrepareTopoInfo(const SubCommInfo& level0CommInfo, const SubCommInfo& level1CommInfo);
    virtual u64 GetLocalReduceSerialThresh() = 0;

    HcomCollOpInfo* opInfo_{nullptr};

    void* usrInMemPtr_ = nullptr;
    void* usrOutMemPtr_ = nullptr;
    u64 count_ = 0; // output中的数量
    u32 unitSize_ = 0;
    u64 curSize_ = 0;
    u64 memSliceSize_ = 0;
    u64 blockSize_ = 0;
    u64 bufferSize_ = 0;
    HcclReduceOp reductionOp_ = HcclReduceOp::HCCL_REDUCE_RESERVED;
    HcclDataType dataType_ = HcclDataType::HCCL_DATA_TYPE_RESERVED;

    std::vector<Stream> subStreams_;
    u32 subStreamNum_ = 0;
    Stream mainStream_;

    std::vector<std::shared_ptr<LocalNotify>> streamNotifyMain_;
    std::vector<std::shared_ptr<LocalNotify>> streamNotifySub_;

    u32 all2allStreamBegin_ = 0; // all2all专用
    u32 all2allStreamSize_ = 0;
    u32 reduceMainStreamIdx_ = 0;
    u32 reduceStreamBegin_ = 0; // local reduce专用
    u32 reduceStreamSize_ = 0;
    u32 intraRankSize_ = 0; // 机内
    u32 serverSize_ = 0;    // 机间
    u32 intraRankId_ = 0;   // 机内
    u32 serverId_ = 0;      // 机间
    u64 offset_ = 0;
    u32 allSteps_ = 0;
    u64 eachRankCclbufferSize_ = 0;

    u32 userRankSize_ = 0;
    u32 userRank_ = 0;
    std::vector<std::vector<u32>> alltoallRecvBlockIdxMap_; // alltoall接收block的idx映射表
    // 本地cclbuffer的偏移
    // allreduce为了保证地址对齐，进行数据分块时除了最后一块数据
    // 其他分块都向上取HCCL_MIN_SLICE_ALIGN_910B倍数的大小，最后一块数据取剩余的大小。
    // reduce scatter每块大小相同向上直接HCCL_MIN_SLICE_ALIGN_910B取整。
    std::vector<Slice> slices_;
    std::vector<LINK> intraLinks_;
    std::vector<LINK> serverLinks_;
};
} // namespace hccl
#endif
