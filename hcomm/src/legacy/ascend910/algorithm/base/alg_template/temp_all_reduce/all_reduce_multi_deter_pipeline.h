/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ALL_REDUCE_MULTI_DETER_PIPELINE_H
#define ALL_REDUCE_MULTI_DETER_PIPELINE_H

#include "alg_template_multi_deter_pipeline.h"

namespace hccl {
class AllReduceMultiDeterPipeline : public MultiDeterPipeline {
public:
    explicit AllReduceMultiDeterPipeline(const HcclDispatcher dispatcher);
    ~AllReduceMultiDeterPipeline() override;

    // 适配新CollExecutor接口
    HcclResult Prepare(
        HcomCollOpInfo* opInfo, DeviceMem& inBuffer, DeviceMem& outBuffer, const u64 count,
        const std::vector<Slice>& slices, const SubCommInfo& level0CommInfo, const SubCommInfo& level1CommInfo,
        Stream& mainStream, std::vector<Stream>& subStream, std::vector<std::shared_ptr<LocalNotify>>& notifyMain,
        std::vector<std::shared_ptr<LocalNotify>>& notifySub) override;
    HcclResult RunAsyncAllgatherPipeline();
    HcclResult RunAsync() override;

protected:
    // 获取device内存部分
    HcclResult
    GetRemoteCclbufferDeviceMem(u32 inputSliceIndex, LINK link, u32 outputSliceIndex, DeviceMem& remoteMem) override;
    HcclResult GetLocalUserDeviceMem(u32 rankIdInAllRanks, DeviceMem& localMem, bool isUserIn);
    HcclResult GetLocalUserInDeviceMem(u32 rankIdInAllRanks, DeviceMem& localMem) override;
    HcclResult GetLocalUserOutDeviceMem(u32 rankIdInAllRanks, DeviceMem& localMem) override;
    HcclResult GetLocalInCclbufferDeviceMem(u32 rankIdInAllRanks, DeviceMem& localMem, bool ifUseLastSize) override;
    HcclResult GetLocalOutCclbufferDeviceMem(u32 rankIdInAllRanks, DeviceMem& localMem, bool ifUseLastSize) override;

    HcclResult RunLocalCopy() override;
    HcclResult RunIntraAlltoallPreSync(u32 step) override;
    // LocalReduce内部函数
    HcclResult BatchPostNotifyForStreams(
        const std::vector<std::vector<std::pair<u32, u32>>>& streamTasks, bool isStartPhase,
        bool useMainStream) override;
    bool IfUseLastSize(u32 step, u32 sendServerId);
    HcclResult RunIntraLocalReduce(u32 step) override;
    HcclResult RunFinalReduce() override;
    // RDMA send部分
    HcclResult RunInterSend(u32 step) override;
    // 主从流同步部分
    HcclResult AlltoallSync(u32 step, bool isStartPhase) override;
    HcclResult LocalReduceSync(u32 step, bool isStartPhase) override;
    // allgather部分
    HcclResult RunAllGatherInterServer(u32 step, const LINK& prevInterLink, const LINK& nextInterLink);
    HcclResult RunAllGatherIntraServer(u32 step);

    u64 GetLocalReduceSerialThresh() override;
    // 初始化部分
    u64 lastSize_ = 0;
    bool isLastRank_ = false;
    u8 serverSizeParity_ = 0;
    DeviceMem outCclBuffer_;
    DeviceMem inCclBuffer_;
    u64 perRankAvgDataSize_ = 0;
};
} // namespace hccl

#endif /* ALL_REDUCE_MULTI_DETER_PIPELINE_H */
