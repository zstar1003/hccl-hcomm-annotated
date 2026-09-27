/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef REDUCE_SCATTER_MULTI_DETER_PIPELINE_H
#define REDUCE_SCATTER_MULTI_DETER_PIPELINE_H

#include "alg_template_multi_deter_pipeline.h"

namespace hccl {
class ReduceScatterMultiDeterPipeline : public MultiDeterPipeline {
public:
    explicit ReduceScatterMultiDeterPipeline(const HcclDispatcher dispatcher);
    ~ReduceScatterMultiDeterPipeline() override;

    // 适配新CollExecutor接口
    HcclResult Prepare(
        HcomCollOpInfo* opInfo, DeviceMem& cclBuffer, const u64 count, const u64 offset,
        const std::vector<Slice>& slices, const SubCommInfo& level0CommInfo, const SubCommInfo& level1CommInfo,
        Stream& mainStream, std::vector<Stream>& subStream, std::vector<std::shared_ptr<LocalNotify>>& notifyMain,
        std::vector<std::shared_ptr<LocalNotify>>& notifySub) override;
    HcclResult RunAsync() override;

protected:
    // 获取device内存部分
    HcclResult
    GetRemoteCclbufferDeviceMem(u32 inputSliceIndex, LINK link, u32 outputSliceIndex, DeviceMem& remoteMem) override;
    HcclResult GetLocalUserInDeviceMem(u32 rankIdInAllRanks, DeviceMem& localMem) override;
    HcclResult GetLocalCclbufferDeviceMem(u32 rankIdInAllRanks, DeviceMem& localMem, u64 sliceOffset);
    HcclResult RunLocalCopy() override;
    HcclResult RunIntraAlltoallPreSync(u32 step) override;
    // LocalReduce内部函数
    HcclResult BatchPostNotifyForStreams(
        const std::vector<std::vector<std::pair<u32, u32>>>& streamTasks, bool isStartPhase,
        bool useMainStream) override;
    HcclResult RunIntraLocalReduce(u32 step) override;
    HcclResult RunFinalReduce() override;
    // RDAM send部分
    HcclResult RunInterSend(u32 step) override;
    // 主从流同步部分
    HcclResult AlltoallSync(u32 step, bool isStartPhase) override;
    HcclResult LocalReduceSync(u32 step, bool isStartPhase) override;
    u64 GetLocalReduceSerialThresh() override;

    DeviceMem cclBuffer_;
    u64 eachRankCclbufferSize_ = 0;
};
} // namespace hccl

#endif /* REDUCE_SCATTER_MULTI_DETER_PIPELINE_H */
