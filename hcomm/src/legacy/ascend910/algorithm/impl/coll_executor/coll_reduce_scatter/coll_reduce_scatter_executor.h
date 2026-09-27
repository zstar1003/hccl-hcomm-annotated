/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COLL_REDUCESCATTER_EXECUTOR_H
#define COLL_REDUCESCATTER_EXECUTOR_H
#include "coll_comm_executor.h"

namespace hccl {

constexpr u64 CCE_REDUCE_ALIGN_FACTOR = 2; // cce reduce数据大小32字节对齐  2是指前后各有

class CollReduceScatterExecutor : public CollCommExecutor {
public:
    explicit CollReduceScatterExecutor(const HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher>& topoMatcher);
    ~CollReduceScatterExecutor() override = default;

    HcclResult Orchestrate(OpParam& param, AlgResourceResponse& algRes) override;

protected:
    // ReduceScatter Loop Executor公共接口
    virtual u64 CalcLoopMaxCount(const u32 unitSize);
    virtual bool IsHugeData(const u64 curSize, OpParam* param = nullptr);
    virtual bool IsSmallData(const u64 totalSize, const u64 curSize);
    virtual bool IsPreloadCopyOptimizeCondition(const OpParam& param, ExecMem& execMem);
    virtual HcclResult RunLoop(OpParam& param, AlgResourceResponse& algRes);
    virtual HcclResult RunLoopV(OpParam& param, AlgResourceResponse& algRes);

    // 工具类
    std::vector<std::vector<Slice>> ReduceScatterRingSlicePrepare(
        u32 ringNum, u32 sliceNum, bool useInlineReduce, const DeviceMem& outputMem, std::vector<Slice>& dataSegsSlice,
        const std::string& tag);
    std::vector<std::vector<Slice>> ReduceScatterRingSlicePrepareContinuous(
        u32 ringNum, u32 sliceNum, bool useInlineReduce, DeviceMem& outputMem, u32 level1RankSize, u32 level2RankSize,
        std::vector<Slice>& dataSegsSlice, const std::string& tag);
    std::vector<std::vector<Slice>> AnyPathReduceScatterRingSlicePrepare(
        u32 ringNum, u32 sliceNum, bool useInlineReduce, DeviceMem& outputMem, std::vector<Slice>& dataSegsSlice,
        const std::string& tag);
    HcclResult PrepareAivBuffers(
        u32 rankSize, u32 rankId, u32 rankOffset, DeviceMem& inputMem, DeviceMem& outputMem, std::vector<LINK>& links,
        void** dataBuffers, void** flagBuffers, UserMemType dataMemType, UserMemType flagMemType, u32 dataMemOffset,
        u32 flagMemOffset) const;
    HcclResult RetryPostSync(OpParam& param, ExecMem& execMem);
    bool CCLMemSlice_{true};     // 每次Loop是否需要对CCLMem进行切片
    bool DMAReduceFlag_{false};  // 是否DMA消减
    bool scratchMemFlag_{false}; // 是否需要申请scratch memory，不需要申请则传入outputmem为scratchmem
    u64 totalSize_{0};           // 总数据量
    bool isReduceScatterV_{false};

private:
    HcclResult RunLoopInner(OpParam& param, const ReduceType& reduceType, ExecMem& execMem);
    HcclResult RunLoopInnerV(OpParam& param, const ReduceType& reduceType, ExecMem& execMem);

    bool CalcCurCountsAndCurDispls(
        const u64 maxTotalCount, std::vector<u64>& countsLeft, std::vector<u64>& displs, std::vector<u64>& curCounts,
        std::vector<u64>& curDispls, u32 unitSize) const;
    void PrintCurCountAndCurDispls(const std::vector<u64>& curCounts, const std::vector<u64>& curDispls) const;
};

} // namespace hccl

#endif
