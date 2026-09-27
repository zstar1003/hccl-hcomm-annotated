/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COLL_REDUCESCATTERV_EXECUTOR_H
#define COLL_REDUCESCATTERV_EXECUTOR_H
#include "coll_comm_executor.h"

namespace hccl {

class CollReduceScatterVExecutor : public CollCommExecutor {
public:
    explicit CollReduceScatterVExecutor(const HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher>& topoMatcher);
    ~CollReduceScatterVExecutor() override = default;

    HcclResult Orchestrate(OpParam& param, AlgResourceResponse& algRes) override;
    HcclResult GetAdjInfo(AlgResourceResponse& algRes, AdjInfo& adjInfo) override;

protected:
    // ReduceScatter Loop Executor公共接口
    virtual u64 CalcLoopMaxCount(const u32 unitSize);
    virtual HcclResult CalcCurCountsAndCurDispls(
        const u64 maxTotalCount, std::vector<u64>& countsLeft, std::vector<u64>& displs, std::vector<u64>& curCounts,
        std::vector<u64>& curDispls, bool& finished);
    virtual bool IsHugeData(const u64 curSize, const OpParam& param);
    virtual HcclResult RunLoop(OpParam& param, AlgResourceResponse& algRes);

    // 工具类
    void PrintCurCountAndCurDispls(const std::vector<u64>& curCounts, const std::vector<u64>& curDispls) const;

    bool CCLMemSlice_{true};     // 每次Loop是否需要对CCLMem进行切片
    bool DMAReduceFlag_{false};  // 是否DMA消减
    bool scratchMemFlag_{false}; // 是否需要申请scratch memory，不需要申请则传入outputmem为scratchmem
    u64 totalSize_{0};           // 总数据量

private:
    HcclResult RunLoopInner(OpParam& param, const ReduceType& reduceType, ExecMem& execMem);
};

} // namespace hccl

#endif
