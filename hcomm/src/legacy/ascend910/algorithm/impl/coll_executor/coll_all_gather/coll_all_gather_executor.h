/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COLL_ALLGATHER_EXECUTOR_H
#define COLL_ALLGATHER_EXECUTOR_H
#include "coll_comm_executor.h"
namespace hccl {
class CollAllGatherExecutor : public CollCommExecutor {
public:
    explicit CollAllGatherExecutor(const HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher>& topoMatcher);
    ~CollAllGatherExecutor() override = default;

    HcclResult Orchestrate(OpParam& param, AlgResourceResponse& algRes) override;

protected:
    // AllGather Loop Executor公共接口
    virtual u64 CalcLoopMaxCount(const u64 cclBuffSize, const u32 unitSize);
    virtual bool IsHugeData(const u64 curSize);
    virtual bool IsSmallData(const u64 size);
    virtual u64 CalcTotalCount(const OpParam& param) const;
    HcclResult RunLoop(OpParam& param, AlgResourceResponse& algRes); // non-virtual

    // agv
    virtual std::vector<u64> GetCounts(const OpParam& param) const;
    virtual std::vector<u64> GetDispls(const OpParam& param) const;
    virtual u64 GetCurrentCount(const OpParam& param, const std::vector<u64>& curCounts) const;
    virtual u64 CalcCurrentTotalCount(const OpParam& param, const std::vector<u64>& curCounts) const;
    virtual HcclOpMetaInfoDef GetOpMetaInfo(u32 algTypeLevel1, bool hugeData, bool smallData, bool dataSplit) const;
    virtual void UpdateOpParam(OpParam& param, std::vector<u64>& curCounts, std::vector<u64>& curDispls) const;
    bool CalcCountsDispls(
        const u64 maxTotalCount, std::vector<u64>& countsLeft, std::vector<u64>& displs, std::vector<u64>& curCounts,
        std::vector<u64>& curDispls);
    void PrintCountsDispls(bool finished, const std::vector<u64>& curCounts, const std::vector<u64>& curDispls) const;
    HcclResult RunLoopV(OpParam& param, AlgResourceResponse& algRes); // non-virtual

    // 工具类
    HcclResult PrepareAllgatherSlice(u32 sliceNum, u64 inputMemSize, std::vector<Slice>& dataSegsSlice) const;

    HcclResult CalculateLevel1AllgatherSlice(
        u64 inputMemSize, u32 level0RankSize, u32 level1RankSize, std::vector<std::vector<Slice>> multRingsSliceZero,
        std::vector<std::vector<Slice>>& multRingsSlice) const;

    HcclResult CalculateLevel2AllgatherSlice(
        u64 inputMemSize, u32 level0RankSize, u32 level1RankSize, u32 level2RankSize,
        std::vector<std::vector<Slice>> multRingsSliceZero, std::vector<Slice>& level2DataSlice, u32 ringIndex) const;

    HcclResult AllGatherLevel2(
        const std::string& tag, DeviceMem& inputMem, DeviceMem& outputMem, u64 count, HcclDataType dataType,
        Stream& stream, HcomCollOpInfo* opInfo = nullptr);

    bool DMAReduceFlag_{false}; // 是否DMA消减的标志
    bool isAllGatherV_{false};
};

} // namespace hccl

#endif
