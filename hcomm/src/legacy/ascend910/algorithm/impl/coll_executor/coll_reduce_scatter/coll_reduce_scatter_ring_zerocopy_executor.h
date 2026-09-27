/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COLL_REDUCESCATTER_RING_ZEROCOPY_EXECUTOR_H
#define COLL_REDUCESCATTER_RING_ZEROCOPY_EXECUTOR_H
#include "coll_reduce_scatter_executor.h"

namespace hccl {
class CollReduceScatterRingZerocopyExecutor : public CollReduceScatterExecutor {
public:
    explicit CollReduceScatterRingZerocopyExecutor(
        const HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher>& topoMatcher);
    ~CollReduceScatterRingZerocopyExecutor() override = default;

protected:
    HcclResult CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport) override;
    u32 level0Rank_ = INVALID_VALUE_RANKID;
    u32 level1Rank_ = INVALID_VALUE_RANKID;
    u32 level2Rank_ = INVALID_VALUE_RANKID;
    u32 level0RankSize_ = INVALID_VALUE_RANKSIZE;
    u32 level1RankSize_ = INVALID_VALUE_RANKSIZE;
    u32 level2RankSize_ = INVALID_VALUE_RANKSIZE;

private:
    void ParseParam(const OpParam& param) override;
    /* *************** 资源计算 *************** */
    std::set<u32> commTargetUserRankSet_;
    HcclResult CalcStreamNum(u32& streamNum) override;
    HcclResult CalcLevel0CommInfo(
        TransportMemType inputType, TransportMemType outputType,
        std::vector<LevelNSubCommTransport>& opTransport) override;
    HcclResult CalcTransportMemType(TransportMemType& inputType, TransportMemType& outputType) const;
    u64 CalcLoopMaxCount(const u32 unitSize) override;

    /* *************** 算法编排 *************** */
    HcclResult SemiRingReduceScatter(
        const std::string& tag, DeviceMem inputMem, DeviceMem outputMem, const u64 count, const HcclDataType dataType,
        const HcclReduceOp reductionOp, const std::vector<std::vector<Slice>> multRingsSliceZero, Stream stream,
        s32 profStage, const u64 baseOffset, const HcomCollOpInfo* opInfo,
        const std::vector<std::vector<Slice>> multRingsUserMemSlice);
    HcclResult RunIntraSeverReduceScatter(
        const std::string& tag, DeviceMem& inputMem, DeviceMem& outputMem, const u64 count,
        const HcclDataType& dataType, const HcclReduceOp& reductionOp,
        const std::vector<std::vector<Slice>>& multRingsSliceZero, const Stream& stream, s32 profStage,
        const u64 baseOffset = 0, const HcomCollOpInfo* opInfo = nullptr,
        const std::vector<std::vector<Slice>>& multRingsUserMemSlice = std::vector<std::vector<Slice>>(0),
        const bool disableDMAReduce = false);
    HcclResult KernelRunIntraServerPre(const OpParam& param, ExecMem& execMem) override;
    HcclResult KernelRunInterServer(const OpParam& param, ExecMem& execMem) override;
    virtual HcclResult KernelRunInterServerPostProcess(const OpParam& param, const ExecMem& execMem);

    virtual HcclResult
    CalcLevel0DataSlices(const OpParam& param, const ExecMem& execMem, std::vector<Slice>& dataSegsSlice);
    virtual HcclResult KernelRunInterServerPreProcess(const OpParam& param, const ExecMem& execMem);
};

} // namespace hccl

#endif
