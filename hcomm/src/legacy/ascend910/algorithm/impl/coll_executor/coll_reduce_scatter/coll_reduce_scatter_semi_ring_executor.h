/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COLL_REDUCE_SCATTER_DOUBLE_RING_MID_COUNT_EXECUTOR_H
#define COLL_REDUCE_SCATTER_DOUBLE_RING_MID_COUNT_EXECUTOR_H
#include "coll_reduce_scatter_ring_for_910_93_executor.h"
namespace hccl {
class CollReduceScatterSemiRingExecutor : public CollReduceScatterRingFor91093Executor {
public:
    explicit CollReduceScatterSemiRingExecutor(
        const HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher>& topoMatcher);
    ~CollReduceScatterSemiRingExecutor() override = default;

private:
    void ParseParam(const OpParam& param) override;
    /* *************** 资源计算 *************** */
    HcclResult CalcNotifyNum(u32 streamNum, u32& notifyNum) override;
    HcclResult CalcStreamNum(u32& streamNum) override;
    HcclResult CalcLevel0CommInfo(
        TransportMemType inputType, TransportMemType outputType,
        std::vector<LevelNSubCommTransport>& opTransport) override;

    /* *************** 算法编排 *************** */
    HcclResult DoubleRingMidCountReduceScatter(
        const std::string& tag, DeviceMem inputMem, DeviceMem outputMem, const u64 count, const HcclDataType dataType,
        const HcclReduceOp reductionOp, const std::vector<std::vector<Slice>> multRingsSliceZero, Stream stream,
        s32 profStage, const u64 baseOffset = 0, const HcomCollOpInfo* opInfo = nullptr,
        const std::vector<std::vector<Slice>> multRingsUserMemSlice = std::vector<std::vector<Slice>>(0),
        const bool retryEnable = false);
    HcclResult RunIntraSeverReduceScatter(
        const std::string& tag, DeviceMem& inputMem, DeviceMem& outputMem, const u64 count,
        const HcclDataType& dataType, const HcclReduceOp& reductionOp,
        const std::vector<std::vector<Slice>>& multRingsSliceZero, const Stream& stream, s32 profStage,
        const u64 baseOffset = 0, const HcomCollOpInfo* opInfo = nullptr,
        const std::vector<std::vector<Slice>>& multRingsUserMemSlice = std::vector<std::vector<Slice>>(0),
        const bool retryEnable = false) override;
};

} // namespace hccl

#endif
