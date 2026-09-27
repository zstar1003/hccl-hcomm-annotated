/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COLL_BROADCAST_RING_ZEROCOPY_EXECUTOR_H
#define COLL_BROADCAST_RING_ZEROCOPY_EXECUTOR_H
#include "coll_broadcast_executor.h"
namespace hccl {
class CollBroadCastRingZerocopyExecutor : public CollBroadcastExecutor {
public:
    CollBroadCastRingZerocopyExecutor(const HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher>& topoMatcher);
    ~CollBroadCastRingZerocopyExecutor() override = default;

private:
    /* *************** 资源计算 *************** */
    HcclResult CalcStreamNum(u32& streamNum) override;
    HcclResult CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport) override;
    HcclResult CalcLevel0CommInfo(
        TransportMemType inputType, TransportMemType outputType,
        std::vector<LevelNSubCommTransport>& opTransport) override;
    HcclResult SetCommRankInfo();

    /* *************** 算法编排 *************** */
    HcclResult
    KernelRunInterServerBroadcastSingleSuperpod(const OpParam& param, ExecMem& execMem, const u64 level1DataSize);
    HcclResult
    KernelRunInterServerBroadcastMultiSuperpod(const OpParam& param, ExecMem& execMem, const u64 level1DataSize);
    HcclResult KernelRunIntraServerPre(const OpParam& param, ExecMem& execMem) override;
    HcclResult KernelRunInterServer(const OpParam& param, ExecMem& execMem) override;
    HcclResult KernelRunIntraServerPost(const OpParam& param, ExecMem& execMem) override;

    HcclResult DoubleRingScatter(
        const std::string& tag, DeviceMem inputMem, DeviceMem outputMem, const u64 count, const HcclDataType dataType,
        const std::vector<std::vector<Slice>> multRingsSliceZero, u32 root, Stream stream, HcomCollOpInfo* opInfo,
        const u64 baseOffset = 0);
    HcclResult DoubleRingAllGather(
        const std::string& tag, DeviceMem inputMem, DeviceMem outputMem, const u64 count, const HcclDataType dataType,
        const std::vector<std::vector<Slice>> multRingsSliceZero, Stream stream, s32 profStage, const u64 baseOffset,
        HcomCollOpInfo* opInfo, const std::vector<std::vector<Slice>> multRingsUserMemSlice);

    u32 level0Rank_ = INVALID_VALUE_RANKID;
    u32 level1Rank_ = INVALID_VALUE_RANKID;
    u32 level2Rank_ = INVALID_VALUE_RANKID;
    u32 level0RankSize_ = INVALID_VALUE_RANKSIZE;
    u32 level1RankSize_ = INVALID_VALUE_RANKSIZE;
    u32 level2RankSize_ = INVALID_VALUE_RANKSIZE;
    std::vector<std::vector<Slice>> level0MultiRingDataSlices_;
};
} // namespace hccl

#endif
