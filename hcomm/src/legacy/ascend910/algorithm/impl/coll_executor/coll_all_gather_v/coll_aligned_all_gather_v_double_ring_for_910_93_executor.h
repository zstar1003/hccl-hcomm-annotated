/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COLL_ALIGNED_ALLGATHER_V_RING_FOR_910_93_EXECUTOR_H
#define COLL_ALIGNED_ALLGATHER_V_RING_FOR_910_93_EXECUTOR_H
#include "coll_aligned_all_gather_double_ring_for_910_93_executor.h"
namespace hccl {
class CollAlignedAllGatherVDoubleRingFor91093Executor : public CollAlignedAllGatherDoubleRingFor91093Executor {
public:
    CollAlignedAllGatherVDoubleRingFor91093Executor(
        const HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher>& topoMatcher);
    ~CollAlignedAllGatherVDoubleRingFor91093Executor() override = default;

private:
    bool IsSmallData(const u64 size) override;

    u64 CalcDstMemOffset(const OpParam& param, u32 perDataSize, u64 inputMemSize) const override;
    HcomCollOpInfo GetHcomCollOpInfo(const OpParam& param, const ExecMem& execMem) const override;
    std::vector<Slice> PrepareSlicesL2(
        const OpParam& param, const SubCommInfo& level2CommInfo, const SubCommInfo& level1CommInfo,
        const SubCommInfo& level0CommInfo, u32 perDataSize, u64 inputMemSize) const override;
    std::vector<Slice> PrepareSlicesL1(
        const OpParam& param, const SubCommInfo& level2CommInfo, const SubCommInfo& level1CommInfo,
        const SubCommInfo& level0CommInfo, u32 perDataSize, u64 inputMemSize) const override;
    HcclResult PrepareSlicesL0(
        std::vector<std::vector<Slice>>& multRingsSlice, const OpParam& param, const SubCommInfo& level2CommInfo,
        const SubCommInfo& level1CommInfo, const SubCommInfo& level0CommInfo, u32 perDataSize,
        u64 inputMemSize) override;
    HcclResult PrepareUserMemSlices(
        std::vector<std::vector<Slice>>& userMemSlices, const std::vector<std::vector<Slice>>& multRingsSlice,
        const OpParam& param, const SubCommInfo& level2CommInfo, const SubCommInfo& level1CommInfo,
        const SubCommInfo& level0CommInfo, u32 perDataSize, u64 inputMemSize) override;
};

} // namespace hccl

#endif
