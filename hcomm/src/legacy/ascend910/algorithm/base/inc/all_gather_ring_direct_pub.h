/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ALL_GATHER_RING_DIRECT_PUB_H
#define ALL_GATHER_RING_DIRECT_PUB_H

#include "alg_template_base_pub.h"
#include "reducer_pub.h"
#include "sender_pub.h"

namespace hccl {
class AllGatherRingDirect : public AlgTemplateBase {
public:
    explicit AllGatherRingDirect(const HcclDispatcher dispatcher);

    ~AllGatherRingDirect() override;

    HcclResult Prepare(
        HcomCollOpInfo* opInfo, u32 userRank, const std::vector<Slice>& userMemOutputSlices,
        bool isSdma = true) override;
    HcclResult RunAsync(const u32 rank, const u32 rankSize, const std::vector<LINK>& links) override;

protected:
private:
    HcclResult CheckParameters(const u32 rank, const u32 rankSize, const std::vector<LINK>& links);
    HcclResult OneRankMemcpy();
    HcclResult GetInitializedNeighborLinks(const u32 rank, const u32 rankSize, const std::vector<LINK>& links);
    HcclResult SetSlices(const u32 rank, const u32 rankSize);
    HcclResult RunInitStep(const u32 rank, const u32 rankSize);
    HcclResult RunAllGather(const u32 rank, const u32 rankSize);
    HcclResult RunAllGatherPartOne(const u32 sliceSize, const u32 step, const u32 txSliceIdx);
    HcclResult RunAllGatherPartTwo(
        const u32 sliceSize, const u32 step, const u32 txSliceIdx, const u32 rxSliceIdx, const u32 rankSize);

    LINK leftLink_;
    LINK rightLink_;

    std::vector<DeviceMem> finalSrc_;
    std::vector<DeviceMem> finalDst_;

    HcomCollOpInfo* opInfo_{nullptr};
    u32 userRank_ = 0;
    std::vector<Slice> userMemOutputSlices_;
    std::vector<Slice> inputSlices_;
    bool isSdma_ = false;
};
} // namespace hccl

#endif /* ALL_GATHER_RING_DIRECT_PUB_H */
