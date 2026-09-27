/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ALL_GATHER_RING_CONCURRENT_DIRECT_PUB_H
#define ALL_GATHER_RING_CONCURRENT_DIRECT_PUB_H

#include "alg_template_base_pub.h"
#include "reducer_pub.h"
#include "sender_pub.h"

namespace hccl {
class AllGatherRingConcurrentDirect : public AlgTemplateBase {
public:
    explicit AllGatherRingConcurrentDirect(const HcclDispatcher dispatcher);

    ~AllGatherRingConcurrentDirect() override;

    // should be called soon after template AllGatherMeshDirect instance created
    HcclResult Prepare(
        HcomCollOpInfo* opInfo, const u32 userRank, std::vector<Stream>& subStreams,
        const std::vector<std::shared_ptr<LocalNotify>>& mainSignals,
        const std::vector<std::shared_ptr<LocalNotify>>& subSignals, const std::vector<u32>& ringsOrder,
        const std::vector<Slice>& userMemSlices, bool isSdma = true) override;

    HcclResult RunAsync(const u32 rank, const u32 rankSize, const std::vector<LINK>& links) override;

protected:
private:
    HcclResult CheckParameters(const u32 rank, const u32 rankSize, const std::vector<LINK>& links);
    HcclResult OneRankMemcpy();
    HcclResult GetInitializedNeighborLinks(const u32 rank, const u32 rankSize, const std::vector<LINK>& links);
    HcclResult SetSlices(const u32 rank, const u32 rankSize);
    HcclResult RunInitStep(const u32 rank, const u32 rankSize);
    HcclResult RunSubStreamSlice(
        const u32 step, const u32 sliceIdx, const std::vector<Slice>& txSliceVector,
        const std::vector<Slice>& subSliceVector);
    HcclResult RunAllGather(u32 rank, u32 rankSize);
    HcclResult MainRecordSub();
    HcclResult SubWaitMain();
    HcclResult SubRecordMain();
    HcclResult MainWaitSub();

    LINK leftLink_;
    LINK rightLink_;

    HcomCollOpInfo* opInfo_{nullptr};
    u32 userRank_;
    std::vector<Stream> subStreams_;
    std::vector<std::shared_ptr<LocalNotify>> mainSignals_;
    std::vector<std::shared_ptr<LocalNotify>> subSignals_;
    std::vector<u32> ringsOrder_;
    std::vector<Slice> userMemOutputSlices_;
    std::vector<Slice> inputSlices_; // 需要吗？
    bool isSdma_;
};
} // namespace hccl

#endif /* ALL_GATHER_RING_CONCURRENT_DIRECT_PUB_H */
