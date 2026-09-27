/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ALIGNED_ALL_GATHER_DOUBLE_RING_PUB_H
#define ALIGNED_ALL_GATHER_DOUBLE_RING_PUB_H

#include "alg_template_base_pub.h"

namespace hccl {
class AlignedAllGatherDoubleRing : public AlgTemplateBase {
public:
    explicit AlignedAllGatherDoubleRing(const HcclDispatcher dispatcher);

    ~AlignedAllGatherDoubleRing() override;

    //  should be called soon after template AlignedAllGatherDoubleRing instance created
    HcclResult Prepare(
        HcomCollOpInfo* opInfo, const u32 userRank, std::vector<Stream>& subStreams,
        std::vector<std::shared_ptr<LocalNotify>>& mainSignals, std::vector<std::shared_ptr<LocalNotify>>& subSignals,
        const std::vector<std::vector<u32>>& ringsOrders,
        const std::vector<std::vector<Slice>>& userMemOutputSlicesOfDoubleRing) override;

    HcclResult RunAsync(const u32 rank, const u32 rankSize, const std::vector<LINK>& links) override;

protected:
private:
    HcclResult CheckParameters(const u32 rank, const u32 rankSize, const std::vector<LINK>& links);
    HcclResult OneRankMemcpy();
    HcclResult GetInitializedNeighborLinks(const u32 rank, const u32 rankSize, const std::vector<LINK>& links);
    HcclResult SetSlices(const u32 rank, const u32 rankSize);
    HcclResult RunInitStep(const u32 rank, const u32 rankSize);
    virtual HcclResult PrepareRunMainStream(u32 ringIndex, Stream& stream, LINK& preLink, LINK& nextLink);
    HcclResult RunAllStreams(
        const u32 step, const u32 rankSize, const std::vector<TxMemoryInfo>& mainTxMems,
        std::vector<RxMemoryInfo>& mainRxMems, const std::vector<TxMemoryInfo>& subTxMems,
        std::vector<RxMemoryInfo>& subRxMems, std::vector<DeviceMem>& mainLocalSrcMems,
        std::vector<DeviceMem>& mainLocalDstMems, std::vector<DeviceMem>& subLocalSrcMems,
        std::vector<DeviceMem>& subLocalDstMems);
    HcclResult RxAsyncMemcpy(const u32 step, const u32 ringIndex, RxMemoryInfo& mem, Stream& stream, LINK& link);
    HcclResult LocalMemcpy(const u32 ringIndex, DeviceMem& localSrcMem, DeviceMem& localDstMem);
    HcclResult PrepareDeviceMems(
        const u32 step, const u32 ringIndex, const u32 rankSize, const u32 txSliceIdx, const u32 rxSliceIdx,
        std::vector<TxMemoryInfo>& txMems, std::vector<RxMemoryInfo>& rxMems, std::vector<DeviceMem>& localSrcMems,
        std::vector<DeviceMem>& localDstMems);
    HcclResult RunAllGather(const u32 rank, const u32 rankSize);
    HcclResult ExecEmptyTasks();
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
    std::vector<std::vector<u32>> ringsOrders_;
    std::vector<Slice> userMemOutputSlices_;
    std::vector<std::vector<Slice>> userMemOutputSlicesOfDoubleRing_;
};
} // namespace hccl

#endif /* ALIGNED_ALL_GATHER_DOUBLE_RING_PUB_H */
