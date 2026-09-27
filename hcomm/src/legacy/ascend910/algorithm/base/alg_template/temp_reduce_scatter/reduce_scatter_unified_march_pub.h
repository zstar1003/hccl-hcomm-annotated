/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef REDUCE_SCATTER_UNIFIED_MARCH_PUB_H
#define REDUCE_SCATTER_UNIFIED_MARCH_PUB_H

#include "alg_template_base_pub.h"

namespace hccl {
class ReduceScatterUnifiedMarch : public AlgTemplateBase {
public:
    explicit ReduceScatterUnifiedMarch(const HcclDispatcher dispatcher);
    ~ReduceScatterUnifiedMarch() override;

    HcclResult Prepare(
        Stream& mainStream, SubCommInfo& level0CommInfo, DeviceMem& userInput, DeviceMem& userOutput,
        DeviceMem& usrInMem, DeviceMem& scratchMem, u64 totalCount, std::vector<Stream>& subStreams,
        const std::vector<std::shared_ptr<LocalNotify>>& meshSignalMainToSub,
        const std::vector<std::shared_ptr<LocalNotify>>& meshSignalSubToMain, const HcclDataType dataType,
        const HcclReduceOp reductionOp, const std::vector<std::vector<Slice>>& multRingsUserMemSlice,
        u64 reduceAttrBitMap) override;
    HcclResult RunAsync() override;

protected:
private:
    std::string GetStreamIndexString();
    HcclResult NotifySubStreamStart(u32 streamSize);
    HcclResult WaitSubStreamFinish(u32 streamSize);
    HcclResult NotifyNeighborsStart(LINK& prevIntraLink, LINK& nextIntralLink, u32 neighbors);
    HcclResult NotifyNeighborsEnd(LINK& prevIntraLink, LINK& nextIntralLink, u32 neighbors);
    HcclResult DoSerialReduce(
        void* remDMAMemPtr, void* dstAddr, u64 memSize, u64 dataCount, Stream& tmpStream, LINK& tmpLink,
        u64 remoteOffsetByte);
    HcclResult RunSingleSliceRead(u32 ringPrevRank, u32 ringNextRank, u32 step, u32 totalStep);
    HcclResult RunHalfSliceRead(u32 ringPrevRank, u32 ringNextRank, u32 step, u32 totalStep);
    HcclResult RunLastStep(u32 ringPrevRank, u32 ringNextRank, u32 totalStep);

    u64 reduceAttr_ = 0;
    Stream mainStream_;
    std::vector<Stream> subStreams_;
    std::vector<std::shared_ptr<LocalNotify>> meshSignalMainToSub_;
    std::vector<std::shared_ptr<LocalNotify>> meshSignalSubToMain_;
    u32 intraRank_ = 0;
    u32 intraRankSize_ = 0;
    std::vector<LINK> links_;
    DeviceMem userInput_;
    DeviceMem userOutput_;
    DeviceMem usrInMem_;
    DeviceMem scratchMem_;
    HcclDataType dataType_;
    HcclReduceOp reductionOp_;
    u64 totalCount_ = 0;
    u64 blockDataByte_ = 0;
    std::vector<std::vector<Slice>> multRingsUserMemSlice_; // 记录server内的要发送的不连续数据块
    u32 notifyIdx_ = 0;                                     // 新增notify资源索引
};
} // namespace hccl
#endif /* ALLTOALL_V_MESH_READ_ONLY_PUB_H */
