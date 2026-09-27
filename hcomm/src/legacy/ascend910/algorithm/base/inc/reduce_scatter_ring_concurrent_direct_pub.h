/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef REDUCE_SCATTER_RING_CONCURRENT_DIRECT_PUB_H
#define REDUCE_SCATTER_RING_CONCURRENT_DIRECT_PUB_H

#include "alg_template_base_pub.h"
#include "reducer_pub.h"
#include "sender_pub.h"

namespace hccl {
class ReduceScatterRingConcurrentDirect : public AlgTemplateBase {
public:
    explicit ReduceScatterRingConcurrentDirect(const HcclDispatcher dispatcher);
    ~ReduceScatterRingConcurrentDirect() override;

    HcclResult Prepare(
        const u64 reduceAttrBitMap, const HcomCollOpInfo* opInfo, const u32 userRank, std::vector<Stream>& subStreams,
        const std::vector<std::shared_ptr<LocalNotify>>& mainSignals,
        const std::vector<std::shared_ptr<LocalNotify>>& subSignals, const std::vector<u32>& ringsOrder,
        const std::vector<Slice>& userMemInputSlices, bool isSdma = true) override;
    HcclResult RunAsync(const u32 rank, const u32 rankSize, const std::vector<LINK>& links) override;

protected:
private:
    HcclResult CheckParameters(const u32 rank, const u32 rankSize, const std::vector<LINK>& links);
    HcclResult OneRankMemcpy();
    HcclResult InitSenderReducer();
    HcclResult GetInitializedNeighborLinks(const u32 rank, const u32 rankSize, const std::vector<LINK>& links);
    HcclResult SetSlices(const u32 rank, const u32 rankSize);
    HcclResult RunInitStep(const u32 rank, const u32 rankSize);
    HcclResult PreSync();
    HcclResult ReducerSpInlineSlice(
        const HcclDispatcher dispatcher, const LINK& link, void* remoteMem, ReducerMemoryInfo reduceMem,
        Stream& stream);
    HcclResult ReducerSdmaRemoteReadSlice(const LINK& link, const ReducerMemoryInfo& reduceMem, Stream& stream) const;
    HcclResult ReducerLocalReduceSuffix(
        const HcclDispatcher dispatcher, const LINK& link, const std::vector<ReducerMemoryInfo>& reducerMems,
        Stream& stream);
    HcclResult ReducerRunSpInlineReduce(
        const HcclDispatcher dispatcher, const LINK& link, const std::vector<ReducerMemoryInfo>& reducerMems,
        Stream& stream);
    HcclResult ReducerRunNoSpInlineReduce(
        const HcclDispatcher dispatcher, const LINK& link, const std::vector<ReducerMemoryInfo>& reducerMems,
        Stream& stream);
    HcclResult ReducerRun(
        const HcclDispatcher dispatcher, const LINK& link, const std::vector<ReducerMemoryInfo>& reducerMems,
        Stream& stream);
    HcclResult RunMainStreamTx(
        const u32 step, const std::vector<Slice>& txSliceVector, const std::vector<Slice>& rxSliceVector,
        const u32 rank, const u32 rankSize, std::vector<ReducerMemoryInfo>& rxReduceMems);
    HcclResult RunMainStream(
        const u32 step, std::vector<Slice> txSliceVector, std::vector<Slice> rxSliceVector, const u32 rank,
        const u32 rankSize);
    HcclResult RunSubStreamPrePost();
    HcclResult RunSubStream(
        const u32 step, std::vector<Slice> subSliceVector, std::vector<Slice> cclSliceVector, const u32 rank,
        const u32 rankSize);
    HcclResult RunSubStreamSlice(
        const u32 step, const u32 sliceIdx, const std::vector<Slice>& subSliceVector,
        const std::vector<Slice>& cclSliceVector, const u32 rankSize);
    HcclResult RunSdmaStepConcurrent(
        const u32 step, const std::vector<ReducerMemoryInfo>& rxReduceMems, const std::vector<Slice>& subSliceVector,
        const std::vector<Slice>& cclSliceVector, const u32 rank, const u32 rankSize);
    HcclResult RunReduceScatter(const u32 rank, const u32 rankSize);
    HcclResult MainRecordSub();
    HcclResult SubWaitMain();
    HcclResult SubRecordMain();
    HcclResult MainWaitSub();

    LINK leftLink_;
    LINK rightLink_;

    std::unique_ptr<Sender> senderInfo_;
    std::unique_ptr<Reducer> reducerInfo_;

    u64 reduceAttr_ = 0; /* 0x1:表示data_type + reduce_type支持inlinereduce  */
    const HcomCollOpInfo* opInfo_{nullptr};
    u32 userRank_ = 0;
    std::vector<Stream> subStreams_;
    std::vector<std::shared_ptr<LocalNotify>> mainSignals_;
    std::vector<std::shared_ptr<LocalNotify>> subSignals_;
    std::vector<u32> ringsOrder_;
    std::vector<Slice> userSlices_;
    u64 lastStepOffset_ = 0;
    bool isSdma_ = false;
    DeviceMem finalSrc_;
    DeviceMem finalDst_;
};
} // namespace hccl

#endif /* REDUCE_SCATTER_RING_CONCURRENT_DIRECT_PUB_H */
