/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef REDUCE_SCATTER_NHR_PUB_H
#define REDUCE_SCATTER_NHR_PUB_H

#include <cmath>

#include "nonuniform_hierarchical_ring_base_pub.h"

#include "mem_host_pub.h"
#include "mem_device_pub.h"
#include "stream_pub.h"

#include "reducer_pub.h"
#include "sender_pub.h"

namespace hccl {
class ReduceScatterNHR : public NHRBase {
public:
    explicit ReduceScatterNHR(const HcclDispatcher dispatcher);
    ~ReduceScatterNHR() override;

    HcclResult Prepare(u64 reduceAttrBitMap, bool needMerge) override;
    HcclResult RunAsync(const u32 rank, const u32 rankSize, const std::vector<LINK>& links) override;
    HcclResult
    GetNslbAdjInfo(const u32 rank, const u32 rankSize, const std::vector<LINK>& links, AdjInfo& nslbAdjInfo) override;

protected:
private:
    HcclResult SimpleCheck(const u32 rank, const u32 rankSize, const std::vector<LINK>& links);
    HcclResult CheckSlices(const std::vector<Slice>& checkSlices, const u32 rankSize);

    HcclResult RunReduceScatterNHR(
        const u32 rank, const u32 rankSize, const std::vector<LINK>& links, const std::vector<Slice>& inputSlices,
        const std::vector<Slice>& outputSlices);
    HcclResult RunSourceSender(
        const LINK& link, InterServerAlgoStep& stepInfo, const std::vector<Slice>& inputSlices,
        const std::vector<Slice>& outputSlices);
    HcclResult RunDestReducer(
        const LINK& link, InterServerAlgoStep& stepInfo, const std::vector<Slice>& inputSlices,
        const std::vector<Slice>& outputSlices);

    HcclResult GetStepInfo(u32 step, u32 nSteps, u32 rank, u32 rankSize, InterServerAlgoStep& stepInfo) override;

    HcclResult InlineReducer(const LINK& linkLeft, const std::vector<ReducerMemoryInfo>& rxReduceMems);

    HcclResult InlineReduceRx(const LINK& linkLeft, std::vector<Slice>& rxSlices, std::vector<Slice>& rxSlicestemp);

    HcclResult InlineReduceRxLastStep(
        const LINK& linkLeft, InterServerAlgoStep& stepInfo, const std::vector<Slice>& inputSlices,
        const std::vector<Slice>& outputSlices);

    HcclResult TbeReduceRx(const LINK& linkLeft, std::vector<Slice>& rxSlices, std::vector<Slice>& rxSlicestemp);

    HcclResult TbeReduceRxLastStep(
        const LINK& linkLeft, InterServerAlgoStep& stepInfo, const std::vector<Slice>& inputSlices,
        const std::vector<Slice>& outputSlices);

    HcclResult RunDestReducerLastStep(
        const LINK& linkLeft, InterServerAlgoStep& stepInfo, const std::vector<Slice>& inputSlices,
        const std::vector<Slice>& outputSlices);

    HcclResult GetRxSlices(
        std::vector<Slice>& rxSlices, std::vector<Slice>& rxSlicestemp, InterServerAlgoStep& stepInfo,
        const std::vector<Slice>& inputSlices, const std::vector<Slice>& outputSlices);

    HcclResult SdmaReducer(
        const u32 nSteps, const LINK& linkLeft, InterServerAlgoStep& stepInfo, const std::vector<Slice>& inputSlices,
        const std::vector<Slice>& outputSlices);

    void GetSliceMap(const u32 rankSize);

    static void Reorder(u32 start, u32 end, u32 len, std::vector<u32>& tree, std::vector<u32>& tmp);

    u64 reduceAttr_ = 0; /* 0x1:表示data_type + reduce_type支持inlinereduce  */

    std::unique_ptr<Sender> senderInfo_;
    std::unique_ptr<Reducer> reducerInfo_;
};
} // namespace hccl

#endif /* REDUCE_SCATTER_NHR_PUB_H */
