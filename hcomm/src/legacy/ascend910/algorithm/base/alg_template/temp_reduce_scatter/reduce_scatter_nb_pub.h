/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef REDUCE_SCATTER_NB_PUB_H
#define REDUCE_SCATTER_NB_PUB_H

#include <cmath>

#include "nonuniform_bruck_base_pub.h"

#include "mem_host_pub.h"
#include "mem_device_pub.h"
#include "stream_pub.h"

#include "reducer_pub.h"
#include "sender_pub.h"

namespace hccl {
class ReduceScatterNB : public NBBase {
public:
    explicit ReduceScatterNB(const HcclDispatcher dispatcher);
    ~ReduceScatterNB() override;

    HcclResult Prepare(u64 reduceAttrBitMap, HcomCollOpInfo* opInfo = nullptr) override;
    HcclResult RunAsync(const u32 rank, const u32 rankSize, const std::vector<LINK>& links) override;
    HcclResult
    GetNslbAdjInfo(const u32 rank, const u32 rankSize, const std::vector<LINK>& links, AdjInfo& nslbAdjInfo) override;

private:
    u64 reduceAttr_ = 0; /* 0x1:表示data_type + reduce_type支持inlinereduce  */

    std::unique_ptr<Sender> senderInfo_;
    std::unique_ptr<Reducer> reducerInfo_;

    HcclResult SimpleCheck(const u32 rank, const u32 rankSize, const std::vector<LINK>& links);
    HcclResult CheckSlices(const std::vector<Slice>& checkSlices, const u32 rankSize);

    HcclResult RunSrcReducerNB(
        const u32 step, const u32 nSlices, const u32 sliceSize, u32 txSliceIdx, const u32 deltaSliceIndex,
        const LINK linkRight, const u32 rank, const u32 rankSize, const std::vector<Slice>& inputSlices,
        const std::vector<Slice>& outputSlices);

    HcclResult RunDestReducerNB(
        const u32 step, const u32 nSteps, const u32 sliceSize, const u32 nSlices, u32 rxSliceIdx,
        const u32 deltaSliceIndex, const LINK linkLeft, const u32 rank, const u32 rankSize,
        const std::vector<Slice>& inputSlices, const std::vector<Slice>& outputSlices);

    HcclResult RunReduceScatterNB(
        const u32 rank, const u32 rankSize, const std::vector<LINK>& links, const std::vector<Slice>& inputSlices,
        const std::vector<Slice>& outputSlices);
    HcclResult
    RunDestReducer(const LINK& link, const std::vector<Slice>& rxSlices, const std::vector<Slice>& rxSlicestemp);
    HcclResult
    RunSourceReducer(const LINK& link, const std::vector<Slice>& txSlices, const std::vector<Slice>& txSlicestemp);
};
} // namespace hccl

#endif /* REDUCE_SCATTER_NB_PUB_H */
