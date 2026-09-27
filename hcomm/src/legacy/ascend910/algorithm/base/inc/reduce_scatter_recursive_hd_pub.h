/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef REDUCE_SCATTER_RECURSIVE_HD_PUB_H
#define REDUCE_SCATTER_RECURSIVE_HD_PUB_H

#include "recursive_halvingdoubling_base_pub.h"
#include "reducer_pub.h"
#include "sender_pub.h"

namespace hccl {
class ReduceScatterRecursiveHalvingDoubling : public RecursiveHalvingDoublingBase {
public:
    explicit ReduceScatterRecursiveHalvingDoubling(const HcclDispatcher dispatcher);
    ~ReduceScatterRecursiveHalvingDoubling() override;

    HcclResult Prepare(u64 reduceAttrBitMap, HcomCollOpInfo* opInfo = nullptr) override;
    HcclResult
    RunAsync(const u32 rank, const u32 rankSize, const std::vector<std::shared_ptr<Transport>>& links) override;
    HcclResult
    GetNslbAdjInfo(const u32 rank, const u32 rankSize, const std::vector<LINK>& links, AdjInfo& nslbAdjInfo) override;

protected:
private:
    HcclResult CalculateSlices(u64 dataBytes, const u32 rankSize) const;

    HcclResult ReduceInPartOne(u32 rank, const std::vector<LINK>& links);

    HcclResult ReduceScatterInBlock(u32 rank, u32 rankSize, const std::vector<LINK>& links);

    HcclResult ScatterInPartOne(u32 rank, u32 rankSize, const std::vector<LINK>& links);

    u64 reduceAttr = 0;

    std::unique_ptr<Sender> senderInfo_;
    std::unique_ptr<Reducer> reducerInfo_;
};
} // namespace hccl

#endif /* __REDUCE_SCATTER_HALVINGDOUBLING_PUB_H__ */
