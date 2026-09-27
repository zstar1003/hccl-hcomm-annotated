/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SCATTER_RING_PUB_H
#define SCATTER_RING_PUB_H

#include "alg_template_base_pub.h"

namespace hccl {
class ScatterRing : public AlgTemplateBase {
public:
    explicit ScatterRing(const HcclDispatcher dispatcher);

    ~ScatterRing() override;

    HcclResult
    RunAsync(const u32 rank, const u32 rankSize, const std::vector<std::shared_ptr<Transport>>& links) override;
    HcclResult
    GetNslbAdjInfo(const u32 rank, const u32 rankSize, const std::vector<LINK>& links, AdjInfo& nslbAdjInfo) override;

protected:
private:
    void PrepareSlicesData(const u32 unitSize, const u64 totalCount, const u32 rankSize) const;
    HcclResult RunScatterOnRootRank();
    HcclResult RunScatterOnEndRank();
    HcclResult RunScatterOnMidRank();
    std::shared_ptr<Transport> linkLeft_;
    std::shared_ptr<Transport> linkRight_;
    u32 interRank_;     // comm内的rank排序
    u32 interRankSize_; // 本comm内ranksize总数

    // scatter ring chunk实现相关函数
    HcclResult RunScatterChunk(const u32 rank, const u32 rankSize, const std::vector<Slice>& outputSlices);
    HcclResult HeadScatterChunk(u32 rank, u32 rankSize, const std::vector<Slice>& outputSlices);
    HcclResult MidScatterChunk(u32 rank, u32 rankSize, u32 sliceIdx, const std::vector<Slice>& outputSlices);
    HcclResult TailScatterChunk(u32 rank, u32 rankSize, u32 sliceIdx, const std::vector<Slice>& outputSlices);
    HcclResult ScatterSlicesPrep(u32 rankSize, u32 nicSize);
};
} // namespace hccl

#endif /* SCATTER_RING_PUB_H */
