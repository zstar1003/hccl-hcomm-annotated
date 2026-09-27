/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ALL_GATHER_HALVING_DOUBLING_PUB_H
#define ALL_GATHER_HALVING_DOUBLING_PUB_H

#include "alg_template_base_pub.h"

namespace hccl {
class AllGatherHalvingDoubling : public AlgTemplateBase {
public:
    explicit AllGatherHalvingDoubling(const HcclDispatcher dispatcher);
    ~AllGatherHalvingDoubling() override;

    // should be called soon after template AllGatherHalvingDoubling instance created
    HcclResult Prepare(u32 blockSize, UserMemType hdInputMemType, UserMemType hdOutputMemType) override;

    HcclResult RunAsync(const u32 rank, const u32 rankSize, const std::vector<LINK>& links) override;
    HcclResult
    GetNslbAdjInfo(const u32 rank, const u32 rankSize, const std::vector<LINK>& links, AdjInfo& nslbAdjInfo) override;

protected:
private:
    HcclResult RunAllGather(u32 rank, u32 stepNum, const std::vector<LINK>& links);
    HcclResult Tx(const LINK& link, const Slice& txSlice);
    HcclResult Rx(const LINK& link, const Slice& rxSlice);

    u32 Log2(u32 antilogarithm) const;
    HcclResult CalculateSlices(u64 size, u32 sliceNum, std::vector<Slice>& sliceOut) const;
    HcclResult CalculateSlices(
        const std::vector<Slice>& inputSlices, u32 stepNum, u32 rank, SliceType type, std::vector<Slice>& sliceOut);

    u32 blockSize_;               // binary block的size, 等于rank_size
    std::vector<Slice> txSlices_; // 下标为step, 标识每个step的tx_size
    std::vector<Slice> rxSlices_; // 下标为step, 标识每个step的rx_size
    u32 interRank_;
    u32 interRankSize_;
    UserMemType hdInputMemType_{UserMemType::OUTPUT_MEM}; // 算法使用的input mem对应用户mem的类型
    UserMemType hdOutputMemType_{UserMemType::INPUT_MEM}; // 算法使用的output mem对应用户mem的类型
};
} // namespace hccl

#endif /* ALL_GATHER_HALVING_DOUBLING_PUB_H */
