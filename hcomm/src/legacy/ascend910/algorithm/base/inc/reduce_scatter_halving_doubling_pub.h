/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef REDUCE_SCATTER_HALVING_DOUBLING_PUB_H
#define REDUCE_SCATTER_HALVING_DOUBLING_PUB_H

#include "alg_template_base_pub.h"
#include "reducer_pub.h"
#include "sender_pub.h"

namespace hccl {
class ReduceScatterHalvingDoubling : public AlgTemplateBase {
public:
    explicit ReduceScatterHalvingDoubling(const HcclDispatcher dispatcher);

    ~ReduceScatterHalvingDoubling() override;

    HcclResult Prepare(
        DeviceMem& inputMem, DeviceMem& outputMem, DeviceMem& scratchMem, const u64 count, const HcclDataType dataType,
        const Stream& stream, const HcclReduceOp reductionOp, const u32 root, const std::vector<Slice>& slices,
        const u64 baseOffset, const u32 blockSize, const u64 reduceAttrBitMap, const UserMemType hdInputMemType,
        const UserMemType hdOutputMemType) override;

    HcclResult RunAsync(const u32 rank, const u32 rankSize, const std::vector<LINK>& links) override;
    HcclResult
    GetNslbAdjInfo(const u32 rank, const u32 rankSize, const std::vector<LINK>& links, AdjInfo& nslbAdjInfo) override;

protected:
private:
    HcclResult RunReduceScatter(
        const u32 rank, const u32 stepNum, const HcclDispatcher dispatcher, const std::vector<LINK>& links);
    HcclResult RunSourceReducer(const LINK& link, const Slice& txSlice);
    HcclResult
    RunDestRducer(const LINK& link, const HcclDispatcher dispatcher, const Slice& rxSlice, const DstMemType reduceDst);

    u32 GetBlockStep(u32 blocksize) const;
    HcclResult CalculateSlices(const u64 size, const u32 sliceNum, std::vector<Slice>& slicesOut) const;
    HcclResult CalcStepSlices(
        const std::vector<Slice>& inputSlices, const u32 stepNum, const u32 rank, const SliceType type,
        std::vector<Slice>& slicesOut);
    u32 blockSize_ = 0;
    std::vector<Slice> txSlices_; // 下标为step, 标识每个step的tx_size
    std::vector<Slice> rxSlices_; // 下标为step, 标识每个step的tx_size
    u64 reduceAttr_ = 0;          /* 0x1:表示data_type + reduce_type支持inlinereduce  */

    UserMemType hdInputMemType_;  // 算法使用的input mem对应用户mem的类型
    UserMemType hdOutputMemType_; // 算法使用的output mem对应用户mem的类型

    std::unique_ptr<Sender> senderInfo_;
    std::unique_ptr<Reducer> reducerInfo_;
};
} // namespace hccl

#endif /* REDUCE_SCATTER_HALVING_DOUBLING_PUB_H */
