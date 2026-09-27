/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef REDUCE_RECURSIVE_HD_PUB_H
#define REDUCE_RECURSIVE_HD_PUB_H

#include "recursive_halvingdoubling_base_pub.h"
#include "reducer_pub.h"
#include "sender_pub.h"

namespace hccl {
class ReduceRecursiveHalvingDoubling : public RecursiveHalvingDoublingBase {
public:
    using AlgTemplateBase::Prepare;
    explicit ReduceRecursiveHalvingDoubling(const HcclDispatcher dispatcher);
    ~ReduceRecursiveHalvingDoubling() override;

    /* 新增的两段式构造函数，获取实例后要无脑调用实现构造函数功能,后续还要调用其它的基类Prepare函数实现其它成员变量初始化
     */
    HcclResult Prepare(u64 reduceAttrBitMap, HcomCollOpInfo* opInfo = nullptr) override;

    HcclResult
    RunAsync(const u32 rank, const u32 rankSize, const std::vector<std::shared_ptr<Transport>>& links) override;
    HcclResult
    GetNslbAdjInfo(const u32 rank, const u32 rankSize, const std::vector<LINK>& links, AdjInfo& nslbAdjInfo) override;

protected:
private:
    HcclResult ReduceInPartOne(u32 rank, const std::vector<LINK>& links);

    HcclResult ReduceScatterInBlock(u32 rank, u32 rankSize, const std::vector<LINK>& links);

    HcclResult GatherInBlock(u32 rank, u32 rankSize, const std::vector<LINK>& links);

    HcclResult CalculateStepSlices(
        const std::vector<Slice>& inputSlices, u32 stepNum, u32 rank, SliceType type, std::vector<Slice>& sliceOut);
    HcclResult BuildRootSubLinks(const std::vector<LINK>& links, std::vector<LINK>& subLinks, u32 rankSize) const;
    std::vector<Slice> txSlices_;
    std::vector<Slice> rxSlices_;
    u64 reduceAttr;

    std::unique_ptr<Sender> senderInfo_;
    std::unique_ptr<Reducer> reducerInfo_;
};
} // namespace hccl

#endif /* __REDUCE_RECURSIVE_HALVINGDOUBLING_PUB_H__ */
