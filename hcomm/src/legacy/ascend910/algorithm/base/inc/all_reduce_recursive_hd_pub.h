/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ALL_REDUCE_RECURSIVE_HD_PUB_H
#define ALL_REDUCE_RECURSIVE_HD_PUB_H

#include "recursive_halvingdoubling_base_pub.h"
#include "reducer_pub.h"
#include "sender_pub.h"

namespace hccl {
class AllReduceRecursiveHalvingDoubling : public RecursiveHalvingDoublingBase {
public:
    using AlgTemplateBase::Prepare;
    explicit AllReduceRecursiveHalvingDoubling(const HcclDispatcher dispatcher);
    ~AllReduceRecursiveHalvingDoubling() override;

    // 新增的两段式构造函数，获取实例后要无脑调用实现构造函数功能，后续还要调用其它的基类Prepare函数实现其它成员变量初始化
    HcclResult Prepare(u64 reduceAttrBitMap, HcomCollOpInfo* opInfo = nullptr) override;

    HcclResult RunAsync(const u32 rank, const u32 rankSize, const std::vector<LINK>& links) override;
    HcclResult
    RunAsyncStaged(const u32 rank, const u32 rankSize, const std::vector<LINK>& links, RunStage stage) override;
    HcclResult GetCommonNslbAdjInfo(
        const u32 rank, const u32 rankSize, const std::vector<LINK>& links, AdjInfo& nslbAdjInfo) const;
    HcclResult
    GetOddNslbAdjInfo(const u32 rank, const u32 rankSize, const std::vector<LINK>& links, AdjInfo& nslbAdjInfo) const;
    HcclResult
    GetNslbAdjInfo(const u32 rank, const u32 rankSize, const std::vector<LINK>& links, AdjInfo& nslbAdjInfo) override;

protected:
    HcclResult PrepareRunAsync(const u32 rank, const u32 rankSize, const std::vector<LINK>& links) override;

private:
    HcclResult ReduceInPartOne(u32 rank, const std::vector<LINK>& links);

    HcclResult ReduceScatterInBlock(u32 rank, u32 rankSize, const std::vector<LINK>& links);

    HcclResult AllGatherInBlock(u32 rank, u32 rankSize, const std::vector<LINK>& links);

    HcclResult GatherInPartOne(u32 rank, const std::vector<LINK>& links);

    u64 reduceAttr;

    std::unique_ptr<Sender> senderInfo_;
    std::unique_ptr<Reducer> reducerInfo_;
};
} // namespace hccl

#endif /* __ALL_REDUCE_RECURSIVE_HALVINGDOUBLING_PUB_H__ */
