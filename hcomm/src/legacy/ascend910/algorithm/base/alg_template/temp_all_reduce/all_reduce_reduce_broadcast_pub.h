/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ALL_REDUCE_REDUCE_BROADCAST_PUB_H
#define ALL_REDUCE_REDUCE_BROADCAST_PUB_H

#include "alg_template_base_pub.h"
#include "alg_template_register.h"

namespace hccl {
class AllReduceReduceBcast : public AlgTemplateBase {
public:
    explicit AllReduceReduceBcast(const HcclDispatcher dispatcher);

    ~AllReduceReduceBcast() override;

    HcclResult RunAsync(const u32 rank, const u32 rankSize, const std::vector<LINK>& links) override;
    HcclResult Prepare(PrepareData& param) override;

protected:
private:
    HcclResult MainRecordSub();
    HcclResult SubWaitMain();
    HcclResult MainWaitSub();
    HcclResult SubRecordMain();
    HcclResult RunReduce(u32 rank, u32 rankSize, const std::vector<LINK>& links);
    HcclResult RunBroadcast(u32 rank, u32 rankSize, const std::vector<LINK>& links);
    HcclResult RunAllReduceBDReduceSend(u32 rank, u32 peer, const std::vector<LINK>& links);
    HcclResult RunAllReduceBDReduceReceive(u32 rank, u32 peer, const std::vector<LINK>& links);
    HcclResult RunAllReduceBDMemcpySend(u32 rank, u32 peer, const std::vector<LINK>& links);
    HcclResult RunAllReduceBDMemcpyReceive(u32 rank, u32 peer, const std::vector<LINK>& links);
    u64 reduceAttr_;
    u32 localRank_;
    u32 localRankSize_;
    u32 userRank_;
    std::vector<Stream> meshStreams_;                                   /* * 多steam* */
    const std::vector<std::shared_ptr<LocalNotify>>* meshSignalPtr_;    /* 每个ring创建一个signal */
    const std::vector<std::shared_ptr<LocalNotify>>* meshSignalAuxPtr_; /* 从stream wait，主steam record */
    const HcomCollOpInfo* opInfo_;
};
} // namespace hccl
#endif /* ALL_REDUCE_REDUCE_BROADCAST_PUB_H */
