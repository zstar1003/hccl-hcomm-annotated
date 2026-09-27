/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef REDUCE_SCATTER_MESH_PUB_H
#define REDUCE_SCATTER_MESH_PUB_H

#include "alg_template_base_pub.h"
#include "reducer_pub.h"
#include "sender_pub.h"

namespace hccl {
class ReduceScatterMesh : public AlgTemplateBase {
public:
    explicit ReduceScatterMesh(const HcclDispatcher dispatcher);

    ~ReduceScatterMesh() override;

    HcclResult Prepare(u64 reduceAttrBitMap, u32 streamIndex) override;
    HcclResult
    RunAsync(const u32 rank, const u32 rankSize, const std::vector<std::shared_ptr<Transport>>& links) override;

protected:
private:
    inline u32 ForwardRank(u32 rank, u32 rankSize, u32 step) const
    {
        if (rankSize == 0) {
            return 0;
        }
        return (rank + rankSize - step) % rankSize;
    }
    inline u32 BackwardRank(u32 rank, u32 rankSize, u32 step) const
    {
        if (rankSize == 0) {
            return 0;
        }
        return (rank + step) % rankSize;
    }
    HcclResult RunSourceReducer(const LINK& link, const Slice& txSlice, const Slice& dstSlice);

    HcclResult RunDestRducer(const LINK& link, const Slice& rxSlice, const Slice& dstSlice);

    HcclResult RunReduceScatter(
        const std::vector<LINK>& links, const std::vector<Slice>& inputSlices, const std::vector<Slice>& scratchSlices);
    std::unique_ptr<Sender> senderInfo_;
    std::unique_ptr<Reducer> reducerInfo_;
    u32 interRank_ = 0;     // 在所有rank环上的rankid?
    u32 interRankSize_ = 0; // 指的服务器的个数? 应当是所有服务器上rank总数和?

    u64 reduceAttr_ = 0; /* 0x1:表示data_type + reduce_type支持inlinereduce  */
    u32 streamIndex_ = 0;
};
} // namespace hccl

#endif /* REDUCE_SCATTER_MESH_PUB_H */
