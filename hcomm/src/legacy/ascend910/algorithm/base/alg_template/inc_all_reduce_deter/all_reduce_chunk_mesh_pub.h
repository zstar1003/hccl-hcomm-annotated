/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ALL_REDUCE_CHUNK_MESH_PUB_H
#define ALL_REDUCE_CHUNK_MESH_PUB_H

#include "alg_template_base_pub.h"

namespace hccl {
class AllReduceChunkMesh : public AlgTemplateBase {
public:
    using AlgTemplateBase::Prepare;
    explicit AllReduceChunkMesh(const HcclDispatcher dispatcher);
    ~AllReduceChunkMesh() override;

    /* 新增的两段式构造函数，获取实例后要无脑调用实现构造函数功能,后续还要调用其它的基类Prepare函数实现其它成员变量初始化
     */
    HcclResult Prepare(
        u64 reduceAttrBitMap, std::vector<Stream>& meshStreams, std::vector<std::shared_ptr<LocalNotify>>& meshSignal,
        std::vector<std::shared_ptr<LocalNotify>>& meshSignalAux, u32 interRank, u32 interRankSize, u32 userRank,
        HcomCollOpInfo* opInfo) override;

    HcclResult RunAsync(const u32 rank, const u32 rankSize, const std::vector<LINK>& links) override;

protected:
private:
    HcclResult MainRecordSub();
    HcclResult SubWaitMain();
    HcclResult MainWaitSub();
    HcclResult SubRecordMain();
    HcclResult PrepareSlice(u64 dataCount, u32 unitSize, u32 sliceNum, std::vector<Slice>& dataSlice) const;
    HcclResult PrepareAllreduceSliceData();
    HcclResult WaitPrevStep(u32 rank, u32 step, const std::vector<LINK>& links);
    HcclResult RecordNextStep(u32 rank, u32 step, const std::vector<LINK>& links);
    HcclResult RunReduceScatter(u32 rank, u32 rankSize, const std::vector<LINK>& links);
    HcclResult RunAllGather(u32 rank, u32 rankSize, const std::vector<LINK>& links);
    HcclResult PrepareAllreduceRing();
    inline u32 BackwardRank(u32 rank, u32 rankSize, u32 step) const
    {
        if (rankSize == 0) {
            return 0;
        }
        return (rank + rankSize - step) % rankSize;
    }
    u64 reduceAttr_;
    u32 localRank_;
    u32 localRankSize_;
    u32 userRank_;
    std::map<u32, std::vector<u32>> ringMap;
    std::map<u32, std::vector<Slice>> sliceMap;
    std::vector<Stream> meshStreams_;                                   /* * 多stream* */
    std::vector<std::shared_ptr<LocalNotify>>* meshSignal_{nullptr};    /* 每个ring创建一个signal */
    std::vector<std::shared_ptr<LocalNotify>>* meshSignalAux_{nullptr}; /* 从stream wait，主stream record */
    HcomCollOpInfo* opInfo_{nullptr};
};
} // namespace hccl
#endif /* ALL_REDUCE_CHUNK_MESH_PUB_H */
