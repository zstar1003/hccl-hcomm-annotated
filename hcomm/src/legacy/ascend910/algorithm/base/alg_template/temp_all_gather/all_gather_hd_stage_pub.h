/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ALL_GATHER_HD_STAGE_PUB_H
#define ALL_GATHER_HD_STAGE_PUB_H

#include "alg_template_base_pub.h"
#include "mem_host_pub.h"
#include "mem_device_pub.h"
#include "stream_pub.h"
#include "alg_template_register.h"

namespace hccl {
class AllGatherHDStage : public AlgTemplateBase {
public:
    explicit AllGatherHDStage(const HcclDispatcher dispatcher);
    ~AllGatherHDStage() override;

    HcclResult RunAsync(const u32 rank, const u32 rankSize, const std::vector<LINK>& links) override;
    HcclResult Prepare(PrepareData& param) override;

protected:
private:
    HcclResult MainRecordSub(u32 streamNum);
    HcclResult SubWaitMain(u32 streamNum);
    HcclResult MainWaitSub(u32 streamNum);
    HcclResult SubRecordMain(u32 streamNum);
    HcclResult ReverseId(u32 oriIdx, u32& revIdx);
    HcclResult
    PrepareSliceData(u32 subRank, u32 subRankSize, u32 size, u32 batchSize, std::vector<Slice>& slices) const;
    HcclResult RunPreCopy(u32 rank, u32 rankSize, const std::vector<LINK>& links);
    HcclResult RunAllGatherStage(u32 rank, u32 rankSize, const std::vector<LINK>& links);
    HcclResult RunAllGatherNoPower(u32 rank, u32 rankSize, const std::vector<LINK>& links);
    HcclResult RunBetweenStep(u32 rank, u32 neighCur, u32 neighNext, const std::vector<LINK>& links);
    HcclResult RunAllGatherPower(u32 rank, u32 rankSize, const std::vector<LINK>& links);
    HcclResult RunAllGatherLast(u32 rank, u32 rankSize, const std::vector<LINK>& links);
    HcclResult RunAllGatherLastOne(u32 rank, u32 rankSize, const std::vector<LINK>& links);
    HcclResult RunAllGatherLastTwo(u32 rank, u32 rankSize, const std::vector<LINK>& links);
    inline u32 BackwardRank(u32 rank, u32 rankSize, u32 step) const
    {
        if (rankSize == 0) {
            return 0;
        }
        return (rank + rankSize - step) % rankSize;
    }
    const u32 base = 2;
    u32 userRank_;
    std::vector<Stream> meshStreams_;                                   /* * 多steam* */
    const std::vector<std::shared_ptr<LocalNotify>>* meshSignalPtr_;    /* 每个ring创建一个signal */
    const std::vector<std::shared_ptr<LocalNotify>>* meshSignalAuxPtr_; /* 从stream wait，主steam record */
    const HcomCollOpInfo* opInfo_;
    std::vector<Slice> sliceNoPower_;
    std::vector<Slice> slicePower_;
    const std::vector<u32> resMap = {0, 2, 1, 3};
    u32 nSteps_ = 0;
    u32 powerSteps_ = 0;
    u32 finalSteps_ = 0;
    u32 noPower_ = 0;
    u32 totalSize_ = 0;
};
} // namespace hccl
#endif /* ALL_GATHER_HD_STAGE_PUB_H */
