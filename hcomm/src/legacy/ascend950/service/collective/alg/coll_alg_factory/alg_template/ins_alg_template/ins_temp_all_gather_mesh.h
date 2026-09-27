/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_INS_TEMP_ALL_GATHER_MESH
#define HCCLV2_INS_TEMP_ALL_GATHER_MESH

#include "string_util.h"
#include "executor_utils.h"
#include "ins_alg_template_base.h"

namespace Hccl {

class InsTempAllGatherMesh1D : public InsAlgTemplateBase {
public:
    explicit InsTempAllGatherMesh1D(
        const RankId virtualRank, const u32 tempRankSize, const std::vector<std::vector<RankId>>& tempVTopo,
        const std::map<RankId, u32>& tempVirtRankMap);
    ~InsTempAllGatherMesh1D() override;

    std::string Describe() const override
    {
        return StringFormat("Instruction based Template of all gather mesh with tempRankSize [%u].", tempRankSize_);
    }

    HcclResult GenExtIns(
        const TempFuncs& tempFuncs, const TemplateDataParams& tempAlgParams, const ResLinks& tempLinks,
        std::vector<InsQuePtr>& tempInsQues);
    HcclResult CalcSliceInfo(const AllignInfo& allignInfo, const u64 dataSize, RankSliceInfo& sliceInfoVec) override;
    HcclResult CalcRes(AlgTempResReq& tempResReq) override;
    u32 CalcScratchMultiple(const BufferType& inBufferTpye, const BufferType& outBufferTpye) const
    {
        (void)inBufferTpye;
        (void)outBufferTpye;
        HCCL_INFO("[InsTempAllGatherMesh1D][CalcScratchMultiple] templateScratchMultiplier[%llu]", tempRankSize_);
        return tempRankSize_;
    }

private:
    HcclResult LocalCopyToScratch(InsQuePtr tempInsQue);
    HcclResult LocalCopyToUsrOut(InsQuePtr tempInsQue);
    HcclResult RunMesh(const u32 myAlgRank, const std::vector<RankId>& vTopo, std::vector<InsQuePtr>& tempInsQues);
    HcclResult SingleRunMesh(const u32 myAlgRank, u32 connectedRank, std::vector<InsQuePtr>& tempInsQues, u32& queIdx);
    void PrepareLinkSlices(
        const u32 myAlgRank, const u32 connectedAlgRank, const std::vector<float>& dataSplitRate, const u32 linkIndex,
        std::vector<DataSlice>& txSrcSlices, std::vector<DataSlice>& txDstSlices, std::vector<DataSlice>& rxSrcSlices,
        std::vector<DataSlice>& rxDstSlices);
    u32 majorQueNum_ = 0;
    u32 queNumPerNeighbor_ = 1;
    bool enableInterRankCounterNotify_ = false;
    TemplateDataParams tempAlgParams_;
    ResLinks tempLinks_;
};

} // namespace Hccl

#endif // !HCCLV2_INS_TEMP_ALL_GATHER_MESH
