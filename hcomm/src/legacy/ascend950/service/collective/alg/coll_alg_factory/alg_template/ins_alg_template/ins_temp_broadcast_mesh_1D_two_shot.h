/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_INS_TEMP_BROADCAST_MESH_TWO_SHOT_H
#define HCCLV2_INS_TEMP_BROADCAST_MESH_TWO_SHOT_H

#include "string_util.h"

#include "ins_alg_template_base.h"
#include "executor_utils.h"

namespace Hccl {

class InsTempBroadcastMesh1DTwoShot : public InsAlgTemplateBase {
public:
    explicit InsTempBroadcastMesh1DTwoShot(
        const RankId virtualRank, const u32 tempRankSize, const std::vector<std::vector<RankId>>& tempVTopo,
        const std::map<RankId, u32>& tempVirtRankMap);
    ~InsTempBroadcastMesh1DTwoShot() override;

    std::string Describe() const override
    {
        return StringFormat("Template of broadcast Mesh two shot with tempRankSize [%u].", tempRankSize_);
    }

    HcclResult GenExtIns(
        const TempFuncs& tempFuncs, const TemplateDataParams& templateDataParams, const ResLinks& tempLinks,
        std::vector<InsQuePtr>& tempInsQues);

    HcclResult CalcRes(AlgTempResReq& tempResReq) override;
    u32 CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType);

private:
    HcclResult PostCopy(const TemplateDataParams& tempAlgParams, std::vector<InsQuePtr>& tempInsQues) const;
    HcclResult CalcCommRankSetforScatter(const u32 groupRankSize, std::vector<u32>& commRanks) const;
    HcclResult CalcCommRankSetforAllGather(const u32 groupRankSize, std::vector<u32>& commRanks) const;
    HcclResult RunScatter(
        const std::vector<u32>& commRanks, const TemplateDataParams& tempAlgParams, const ResLinks& tempLinks,
        std::vector<InsQuePtr>& queues, const RankSliceInfo& sliceInfoVec) const;
    HcclResult RunAllGather(
        const std::vector<u32>& commRanks, const TemplateDataParams& tempAlgParams, const ResLinks& tempLinks,
        std::vector<InsQuePtr>& queues, const RankSliceInfo& sliceInfoVec) const;
    HcclResult RootSendData(
        const u64 memOffset, const s32 remoteRank, const TemplateDataParams& tempAlgParams, const InsQuePtr& queue,
        const LinkData& link, const RankSliceInfo& sliceInfoVec) const;
    HcclResult RankRecvData(
        const u64 memOffset, const TemplateDataParams& tempAlgParams, const InsQuePtr& queue, const LinkData& link,
        const RankSliceInfo& sliceInfoVec) const;

    HcclResult CalcDataSliceInfo(const u64 dataSize, RankSliceInfo& sliceInfoVec);

    BufferType srcBufferType_ = BufferType::INPUT;
    BufferType dstBufferType_ = BufferType::INPUT;
};

} // namespace Hccl

#endif // !HCCLV2_INS_TEMP_BROADCAST_MESH_TWO_SHOT_H
