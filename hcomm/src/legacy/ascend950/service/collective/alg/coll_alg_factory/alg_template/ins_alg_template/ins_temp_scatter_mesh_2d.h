/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_INS_TEMP_SCATTER_MESH_2D
#define HCCLV2_INS_TEMP_SCATTER_MESH_2D

#include "string_util.h"

#include "ins_alg_template_base.h"
#include "executor_utils.h"

namespace Hccl {

class InsTempScatterMesh2D : public InsAlgTemplateBase {
public:
    explicit InsTempScatterMesh2D(
        const RankId virtualRank, const u32 tempRankSize, const std::vector<std::vector<RankId>>& tempVTopo,
        const std::map<RankId, u32>& tempVirtRankMap);
    ~InsTempScatterMesh2D() override;

    std::string Describe() const override
    {
        return StringFormat("Instruction based Template of scatter mesh 2D with tempRankSize [%u].", tempRankSize_);
    }

    HcclResult GenExtIns(
        TempFuncs& tempFuncs, TemplateDataParams& tempAlgParams, ResLinks& tempResLinks,
        std::vector<InsQuePtr>& tempInsQues);
    HcclResult CalcRes(AlgTempResReq& tempResReq) override;
    u32 CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType) const;

private:
    HcclResult PreDataCopy(const TemplateDataParams& tempAlgParams, std::vector<InsQuePtr>& tempInsQues) const;
    HcclResult PostDataCopy(const TemplateDataParams& tempAlgParams, std::vector<InsQuePtr>& tempInsQues) const;

    HcclResult RootMeshSend(
        TemplateDataParams& tempAlgParams, ResLinks& tempResLinks, const std::vector<RankId> vTopo,
        const u32 xyRankSize, u32 rankDistX, u32 rankDistY, u64 dataOffSet, u64 tranDataSize,
        std::vector<InsQuePtr>& tempInsQues) const;
    HcclResult RankRecvFromRoot(
        TemplateDataParams& tempAlgParams, ResLinks& tempResLinks, const u32 xyRankSize, u32 rankDist, u64 dataOffSet,
        u64 tranDataSize, std::vector<InsQuePtr>& tempInsQues) const;
    HcclResult RunDataCombine(
        TemplateDataParams& tempAlgParams, ResLinks& tempResLinks, u64 tranDataSizeX, u64 tranDataSizeY,
        std::vector<InsQuePtr>& tempInsQues);

    HcclResult DataCombineSend(
        const TemplateDataParams& tempAlgParams, ResLinks& tempResLinks, const std::vector<RankId> vTopo,
        u64 dataOffSet, u64 tranDataSize, std::vector<InsQuePtr>& tempInsQues) const;
    HcclResult DataCombineRecv(
        const TemplateDataParams& tempAlgParams, LinkData link, u64 dataOffSet, u64 tranDataSize,
        InsQuePtr queue) const;

    HcclResult GetRankId(u32 xRank, u32 yRank, u32& rank) const;

    HcclResult
    SendDirect(TemplateDataParams& tempAlgParams, InsQuePtr queue, const LinkData link, u32 remoteRank) const;
    HcclResult SendTransit(
        const TemplateDataParams& tempAlgParams, InsQuePtr queue, const LinkData link, u32 remoteRank, u32 xyRankSize,
        u32 rankDistX, u32 rankDistY, u64 xyOffSet, u64 tranDataSize) const;
    HcclResult
    RecvDirect(TemplateDataParams& tempAlgParams, InsQuePtr queue, const LinkData link, u32 remoteRank) const;
    HcclResult RecvTransit(
        const TemplateDataParams& tempAlgParams, InsQuePtr queue, const LinkData link, u32 remoteRank, u32 xyRankSize,
        u32 rankDist, u64 xyOffSet, u64 tranDataSize) const;

    u32 queNum_{0};
    u32 xQueNum_{0};
    u32 yQueNum_{0};

    u32 xRankSize_{0};
    u32 yRankSize_{0};

    u32 myRankX_{0};
    u32 myRankY_{0};
    u32 rootX_{0};
    u32 rootY_{0};

    std::vector<InsQuePtr> xInsQues_;
    std::vector<InsQuePtr> yInsQues_;

    bool enableInterRankCounterNotify_{false};
};

} // namespace Hccl

#endif // HCCLV2_INS_TEMP_SCATTER_MESH_2D
