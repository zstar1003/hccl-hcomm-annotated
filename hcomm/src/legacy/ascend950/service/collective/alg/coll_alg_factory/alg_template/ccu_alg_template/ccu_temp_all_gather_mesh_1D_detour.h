/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_CCU_TEMP_ALL_GATHER_MESH_DETOUR_1D_H
#define HCCLV2_CCU_TEMP_ALL_GATHER_MESH_DETOUR_1D_H

#include "string_util.h"
#include "ccu_alg_template_base.h"
#include "ccu_instruction_all_gather_mesh1d_detour.h"

namespace Hccl {

class CcuTempAllGatherMeshDetour1D : public CcuAlgTemplateBase {
public:
    explicit CcuTempAllGatherMeshDetour1D(
        const RankId virtualRank, const u32 tempRankSize, const std::vector<std::vector<RankId>>& tempVTopo,
        const std::map<RankId, u32>& tempVirtRankMap);
    ~CcuTempAllGatherMeshDetour1D() override;

    std::string Describe() const override
    {
        return StringFormat("Template of all gather ccu mesh Detour 1D with tempRankSize [%u].", tempRankSize_);
    }

    HcclResult
    Run(const TempFuncs& tempFuncs, const RankSliceInfo& sliceInfoVec, const BuffInfo& buffInfo,
        const ResLinks& tempLinks, std::vector<InsQuePtr>& tempInsQues) override;
    HcclResult CalcResDetour(const RankGraph* rankGraph, AlgTempResReq& tempResReq) override;
    HcclResult CalcResDetour(ConnectedLinkMgr* linkMgr, AlgTempResReq& tempResReq) override;
    HcclResult CalcSliceInfo(const AllignInfo& allignInfo, const u64 dataSize, RankSliceInfo& sliceInfoVec) override;

private:
    void GetAddrInfo(
        const TempFuncs& tempFuncs, const RankSliceInfo& sliceInfoVec, uint64_t& inputAddr, uint64_t& outputAddr,
        uint64_t& offset);
    void CalcDetourOffset(uint64_t sliceSize, uint64_t& tailOffset, uint64_t& tailSize, uint64_t& loopIterNum);
    void ProcessLinks(std::vector<LinkData>& links, const ResLinks& tempLinks) const;

    uint64_t detourPathNum_{0}; // 到每个对端有几个绕路路径
    uint64_t pathNumPerPeer_{0};
    std::vector<uint64_t> lengths_;
    uint64_t singleTransportSize_{0};
};

} // namespace Hccl

#endif // HCCLV2_CCU_TEMP_ALL_GATHER_MESH_1D_H_
