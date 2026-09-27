/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_CCU_TEMP_ALL_REDUCE_NHR_1D_MEM2MEM_H_
#define HCCLV2_CCU_TEMP_ALL_REDUCE_NHR_1D_MEM2MEM_H_
#include "string_util.h"
#include "env_config_v2.h"
#include "ccu_alg_template_base.h"
#include "ccu_instruction_all_reduce_nhr1d_mem2mem.h"
#include "executor_utils.h"

namespace Hccl {
class CcuTempAllReduceNHRMem2Mem1D : public CcuAlgTemplateBase {
public:
    explicit CcuTempAllReduceNHRMem2Mem1D(
        const RankId virtualRank, const u32 tempRankSize, const std::vector<std::vector<RankId>>& tempVTopo,
        const std::map<RankId, u32>& tempVirtRankMap);
    ~CcuTempAllReduceNHRMem2Mem1D() override;

    std::string Describe() const override
    {
        return StringFormat("Template of AllReduce ccu nhr 1D mem2mem with tempRankSize [%u].", tempRankSize_);
    }

    HcclResult GenExtIns(
        const TempFuncs& tempFuncs, TemplateDataParams& tempAlgParams, const ResLinks& tempLinks,
        std::vector<InsQuePtr>& tempInsQues);
    HcclResult CalcRes(AlgTempResReq& tempResReq) override;
    uint64_t GetMaxSliceSize() const;
    void InitReduceInfo(const ReduceOp& reduceOp, const DataType& dataType);

private:
    HcclResult CalcSlice(const u64 dataSize, RankSliceInfo& sliceInfoVec);
    HcclResult GetStepInfo(u32 step, u32 nSteps, NHRStepInfo& stepInfo);
    HcclResult GetAllGatherStepInfo(u32 step, u32 nSteps, NHRStepInfo& stepInfo);
    HcclResult GetReduceScatterStepInfo(u32 step, NHRStepInfo& stepInfo);
    HcclResult ProcessNHRStepInfo(
        std::vector<NHRStepInfo>& stepInfoVector, RankGroup& rankGroup, std::map<u32, u32>& indexMap,
        std::vector<LinkData>& linksDie0, std::vector<LinkData>& linksDie1, const ResLinks& tempLinks,
        uint32_t axisSize);
    HcclResult
    SplitDataFor2Dies(uint64_t dataCount, const ResLinks& tempLinks, uint64_t& die0Size, uint64_t& die1Size) const;
    uint32_t virtRankId2RankId(const u32 virtRankId);
    ReduceOp reduceOp_;
    DataType dataType_;
};

} // namespace Hccl

#endif // HCCLV2_CCU_TEMP_ALL_REDUCE_MESH_1D_MEM2MEM_H_
