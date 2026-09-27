/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_CCU_TEMP_REDUCE_SCATTER_NHR_1D_H_
#define HCCLV2_CCU_TEMP_REDUCE_SCATTER_NHR_1D_H_
#include "string_util.h"
#include "env_config_v2.h"
#include "ccu_alg_template_base.h"
#include "ccu_instruction_reduce_scatter_nhr1d_mem2mem.h"
#include "executor_utils.h"

namespace Hccl {
class CcuTempReduceScatterNHR1DMem2Mem : public CcuAlgTemplateBase {
public:
    explicit CcuTempReduceScatterNHR1DMem2Mem(
        const RankId virtualRank, const u32 tempRankSize, const std::vector<std::vector<RankId>>& tempVTopo,
        const std::map<RankId, u32>& tempVirtRankMap);
    ~CcuTempReduceScatterNHR1DMem2Mem() override;

    std::string Describe() const override
    {
        return StringFormat("Template of Reduce Scatter ccu nhr 1D mem2mem with tempRankSize [%u].", tempRankSize_);
    }

    HcclResult GenExtIns(
        const TempFuncs& tempFuncs, TemplateDataParams& tempAlgParams, const ResLinks& tempLinks,
        std::vector<InsQuePtr>& tempInsQues);
    HcclResult CalcRes(AlgTempResReq& tempResReq) override;
    uint64_t GetMaxSliceSize() const;
    void InitReduceInfo(const ReduceOp& reduceOp, const DataType& dataType);
    u32 CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType) override;

private:
    HcclResult GetStepInfo(u32 step, NHRStepInfo& stepInfo);
    RankId GetRankFromMap(const u32 rankIdx);
    ReduceOp reduceOp_;
    DataType dataType_;
    bool isSipportTwoDie_{false};
    u32 linkNum_{1};
};

} // namespace Hccl

#endif // HCCLV2_CCU_TEMP_REDUCE_SCATTER_NHR_1D_H_
