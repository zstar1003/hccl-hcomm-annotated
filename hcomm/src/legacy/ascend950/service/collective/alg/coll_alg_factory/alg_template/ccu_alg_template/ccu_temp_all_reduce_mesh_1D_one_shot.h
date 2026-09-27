/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_CCU_TEMP_ALL_REDUCE_MESH_1D_ONE_SHOT_H_
#define HCCLV2_CCU_TEMP_ALL_REDUCE_MESH_1D_ONE_SHOT_H_
#include <vector>
#include <map>
#include <string>
#include <hccl/hccl_types.h>
#include "template_utils.h"
#include "reduce_op.h"
#include "hccl/base.h"
#include "types/types.h"
#include "string_util.h"
#include "env_config_v2.h"
#include "data_type.h"
#include "ccu_alg_template_base.h"
#include "ccu_instruction_all_reduce_mesh1d.h"
#include "ccu_alg_template_base.h"

namespace Hccl {

class CcuTempAllReduceMesh1DOneShot : public CcuAlgTemplateBase {
public:
    explicit CcuTempAllReduceMesh1DOneShot(
        const RankId virtualRank, const u32 tempRankSize, const std::vector<std::vector<RankId>>& tempVTopo,
        const std::map<RankId, u32>& tempVirtRankMap);
    ~CcuTempAllReduceMesh1DOneShot() override;

    std::string Describe() const override
    {
        return StringFormat("Template of CcuTempAllReduceMesh1DOneShot tempRankSize [%u].", tempRankSize_);
    }

    HcclResult
    Run(const TempFuncs& tempFuncs, const RankSliceInfo& sliceInfoVec, const BuffInfo& buffInfo,
        const ResLinks& tempLinks, std::vector<InsQuePtr>& tempInsQues) override;
    HcclResult CalcRes(AlgTempResReq& tempResReq) override;
    HcclResult CalcSliceInfo(const AllignInfo& allignInfo, const u64 dataSize, RankSliceInfo& sliceInfoVec) override;
    // init reduceInfo
    void InitReduceInfo(const ReduceOp& reduceOp, const DataType& dataType);

private:
    HcclResult CalcInputOutputAddr(const TempFuncs& tempFuncs, uint64_t& inputAddr, uint64_t& outputAddr);
    HcclResult PrepareLinks(const ResLinks& tempLinks, std::vector<LinkData>& links) const;
    HcclResult PrepareRankGroup(RankGroup& rankGroup);

private:
    ReduceOp reduceOp_;
    DataType dataType_;
};

} // namespace Hccl

#endif // HCCLV2_CCU_TEMP_ALL_REDUCE_MESH_1D_H_
