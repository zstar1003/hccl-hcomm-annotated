/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_INS_TEMP_ALL_TO_ALL_MESH_2D
#define HCCLV2_INS_TEMP_ALL_TO_ALL_MESH_2D

#include "string_util.h"
#include "executor_utils.h"
#include "ins_alg_template_base.h"

namespace Hccl {

constexpr u32 TEMPVTOPOSIZE = 2;

class InsTempAlltoAllMesh2D : public InsAlgTemplateBase {
public:
    explicit InsTempAlltoAllMesh2D(
        const RankId virtualRank, const u32 tempRankSize, const std::vector<std::vector<RankId>>& tempVTopo,
        const std::map<RankId, u32>& tempVirtRankMap);
    ~InsTempAlltoAllMesh2D() override;

    std::string Describe() const override
    {
        return StringFormat("Instruction based Template of all to all mesh 2d with tempRankSize [%u].", tempRankSize_);
    }
    u32 CalcScratchMultiple(const BufferType& inBufferTpye, const BufferType& outBufferTpye) const
    {
        (void)inBufferTpye;
        (void)outBufferTpye;
        // 单算子和图模式一致，AlltoAll的usrIn、scratchBuffer，usrOut大小一致
        return 1;
    }

    HcclResult GenExtIns(
        const TempFuncs& tempFuncs, const TemplateDataParams& tempAlgParams, const ResLinks& tempLinks,
        std::vector<InsQuePtr>& tempInsQues) const;

    HcclResult CalcRes(AlgTempResReq& tempResReq) override;

private:
    HcclResult LocalDataCopy(const TemplateDataParams& tempAlgParams, std::vector<InsQuePtr>& tempInsQues);
    HcclResult PostLocalCopy(const TemplateDataParams& tempAlgParams, std::vector<InsQuePtr>& tempInsQues);

    HcclResult RunMeshX(
        std::vector<u64>& xDataInAddr, std::vector<u64>& xDataOutAddr, u64 xSize, BufferType srcBufferType,
        BufferType dstBufferType, DmaMode dmaMode, std::vector<InsQuePtr>& xInsQues, const ResLinks& tempLinks) const;
    HcclResult RunMeshY(
        std::vector<u64>& yDataInAddr, std::vector<u64>& yDataOutAddr, u64 ySize, BufferType srcBufferType,
        BufferType dstBufferType, DmaMode dmaMode, std::vector<InsQuePtr>& yInsQues, const ResLinks& tempLinks) const;

    u32 rankId_ = 0;
    u32 xRankId_ = 0;
    u32 yRankId_ = 0;
    u32 rankSize_ = 0;
    u32 xRankSize_ = 0;
    u32 yRankSize_ = 0;
};

} // namespace Hccl

#endif // !HCCLV2_INS_TEMP_ALL_TO_ALL_MESH_2D
