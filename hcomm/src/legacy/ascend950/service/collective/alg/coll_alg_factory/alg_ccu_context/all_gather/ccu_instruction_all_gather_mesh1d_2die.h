/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_CCU_INSTRUCTION_ALL_GATHER_MESH_1D_2DIE_H_
#define HCCLV2_CCU_INSTRUCTION_ALL_GATHER_MESH_1D_2DIE_H_

#include "template_utils.h"
#include "instruction.h"
#include "ins_queue.h"
#include "ccu_context_utils.h"
#include "ccu_ctx_signature.h"
#include "ccu_ins.h"
#include "ccu_rank_group.h"

namespace Hccl {

// 为AllGatherMesh1D实现的CCUIns、CCUCtxArg与CCUTaskArg
class CcuCtxArgAllGatherMesh1D2Die : public CcuCtxArg {
public:
    explicit CcuCtxArgAllGatherMesh1D2Die(
        const std::vector<uint64_t>& dSize, uint32_t rId, bool withMyRank, const CollAlgOperator& op,
        const std::vector<std::vector<RankId>>& tempVTopo)
        : dimSize_(dSize),
          rankId_(rId),
          withMyRank_(withMyRank),
          op_(op),
          tempVTopo_(tempVTopo)
    {}
    CcuCtxSignature GetCtxSignature() const override
    {
        CcuCtxSignature signature;
        GenerateCcuCtxSignature(signature, CcuInstType::CCU_ALLGATHER_MESH_1D_2DIE, op_, tempVTopo_);
        return signature;
    }
    std::vector<uint64_t> dimSize_;
    uint32_t rankId_;
    bool withMyRank_;
    CollAlgOperator op_;
    std::vector<std::vector<RankId>> tempVTopo_;
};

class CcuTaskArgAllGatherMesh1D2Die : public CcuTaskArg {
public:
    explicit CcuTaskArgAllGatherMesh1D2Die(
        uint64_t inputAddr, uint64_t outputAddr, uint64_t sliceSize, uint64_t outputSliceStride, uint64_t token)
        : inputAddr_(inputAddr),
          outputAddr_(outputAddr),
          sliceSize_(sliceSize),
          outputSliceStride_(outputSliceStride),
          token_(token)
    {}

    uint64_t inputAddr_;
    uint64_t outputAddr_;
    uint64_t sliceSize_;
    uint64_t outputSliceStride_;
    uint64_t token_;
};

class CcuInstructionAllGatherMesh1D2Die : public CcuInstruction {
public:
    CcuInstructionAllGatherMesh1D2Die() : CcuInstruction() {}

    void Init(
        uint32_t rankId, uint64_t inputAddr, uint64_t outputAddr, uint64_t sliceSize, uint64_t outputSliceStride,
        uint64_t token, uint64_t withMyRank, CollAlgOperator& op, std::vector<std::vector<RankId>>& tempVTopo)
    {
        u32 maxDimNum = 1;
        if (tempVTopo.size() != maxDimNum) {
            THROW<InvalidParamsException>(StringFormat(
                "[CcuInstructionAllGatherMesh1D2Die] tempVTopo size is not 1, size is [%zu].", tempVTopo.size()));
        }
        dimSize_.push_back(tempVTopo[0].size());
        rankId_ = rankId;
        inputAddr_ = inputAddr;
        outputAddr_ = outputAddr;
        token_ = token;
        outputSliceStride_ = outputSliceStride;
        withMyRank_ = withMyRank;
        op_ = op;
        tempVTopo_ = tempVTopo;
        sliceSize_ = sliceSize;
        return;
    }

    void SetInstType(CcuInstType instType) { instType_ = instType; }

    std::string Describe() const override
    {
        return StringFormat(
            "CcuInstructionAllGatherMesh1D2Die rankId [%u], instType[%s]", rankId_, instType_.Describe().c_str());
    }

    CcuInstType GetInstType() const override
    {
        HCCL_INFO("CcuInstructionAllGatherMesh1D2Die instype is CCU_ALLGATHER_MESH_1D_2DIE.");
        return instType_;
    }

    std::unique_ptr<CcuCtxArg> GetCtxArg() const override
    {
        return std::make_unique<CcuCtxArgAllGatherMesh1D2Die>(dimSize_, rankId_, withMyRank_, op_, tempVTopo_);
    }

    std::unique_ptr<CcuTaskArg> GetTaskArg() const override
    {
        return std::make_unique<CcuTaskArgAllGatherMesh1D2Die>(
            inputAddr_, outputAddr_, sliceSize_, outputSliceStride_, token_);
    }

private:
    CcuInstType instType_ = CcuInstType::CCU_ALLGATHER_MESH_1D_2DIE;
    std::vector<uint64_t> dimSize_;
    uint32_t rankId_{0};
    bool withMyRank_{false};
    uint64_t inputAddr_{0};
    uint64_t outputAddr_{0};
    uint64_t sliceSize_{0};
    uint64_t outputSliceStride_{0};
    uint64_t token_{0};
    CollAlgOperator op_;
    std::vector<std::vector<RankId>> tempVTopo_;
};

} // namespace Hccl
#endif // HCCLV2_CCU_INSTRUCTION_ALL_GATHER_MESH_1D_2DIE_H_
