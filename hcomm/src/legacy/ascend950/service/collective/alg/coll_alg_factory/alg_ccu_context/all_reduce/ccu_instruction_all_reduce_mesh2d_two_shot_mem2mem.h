/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_CCU_INSTRUCTION_ALL_REDUCE_MESH_2D_TWO_SHOT_MEM2MEM_H_
#define HCCLV2_CCU_INSTRUCTION_ALL_REDUCE_MESH_2D_TWO_SHOT_MEM2MEM_H_

#include "template_utils.h"
#include "instruction.h"
#include "ins_queue.h"
#include "ccu_context_utils.h"
#include "ccu_ctx_signature.h"
#include "ccu_ins.h"
#include "ccu_rank_group.h"

namespace Hccl {

// 为AllReduceMesh2DTwoShot实现的CCUIns、CCUCtxArg与CCUTaskArg
class CcuCtxArgAllReduceMeshTwoShotMem2Mem2D : public CcuCtxArg {
public:
    explicit CcuCtxArgAllReduceMeshTwoShotMem2Mem2D(
        const std::vector<uint64_t>& dSize, uint32_t rId, uint32_t axisId, const CollAlgOperator& op,
        const std::vector<std::vector<RankId>>& tempVTopo)
        : dimSize_(dSize),
          rankId_(rId),
          axisId_(axisId),
          op_(op),
          tempVTopo_(tempVTopo)
    {}
    CcuCtxSignature GetCtxSignature() const override
    {
        CcuCtxSignature signature;
        GenerateCcuCtxSignature(signature, CcuInstType::CCU_ALL_REDUCE_MESH_2D_TWO_SHOT_MEM2MEM, op_, tempVTopo_);
        return signature;
    }

    std::vector<uint64_t> dimSize_;
    uint32_t rankId_;
    uint32_t axisId_;
    CollAlgOperator op_;
    std::vector<std::vector<RankId>> tempVTopo_;
};

class CcuTaskArgAllReduceMeshTwoShotMem2Mem2D : public CcuTaskArg {
public:
    explicit CcuTaskArgAllReduceMeshTwoShotMem2Mem2D(
        uint64_t inputAddr, uint64_t outputAddr, uint64_t normalRankXSliceSize, uint64_t normalRankYSliceSize,
        uint64_t lastRankXSliceSize, uint64_t lastRankYSliceSize, uint64_t token)
        : inputAddr_(inputAddr),
          outputAddr_(outputAddr),
          normalRankXSliceSize_(normalRankXSliceSize),
          normalRankYSliceSize_(normalRankYSliceSize),
          lastRankXSliceSize_(lastRankXSliceSize),
          lastRankYSliceSize_(lastRankYSliceSize),
          token_(token)
    {}
    uint64_t inputAddr_;
    uint64_t outputAddr_;
    uint64_t normalRankXSliceSize_;
    uint64_t normalRankYSliceSize_;
    uint64_t lastRankXSliceSize_;
    uint64_t lastRankYSliceSize_;
    uint64_t token_;
};

class CcuInstructionAllReduceMeshTwoShotMem2Mem2D : public CcuInstruction {
public:
    CcuInstructionAllReduceMeshTwoShotMem2Mem2D() : CcuInstruction() {}

    void Init(
        std::vector<uint64_t>& dimSize, uint32_t rankId, uint64_t inputAddr, uint64_t outputAddr, uint32_t axisId,
        uint64_t normalRankXSliceSize, uint64_t normalRankYSliceSize, uint64_t lastRankXSliceSize,
        uint64_t lastRankYSliceSize, uint64_t token, CollAlgOperator& op, std::vector<std::vector<RankId>>& tempVTopo)
    {
        dimSize_ = dimSize;
        rankId_ = rankId;
        inputAddr_ = inputAddr;
        outputAddr_ = outputAddr;

        axisId_ = axisId;

        normalRankXSliceSize_ = normalRankXSliceSize;
        normalRankYSliceSize_ = normalRankYSliceSize;
        lastRankXSliceSize_ = lastRankXSliceSize;
        lastRankYSliceSize_ = lastRankYSliceSize;

        token_ = token;
        op_ = op;
        tempVTopo_ = tempVTopo;
        return;
    }

    CcuInstType GetInstType() const override
    {
        HCCL_INFO("CcuInstructionAllReduceMeshTwoShotMem2Mem2D instype is CCU_ALL_REDUCE_MESH_2D_TWO_SHOT_MEM2MEM.");
        return instType_;
    }

    std::string Describe() const override
    {
        return StringFormat(
            "CcuInstructionAllReduceMeshTwoShotMem2Mem2D rankId [%u], instType[%s]", rankId_,
            instType_.Describe().c_str());
    }

    void SetInstType(CcuInstType instType) { instType_ = instType; }

    std::unique_ptr<CcuCtxArg> GetCtxArg() const override
    {
        return std::make_unique<CcuCtxArgAllReduceMeshTwoShotMem2Mem2D>(dimSize_, rankId_, axisId_, op_, tempVTopo_);
    }

    std::unique_ptr<CcuTaskArg> GetTaskArg() const override
    {
        return std::make_unique<CcuTaskArgAllReduceMeshTwoShotMem2Mem2D>(
            inputAddr_, outputAddr_, normalRankXSliceSize_, normalRankYSliceSize_, lastRankXSliceSize_,
            lastRankYSliceSize_, token_);
    }

private:
    CcuInstType instType_ = CcuInstType::CCU_ALL_REDUCE_MESH_2D_TWO_SHOT_MEM2MEM;
    std::vector<uint64_t> dimSize_;
    uint32_t rankId_{0};
    uint32_t axisId_{0};
    uint64_t inputAddr_{0};
    uint64_t outputAddr_{0};
    uint64_t normalRankXSliceSize_{0};
    uint64_t normalRankYSliceSize_{0};
    uint64_t lastRankXSliceSize_{0};
    uint64_t lastRankYSliceSize_{0};
    uint64_t token_{0};
    CollAlgOperator op_;
    std::vector<std::vector<RankId>> tempVTopo_;
};

} // namespace Hccl
#endif // HCCLV2_CCU_INSTRUCTION_ALL_REDUCE_MESH_2D_TWO_SHOT_MEM2MEM_H_
