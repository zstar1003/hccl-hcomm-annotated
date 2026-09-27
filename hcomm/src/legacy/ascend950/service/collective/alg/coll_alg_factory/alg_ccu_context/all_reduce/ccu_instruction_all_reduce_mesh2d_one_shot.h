/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_CCU_INSTRUCTION_ALL_REDUCE_MESH_2D_ONE_SHOT_H_
#define HCCLV2_CCU_INSTRUCTION_ALL_REDUCE_MESH_2D_ONE_SHOT_H_

#include "template_utils.h"
#include "instruction.h"
#include "ins_queue.h"
#include "ccu_context_utils.h"
#include "ccu_ctx_signature.h"
#include "ccu_ins.h"
#include "ccu_rank_group.h"

namespace Hccl {

// 为AllReduceMesh2DOneShot实现的CCUIns、CCUCtxArg与CCUTaskArg
class CcuCtxArgAllReduceMesh2DOneShot : public CcuCtxArg {
public:
    explicit CcuCtxArgAllReduceMesh2DOneShot(
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
        GenerateCcuCtxSignature(signature, CcuInstType::CCU_ALL_REDUCE_MESH_2D_ONE_SHOT_DIRECT, op_, tempVTopo_);
        return signature;
    }

    std::vector<uint64_t> dimSize_;
    uint32_t rankId_;
    uint32_t axisId_;
    CollAlgOperator op_;
    std::vector<std::vector<RankId>> tempVTopo_;
};

class CcuTaskArgAllReduceMesh2DOneShot : public CcuTaskArg {
public:
    explicit CcuTaskArgAllReduceMesh2DOneShot(
        uint64_t inputAddr, uint64_t outputAddr, uint64_t scratchAddr, uint64_t xSliceSize, uint64_t ySliceSize,
        uint64_t xSliceOffset, uint64_t ySliceOffset, uint64_t token)
        : inputAddr_(inputAddr),
          outputAddr_(outputAddr),
          scratchAddr_(scratchAddr),
          xSliceSize_(xSliceSize),
          ySliceSize_(ySliceSize),
          xSliceOffset_(xSliceOffset),
          ySliceOffset_(ySliceOffset),
          token_(token)
    {}
    uint64_t inputAddr_;
    uint64_t outputAddr_;
    uint64_t scratchAddr_;
    uint64_t xSliceSize_;
    uint64_t ySliceSize_;
    uint64_t xSliceOffset_;
    uint64_t ySliceOffset_;
    uint64_t token_;
};

class CcuInstructionAllReduceMesh2DOneShot : public CcuInstruction {
public:
    CcuInstructionAllReduceMesh2DOneShot() : CcuInstruction() {}

    void Init(
        std::vector<uint64_t>& dimSize, uint32_t rankId, uint64_t inputAddr, uint64_t outputAddr, uint64_t scratchAddr,
        uint32_t axisId, uint64_t xSliceSize, uint64_t ySliceSize, uint64_t xSliceOffset, uint64_t ySliceOffset,
        uint64_t token, CollAlgOperator& op, std::vector<std::vector<RankId>>& tempVTopo)
    {
        dimSize_ = dimSize;
        rankId_ = rankId;
        inputAddr_ = inputAddr;
        outputAddr_ = outputAddr;
        scratchAddr_ = scratchAddr;

        axisId_ = axisId;

        xSliceSize_ = xSliceSize;
        ySliceSize_ = ySliceSize;

        xSliceOffset_ = xSliceOffset;
        ySliceOffset_ = ySliceOffset;

        token_ = token;
        op_ = op;
        tempVTopo_ = tempVTopo;
        return;
    }

    CcuInstType GetInstType() const override
    {
        HCCL_INFO("CcuInstructionAllReduceMesh2DOneShot instype is CCU_ALL_REDUCE_MESH_2D_ONE_SHOT_DIRECT.");
        return instType_;
    }

    std::string Describe() const override
    {
        return StringFormat(
            "CcuInstructionAllReduceMesh2DOneShot rankId [%u], instType[%s]", rankId_, instType_.Describe().c_str());
    }

    std::unique_ptr<CcuCtxArg> GetCtxArg() const override
    {
        return std::make_unique<CcuCtxArgAllReduceMesh2DOneShot>(dimSize_, rankId_, axisId_, op_, tempVTopo_);
    }

    void SetInstType(CcuInstType instType) { instType_ = instType; }

    std::unique_ptr<CcuTaskArg> GetTaskArg() const override
    {
        return std::make_unique<CcuTaskArgAllReduceMesh2DOneShot>(
            inputAddr_, outputAddr_, scratchAddr_, xSliceSize_, ySliceSize_, xSliceOffset_, ySliceOffset_, token_);
    }

private:
    CcuInstType instType_ = CcuInstType::CCU_ALL_REDUCE_MESH_2D_ONE_SHOT_DIRECT;
    std::vector<uint64_t> dimSize_;
    uint32_t rankId_{0};

    uint64_t inputAddr_{0};
    uint64_t outputAddr_{0};
    uint64_t scratchAddr_{0};

    uint32_t axisId_{0};

    uint64_t xSliceSize_{0};
    uint64_t ySliceSize_{0};
    uint64_t xSliceOffset_{0};
    uint64_t ySliceOffset_{0};

    uint64_t token_{0};
    CollAlgOperator op_;
    std::vector<std::vector<RankId>> tempVTopo_;
};

} // namespace Hccl
#endif // HCCLV2_CCU_INSTRUCTION_ALL_REDUCE_MESH_2D_ONE_SHOT_H_
