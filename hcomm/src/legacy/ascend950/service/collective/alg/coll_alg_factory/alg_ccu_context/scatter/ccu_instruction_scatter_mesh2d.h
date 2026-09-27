/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_CCU_INSTRUCTION_SCATTER_MESH_2D_H_
#define HCCLV2_CCU_INSTRUCTION_SCATTER_MESH_2D_H_

#include "template_utils.h"
#include "instruction.h"
#include "ins_queue.h"
#include "ccu_context_utils.h"
#include "ccu_ctx_signature.h"
#include "ccu_ins.h"
#include "ccu_rank_group.h"

namespace Hccl {

// 为ScatterMesh2D实现的CCUIns、CCUCtxArg与CCUTaskArg
class CcuCtxArgScatterMesh2D : public CcuCtxArg {
public:
    explicit CcuCtxArgScatterMesh2D(
        const std::vector<uint64_t>& dSize, uint32_t rankSize, uint32_t rId, uint32_t axisId, uint32_t root,
        const CollAlgOperator& op, const std::vector<std::vector<RankId>>& tempVTopo)
        : dimSize_(dSize),
          rankSize_(rankSize),
          rankId_(rId),
          axisId_(axisId),
          root_(root),
          op_(op),
          tempVTopo_(tempVTopo)
    {}

    CcuCtxSignature GetCtxSignature() const override
    {
        CcuCtxSignature signature;
        GenerateCcuCtxSignature(signature, CcuInstType::CCU_SCATTER_MESH_2D_DIRECT, op_, tempVTopo_);
        return signature;
    }

    std::vector<uint64_t> dimSize_;
    uint32_t rankSize_;
    uint32_t rankId_;
    uint32_t axisId_;
    uint32_t root_;
    CollAlgOperator op_;
    std::vector<std::vector<RankId>> tempVTopo_;
};

class CcuTaskArgScatterMesh2D : public CcuTaskArg {
public:
    explicit CcuTaskArgScatterMesh2D(
        uint64_t inputAddr, uint64_t outputAddr, uint64_t scratchAddr, uint64_t token, uint64_t sliceSize,
        uint64_t stride, uint64_t xSliceSize, uint64_t ySliceSize)
        : inputAddr_(inputAddr),
          outputAddr_(outputAddr),
          scratchAddr_(scratchAddr),
          token_(token),
          sliceSize_(sliceSize),
          stride_(stride),
          xSliceSize_(xSliceSize),
          ySliceSize_(ySliceSize)
    {}

    uint64_t inputAddr_;
    uint64_t outputAddr_;
    uint64_t scratchAddr_;
    uint64_t token_;
    uint64_t sliceSize_;
    uint64_t stride_;
    uint64_t xSliceSize_;
    uint64_t ySliceSize_;
};

class CcuInstructionScatterMesh2D : public CcuInstruction {
public:
    CcuInstructionScatterMesh2D() : CcuInstruction() {}

    void Init(
        uint32_t rankId, uint32_t rankSize, uint32_t axisId, uint64_t inputAddr, uint64_t outputAddr,
        uint64_t scratchAddr, uint64_t token, uint64_t sliceSize, uint64_t stride, uint64_t xSliceSize,
        uint64_t ySliceSize, CollAlgOperator& op, std::vector<std::vector<RankId>>& tempVTopo)
    {
        dimSize_.push_back(tempVTopo[0].size());
        dimSize_.push_back(tempVTopo[1].size());
        rankId_ = rankId;
        rankSize_ = rankSize;
        axisId_ = axisId;
        inputAddr_ = inputAddr;
        outputAddr_ = outputAddr;
        scratchAddr_ = scratchAddr;
        sliceSize_ = sliceSize;
        token_ = token;
        stride_ = stride;
        xSliceSize_ = xSliceSize;
        ySliceSize_ = ySliceSize;
        op_ = op;
        tempVTopo_ = tempVTopo;

        if (dimSize_[0] * dimSize_[1] != rankSize_) {
            THROW<InvalidParamsException>(StringFormat(
                "[CcuInstructionScatterMesh2D] each DimSize and rankSize is NOT match. dimSize[0][%llu], "
                "dimSize[1][%llu], rankSize[%llu]",
                dimSize_[0], dimSize_[1], rankSize_));
        }

        return;
    }

    CcuInstType GetInstType() const override
    {
        HCCL_INFO("CcuInstructionScatterMesh2D instype is CCU_SCATTER_MESH_2D_DIRECT.");
        return instType_;
    }

    void SetInstType(CcuInstType instType) { instType_ = instType; }

    std::string Describe() const override
    {
        return StringFormat(
            "CcuInstructionScatterMesh2D rankId [%u], instType[%s]", rankId_, instType_.Describe().c_str());
    }

    std::unique_ptr<CcuCtxArg> GetCtxArg() const override
    {
        return std::make_unique<CcuCtxArgScatterMesh2D>(
            dimSize_, rankSize_, rankId_, axisId_, op_.root, op_, tempVTopo_);
    }

    std::unique_ptr<CcuTaskArg> GetTaskArg() const override
    {
        return std::make_unique<CcuTaskArgScatterMesh2D>(
            inputAddr_, outputAddr_, scratchAddr_, token_, sliceSize_, stride_, xSliceSize_, ySliceSize_);
    }

private:
    CcuInstType instType_ = CcuInstType::CCU_SCATTER_MESH_2D_DIRECT;
    std::vector<uint64_t> dimSize_;
    uint32_t rankId_{0};
    uint32_t rankSize_{0};
    uint32_t axisId_{0};
    uint64_t inputAddr_{0};
    uint64_t outputAddr_{0};
    uint64_t scratchAddr_{0};
    uint64_t token_{0};
    uint64_t stride_{0};
    uint64_t sliceSize_{0};
    uint64_t xSliceSize_{0};
    uint64_t ySliceSize_{0};
    CollAlgOperator op_;
    std::vector<std::vector<RankId>> tempVTopo_;
};

} // namespace Hccl
#endif // HCCLV2_CCU_INSTRUCTION_SCATTER_MESH_2D_H_
