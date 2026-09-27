/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_CCU_INSTRUCTION_REDUCE_MESH_2D_H_
#define HCCLV2_CCU_INSTRUCTION_REDUCE_MESH_2D_H_

#include "template_utils.h"
#include "instruction.h"
#include "ins_queue.h"
#include "ccu_context_utils.h"
#include "ccu_ctx_signature.h"
#include "ccu_ins.h"
#include "ccu_rank_group.h"

namespace Hccl {

// 为ReduceMesh2D实现的CCUIns、CCUCtxArg与CCUTaskArg
class CcuCtxArgReduceMesh2D : public CcuCtxArg {
public:
    explicit CcuCtxArgReduceMesh2D(
        const std::vector<uint64_t>& dSize, uint32_t rId, uint32_t rootId, uint32_t aId, const CollAlgOperator& op,
        const std::vector<std::vector<RankId>>& tempVTopo)
        : dimSize_(dSize),
          rankId_(rId),
          rootId_(rootId),
          axisId_(aId),
          op_(op),
          tempVTopo_(tempVTopo)
    {}
    CcuCtxSignature GetCtxSignature() const override
    {
        CcuCtxSignature signature;
        GenerateCcuCtxSignature(signature, CcuInstType::CCU_REDUCE_MESH_2D_DIRECT, op_, tempVTopo_);
        return signature;
    }

    std::vector<uint64_t> dimSize_;
    uint32_t rankId_;
    uint32_t rootId_;
    uint32_t axisId_;
    CollAlgOperator op_;
    std::vector<std::vector<RankId>> tempVTopo_;
};

class CcuTaskArgReduceMesh2D : public CcuTaskArg {
public:
    explicit CcuTaskArgReduceMesh2D(
        uint64_t inputAddr, uint64_t outputAddr, uint64_t sliceSize, uint64_t xAxisSize, uint64_t yAxisSize,
        uint64_t offset, uint64_t token)
        : inputAddr_(inputAddr),
          outputAddr_(outputAddr),
          sliceSize_(sliceSize),
          xAxisSize_(xAxisSize),
          yAxisSize_(yAxisSize),
          offset_(offset),
          token_(token)
    {}

    uint64_t inputAddr_;
    uint64_t outputAddr_;
    uint64_t sliceSize_;
    uint64_t xAxisSize_;
    uint64_t yAxisSize_;
    uint64_t offset_;
    uint64_t token_;
};

class CcuInstructionReduceMesh2D : public CcuInstruction {
public:
    CcuInstructionReduceMesh2D() : CcuInstruction() {}

    void Init(
        std::vector<uint64_t> dimSize, uint32_t rankId, uint32_t rootId, uint64_t axisId, uint64_t inputAddr,
        uint64_t outputAddr, uint64_t sliceSize, uint64_t xAxisSize, uint64_t yAxisSize, uint64_t offset,
        uint64_t token, CollAlgOperator& op, std::vector<std::vector<RankId>>& tempVTopo)
    {
        dimSize_ = dimSize;
        rankId_ = rankId;
        rootId_ = rootId;
        axisId_ = axisId;
        inputAddr_ = inputAddr;
        outputAddr_ = outputAddr;
        sliceSize_ = sliceSize;
        xAxisSize_ = xAxisSize;
        yAxisSize_ = yAxisSize;
        offset_ = offset;
        token_ = token;
        op_ = op;
        tempVTopo_ = tempVTopo;
        return;
    }

    CcuInstType GetInstType() const override
    {
        HCCL_INFO("CcuInstructionReduceMesh2D instype is CCU_REDUCE_MESH_2D_DIRECT.");
        return instType_;
    }

    std::string Describe() const override
    {
        return StringFormat("CcuInstructionReduceMesh2D rankId [%u], instType[%d]", rankId_, instType_);
    }

    void SetInstType(CcuInstType instType) { instType_ = instType; }

    std::unique_ptr<CcuCtxArg> GetCtxArg() const override
    {
        HCCL_INFO("[CcuInstructionReduceMesh2D] GetCtxArg begin");
        return std::make_unique<CcuCtxArgReduceMesh2D>(dimSize_, rankId_, rootId_, axisId_, op_, tempVTopo_);
    }

    std::unique_ptr<CcuTaskArg> GetTaskArg() const override
    {
        HCCL_INFO("[CcuInstructionReduceMesh2D] GetTaskArg begin");
        return std::make_unique<CcuTaskArgReduceMesh2D>(
            inputAddr_, outputAddr_, sliceSize_, xAxisSize_, yAxisSize_, offset_, token_);
    }

private:
    CcuInstType instType_ = CcuInstType::CCU_REDUCE_MESH_2D_DIRECT;
    std::vector<uint64_t> dimSize_;
    uint32_t rankId_{0};
    uint32_t rootId_{0};
    uint64_t axisId_{0};
    uint64_t offset_{0};
    uint64_t token_{0};
    uint64_t inputAddr_{0};
    uint64_t outputAddr_{0};
    uint64_t sliceSize_{0};
    uint64_t xAxisSize_{0};
    uint64_t yAxisSize_{0};
    CollAlgOperator op_;
    std::vector<std::vector<RankId>> tempVTopo_;
};

} // namespace Hccl
#endif // HCCLV2_CCU_INSTRUCTION_REDUCE_MESH_2D_H_
