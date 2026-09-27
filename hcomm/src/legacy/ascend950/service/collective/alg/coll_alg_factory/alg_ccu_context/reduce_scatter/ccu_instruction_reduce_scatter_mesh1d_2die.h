/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_CCU_INSTRUCTION_REDUCE_SCATTER_MESH_1D_2DIE_H_
#define HCCLV2_CCU_INSTRUCTION_REDUCE_SCATTER_MESH_1D_2DIE_H_

#include "template_utils.h"
#include "instruction.h"
#include "ins_queue.h"
#include "ccu_context_utils.h"
#include "ccu_ctx_signature.h"
#include "ccu_ins.h"
#include "ccu_rank_group.h"

namespace Hccl {

// 为ReduceScatterMesh1D实现的CCUIns、CCUCtxArg与CCUTaskArg
class CcuCtxArgReduceScatterMesh1D2Die : public CcuCtxArg {
public:
    explicit CcuCtxArgReduceScatterMesh1D2Die(
        const std::vector<uint64_t>& dSize, uint32_t rId, const CollAlgOperator& op, bool rmtReduceWithMyRank,
        const std::vector<std::vector<RankId>>& tempVTopo)
        : dimSize_(dSize),
          rankId_(rId),
          rmtReduceWithMyRank_(rmtReduceWithMyRank),
          op_(op),
          tempVTopo_(tempVTopo)
    {}
    CcuCtxSignature GetCtxSignature() const override
    {
        CcuCtxSignature signature;
        GenerateCcuCtxSignature(signature, CcuInstType::CCU_REDUCE_SCATTER_MESH_1D_DIRECT, op_, tempVTopo_);
        return signature;
    }
    std::vector<uint64_t> dimSize_;
    u32 rankId_;
    bool rmtReduceWithMyRank_;
    CollAlgOperator op_;
    std::vector<std::vector<RankId>> tempVTopo_;
};

class CcuTaskArgReduceScatterMesh1D2Die : public CcuTaskArg {
public:
    explicit CcuTaskArgReduceScatterMesh1D2Die(
        uint64_t inputAddr, uint64_t outputAddr, uint64_t scratchAddr, uint64_t sliceSize, uint64_t token)
        : inputAddr_(inputAddr),
          outputAddr_(outputAddr),
          scratchAddr_(scratchAddr),
          sliceSize_(sliceSize),
          token_(token)
    {}

    uint64_t inputAddr_{0};
    uint64_t outputAddr_{0};
    uint64_t scratchAddr_{0};
    uint64_t sliceSize_{0};
    uint64_t token_{0};
};

class CcuInstructionReduceScatterMesh1D2Die : public CcuInstruction {
public:
    CcuInstructionReduceScatterMesh1D2Die() : CcuInstruction() {}

    void Init(
        uint32_t rankId, CollAlgOperator& op, bool rmtReduceWithMyRank, std::vector<std::vector<RankId>>& tempVTopo,
        uint64_t inputAddr, uint64_t outputAddr, uint64_t scratchAddr, uint64_t sliceSize, uint64_t token)
    {
        u32 maxDimNum = 1;
        if (tempVTopo.size() != maxDimNum) {
            THROW<InvalidParamsException>(StringFormat(
                "[CcuInstructionReduceScatterMesh1D2Die] tempVTopo size is not 1, size is [%zu].", tempVTopo.size()));
        }

        dimSize_.push_back(tempVTopo[0].size());
        rankId_ = rankId;
        op_ = op;
        rmtReduceWithMyRank_ = rmtReduceWithMyRank;
        tempVTopo_ = tempVTopo;
        inputAddr_ = inputAddr;
        outputAddr_ = outputAddr;
        scratchAddr_ = scratchAddr;
        sliceSize_ = sliceSize;
        token_ = token;
    }

    CcuInstType GetInstType() const override
    {
        HCCL_INFO("CcuInstructionReduceScatterMesh1D2Die instype is CCU_REDUCE_SCATTER_MESH_1D_2DIE.");
        return instType_;
    }

    std::string Describe() const override
    {
        return StringFormat(
            "CcuInstructionReduceScatterMesh1D2Die rankId [%u], instType[%s]", rankId_, instType_.Describe().c_str());
    }

    void SetInstType(CcuInstType instType) { instType_ = instType; }

    std::unique_ptr<CcuCtxArg> GetCtxArg() const override
    {
        return std::make_unique<CcuCtxArgReduceScatterMesh1D2Die>(
            dimSize_, rankId_, op_, rmtReduceWithMyRank_, tempVTopo_);
    }

    std::unique_ptr<CcuTaskArg> GetTaskArg() const override
    {
        return std::make_unique<CcuTaskArgReduceScatterMesh1D2Die>(
            inputAddr_, outputAddr_, scratchAddr_, sliceSize_, token_);
    }

private:
    CcuInstType instType_ = CcuInstType::CCU_REDUCE_SCATTER_MESH_1D_2DIE;

    std::vector<uint64_t> dimSize_;
    u32 rankId_{0};
    CollAlgOperator op_;
    bool rmtReduceWithMyRank_{true};
    std::vector<std::vector<RankId>> tempVTopo_;
    uint64_t inputAddr_{0};
    uint64_t outputAddr_{0};
    uint64_t scratchAddr_{0};
    uint64_t sliceSize_{0};
    uint64_t token_{0};
};

} // namespace Hccl
#endif // HCCLV2_CCU_INSTRUCTION_REDUCE_SCATTER_MESH_1D_2DIE_H_
