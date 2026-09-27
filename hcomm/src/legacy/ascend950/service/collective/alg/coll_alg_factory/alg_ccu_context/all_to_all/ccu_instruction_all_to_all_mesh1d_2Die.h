/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_CCU_INSTRUCTION_ALL_TO_ALL_MESH_1D_2DIE_H_
#define HCCLV2_CCU_INSTRUCTION_ALL_TO_ALL_MESH_1D_2DIE_H_

#include "template_utils.h"
#include "instruction.h"
#include "ins_queue.h"
#include "ccu_context_utils.h"
#include "ccu_ctx_signature.h"
#include "ccu_ins.h"
#include "ccu_rank_group.h"

namespace Hccl {

class CcuCtxArgAllToAllMesh1D2Die : public CcuCtxArg {
public:
    explicit CcuCtxArgAllToAllMesh1D2Die(
        const std::vector<uint64_t>& dimSize, uint32_t rankId, const CollAlgOperator& op,
        const std::vector<std::vector<RankId>>& tempVTopo, bool withMyRank, const std::vector<RankId>& rankGroup,
        u32 bitNum)
        : dimSize_(dimSize),
          rankId_(rankId),
          op_(op),
          tempVTopo_(tempVTopo),
          withMyRank_(withMyRank),
          rankGroup(rankGroup),
          bitNum_(bitNum)
    {}
    CcuCtxSignature GetCtxSignature() const override
    {
        CcuCtxSignature signature;
        GenerateCcuCtxSignature(signature, CcuInstType::CCU_ALLTOALL_MESH_1D_2DIE, op_, tempVTopo_);
        return signature;
    }
    std::vector<uint64_t> dimSize_;
    uint32_t rankId_;
    CollAlgOperator op_;
    std::vector<std::vector<RankId>> tempVTopo_;
    bool withMyRank_;
    std::vector<RankId> rankGroup;
    u32 bitNum_;
};

class CcuTaskArgAllToAllMesh1D2Die : public CcuTaskArg {
public:
    explicit CcuTaskArgAllToAllMesh1D2Die(
        uint64_t inputAddr, uint64_t outputAddr, uint64_t sliceSize, uint64_t token, uint64_t inputSliceStride,
        uint64_t outputSliceStride, uint64_t outBuffBaseOff)
        : inputAddr_(inputAddr),
          outputAddr_(outputAddr),
          sliceSize_(sliceSize),
          token_(token),
          inputSliceStride_(inputSliceStride),
          outputSliceStride_(outputSliceStride),
          outBuffBaseOff_(outBuffBaseOff)
    {}

    uint64_t inputAddr_;
    uint64_t outputAddr_;
    uint64_t sliceSize_;
    uint64_t token_;
    uint64_t inputSliceStride_;
    uint64_t outputSliceStride_;
    uint64_t outBuffBaseOff_;
};

class CcuInstructionAllToAllMesh1D2Die : public CcuInstruction {
public:
    CcuInstructionAllToAllMesh1D2Die() : CcuInstruction() {}

    void Init(
        uint32_t rankId, uint64_t inputAddr, uint64_t outputAddr, uint64_t sliceSize, uint64_t token,
        uint64_t inputSliceStride, uint64_t outputSliceStride, uint64_t outBuffBaseOff, CollAlgOperator& op,
        std::vector<std::vector<RankId>>& tempVTopo, bool withMyRank, u32 bitNum)
    {
        u32 maxDimNum = 1;
        if (tempVTopo.size() != maxDimNum) {
            THROW<InvalidParamsException>(StringFormat(
                "[CcuInstructionAllToAllMesh1D2Die] tempVTopo size is not 1, size is [%zu].", tempVTopo.size()));
        }
        dimSize_.push_back(tempVTopo[0].size());
        rankId_ = rankId;
        inputAddr_ = inputAddr;
        outputAddr_ = outputAddr;
        sliceSize_ = sliceSize;
        token_ = token;
        inputSliceStride_ = inputSliceStride;
        outputSliceStride_ = outputSliceStride;
        outBuffBaseOff_ = outBuffBaseOff;
        op_ = op;
        tempVTopo_ = tempVTopo;
        withMyRank_ = withMyRank;
        bitNum_ = bitNum;
        return;
    }

    CcuInstType GetInstType() const override
    {
        HCCL_INFO("CcuInstructionAllToAllMesh1D2Die instype is CCU_ALLTOALL_MESH_1D_2DIE.");
        return instType_;
    }

    std::string Describe() const override
    {
        return StringFormat(
            "CcuInstructionAllToAllMesh1D2Die rankId [%u], instType[%s]", rankId_, instType_.Describe().c_str());
    }

    std::unique_ptr<CcuCtxArg> GetCtxArg() const override
    {
        return std::make_unique<CcuCtxArgAllToAllMesh1D2Die>(
            dimSize_, rankId_, op_, tempVTopo_, withMyRank_, rankGroup_.GetRanks(), bitNum_);
    }

    void SetInstType(CcuInstType instType) { instType_ = instType; }

    std::unique_ptr<CcuTaskArg> GetTaskArg() const override
    {
        return std::make_unique<CcuTaskArgAllToAllMesh1D2Die>(
            inputAddr_, outputAddr_, sliceSize_, token_, inputSliceStride_, outputSliceStride_, outBuffBaseOff_);
    }

private:
    CcuInstType instType_ = CcuInstType::CCU_ALLTOALL_MESH_1D_2DIE;
    std::vector<uint64_t> dimSize_;
    uint32_t rankId_{0};
    uint64_t inputAddr_{0};
    uint64_t outputAddr_{0};
    uint64_t sliceSize_{0};
    uint64_t token_{0};
    uint64_t inputSliceStride_{0};
    uint64_t outputSliceStride_{0};
    uint64_t outBuffBaseOff_{0};
    CollAlgOperator op_;
    std::vector<std::vector<RankId>> tempVTopo_;
    bool withMyRank_{false};
    u32 bitNum_{0};
};

} // namespace Hccl
#endif // HCCLV2_CCU_INSTRUCTION_ALL_ALL_TO_ALL_MESH_1D_2Die_H
