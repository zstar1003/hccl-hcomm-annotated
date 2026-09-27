/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_CCU_INSTRUCTION_REDUCE_SCATTER_MESH_1D_MEM2MEM_H_
#define HCCLV2_CCU_INSTRUCTION_REDUCE_SCATTER_MESH_1D_MEM2MEM_H_

#include "template_utils.h"
#include "instruction.h"
#include "ins_queue.h"
#include "ccu_context_utils.h"
#include "ccu_ctx_signature.h"
#include "ccu_ins.h"
#include "ccu_rank_group.h"

namespace Hccl {

class CcuCtxArgReduceScatterMeshMem2Mem1D : public CcuCtxArg {
public:
    explicit CcuCtxArgReduceScatterMeshMem2Mem1D(
        const std::vector<uint64_t>& dimSize, uint32_t rankId, const CollAlgOperator& op,
        const std::vector<std::vector<RankId>>& tempVTopo)
        : dimSize_(dimSize),
          rankId_(rankId),
          op_(op),
          tempVTopo_(tempVTopo)
    {}
    CcuCtxSignature GetCtxSignature() const override
    {
        CcuCtxSignature signature;
        GenerateCcuCtxSignature(signature, CcuInstType::CCU_REDUCE_SCATTER_MESH_1D_MEM2MEM, op_, tempVTopo_);
        return signature;
    }
    std::vector<uint64_t> dimSize_;
    uint32_t rankId_;
    CollAlgOperator op_;
    std::vector<std::vector<RankId>> tempVTopo_;
};

class CcuTaskArgReduceScatterMeshMem2Mem1D : public CcuTaskArg {
public:
    explicit CcuTaskArgReduceScatterMeshMem2Mem1D(
        uint64_t inputAddr, uint64_t outputAddr, uint64_t token, uint64_t scratchAddr, uint64_t inputSliceStride,
        uint64_t inputRepeatStride, uint64_t outputRepeatStride, uint64_t normalSliceSize, uint64_t lastSliceSize,
        uint64_t repeatNum)
        : inputAddr_(inputAddr),
          outputAddr_(outputAddr),
          token_(token),
          scratchAddr_(scratchAddr),
          inputSliceStride_(inputSliceStride),
          inputRepeatStride_(inputRepeatStride),
          outputRepeatStride_(outputRepeatStride),
          normalSliceSize_(normalSliceSize),
          lastSliceSize_(lastSliceSize),
          repeatNum_(repeatNum)
    {}

    uint64_t inputAddr_;
    uint64_t outputAddr_;
    uint64_t token_;
    uint64_t scratchAddr_;
    uint64_t inputSliceStride_;
    uint64_t inputRepeatStride_;
    uint64_t outputRepeatStride_;
    uint64_t normalSliceSize_;
    uint64_t lastSliceSize_;
    uint64_t repeatNum_;
};

class CcuInstructionReduceScatterMeshMem2Mem1D : public CcuInstruction {
public:
    CcuInstructionReduceScatterMeshMem2Mem1D() : CcuInstruction() {}

    void Init(
        uint32_t rankId, uint64_t repeatNum, const CollAlgOperator& op,
        const std::vector<std::vector<RankId>>& tempVTopo, uint64_t inputAddr, uint64_t outputAddr, uint64_t token,
        uint64_t scratchAddr, uint64_t inputSliceStride, uint64_t inputRepeatStride, uint64_t outputRepeatStride,
        uint64_t normalSliceSize, uint64_t lastSliceSize)
    {
        dimSize_.push_back(tempVTopo[0].size());
        rankId_ = rankId;
        repeatNum_ = repeatNum;
        op_ = op;
        tempVTopo_ = tempVTopo;
        inputAddr_ = inputAddr;
        outputAddr_ = outputAddr;
        token_ = token;
        scratchAddr_ = scratchAddr;
        inputSliceStride_ = inputSliceStride;
        inputRepeatStride_ = inputRepeatStride;
        outputRepeatStride_ = outputRepeatStride;
        normalSliceSize_ = normalSliceSize;
        lastSliceSize_ = lastSliceSize;

        return;
    }

    CcuInstType GetInstType() const override
    {
        HCCL_INFO("CcuInstructionReduceScatterMeshMem2Mem1D instype is CCU_REDUCE_SCATTER_MESH_1D_MEM2MEM.");
        return instType_;
    }

    std::string Describe() const override
    {
        return StringFormat(
            "CcuInstructionReduceScatterMeshMem2Mem1D rankId [%u], instType[%s]", rankId_,
            instType_.Describe().c_str());
    }

    void SetInstType(CcuInstType instType) { instType_ = instType; }

    std::unique_ptr<CcuCtxArg> GetCtxArg() const override
    {
        return std::make_unique<CcuCtxArgReduceScatterMeshMem2Mem1D>(dimSize_, rankId_, op_, tempVTopo_);
    }

    std::unique_ptr<CcuTaskArg> GetTaskArg() const override
    {
        return std::make_unique<CcuTaskArgReduceScatterMeshMem2Mem1D>(
            inputAddr_, outputAddr_, token_, scratchAddr_, inputSliceStride_, inputRepeatStride_, outputRepeatStride_,
            normalSliceSize_, lastSliceSize_, repeatNum_);
    }

private:
    CcuInstType instType_ = CcuInstType::CCU_REDUCE_SCATTER_MESH_1D_MEM2MEM;
    std::vector<uint64_t> dimSize_;
    uint32_t rankId_{0};
    uint64_t repeatNum_{0};
    CollAlgOperator op_;
    std::vector<std::vector<RankId>> tempVTopo_;
    uint64_t inputAddr_{0};
    uint64_t outputAddr_{0};
    uint64_t scratchAddr_{0};
    uint64_t token_{0};
    uint64_t inputSliceStride_{0};
    uint64_t inputRepeatStride_{0};
    uint64_t outputRepeatStride_{0};
    uint64_t normalSliceSize_{0};
    uint64_t lastSliceSize_{0};
};
} // namespace Hccl
#endif // HCCLV2_CCU_INSTRUCTION_REDUCE_SCATTER_MESH_1D_MEM2MEM
