/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_CCU_INSTRUCTION_ALL_TO_ALL_MESH_1D_H_
#define HCCLV2_CCU_INSTRUCTION_ALL_TO_ALL_MESH_1D_H_

#include "template_utils.h"
#include "instruction.h"
#include "ins_queue.h"
#include "ccu_context_utils.h"
#include "ccu_ctx_signature.h"
#include "ccu_ins.h"
#include "ccu_rank_group.h"

namespace Hccl {

// 为AllToAllMesh1D实现的CCUIns、CCUCtxArg与CCUTaskArg
class CcuCtxArgAllToAllMesh1D : public CcuCtxArg {
public:
    explicit CcuCtxArgAllToAllMesh1D(
        const std::vector<uint64_t>& dSize, uint32_t rId, const CollAlgOperator& op,
        const std::vector<std::vector<RankId>>& tempVTopo, bool loadFromMem = false)
        : dimSize(dSize),
          rankId(rId),
          op(op),
          tempVTopo(tempVTopo),
          loadFromMem(loadFromMem)
    {}
    CcuCtxSignature GetCtxSignature() const override
    {
        CcuCtxSignature signature;
        GenerateCcuCtxSignature(signature, CcuInstType::CCU_ALLTOALL_MESH_1D_DIRECT, op, tempVTopo);
        return signature;
    }
    std::vector<uint64_t> dimSize;
    uint32_t rankId;
    CollAlgOperator op;
    std::vector<std::vector<RankId>> tempVTopo;
    bool loadFromMem;
};

class CcuTaskArgAllToAllMesh1D : public CcuTaskArg {
public:
    explicit CcuTaskArgAllToAllMesh1D(
        uint64_t inputAddr, uint64_t outputAddr, uint64_t sliceSize, uint64_t token, uint64_t srcOffset,
        uint64_t dstOffset, uint64_t srcStride)
        : inputAddr(inputAddr),
          outputAddr(outputAddr),
          sliceSize(sliceSize),
          token(token),
          srcOffset(srcOffset),
          dstOffset(dstOffset),
          srcStride(srcStride)
    {}

    uint64_t inputAddr;
    uint64_t outputAddr;
    uint64_t sliceSize;
    uint64_t token;
    uint64_t srcOffset;
    uint64_t dstOffset;
    uint64_t srcStride;
};

class CcuInstructionAllToAllMesh1D : public CcuInstruction {
public:
    CcuInstructionAllToAllMesh1D() : CcuInstruction() {}

    void Init(
        uint32_t rankId, uint64_t inputAddr, uint64_t outputAddr, uint64_t sliceSize, uint64_t token,
        uint64_t srcOffset, uint64_t dstOffset, uint64_t srcStride, CollAlgOperator& op,
        std::vector<std::vector<RankId>>& tempVTopo, bool loadFromMem = false)
    {
        u32 maxDimNum = 1;
        if (tempVTopo.size() != maxDimNum) {
            THROW<InvalidParamsException>(StringFormat(
                "[CcuInstructionAllToAllMesh1D] tempVTopo size is not 1, size is [%zu].", tempVTopo.size()));
        }
        dimSize_.push_back(tempVTopo[0].size());
        rankId_ = rankId;
        inputAddr_ = inputAddr;
        outputAddr_ = outputAddr;
        sliceSize_ = sliceSize;
        token_ = token;
        srcOffset_ = srcOffset;
        dstOffset_ = dstOffset;
        srcStride_ = srcStride;
        op_ = op;
        tempVTopo_ = tempVTopo;
        loadFromMem_ = loadFromMem;
        return;
    }

    CcuInstType GetInstType() const override
    {
        HCCL_INFO("CcuInstructionAllToAllMesh1D instype is CCU_ALLTOALL_MESH_1D_DIRECT.");
        return instType_;
    }

    std::string Describe() const override
    {
        return StringFormat(
            "CcuInstructionAllToAllMesh1D rankId [%u], instType[%s]", rankId_, instType_.Describe().c_str());
    }

    std::unique_ptr<CcuCtxArg> GetCtxArg() const override
    {
        return std::make_unique<CcuCtxArgAllToAllMesh1D>(dimSize_, rankId_, op_, tempVTopo_, loadFromMem_);
    }

    void SetInstType(CcuInstType instType) { instType_ = instType; }

    std::unique_ptr<CcuTaskArg> GetTaskArg() const override
    {
        return std::make_unique<CcuTaskArgAllToAllMesh1D>(
            inputAddr_, outputAddr_, sliceSize_, token_, srcOffset_, dstOffset_, srcStride_);
    }

private:
    CcuInstType instType_ = CcuInstType::CCU_ALLTOALL_MESH_1D_DIRECT;
    std::vector<uint64_t> dimSize_;
    uint32_t rankId_{0};
    uint64_t inputAddr_{0};
    uint64_t outputAddr_{0};
    uint64_t sliceSize_{0};
    uint64_t token_{0};
    uint64_t srcOffset_{0};
    uint64_t dstOffset_{0};
    uint64_t srcStride_{0};
    CollAlgOperator op_;
    std::vector<std::vector<RankId>> tempVTopo_;
    bool loadFromMem_{false};
};

} // namespace Hccl
#endif // HCCLV2_CCU_INSTRUCTION_ALL_ALL_TO_ALL_MESH_1D_H
