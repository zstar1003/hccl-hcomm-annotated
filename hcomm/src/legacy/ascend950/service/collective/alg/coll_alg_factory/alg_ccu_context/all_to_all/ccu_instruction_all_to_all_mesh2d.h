/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_CCU_INSTRUCTION_ALL_TO_ALL_MESH_2D_H
#define HCCLV2_CCU_INSTRUCTION_ALL_TO_ALL_MESH_2D_H

#include "template_utils.h"
#include "instruction.h"
#include "ins_queue.h"
#include "ccu_context_utils.h"
#include "ccu_ctx_signature.h"
#include "ccu_ins.h"
#include "ccu_rank_group.h"

namespace Hccl {

// 为AllGatherMesh1D实现的CCUIns、CCUCtxArg与CCUTaskArg
class CcuCtxArgAlltoAllMesh2D : public CcuCtxArg {
public:
    CcuCtxArgAlltoAllMesh2D(
        const std::vector<uint32_t>& dSize, uint32_t rId, uint32_t aId, const CollAlgOperator& op,
        const std::vector<std::vector<RankId>>& tempVTopo)
        : CcuCtxArg(),
          dimSize(dSize),
          rankId(rId),
          axisId(aId),
          op(op),
          tempVTopo(tempVTopo)
    {}

    ~CcuCtxArgAlltoAllMesh2D() override {}

    CcuCtxSignature GetCtxSignature() const override
    {
        CcuCtxSignature signature;
        GenerateCcuCtxSignature(signature, CcuInstType::CCU_ALLTOALL_MESH_2D_DIRECT, op, tempVTopo);
        HCCL_INFO("[CcuCtxArgAlltoAllMesh2D][GetCtxSignature] signature[%s]", signature.GetData().c_str());
        return signature;
    }

    // 需要存储，传递给算法
    std::vector<uint32_t> dimSize;
    uint32_t rankId;
    uint32_t axisId;

    const CollAlgOperator& op;
    const std::vector<std::vector<RankId>>& tempVTopo;
};

class CcuTaskArgAlltoAllMesh2D : public CcuTaskArg {
public:
    explicit CcuTaskArgAlltoAllMesh2D(
        uint64_t inputAddr, uint64_t outputAddr, uint64_t scratchAddr, uint64_t sendStride, uint64_t recvStride,
        uint64_t sendLength, uint64_t aSize, uint64_t bSize, uint64_t baseOffset, uint64_t token)
        : CcuTaskArg(),
          inputAddr(inputAddr),
          outputAddr(outputAddr),
          scratchAddr(scratchAddr),
          sendStride(sendStride),
          recvStride(recvStride),
          sendLength(sendLength),
          aSize(aSize),
          bSize(bSize),
          baseOffset(baseOffset),
          token(token)
    {}

    uint64_t inputAddr;
    uint64_t outputAddr;
    uint64_t scratchAddr;
    uint64_t sendStride;
    uint64_t recvStride;
    uint64_t sendLength;
    uint64_t aSize; // X方向第一轮传输的数据量
    uint64_t bSize;
    uint64_t baseOffset; // 多轮执行时的基础偏移，等于step*(aSize+bSize)
    uint64_t token;
};

class CcuInstructionAlltoAllMesh2D : public CcuInstruction {
public:
    CcuInstructionAlltoAllMesh2D(
        const CollAlgOperator& op, const std::vector<uint32_t>& dimSize,
        const std::vector<std::vector<RankId>>& tempVTopo)
        : CcuInstruction(),
          op_(op),
          dimSize_(dimSize),
          tempVTopo_(tempVTopo)
    {}

    void Init(
        uint32_t rankId, uint64_t inputAddr, uint64_t outputAddr, uint64_t scratchAddr, uint64_t axisId,
        uint64_t sendStride, uint64_t recvStride, uint64_t sendLength, uint64_t aSize, uint64_t bSize,
        uint64_t baseOffset, uint64_t token)
    {
        rankId_ = rankId;
        inputAddr_ = inputAddr;
        outputAddr_ = outputAddr;
        scratchAddr_ = scratchAddr;
        axisId_ = axisId;
        sendStride_ = sendStride;
        recvStride_ = recvStride;
        sendLength_ = sendLength;
        aSize_ = aSize;
        bSize_ = bSize;
        baseOffset_ = baseOffset;
        token_ = token;
        HCCL_INFO(
            "[CcuInstructionAlltoAllMesh2D][Init] rankId[%u] inputAddr[%llu] outputAddr[%llu] scratchAddr[%llu],\
axisId[%u], sendStride[%llu], recvStride[%llu], sendLength[%llu], aSize[%llu], bSize[%llu], baseOffset[%llu], \
dimSize.size[%u], tempVTopo.size[%u]",
            rankId_, inputAddr_, outputAddr_, scratchAddr_, axisId_, sendStride_, recvStride_, sendLength_, aSize_,
            bSize_, baseOffset_, dimSize_.size(), tempVTopo_.size());
        return;
    }

    CcuInstType GetInstType() const override
    {
        HCCL_INFO("CcuInstructionAllGatherMesh1D instype is CCU_ALLTOALL_MESH_2D_DIRECT.");
        return instType_;
    }

    std::string Describe() const override
    {
        return StringFormat("[CcuInstructionAllGatherMesh1D]RankId[%u] Ins[%s]", rankId_, instType_.Describe().c_str());
    }

    std::unique_ptr<CcuCtxArg> GetCtxArg() const override
    {
        HCCL_INFO(
            "[CcuInstructionAlltoAllMesh2D][GetCtxArg] dimSize.size[%u], rankId[%u], axisId[%u], tempVTopo.size[%u]",
            dimSize_.size(), rankId_, axisId_, tempVTopo_.size());
        return std::make_unique<CcuCtxArgAlltoAllMesh2D>(dimSize_, rankId_, axisId_, op_, tempVTopo_);
    }

    void SetInstType(CcuInstType instType) { instType_ = instType; }

    std::unique_ptr<CcuTaskArg> GetTaskArg() const override
    {
        return std::make_unique<CcuTaskArgAlltoAllMesh2D>(
            inputAddr_, outputAddr_, scratchAddr_, sendStride_, recvStride_, sendLength_, aSize_, bSize_, baseOffset_,
            token_);
    }

private:
    CollAlgOperator op_;
    std::vector<uint32_t> dimSize_;
    std::vector<std::vector<RankId>> tempVTopo_;

    CcuInstType instType_ = CcuInstType::CCU_ALLTOALL_MESH_2D_DIRECT;

    uint32_t rankId_{0};
    uint64_t inputAddr_{0};
    uint64_t outputAddr_{0};
    uint64_t scratchAddr_{0};
    uint64_t sendStride_{0};
    uint64_t recvStride_{0};
    uint64_t axisId_{0};
    uint64_t sendLength_{0}; // 多轮时的单个数据块总大小
    uint64_t aSize_{0};
    uint64_t bSize_{0};
    uint64_t baseOffset_{0};
    uint64_t token_{0};
};

} // namespace Hccl
#endif // HCCLV2_CCU_INSTRUCTION_ALL_TO_ALL_MESH_2D_H
