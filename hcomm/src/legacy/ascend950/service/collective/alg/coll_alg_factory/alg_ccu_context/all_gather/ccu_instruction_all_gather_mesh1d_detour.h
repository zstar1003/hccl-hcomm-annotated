/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_CCU_INSTRUCTION_ALL_GATHER_MESH_1D_DETOUR_H_
#define HCCLV2_CCU_INSTRUCTION_ALL_GATHER_MESH_1D_DETOUR_H_

#include "template_utils.h"
#include "instruction.h"
#include "ins_queue.h"
#include "ccu_context_utils.h"
#include "ccu_ctx_signature.h"
#include "ccu_ins.h"
#include "ccu_rank_group.h"

namespace Hccl {

class CcuCtxArgAllGatherMeshDetour1D : public CcuCtxArg {
public:
    explicit CcuCtxArgAllGatherMeshDetour1D(
        const std::vector<uint64_t>& dSize, uint32_t rId, const CollAlgOperator& op,
        const std::vector<std::vector<RankId>>& tempVTopo, uint64_t singleTransportSize, uint64_t detourPathNum,
        uint64_t pathNumPerPeer)
        : dimSize_(dSize),
          rankId_(rId),
          op_(op),
          tempVTopo_(tempVTopo),
          singleTransportSize_(singleTransportSize),
          detourPathNum_(detourPathNum),
          pathNumPerPeer_(pathNumPerPeer)
    {}

    CcuCtxSignature GetCtxSignature() const override
    {
        CcuCtxSignature signature;
        GenerateCcuCtxSignature(signature, CcuInstType::CCU_ALLGATHER_MESH_1D_DETOUR, op_, tempVTopo_);
        return signature;
    }

    std::vector<uint64_t> dimSize_;
    uint32_t rankId_;
    CollAlgOperator op_;
    std::vector<std::vector<RankId>> tempVTopo_;
    uint64_t singleTransportSize_;
    uint64_t detourPathNum_;
    uint64_t pathNumPerPeer_;
};

class CcuTaskArgAllGatherMeshDetour1D : public CcuTaskArg {
public:
    explicit CcuTaskArgAllGatherMeshDetour1D(
        uint64_t inputAddr, uint64_t outputAddr, uint64_t token, uint64_t baseOffset, uint64_t tailOffset,
        uint64_t tailSize, uint64_t loopIterNum, const std::vector<uint64_t>& lengths)
        : inputAddr_(inputAddr),
          outputAddr_(outputAddr),
          token_(token),
          baseOffset_(baseOffset),
          tailOffset_(tailOffset),
          tailSize_(tailSize),
          loopIterNum_(loopIterNum),
          lengths_(lengths)
    {}

    uint64_t inputAddr_;
    uint64_t outputAddr_;
    uint64_t token_;
    uint64_t baseOffset_;
    uint64_t tailOffset_;
    uint64_t tailSize_; // 尾块数据量
    uint64_t loopIterNum_;
    std::vector<uint64_t> lengths_; // 每个loop迭代一次搬运时每个ms上的数据量
};

class CcuInstructionAllGatherMeshDetour1D : public CcuInstruction {
public:
    CcuInstructionAllGatherMeshDetour1D() : CcuInstruction() {}

    void InitDetourInfo(
        uint32_t rankId, uint64_t inputAddr, uint64_t outputAddr, uint64_t token, uint64_t baseOffset,
        uint64_t tailOffset, uint64_t tailSize, uint64_t loopIterNum, std::vector<uint64_t>& lengths,
        uint64_t singleTransportSize, uint64_t detourPathNum, uint64_t pathNumPerPeer, CollAlgOperator& op,
        std::vector<std::vector<RankId>>& tempVTopo)
    {
        u32 maxDimNum = 1;
        if (tempVTopo.size() != maxDimNum) {
            THROW<InvalidParamsException>(StringFormat(
                "[CcuInstructionAllGatherMeshDetour1D] tempVTopo size is not 1, size is [%zu].", tempVTopo.size()));
        }
        dimSize_.push_back(tempVTopo[0].size());
        rankId_ = rankId;
        inputAddr_ = inputAddr;
        outputAddr_ = outputAddr;
        token_ = token;
        baseOffset_ = baseOffset;
        tailOffset_ = tailOffset;
        tailSize_ = tailSize;
        loopIterNum_ = loopIterNum;
        lengths_ = lengths;
        singleTransportSize_ = singleTransportSize;
        detourPathNum_ = detourPathNum;
        pathNumPerPeer_ = pathNumPerPeer;
        op_ = op;
        tempVTopo_ = tempVTopo;
        return;
    }

    CcuInstType GetInstType() const override
    {
        HCCL_INFO("CcuInstructionAllGatherMeshDetour1D instype is CCU_ALLGATHER_MESH_1D_DETOUR.");
        return instType_;
    }

    std::string Describe() const override
    {
        return StringFormat(
            "CcuInstructionAllGatherMeshDetour1D rankId [%u], instType[%s]", rankId_, instType_.Describe().c_str());
    }

    std::unique_ptr<CcuCtxArg> GetCtxArg() const override
    {
        return std::make_unique<CcuCtxArgAllGatherMeshDetour1D>(
            dimSize_, rankId_, op_, tempVTopo_, singleTransportSize_, detourPathNum_, pathNumPerPeer_);
    }

    std::unique_ptr<CcuTaskArg> GetTaskArg() const override
    {
        return std::make_unique<CcuTaskArgAllGatherMeshDetour1D>(
            inputAddr_, outputAddr_, token_, baseOffset_, tailOffset_, tailSize_, loopIterNum_, lengths_);
    }

    void SetInstType(CcuInstType instType) { instType_ = instType; }

private:
    CcuInstType instType_ = CcuInstType::CCU_ALLGATHER_MESH_1D_DETOUR;
    std::vector<uint64_t> dimSize_;
    uint32_t rankId_{0};
    uint64_t inputAddr_{0};
    uint64_t outputAddr_{0};
    uint64_t token_{0};
    uint64_t baseOffset_{0};
    uint64_t tailOffset_{0};
    uint64_t tailSize_{0};
    uint64_t loopIterNum_{0};
    std::vector<uint64_t> lengths_;

    uint64_t singleTransportSize_{0};
    uint64_t detourPathNum_{0}; // 到每个对端有几个绕路路径
    uint64_t pathNumPerPeer_{0};
    CollAlgOperator op_;
    std::vector<std::vector<RankId>> tempVTopo_;
};

} // namespace Hccl
#endif // HCCLV2_CCU_INSTRUCTION_ALL_GATHER_MESH_1D_H_
