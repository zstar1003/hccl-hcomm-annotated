/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_CCU_INSTRUCTION_REDUCE_MESH_1D_TWO_SHOT_MEM2MEM_H_
#define HCCLV2_CCU_INSTRUCTION_REDUCE_MESH_1D_TWO_SHOT_MEM2MEM_H_

#include <memory>
#include <map>
#include <vector>
#include <queue>
#include <string>

#include <sstream>
#include <ios>
#include <iostream>

#include "template_utils.h"
#include "instruction.h"
#include "ins_queue.h"
#include "ccu_context_utils.h"
#include "ccu_ctx_signature.h"
#include "ccu_ins.h"
#include "ccu_rank_group.h"

namespace Hccl {

// 为ReduceMeshTwoShot1D实现的CCUIns、CCUCtxArg与CCUTaskArg
class CcuCtxArgReduceMeshTwoShotMem2Mem1D : public CcuCtxArg {
public:
    explicit CcuCtxArgReduceMeshTwoShotMem2Mem1D(
        const std::vector<uint64_t>& dimSize, uint32_t rankId, uint32_t rootId, const CollAlgOperator& op,
        const std::vector<std::vector<RankId>>& tempVTopo)
        : dimSize_(dimSize),
          rankId_(rankId),
          rootId_(rootId),
          op_(op),
          tempVTopo_(tempVTopo)
    {}
    CcuCtxSignature GetCtxSignature() const override
    {
        CcuCtxSignature signature;
        GenerateCcuCtxSignature(signature, CcuInstType::CCU_REDUCE_MESH_1D_TWO_SHOT_MEM2MEM, op_, tempVTopo_);
        return signature;
    }
    std::vector<uint64_t> dimSize_;
    uint32_t rankId_;
    uint32_t rootId_;
    CollAlgOperator op_;
    std::vector<std::vector<RankId>> tempVTopo_;
};

class CcuTaskArgReduceMeshTwoShotMem2Mem1D : public CcuTaskArg {
public:
    explicit CcuTaskArgReduceMeshTwoShotMem2Mem1D(
        uint64_t inputAddr, uint64_t outputAddr, uint64_t token, uint64_t scratchAddr, uint64_t normalSliceSize,
        uint64_t lastSliceSize, uint64_t mySliceSize)
        : inputAddr_(inputAddr),
          outputAddr_(outputAddr),
          token_(token),
          scratchAddr_(scratchAddr),
          normalSliceSize_(normalSliceSize),
          lastSliceSize_(lastSliceSize),
          mySliceSize_(mySliceSize)
    {}
    uint64_t inputAddr_;
    uint64_t outputAddr_;
    uint64_t token_;
    uint64_t scratchAddr_;
    uint64_t normalSliceSize_;
    uint64_t lastSliceSize_;
    uint64_t mySliceSize_;
};

class CcuInstructionReduceMeshTwoShotMem2Mem1D : public CcuInstruction {
public:
    CcuInstructionReduceMeshTwoShotMem2Mem1D() : CcuInstruction() {}

    void Init(
        uint32_t rankId, uint32_t rootId, const CollAlgOperator& op, const std::vector<std::vector<RankId>>& tempVTopo,
        uint64_t inputAddr, uint64_t outputAddr, uint64_t scratchAddr, uint64_t token, uint64_t normalSliceSize,
        uint64_t lastSliceSize, uint64_t mySliceSize)
    {
        dimSize_.push_back(tempVTopo[0].size());
        rankId_ = rankId;
        rootId_ = rootId;
        op_ = op;
        tempVTopo_ = tempVTopo;
        inputAddr_ = inputAddr;
        outputAddr_ = outputAddr;
        scratchAddr_ = scratchAddr;
        token_ = token;
        normalSliceSize_ = normalSliceSize;
        lastSliceSize_ = lastSliceSize;
        mySliceSize_ = mySliceSize;

        return;
    }

    std::string Describe() const override
    {
        return StringFormat(
            "CcuInstructionReduceMeshTwoShot1D rankId [%u], instType[%s]", rankId_, instType_.Describe().c_str());
    }

    CcuInstType GetInstType() const override { return instType_; }

    void SetInstType(CcuInstType instType) { instType_ = instType; }

    std::unique_ptr<CcuCtxArg> GetCtxArg() const override
    {
        return std::make_unique<CcuCtxArgReduceMeshTwoShotMem2Mem1D>(dimSize_, rankId_, rootId_, op_, tempVTopo_);
    }

    std::unique_ptr<CcuTaskArg> GetTaskArg() const override
    {
        return std::make_unique<CcuTaskArgReduceMeshTwoShotMem2Mem1D>(
            inputAddr_, outputAddr_, token_, scratchAddr_, normalSliceSize_, lastSliceSize_, mySliceSize_);
    }

    std::vector<LinkData> GetLinks() const override { return links_; }

    void SetLinks(std::vector<LinkData>& links) override { links_ = links; }

    RankGroup GetRankGroup() const override { return rankGroup_; }

    void SetRankGroup(RankGroup& rankGroup) override { rankGroup_ = rankGroup; }

private:
    CcuInstType instType_ = CcuInstType::CCU_REDUCE_MESH_1D_TWO_SHOT_MEM2MEM;
    std::vector<uint64_t> dimSize_;
    uint32_t rankId_{0};
    uint32_t rootId_{0};
    CollAlgOperator op_;
    std::vector<std::vector<RankId>> tempVTopo_;
    uint64_t inputAddr_{0};
    uint64_t outputAddr_{0};
    uint64_t token_{0};
    uint64_t scratchAddr_{0};
    uint64_t normalSliceSize_{0};
    uint64_t lastSliceSize_{0};
    uint64_t mySliceSize_{0};
    RankGroup rankGroup_;
    std::vector<LinkData> links_;
};

} // namespace Hccl
#endif // HCCLV2_CCU_INSTRUCTION_REDUCE_MESH_1D_TWO_SHOT_MEM2MEM_H_
