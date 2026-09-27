/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_CCU_INSTRUCTION_SCATTER_NHR_1D_H_
#define HCCLV2_CCU_INSTRUCTION_SCATTER_NHR_1D_H_

#include "template_utils.h"
#include "instruction.h"
#include "ins_queue.h"
#include "ccu_context_utils.h"
#include "ccu_ctx_signature.h"
#include "ccu_ins.h"
#include "ccu_rank_group.h"

namespace Hccl {
using NHRStepInfo = struct NHRStepInfoDef {
    u32 step = 0;
    u32 myRank = 0;
    u32 nSlices;
    u32 toRank = 0;
    u32 fromRank = 0;
    std::vector<u32> txSliceIdxs;
    std::vector<u32> rxSliceIdxs;
    NHRStepInfoDef() : nSlices(0) {}
};

// 为ScatterNHR1D实现的CCUIns、CCUCtxArg与CCUTaskArg
class CcuCtxArgScatterNHR1D : public CcuCtxArg {
public:
    explicit CcuCtxArgScatterNHR1D(
        const std::vector<uint64_t>& dimSize, uint32_t rankId, uint32_t rootId, uint32_t axisId, uint32_t axisSize,
        const std::vector<NHRStepInfo> stepInfoVector, const std::map<u32, u32> indexMap, const CollAlgOperator& op,
        const std::vector<std::vector<RankId>>& tempVTopo)
        : dimSize_(dimSize),
          rankId_(rankId),
          rootId_(rootId),
          axisId_(axisId),
          axisSize_(axisSize),
          stepInfoVector_(stepInfoVector),
          indexMap_(indexMap),
          op_(op),
          tempVTopo_(tempVTopo)
    {}
    CcuCtxSignature GetCtxSignature() const override
    {
        CcuCtxSignature signature;
        GenerateCcuCtxSignature(signature, CcuInstType::CCU_SCATTER_NHR_1D_MEM2MEM, op_, tempVTopo_);
        return signature;
    }
    std::vector<uint64_t> dimSize_;
    uint64_t rankId_;
    uint64_t rootId_;
    uint64_t axisId_;
    uint64_t axisSize_;
    std::vector<NHRStepInfo> stepInfoVector_;
    std::map<u32, u32> indexMap_;
    CollAlgOperator op_;
    std::vector<std::vector<RankId>> tempVTopo_;
};

class CcuTaskArgScatterNHR1D : public CcuTaskArg {
public:
    explicit CcuTaskArgScatterNHR1D(
        uint64_t inputAddr, uint64_t outputAddr, uint64_t scratchAddr, uint64_t token, uint64_t sliceSize,
        uint64_t die0Size, uint64_t die1Size, uint64_t inputSliceStride, uint64_t outputSliceStride,
        uint64_t inputRepeatStride, uint64_t outputRepeatStride, uint64_t repeatNum, uint64_t repeatNumVar,
        uint64_t isOutputScratch)
        : inputAddr_(inputAddr),
          outputAddr_(outputAddr),
          scratchAddr_(scratchAddr),
          token_(token),
          sliceSize_(sliceSize),
          die0Size_(die0Size),
          die1Size_(die1Size),
          inputSliceStride_(inputSliceStride),
          outputSliceStride_(outputSliceStride),
          inputRepeatStride_(inputRepeatStride),
          outputRepeatStride_(outputRepeatStride),
          repeatNum_(repeatNum),
          repeatNumVar_(repeatNumVar),
          isOutputScratch_(isOutputScratch)
    {}

    uint64_t inputAddr_;
    uint64_t outputAddr_;
    uint64_t scratchAddr_;
    uint64_t token_;
    uint64_t sliceSize_;
    uint64_t die0Size_;
    uint64_t die1Size_;
    uint64_t inputSliceStride_;
    uint64_t outputSliceStride_;
    uint64_t inputRepeatStride_;
    uint64_t outputRepeatStride_;
    uint64_t repeatNum_;
    uint64_t repeatNumVar_;
    uint64_t isOutputScratch_;
};

class CcuInstructionScatterNHR1D : public CcuInstruction {
public:
    CcuInstructionScatterNHR1D() : CcuInstruction() {}

    void Init(
        uint32_t rankId, uint32_t rootId, uint64_t inputAddr, uint64_t outputAddr, uint64_t scratchAddr,
        uint64_t axisId, uint64_t axisSize, uint64_t sliceSize, uint64_t die0Size, uint64_t die1Size,
        uint64_t inputSliceStride, uint64_t outputSliceStride, uint64_t inputRepeatStride, uint64_t outputRepeatStride,
        uint32_t repeatNum, uint64_t repeatNumVar, std::vector<NHRStepInfo> stepInfoVector, std::map<u32, u32> indexMap,
        uint64_t token, CollAlgOperator& op, std::vector<std::vector<RankId>>& tempVTopo, uint64_t isOutputScratch)
    {
        u32 maxDimNum = 1;
        if (tempVTopo.size() != maxDimNum) {
            THROW<InvalidParamsException>(
                StringFormat("[CcuInstructionScatterNHR1D] tempVTopo size is not 1, size is [%zu].", tempVTopo.size()));
        }
        dimSize_.push_back(tempVTopo[0].size());
        rankId_ = rankId;
        rootId_ = rootId;
        token_ = token;
        inputAddr_ = inputAddr;
        outputAddr_ = outputAddr;
        scratchAddr_ = scratchAddr;
        axisId_ = axisId;
        axisSize_ = axisSize;
        sliceSize_ = sliceSize;
        die0Size_ = die0Size;
        die1Size_ = die1Size;
        inputSliceStride_ = inputSliceStride;
        outputSliceStride_ = outputSliceStride;
        inputRepeatStride_ = inputRepeatStride;
        outputRepeatStride_ = outputRepeatStride;
        stepInfoVector_ = stepInfoVector;
        indexMap_ = indexMap;
        token_ = token;
        op_ = op;
        tempVTopo_ = tempVTopo;
        repeatNum_ = repeatNum;
        repeatNumVar_ = repeatNumVar;
        isOutputScratch_ = isOutputScratch;
        return;
    }

    CcuInstType GetInstType() const override
    {
        HCCL_INFO("CcuInstructionScatterNHR1D instype is CCU_SCATTER_NHR_1D_MEM2MEM.");
        return instType_;
    }

    std::string Describe() const override
    {
        return StringFormat(
            "CcuInstructionScatterNHR1D rankId [%u], instType[%s]", rankId_, instType_.Describe().c_str());
    }

    void SetInstType(CcuInstType instType) { instType_ = instType; }

    std::unique_ptr<CcuCtxArg> GetCtxArg() const override
    {
        return std::make_unique<CcuCtxArgScatterNHR1D>(
            dimSize_, rankId_, rootId_, axisId_, axisSize_, stepInfoVector_, indexMap_, op_, tempVTopo_);
    }

    std::unique_ptr<CcuTaskArg> GetTaskArg() const override
    {
        return std::make_unique<CcuTaskArgScatterNHR1D>(
            inputAddr_, outputAddr_, scratchAddr_, token_, sliceSize_, die0Size_, die1Size_, inputSliceStride_,
            outputSliceStride_, inputRepeatStride_, outputRepeatStride_, repeatNum_, repeatNumVar_, isOutputScratch_);
    }

private:
    CcuInstType instType_ = CcuInstType::CCU_SCATTER_NHR_1D_MEM2MEM;
    std::vector<uint64_t> dimSize_;
    CollAlgOperator op_;
    uint32_t rankId_{0};
    uint32_t rootId_{0};
    uint32_t axisId_{0};
    uint32_t axisSize_{0};
    uint32_t sliceSize_{0};
    uint64_t inputAddr_{0};
    uint64_t outputAddr_{0};
    uint64_t scratchAddr_{0};
    uint64_t die0Size_{0};
    uint64_t die1Size_{0};
    uint64_t inputSliceStride_{0};
    uint64_t outputSliceStride_{0};
    uint64_t inputRepeatStride_{0};
    uint64_t outputRepeatStride_{0};
    uint32_t repeatNum_{0};
    uint64_t repeatNumVar_{0};
    uint64_t isOutputScratch_{0};
    std::vector<NHRStepInfo> stepInfoVector_;
    std::map<u32, u32> indexMap_;
    uint64_t token_{0};
    std::vector<std::vector<RankId>> tempVTopo_;
};
} // namespace Hccl

#endif // HCCLV2_CCU_INSTRUCTION_SCATTER_NHR_1D_H_
