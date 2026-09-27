/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_CCU_INSTRUCTION_REDUCE_NHR_1D_H_
#define HCCLV2_CCU_INSTRUCTION_REDUCE_NHR_1D_H_

#include "template_utils.h"
#include "instruction.h"
#include "ins_queue.h"
#include "ccu_context_utils.h"
#include "ccu_ctx_signature.h"
#include "ccu_ins.h"
#include "ccu_rank_group.h"
#include "ccu_instruction_all_reduce_nhr1d_mem2mem.h"

namespace Hccl {

using NHRStepInfo = struct NHRStepInfoDef;

// 为ReduceNHR1D实现的CCUIns、CCUCtxArg与CCUTaskArg
class CcuCtxArgReduceNHR1D : public CcuCtxArg {
public:
    explicit CcuCtxArgReduceNHR1D(
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
        GenerateCcuCtxSignature(signature, CcuInstType::CCU_REDUCE_NHR_1D_MEM2MEM, op_, tempVTopo_);
        return signature;
    }
    std::vector<uint64_t> dimSize_;
    uint32_t rankId_;
    uint32_t rootId_;
    uint32_t axisId_;
    uint32_t axisSize_;
    std::vector<NHRStepInfo> stepInfoVector_;
    std::map<u32, u32> indexMap_;
    CollAlgOperator op_;
    std::vector<std::vector<RankId>> tempVTopo_;
};

class CcuTaskArgReduceNHR1D : public CcuTaskArg {
public:
    explicit CcuTaskArgReduceNHR1D(
        uint64_t inputAddr, uint64_t outputAddr, uint64_t token, uint64_t isInputOutputEqual, uint64_t die0Size,
        uint64_t die1Size, uint64_t die0SliceSize, uint64_t die1SliceSize, uint64_t die0LastSliceSize,
        uint64_t die1LastSliceSize)
        : inputAddr_(inputAddr),
          outputAddr_(outputAddr),
          token_(token),
          isInputOutputEqual_(isInputOutputEqual),
          die0Size_(die0Size),
          die1Size_(die1Size),
          die0SliceSize_(die0SliceSize),
          die1SliceSize_(die1SliceSize),
          die0LastSliceSize_(die0LastSliceSize),
          die1LastSliceSize_(die1LastSliceSize)
    {}

    uint64_t inputAddr_;
    uint64_t outputAddr_;
    uint64_t token_;
    uint64_t isInputOutputEqual_;
    uint64_t die0Size_;
    uint64_t die1Size_;
    uint64_t die0SliceSize_;
    uint64_t die1SliceSize_;
    uint64_t die0LastSliceSize_;
    uint64_t die1LastSliceSize_;
};

class CcuInstructionReduceNHR1D : public CcuInstruction {
public:
    CcuInstructionReduceNHR1D() : CcuInstruction() {}

    void Init(
        uint32_t rankId, uint32_t rootId, uint64_t inputAddr, uint64_t outputAddr, uint32_t axisId, uint32_t axisSize,
        uint64_t die0Size, uint64_t die1Size, uint64_t die0SliceSize, uint64_t die1SliceSize,
        uint64_t die0LastSliceSize, uint64_t die1LastSliceSize, std::vector<NHRStepInfo> stepInfoVector,
        std::map<u32, u32> indexMap, uint64_t token, uint64_t isInputOutputEqual, CollAlgOperator& op,
        std::vector<std::vector<RankId>>& tempVTopo)
    {
        u32 maxDimNum = 1;
        if (tempVTopo.size() != maxDimNum) {
            THROW<InvalidParamsException>(
                StringFormat("[CcuInstructionReduceNHR1D] tempVTopo size is not 1, size is [%zu].", tempVTopo.size()));
        }
        dimSize_.push_back(tempVTopo[0].size());
        rankId_ = rankId;
        rootId_ = rootId;
        inputAddr_ = inputAddr;
        outputAddr_ = outputAddr;
        axisId_ = axisId;
        axisSize_ = axisSize;
        die0Size_ = die0Size;
        die1Size_ = die1Size;
        die0SliceSize_ = die0SliceSize;
        die1SliceSize_ = die1SliceSize;
        die0LastSliceSize_ = die0LastSliceSize;
        die1LastSliceSize_ = die1LastSliceSize;
        stepInfoVector_ = stepInfoVector;
        indexMap_ = indexMap;
        token_ = token;
        isInputOutputEqual_ = isInputOutputEqual;
        op_ = op;
        tempVTopo_ = tempVTopo;
        return;
    }

    CcuInstType GetInstType() const override
    {
        HCCL_INFO("CcuInstructionReduceNHR1D instype is CCU_REDUCE_NHR_1D_MEM2MEM.");
        return instType_;
    }

    std::string Describe() const override
    {
        return StringFormat(
            "CcuInstructionReduceNHR1D rankId [%u], instType[%s]", rankId_, instType_.Describe().c_str());
    }

    void SetInstType(CcuInstType instType) { instType_ = instType; }

    std::unique_ptr<CcuCtxArg> GetCtxArg() const override
    {
        return std::make_unique<CcuCtxArgReduceNHR1D>(
            dimSize_, rankId_, rootId_, axisId_, axisSize_, stepInfoVector_, indexMap_, op_, tempVTopo_);
    }

    std::unique_ptr<CcuTaskArg> GetTaskArg() const override
    {
        return std::make_unique<CcuTaskArgReduceNHR1D>(
            inputAddr_, outputAddr_, token_, isInputOutputEqual_, die0Size_, die1Size_, die0SliceSize_, die1SliceSize_,
            die0LastSliceSize_, die1LastSliceSize_);
    }

private:
    CcuInstType instType_ = CcuInstType::CCU_REDUCE_NHR_1D_MEM2MEM;
    std::vector<uint64_t> dimSize_;
    uint32_t rankId_{0};
    uint32_t rootId_{0};
    uint32_t axisId_{0};
    uint32_t axisSize_{0};
    uint64_t inputAddr_{0};
    uint64_t outputAddr_{0};
    uint64_t die0Size_{0};
    uint64_t die1Size_{0};
    uint64_t die0SliceSize_{0};
    uint64_t die1SliceSize_{0};
    uint64_t die0LastSliceSize_{0};
    uint64_t die1LastSliceSize_{0};
    uint64_t isInputOutputEqual_{0};
    CollAlgOperator op_;
    std::vector<NHRStepInfo> stepInfoVector_;
    std::map<u32, u32> indexMap_;
    uint64_t token_{0};
    std::vector<std::vector<RankId>> tempVTopo_;
};
} // namespace Hccl

#endif // HCCLV2_CCU_INSTRUCTION_ALL_REDUCE_NHR_1D_H_
