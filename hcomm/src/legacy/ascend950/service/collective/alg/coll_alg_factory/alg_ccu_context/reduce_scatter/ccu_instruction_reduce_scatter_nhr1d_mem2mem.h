/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_CCU_INSTRUCTION_REDUCE_SCATTER_NHR_1D_H_
#define HCCLV2_CCU_INSTRUCTION_REDUCE_SCATTER_NHR_1D_H_

#include "template_utils.h"
#include "instruction.h"
#include "ins_queue.h"
#include "ccu_context_utils.h"
#include "ccu_ctx_signature.h"
#include "ccu_ins.h"
#include "ccu_rank_group.h"

namespace Hccl {

using NHRStepInfo = struct NHRStepInfo {
    u32 step = 0;
    u32 myRank = 0;
    u32 nSlices;
    u32 toRank = 0;
    u32 fromRank = 0;
    std::vector<u32> txSliceIdxs;
    std::vector<u32> rxSliceIdxs;

    NHRStepInfo() : nSlices(0) {}
};

// 为ReduceScatterNHR1D实现的CCUIns、CCuCtxArg与CCUTaskArg
class CcuCtxArgReduceScatterNHR1D : public CcuCtxArg {
public:
    explicit CcuCtxArgReduceScatterNHR1D(
        const std::vector<uint64_t>& dimSize, uint32_t rankId, uint32_t axisId,
        const std::vector<NHRStepInfo> stepInfoVector, const std::map<u32, u32> indexMap, const CollAlgOperator& op,
        const std::vector<std::vector<RankId>>& tempVTopo, uint32_t linkNum)
        : dimSize_(dimSize),
          rankId_(rankId),
          axisId_(axisId),
          stepInfoVector_(stepInfoVector),
          indexMap_(indexMap),
          op_(op),
          tempVTopo_(tempVTopo),
          linkNum_(linkNum)
    {}
    CcuCtxSignature GetCtxSignature() const override
    {
        CcuCtxSignature signature;
        GenerateCcuCtxSignature(signature, CcuInstType::CCU_REDUCE_SCATTER_NHR_1D_MEM2MEM, op_, tempVTopo_);
        // signature.Append<uint8_t>(uint8_t(op.outputDataType));
        return signature;
    }
    std::vector<uint64_t> dimSize_; // 记录x轴和y轴的卡数
    uint64_t rankId_;
    uint64_t axisId_;                         // 记录自己在哪个轴
    std::vector<NHRStepInfo> stepInfoVector_; // nhr每一步的信息（发送/接受给谁，发/收哪片数据）
    std::map<u32, u32> indexMap_; // 因为想要收发连续的数据，所以会把原rank映射一个虚拟的rank
    CollAlgOperator op_;
    std::vector<std::vector<RankId>> tempVTopo_;
    uint32_t linkNum_; // 因为当前不支持双die，用此变量确认单die还是双die
};

class CcuTaskArgReduceScatterNHR1D : public CcuTaskArg {
public:
    explicit CcuTaskArgReduceScatterNHR1D(
        uint64_t inputAddr, uint64_t outputAddr, uint64_t token, uint64_t die0Size, uint64_t die1Size,
        uint64_t inputSliceStride, uint64_t outputSliceStride, uint64_t inputRepeatStride, uint64_t outputRepeatStride,
        uint64_t repeatNum, uint64_t isBottom)
        : inputAddr_(inputAddr),
          outputAddr_(outputAddr),
          token_(token),
          die0Size_(die0Size),
          die1Size_(die1Size),
          inputSliceStride_(inputSliceStride),
          outputSliceStride_(outputSliceStride),
          inputRepeatStride_(inputRepeatStride),
          outputRepeatStride_(outputRepeatStride),
          repeatNum_(repeatNum),
          isBottom_(isBottom)
    {}

    uint64_t inputAddr_;
    uint64_t outputAddr_;
    uint64_t token_;
    uint64_t die0Size_;
    uint64_t die1Size_;
    uint64_t inputSliceStride_;
    uint64_t outputSliceStride_;
    uint64_t inputRepeatStride_;
    uint64_t outputRepeatStride_;
    uint64_t repeatNum_;
    uint64_t isBottom_;
};

class CcuInstructionReduceScatterNHR1D : public CcuInstruction {
public:
    CcuInstructionReduceScatterNHR1D() : CcuInstruction() {}

    void Init(
        uint32_t rankId, uint64_t inputAddr, uint64_t outputAddr, uint64_t axisId, uint64_t die0Size, uint64_t die1Size,
        uint64_t repeatNum, uint64_t inputSliceStride, uint64_t outputSliceStride, uint64_t inputRepeatStride,
        uint64_t outputRepeatStride, std::vector<NHRStepInfo> stepInfoVector, std::map<u32, u32> indexMap,
        uint64_t token, CollAlgOperator& op, std::vector<std::vector<RankId>>& tempVTopo, u32 linkNum, bool isBottom)
    {
        u32 maxDimNum = 1;
        if (tempVTopo.size() != maxDimNum) {
            THROW<InvalidParamsException>(StringFormat(
                "[CcuInstructionReduceScatterNHR1D] tempVTopo size is not 1, size is [%zu].", tempVTopo.size()));
        }
        dimSize_.push_back(tempVTopo[0].size());
        rankId_ = rankId;
        inputAddr_ = inputAddr;
        outputAddr_ = outputAddr;
        axisId_ = axisId;
        die0Size_ = die0Size;
        die1Size_ = die1Size;
        repeatNum_ = repeatNum;
        inputSliceStride_ = inputSliceStride;
        outputSliceStride_ = outputSliceStride;
        inputRepeatStride_ = inputRepeatStride;
        outputRepeatStride_ = outputRepeatStride;
        stepInfoVector_ = stepInfoVector;
        indexMap_ = indexMap;
        token_ = token;
        op_ = op;
        tempVTopo_ = tempVTopo;
        linkNum_ = linkNum;
        isBottom_ = isBottom;
        return;
    }

    std::string Describe() const override
    {
        return StringFormat(
            "CcuInstructionReduceScatterNHR1D rankId [%u], instType[%s]", rankId_, instType_.Describe().c_str());
    }

    CcuInstType GetInstType() const override
    {
        HCCL_INFO("CcuInstructionReduceScatterNHR1D instype is CCU_REDUCE_SCATTER_NHR_1D_MEM2MEM.");
        return instType_;
    }

    void SetInstType(CcuInstType instType) { instType_ = instType; }

    std::unique_ptr<CcuCtxArg> GetCtxArg() const override
    {
        return std::make_unique<CcuCtxArgReduceScatterNHR1D>(
            dimSize_, rankId_, axisId_, stepInfoVector_, indexMap_, op_, tempVTopo_, linkNum_);
    }

    std::unique_ptr<CcuTaskArg> GetTaskArg() const override
    {
        return std::make_unique<CcuTaskArgReduceScatterNHR1D>(
            inputAddr_, outputAddr_, token_, die0Size_, die1Size_, inputSliceStride_, outputSliceStride_,
            inputRepeatStride_, outputRepeatStride_, repeatNum_, isBottom_);
    }

private:
    CcuInstType instType_ = CcuInstType::CCU_REDUCE_SCATTER_NHR_1D_MEM2MEM;
    std::vector<uint64_t> dimSize_;
    uint32_t rankId_{0};
    uint32_t axisId_{0};
    uint64_t inputAddr_{0};
    uint64_t outputAddr_{0};
    uint64_t die0Size_{0};
    uint64_t die1Size_{0};
    uint64_t repeatNum_{0};
    uint64_t inputSliceStride_{0};
    uint64_t outputSliceStride_{0};
    uint64_t inputRepeatStride_{0};
    uint64_t outputRepeatStride_{0};
    uint64_t isBottom_{0};
    std::vector<NHRStepInfo> stepInfoVector_;
    std::map<u32, u32> indexMap_;
    uint64_t token_{0};
    CollAlgOperator op_;
    std::vector<std::vector<RankId>> tempVTopo_;
    uint32_t linkNum_{0};
};
} // namespace Hccl

#endif // HCCLV2_CCU_INSTRUCTION_REDUCE_SCATTER_NHR_1D_H_
