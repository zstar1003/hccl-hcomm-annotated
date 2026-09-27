/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_CCU_INSTRUCTION_BROADCAST_NHR_1D_H_
#define HCCLV2_CCU_INSTRUCTION_BROADCAST_NHR_1D_H_

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

// 为BroadcastNHRMem2Mem1D实现的CCUIns、CCUCtxArg与CCUTaskArg
class CcuCtxArgBroadcastNHRMem2Mem1D : public CcuCtxArg {
public:
    explicit CcuCtxArgBroadcastNHRMem2Mem1D(
        const std::vector<uint64_t>& dimSize, uint32_t rankId, uint32_t axisId, uint32_t axisSize,
        const std::vector<NHRStepInfo> stepInfoVector, const std::map<u32, u32> indexMap, const CollAlgOperator& op,
        const std::vector<std::vector<RankId>>& tempVTopo)
        : dimSize_(dimSize),
          rankId_(rankId),
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
        GenerateCcuCtxSignature(signature, CcuInstType::CCU_BROADCAST_NHR_1D_MEM2MEM, op_, tempVTopo_);
        return signature;
    }
    std::vector<uint64_t> dimSize_; // 记录x轴和y轴的卡数
    uint64_t rankId_;
    uint64_t axisId_; // 记录自己在哪个轴
    uint64_t axisSize_;
    std::vector<NHRStepInfo> stepInfoVector_; // nhr每一步的信息（发送/接受给谁，发/收哪片数据）
    std::map<u32, u32> indexMap_; // 因为想要收发连续的数据，所以会把原rank映射一个虚拟的rank
    CollAlgOperator op_;
    std::vector<std::vector<RankId>> tempVTopo_;
};

class CcuTaskArgBroadcastNHRMem2Mem1D : public CcuTaskArg {
public:
    explicit CcuTaskArgBroadcastNHRMem2Mem1D(
        uint64_t inputAddr, uint64_t outputAddr, uint64_t token, uint64_t die0Size, uint64_t die1Size,
        uint64_t die0SliceSize, uint64_t die1SliceSize, uint64_t die0LastSliceSize, uint64_t die1LastSliceSize)
        : inputAddr_(inputAddr),
          outputAddr_(outputAddr),
          token_(token),
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
    uint64_t die0Size_;
    uint64_t die1Size_;
    uint64_t die0SliceSize_;
    uint64_t die1SliceSize_;
    uint64_t die0LastSliceSize_;
    uint64_t die1LastSliceSize_;
};

class CcuInstructionBroadcastNHRMem2Mem1D : public CcuInstruction {
public:
    CcuInstructionBroadcastNHRMem2Mem1D() : CcuInstruction() {}

    void Init(
        uint32_t rankId, uint64_t inputAddr, uint64_t outputAddr, uint32_t axisId, uint32_t axisSize, uint64_t die0Size,
        uint64_t die1Size, uint64_t die0SliceSize, uint64_t die1SliceSize, uint64_t die0LastSliceSize,
        uint64_t die1LastSliceSize, std::vector<NHRStepInfo> stepInfoVector, std::map<u32, u32> indexMap,
        uint64_t token, CollAlgOperator& op, std::vector<std::vector<RankId>>& tempVTopo)
    {
        u32 maxDimNum = 1;
        if (tempVTopo.size() != maxDimNum) {
            THROW<InvalidParamsException>(StringFormat(
                "[CcuInstructionBroadcastNHRMem2Mem1D] tempVTopo size is not 1, size is [%zu].", tempVTopo.size()));
        }

        dimSize_.push_back(tempVTopo[0].size());
        rankId_ = rankId;
        inputAddr_ = inputAddr;
        outputAddr_ = outputAddr;
        die0Size_ = die0Size;
        die1Size_ = die1Size;
        axisId_ = axisId;
        axisSize_ = axisSize;
        die0SliceSize_ = die0SliceSize;
        die1SliceSize_ = die1SliceSize;
        die0LastSliceSize_ = die0LastSliceSize;
        die1LastSliceSize_ = die1LastSliceSize;
        stepInfoVector_ = stepInfoVector;
        token_ = token;
        indexMap_ = indexMap;
        tempVTopo_ = tempVTopo;
        op_ = op;
        return;
    }

    CcuInstType GetInstType() const override
    {
        HCCL_INFO("CcuInstructionBroadcastNHRMem2Mem1D instype is CCU_BROADCAST_NHR_1D_MEM2MEM.");
        return instType_;
    }

    std::string Describe() const override
    {
        return StringFormat(
            "CcuInstructionBroadcastNHRMem2Mem1D rankId [%u], instType[%s]", rankId_, instType_.Describe().c_str());
    }

    void SetInstType(CcuInstType instType) { instType_ = instType; }

    std::unique_ptr<CcuCtxArg> GetCtxArg() const override
    {
        return std::make_unique<CcuCtxArgBroadcastNHRMem2Mem1D>(
            dimSize_, rankId_, axisId_, axisSize_, stepInfoVector_, indexMap_, op_, tempVTopo_);
    }

    std::unique_ptr<CcuTaskArg> GetTaskArg() const override
    {
        return std::make_unique<CcuTaskArgBroadcastNHRMem2Mem1D>(
            inputAddr_, outputAddr_, token_, die0Size_, die1Size_, die0SliceSize_, die1SliceSize_, die0LastSliceSize_,
            die1LastSliceSize_);
    }

private:
    CcuInstType instType_ = CcuInstType::CCU_BROADCAST_NHR_1D_MEM2MEM;
    std::vector<uint64_t> dimSize_;
    uint32_t rankId_{0};
    uint32_t axisId_{0};
    uint32_t axisSize_{0};
    uint64_t inputAddr_{0};
    uint64_t outputAddr_{0};
    uint64_t die0Size_{0};
    uint64_t die1Size_{0};
    uint64_t die0LastSliceSize_{0};
    uint64_t die1LastSliceSize_{0};
    uint64_t die0SliceSize_{0};
    uint64_t die1SliceSize_{0};
    uint64_t token_{0};

    std::vector<std::vector<RankId>> tempVTopo_;
    std::vector<NHRStepInfo> stepInfoVector_;
    std::map<u32, u32> indexMap_;
    CollAlgOperator op_;
};
} // namespace Hccl

#endif // HCCLV2_CCU_INSTRUCTION_BROADCAST_NHR_1D_H_
