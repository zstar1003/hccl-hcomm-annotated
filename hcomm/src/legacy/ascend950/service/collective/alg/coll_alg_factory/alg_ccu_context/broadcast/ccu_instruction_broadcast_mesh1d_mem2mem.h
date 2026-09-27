/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_CCU_INSTRUCTION_BROADCAST_MESH_1D_MEM2MEM_H_
#define HCCLV2_CCU_INSTRUCTION_BROADCAST_MESH_1D_MEM2MEM_H_
#include "template_utils.h"
#include "instruction.h"
#include "ins_queue.h"
#include "ccu_context_utils.h"
#include "ccu_ctx_signature.h"
#include "ccu_ins.h"
#include "ccu_rank_group.h"
namespace Hccl {
// 为BroadcastMesh1D实现的CCUIns、CCUCtxArg与CCUTaskArg
class CcuCtxArgBroadcastMesh1DMem2Mem : public CcuCtxArg {
public:
    explicit CcuCtxArgBroadcastMesh1DMem2Mem(
        const std::vector<uint64_t>& dimSize, uint64_t rankId, uint64_t rootId, const CollAlgOperator& op,
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
        GenerateCcuCtxSignature(signature, CcuInstType::CCU_BROADCAST_MESH_1D_MEM2MEM, op_, tempVTopo_);
        return signature;
    }
    std::vector<uint64_t> dimSize_;
    uint32_t rankId_;
    uint32_t rootId_;
    CollAlgOperator op_;
    std::vector<std::vector<RankId>> tempVTopo_;
};
class CcuTaskArgBroadcastMesh1DMem2Mem : public CcuTaskArg {
public:
    explicit CcuTaskArgBroadcastMesh1DMem2Mem(
        uint64_t inputAddr, uint64_t outputAddr, uint64_t token, uint64_t inputSliceStride, uint64_t outputSliceStride,
        uint64_t repeatNum, uint64_t inputRepeatStride, uint64_t outputRepeatStride, uint64_t normalSliceSize,
        uint64_t lastSliceSize, uint64_t allgatherOffset, uint64_t repeatNumVar)
        : inputAddr_(inputAddr),
          outputAddr_(outputAddr),
          token_(token),
          inputSliceStride_(inputSliceStride),
          outputSliceStride_(outputSliceStride),
          repeatNum_(repeatNum),
          inputRepeatStride_(inputRepeatStride),
          outputRepeatStride_(outputRepeatStride),
          normalSliceSize_(normalSliceSize),
          lastSliceSize_(lastSliceSize),
          allgatherOffset_(allgatherOffset),
          repeatNumVar_(repeatNumVar)
    {}
    uint64_t inputAddr_;
    uint64_t outputAddr_;
    uint64_t token_;
    uint64_t inputSliceStride_;
    uint64_t outputSliceStride_;
    uint64_t repeatNum_;
    uint64_t inputRepeatStride_;
    uint64_t outputRepeatStride_;
    uint64_t normalSliceSize_;
    uint64_t lastSliceSize_;
    uint64_t allgatherOffset_;
    uint64_t repeatNumVar_;
};

class CcuInstructionBroadcastMesh1DMem2Mem : public CcuInstruction {
public:
    CcuInstructionBroadcastMesh1DMem2Mem() : CcuInstruction() {}

    void Init(
        uint32_t rankId, uint32_t rootId, const CollAlgOperator& op, const std::vector<std::vector<RankId>>& tempVTopo,
        uint64_t inputAddr, uint64_t outputAddr, uint64_t token, uint64_t inputSliceStride, uint64_t outputSliceStride,
        uint32_t repeatNum, uint64_t inputRepeatStride, uint64_t outputRepeatStride, uint64_t normalSliceSize,
        uint64_t lastSliceSize, uint64_t allgatherOffset, uint64_t repeatNumVar)
    {
        u32 maxDimNum = 1;
        if (tempVTopo.size() != maxDimNum) {
            THROW<InvalidParamsException>(StringFormat(
                "[CcuInstructionBroadcastMesh1DMem2Mem] tempVTopo size is not 1, size is [%zu].", tempVTopo.size()));
        }
        dimSize_.push_back(tempVTopo[0].size());
        rankId_ = rankId;
        rootId_ = rootId;
        op_ = op;
        tempVTopo_ = tempVTopo;
        inputAddr_ = inputAddr;
        outputAddr_ = outputAddr;
        token_ = token;
        inputSliceStride_ = inputSliceStride;
        outputSliceStride_ = outputSliceStride;
        repeatNum_ = repeatNum;
        inputRepeatStride_ = inputRepeatStride;
        outputRepeatStride_ = outputRepeatStride;
        normalSliceSize_ = normalSliceSize;
        lastSliceSize_ = lastSliceSize;
        allgatherOffset_ = allgatherOffset;
        repeatNumVar_ = repeatNumVar;
        return;
    }

    CcuInstType GetInstType() const override
    {
        HCCL_INFO("CcuInstructionBroadcastMesh1DMem2Mem instype is CCU_BROADCAST_MESH_1D_MEM2MEM.");
        return instType_;
    }

    std::string Describe() const override
    {
        return StringFormat(
            "CcuInstructionBroadcastMesh1DMem2Mem rankId [%u], instType[%s]", rankId_, instType_.Describe().c_str());
    }

    std::unique_ptr<CcuCtxArg> GetCtxArg() const override
    {
        return std::make_unique<CcuCtxArgBroadcastMesh1DMem2Mem>(dimSize_, rankId_, rootId_, op_, tempVTopo_);
    }

    void SetInstType(CcuInstType instType) { instType_ = instType; }

    std::unique_ptr<CcuTaskArg> GetTaskArg() const override
    {
        return std::make_unique<CcuTaskArgBroadcastMesh1DMem2Mem>(
            inputAddr_, outputAddr_, token_, inputSliceStride_, outputSliceStride_, repeatNum_, inputRepeatStride_,
            outputRepeatStride_, normalSliceSize_, lastSliceSize_, allgatherOffset_, repeatNumVar_);
    }

private:
    CcuInstType instType_ = CcuInstType::CCU_BROADCAST_MESH_1D_MEM2MEM;
    std::vector<uint64_t> dimSize_;
    uint32_t rankId_{0};
    uint32_t rootId_{0};
    CollAlgOperator op_;
    std::vector<std::vector<RankId>> tempVTopo_;
    uint64_t inputAddr_{0};
    uint64_t outputAddr_{0};
    uint64_t token_{0};
    uint64_t inputSliceStride_{0};
    uint64_t outputSliceStride_{0};
    uint32_t repeatNum_{0};
    uint64_t inputRepeatStride_{0};
    uint64_t outputRepeatStride_{0};
    uint64_t allgatherOffset_{0};
    uint64_t normalSliceSize_{0};
    uint64_t lastSliceSize_{0};
    uint64_t repeatNumVar_{0};
};
} // namespace Hccl
#endif // HCCLV2_CCU_INSTRUCTION_BROADCAST_MESH_1D_MEM2MEM_WITH_STRIDE_H_
