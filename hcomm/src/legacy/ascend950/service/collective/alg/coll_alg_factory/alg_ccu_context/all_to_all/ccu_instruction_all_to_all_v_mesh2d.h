/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_CCU_INSTRUCTION_ALL_TO_ALL_V_MESH_2D_H_
#define HCCLV2_CCU_INSTRUCTION_ALL_TO_ALL_V_MESH_2D_H_

#include "template_utils.h"
#include "instruction.h"
#include "ins_queue.h"
#include "ccu_context_utils.h"
#include "ccu_ctx_signature.h"
#include "ccu_ins.h"
#include "ccu_rank_group.h"

namespace Hccl {

// 为AllToAllVMesh1D实现的CCUIns、CCUCtxArg与CCUTaskArg
class CcuCtxArgAllToAllVMesh2D : public CcuCtxArg {
public:
    CcuCtxArgAllToAllVMesh2D(
        const std::vector<uint32_t>& dSize, uint32_t rId, uint64_t aId, const CollAlgOperator& op,
        const std::vector<std::vector<RankId>>& tempVTopo)
        : CcuCtxArg(),
          dimSize(dSize),
          rankId(rId),
          axisId(aId),
          op(op),
          tempVTopo(tempVTopo)
    {}

    ~CcuCtxArgAllToAllVMesh2D() override {}

    CcuCtxSignature GetCtxSignature() const override
    {
        CcuCtxSignature signature;
        GenerateCcuCtxSignature(signature, CcuInstType::CCU_ALLTOALLV_MESH_2D_DIRECT, op, tempVTopo);
        return signature;
    }
    std::vector<uint32_t> dimSize;
    uint32_t rankId;
    uint64_t axisId;
    CollAlgOperator op;
    std::vector<std::vector<RankId>> tempVTopo;
};

class CcuTaskArgAllToAllVMesh2D : public CcuTaskArg {
public:
    explicit CcuTaskArgAllToAllVMesh2D(
        uint64_t inputAddr, uint64_t outputAddr, uint64_t scratchAddr, uint64_t token, uint64_t scratchSliceSize,
        uint64_t scratchSliceBias, const A2ASendRecvInfo& localSendRecvInfo)
        : inputAddr(inputAddr),
          outputAddr(outputAddr),
          scratchAddr(scratchAddr),
          token(token),
          scratchSliceSize(scratchSliceSize),
          scratchSliceBias(scratchSliceBias),
          localSendRecvInfo(localSendRecvInfo)
    {}

    uint64_t inputAddr;
    uint64_t outputAddr;
    uint64_t scratchAddr;
    uint64_t token;
    uint64_t scratchSliceSize;
    uint64_t scratchSliceBias;
    A2ASendRecvInfo localSendRecvInfo;
};

class CcuInstructionAllToAllVMesh2D : public CcuInstruction {
public:
    CcuInstructionAllToAllVMesh2D(
        const CollAlgOperator& op, const std::vector<uint32_t>& dimSize,
        const std::vector<std::vector<RankId>>& tempVTopo)
        : CcuInstruction(),
          op_(op),
          dimSize_(dimSize),
          tempVTopo_(tempVTopo)
    {}

    void Init(
        uint32_t rankId, uint64_t axisId, uint64_t inputAddr, uint64_t outputAddr, uint64_t scratchAddr, uint64_t token,
        uint64_t scratchSliceSize, uint64_t scratchSliceBias, const A2ASendRecvInfo& localSendRecvInfo)
    {
        axisId_ = axisId;
        rankId_ = rankId;
        inputAddr_ = inputAddr;
        outputAddr_ = outputAddr;
        scratchAddr_ = scratchAddr;
        token_ = token;
        scratchSliceSize_ = scratchSliceSize;
        scratchSliceBias_ = scratchSliceBias;
        localSendRecvInfo_ = localSendRecvInfo;
        return;
    }

    CcuInstType GetInstType() const override
    {
        HCCL_INFO("CcuInstructionAllToAllVMesh2D instype is CCU_ALLTOALLV_MESH_2D_DIRECT.");
        return instType_;
    }

    std::string Describe() const override
    {
        return StringFormat(
            "CcuInstructionAllToAllVMesh2D rankId [%u], instType[%s]", rankId_, instType_.Describe().c_str());
    }

    void SetInstType(CcuInstType instType) { instType_ = instType; }

    std::unique_ptr<CcuCtxArg> GetCtxArg() const override
    {
        HCCL_INFO(
            "[CcuInstructionAlltoAllVMesh2D][GetCtxArg] dimSize.size[%u], rankId[%u], axisId[%llu], tempVTopo.size[%u]",
            dimSize_.size(), rankId_, axisId_, tempVTopo_.size());
        return std::make_unique<CcuCtxArgAllToAllVMesh2D>(dimSize_, rankId_, axisId_, op_, tempVTopo_);
    }

    std::unique_ptr<CcuTaskArg> GetTaskArg() const override
    {
        return std::make_unique<CcuTaskArgAllToAllVMesh2D>(
            inputAddr_, outputAddr_, scratchAddr_, token_, scratchSliceSize_, scratchSliceBias_, localSendRecvInfo_);
    }

private:
    CollAlgOperator op_;
    std::vector<uint32_t> dimSize_;
    std::vector<std::vector<RankId>> tempVTopo_;

    CcuInstType instType_ = CcuInstType::CCU_ALLTOALLV_MESH_2D_DIRECT;
    uint32_t rankId_{0};
    uint64_t axisId_{0};
    uint64_t inputAddr_{0};
    uint64_t outputAddr_{0};
    uint64_t scratchAddr_{0};
    uint64_t token_{0};
    uint64_t scratchSliceSize_{0};
    uint64_t scratchSliceBias_{0};
    A2ASendRecvInfo localSendRecvInfo_;
};

} // namespace Hccl
#endif // HCCLV2_CCU_INSTRUCTION_ALL_TO_ALL_V_MESH_2D_H_
