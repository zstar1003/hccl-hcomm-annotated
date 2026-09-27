/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_CCU_INSTRUCTION_HALF_ALL_TO_ALL_V_MESH_1D_H_
#define HCCLV2_CCU_INSTRUCTION_HALF_ALL_TO_ALL_V_MESH_1D_H_

#include "template_utils.h"
#include "instruction.h"
#include "ins_queue.h"
#include "ccu_context_utils.h"
#include "ccu_ctx_signature.h"
#include "ccu_ins.h"
#include "ccu_rank_group.h"

namespace Hccl {

// 为AllToAllVMesh1D实现的CCUIns、CCUCtxArg与CCUTaskArg
class CcuCtxArgHalfAllToAllVMesh1D : public CcuCtxArg {
public:
    explicit CcuCtxArgHalfAllToAllVMesh1D(
        const std::vector<uint64_t>& dSize, uint32_t rId, const CollAlgOperator& op, uint32_t mId,
        uint64_t cclBufferAddr, const std::vector<std::vector<RankId>>& tempVTopo)
        : dimSize(dSize),
          rankId(rId),
          op(op),
          missionId(mId),
          cclBufferAddr(cclBufferAddr),
          tempVTopo(tempVTopo)
    {}
    CcuCtxSignature GetCtxSignature() const override
    {
        CcuCtxSignature signature;
        GenerateCcuCtxSignature(signature, CcuInstType::CCU_HALF_ALLTOALLV_MESH_1D, op, tempVTopo);
        return signature;
    }
    std::vector<uint64_t> dimSize;
    uint32_t rankId;
    CollAlgOperator op;
    uint32_t missionId;
    uint64_t cclBufferAddr;
    std::vector<std::vector<RankId>> tempVTopo;
};

class CcuTaskArgHalfAllToAllVMesh1D : public CcuTaskArg {
public:
    explicit CcuTaskArgHalfAllToAllVMesh1D() {}
};

class CcuInstructionHalfAllToAllVMesh1D : public CcuInstruction {
public:
    CcuInstructionHalfAllToAllVMesh1D() : CcuInstruction() {}

    void Init(
        uint32_t mId, uint32_t rankId, uint64_t scratchAddr, CollAlgOperator& op,
        std::vector<std::vector<RankId>>& tempVTopo)
    {
        u32 maxDimNum = 1;
        if (tempVTopo.size() != maxDimNum) {
            THROW<InvalidParamsException>(StringFormat(
                "[CcuInstructionHalfAllToAllVMesh1D] tempVTopo size is not 1, size is [%zu].", tempVTopo.size()));
        }
        dimSize_.push_back(tempVTopo[0].size());
        missionId_ = mId;
        rankId_ = rankId;
        cclBufferAddr_ = scratchAddr;
        op_ = op;
        tempVTopo_ = tempVTopo;
        return;
    }

    CcuInstType GetInstType() const override
    {
        HCCL_INFO("CcuInstructionHalfAllToAllVMesh1D instype is CCU_HALF_ALLTOALLV_MESH_1D.");
        return instType_;
    }

    std::string Describe() const override
    {
        return StringFormat(
            "CcuInstructionHalfAllToAllVMesh1D rankId [%u], instType[%s]", rankId_, instType_.Describe().c_str());
    }

    void SetInstType(CcuInstType instType) { instType_ = instType; }

    std::unique_ptr<CcuCtxArg> GetCtxArg() const override
    {
        return std::make_unique<CcuCtxArgHalfAllToAllVMesh1D>(
            dimSize_, rankId_, op_, missionId_, cclBufferAddr_, tempVTopo_);
    }

    std::unique_ptr<CcuTaskArg> GetTaskArg() const override
    {
        return std::make_unique<CcuTaskArgHalfAllToAllVMesh1D>();
    }

private:
    CcuInstType instType_ = CcuInstType::CCU_HALF_ALLTOALLV_MESH_1D;
    std::vector<uint64_t> dimSize_;
    uint32_t rankId_{0};
    uint32_t missionId_{0};
    uint64_t cclBufferAddr_{0};
    CollAlgOperator op_;
    std::vector<std::vector<RankId>> tempVTopo_;
};

} // namespace Hccl
#endif // HCCLV2_CCU_INSTRUCTION_HALF_ALL_TO_ALL_V_MESH_1D_H_
