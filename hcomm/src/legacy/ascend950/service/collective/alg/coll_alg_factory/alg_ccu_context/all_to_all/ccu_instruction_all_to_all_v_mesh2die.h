/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_CCU_INSTRUCTION_ALL_TO_ALL_V_MESH_2DIE_H_
#define HCCLV2_CCU_INSTRUCTION_ALL_TO_ALL_V_MESH_2DIE_H_

#include "template_utils.h"
#include "instruction.h"
#include "ins_queue.h"
#include "ccu_context_utils.h"
#include "ccu_ctx_signature.h"
#include "ccu_ins.h"
#include "ccu_rank_group.h"

namespace Hccl {

// 为AllToAllVMesh2Die实现的CCUIns、CCUCtxArg与CCUTaskArg
class CcuCtxArgAllToAllVMesh2Die : public CcuCtxArg {
public:
    CcuCtxArgAllToAllVMesh2Die(
        const std::vector<uint32_t>& dSize, uint32_t rId, bool withMyRank, const CollAlgOperator& op,
        const std::vector<std::vector<RankId>>& tempVTopo, const std::vector<RankId>& rankGroup)
        : CcuCtxArg(),
          dimSize(dSize),
          rankId(rId),
          withMyRank(withMyRank),
          op(op),
          tempVTopo(tempVTopo),
          rankGroup(rankGroup)
    {}

    ~CcuCtxArgAllToAllVMesh2Die() override {}

    CcuCtxSignature GetCtxSignature() const override
    {
        CcuCtxSignature signature;
        GenerateCcuCtxSignature(signature, CcuInstType::CCU_ALLTOALLV_MESH_2DIE_DIRECT, op, tempVTopo);
        return signature;
    }

    std::vector<uint32_t> dimSize;
    uint32_t rankId;
    bool withMyRank;
    CollAlgOperator op;
    std::vector<std::vector<RankId>> tempVTopo;
    std::vector<RankId> rankGroup;
};

class CcuTaskArgAllToAllVMesh2Die : public CcuTaskArg {
public:
    explicit CcuTaskArgAllToAllVMesh2Die(
        uint64_t inputAddr, uint64_t outputAddr, uint64_t scratchAddr, uint64_t token,
        const A2ASendRecvInfo& localSendRecvInfo)
        : inputAddr(inputAddr),
          outputAddr(outputAddr),
          scratchAddr(scratchAddr),
          token(token),
          localSendRecvInfo(localSendRecvInfo)
    {}

    uint64_t inputAddr;
    uint64_t outputAddr;
    uint64_t scratchAddr;
    uint64_t token;
    A2ASendRecvInfo localSendRecvInfo;
};

class CcuInstructionAllToAllVMesh2Die : public CcuInstruction {
public:
    CcuInstructionAllToAllVMesh2Die(
        const CollAlgOperator& op, const std::vector<uint32_t>& dimSize,
        const std::vector<std::vector<RankId>>& tempVTopo)
        : CcuInstruction(),
          op_(op),
          dimSize_(dimSize),
          tempVTopo_(tempVTopo)
    {}

    void Init(
        uint32_t rankId, bool withMyRank, uint64_t inputAddr, uint64_t outputAddr, uint64_t scratchAddr, uint64_t token,
        const A2ASendRecvInfo& localSendRecvInfo)
    {
        rankId_ = rankId;
        withMyRank_ = withMyRank;
        inputAddr_ = inputAddr;
        outputAddr_ = outputAddr;
        scratchAddr_ = scratchAddr;
        token_ = token;
        localSendRecvInfo_ = localSendRecvInfo;
        return;
    }

    CcuInstType GetInstType() const override
    {
        HCCL_INFO("CcuInstructionAllToAllVMesh2Die instype is CCU_ALLTOALLV_MESH_2DIE_DIRECT.");
        return instType_;
    }

    std::string Describe() const override
    {
        return StringFormat(
            "CcuInstructionAllToAllVMesh2Die rankId [%u], instType[%s]", rankId_, instType_.Describe().c_str());
    }

    void SetInstType(CcuInstType instType) { instType_ = instType; }

    std::unique_ptr<CcuCtxArg> GetCtxArg() const override
    {
        HCCL_INFO(
            "[CcuInstructionAllToAllVMesh2Die][GetCtxArg] dimSize.size[%u], rankId[%u], withMyRank[%u], "
            "tempVTopo.size[%u]",
            dimSize_.size(), rankId_, withMyRank_, tempVTopo_.size());
        return std::make_unique<CcuCtxArgAllToAllVMesh2Die>(
            dimSize_, rankId_, withMyRank_, op_, tempVTopo_, rankGroup_.GetRanks());
    }

    std::unique_ptr<CcuTaskArg> GetTaskArg() const override
    {
        // 2个Die依赖的TaskArg要一样，因为当前ccu_ins_group只会用第一个Instruction获取TaskArg
        return std::make_unique<CcuTaskArgAllToAllVMesh2Die>(
            inputAddr_, outputAddr_, scratchAddr_, token_, localSendRecvInfo_);
    }

private:
    CollAlgOperator op_;
    std::vector<uint32_t> dimSize_;
    std::vector<std::vector<RankId>> tempVTopo_;

    CcuInstType instType_ = CcuInstType::CCU_ALLTOALLV_MESH_2DIE_DIRECT;
    uint32_t rankId_{0};
    bool withMyRank_{false};
    uint64_t inputAddr_{0};
    uint64_t outputAddr_{0};
    uint64_t scratchAddr_{0};
    uint64_t token_{0};
    A2ASendRecvInfo localSendRecvInfo_;
};

} // namespace Hccl
#endif // HCCLV2_CCU_INSTRUCTION_ALL_TO_ALL_V_MESH_2DIE_H_
