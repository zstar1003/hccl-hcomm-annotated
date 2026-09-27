/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_CCU_INSTRUCTION_REDUCE_TAIL_BLOCK_H_
#define HCCLV2_CCU_INSTRUCTION_REDUCE_TAIL_BLOCK_H_

#include "template_utils.h"
#include "instruction.h"
#include "ins_queue.h"
#include "ccu_context_utils.h"
#include "ccu_ctx_signature.h"
#include "ccu_ins.h"
#include "ccu_rank_group.h"

namespace Hccl {
class CcuCtxArgReduceTailBlock : public CcuCtxArg {
public:
    explicit CcuCtxArgReduceTailBlock(
        const std::vector<uint64_t>& dSize, uint32_t rId, const std::string& notifySignal, const CollAlgOperator& op,
        const std::vector<std::vector<RankId>>& tempVTopo)
        : dimSize_(dSize),
          rankId_(rId),
          notifySignal_(notifySignal),
          op_(op),
          tempVTopo_(tempVTopo)
    {}
    CcuCtxSignature GetCtxSignature() const override
    {
        CcuCtxSignature signature;
        GenerateCcuCtxSignature(signature, CcuInstType::CCU_REDUCE_TAILBLOCK_DIRECT, op_, tempVTopo_);
        return signature;
    }

    std::vector<uint64_t> dimSize_;
    uint32_t rankId_;
    std::string notifySignal_;
    CollAlgOperator op_;
    std::vector<std::vector<RankId>> tempVTopo_;
};

class CcuTaskArgReduceTailBlock : public CcuTaskArg {
public:
    explicit CcuTaskArgReduceTailBlock() {}
};

class CcuInstructionReduceTailBlock : public CcuInstruction {
public:
    CcuInstructionReduceTailBlock() : CcuInstruction() {}

    void Init(
        uint32_t rankId, const std::string& notifySignal, CollAlgOperator& op,
        std::vector<std::vector<RankId>>& tempVTopo)
    {
        u32 maxDimNum = 1;
        if (tempVTopo.size() != maxDimNum) {
            THROW<InvalidParamsException>(StringFormat(
                "[CcuInstructionReduceTailBlock] tempVTopo size is not 1, size is [%zu].", tempVTopo.size()));
        }
        dimSize_.push_back(tempVTopo[0].size());
        rankId_ = rankId;
        op_ = op;
        tempVTopo_ = tempVTopo;
        notifySignal_ = notifySignal;
        return;
    }

    CcuInstType GetInstType() const override
    {
        HCCL_INFO("CcuInstructionReduceTailBlock instype is CCU_REDUCE_TAILBLOCK_DIRECT.");
        return instType_;
    }

    std::string Describe() const override { return StringFormat("CcuInstructionReduceTailBlock"); }

    void SetInstType(CcuInstType instType) { instType_ = instType; }

    std::unique_ptr<CcuCtxArg> GetCtxArg() const override
    {
        return std::make_unique<CcuCtxArgReduceTailBlock>(dimSize_, rankId_, notifySignal_, op_, tempVTopo_);
    }

    std::unique_ptr<CcuTaskArg> GetTaskArg() const override { return std::make_unique<CcuTaskArgReduceTailBlock>(); }

private:
    CcuInstType instType_ = CcuInstType::CCU_REDUCE_TAILBLOCK_DIRECT;
    std::string notifySignal_ = "Reduce_defalut";
    std::vector<uint64_t> dimSize_;
    uint32_t rankId_{0};
    CollAlgOperator op_;
    std::vector<std::vector<RankId>> tempVTopo_;
};

} // namespace Hccl
#endif // HCCLV2_CCU_INSTRUCTION_REDUCE_TAIL_BLOCK_H_
