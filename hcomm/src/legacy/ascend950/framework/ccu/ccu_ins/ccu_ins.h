/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CCU_INSTRUCTION_H
#define CCU_INSTRUCTION_H

#include <memory>
#include "instruction.h"
#include "ccu_ctx_arg.h"
#include "ccu_task_arg.h"
#include "ccu_rank_group.h"
#include "ccu_ctx_signature.h"
#include "ccu_task_param.h"
#include "internal_exception.h"

namespace Hccl {
MAKE_ENUM(
    CcuInstType, CCU_INS_GROUP, CCU_ALLTOALL_MESH_2D_DIRECT, CCU_ALLGATHER_MESH_1D_DIRECT, CCU_ALLGATHER_MESH_2D_DIRECT,
    CCU_REDUCE_SCATTER_MESH_1D_DIRECT, CCU_ALL_REDUCE_MESH_1D_DIRECT, CCU_ALLTOALL_MESH_1D_DIRECT,
    CCU_REDUCE_MESH_1D_DIRECT, CCU_ALLTOALLV_MESH_1D_DIRECT, CCU_SCATTER_MESH_1D_DIRECT, CCU_BROADCAST_MESH_1D_DIRECT,
    CCU_REDUCE_SCATTER_MESH_2D_DIRECT, CCU_REDUCE_SCATTER_MESH_2D_MULTI_MISSION, CCU_BROADCAST_MESH_2D_DIRECT,
    CCU_ALL_REDUCE_MESH_2D_ONE_SHOT_DIRECT, CCU_ALL_REDUCE_MESH_2D_TWO_SHOT_DIRECT, CCU_SCATTER_MESH_2D_DIRECT,
    CCU_REDUCE_MESH_2D_DIRECT, CCU_ALLTOALLV_MESH_2D_DIRECT, CCU_ALLGATHER_MESH_1D_DETOUR,
    CCU_ALL_REDUCE_MESH_1D_DETOUR, CCU_REDUCE_SCATTER_MESH_1D_DETOUR, CCU_BROADCAST_MESH_1D_MEM2MEM,
    CCU_BROADCAST_MESH_1D_MULTIMISSION, CCU_REDUCE_MESH_1D_MULTI_MISSION, CCU_ALL_REDUCE_MESH_1D_ONE_SHOT_DIRECT,
    CCU_REDUCE_TAILBLOCK_DIRECT, CCU_ALL_REDUCE_MESH_1D_MEM2MEM, CCU_ALL_REDUCE_MESH_1D_MULTI_MISSION,
    CCU_ALLGATHER_MESH_1D_MULTI_MISSION, CCU_ALLGATHER_MESH_1D_MEM2MEM, CCU_HALF_ALLTOALLV_MESH_1D,
    CCU_REDUCE_SCATTER_MESH_1D_MULTI_MISSION, CCU_REDUCE_SCATTER_MESH_1D_MEM2MEM, CCU_REDUCE_SCATTER_NHR_1D_MEM2MEM,
    CCU_ALLGATHER_MESH_1D_MEM2MEM_WITH_STRIDE_DIRECT, CCU_ALLGATHER_NHR_1D_MEM2MEM, CCU_ALLGATHER_MESH_2D_MULTI_MISSION,
    CCU_BROADCAST_MESH_2D_MEM2MEM, CCU_ALLGATHER_MESH_2D_MEM2MEM, CCU_ALL_GATHER_V_MESH_1D_DIRECT,
    CCU_REDUCE_SCATTER_V_MESH_1D_DIRECT, CCU_REDUCE_SCATTER_V_MESH_1D_MEM2MEM_DIRECT,
    CCU_ALL_REDUCE_MESH_2D_TWO_SHOT_MULTI_MISSION, CCU_ALL_REDUCE_MESH_2D_TWO_SHOT_MEM2MEM, CCU_REDUCE_MESH_1D_MEM2MEM,
    CCU_REDUCE_SCATTER_MESH_2D_MEM2MEM, CCU_BROADCAST_MESH_2D_MULTI_MISSION, CCU_REDUCE_MESH_2D_MEM2MEM,
    CCU_ALLREDUCE_NHR_1D_MEM2MEM, CCU_SCATTER_NHR_1D_MEM2MEM, CCU_BROADCAST_NHR_1D_MEM2MEM, CCU_REDUCE_NHR_1D_MEM2MEM,
    CCU_ALLTOALLV_MESH_2DIE_DIRECT, CCU_ALLGATHER_MESH_1D_2DIE, CCU_ALLTOALL_MESH_1D_2DIE,
    CCU_REDUCE_SCATTER_MESH_1D_2DIE, CCU_REDUCE_MESH_1D_TWO_SHOT_MEM2MEM);

class CcuInstruction : public Instruction {
public:
    CcuInstruction() : Instruction(InstructionType::CCU_INS) {}

    virtual void SetExecId(u64 id) { execId = id; }

    virtual u64 GetExecId() const { return execId; }

    u32 GetCntCkeNum() const { return cntCkeNum; }

    void SetCntCkeNum(u32 num) { cntCkeNum = num; }

    virtual CcuCtxSignature GetCtxSignature() const
    {
        std::unique_ptr<CcuCtxArg> ccuTaskArg = GetCtxArg();
        if (ccuTaskArg == nullptr) {
            THROW<InternalException>("[CcuInstruction][GetCtxSignature]ccuTaskArg is null");
        }
        return ccuTaskArg->GetCtxSignature();
    }

    virtual std::vector<LinkData> GetLinks() const { return links_; }

    virtual void SetLinks(std::vector<LinkData>& links) { links_ = links; }

    virtual RankGroup GetRankGroup() const { return rankGroup_; }

    virtual void SetRankGroup(RankGroup& rankGroup) { rankGroup_ = rankGroup; }

    virtual CcuInstType GetInstType() const { return instType_; }

    virtual void Translate(std::vector<std::vector<CcuTaskParam>>& taskParam) const;

    virtual std::unique_ptr<CcuCtxArg> GetCtxArg() const = 0;
    virtual std::unique_ptr<CcuTaskArg> GetTaskArg() const = 0;
    std::string Describe() const override = 0;

protected:
    u64 execId{0};
    RankGroup rankGroup_;
    std::vector<LinkData> links_;
    CcuInstType instType_;

private:
    u32 cntCkeNum{0};
};

} // namespace Hccl

#endif // CCU_INSTRUCTION_H
