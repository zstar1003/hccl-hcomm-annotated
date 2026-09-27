/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_CCU_CONTEXT_ALL_TO_ALL_V_MESH_2DIE_H_
#define HCCLV2_CCU_CONTEXT_ALL_TO_ALL_V_MESH_2DIE_H_

#include <vector>
#include <ios>

#include "log.h"
#include "ccu_context_alg_base.h"
#include "ccu_datatype.h"
#include "ccu_instruction_all_to_all_v_mesh2die.h"

namespace Hccl {

class CcuContextAllToAllVMesh2Die : public CcuContextAlgBase {
public:
    CcuContextAllToAllVMesh2Die(
        const CcuCtxArg& arg, const std::vector<CcuTransport*>& transports, const CcuTransportGroup& group);
    ~CcuContextAllToAllVMesh2Die() override {}

    void Algorithm() override;
    std::vector<uint64_t> GeneArgs(const CcuTaskArg& arg) override;

protected:
    // a2a 对每个对端的发送接收信息
    struct A2AVsingleSendRecvInfo {
        CcuRep::Variable sendOffset;
        CcuRep::Variable recvOffset;
        CcuRep::Variable sendTailSize; // 本rank给其他所有rank要发的尾块数据
        GroupOpSize sendTailGoSize;
        CcuRep::Variable sendLoopNum; // 本rank给其他所有rank要发的轮数
    };

private:
    void InitResources();
    void LoadArgs();
    void ExchangeInfoAndSync();
    void PostSync();

    void DoAll2AllVMultiLoop();
    void CalcGroupSrcDst();
    void LoopStep();
    uint32_t CalcDstRank(uint32_t peerId) const;
    uint32_t CalcTransIdx(uint32_t peerId) const;
    void GroupCopyToDstOutput(uint32_t peerId);
    void WriteToDstOutput(uint32_t peerId);

    static constexpr uint32_t RANK_EVEN = 2; // 只支持矩形rank分布
    static constexpr uint64_t MAX_TRANSPORT_SIZE = UB_MAX_TRANS_SIZE;

    static constexpr uint32_t GO_ADDR_OFFSET_IDX = 0;
    static constexpr uint32_t GO_LOOP_PARAM_IDX = 1;
    static constexpr uint32_t GO_PARALLEL_PARAM_IDX = 2;
    static constexpr uint32_t GO_RESIDUAL_IDX = 3;

    CcuRep::Variable input_;
    std::vector<CcuRep::Variable> output_;
    std::vector<CcuRep::Variable> token_;

    CcuRep::Variable xnConst1_;
    CcuRep::Variable completedRankCount_;
    CcuRep::Variable xnMaxTransportSize_;
    GroupOpSize xnMaxTransportGoSize_;
    CcuRep::Variable curSendTailSize_;
    GroupOpSize curSendTailGoSize_;
    std::vector<A2AVsingleSendRecvInfo> sendRecvInfo_;

    bool withMyRank_{false};
    uint32_t localSize_{0}; // 本rank所在DIE的总rank数
    uint32_t localId_{0};   // 本rank所在DIE的编号，固定放在末尾
    uint32_t peerSize_{0};
    uint32_t logicId_{0};
    std::vector<RankId> rankGroup_;

    std::vector<CcuRep::Memory> src_;
    std::vector<CcuRep::Memory> dst_;

    uint16_t selfBit_{0};
    uint16_t allBit_{0};

    // 在本地的搬运完成标记
    CcuRep::MaskSignal locSignal_;
};
} // namespace Hccl

#endif // HCCLV2_CCU_CONTEXT_ALL_TO_ALL_V_MESH_2DIE_H_
