/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_CCU_CONTEXT_ALL_TO_ALL_MESH_1D_2DIE_H_
#define HCCLV2_CCU_CONTEXT_ALL_TO_ALL_MESH_1D_2DIE_H_

#include <vector>
#include <ios>
#include "log.h"
#include "ccu_context_alg_base.h"
#include "ccu_datatype.h"
#include "ccu_assist.h"

namespace Hccl {

class CcuContextAllToAllMesh1D2Die : public CcuContextAlgBase {
public:
    CcuContextAllToAllMesh1D2Die(
        const CcuCtxArg& arg, const std::vector<CcuTransport*>& transports, const CcuTransportGroup& group);
    ~CcuContextAllToAllMesh1D2Die() override {}

    void Algorithm() override;
    std::vector<uint64_t> GeneArgs(const CcuTaskArg& arg) override;

private:
    uint16_t virRankSize{0};
    uint64_t logicRankSize{0};
    uint16_t selfBit{0};
    uint16_t allBit{0};
    std::vector<CcuRep::Variable> input_;
    std::vector<CcuRep::Variable> output_;
    std::vector<CcuRep::Variable> token_;
    bool withMyRank_ = true; // 发数据是否包含本rank
    std::vector<RankId> rankGroup_;
    CcuRep::Variable sliceSize_;
    CcuRep::Variable inputSliceStride_;
    CcuRep::Variable outputoffset_;
    CcuRep::Variable outBuffBaseOff_;
    GroupOpSize groupOpSize_;
    uint32_t signalNum_{0}; // 需要使用的signal数量
    uint16_t bitNumPerCKE_{0};
    void CreateLocalCopyLoop();
    void LocalCopyByLoopGroup(CcuRep::Memory dst, CcuRep::Memory src);
    void InitResource();
    void LoadArgs();
    void PreSync();
    void PostSync();
    uint32_t CalcDstRank(uint32_t peerId) const;
    void DoRepeatAllToAll();
};
} // namespace Hccl

#endif // HCCLV2_CCU_CONTEXT_ALL_TO_ALL_MESH_1D_2DIE_H_
