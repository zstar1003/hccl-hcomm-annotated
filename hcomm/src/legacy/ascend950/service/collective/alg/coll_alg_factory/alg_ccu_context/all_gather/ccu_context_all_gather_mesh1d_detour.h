/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_CCU_CONTEXT_ALL_GATHER_MESH_1D_DETOUR_H
#define HCCLV2_CCU_CONTEXT_ALL_GATHER_MESH_1D_DETOUR_H

#include <vector>

#include <ios>
#include "log.h"
#include "ccu_context_alg_base.h"
#include "ccu_datatype.h"

namespace Hccl {

class CcuContextAllGatherMeshDetour1D : public CcuContextAlgBase {
public:
    CcuContextAllGatherMeshDetour1D(
        const CcuCtxArg& arg, const std::vector<CcuTransport*>& transports, const CcuTransportGroup& group);
    ~CcuContextAllGatherMeshDetour1D() override {}

    void Algorithm() override;
    std::vector<uint64_t> GeneArgs(const CcuTaskArg& arg) override;

private:
    void ProcessTransports(const std::vector<CcuTransport*>& transports);
    void AllocDetourRes();
    void CreateMultiOpBroadcastDetour();
    void GroupBroadcastDetour(
        std::vector<CcuRep::Variable>& lengths, std::vector<CcuRep::Memory>& src, std::vector<CcuRep::Memory>& dst);
    void FirstStep();
    void SecondStep();

    // ctx args
    uint64_t singleTransportSize_{0}; // 每个loop单次传输的总数据量，通信域级别
    uint64_t detourPathNum_{0};
    uint64_t pathNumPerPeer_{0}; // 到每个rank有几个transport，包括重复的
    std::vector<std::vector<CcuTransport*>>
        detourTransports_; // 默认是transport的每rankSize-1个为一组，第一组是直连链路

    // task args
    CcuRep::Variable input_;
    std::vector<CcuRep::Variable> output_;
    std::vector<CcuRep::Variable> token_;
    CcuRep::Variable baseOffset_;           // rankId*sliceSize，待广播分片的基础偏移
    CcuRep::Variable tailOffset_;           // 尾块相对偏移，singleTransportSize*128*iterNum
    CcuRep::Variable loopIterNum_;          // for detour loopgroup only
    std::vector<CcuRep::Variable> lengths_; // 每组transport对应一个len
    GroupOpSize groupOpSize_;               // 只处理尾块
};
} // namespace Hccl

#endif // HCCLV2_CCU_CONTEXT_ALL_GATHER_MESH_1D_DETOUR_H
