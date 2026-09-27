/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_CCU_CONTEXT_REDUCE_MESH_1D_H_
#define HCCLV2_CCU_CONTEXT_REDUCE_MESH_1D_H_

#include <vector>
#include <ios>
#include "log.h"
#include "ccu_context_alg_base.h"
#include "ccu_datatype.h"

namespace Hccl {

class CcuContextReduceMesh1D : public CcuContextAlgBase {
public:
    CcuContextReduceMesh1D(
        const CcuCtxArg& arg, const std::vector<CcuTransport*>& transports, const CcuTransportGroup& group);

    void Algorithm() override;
    std::vector<uint64_t> GeneArgs(const CcuTaskArg& arg) override;

private:
    void InitResource();
    void LoadArgs();
    void PreSync();
    void DoRepeatReduce();
    void PostSync();

    uint32_t rootId_{0}; // 当rankid == rootid时，为root节点 则跳过write操作
    DataType dataType_;
    DataType outputDataType_;
    std::vector<CcuRep::Variable> input_;
    std::vector<CcuRep::Variable> output_;
    std::vector<CcuRep::Variable> token_;
    uint16_t selfBit_{0};
    uint16_t allBit_{0};
    CcuRep::Variable currentRankSliceInputOffset_;
    CcuRep::Variable currentRankSliceOutputOffset_;
    CcuRep::Variable repeatNum_;
    CcuRep::Variable inputRepeatStride_;
    CcuRep::Variable outputRepeatStride_;
    CcuRep::Variable normalSliceSize_;
    CcuRep::Variable lastSliceSize_;
    CcuRep::Variable repeatNumVar_;
    CcuRep::Variable flag_;

    GroupOpSize groupOpSize_;
    CcuRep::Variable srcOffset_;
    CcuRep::Variable dstOffset_;

    CcuRep::Memory localMem_;
    std::vector<CcuRep::Memory> reomteMem_;

    CcuRep::MaskSignal localSignal_;
};
} // namespace Hccl

#endif // HCCLV2_CCU_CONTEXT_REDUCE_MESH_1D_H_
