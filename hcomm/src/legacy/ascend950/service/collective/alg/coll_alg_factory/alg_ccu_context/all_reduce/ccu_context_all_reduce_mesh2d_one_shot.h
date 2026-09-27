/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_CCU_CONTEXT_ALL_REDUCE_MESH_2D_ONE_SHOT_H_
#define HCCLV2_CCU_CONTEXT_ALL_REDUCE_MESH_2D_ONE_SHOT_H_

#include <vector>
#include <ios>
#include "log.h"
#include "ccu_context_alg_base.h"
#include "ccu_datatype.h"

namespace Hccl {

class CcuContextAllReduceMesh2DOneShot : public CcuContextAlgBase {
public:
    CcuContextAllReduceMesh2DOneShot(
        const CcuCtxArg& arg, const std::vector<CcuTransport*>& transports, const CcuTransportGroup& group);
    ~CcuContextAllReduceMesh2DOneShot() override {}

    void Algorithm() override;
    std::vector<uint64_t> GeneArgs(const CcuTaskArg& arg) override;

private:
    void InitVariables();
    void LoadArgs();
    void DoAxisSync(uint32_t signalIdx);
    void DoGroupSync(int ckeIdx, uint16_t selfBit, uint16_t allBit);
    void DoGroupReduce(
        std::vector<CcuRep::Variable>& srcBase, CcuRep::Variable& dstBase, CcuRep::Variable& offset,
        GroupOpSize& goSize);
    std::vector<uint64_t> dimSize_;
    std::vector<uint64_t> myRankIdxInAxis_;
    uint64_t myRankIdxInCurrentAxis_{0};
    uint64_t currentAxisRankSize_{0};

    uint32_t axisId_{0};
    DataType dataType_;
    DataType outputDataType_;

    std::vector<CcuRep::Variable> inputAddr_;
    std::vector<CcuRep::Variable> outputAddr_;
    std::vector<CcuRep::Variable> scratchAddr_;
    std::vector<CcuRep::Variable> token_;
    CcuRep::Variable xSliceOffset_;
    CcuRep::Variable ySliceOffset_;
    GroupOpSize xGoSize_;
    GroupOpSize yGoSize_;

    std::string currAxisSignalName_;
    std::string otherAxisSignalName_;
    CcuRep::MaskSignal currAxisSignal_;
    CcuRep::MaskSignal otherAxisSignal_;
};
} // namespace Hccl

#endif // HCCLV2_CCU_CONTEXT_ALL_REDUCE_MESH_2D_ONE_SHOT_H_
