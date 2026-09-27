/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_CCU_CONTEXT_ALL_REDUCE_MESH_1D_ONE_SHOT_H_
#define HCCLV2_CCU_CONTEXT_ALL_REDUCE_MESH_1D_ONE_SHOT_H_

#include <vector>
#include <ios>
#include "log.h"
#include "ccu_context_alg_base.h"
#include "ccu_datatype.h"

namespace Hccl {

class CcuContextAllReduceMesh1DOneShot : public CcuContextAlgBase {
public:
    CcuContextAllReduceMesh1DOneShot(
        const CcuCtxArg& arg, const std::vector<CcuTransport*>& transports, const CcuTransportGroup& group);
    ~CcuContextAllReduceMesh1DOneShot() override {}

    void Algorithm() override;
    std::vector<uint64_t> GeneArgs(const CcuTaskArg& arg) override;

private:
    void InitResource();
    void ImportReduceVariables();
    void LoadArgs();
    void Presync();
    void Postsync();
    void SyncTailBlock(uint32_t ctxSignalIndex);
    void DoGroupReduce();
    void
    CalcMissionOffset(uint64_t sliceSize, uint64_t missionId, uint64_t& missionSize, uint64_t& missionOffset) const;

    DataType dataType_;
    DataType outputDataType_;

    std::string notifySignal_ = "empty_signal";

    std::vector<CcuRep::Variable> input_;
    CcuRep::Variable output_;
    std::vector<CcuRep::Variable> token_;
    GroupOpSize groupOpSize_;

    CcuRep::Variable tailBlockOffSet_;
    GroupOpSize tailBlockGroupOpSize_;

    CcuRep::Variable reduceInput_;
    CcuRep::Variable reduceOutput_;
    CcuRep::Variable reducetoken_;
    CcuRep::Variable reduceInputOffSet_;
    CcuRep::Variable reduceOutputOffSet_;
    GroupOpSize reduceGroupOpSize_;

    CcuRep::MaskSignal mainBlockCtxSignal_;
    CcuRep::MaskSignal tailBlockCtxSignal_;
};
} // namespace Hccl

#endif // HCCLV2_CCU_CONTEXT_ALL_REDUCE_MESH_1D_H_
