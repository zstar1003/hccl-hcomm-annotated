/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_CCU_CONTEXT_ALL_REDUCE_MESH_2D_TWO_SHOT_H_
#define HCCLV2_CCU_CONTEXT_ALL_REDUCE_MESH_2D_TWO_SHOT_H_

#include <vector>
#include <ios>
#include "log.h"
#include "ccu_context_alg_base.h"
#include "ccu_datatype.h"

namespace Hccl {

class CcuContextAllReduceMesh2DTwoShot : public CcuContextAlgBase {
public:
    CcuContextAllReduceMesh2DTwoShot(
        const CcuCtxArg& arg, const std::vector<CcuTransport*>& transports, const CcuTransportGroup& group);
    ~CcuContextAllReduceMesh2DTwoShot() override {}

    void Algorithm() override;
    std::vector<uint64_t> GeneArgs(const CcuTaskArg& arg) override;

private:
    void InitVariables();
    void LoadArgs();
    void SyncAll(int ckeIdx);
    void PreSync();
    void GetSliceOffsetAndGoSize(
        uint64_t currentSliceRankIdx, uint64_t currStepSliceType, CcuRep::Variable& currOffset,
        GroupOpSize& currGoSize);
    void DoAxisSync(uint32_t signalIdx);
    void DoGroupSync(int ckeIdx, uint16_t selfBit, uint16_t allBit);
    void DoGroupReduce(
        std::vector<CcuRep::Variable>& srcBase, CcuRep::Variable& dstBase, CcuRep::Variable& offset,
        GroupOpSize& goSize);
    void DoGroupBroadcast(
        CcuRep::Variable& srcBase, std::vector<CcuRep::Variable>& dstBase, CcuRep::Variable& offset,
        GroupOpSize& goSize);

    uint32_t axisId_{0};
    DataType dataType_;
    DataType outputDataType_;
    uint16_t selfBit_{0};
    uint16_t allBit_{0};
    uint32_t otherAxisId_{0};
    std::vector<uint64_t> myRankIdxInAxis_;
    uint64_t myRankIdxInCurrentAxis_{0};
    uint64_t currentAxisRankSize_{0};
    uint64_t myRankIdxInOtherAxis_{0};
    uint64_t otherAxisRankSize_{0};

    std::vector<uint64_t> dimSize_;
    std::vector<CcuRep::Variable> inputAddr_;
    std::vector<CcuRep::Variable> outputAddr_;
    std::vector<CcuRep::Variable> token_;

    CcuRep::Variable normalRankXSliceSize_;
    CcuRep::Variable normalRankYSliceSize_;
    CcuRep::Variable lastRankXSliceSize_;
    CcuRep::Variable lastRankYSliceSize_;

    GroupOpSize normalRankXGoSize_;
    GroupOpSize normalRankYGoSize_;
    GroupOpSize lastRankXGoSize_;
    GroupOpSize lastRankYGoSize_;

    std::string currAxisSignalName_;
    std::string otherAxisSignalName_;
    CcuRep::MaskSignal currAxisSignal_;
    CcuRep::MaskSignal otherAxisSignal_;

    // 算法模板运行时的参数
    std::vector<CcuRep::Memory> tmpAddrList_;
    CcuRep::Memory tmpAddr_;
};
} // namespace Hccl

#endif // HCCLV2_CCU_CONTEXT_ALL_REDUCE_MESH_2D_TWO_SHOT_H_
