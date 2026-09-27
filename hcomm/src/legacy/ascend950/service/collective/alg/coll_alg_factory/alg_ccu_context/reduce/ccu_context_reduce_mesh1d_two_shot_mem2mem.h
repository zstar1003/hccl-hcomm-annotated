/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_CCU_CONTEXT_REDUCE_MESH_1D_TWO_SHOT_MEM2MEM_H_
#define HCCLV2_CCU_CONTEXT_REDUCE_MESH_1D_TWO_SHOT_MEM2MEM_H_

#include <vector>

#include <ios>
#include "log.h"
#include "ccu_ctx.h"
#include "ccu_datatype.h"

namespace Hccl {

class CcuContextReduceMeshTwoShotMem2Mem1D : public CcuContext {
public:
    CcuContextReduceMeshTwoShotMem2Mem1D(
        const CcuCtxArg& arg, const std::vector<CcuTransport*>& transports, const CcuTransportGroup& group);
    ~CcuContextReduceMeshTwoShotMem2Mem1D() override {}

    void Algorithm() override;
    std::vector<uint64_t> GeneArgs(const CcuTaskArg& arg) override;

protected:
    void InitResource();
    void LoadArgs();
    void PreSync();
    void PostSync();
    void DoRepeatReduceTwoShot();
    void ReduceRmtToLoc(const std::vector<CcuRep::Variable>& srcAddr, const CcuRep::Variable& dstAddr);
    void BcastLocToRmt(const CcuRep::Variable& srcAddr, const std::vector<CcuRep::Variable>& dstAddr);

private:
    uint64_t rankSize_{0};
    uint32_t rankId_{0};
    uint32_t rootId_{0};
    DataType dataType_;
    DataType outputDataType_;
    ReduceOp reduceOp_;

    std::vector<CcuRep::Variable> input_;
    std::vector<CcuRep::Variable> output_;
    std::vector<CcuRep::Variable> token_;
    std::vector<CcuRep::Variable> scratch_;

    CcuRep::Variable normalSliceSize_;
    CcuRep::Variable lastSliceSize_;
    CcuRep::Variable mySliceSize_;
    std::vector<CcuRep::Variable> sliceOffset_;
    std::vector<CcuRep::Variable> len_;
    CcuRep::Variable isInputOutputEqual_;
    CcuRep::Variable sliceSize_;
    CcuRep::MaskSignal locMask_;

    CcuRep::Memory srcMem_;
    CcuRep::Memory dstMem_;
    std::vector<CcuRep::Memory> reduceScatterSrc_;
    std::vector<CcuRep::Memory> reduceScatterDst_;
    std::vector<CcuRep::Memory> gatherSrc_;
    std::vector<CcuRep::Memory> gatherDst_;

    uint16_t selfBit_{0};
    uint16_t allBit_{0};
    GroupOpSize localGoSize_;

    std::string GetLoopBlockTag(std::string loopType, int32_t index);
    void CreateReduceLoop(uint32_t size, DataType dataType, DataType outputDataType, ReduceOp opType);
    void ReduceLoopGroup(
        CcuRep::Memory outDstOrg, CcuRep::Memory srcOrg, std::vector<CcuRep::Memory>& scratchOrg, GroupOpSize goSize,
        DataType dataType, DataType outputDataType, ReduceOp opType);
    const std::string LOOP_BLOCK_TAG{"_local_copy_reduce_loop_"};
};
} // namespace Hccl

#endif // HCCLV2_CCU_CONTEXT_ALL_REDUCE_MESH_1D_H_
