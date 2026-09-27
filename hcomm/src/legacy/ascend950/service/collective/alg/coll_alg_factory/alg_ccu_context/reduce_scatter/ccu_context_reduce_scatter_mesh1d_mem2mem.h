/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_CCU_CONTEXT_REDUCE_SCATTER_MESH_1D_MEM2MEM_H_
#define HCCLV2_CCU_CONTEXT_REDUCE_SCATTER_MESH_1D_MEM2MEM_H_

#include <vector>
#include <ios>
#include "log.h"
#include "ccu_context_alg_base.h"
#include "ccu_datatype.h"

namespace Hccl {
class CcuContextReduceScatterMeshMem2Mem1D : public CcuContextAlgBase {
public:
    CcuContextReduceScatterMeshMem2Mem1D(
        const CcuCtxArg& arg, const std::vector<CcuTransport*>& transports, const CcuTransportGroup& group);
    ~CcuContextReduceScatterMeshMem2Mem1D() override {}

    void Algorithm() override;
    std::vector<uint64_t> GeneArgs(const CcuTaskArg& arg) override;

private:
    void InitResource();
    void LoadArgs();
    void PreSync();
    void PostSync();
    void DoRepeatReduceScatter();
    void DoReduceScatter();

    std::string GetLoopBlockTag(std::string loopType, int32_t index);
    void CreateReduceLoop(uint32_t size, DataType dataType, DataType outputDataType, ReduceOp opType);
    void ReduceLoopGroup(
        CcuRep::Memory outDstOrg, CcuRep::Memory srcOrg, std::vector<CcuRep::Memory>& scratchOrg, GroupOpSize goSize,
        DataType dataType, DataType outputDataType, ReduceOp opType);

    const std::string LOOP_BLOCK_TAG{"_local_copy_reduce_loop_"};

    DataType dataType_;
    DataType outputDataType_;
    CcuRep::Variable repeatNum_;
    std::vector<CcuRep::Variable> input_;
    CcuRep::Variable output_;
    std::vector<CcuRep::Variable> scratch_;
    std::vector<CcuRep::Variable> token_;
    CcuRep::Variable currentRankSliceInputOffset_;
    CcuRep::Variable inputRepeatStride_;
    CcuRep::Variable outputRepeatStride_;
    CcuRep::Variable normalSliceSize_;
    GroupOpSize normalGoSize_;
    uint16_t selfBit_{0};
    uint16_t allBit_{0};
    std::vector<CcuRep::Memory> localMem_;
    std::vector<CcuRep::Memory> remoteMem_;
    CcuRep::MaskSignal localSignal_;
    CcuRep::Variable flag_; // 用以判断是否是第一次重复
};
} // namespace Hccl

#endif // HCCLV2_CCU_CONTEXT_REDUCE_SCATTER_MESH_1D_MEM2MEM_H_
