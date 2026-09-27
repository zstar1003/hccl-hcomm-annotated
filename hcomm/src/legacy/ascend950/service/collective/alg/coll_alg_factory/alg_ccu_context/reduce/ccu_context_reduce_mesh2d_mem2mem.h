/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_CCU_CONTEXT_REDUCE_MESH_2D_MEM2MEM_H_
#define HCCLV2_CCU_CONTEXT_REDUCE_MESH_2D_MEM2MEM_H_

#include <vector>
#include <ios>
#include "log.h"
#include "ccu_context_alg_base.h"
#include "ccu_datatype.h"
#include "ccu_assist.h"

namespace Hccl {
constexpr int INPUT_XN_ID = 1;
constexpr int TOKEN_XN_ID = 2;
constexpr int CKE_IDX_0 = 0;
constexpr int CKE_IDX_1 = 1;
constexpr int CKE_IDX_2 = 2;
constexpr int CKE_IDX_3 = 3;
constexpr int CKE_IDX_4 = 4;
constexpr uint64_t CCU_MS_SIZE = 4096;
constexpr uint64_t LOCAL_COPY_MS = 8;
constexpr int X_AXIS_ID = 0;
constexpr int Y_AXIS_ID = 1;

class CcuContextReduceMeshMem2Mem2D : public CcuContextAlgBase {
public:
    CcuContextReduceMeshMem2Mem2D(
        const CcuCtxArg& arg, const std::vector<CcuTransport*>& transports, const CcuTransportGroup& group);
    ~CcuContextReduceMeshMem2Mem2D() override {}

    void Algorithm() override;
    std::vector<uint64_t> GeneArgs(const CcuTaskArg& arg) override;

private:
    void InitResources();
    void LoadArgs();
    void PreSync();
    void PostSync(uint32_t signalIndex);
    void AxisSync(uint32_t signalIndex);
    void ReduceStep1();
    void ReduceStep2();
    std::vector<uint64_t> CalMeshChunkSlice(uint64_t dataSize, uint64_t sliceNum); // for mesh-chunk

    std::vector<uint64_t> dimSize_;
    uint32_t axisId_{0}; // 0 : X轴， 1 : Y轴

    std::vector<uint32_t> dimId_; // 本rank所在行或列的编号
    uint32_t localId_{0};         // 本chip所在行或列的编号
    uint32_t localSize_{0};       // 本rank所在行或列的总rank数

    std::vector<uint32_t> rootDimId_; // root所在行或列的编号
    uint32_t rootId_{0};              // 当rankid == rootid时，为root节点 则跳过write操作
    DataType dataType_;
    DataType outputDataType_;
    ReduceOp reduceOp_;
    std::vector<CcuRep::Variable> input_;
    CcuRep::Variable output_;
    std::vector<CcuRep::Variable> token_;
    CcuRep::Variable xAxisSize_;
    CcuRep::Variable yAxisSize_;
    CcuRep::Variable yAxisOffset_;
    CcuRep::MaskSignal locMask_;

    GroupOpSize xAxisGroupOpSize_;
    GroupOpSize yAxisGroupOpSize_;
    GroupOpSize curGoSize_; // for loop-group local copy
    // variables for mesh-chunk
    std::vector<CcuRep::Variable> xChunkSize_; // for x slice size
    std::vector<CcuRep::Variable> yChunkSize_; // for y slice size
    std::vector<CcuRep::Variable> chunkSize_;  // for current axis
    CcuRep::Variable chunkOffset_;

    // 跨轴同步信号
    std::string localAxisSignalName_;
    std::string anotherAxisSignalName_;
    CcuRep::MaskSignal localAxisSignal_;
    CcuRep::MaskSignal anotherAxisSignal_;
};
} // namespace Hccl

#endif // HCCLV2_CCU_CONTEXT_REDUCE_MESH_2D_MEM2MEM_H_
