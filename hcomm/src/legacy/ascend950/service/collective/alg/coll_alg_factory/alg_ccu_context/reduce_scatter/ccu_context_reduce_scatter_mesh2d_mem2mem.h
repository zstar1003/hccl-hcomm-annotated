/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_CCU_CONTEXT_REDUCE_SCATTER_MESH_2D_MEM2MEM_H_
#define HCCLV2_CCU_CONTEXT_REDUCE_SCATTER_MESH_2D_MEM2MEM_H_

#include <vector>
#include <ios>
#include "log.h"
#include "ccu_context_alg_base.h"
#include "ccu_datatype.h"
#include "ccu_assist.h"

namespace Hccl {

class CcuContextReduceScatterMeshMem2Mem2D : public CcuContextAlgBase {
public:
    CcuContextReduceScatterMeshMem2Mem2D(
        const CcuCtxArg& arg, const std::vector<CcuTransport*>& transports, const CcuTransportGroup& group);
    ~CcuContextReduceScatterMeshMem2Mem2D() override {}

    void Algorithm() override;
    std::vector<uint64_t> GeneArgs(const CcuTaskArg& arg) override;

protected:
    void InitResources();

private:
    void CreateLocalCopyLoop();                                                    // for loop-group local copy
    void LocalCopyByLoopGroup(CcuRep::Memory dst, CcuRep::Memory src);             // for loop-group local copy
    std::vector<uint64_t> CalMeshChunkSlice(uint64_t dataSize, uint64_t sliceNum); // for mesh-chunk
    void PreSync();
    void PostSync(uint32_t signalIndex);
    void LoadArgs();
    void AxisSync(uint32_t signalIndex);
    void Step1Reduce();
    void Step2Reduce();

    // 在构造函数中初始化
    uint32_t axisId_{0};            // 值为 0 表示 x 轴, 值为 1 表示 y 轴
    std::vector<uint64_t> dimSize_; // [0] mesh2d 拓扑的列数
    std::vector<uint32_t> dimId_;   // [0] 当前 NPU 所在列编号, [1] 当前 NPU 所在行编号
    uint32_t localId_{0};           // 当前 rank 在指定 axisId_ 上的编号
    uint32_t localSize_{0};         // mesh2d 拓扑在指定 axisId_ 轴上的维度
    uint32_t oppsiteSize_{0};       // mesh2d 拓扑在与指定 axisId_ 轴相反的轴上的维度
    DataType dataType_;
    DataType outputDataType_;
    // load 进来的参数
    std::vector<CcuRep::Variable> input_;
    std::vector<CcuRep::Variable> output_;
    std::vector<CcuRep::Variable> token_;
    CcuRep::Variable xAxisSize_;
    CcuRep::Variable yAxisSize_;
    CcuRep::Variable step0BaseOffset_;
    CcuRep::Variable step0AddOffset_;
    CcuRep::Variable step1AddOffset_;
    CcuRep::Variable yAxisOffset_;
    GroupOpSize xAxisGroupOpSize_;
    GroupOpSize yAxisGroupOpSize_;
    GroupOpSize curGoSize_; // for loop-group local copy

    // For mesh-chunk
    std::vector<CcuRep::Variable> xlocalSlice_;
    std::vector<CcuRep::Variable> ylocalSlice_;

    CcuRep::Variable inputSize_;
    CcuRep::Variable offset_;
    CcuRep::Variable sliceOffset_;
    CcuRep::Variable strideSize_;

    // 跨轴同步信号
    std::string localAxisSignalName_;   // 由 axisId_ 初始化
    std::string anotherAxisSignalName_; // 由 axisId_ 初始化
    CcuRep::MaskSignal localAxisSignal_;
    CcuRep::MaskSignal anotherAxisSignal_;
}; // class CcuContextReduceScatterMeshMem2Mem2D
} // namespace Hccl

#endif // HCCLV2_CCU_CONTEXT_REDUCE_SCATTER_MESH_2D_MEM2MEM_H_
