/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_CCU_CONTEXT_REDUCE_MESH_2D_H_
#define HCCLV2_CCU_CONTEXT_REDUCE_MESH_2D_H_

#include <vector>
#include <ios>
#include "log.h"
#include "ccu_context_alg_base.h"
#include "ccu_datatype.h"

namespace Hccl {

class CcuContextReduceMesh2D : public CcuContextAlgBase {
public:
    CcuContextReduceMesh2D(
        const CcuCtxArg& arg, const std::vector<CcuTransport*>& transports, const CcuTransportGroup& group);
    ~CcuContextReduceMesh2D() override {}

    void Algorithm() override;
    std::vector<uint64_t> GeneArgs(const CcuTaskArg& arg) override;

private:
    void InitResources();
    void LoadArgs();
    void PreSync();
    void PostSync(uint32_t signalIndex);
    void AxisSync(uint32_t signalIndex);
    void Step1Reduce();
    void Step2ReduceForRoot();

    std::vector<uint64_t> dimSize_;
    uint32_t axisId_{0}; // 0 : X轴， 1 : Y轴

    std::vector<uint32_t> dimId_; // 本rank所在行或列的编号
    uint32_t localId_{0};         // 本chip所在行或列的编号
    uint32_t localSize_{0};       // 本rank所在行或列的总rank数

    std::vector<uint32_t> rootDimId_; // root所在行或列的编号
    uint32_t rootId_{0};              // 当rankid == rootid时，为root节点 则跳过write操作
    uint32_t rootLocalId_{0};         // root所在行或列的编号
    DataType dataType_;
    DataType outputDataType_;
    ReduceOp reduceOp_;
    std::vector<CcuRep::Variable> input_;
    std::vector<CcuRep::Variable> output_;
    std::vector<CcuRep::Variable> token_;
    CcuRep::Variable offset_;
    GroupOpSize xAxisGroupOpSize_;
    GroupOpSize yAxisGroupOpSize_;

    // 跨轴同步信号
    std::string localAxisSignalName_;
    std::string anotherAxisSignalName_;
    CcuRep::MaskSignal localAxisSignal_;
    CcuRep::MaskSignal anotherAxisSignal_;
};
} // namespace Hccl

#endif // HCCLV2_CCU_CONTEXT_REDUCE_MESH_2D_H_
