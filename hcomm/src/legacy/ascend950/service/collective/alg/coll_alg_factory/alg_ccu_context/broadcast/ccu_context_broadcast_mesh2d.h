/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_CCU_CONTEXT_BROADCAST_MESH_2D_H_
#define HCCLV2_CCU_CONTEXT_BROADCAST_MESH_2D_H_

#include <vector>
#include <ios>
#include "log.h"
#include "ccu_context_alg_base.h"
#include "ccu_datatype.h"

namespace Hccl {

class CcuContextBroadcastMesh2D : public CcuContextAlgBase {
public:
    CcuContextBroadcastMesh2D(
        const CcuCtxArg& arg, const std::vector<CcuTransport*>& transports, const CcuTransportGroup& group);
    ~CcuContextBroadcastMesh2D() override {}

    void Algorithm() override;
    std::vector<uint64_t> GeneArgs(const CcuTaskArg& arg) override;

private:
    void InitResources();
    void PreSync();
    void PostSync(uint32_t signalIndex);
    void LoadArgs();
    void AxisSync(uint32_t signalIndex);
    void Step1BroadcastForRoot();
    void Step2BroadcastForRoot();
    void Step2Broadcast();

    // 构造函数中
    std::vector<uint64_t> dimSize_;
    uint32_t axisId_{0};
    std::vector<uint32_t> dimId_;     // 本rank所在行或列的编号
    std::vector<uint32_t> rootDimId_; // root所在行或列的编号
    uint32_t localId_{0};             // 本rank所在行或列的编号
    uint32_t rootLocalId_{0};         // root所在行或列的编号
    uint32_t localSize_{0};           // 本rank所在行或列的总rank数
    DataType dataType_;
    uint32_t rootId_{0};

    // load进来参数
    std::vector<CcuRep::Variable> input_;
    std::vector<CcuRep::Variable> token_;
    CcuRep::Variable yAxisOffset_;
    GroupOpSize xAxisGroupOpSize_;
    GroupOpSize yAxisGroupOpSize_;

    // 跨轴同步信号
    CcuRep::MaskSignal localAxisSignal_;
    CcuRep::MaskSignal anotherAxisSignal_;
    std::string localAxisSignalName_;
    std::string anotherAxisSignalName_;
};
} // namespace Hccl

#endif // HCCLV2_CCU_CONTEXT_BROADCAST_MESH_2D_H_
