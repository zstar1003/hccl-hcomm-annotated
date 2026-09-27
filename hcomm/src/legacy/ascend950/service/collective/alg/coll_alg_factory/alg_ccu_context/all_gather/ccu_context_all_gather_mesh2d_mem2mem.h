/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_CCU_CONTEXT_ALL_GATHER_MESH_2D_MEM2MEM_H_
#define HCCLV2_CCU_CONTEXT_ALL_GATHER_MESH_2D_MEM2MEM_H_

#include <vector>

#include <ios>
#include "log.h"
#include "ccu_context_alg_base.h"
#include "ccu_datatype.h"

namespace Hccl {

class CcuContextAllGatherMeshMem2Mem2D : public CcuContextAlgBase {
public:
    CcuContextAllGatherMeshMem2Mem2D(
        const CcuCtxArg& arg, const std::vector<CcuTransport*>& transports, const CcuTransportGroup& group);
    ~CcuContextAllGatherMeshMem2Mem2D() override {}

    void Algorithm() override;
    std::vector<uint64_t> GeneArgs(const CcuTaskArg& arg) override;

private:
    void InitResources();
    void LoadArgs();
    void ExchangeInfoAndSync();
    void RankSync(uint32_t signalIndex);
    void AxisSync(uint32_t signalIndex);
    void FirstStep();
    void SecondStep();

    static constexpr uint32_t DIM_SIZE = 2;
    std::vector<uint64_t> dimSize_;
    uint32_t axisId_{0}; // 0 : X轴， 1 : Y轴

    std::vector<uint32_t> dimId_; // 本rank所在行或列的编号
    uint32_t localId_{0};         // 本chip所在行或列的编号
    uint32_t localSize_{0};       // 本rank所在行或列的总rank数

    // 从外部获取的参数
    std::vector<CcuRep::Variable> input_;
    std::vector<CcuRep::Variable> output_;
    std::vector<CcuRep::Variable> token_;

    CcuRep::Variable xAxisSize_; // 沿X轴搬移的数据量大小
    CcuRep::Variable yAxisSize_; // 沿Y轴搬移的数据量大小
    CcuRep::Variable offset_;    // Rank偏移+loop偏移
    CcuRep::Variable sliceSize_; // aSize + bSize 应该可以去掉

    CcuRep::Variable firstInOffset_;
    CcuRep::Variable firstOutOffset_;
    CcuRep::Variable secondInOutBaseOffset_;
    CcuRep::Variable secondInOutStepOffset_;

    GroupOpSize goASize_;
    GroupOpSize goBSize_;

    // 跨轴同步信号
    std::string localAxisSignalName_;
    std::string anotherAxisSignalName_;
    CcuRep::MaskSignal localAxisSignal_;
    CcuRep::MaskSignal anotherAxisSignal_;
};
} // namespace Hccl

#endif // HCCLV2_CCU_CONTEXT_ALL_GATHER_MESH_2D_MEM2MEM_H_
