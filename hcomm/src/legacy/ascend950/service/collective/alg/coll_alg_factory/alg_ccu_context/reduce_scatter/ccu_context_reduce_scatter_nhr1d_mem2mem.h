/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CUPT_KERNEL_CCU_CONTEXT_REDUCE_SCATTER_NHR_1D_H_
#define CUPT_KERNEL_CCU_CONTEXT_REDUCE_SCATTER_NHR_1D_H_
#include <vector>
#include <ios>
#include "log.h"
#include "ccu_context_alg_base.h"
#include "ccu_datatype.h"
#include "ccu_instruction_reduce_scatter_nhr1d_mem2mem.h"

namespace Hccl {

class CcuContextReduceScatterNHR1DMem2Mem : public CcuContextAlgBase {
public:
    CcuContextReduceScatterNHR1DMem2Mem(
        const CcuCtxArg& arg, const std::vector<CcuTransport*>& transports, const CcuTransportGroup& group);
    ~CcuContextReduceScatterNHR1DMem2Mem() override {}

    void Algorithm() override;
    std::vector<uint64_t> GeneArgs(const CcuTaskArg& arg) override;

private:
    void LoadArgs();
    void InitResources();
    void PreSync();
    void PostSync();
    void AxisSync(uint32_t signalIndex);
    void DoRepeatReduceScatterNHR();
    void DoRepeatReduceScatterNHRSingleStep(
        const NHRStepInfo& nhrStepInfo, const std::vector<CcuRep::Variable>& inputSliceOffset);
    void DoRepeatSendRecvSlices(const u32& toRank, CcuRep::Memory& src, CcuRep::Memory& dst);

    // 构造函数中
    uint64_t dimSize_{0};
    uint32_t axisId_{0};
    uint32_t localSize_{0}; // 本rank所在行或列的总rank数
    uint32_t myRankIdx_{0};
    uint32_t signalNum_{0}; // 需要使用的signal数量
    DataType dataType_;
    DataType outputDataType_;
    std::vector<NHRStepInfo> stepInfoVector_; // nhr算法执行过程中的参数
    std::map<u32, u32> indexMap_;
    uint32_t linkNum_{0};

    // load进来参数
    std::vector<CcuRep::Variable> input_;
    CcuRep::Variable output_;
    std::vector<CcuRep::Variable> token_;
    CcuRep::Variable die0Size_;
    CcuRep::Variable die1Size_;
    CcuRep::Variable inputSliceStride_;
    CcuRep::Variable outputSliceStride_;
    CcuRep::Variable inputRepeatStride_;
    CcuRep::Variable outputRepeatStride_;
    CcuRep::Variable repeatNumVar_;
    CcuRep::Variable isBottom_;
    CcuRep::Variable repeatNumVarTemp_;
    // 用于记录每次writereduce的数据
    CcuRep::Variable sliceSize_;

    // 跨轴同步信号
    std::string localAxisSignalName_;
    std::string anotherAxisSignalName_;
    CcuRep::MaskSignal localAxisSignal_;
    CcuRep::MaskSignal anotherAxisSignal_;
    CcuRep::MaskSignal localSignal_;

    CcuRep::Variable repeatInputOffset_;
    CcuRep::Variable repeatOutputOffset_;
    CcuRep::Variable myrankInputSliceOffset_;

    CcuRep::Memory srcMem_;
    CcuRep::Memory dstMem_;
    CcuRep::Variable flag_; // 用于判断是否是第一次循环
};
} // namespace Hccl

#endif // HCCLV2_CCU_CONTEXT_REDUCE_SCATTER_NHR_1D_H_
