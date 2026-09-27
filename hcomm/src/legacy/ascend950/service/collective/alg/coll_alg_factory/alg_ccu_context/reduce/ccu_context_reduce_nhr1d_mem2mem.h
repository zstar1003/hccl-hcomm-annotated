/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CUPT_KERNEL_CCU_CONTEXT_REDUCE_NHR_1D_H_
#define CUPT_KERNEL_CCU_CONTEXT_REDUCE_NHR_1D_H_
#include <vector>

#include <ios>
#include "log.h"
#include "ccu_context_alg_base.h"
#include "ccu_datatype.h"
#include "ccu_instruction_reduce_nhr1d_mem2mem.h"

namespace Hccl {

class CcuContextReduceNHR1DMem2mem : public CcuContextAlgBase {
public:
    CcuContextReduceNHR1DMem2mem(
        const CcuCtxArg& arg, const std::vector<CcuTransport*>& transports, const CcuTransportGroup& group);
    ~CcuContextReduceNHR1DMem2mem() override {}

    void Algorithm() override;
    std::vector<uint64_t> GeneArgs(const CcuTaskArg& arg) override;

private:
    void LoadArgs();
    void InitResources();
    void PreSync();
    void PostSync();
    void AxisSync(uint32_t signalIndex);
    void LocalCopySlices();
    void DoLocalCopySlice(CcuRep::Memory& src, CcuRep::Memory& dst, const u32& copySliceIdx, u32 signalIndex);
    std::vector<u32> GetNonTxSliceIdxs(const std::vector<u32>& txSliceIdxs) const;
    void DoReduceScatterNHR();
    void DoReduceScatterNHRSingleStep(const NHRStepInfo& nhrStepInfo);
    void DoWriteReduceSlice(
        const u32& toRank, CcuRep::Memory& src, CcuRep::Memory& dst, const u32& sendSliceIdx, u32 signalIndex);
    void DoGatherNHR();
    void DoGatherNHRSingleStep(const NHRStepInfo& nhrStepInfo);
    void DoSendRecvSlice(
        const u32& toRank, CcuRep::Memory& src, CcuRep::Memory& dst, const u32& sendSliceIdx, u32 signalIndex);

    // 构造函数中
    uint32_t rootId_{0};
    uint64_t dimSize_{0};
    uint32_t axisId_{0};
    uint32_t axisSize_{0};
    uint32_t localSize_{0}; // 本rank所在行或列的总rank数
    uint32_t myRankIdx_{0};
    uint32_t signalNum_{0}; // 需要使用的signal数量
    uint32_t repeatNum_{0};
    DataType dataType_;
    // DataType outputDataType_;
    std::vector<NHRStepInfo> stepInfoVector_; // nhr算法执行过程中的参数
    std::map<u32, u32> indexMap_;

    // load进来参数
    CcuRep::Variable input_;
    std::vector<CcuRep::Variable> output_;
    std::vector<CcuRep::Variable> token_;
    CcuRep::Variable isInputOutputEqual_;
    CcuRep::Variable die0LastSliceSize_;
    CcuRep::Variable die1LastSliceSize_;
    CcuRep::Variable repeatInputOffset_;
    CcuRep::Variable repeatOutputOffset_;
    CcuRep::Variable die0Size_;
    CcuRep::Variable die1Size_;
    CcuRep::Variable sliceSize_;
    CcuRep::Variable die0SliceSize_;
    CcuRep::Variable die1SliceSize_;

    // 跨轴同步信号
    std::string localAxisSignalName_;
    std::string anotherAxisSignalName_;
    CcuRep::MaskSignal localAxisSignal_;
    CcuRep::MaskSignal anotherAxisSignal_;
    CcuRep::MaskSignal localSignal_;

    std::vector<CcuRep::Variable> sliceOffset_;

    CcuRep::Memory srcMem_;
    CcuRep::Memory dstMem_;
};
} // namespace Hccl

#endif // HCCLV2_CCU_CONTEXT_ALL_REDUCE_NHR_1D_H_
