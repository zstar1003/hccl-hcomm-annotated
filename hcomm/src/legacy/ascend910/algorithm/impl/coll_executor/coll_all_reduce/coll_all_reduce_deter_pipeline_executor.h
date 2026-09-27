/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COLL_ALLREDUCE_DETER_PIPELINE_EXECUTOR_H
#define COLL_ALLREDUCE_DETER_PIPELINE_EXECUTOR_H
#include "coll_all_reduce_executor.h"

namespace hccl {
class CollAllReduceDeterPipelineExecutor : public CollAllReduceExecutor {
public:
    explicit CollAllReduceDeterPipelineExecutor(
        const HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher>& topoMatcher);
    ~CollAllReduceDeterPipelineExecutor() override = default;

private:
    void ParseParam(const OpParam& param) override;
    /* *************** 资源计算 *************** */
    HcclResult CalcStreamNum(u32& streamNum) override;
    HcclResult CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport) override;
    HcclResult CalcLevel0CommInfo(
        TransportMemType inputType, TransportMemType outputType,
        std::vector<LevelNSubCommTransport>& opTransport) override;
    HcclResult CalcLevel1CommInfo(
        TransportMemType inputType, TransportMemType outputType,
        std::vector<LevelNSubCommTransport>& opTransport) override;
    HcclResult CalcTransportMemType(TransportMemType& inputType, TransportMemType& outputType) const;

    /* *************** 算法编排 *************** */
    HcclResult RunLoopInner(OpParam& param, const ReduceType& reduceType, ExecMem& execMem) override;
    HcclResult KernelRun(const OpParam& param, ExecMem& execMem) override;

    /* **************** 数据准备*************** */
    u64 CalcCountPerSlice(const u64& totalCount, const u32& unitSize);
    HcclResult PrepareDataSlice(const ExecMem& execMem, const u32& unitSize, std::vector<Slice>& bufferSlices);
    u64 totalSize_{0U};
};

} // namespace hccl

#endif
