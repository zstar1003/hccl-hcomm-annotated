/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COLL_REDUCESCATTERV_MESH_OPBASE_EXECUTOR_H
#define COLL_REDUCESCATTERV_MESH_OPBASE_EXECUTOR_H
#include "coll_reduce_scatter_v_executor.h"

// 对应 CollReduceScatterMeshDmaEliminationExecutor
namespace hccl {
class CollReduceScatterVMeshOpbaseExecutor : public CollReduceScatterVExecutor {
public:
    explicit CollReduceScatterVMeshOpbaseExecutor(
        const HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher>& topoMatcher);
    ~CollReduceScatterVMeshOpbaseExecutor() override = default;

private:
    void ParseParam(const OpParam& param) override;
    /* *************** 资源计算 *************** */
    HcclResult CalcStreamNum(u32& streamNum) override;
    HcclResult CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport) override;
    HcclResult CalcLevel0CommInfo(
        TransportMemType inputType, TransportMemType outputType,
        std::vector<LevelNSubCommTransport>& opTransport) override;
    HcclResult CalcTransportMemType(TransportMemType& inputType, TransportMemType& outputType) const;

    /* *************** 算法编排 *************** */
    HcclResult CalReduceScatterVSliceData(
        const OpParam& param, u32 level0RankSize, u32 level1RankSize, std::vector<Slice>& dataSlices) const;
    HcclResult CalcCurCountsAndCurDisplsSingleModule(
        const u64 maxTotalCount, std::vector<u64>& countsLeft, std::vector<u64>& displs, std::vector<u64>& curCounts,
        std::vector<u64>& curDispls, bool& finished) const;
    HcclResult CalcCurCountsAndCurDisplsMultiModule(
        const u64 maxTotalCount, std::vector<u64>& countsLeft, std::vector<u64>& displs, std::vector<u64>& curCounts,
        std::vector<u64>& curDispls, bool& finished) const;
    HcclResult CalcCurCountsAndCurDispls(
        const u64 maxTotalCount, std::vector<u64>& countsLeft, std::vector<u64>& displs, std::vector<u64>& curCounts,
        std::vector<u64>& curDispls, bool& finished) override;
    bool IsHugeData(const u64 curSize, const OpParam& param) override;
    HcclResult RunReduceScattervLevel0SingleModule(const OpParam& param, ExecMem& execMem, SubCommInfo& level0CommInfo);
    HcclResult RunReduceScattervLevel0(const OpParam& param, ExecMem& execMem, SubCommInfo& level0CommInfo);
    HcclResult RunReduceScattervLevel1(const OpParam& param, ExecMem& execMem, const SubCommInfo& level0CommInfo);
    HcclResult KernelRun(const OpParam& param, ExecMem& execMem) override;
};

} // namespace hccl

#endif
