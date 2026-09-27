/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COLL_ALLGATHERV_MESH_OPBASE_EXECUTOR_H
#define COLL_ALLGATHERV_MESH_OPBASE_EXECUTOR_H
#include "coll_all_gather_v_executor.h"
namespace hccl {
class CollAllGatherVMeshExecutor : public CollAllGatherVExecutor {
public:
    explicit CollAllGatherVMeshExecutor(const HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher>& topoMatcher);
    ~CollAllGatherVMeshExecutor() override = default;

private:
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
    HcclResult CalcCurCountsAndCurDispls(
        const u64 maxTotalCount, std::vector<u64>& countsLeft, std::vector<u64>& displs, std::vector<u64>& curCounts,
        std::vector<u64>& curDispls, bool& finished) override;
    HcclResult CalcCurCountsAndCurDisplsMultiModule(
        const u64 maxTotalCount, std::vector<u64>& countsLeft, std::vector<u64>& displs, std::vector<u64>& curCounts,
        std::vector<u64>& curDispls, bool& finished) const;
    HcclResult CalcCurCountsAndCurDisplsSingleModule(
        const u64 maxTotalCount, std::vector<u64>& countsLeft, std::vector<u64>& displs, std::vector<u64>& curCounts,
        std::vector<u64>& curDispls, bool& finished) const;
    u64 CalcLoopMaxCount(const u64 cclBuffSize, const u32 unitSize) override;
    bool IsHugeData(const u64 curSize) override;
    HcclResult KernelRun(const OpParam& param, ExecMem& execMem) override;
    HcclResult RunSingleMesh(const OpParam& param, ExecMem& execMem);
    HcclResult RunLevel1(
        const OpParam& param, ExecMem& execMem, const SubCommInfo& level0CommInfo, SubCommInfo& level1CommInfo) const;
    HcclResult
    RunLevel0(const OpParam& param, ExecMem& execMem, SubCommInfo& level0CommInfo, const SubCommInfo& level1CommInfo);
};

} // namespace hccl

#endif
