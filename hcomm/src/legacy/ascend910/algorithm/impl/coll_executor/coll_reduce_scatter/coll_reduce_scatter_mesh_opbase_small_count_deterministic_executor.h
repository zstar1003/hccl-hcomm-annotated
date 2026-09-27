/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COLL_REDUCE_SCATTER_MESH_OPBASE_SMALL_COUNT_DETERMINISTIC_EXECUTOR_H
#define COLL_REDUCE_SCATTER_MESH_OPBASE_SMALL_COUNT_DETERMINISTIC_EXECUTOR_H

#include "coll_reduce_scatter_executor.h"

namespace hccl {
class CollReduceScatterMeshOpbaseSmallCountDeterministicExecutor : public CollReduceScatterExecutor {
public:
    CollReduceScatterMeshOpbaseSmallCountDeterministicExecutor(
        const HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher>& topoMatcher);
    ~CollReduceScatterMeshOpbaseSmallCountDeterministicExecutor() override = default;

private:
    /**************** 资源计算 *************** */
    HcclResult CalcStreamNum(u32& streamNum) override;
    HcclResult CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport) override;
    HcclResult CalcLevel0CommInfo(
        TransportMemType inputType, TransportMemType outputType,
        std::vector<LevelNSubCommTransport>& opTransport) override;
    HcclResult CalcLevel1CommInfo(
        TransportMemType inputType, TransportMemType outputType,
        std::vector<LevelNSubCommTransport>& opTransport) override;
    HcclResult CalcTransportMemType(TransportMemType& inputType, TransportMemType& outputType) const;
    /* *************** 任务编排 *************** */
    u64 CalcLoopMaxCount(const u32 unitSize) override;
    bool IsHugeData(const u64 curSize, OpParam* param = nullptr) override;
    bool IsSmallData(const u64 totalSize, const u64 curSize) override;
    bool IsPreloadCopyOptimizeCondition(const OpParam& param, ExecMem& execMem) override;
    HcclResult KernelRun(const OpParam& param, ExecMem& execMem) override;
    bool IsPowerOfTwo(u32 num);
    HcclResult CopyFromUserInToCclIn(const OpParam& param, ExecMem& execMem);
    HcclResult RunAlgLevel1(const OpParam& param, u64 reduceAttr, ExecMem& execMem, SubCommInfo& level1CommInfo);
    HcclResult RunAlgLevel0(
        const OpParam& param, u64 reduceAttr, ExecMem& execMem, SubCommInfo& level0CommInfo,
        SubCommInfo& level1CommInfo);
};

} // namespace hccl

#endif
