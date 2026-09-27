/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COLL_ALLREDUCE_MESH_OPBASE_MID_COUNT_DETERMINISTIC_EXECUTOR_H
#define COLL_ALLREDUCE_MESH_OPBASE_MID_COUNT_DETERMINISTIC_EXECUTOR_H

#include "coll_all_reduce_executor.h"

namespace hccl {
class CollAllReduceMeshOpbaseMidCountDeterministicExecutor : public CollAllReduceExecutor {
public:
    CollAllReduceMeshOpbaseMidCountDeterministicExecutor(
        const HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher>& topoMatcher);
    ~CollAllReduceMeshOpbaseMidCountDeterministicExecutor() override = default;

private:
    /**************** 资源计算 *************** */
    HcclResult CalcStreamNum(u32& streamNum) override;
    HcclResult CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport) override;
    HcclResult CalcLevel0CommInfo(
        TransportMemType inputType, TransportMemType outputType,
        std::vector<LevelNSubCommTransport>& opTransport) override;
    HcclResult CalcTransportMemType(TransportMemType& inputType, TransportMemType& outputType) const;
    /* *************** 任务编排 *************** */
    bool IsHugeData(const u64 curSize) override;
    bool IsSmallData(const u64 totalSize, const u64 curSize) override;
    HcclResult RunLoopInner(OpParam& param, const ReduceType& reduceType, ExecMem& execMem) override;
    HcclResult KernelRun(const OpParam& param, ExecMem& execMem) override;
    HcclResult PrepareSlicesInfo(
        const OpParam& param, const ExecMem& execMem, std::vector<Slice>& dataSegsSlice,
        GroupSlicesInfo& groupSlicesInfo, const u32 sliceSize) const;
    HcclResult RunReduceScatterLevel0(const OpParam& param, ExecMem& execMem, GroupSlicesInfo& groupSlicesInfo);
    HcclResult RunAllReduceLevel1(const OpParam& param, ExecMem& execMem, const std::vector<Slice>& dataSegsSlice);
    HcclResult RunAllGatherLevel0(const OpParam& param, ExecMem& execMem, const std::vector<Slice>& dataSegsSlice);
};

} // namespace hccl

#endif
