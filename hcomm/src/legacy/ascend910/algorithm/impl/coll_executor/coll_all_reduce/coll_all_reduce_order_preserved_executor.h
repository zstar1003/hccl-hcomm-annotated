/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COLL_ALL_REDUCE_ORDER_PRESERVED_EXECUTOR_H
#define COLL_ALL_REDUCE_ORDER_PRESERVED_EXECUTOR_H
#include "coll_all_reduce_executor.h"

namespace hccl {
class CollAllReduceOrderPreservedExecutor : public CollAllReduceExecutor {
public:
    CollAllReduceOrderPreservedExecutor(const HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher>& topoMatcher);

private:
    void ParseParam(const OpParam& param) override;
    /* *************** 资源计算 *************** */
    HcclResult CalcScratchMemSize(u64& scratchMemSize) override;
    u32 CalReduceStreamNum(const u32& localRankSize);
    HcclResult CalcStreamNum(u32& streamNum) override;
    HcclResult CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport) override;
    HcclResult CalcLevel0CommInfo(
        TransportMemType inputType, TransportMemType outputType,
        std::vector<LevelNSubCommTransport>& opTransport) override;
    HcclResult CalcLevel1CommInfo(
        TransportMemType inputType, TransportMemType outputType,
        std::vector<LevelNSubCommTransport>& opTransport) override;
    HcclResult CalcTransportMemType(TransportMemType& inputType, TransportMemType& outputType) const;
    void CalGroupSlices(const OpParam& param, const ExecMem& execMem);
    void CalcSizePerBlock(const OpParam& param, ExecMem& execMem);
    HcclResult
    RunReduceScatterLevel0SingleRank(const OpParam& param, ExecMem& execMem, const SubCommInfo& level0CommInfo) const;

    /* *************** 算法编排 *************** */
    bool IsHugeData(const u64 curSize) override;
    HcclResult KernelRun(const OpParam& param, ExecMem& execMem) override;
    HcclResult RunReduceScatterLevel0(const OpParam& param, ExecMem& execMem, SubCommInfo& level0CommInfo);
    HcclResult RunReduceScatterLevel1(const OpParam& param, ExecMem& execMem, SubCommInfo& level0CommInfo);
    HcclResult RunAllGatherLevel0(const OpParam& param, ExecMem& execMem, SubCommInfo& level0CommInfo);
    HcclResult RunAllGatherLevel1(const OpParam& param, ExecMem& execMem, const SubCommInfo& level0CommInfo);

    u64 sizePerBlock_{0};        // 单块数据的大小
    std::vector<u64> groupSize_; // input切分每块数据的大小
    bool scratchMemFlag_{false}; // 是否需要申请scratch memory，不需要申请则传入outputmem为scratchmem
    u64 totalSize_{0};           // 总数据量
    u32 all2allOffset_ = 0;
};

} // namespace hccl

#endif
