/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COLL_ALLGATHER_MID_COUNT_FOR_910_93_EXECUTOR_H
#define COLL_ALLGATHER_MID_COUNT_FOR_910_93_EXECUTOR_H
#include "coll_all_gather_executor.h"
namespace hccl {
class CollAllGatherMidCountFor91093Executor : public CollAllGatherExecutor {
public:
    explicit CollAllGatherMidCountFor91093Executor(
        const HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher>& topoMatcher);
    ~CollAllGatherMidCountFor91093Executor() override = default;

private:
    /* *************** 资源计算 *************** */
    // HcclResult CalcStreamNum(u32& streamNum) override;
    HcclResult CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport) override;
    HcclResult CalcTransportMemType(TransportMemType& inputType, TransportMemType& outputType) const;
    HcclResult CalcLevel1CommInfo(
        TransportMemType inputType, TransportMemType outputType,
        std::vector<LevelNSubCommTransport>& opTransport) override;
    HcclResult CalcLevel2CommInfo(
        TransportMemType inputType, TransportMemType outputType,
        std::vector<LevelNSubCommTransport>& opTransport) override;

    /* *************** 算法编排 *************** */
    u64 CalcLoopMaxCount(const u64 cclBuffSize, const u32 unitSize) override;
    u64 CalcDstMemOffset(const OpParam& param, u64 inputMemSize) const;
    HcclResult PrepareL2DataSlices(
        const OpParam& param, const SubCommInfo& level1CommInfo, const SubCommInfo& level2CommInfo, u64 inputMemSize,
        std::vector<Slice>& dataSlices) const;
    HcclResult RunLevel2ByNHR(
        const OpParam& param, ExecMem& execMem, SubCommInfo& level1CommInfo, SubCommInfo& level2CommInfo) const;
    HcclResult PrepareL1DataSlices(
        const OpParam& param, const SubCommInfo& level1CommInfo, const SubCommInfo& level2CommInfo, u64 inputMemSize,
        u32 moduleId, std::vector<Slice>& dataSlices) const;
    HcclResult RunLevel1ByNHR(
        const OpParam& param, ExecMem& execMem, SubCommInfo& level1CommInfo, SubCommInfo& level2CommInfo) const;
    HcclResult KernelRun(const OpParam& param, ExecMem& execMem) override;
};

} // namespace hccl

#endif
