/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COLL_BROADCAST_MID_COUNT_FOR_910_93_EXECUTOR_H
#define COLL_BROADCAST_MID_COUNT_FOR_910_93_EXECUTOR_H
#include "coll_broadcast_executor.h"
namespace hccl {
class CollBroadcastMidCountFor91093Executor : public CollBroadcastExecutor {
public:
    explicit CollBroadcastMidCountFor91093Executor(
        const HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher>& topoMatcher);
    ~CollBroadcastMidCountFor91093Executor() override = default;
    HcclResult Orchestrate(OpParam& param, AlgResourceResponse& algRes) override;

private:
    /* *************** 资源计算 *************** */
    HcclResult CalcStreamNum(u32& streamNum) override;
    HcclResult CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport) override;
    HcclResult CalcTransportMemType(TransportMemType& inputType, TransportMemType& outputType) const;
    HcclResult CalcLevel1CommInfo(
        TransportMemType inputType, TransportMemType outputType,
        std::vector<LevelNSubCommTransport>& opTransport) override;
    HcclResult CalcLevel2CommInfo(
        TransportMemType inputType, TransportMemType outputType,
        std::vector<LevelNSubCommTransport>& opTransport) override;
    u64 CalcLoopMaxCount(const u64 cclBuffSize, const u32 unitSize) override;
    /* *************** 算法编排 *************** */
    HcclResult RunLevel2ByNHR(
        const OpParam& param, ExecMem& execMem, SubCommInfo& level1CommInfo, SubCommInfo& level2CommInfo) const;
    HcclResult
    RunLevel1ByNHR(const OpParam& param, ExecMem& execMem, SubCommInfo& level1CommInfo, SubCommInfo& level2CommInfo);
    HcclResult KernelRun(const OpParam& param, ExecMem& execMem) override;
    HcclResult RunLoopInner(OpParam& param, ExecMem& execMem);
};

} // namespace hccl

#endif
