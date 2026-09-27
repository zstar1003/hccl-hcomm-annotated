/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COLL_ALLGATHER_RING_ZEROCOPY_EXCHANGE_EXECUTOR_H
#define COLL_ALLGATHER_RING_ZEROCOPY_EXCHANGE_EXECUTOR_H
#include "coll_all_gather_ring_zerocopy_executor.h"
#include "coll_reduce_scatter_ring_zerocopy_exchange_executor.h"

namespace hccl {
class CollAllGatherRingZerocopyExchangeExecutor : public CollAllGatherRingZerocopyExecutor {
public:
    explicit CollAllGatherRingZerocopyExchangeExecutor(
        const HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher>& topoMatcher);
    ~CollAllGatherRingZerocopyExchangeExecutor() override = default;

private:
    /* *************** 资源计算 *************** */
    HcclResult CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport) override;
    HcclResult CalcExchangeCommInfo(std::vector<LevelNSubCommTransport>& opTransport);
    HcclResult CalExchangeRemoteRank(u32& remoteRankSend, u32& remoteRankRecv);

    /* *************** 算法编排 *************** */
    HcclResult
    CalcLevel0DataSlices(const OpParam& param, const ExecMem& execMem, std::vector<Slice>& dataSegsSlice) override;
    HcclResult KernelRunInterServerPreProcess(const OpParam& param, const ExecMem& execMem) override;
    HcclResult KernelRunInterServerPostProcess(const OpParam& param, const ExecMem& execMem) override;
};

} // namespace hccl

#endif
