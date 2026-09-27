/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COLL_REDUCESCATTER_RING_ZEROCOPY_EXCHANGE_EXECUTOR_H
#define COLL_REDUCESCATTER_RING_ZEROCOPY_EXCHANGE_EXECUTOR_H
#include "coll_reduce_scatter_ring_zerocopy_executor.h"

namespace hccl {
class CollReduceScatterRingZerocopyExchangeExecutor : public CollReduceScatterRingZerocopyExecutor {
public:
    explicit CollReduceScatterRingZerocopyExchangeExecutor(
        const HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher>& topoMatcher);
    ~CollReduceScatterRingZerocopyExchangeExecutor() override = default;

private:
    HcclResult CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport) override;
    HcclResult CalcExchangeCommInfo(std::vector<LevelNSubCommTransport>& opTransport);

    HcclResult KernelRunIntraServerPreProcess();
    HcclResult KernelRunInterServerPostProcess(const OpParam& param, const ExecMem& execMem) override;

    HcclResult KernelRunInterServerPreProcess(const OpParam& param, const ExecMem& execMem) override;
    HcclResult
    CalcLevel0DataSlices(const OpParam& param, const ExecMem& execMem, std::vector<Slice>& dataSegsSlice) override;
};

} // namespace hccl

#endif
