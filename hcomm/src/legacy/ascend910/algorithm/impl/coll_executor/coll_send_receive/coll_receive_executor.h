/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COLL_RECEIVE_EXECUTOR_H
#define COLL_RECEIVE_EXECUTOR_H
#include "coll_comm_executor.h"
#include "alg_template_base_pub.h"

namespace hccl {
class CollReceiveExecutor : public CollNativeExecutorBase {
public:
    CollReceiveExecutor(const HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher>& topoMatcher);
    ~CollReceiveExecutor() override = default;

    HcclResult Orchestrate(OpParam& param, AlgResourceResponse& algRes) override;
    HcclResult GetAdjInfo(AlgResourceResponse& algRes, AdjInfo& adjInfo) override;

private:
    using CollNativeExecutorBase::CalcCommInfo;
    /* *************** 资源计算 *************** */
    HcclResult CalcResRequest(const OpParam& param, AlgResourceRequest& resourceRequest) override;
    HcclResult CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport, u32 srcRank);
    HcclResult CalcTransportMemType(TransportMemType& inputType, TransportMemType& outputType);
    HcclResult CalcP2PCommInfo(
        TransportMemType inputType, TransportMemType outputType, std::vector<LevelNSubCommTransport>& opTransport,
        u32 srcRank);

    /* *************** 算法编排 *************** */
    HcclResult RunLoop(OpParam& param, AlgResourceResponse& algRes);
    HcclResult RunTemplate(const OpParam& param, DeviceMem& outputMem);

    bool DMAReduceFlag_{false}; // 是否DMA消减的标志
};

} // namespace hccl

#endif
