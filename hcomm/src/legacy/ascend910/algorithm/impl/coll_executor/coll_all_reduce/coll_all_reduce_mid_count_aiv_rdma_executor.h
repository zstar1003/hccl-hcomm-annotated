/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COLL_ALLREDUCE_MID_COUNT_AIV_RDMA_EXECUTOR_H
#define COLL_ALLREDUCE_MID_COUNT_AIV_RDMA_EXECUTOR_H

#include "coll_all_reduce_executor.h"
#include "hccl_aiv.h"
#include "executor_impl.h"
#include "sender_pub.h"

namespace hccl {
class CollAllReduceMidCountAivRdmaExecutor : public CollAllReduceExecutor {
public:
    CollAllReduceMidCountAivRdmaExecutor(const HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher>& topoMatcher);
    ~CollAllReduceMidCountAivRdmaExecutor() override = default;

    HcclResult Orchestrate(OpParam& param, AlgResourceResponse& algRes) override;
    HcclResult GetAdjInfo(AlgResourceResponse& algRes, AdjInfo& adjInfo) override;

private:
    /* *************** 资源计算 *************** */
    HcclResult CalcStreamNum(u32& streamNum) override;
    HcclResult CalcLevel0CommInfo(
        TransportMemType inputType, TransportMemType outputType,
        std::vector<LevelNSubCommTransport>& opTransport) override;
    HcclResult CalcTransportMemType(TransportMemType& inputType, TransportMemType& outputType) const;
    HcclResult CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport) override;
    HcclResult CalNumBlocks(
        u32& numBlocks, u32 rankSize, u64 dataSize = 0, HcclCMDType cmdType = HcclCMDType::HCCL_CMD_INVALID) override;

    /* *************** 算法编排 *************** */

    HcclResult KernelRun(const OpParam& param, ExecMem& execMem) override;
};

} // namespace hccl

#endif
