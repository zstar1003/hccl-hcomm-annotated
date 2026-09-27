/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COLL_RUN_ALLTOALL_STAGED_AIV_RDMA_EXECUTOR_H
#define COLL_RUN_ALLTOALL_STAGED_AIV_RDMA_EXECUTOR_H
#include "coll_all_to_all_executor.h"
#include "hccl_aiv.h"

namespace hccl {
class CollRunAlltoAllStagedAivRdmaExecutor : public CollAlltoAllExecutor {
public:
    CollRunAlltoAllStagedAivRdmaExecutor(const HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher>& topoMatcher);
    ~CollRunAlltoAllStagedAivRdmaExecutor() override = default;

    HcclResult Orchestrate(OpParam& param, AlgResourceResponse& algRes) override;

private:
    HcclResult CalcLevel0CommInfo(
        TransportMemType inputType, TransportMemType outputType,
        std::vector<LevelNSubCommTransport>& opTransport) override;
    HcclResult CalcLevel1CommInfo(
        TransportMemType inputType, TransportMemType outputType,
        std::vector<LevelNSubCommTransport>& opTransport) override;
    HcclResult CalcStreamNum(u32& streamNum) override;
    HcclResult CalcScratchMemSize(u64& scratchMemSize) override;
    HcclResult CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport) override;
    HcclResult KernelRun(const OpParam& param, ExecMem& execMem) override;

    HcclResult RunAlltoAllStaged1InAIV(const OpParam& param, ExecMem& execMem);
    HcclResult PrepareAivBuffers(DeviceMem& inputMem, DeviceMem& outputMem, void** dataBuffers, void** flagBuffers);
    HcclResult RunAlltoAllStaged2(const OpParam& param, ExecMem& execMem);
    void CalcInterMeshAggregationAlltoAllMemInfo(
        const OpParam& param, std::map<u32, std::list<OneSendRecvAddrInfo>>& sendAddrInfosInter,
        std::map<u32, std::list<OneSendRecvAddrInfo>>& recvAddrInfosInter);
    HcclResult CalNumBlocks(
        u32& numBlocks, u32 rankSize, u64 dataSize = 0, HcclCMDType cmdType = HcclCMDType::HCCL_CMD_INVALID) override;

    /* *************** 算法参数 *************** */
    u32 sendDataSize_ = 0;
    u32 recvDataSize_ = 0;
    SubCommInfo innerCommInfo_ = {0, 0, std::vector<LINK>(), std::vector<LINK>()};
    SubCommInfo outerCommInfo_ = {0, 0, std::vector<LINK>(), std::vector<LINK>()};
};

} // namespace hccl

#endif
