/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COLL_RUN_ALLTOALL_SYMMETRIC_MEMORY_EXECUTOR_H
#define COLL_RUN_ALLTOALL_SYMMETRIC_MEMORY_EXECUTOR_H
#include "coll_all_to_all_executor.h"
namespace hccl {
class CollRunAlltoAllFullMeshSymmetricMemory : public CollAlltoAllExecutor {
public:
    CollRunAlltoAllFullMeshSymmetricMemory(const HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher>& topoMatcher);
    ~CollRunAlltoAllFullMeshSymmetricMemory() override = default;
    HcclResult Orchestrate(OpParam& param, AlgResourceResponse& algRes) override;

private:
    HcclResult CalcStreamNum(u32& streamNum) override;
    HcclResult CalcLevel0CommInfo(
        TransportMemType inputType, TransportMemType outputType,
        std::vector<LevelNSubCommTransport>& opTransport) override;
    HcclResult CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport) override;
    HcclResult KernelRun(const OpParam& param, ExecMem& execMem) override;
    HcclResult GetAlltoAllTmpRankSendRecvInfo(const OpParam& param);
    HcclResult GetLocalSendRecvInfoforAlltoall(const OpParam& param);
    HcclResult GetLocalSendRecvInfoforAlltoallVC(const OpParam& param);
    HcclResult CalcTransportMemType(TransportMemType& inputType, TransportMemType& outputType) const;
    HcclResult GetLocalSDMAGroupInfo(u32& devNumInlocalPod, u32& rankIdxInPod) const;
    ZCopySendRecvInfo sendRecvInfo_;
    const ZCopySendRecvInfo* sendRecvInfoPtr = nullptr;
};
} // namespace hccl
#endif
