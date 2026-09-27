/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COLL_ALLTOALLV_PIPELINE_FOR_910_93_EXECUTOR_H
#define COLL_ALLTOALLV_PIPELINE_FOR_910_93_EXECUTOR_H
#include "coll_all_to_all_executor.h"
namespace hccl {
class CollAlltoAllVPipelineFor91093 : public CollAlltoAllExecutor { // A3 背靠背机型算法

public:
    CollAlltoAllVPipelineFor91093(const HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher>& topoMatcher);
    ~CollAlltoAllVPipelineFor91093() override = default;
    HcclResult Orchestrate(OpParam& param, AlgResourceResponse& algRes) override;

private:
    HcclResult CalcStreamNum(u32& streamNum) override;
    HcclResult CalcTransportMemType(TransportMemType& inputType, TransportMemType& outputType) const;
    HcclResult CalcLevel1CommInfo(
        TransportMemType inputType, TransportMemType outputType,
        std::vector<LevelNSubCommTransport>& opTransport) override;
    HcclResult CalcLevel2CommInfo(
        TransportMemType inputType, TransportMemType outputType,
        std::vector<LevelNSubCommTransport>& opTransport) override;
    HcclResult CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport) override;
    HcclResult KernelRun(const OpParam& param, ExecMem& execMem) override;
    HcclResult CalLocalSendRecvInfo(const OpParam& param, SendRecvInfo& info);
    HcclResult CalcA2ASendRecvInfo(const OpParam& param, SendRecvInfo& info);
    HcclResult CalcA2AvSendRecvInfo(const OpParam& param, SendRecvInfo& info);
    HcclResult CalcA2AvcSendRecvInfo(const OpParam& param, SendRecvInfo& info);
};

} // namespace hccl
#endif
