/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COLL_REDUCESCATTER_RING_ZEROCOPY_EXCHANGE_PIPELINE_EXECUTOR_H
#define COLL_REDUCESCATTER_RING_ZEROCOPY_EXCHANGE_PIPELINE_EXECUTOR_H

#include "coll_reduce_scatter_executor.h"

namespace hccl {
class CollReduceScatterRingZerocopyExchangePipelineExecutor : public CollReduceScatterExecutor {
public:
    explicit CollReduceScatterRingZerocopyExchangePipelineExecutor(
        const HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher>& topoMatcher);
    ~CollReduceScatterRingZerocopyExchangePipelineExecutor() override = default;

protected:
    HcclResult CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport) override;
    u32 level0Rank_ = INVALID_VALUE_RANKID;
    u32 level1Rank_ = INVALID_VALUE_RANKID;
    u32 level2Rank_ = INVALID_VALUE_RANKID;
    u32 level0RankSize_ = INVALID_VALUE_RANKSIZE;
    u32 level1RankSize_ = INVALID_VALUE_RANKSIZE;
    u32 level2RankSize_ = INVALID_VALUE_RANKSIZE;

private:
    void ParseParam(const OpParam& param) override;
    /* *************** 资源计算 *************** */
    HcclResult CalcStreamNum(u32& streamNum) override;
    HcclResult CalcLevel0CommInfo(
        TransportMemType inputType, TransportMemType outputType,
        std::vector<LevelNSubCommTransport>& opTransport) override;
    HcclResult CalcExchangeCommInfo(std::vector<LevelNSubCommTransport>& opTransport);
    u64 CalcLoopMaxCount(const u32 unitSize) override;

    /* *************** 算法编排 *************** */
    HcclResult KernelRunIntraServerPre(const OpParam& param, ExecMem& execMem) override;
    HcclResult KernelRunInterServer(const OpParam& param, ExecMem& execMem) override;

    HcclResult RunIntraServer(const OpParam& param, const ExecMem& execMem, u32 step);
    HcclResult RunInterServerPreProcess(const OpParam& param, const ExecMem& execMem, u32 step);
    HcclResult RunInterServer(const OpParam& param, ExecMem& execMem, u32 step);
    HcclResult RunInterServerPostProcess(const OpParam& param, const ExecMem& execMem, u32 step);
    HcclResult
    ExchangeData(const OpParam& param, const ExecMem& execMem, u32 step, u32 remoteRankSend, u32 remoteRankRecv);
    HcclResult RunSuperPodPreSync(const OpParam& param);
    HcclResult RunSuperPod(const OpParam& param, const ExecMem& execMem, u32 step);
    HcclResult RunSuperPodPostSync(const OpParam& param);
    HcclResult RunSuperPodAndInterServerPostProcess(const OpParam& param, const ExecMem& execMem, u32 step);
    HcclResult RunFinallyProcess(const OpParam& param, const ExecMem& execMem) const;

    HcclResult CalExchangeRemoteRank(u32& remoteRankSend, u32& remoteRankRecv) const;
    HcclResult SemiRingReduceScatter(
        const std::string& tag, DeviceMem inputMem, DeviceMem outputMem, const u64 count, const HcclDataType dataType,
        const HcclReduceOp reductionOp, const std::vector<std::vector<Slice>> multRingsSliceZero, Stream stream,
        s32 profStage, const u64 baseOffset, const HcomCollOpInfo* opInfo,
        const std::vector<std::vector<Slice>> multRingsUserMemSlice);

    bool intraServerDone_{false};
    u64 curSize_{0};
    u32 unitSize_{0};
    u32 exchangeRemoteRankSend_{0};
    u32 exchangeRemoteRankRecv_{0};
};

} // namespace hccl

#endif
