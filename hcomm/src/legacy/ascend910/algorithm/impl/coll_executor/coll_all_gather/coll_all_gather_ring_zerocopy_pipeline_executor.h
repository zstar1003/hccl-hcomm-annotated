/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COLL_ALLGATHER_RING_ZEROCOPY_PIPELINE_EXECUTOR_H
#define COLL_ALLGATHER_RING_ZEROCOPY_PIPELINE_EXECUTOR_H

#include "coll_all_gather_executor.h"

namespace hccl {
class CollAllGatherRingZerocopyPipelineExecutor : public CollAllGatherExecutor {
public:
    explicit CollAllGatherRingZerocopyPipelineExecutor(
        const HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher>& topoMatcher);
    ~CollAllGatherRingZerocopyPipelineExecutor() override = default;

    HcclResult Orchestrate(OpParam& param, AlgResourceResponse& algRes) override;

private:
    /* *************** 资源计算 *************** */
    HcclResult CalcStreamNum(u32& streamNum) override;
    HcclResult CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport) override;
    HcclResult CalcLevel0CommInfo(
        TransportMemType inputType, TransportMemType outputType,
        std::vector<LevelNSubCommTransport>& opTransport) override;
    HcclResult CalcLevel2CommInfo(
        TransportMemType inputType, TransportMemType outputType,
        std::vector<LevelNSubCommTransport>& opTransport) override;
    HcclResult CalcExchangeCommInfo(std::vector<LevelNSubCommTransport>& opTransport);
    HcclResult CalcTransportMemType(TransportMemType& inputType, TransportMemType& outputType) const;

    /* *************** 算法编排 *************** */
    u64 CalcLoopMaxCount(const u64 cclBuffSize, const u32 unitSize) override;
    HcclResult RunLoop(OpParam& param);
    HcclResult KernelRunWithLoop(const OpParam& param, ExecMem& execMem, bool isLastLoop);

    HcclResult KernelRunInterSuperPod(const OpParam& param, ExecMem& execMem);

    HcclResult KernelRunIntraServerPost(const OpParam& param, ExecMem& execMem) override;
    HcclResult SemiRingAllGather(
        const std::string& tag, DeviceMem& inputMem, DeviceMem& outputMem, const u64 count,
        const HcclDataType& dataType, const std::vector<std::vector<Slice>>& multRingsSliceZero, const Stream& stream,
        s32 profStage, const u64 baseOffset, const HcomCollOpInfo* opInfo,
        const std::vector<std::vector<Slice>>& multRingsUserMemSlice);

    HcclResult KernelRunInterServer(const OpParam& param, ExecMem& execMem) override;
    HcclResult KernelRunInterServerPreProcess(const OpParam& param, ExecMem& execMem);
    HcclResult KernelRunInterServerPostProcess(const OpParam& param, ExecMem& execMem);

    HcclResult NotifyRdmaStreamStart();
    HcclResult WaitRdmaStreamFinish();

    HcclResult CalExchangeRemoteRank(u32& remoteRankSend, u32& remoteRankRecv);
    HcclResult CalcDataSlices(u64 sliceSize, u32 rankSize, std::vector<Slice>& dataSegsSlice) const;

    u32 unitSize_ = 0;
    u64 totalSize_ = 0; // 输入总数据量

    u32 level0Rank_ = INVALID_VALUE_RANKID;
    u32 level1Rank_ = INVALID_VALUE_RANKID;
    u32 level2Rank_ = INVALID_VALUE_RANKID;
    u32 level0RankSize_ = INVALID_VALUE_RANKSIZE;
    u32 level1RankSize_ = INVALID_VALUE_RANKSIZE;
    u32 level2RankSize_ = INVALID_VALUE_RANKSIZE;

    Stream mainStream_; // SDMA+localcopy
    std::vector<Stream> subStreams_;
    Stream rdmaMainStream_;
    std::vector<Stream> sdmaSubStreams_;
    std::shared_ptr<LocalNotify> notifyMainToRdma_;
    std::shared_ptr<LocalNotify> notifyRdmaToMain_;
    std::vector<std::shared_ptr<LocalNotify>> notifySdmaMain_;
    std::vector<std::shared_ptr<LocalNotify>> notifySdmaSub_;

    u32 memIdx_ = 0; // 表示当前算法编排使用CCLIN的第几块（0/1），是RDMA通信的源和目的，也是数据交换的起始
    u32 blockIdx_ = 0;  // 表示当前超节点内在处理来自于第几个超节点的数据
    u64 blockSize_ = 0; // 表示来自于一个超节点的数据的大小
};

} // namespace hccl

#endif
