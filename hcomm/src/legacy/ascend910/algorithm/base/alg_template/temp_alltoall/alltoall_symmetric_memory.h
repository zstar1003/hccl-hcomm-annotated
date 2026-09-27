/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ALLTOALL_V_SYMMETRIC_MEMORY_H
#define ALLTOALL_V_SYMMETRIC_MEMORY_H

#include "alg_template_register.h"

namespace hccl {
class AlltoAllFullMeshSymmetricMemory : public AlgTemplateBase {
public:
    explicit AlltoAllFullMeshSymmetricMemory(const HcclDispatcher dispatcher);
    ~AlltoAllFullMeshSymmetricMemory() override;
    HcclResult RunAsync() override;
    HcclResult Prepare(PrepareData& param) override;

protected:
private:
    HcclResult GenerateSubStreamInfo(
        const std::vector<Stream>& subStreams, const std::vector<std::shared_ptr<LocalNotify>>& meshSignalMainToSub,
        const std::vector<std::shared_ptr<LocalNotify>>& meshSignalSubToMain);
    std::string GetStreamIndexString();
    HcclResult NotifySubStreamStart();
    HcclResult WaitSubStreamFinish();
    HcclResult NotifyRemoteRankStart();
    HcclResult SDMAwithRemoteRankAndNotifyEnd(u32 roundIdx);
    HcclResult SendRecvData(u32 roundIdx);

    void UpdateCurrRankRecvInfo(u32 roundIdx, u32 side, u32 destRank, ReadDataBlock& readInfo);
    void UpdateRemoteRankSet(u32 roundIdx, u32 groupRankSize);
    void UpdatePartialCommunicationRankSetPairWise(u32 roundIdx, u32 groupRankSize);
    void UpdatePartialCommunicationRankSet(
        u32 roundIdx, u32 groupRankSize, std::vector<std::vector<std::pair<u32, u32>>>& partialCommRankSet);
    void UpdateSendRecvInfo(
        u32 roundIdx, std::unordered_map<u32, ReadDataBlock>& subStreamReadInfo,
        const std::vector<std::vector<std::pair<u32, u32>>>& partialCommRankSet);
    HcclResult LocalCopy();
    HcclResult RunGroupFullMeshAlltoall(u32 roundIdx);
    HcclResult RunSDMA(HcclOpMetaInfoDef& opMeta);
    HcclResult RunSDMATasks(u32 roundIdx, u32 groupRankSize, u32 leftRankSize);

    // 后同步处理相关函数
    bool IsPostSyncEnable(u32 roundIdx);
    HcclResult SdmaMainStreamWait(u32 roundIdx);
    HcclResult SdmaMainStreamPost(u32 roundIdx);
    HcclResult SetPostSyncTasks(u32 roundIdx);

    Stream mainStream_;
    u32 userRank_ = 0;
    u32 userRankSize_ = 0;
    u32 podStartRank_ = 0; // 表示一个pod内起始的userRankId
    u32 podEndRank_ = 0;   // 表示一个pod内结束的userRankId
    std::vector<LINK> links_;
    const ZCopySendRecvInfo* sendRecvInfoPtr_ = nullptr;
    u32 devNumInlocalPod_ = 0;
    u32 rankIdxInPod_ = 0;
    bool isSuPodAsym_ = false;
    HcclCMDType opType_ = HcclCMDType::HCCL_CMD_MAX;
    bool islocalCpyDone_ = false;

    DeviceMem userInput_;
    DeviceMem userOutput_;
    HcclWorkflowMode workMode_ = HcclWorkflowMode::HCCL_WORKFLOW_MODE_RESERVED;
    u64 sdmaDataBlockSize_ = 0;

    std::unordered_map<u32, ReadDataBlock> subStreamReadInfo_; // 从流当前接收长度和接收到的本地偏移
    u32 sdmaConcurrentNum_ = 0;                                // 分组mesh-每组group的ranksize
    std::vector<std::vector<std::pair<u32, u32>>>
        partialCommRankSet_; // 参与通信的rank组合, 第0、1、2个vector分别存放左、右、中的rank
    u64 commRounds_ = 0;     // 每个rank分组fullmesh后需要通信的轮次

    // SDMA处理相关
    std::vector<Stream> sdmaSubStream_;
    std::vector<std::shared_ptr<LocalNotify>> sdmaMeshSignalMainToSub_;
    std::vector<std::shared_ptr<LocalNotify>> sdmaMeshSignalSubToMain_;
    // 重执行后同步优化需要在最后一个step插入收发任务做拉齐操作
    u32 lastStep_ = 0;
    u32 lastRoundIdx_ = 0;
};
} // namespace hccl
#endif /* ALLTOALL_V_SYMMETRIC_MEMORY_H */
