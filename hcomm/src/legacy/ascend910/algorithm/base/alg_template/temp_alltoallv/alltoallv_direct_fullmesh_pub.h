/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ALLTOALL_V_DIRECT_FULLMESH_PUB_H
#define ALLTOALL_V_DIRECT_FULLMESH_PUB_H

#include "mc2_handler_pub.h"
#include "alg_template_register.h"

namespace hccl {
class AlltoAllVDirectFullMesh : public AlgTemplateBase {
public:
    explicit AlltoAllVDirectFullMesh(const HcclDispatcher dispatcher);
    ~AlltoAllVDirectFullMesh() override;
    HcclResult RunAsync() override;
    HcclResult Prepare(PrepareData& param) override;
    HcclResult
    GetNslbAdjInfo(const u32 rank, const u32 rankSize, const std::vector<LINK>& links, AdjInfo& nslbAdjInfo) override;

    // 用于alltoallv类算子的aicpu cache
    // 注意: 如果新增alltoallv的算法实现并且想使用aicpu cache, 需要在更新cache白名单前, 在算法实现中做以下适配:
    // (i) Zero-length recv/send memcpy需要保证调用HcclD2DMemcpyAsync/TxAck/RxAck/TxDataSignal/RxDataSignal
    // 从而dispatcher可以感知并下发对应placeholder
    // (ii) 需要维护hccl offset <-> rank vector之间的mapping, 保证cache命中后的正确刷新
    // 可以在alltoallv_direct_fullmesh.cc下搜索needAlltoallvCache_获得参考
    HcclResult
    GetHcclOffsetDstRanksMap(std::unordered_map<uint64_t, std::vector<uint32_t>>& hcclOffsetDstRanksMap) const override;

protected:
private:
    HcclResult GenerateSubStreamInfo(
        const std::vector<Stream>& subStreams, const std::vector<std::shared_ptr<LocalNotify>>& meshSignalMainToSub,
        const std::vector<std::shared_ptr<LocalNotify>>& meshSignalSubToMain);
    std::string GetStreamIndexString();
    u64 CalcMaxSendLen();
    HcclResult NotifySubStreamStart();
    HcclResult WaitSubStreamFinish();
    HcclResult NotifyLocalSubStreamStart();
    HcclResult WaitLocalSubStreamFinish();
    u32 CalcNumSubStep();
    HcclResult NotifyRemoteRankStart(u32 step);
    HcclResult SDMAwithRemoteRankAndNotifyEnd(u32 step, u32 roundIdx);
    HcclResult SendRecvData(u32 step, u32 roundIdx);

    HcclResult UpdateCurrRankSendInfo(
        u32 step, u32 roundIdx, u32 side, u32 destRank, std::vector<SendDataBlock>& sendInfo,
        std::unordered_map<u32, SendDataBlock>& subStreamZcopySendInfo, u32 maxSendStep);
    HcclResult UpdateCurrRankRecvInfo(
        u32 step, u32 roundIdx, u32 side, u32 destRank, std::vector<ReadDataBlock>& readInfo,
        std::unordered_map<u32, ReadDataBlock>& subStreamZcopyReadInfo, u32 maxRecvStep);
    void UpdateOpBaseSubStreamInfo(u32 step, u32 roundIdx);
    void UpdateRemoteRankSet(u32 roundIdx, u32 groupRankSize);
    void UpdatePartialCommunicationRankSetPairWise(u32 roundIdx, u32 groupRankSize);
    void UpdatePartialCommunicationRankSet(
        u32 roundIdx, u32 groupRankSize, std::vector<std::vector<std::pair<u32, u32>>>& partialCommRankSet);
    HcclResult PrepareIntraData(
        u32 step, std::unordered_map<u32, std::vector<SendDataBlock>>& subStreamSendInfo,
        std::unordered_map<u32, SendDataBlock>& subStreamZcopySendInfo);
    void UpdateSendRecvInfo(
        u32 step, u32 roundIdx, std::unordered_map<u32, std::vector<ReadDataBlock>>& subStreamReadInfo,
        std::unordered_map<u32, std::vector<SendDataBlock>>& subStreamSendInfo,
        std::unordered_map<u32, ReadDataBlock>& subStreamZcopyReadInfo,
        std::unordered_map<u32, SendDataBlock>& subStreamZcopySendInfo,
        const std::vector<std::vector<std::pair<u32, u32>>>& partialCommRankSet);
    HcclResult LocalCopy();
    HcclResult RunGroupFullMeshAlltoall(u32 roundIdx, u32 step);
    HcclResult RunSDMA(HcclOpMetaInfoDef& opMeta);
    HcclResult RunSDMATasks(u32 roundIdx, u32 step, u32 groupRankSize, u32 leftRankSize);
    HcclResult RunSDMAFineGrained(u32 totalStep, HcclOpMetaInfoDef& opMeta);

    // RDMA处理相关函数
    HcclResult MainNotifyRdmaControlStart();
    HcclResult RdmaControlNotifyMainFinish();
    HcclResult RdmaControlNotifySubStart();
    HcclResult SubNotifyRdmaControlFinish();
    u32 GetNextDstRank(u32& curDstRank);
    u32 GetPreSrcRank(u32& curDstRank);
    void GenRdmaSendInfo(u32 dstRank, std::vector<SendDataBlock>& sendInfo);
    void GenRdmaRecvInfo(u32 srcRank, std::vector<RecvDataBlock>& recvInfo);
    HcclResult CopyDataForSend(u32 dstRank, std::vector<SendDataBlock>& sendInfo, u32 curStep, Stream stream);
    HcclResult SendRecvRdmaData(
        u32 dstRank, u32 srcRank, std::vector<SendDataBlock>& sendInfo, std::vector<RecvDataBlock>& recvInfo, u32 round,
        u32 index, u32 curStep, Stream stream);
    HcclResult CopyRecvDataToOutput(u32 srcRank, std::vector<RecvDataBlock>& recvInfo, u32 curStep, Stream stream);
    HcclResult ProcessSingleGroupRdmaData(std::vector<u32>& dstRanks, std::vector<u32>& srcRanks, u32 round);
    HcclResult ProcessRdmaData();
    HcclResult RunRDMA();

    // 后同步处理相关函数
    bool IsPostSyncEnable(u32 step, u32 roundIdx);
    HcclResult SdmaMainStreamWait(u32 step, u32 roundIdx);
    HcclResult SdmaMainStreamPost(u32 step, u32 roundIdx);
    HcclResult RdmaPostSync(Stream& stream);
    HcclResult SetPostSyncTasks(u32 step, u32 roundIdx);

    Stream mainStream_;
    u32 userRank_;
    u32 userRankSize_;
    u32 podStartRank_; // 表示一个pod内起始的userRankId
    u32 podEndRank_;   // 表示一个pod内结束的userRankId
    std::vector<LINK> links_;
    const SendRecvInfo* localSendRecvInfoPtr_;
    u32 podNum_;
    u32 devNumInlocalPod_;
    u32 rankIdxInPod_;
    u32 totalRdmaRankNum_; // 需要通信的rdma对端
    bool isSuPodAsym_;
    HcclCMDType opType_;
    bool isBigCount_;

    DeviceMem userInput_;
    DeviceMem userOutput_;
    DeviceMem cclInMem_;
    DeviceMem cclOutMem_;
    HcclWorkflowMode workMode_;
    u64 sdmaDataBlockSize_ = 0;

    bool islocalCpyDone_ = false;
    std::unordered_map<u32, std::vector<SendDataBlock>> subStreamSendInfo_; // 从流当前发送长度和发送的本地偏移
    std::unordered_map<u32, std::vector<ReadDataBlock>> subStreamReadInfo_; // 从流当前接收长度和接收到的本地偏移
    std::unordered_map<u32, std::vector<SendDataBlock>> nextSubStreamSendInfo_; // 下一轮从流发送长度和发送的本地偏移
    std::unordered_map<u32, std::vector<ReadDataBlock>> nextSubStreamReadInfo_; // 下一轮从流接收长度和接收到的本地偏移
    std::unordered_map<u32, u32> sendNumSubStep_; // 需要向对应对端rank发几次数据
    std::unordered_map<u32, u32> recvNumSubStep_; // 需要从对应对端rank收几次数据
    u32 sdmaConcurrentNum_;                       // 分组mesh-每组group的ranksize
    std::vector<std::vector<std::pair<u32, u32>>>
        partialCommRankSet_; // 参与通信的rank组合, 第0、1、2个vector分别存放左、右、中的rank
    std::vector<std::vector<std::pair<u32, u32>>> nextPartialCommRankSet_; // 下一轮参与通信的rank组合
    u64 commRounds_ = 0; // 每个rank分组fullmesh后需要通信的轮次

    // 本地拷贝处理相关
    std::vector<Stream> localSubStream_;
    std::vector<std::shared_ptr<LocalNotify>> localSignalMainToSub_;
    std::vector<std::shared_ptr<LocalNotify>> localSignalSubToMain_;
    // SDMA处理相关
    std::vector<Stream> sdmaSubStream_;
    std::vector<std::shared_ptr<LocalNotify>> sdmaMeshSignalMainToSub_;
    std::vector<std::shared_ptr<LocalNotify>> sdmaMeshSignalSubToMain_;
    // RDMA处理相关
    u64 rdmaDataBlockSize_ = 0;
    // RDMA并发数量
    u32 rdmaConcurrentNum_;
    std::shared_ptr<LocalNotify> main2RdmaControlStreamNotify_;
    std::shared_ptr<LocalNotify> rdmaControl2MainStreamNotify_;
    // RDMA从流，以及RDMA控制流与从流同步的notify
    std::vector<Stream> rdmaSubStreams_;
    std::vector<std::shared_ptr<LocalNotify>> rdmaControl2SubNotifies_;
    std::vector<std::shared_ptr<LocalNotify>> rdmaSub2ControlNotifies_;
    Mc2HandlerPub mc2HandlerPub;
    // 重执行后同步优化需要在最后一个step插入收发任务做拉齐操作
    u32 lastStep_ = 0;
    u32 lastRoundIdx_ = 0;
    u32 lastRdmaRoundIdx_ = 0;
    u32 lastRdmaDstRanksIdx_ = 0;
    u32 lastRdmaStep_ = 0;

    // 用于alltoallv类算子的aicpu cache
    bool needAlltoallvCache_ = false; // 是否需要对当前alltoallv算子做aicpu cache
    // 注意: 对零长拷贝保留send/read info, 是为了在使能alltoallv类算子aicpu cache时, 保证cache-memcpy placeholder
    // SQE的正确下发 注意: 这里Zcopy指的是zero-length memory copy, 而不是zero copy 注意:
    // 从流相关的偏移信息每个step每个round UpdateSendRecvInfo前都会clear, 所以可以重复设置
    std::unordered_map<u32, SendDataBlock> subStreamZcopySendInfo_;     // 从流当前零长拷贝的发送偏移
    std::unordered_map<u32, ReadDataBlock> subStreamZcopyReadInfo_;     // 从流当前零长拷贝的接收偏移
    std::unordered_map<u32, SendDataBlock> nextSubStreamZcopySendInfo_; // 下一轮从流零长拷贝的发送偏移
    std::unordered_map<u32, ReadDataBlock> nextSubStreamZcopyReadInfo_; // 下一轮从流零长拷贝的接收偏移
    // 注意: hccl offset <-> remote rank之间的mapping不能每个step每个round清理, aicpu cache需要获得step
    // 0所有rounds下的映射信息, 所以只有在step为0时才会更新 (其实hccl offsets在不同steps下不变, 变化的只有user
    // input/output offsets)
    std::unordered_map<uint64_t, std::vector<uint32_t>>
        hcclOffsetDstRanksMap_; // Local hccl input buffer中的local hccl offset到remote dst ranks的映射
                                // (用于PrepareIntraData; 当rankSize较大触发multi-round场景时, 同一个hccl
                                // offset可能会对应多个dst ranks)
};
} // namespace hccl
#endif /* ALLTOALL_V_MESH_READ_ONLY_PUB_H */
