/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ALLTOALL_V_CONTINUOUS_PIPELINE_PUB_H
#define ALLTOALL_V_CONTINUOUS_PIPELINE_PUB_H

#include "alg_template_base_pub.h"

namespace hccl {

constexpr u32 PINGPONG_MEM_NUM = 2;

struct TaskState {
    u32 stepNum{0};     // stepNum来分别表示当前做到哪一步
    u32 stepNumNext{0}; // next表示每种task在这一轮要做到哪一步，步长是RDMA并发度
    u32 loopNum{0};     // 记录每种task现在进行了几轮
};

enum class SdmaSyncType {
    PRE_SYNC, // 前同步
    POST_SYNC // 尾同步
};

class AlltoallvContinuousPipeline : public AlgTemplateBase {
public:
    explicit AlltoallvContinuousPipeline(const HcclDispatcher dispatcher);
    ~AlltoallvContinuousPipeline() override;
    HcclResult Prepare(
        const u32 userRank, const A2aPipelineMemory& a2aPipelineMemory, const SubCommInfo& level0CommInfo,
        const SubCommInfo& level1CommInfo, const Stream& mainStream, std::vector<Stream>& subStream,
        std::vector<std::shared_ptr<LocalNotify>>& notifyMain, std::vector<std::shared_ptr<LocalNotify>>& notifySub,
        std::vector<SendRecvInfo>& sendRecvInfoList, const HcclDataType dataType, const HcclWorkflowMode workMode,
        u32 timeOut) override;
    HcclResult RunAsync() override;

private:
    // Prepare子函数
    HcclResult PrepareSendRecvInfo(std::vector<SendRecvInfo>& sendRecvInfoList);
    HcclResult
    PrepareTopoInfo(const u32 userRank, const SubCommInfo& level0CommInfo, const SubCommInfo& level1CommInfo);

    // 对buffer进行切分，in和out buffer都分为数据区域和信息区域，数据区域和信息区域里再分成user rank size块
    HcclResult SplitBuffer(const bool enablePingPong);

    // 将从流资源划分为RDMA并发流和SDMA并发流，对Notify资源也进行相应的划分
    HcclResult PartitionSubStreamsAndNotifies(
        const std::vector<Stream>& subStreams, const std::vector<std::shared_ptr<LocalNotify>>& signalMainToSub,
        const std::vector<std::shared_ptr<LocalNotify>>& signalSubToMain);

    // 将一块准备要发送到remoteRank的数据从input拷贝到in buffer对应分块里，拷贝size为MIN(剩余要发送的size，ccl分块大小)
    HcclResult LocalCopyFromInputToInBuffer(const u32 targetRank, Stream& stream, const u32 loopIdx);

    // 将一块来自remoteRank的数据从out buffer对应分块里拷贝到output，拷贝size为MIN(剩余要接收的size，ccl分块大小)
    HcclResult LocalCopyFromOutBufferToOutput(const u32 sourceRank, Stream& stream, const u32 loopIdx);

    // 将自己发送到自己的那块数据，从input拷贝到output对应位置，拷贝size为完整数据大小
    HcclResult LocalCopySelfDataFromInputToOutput(Stream& stream);

    // 将一块要通过SDMA发送给机内另一个rank的数据，从本端input发送到对端的out
    // buffer对应分块里，拷贝size为MIN(剩余要发送的size，ccl分块大小)
    HcclResult SdmaSendFromInputToRemoteOutBuffer(const u32 targetRank, Stream& stream, const u32 loopIdx);

    // 通过SDMA从机内另一个rank的out
    // buffer对应分块读取自己接收的数据，放到本端output，拷贝size为MIN(剩余要接收的size，ccl分块大小)
    HcclResult SdmaReadFromRemoteOutBufferToOutput(const u32 sourceRank, Stream& stream, const u32 loopIdx);

    // 跨module，向sendRank发送数据，从recvRank接收数据，数据从in buffer到out buffer
    HcclResult InterSendAndReceive(const u32 sendRank, const u32 recvRank, Stream& stream, const u32 loopIdx);

    // 处理发送给跨module rank数据的本地拷贝，根据起始stepNum和步长计算具体要处理哪些rank的数据
    HcclResult DoLocalCopy(const u32 beginStepNum, const u32 endStepNum, const u32 loopIdx);

    // 在module内分发来自跨module rank发送过来的数据，根据起始stepNum和步长计算具体要处理哪些rank的数据
    HcclResult DoIntraDistribution(const u32 beginStepNum, const u32 endStepNum, const u32 loopIdx);

    // 跨module rank的数据收发，根据起始stepNum和步长计算具体要处理哪些rank的数据
    HcclResult DoInterSendReceive(const u32 beginStepNum, const u32 endStepNum, const u32 loopIdx);

    // 拷贝来自module内其他rank发来的数据到output
    HcclResult DoLevel0LocalCopy(const u32 loopIdx);

    // 将要给module内其他rank的数据发送到对端out buffer
    HcclResult DoLevel0SdmaSend(const u32 loopIdx);

    // Sdma前后同步
    HcclResult DoSdmaSync(const SdmaSyncType syncType);

    // 向module内其他rank广播本rank的counts信息
    HcclResult DoIntraInfoBroadcast();

    // 本地将info和flag写到buffer对应位置，同时做一次level1的同步
    HcclResult DoLocalWriteInfoAndFlagAndInterSync();

    // 轮询等待某个rank的flag，阻塞函数
    HcclResult WaitValueOfRank(const u32 rank, const HcclUs& startTimeUs, u32& value);

    // 等待counts信息并刷新receive info，阻塞函数
    HcclResult WaitAndCalReceiveInfo();

    // 获取module内某个rank对应的Sdma从流idx
    inline u32 GetSdmaSubStreamIdx(const u32 remoteRank) const;

    // 主流通知SDMA从流，SDMA从流等待主流
    HcclResult NotifySdmaSubStreamStart();
    // 主流等待SDMA从流，SDMA从流通知主流
    HcclResult WaitSdmaSubStreamFinish();

    // 主流通知RDMA从流，RDMA从流等待主流
    HcclResult NotifyRdmaSubStreamStart();
    // 主流等待RDMA从流，RDMA从流通知主流
    HcclResult WaitRdmaSubStreamFinish();

    // 跨module通信，通过SDMA从link left读
    HcclResult InterSdmaRx(
        const LINK& linkLeft, const LINK& linkRight, std::vector<TxMemoryInfo>& sendMems,
        std::vector<RxMemoryInfo>& recvMems, Stream& stream) const;

    // 跨module通信，通过RDMA从link left读或向link right写
    HcclResult InterRdmaTxRx(
        const LINK& linkLeft, const LINK& linkRight, std::vector<TxMemoryInfo>& sendMems,
        std::vector<RxMemoryInfo>& recvMems, Stream& stream) const;

    // 获取本rank的counts和displacements信息
    inline u64 GetLocalSendCountOfRank(const u32 targetRank) const;
    inline u64 GetLocalSendDisplOfRank(const u32 targetRank) const;
    inline u64 GetLocalRecvCountOfRank(const u32 sourceRank) const;
    inline u64 GetLocalRecvDisplOfRank(const u32 sourceRank) const;

    // 获取每个rank对应的内存分块偏移值，bufferIdx为0或1表示启用乒乓时的两组内存
    inline u64 GetDataBlockOffset(const u32 rank, const u32 bufferIdx) const;

    // 刷新某个rank的send info，counts-[count]，displ+[count]
    HcclResult UpdateLocalSendInfo(const u32 targetRank, const u64 count);
    // 刷新某个rank的receive info，counts-[count]，displ+[count]
    HcclResult UpdateLocalRecvInfo(const u32 sourceRank, const u64 count);

    // 计算本地loop数，后续和module内其他rank进行比较，取最大的loop数作为总loop数
    u32 GetLocalLoopNum() const;

    HcclWorkflowMode workMode_ = HcclWorkflowMode::HCCL_WORKFLOW_MODE_RESERVED;

    DeviceMem inBuffer_;
    DeviceMem outBuffer_;

    u32 userRankSize_{0};
    u32 intraRankSize_{0};
    u32 interRankSize_{0};

    u32 userRank_{0};
    u32 intraRankId_{0};
    u32 interRankId_{0};

    std::vector<std::vector<u32>> ranksPerModule_; // 按照module将rank分组

    u32 rdmaConcurrentNum_{0}; // 跨module通信并发度

    bool enablePingPong_{false}; // 跨module通信只有1步时需要将in buffer和out buffer的data区域平分成两份，避免内存冲突

    Stream mainStream_;
    std::vector<Stream> subStreams_;

    std::vector<Stream> sdmaSubStreams_;
    std::vector<std::shared_ptr<LocalNotify>> streamNotifyMainToSdmaSub_;
    std::vector<std::shared_ptr<LocalNotify>> streamNotifySdmaSubToMain_;

    std::vector<Stream> rdmaSubStreams_;
    std::vector<std::shared_ptr<LocalNotify>> streamNotifyMainToRdmaSub_;
    std::vector<std::shared_ptr<LocalNotify>> streamNotifyRdmaSubToMain_;

    std::vector<LINK> intraLinks_;
    std::vector<LINK> interLinks_;

    u32 unitSize_{0}; // 数据类型对应的字节数

    u64 sizePerBlock_{0};               // buffer中每个数据分块的大小，一共有user rank size块
    u64 countsPerBlock_{0};             // buffer中每个数据分块的count数
    std::vector<u64> dataBlockOffsets_; // 每个数据分块的首地址，数组长度为user rank size，索引是user rank
    std::vector<u64> infoOffsets_; // counts或flag分块的首地址，数组长度为user rank size，索引是user rank

    std::vector<u64> inBufferDataSize_; // 记录在in buffer里每个分块目前存放的数据大小

    bool needCollectInfo_{false}; // 是否需要先收集其他module的counts信息
    const SendRecvInfo* localSendRecvInfoPtr_ = nullptr;
    std::vector<std::vector<u64>> intraRecvCounts_; // 记录module内所有卡的recv counts

    std::vector<u64> localSendCounts_;
    std::vector<u64> localSendDispls_;
    std::vector<u64> localRecvCounts_;
    std::vector<u64> localRecvDispls_;

    std::vector<u32> flagAreaRefreshData_;

    u32 waitFlagTimeoutSec_{NOTIFY_DEFAULT_WAIT_TIME}; // 等待Flag的超时时间，单位秒

    u32 flagAreaRefreshFlag_{0}; // 标识flag区域已经被刷0，避免刷0的任务还没执行，下发态kernel就开始轮询
    u32 flagAreaRefreshValue_{1};

    u32 intraLoopNum_{0}; // 本module内所有rank的最大loop数
};

} // namespace hccl

#endif /* * ALLTOALL_V_CONTINUOUS_PIPELINE_PUB_H */
