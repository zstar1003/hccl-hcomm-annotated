/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COLL_BATCH_SEND_RECV_GROUP_EXECUTOR_H
#define COLL_BATCH_SEND_RECV_GROUP_EXECUTOR_H

#include "coll_comm_executor.h"
#include "coll_batch_send_recv_executor.h"
#include <map>

namespace hccl {
class CollBatchSendRecvGroupExecutor : public CollBatchSendRecvExecutor {
public:
    CollBatchSendRecvGroupExecutor(const HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher>& topoMatcher);
    ~CollBatchSendRecvGroupExecutor() override = default;
    HcclResult Orchestrate(OpParam& param, AlgResourceResponse& algResource) override;

protected:
    /* *************** 算法编排 *************** */
    u64 CalcSendLoopMaxCount(const u32 unitSize) const;
    u64 CalcRecvLoopMaxCount(const u32 unitSize) const;
    HcclResult CalcSendSlices();
    HcclResult CalcRecvSlices();
    HcclResult OrganizeSendItemByStream();
    HcclResult OrganizeRecvItemByStream();
    HcclResult CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport) override;
    struct SendRecvSlice {
        u8* addr;
        u64 size;
        u32 remoteRank;
        bool isRdma; // true=RDMA任务(用CCLOut拆半，无ping-pong)；false=SDMA任务(用CCLIn，ping-pong)
        SendRecvSlice(u8* addr, u64 size, u32 remoteRank, bool isRdma = false)
            : addr(addr),
              size(size),
              remoteRank(remoteRank),
              isRdma(isRdma)
        {}
    };

    // SDMA slice按remoteRank % streamNum分发到各SDMA从流(仅SDMA，ping-pong)。
    std::vector<std::deque<SendRecvSlice>> sendDataSlicesBySendStream_;
    std::vector<std::deque<SendRecvSlice>> recvDataSlicesByRecvStream_;

    // RDMA与SDMA分离：SDMA占sendStreamNum_+recvStreamNum_条从流(CCLIn ping-pong)；
    // RDMA单独占2条从流(1 send + 1 recv)，对端顺序按alltoallv_direct_fullmesh对称场景规则
    // (send前向递增、recv后向递减，均跳过本pod；对端无任务则跳过)。
    // RDMA使用CCLOut拆半：A(offset 0)=send scratch(单slot), B(offset rdmaDataBlockSize_)=recv scratch(单slot)。
    // RDMA不做ping-pong，逐slice处理。
    static constexpr u32 RDMA_CCLOUT_HALF_NUM = 2; // CCLOut拆成两半：A=send, B=recv
    static constexpr u32 RDMA_STREAM_NUM = 2;      // RDMA专用从流数：1 send + 1 recv

    // RDMA slice(已按对称场景规则排序)，分别由rdmaSendStreamIdx_/rdmaRecvStreamIdx_对应的从流处理。
    std::deque<SendRecvSlice> rdmaSendSlices_;
    std::deque<SendRecvSlice> rdmaRecvSlices_;

private:
    HcclResult RunLoop(OpParam& param);
    HcclResult RunTasks(OpParam& param);
    HcclResult ProcessPreloadedSendSlice(u32 streamIdx, u32& pendingSendCount, u32& nonEmptySendStream);
    HcclResult ProcessNewRankSendSlice(u32 streamIdx, u32& pendingSendCount, u32& nonEmptySendStream);
    HcclResult ProcessRecvSlice(u32 streamIdx, u32& nonEmptyRecvStream);
    // RDMA专用从流(单send/单recv)上处理一个slice，对端顺序由rdmaSendSlices_/rdmaRecvSlices_保证。
    HcclResult ProcessRdmaSendSlice();
    HcclResult ProcessRdmaRecvSlice();
    HcclResult CalcPodRange();
    bool IsRemoteRankRdma(u32 remoteRank) const;
    HcclResult SetNormalModeIfDeviceDirect();
    HcclResult CalcStreamNum(u32& streamNum) override;
    HcclResult CalcPingPongHalfSize();

    HcclResult MainPostSubWait(Stream& mainStream);
    HcclResult MainWaitSubPost(Stream& mainStream);
    // 统计各从流(SDMA send/recv + RDMA send/recv)是否有任务并记录到成员变量，
    // 同时返回SDMA非空流数供RunTasks循环使用。循环中不再更新。
    HcclResult CalcStreamTaskStatus(u32& nonEmptySendStream, u32& nonEmptyRecvStream);

    // 对称场景规则：send前向递增/recv后向递减遍历，跳过本pod。返回当前候选rank并推进游标。
    u32 GetNextDstRank(u32& curDstRank) const;
    u32 GetPreSrcRank(u32& curSrcRank) const;
    // 计算是否为非对称超节点场景(参考alltoallv_direct_fullmesh executor中的isSuPodAsym判断)
    void CalcIsSuPodAsym(bool isA2MultiModule);
    // 按对称场景规则对RDMA slice排序：isSend=true用前向(GetNextDstRank)，false用后向(GetPreSrcRank)。
    // 遍历所有跨pod对端，仅输出存在任务的对端的slice(起点初始化与每次更新均跳过无任务对端)。
    void OrderRdmaSlices(
        bool isSend, const std::map<u32, std::deque<SendRecvSlice>>& byRank, std::deque<SendRecvSlice>& out);

    // RDMA专用从流在slaveStreams中的索引。
    u32 RdmaSendStreamIdx() const { return sendStreamNum_ + recvStreamNum_; }
    u32 RdmaRecvStreamIdx() const { return sendStreamNum_ + recvStreamNum_ + 1; }

private:
    std::vector<std::deque<HcclSendRecvItem*>> sendQueueBySendstream_;
    std::vector<std::deque<HcclSendRecvItem*>> recvQueueByRecvstream_;
    u32 sendStreamNum_ = 0;
    u32 recvStreamNum_ = 0;
    u64 bufferSliceSize_ = 0;
    u64 rdmaDataBlockSize_ = 0; // RDMA单流slot大小 = CCLOut/2(A=send半区, B=recv半区)
    u32 podStartRank_ = 0; // pod(rank)范围，用于判定isRdma：[podStartRank_, podEndRank_]内为SDMA，跨pod为RDMA
    u32 podEndRank_ = 0;
    u32 devNumInlocalPod_ = 0; // 本pod内rank数，用于对称场景规则起点的计算
    bool isSuPodAsym_ = false; // 非对称场景：A2A3卡数不一致或A3多超节点server数不同时为true，send/recv使用相同遍历顺序
    // Ping-pong state
    std::vector<u32> sendCurPhase_;         // which half has loaded data, ready to Record+Send (0=A, 1=B)
    std::vector<u64> sendLoadedSize_;       // size of data loaded in current sendCurPhase_ half (0 = nothing)
    std::vector<u32> sendLoadedRemoteRank_; // remote rank for the data loaded in current sendCurPhase_
    std::vector<u32> recvCurPhase_;         // which half to read from for recv (0=A, 1=B)
    std::vector<u32> recvCurRemoteRank_;    // current remote rank being received on this stream

    // 各从流是否有任务(在RunTasks起始时记录，循环中不更新)，用于头尾同步只唤醒有任务的从流。
    std::vector<bool> sendStreamHasTask_; // send从流(共sendStreamNum_条)是否有任务
    std::vector<bool> recvStreamHasTask_; // recv从流(共recvStreamNum_条)是否有任务
    bool rdmaSendHasTask_ = false;        // RDMA专用send从流是否有任务
    bool rdmaRecvHasTask_ = false;        // RDMA专用recv从流是否有任务
};
} // namespace hccl

#endif
