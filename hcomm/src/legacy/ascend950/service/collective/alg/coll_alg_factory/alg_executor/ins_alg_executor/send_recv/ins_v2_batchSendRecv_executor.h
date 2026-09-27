/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_INS_V2_BATCH_SEND_RECV_EXECUTOR_H
#define HCCLV2_INS_V2_BATCH_SEND_RECV_EXECUTOR_H

#include <unordered_set>
#include <algorithm>

#include "ins_coll_alg_base.h"
#include "ins_temp_all_gather_mesh.h"
#include "topo_match_partial_mesh.h"
#include "instruction.h"
#include "data_buffer.h"
#include "rmt_data_buffer_mgr.h"

namespace Hccl {
constexpr u32 MULTIPLY_TWO = 2;
constexpr u64 HCCL_CHUNK_SIZE = 1024 * 1024 * 1024; // 1024*1024*1024的size

template <typename AlgTopoMatch>
class InsV2BatchSendRecvExecutor : public InsCollAlgBase {
public:
    explicit InsV2BatchSendRecvExecutor();
    ~InsV2BatchSendRecvExecutor() override;

    std::string Describe() const override { return "Instruction based Send Executor."; }

    HcclResult Orchestrate(
        const RankGraph* rankGraph, const CollAlgOperator& op, const CollAlgParams& params, InsQuePtr insQue) override;

    HcclResult CalcResOffload(const RankGraph* rankGraph, const u64& dataSize, CollOffloadOpResReq& resReq) override;

    HcclResult CalcRes(const RankGraph* rankGraph, CollAlgResReq& algResReq) override;

    HcclResult Orchestrate(
        const AlgTopoInfo& topoInfo, const CollAlgOperator& op, const CollAlgParams& params, ConnectedLinkMgr* linkMgr,
        InsQuePtr insQue) override;

    void SetRmaDataBufferMgr(const RmtDataBufferMgr* rmaDataBufferMgr) override;
    HcclResult CalNumBlocks(u32& blockDim, u64 dataSize, u32 blockDimLimit) override;

protected:
    struct SendRecvSlice {
        uintptr_t addr_;
        u64 size_;
        u32 remoteRank_;
        SendRecvSlice(uintptr_t addr, u64 size, u32 remoteRank) : addr_(addr), size_(size), remoteRank_(remoteRank) {}
    };

    u32 remoteUserRank_ = 0;
    const u32 MAX_LOOP_IN_ONCE_LAUNCH = 200;
    std::deque<SendRecvSlice> sendDataSilces_;
    std::deque<SendRecvSlice> recvDataSilces_;

private:
    bool SortSendItems(HcclSendRecvItem* a, HcclSendRecvItem* b) const;
    bool SortRecvItems(HcclSendRecvItem* a, HcclSendRecvItem* b) const;
    HcclResult InitParams(const CollAlgOperator& op, const CollAlgParams& params) override;
    HcclResult GetPairWiseList();

    // 收发数据准备
    HcclResult CalcSendSlices(u64 maxRoundTransferSize);
    HcclResult CalcRecvSlices(u64 maxRoundTransferSize);
    HcclResult GenSendSlicesMapRank();
    HcclResult GenRecvSlicesMapRank();

    // 实现自发自收
    HcclResult ProcessSelfSendRecvTasks(InsQuePtr& queue);

    // 实现数据发送&接收
    HcclResult ProcessSendRecv(
        const CollAlgOperator& op, InsQuePtr& queue, u32 remoteRank, std::vector<SendRecvSlice>& sendRemoteSlices,
        std::vector<SendRecvSlice>& recvRemoteSlices, LinkData& link) const;
    HcclResult ProcessSendDataSlice(
        InsQuePtr& queue, SendRecvSlice& sendRemoteSlice, u32 remoteRank, uint64_t scratchBufferAddr,
        LinkData& link) const;
    HcclResult SendRun(DataBuffer& execBufferSlice, u32 remoteUserRank, InsQuePtr& queue, LinkData& link) const;
    HcclResult
    CopyRecvDataSliceToUsrOut(InsQuePtr& queue, SendRecvSlice& slice, u32 remoteRank, uint64_t scratchBufferAddr) const;

    HcclResult
    RunLoopSendRecv(const CollAlgOperator& op, std::vector<InsQuePtr>& queues, InsTempAllGatherMesh1D& tempAlg);
    HcclResult CalcResLinksPartialMesh(
        const RankId myRank, const std::vector<std::vector<RankId>>& tempVTopo, const u32 linkNumBtwPeers,
        AlgTempResReq& tempResReq);
    HcclResult CalcRes(AlgTempResReq& tempResReq);
    HcclResult ExecAiv();

    HcclResult InitCommInfo(const RankGraph* rankGraph);
    HcclResult InitCommInfo(const AlgTopoInfo& topoInfo);

    std::set<u32> commTargetUserRankSet_;
    std::deque<HcclSendRecvItem*> sendToSelfDeque_;
    std::deque<HcclSendRecvItem*> recvFromSelfDeque_;
    std::deque<HcclSendRecvItem*> sendDeque_;
    std::deque<HcclSendRecvItem*> recvDeque_;
    BuffInfo buffInfo_;

    std::vector<RankId> virtRanks_;
    std::map<RankId, u32> virtRankMap_; // map<virtRank, virtRankOrder>
    std::vector<std::vector<RankId>> vTopo_;

    std::vector<InsQuePtr> requiredQue_;
    ResLinks tempResLinks_;

    std::map<u32, std::vector<SendRecvSlice>> SendSliceMapByRemoteRank_;
    std::map<u32, std::vector<SendRecvSlice>> RecvSliceMapByRemoteRank_;
    u64 maxRoundTransferSize_ = 0; // 单轮最多能够传输的size
    u32 sliceId_{0};               // 用于组装aivTag
};
} // namespace Hccl
#endif // !HCCLV2_INS_V2_BATCH_SEND_RECV_EXECUTOR_H
